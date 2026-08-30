"""Durable season/series registry layered above the single-episode task store."""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping

from runtime_config import projects_dir, state_dir


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _safe(value: str, label: str) -> str:
    text = str(value).strip()
    if not text or not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"{label} must contain only letters, digits, dot, underscore or hyphen")
    return text


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_bundle_hash(values: Iterable[Any], base: Path) -> str | None:
    hashes: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("path") or value.get("source_path")
        path = Path(str(value or ""))
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            return None
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return _json_hash(hashes) if hashes else None


_RUNTIME_ASSET_FIELDS = {
    "reference_images", "asset_status", "asset_hash", "asset_manifest_path",
    "asset_approval", "asset_rejection_history", "approved", "approved_at", "error",
}


def canonical_series(spec: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(spec), ensure_ascii=False))
    for collection in ("character_bible", "scene_bible"):
        for item in payload.get(collection) or []:
            if isinstance(item, dict):
                for key in _RUNTIME_ASSET_FIELDS:
                    item.pop(key, None)
    # A complete V4 envelope is retained under ``runtime.v4_contract`` for
    # restart/recovery.  The structural V4 copy deliberately excludes
    # per-episode generation/approval state so completing later episodes does
    # not revoke an already approved season contract.
    v4 = payload.get("v4_contract")
    if isinstance(v4, dict):
        for key in (
            "episode_contracts", "episode_approvals", "season_approved",
            "quality_warnings", "backend_status",
        ):
            v4.pop(key, None)
        for collection in ("shared_character_bible", "shared_scene_bible"):
            for item in v4.get(collection) or []:
                if isinstance(item, dict):
                    for key in _RUNTIME_ASSET_FIELDS:
                        item.pop(key, None)
    for key in ("status", "approval", "episodes", "assets", "deliveries", "runtime"):
        payload.pop(key, None)
    return payload


def series_contract_hash(spec: Mapping[str, Any]) -> str:
    return _json_hash(canonical_series(spec))


class SeriesStore:
    def __init__(self, path: str | Path | None = None):
        configured = path or os.environ.get("AI_MANGA_SERIES_DB") or os.environ.get("AI_FACTORY_SERIES_DB")
        self.path = Path(configured or state_dir() / "series.sqlite3").resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS series (
                series_id TEXT PRIMARY KEY, title TEXT, theme TEXT NOT NULL, synopsis TEXT NOT NULL,
                episode_count INTEGER NOT NULL, episode_seconds REAL NOT NULL,
                contract_hash TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft',
                approved_at TEXT, shared_assets_status TEXT NOT NULL DEFAULT 'pending',
                shared_assets_hash TEXT, spec TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS series_episodes (
                series_id TEXT NOT NULL, episode_number INTEGER NOT NULL, ep_id TEXT NOT NULL UNIQUE,
                predecessor_ep_id TEXT, contract_hash TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'registered',
                continuity_state_in TEXT NOT NULL DEFAULT '{}', continuity_state_out TEXT NOT NULL DEFAULT '{}',
                last_clip_path TEXT, delivery_manifest TEXT, error TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(series_id, episode_number)
            );
            CREATE TABLE IF NOT EXISTS series_assets (
                asset_id TEXT PRIMARY KEY, series_id TEXT NOT NULL, asset_type TEXT NOT NULL, source_id TEXT NOT NULL,
                prompt_hash TEXT NOT NULL, content_hash TEXT, reference_images TEXT NOT NULL DEFAULT '[]',
                manifest_path TEXT, status TEXT NOT NULL DEFAULT 'queued', approved INTEGER NOT NULL DEFAULT 0,
                prompt_id TEXT, error TEXT, metadata TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL,
                UNIQUE(series_id, asset_type, source_id)
            );
            CREATE TABLE IF NOT EXISTS series_workers (
                series_id TEXT PRIMARY KEY, owner TEXT NOT NULL, pid INTEGER NOT NULL, heartbeat REAL NOT NULL
            );
            """)

    @staticmethod
    def _decode_series(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if not row:
            return None
        result = dict(row)
        result["spec"] = json.loads(result["spec"])
        return result

    @staticmethod
    def _decode_episode(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in ("continuity_state_in", "continuity_state_out", "delivery_manifest"):
            result[key] = json.loads(result[key]) if result.get(key) else {}
        return result

    @staticmethod
    def _decode_asset(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["approved"] = bool(result["approved"])
        result["reference_images"] = json.loads(result["reference_images"] or "[]")
        result["metadata"] = json.loads(result["metadata"] or "{}")
        return result

    def get_series(self, series_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            return self._decode_series(conn.execute("SELECT * FROM series WHERE series_id=?", (series_id,)).fetchone())

    def save_series(self, series_id: str, spec: Mapping[str, Any]) -> dict[str, Any]:
        series_id = _safe(series_id, "series_id")
        now = _now()
        current_hash = series_contract_hash(spec)
        existing = self.get_series(series_id)
        same = bool(existing and existing["contract_hash"] == current_hash)
        status = existing["status"] if same else "draft"
        with self.connection() as conn:
            conn.execute("""INSERT INTO series(
                series_id,title,theme,synopsis,episode_count,episode_seconds,contract_hash,status,approved_at,
                shared_assets_status,shared_assets_hash,spec,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(series_id) DO UPDATE SET
                title=excluded.title,theme=excluded.theme,synopsis=excluded.synopsis,
                episode_count=excluded.episode_count,episode_seconds=excluded.episode_seconds,
                contract_hash=excluded.contract_hash,status=excluded.status,approved_at=excluded.approved_at,
                spec=excluded.spec,updated_at=excluded.updated_at""", (
                series_id, str(spec.get("title") or series_id), str(spec["theme"]), str(spec["synopsis"]),
                int(spec["episode_count"]), float(spec["episode_seconds"]), current_hash, status,
                existing.get("approved_at") if same and existing else None,
                existing.get("shared_assets_status", "pending") if existing else "pending",
                existing.get("shared_assets_hash") if existing else None,
                json.dumps(dict(spec), ensure_ascii=False), now, now,
            ))
        return self.get_series(series_id) or {}

    def update_series(self, series_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {"status", "approved_at", "shared_assets_status", "shared_assets_hash", "spec"}
        if set(changes) - allowed:
            raise ValueError(f"unsupported series fields: {sorted(set(changes) - allowed)}")
        if "spec" in changes:
            changes["spec"] = json.dumps(changes["spec"], ensure_ascii=False)
        changes["updated_at"] = _now()
        with self.connection() as conn:
            cur = conn.execute(
                f"UPDATE series SET {', '.join(f'{key}=?' for key in changes)} WHERE series_id=?",
                (*changes.values(), series_id),
            )
            if cur.rowcount != 1:
                raise KeyError(series_id)
        return self.get_series(series_id) or {}

    def replace_episodes(self, series_id: str, episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows = list(episodes)
        now = _now()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in rows:
                existing = conn.execute(
                    "SELECT * FROM series_episodes WHERE series_id=? AND episode_number=?",
                    (series_id, int(item["episode_number"])),
                ).fetchone()
                same = bool(existing and existing["contract_hash"] == item["contract_hash"])
                preserve_success = bool(same and existing["status"] in {"succeeded", "exported"})
                status = existing["status"] if preserve_success else "registered"
                conn.execute("""INSERT INTO series_episodes(
                    series_id,episode_number,ep_id,predecessor_ep_id,contract_hash,status,
                    continuity_state_in,continuity_state_out,last_clip_path,delivery_manifest,error,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(series_id,episode_number) DO UPDATE SET
                    ep_id=excluded.ep_id,predecessor_ep_id=excluded.predecessor_ep_id,
                    contract_hash=excluded.contract_hash,status=excluded.status,
                    continuity_state_in=excluded.continuity_state_in,
                    continuity_state_out=excluded.continuity_state_out,last_clip_path=excluded.last_clip_path,
                    delivery_manifest=excluded.delivery_manifest,error=excluded.error,updated_at=excluded.updated_at""", (
                    series_id, int(item["episode_number"]), item["ep_id"], item.get("predecessor_ep_id"),
                    item["contract_hash"], status,
                    existing["continuity_state_in"] if preserve_success else json.dumps(
                        item.get("continuity_state_in") or {}, ensure_ascii=False
                    ),
                    existing["continuity_state_out"] if preserve_success else "{}",
                    existing["last_clip_path"] if preserve_success else None,
                    existing["delivery_manifest"] if preserve_success else None,
                    existing["error"] if preserve_success else None, now, now,
                ))
            numbers = [int(item["episode_number"]) for item in rows]
            placeholders = ",".join("?" for _ in numbers)
            conn.execute(
                f"DELETE FROM series_episodes WHERE series_id=? AND episode_number NOT IN ({placeholders})",
                (series_id, *numbers),
            )
        return self.list_episodes(series_id)

    def list_episodes(self, series_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM series_episodes WHERE series_id=? ORDER BY episode_number", (series_id,)
            ).fetchall()
        return [self._decode_episode(row) for row in rows]

    def get_episode(self, series_id: str, episode_number: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM series_episodes WHERE series_id=? AND episode_number=?",
                (series_id, int(episode_number)),
            ).fetchone()
        return self._decode_episode(row) if row else None

    def update_episode(self, series_id: str, episode_number: int, **changes: Any) -> dict[str, Any]:
        allowed = {"status", "contract_hash", "continuity_state_in", "continuity_state_out", "last_clip_path", "delivery_manifest", "error"}
        if set(changes) - allowed:
            raise ValueError(f"unsupported episode fields: {sorted(set(changes) - allowed)}")
        for key in ("continuity_state_in", "continuity_state_out", "delivery_manifest"):
            if key in changes:
                changes[key] = json.dumps(changes[key] or {}, ensure_ascii=False)
        changes["updated_at"] = _now()
        with self.connection() as conn:
            cur = conn.execute(
                f"UPDATE series_episodes SET {', '.join(f'{key}=?' for key in changes)} WHERE series_id=? AND episode_number=?",
                (*changes.values(), series_id, int(episode_number)),
            )
            if cur.rowcount != 1:
                raise KeyError(f"{series_id}/{episode_number}")
        return self.get_episode(series_id, episode_number) or {}

    def replace_assets(self, series_id: str, assets: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows = list(assets)
        now = _now()
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in rows:
                existing = conn.execute("SELECT * FROM series_assets WHERE asset_id=?", (item["asset_id"],)).fetchone()
                same = bool(existing and existing["prompt_hash"] == item["prompt_hash"] and existing["content_hash"] == item.get("content_hash"))
                ready = bool(item.get("content_hash") and item.get("reference_images"))
                status = existing["status"] if same else ("succeeded" if ready else "queued")
                approved = existing["approved"] if same else 0
                if existing and existing["prompt_hash"] != item["prompt_hash"]:
                    ready = False
                    status = "queued"
                    approved = 0
                conn.execute("""INSERT INTO series_assets(
                    asset_id,series_id,asset_type,source_id,prompt_hash,content_hash,reference_images,
                    manifest_path,status,approved,prompt_id,error,metadata,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(asset_id) DO UPDATE SET
                    prompt_hash=excluded.prompt_hash,content_hash=excluded.content_hash,
                    reference_images=excluded.reference_images,manifest_path=excluded.manifest_path,
                    status=excluded.status,approved=excluded.approved,prompt_id=excluded.prompt_id,
                    error=excluded.error,metadata=excluded.metadata,updated_at=excluded.updated_at""", (
                    item["asset_id"], series_id, item["asset_type"], item["source_id"], item["prompt_hash"],
                    item.get("content_hash") if ready else None,
                    json.dumps(item.get("reference_images") or [], ensure_ascii=False) if ready else "[]",
                    existing["manifest_path"] if same else item.get("manifest_path"), status, approved,
                    existing["prompt_id"] if same else None, existing["error"] if same else None,
                    json.dumps(item.get("metadata") or {}, ensure_ascii=False), now,
                ))
            ids = [item["asset_id"] for item in rows]
            if ids:
                conn.execute(
                    f"DELETE FROM series_assets WHERE series_id=? AND asset_id NOT IN ({','.join('?' for _ in ids)})",
                    (series_id, *ids),
                )
            else:
                conn.execute("DELETE FROM series_assets WHERE series_id=?", (series_id,))
        return self.list_assets(series_id)

    def list_assets(self, series_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM series_assets WHERE series_id=? ORDER BY asset_type,source_id", (series_id,)
            ).fetchall()
        return [self._decode_asset(row) for row in rows]

    def update_asset(self, asset_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {"content_hash", "reference_images", "manifest_path", "status", "approved", "prompt_id", "error", "metadata"}
        if set(changes) - allowed:
            raise ValueError(f"unsupported series asset fields: {sorted(set(changes) - allowed)}")
        for key in ("reference_images", "metadata"):
            if key in changes:
                changes[key] = json.dumps(changes[key], ensure_ascii=False)
        if "approved" in changes:
            changes["approved"] = int(bool(changes["approved"]))
        changes["updated_at"] = _now()
        with self.connection() as conn:
            cur = conn.execute(
                f"UPDATE series_assets SET {', '.join(f'{key}=?' for key in changes)} WHERE asset_id=?",
                (*changes.values(), asset_id),
            )
            if cur.rowcount != 1:
                raise KeyError(asset_id)
            row = conn.execute("SELECT * FROM series_assets WHERE asset_id=?", (asset_id,)).fetchone()
        return self._decode_asset(row)

    def acquire_worker(self, series_id: str, stale_after: float = 120.0) -> bool:
        now = time.time()
        owner = f"{socket.gethostname()}:{os.getpid()}"
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM series_workers WHERE series_id=?", (series_id,)).fetchone()
            if row and now - float(row["heartbeat"]) <= stale_after and row["owner"] != owner:
                return False
            conn.execute("INSERT OR REPLACE INTO series_workers VALUES(?,?,?,?)", (series_id, owner, os.getpid(), now))
        return True

    def heartbeat(self, series_id: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE series_workers SET heartbeat=? WHERE series_id=? AND pid=?", (time.time(), series_id, os.getpid()))

    def worker_info(self, series_id: str, stale_after: float = 120.0) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM series_workers WHERE series_id=?", (series_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["active"] = time.time() - float(result["heartbeat"]) <= stale_after
        return result

    def release_worker(self, series_id: str) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM series_workers WHERE series_id=? AND pid=?", (series_id, os.getpid()))


_default: SeriesStore | None = None


def default_series_store() -> SeriesStore:
    global _default
    configured = os.environ.get("AI_MANGA_SERIES_DB") or os.environ.get("AI_FACTORY_SERIES_DB")
    expected = Path(configured or state_dir() / "series.sqlite3").resolve()
    if _default is None or _default.path != expected:
        _default = SeriesStore(expected)
    return _default


def series_project_dir(series_id: str) -> Path:
    return projects_dir() / "_series" / _safe(series_id, "series_id")


__all__ = [
    "SeriesStore", "default_series_store", "series_project_dir", "canonical_series",
    "series_contract_hash", "_file_bundle_hash", "_json_hash", "_now", "_safe",
]
