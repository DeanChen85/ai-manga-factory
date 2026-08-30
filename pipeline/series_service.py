"""Public durable facade for multi-episode/season production.

This module composes the existing single-episode pipeline.  Web callers only
prepare durable state or launch a hidden worker; GPU work never runs on the UI
thread.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import task_store
import video_delivery
import worker
from prompt_contracts import SERIES_SCHEMA_VERSION, validate_series_contract
from runtime_config import ffmpeg_executable, project_root
from series_store import (
    _file_bundle_hash, _json_hash, _now, _safe, canonical_series,
    default_series_store, series_contract_hash, series_project_dir,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _validate_series_spec(spec: Mapping[str, Any]) -> None:
    if not str(spec.get("theme") or "").strip():
        raise ValueError("series.theme is required")
    if not str(spec.get("synopsis") or "").strip():
        raise ValueError("series.synopsis is required")
    if int(spec.get("episode_count") or 0) <= 0:
        raise ValueError("series.episode_count must be positive")
    if float(spec.get("episode_seconds") or 0) <= 0:
        raise ValueError("series.episode_seconds must be positive")
    if not isinstance(spec.get("character_bible") or [], list):
        raise ValueError("series.character_bible must be a list")
    if not isinstance(spec.get("scene_bible") or [], list):
        raise ValueError("series.scene_bible must be a list")


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _structural_v4(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable V4 season contract used for backend identity."""
    structural = _copy(dict(contract))
    for key in (
        "episode_contracts", "episode_approvals", "season_approved",
        "quality_warnings", "backend_status",
    ):
        structural.pop(key, None)
    runtime_fields = {
        "reference_images", "asset_status", "asset_hash", "asset_manifest_path",
        "asset_approval", "asset_rejection_history", "approved", "approved_at", "error",
    }
    for collection in ("shared_character_bible", "shared_scene_bible"):
        for item in structural.get(collection) or []:
            if isinstance(item, dict):
                for key in runtime_fields:
                    item.pop(key, None)
    return structural


def _v4_declared_hash(contract: Mapping[str, Any]) -> str:
    """Reproduce the hash emitted by prompt_contracts.normalize_series_contract."""
    return hashlib.sha256(repr((
        contract.get("series_bible") or {},
        contract.get("shared_character_bible") or [],
        contract.get("world_bible") or {},
        contract.get("shared_scene_bible") or [],
        contract.get("season_outline") or [],
    )).encode("utf-8")).hexdigest()


def _validate_v4_contract(
    contract: Mapping[str, Any], *, require_all_approved: bool = False,
) -> dict[str, Any]:
    payload = _copy(dict(contract))
    errors = list(validate_series_contract(payload))
    declared = str(payload.get("series_sha256") or "")
    actual = _v4_declared_hash(payload)
    if not declared or declared != actual:
        errors.append("series_sha256 does not match the immutable shared V4 contract")
    outline = payload.get("season_outline") or []
    episode_ids = [str(item.get("episode_id") or "") for item in outline]
    expected_ids = [f"ep_{number:03d}" for number in range(1, int(payload.get("episode_count") or 0) + 1)]
    if episode_ids != expected_ids:
        errors.append("season_outline episode IDs must be the exact ordered 1..N set")
    approvals = payload.get("episode_approvals") or {}
    if set(approvals) != set(expected_ids):
        errors.append("episode_approvals keys must exactly match season_outline")
    contracts = payload.get("episode_contracts") or {}
    if require_all_approved:
        if set(contracts) != set(expected_ids):
            errors.append("episode_contracts must contain exactly N V3 contracts")
        missing = [episode_id for episode_id in expected_ids if not approvals.get(episode_id)]
        if missing:
            errors.append(f"all V4 episode contracts must be approved: {', '.join(missing)}")
        if not payload.get("season_approved"):
            errors.append("season_approved must be true before backend registration")
    if errors:
        raise ValueError("invalid V4 series contract: " + "; ".join(dict.fromkeys(errors)))
    return payload


def _series_spec_from_v4(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = _copy(dict(contract))
    bible = payload.get("series_bible") or {}
    brief = payload.get("creative_brief") or {}
    title = str(bible.get("title") or brief.get("topic") or "").strip()
    theme = str(brief.get("topic") or ", ".join(bible.get("themes") or []) or title).strip()
    synopsis = str(bible.get("premise") or brief.get("synopsis") or "").strip()
    if not title or not theme or not synopsis:
        raise ValueError("V4 series title/theme/synopsis are required")
    structural = _structural_v4(payload)
    return {
        "schema_version": "ai-manga.series-service-spec/v1",
        "series_id": str(bible.get("series_id") or ""),
        "title": title,
        "theme": theme,
        "synopsis": synopsis,
        "episode_count": int(payload.get("episode_count") or 0),
        "episode_seconds": float(payload.get("seconds_per_episode") or 0),
        "story_bible": _copy(bible),
        "visual_bible": _copy(payload.get("visual_bible") or {}),
        "world_bible": _copy(payload.get("world_bible") or {}),
        "character_bible": _copy(payload.get("shared_character_bible") or []),
        "scene_bible": _copy(payload.get("shared_scene_bible") or []),
        "season_outline": _copy(payload.get("season_outline") or []),
        "v4_series_sha256": payload.get("series_sha256"),
        "v4_shared_contract_hash": _json_hash(structural),
        "v4_contract": structural,
        # Full outer V4 envelope, including V3 episode contracts and review
        # state, is restart state rather than creative identity.
        "runtime": {"v4_contract": payload},
    }


def _stored_v4(series: Mapping[str, Any]) -> dict[str, Any] | None:
    spec = series.get("spec") if isinstance(series.get("spec"), Mapping) else {}
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), Mapping) else {}
    value = runtime.get("v4_contract") or spec.get("v4_contract")
    return _copy(value) if isinstance(value, Mapping) and value else None


def _v4_outline_item(series: Mapping[str, Any], episode_number: int) -> dict[str, Any] | None:
    contract = _stored_v4(series)
    if not contract:
        return None
    outline = contract.get("season_outline") or []
    index = int(episode_number) - 1
    return _copy(outline[index]) if 0 <= index < len(outline) else None


def _preserve_asset_runtime(series_id: str, spec: dict[str, Any]) -> None:
    """Keep approved generated assets when only V4 episode runtime changed."""
    store = default_series_store()
    existing = {
        (item["asset_type"], item["source_id"]): item
        for item in store.list_assets(series_id)
    }
    proposed = {
        (item["asset_type"], item["source_id"]): item
        for item in _asset_specs(series_id, spec)
    }
    for asset_type, collection, id_key in (
        ("character", spec.get("character_bible") or [], "character_id"),
        ("scene", spec.get("scene_bible") or [], "scene_id"),
    ):
        for source in collection:
            source_id = str(source.get(id_key) or source.get("id") or "")
            old = existing.get((asset_type, source_id))
            new = proposed.get((asset_type, source_id))
            if not old or not new or old.get("prompt_hash") != new.get("prompt_hash"):
                continue
            refs = list(old.get("reference_images") or [])
            if not old.get("content_hash") or _file_bundle_hash(refs, series_project_dir(series_id)) != old.get("content_hash"):
                continue
            source["reference_images"] = refs
            source["asset_status"] = old.get("status")
            source["asset_hash"] = old.get("content_hash")
            source["asset_manifest_path"] = old.get("manifest_path")
            source["asset_approval"] = {"state": "approved" if old.get("approved") else "pending"}


def _asset_specs(series_id: str, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    project = series_project_dir(series_id)
    visual = spec.get("visual_bible") or {}
    rows: list[dict[str, Any]] = []
    runtime_fields = {
        "reference_images", "asset_status", "asset_hash", "asset_manifest_path",
        "asset_approval", "asset_rejection_history", "approved", "approved_at", "error",
    }
    for asset_type, collection, id_key in (
        ("character", spec.get("character_bible") or [], "character_id"),
        ("scene", spec.get("scene_bible") or [], "scene_id"),
    ):
        for index, source in enumerate(collection, 1):
            source_id = str(source.get(id_key) or f"{asset_type}_{index:02d}")
            refs = list(source.get("reference_images") or [])
            prompt_source = {
                "asset_type": asset_type,
                "source": {key: value for key, value in source.items() if key not in runtime_fields},
                "visual_bible": visual,
            }
            if asset_type == "scene":
                prompt_source["world_bible"] = spec.get("world_bible") or {}
            rows.append({
                "asset_id": f"{series_id}:{asset_type}:{source_id}", "asset_type": asset_type,
                "source_id": source_id, "prompt_hash": _json_hash(prompt_source),
                "content_hash": _file_bundle_hash(refs, project), "reference_images": refs,
                "manifest_path": source.get("asset_manifest_path"),
                "metadata": {"prompt_source": prompt_source},
            })
    return rows


def prepare_series(series_id: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a draft season and register every shared asset; no GPU work."""
    series_id = _safe(series_id, "series_id")
    _validate_series_spec(spec)
    payload = json.loads(json.dumps(dict(spec), ensure_ascii=False))
    payload["series_id"] = series_id
    store = default_series_store()
    row = store.save_series(series_id, payload)
    assets = store.replace_assets(series_id, _asset_specs(series_id, payload))
    asset_status = "ready_for_approval" if all(item["status"] == "succeeded" for item in assets) else "pending"
    if row.get("shared_assets_status") != "approved" or not all(item["approved"] for item in assets):
        store.update_series(series_id, shared_assets_status=asset_status, shared_assets_hash=None)
    _atomic_json(series_project_dir(series_id) / "series.json", payload)
    return status_series(series_id, reconcile=False)


def prepare_series_contract(series_id: str, v4: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and durably prepare a normalized V4 creative contract.

    This is a persistence-only facade: it registers shared assets but never
    starts an API, FFmpeg, or GPU worker.
    """
    series_id = _safe(series_id, "series_id")
    contract = _validate_v4_contract(v4)
    declared_id = str((contract.get("series_bible") or {}).get("series_id") or "")
    if declared_id != series_id:
        raise ValueError(f"series_id mismatch: facade={series_id}, V4={declared_id or '<missing>'}")
    spec = _series_spec_from_v4(contract)
    _preserve_asset_runtime(series_id, spec)
    return prepare_series(series_id, spec)


def approve_series(series_id: str, *, expected_hash: str | None = None) -> dict[str, Any]:
    store = default_series_store()
    row = store.get_series(_safe(series_id, "series_id"))
    if not row:
        raise KeyError(series_id)
    if expected_hash and expected_hash != row["contract_hash"]:
        raise RuntimeError("series changed since review")
    _validate_series_spec(row["spec"])
    store.update_series(series_id, status="approved", approved_at=_now())
    return status_series(series_id, reconcile=False)


def _materialize_episode(series: Mapping[str, Any], raw: Mapping[str, Any], number: int) -> dict[str, Any]:
    episode = json.loads(json.dumps(dict(raw), ensure_ascii=False))
    series_id = series["series_id"]
    ep_id = _safe(str(episode.get("ep_id") or f"{series_id}_ep_{number:03d}"), "ep_id")
    episode["ep_id"] = ep_id
    episode["episode_number"] = number
    panels = episode.get("panels") or []
    if not panels:
        raise ValueError(f"episode {number} must contain panels")
    target = float(series["episode_seconds"])
    explicit_edits = [panel.get("edit_duration_seconds") for panel in panels]
    uses_shot_plan = any(value is not None for value in explicit_edits)
    if uses_shot_plan:
        if not all(value is not None for value in explicit_edits):
            raise ValueError(
                f"episode {number} must specify edit_duration_seconds on every panel"
            )
        planned = sum(float(value) for value in explicit_edits)
        if abs(planned - target) > 1 / 24:
            raise ValueError(
                f"episode {number} edit duration {planned} does not equal required {target}"
            )
        for panel in panels:
            source = panel.get(
                "source_generation_duration_seconds", panel.get("duration_seconds")
            )
            if source is None or abs(float(source) - 10.125) > 1e-6:
                raise ValueError(
                    f"episode {number} source_generation_duration_seconds must be 10.125"
                )
            panel["source_generation_duration_seconds"] = 10.125
            panel["duration_seconds"] = 10.125
    else:
        explicit = [panel.get("duration_seconds") for panel in panels]
        if any(value is not None for value in explicit) and not all(value is not None for value in explicit):
            raise ValueError(f"episode {number} must specify duration_seconds on every panel or none")
        if all(value is not None for value in explicit):
            planned = sum(float(value) for value in explicit)
            if abs(planned - target) > 1 / 24:
                raise ValueError(f"episode {number} duration {planned} does not equal required {target}")
        else:
            per_panel = target / len(panels)
            for panel in panels:
                panel["duration_seconds"] = per_panel
    for panel in panels:
        package = panel.setdefault("prompt_package", {})
        package.setdefault("render_settings", {})["duration_seconds"] = float(panel["duration_seconds"])
    if episode.get("target_duration_seconds") is not None and abs(float(episode["target_duration_seconds"]) - target) > 1 / 24:
        raise ValueError(f"episode {number} target_duration_seconds does not equal series episode_seconds")
    episode["target_duration_seconds"] = target
    if uses_shot_plan:
        episode.setdefault("render_settings", {})["target_edit_duration_seconds"] = target
    episode["story_bible"] = {
        **(series["spec"].get("story_bible") or {}),
        "series_theme": series["theme"], "series_synopsis": series["synopsis"],
        **(episode.get("story_bible") or {}),
    }
    episode["visual_bible"] = json.loads(json.dumps(series["spec"].get("visual_bible") or {}))
    episode["world_bible"] = json.loads(json.dumps(series["spec"].get("world_bible") or {}))
    episode["character_bible"] = json.loads(json.dumps(series["spec"].get("character_bible") or []))
    panel_ids = {str(panel.get("panel_id") or panel.get("name") or "") for panel in panels}
    episode["scene_bible"] = json.loads(json.dumps(series["spec"].get("scene_bible") or []))
    for scene in episode["scene_bible"]:
        scene_id = str(scene.get("scene_id") or "")
        scene["panel_ids"] = [
            panel_id for panel_id in sorted(panel_ids)
            if any(str(panel.get("panel_id") or panel.get("name") or "") == panel_id and str(panel.get("scene_id") or "") == scene_id for panel in panels)
        ]
    episode["series_context"] = {
        **(episode.get("series_context") or {}),
        "series_id": series_id, "episode_number": number,
        "predecessor_ep_id": None if number == 1 else f"{series_id}_ep_{number - 1:03d}",
        "episode_seconds": target, "shared_assets_hash": series.get("shared_assets_hash"),
        "series_sha256": series["spec"].get("v4_series_sha256"),
    }
    return episode


def register_episodes(series_id: str, episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Register exactly N episode contracts atomically at the series layer."""
    store = default_series_store()
    series = store.get_series(_safe(series_id, "series_id"))
    if not series or series["status"] != "approved":
        raise RuntimeError("series must be approved before episode registration")
    raw_rows = list(episodes)
    expected_count = int(series["episode_count"])
    if len(raw_rows) != expected_count:
        raise ValueError(f"exactly {expected_count} episodes are required; received {len(raw_rows)}")
    numbered: dict[int, Mapping[str, Any]] = {}
    for index, item in enumerate(raw_rows, 1):
        number = int(item.get("episode_number") or index)
        if number in numbered:
            raise ValueError(f"duplicate episode_number: {number}")
        numbered[number] = item
    if set(numbered) != set(range(1, expected_count + 1)):
        raise ValueError(f"episode_number must be exactly 1..{expected_count}")
    prepared: list[dict[str, Any]] = []
    for number in range(1, expected_count + 1):
        episode = _materialize_episode(series, numbered[number], number)
        episode["series_context"]["predecessor_ep_id"] = None if number == 1 else prepared[-1]["ep_id"]
        prepared.append(episode)
    # Only mutate per-episode stores after every contract has passed the exact
    # count/duration/schema preflight above.
    episode_rows: list[dict[str, Any]] = []
    existing_by_number = {item["episode_number"]: item for item in store.list_episodes(series_id)}
    for number, episode in enumerate(prepared, 1):
        creative_hash = task_store.contract_hash(episode)
        existing = existing_by_number.get(number)
        preserve_complete = bool(
            existing and existing["contract_hash"] == creative_hash
            and existing["status"] in {"succeeded", "exported"}
        )
        if not preserve_complete:
            snapshot = task_store.prepare_contract(episode["ep_id"], episode)
            task_store.approve_contract(episode["ep_id"], expected_hash=snapshot["pipeline"]["contract_hash"])
            if series.get("shared_assets_status") == "approved":
                task_store.approve_assets(episode["ep_id"])
        episode_rows.append({
            "episode_number": number, "ep_id": episode["ep_id"],
            "predecessor_ep_id": None if number == 1 else prepared[number - 2]["ep_id"],
            "contract_hash": creative_hash,
            "continuity_state_in": ({
                "contract_state": _copy(episode.get("continuity_state_in") or {}),
                "series_sha256": episode.get("series_sha256") or episode["series_context"].get("series_sha256"),
            } if isinstance(episode.get("continuity_state_in"), dict) else {}),
        })
    store.replace_episodes(series_id, episode_rows)
    return status_series(series_id)


def register_series_contract_episodes(
    series_id: str, v4: Mapping[str, Any], *, require_all_approved: bool = True,
) -> dict[str, Any]:
    """Register the exact ordered V3 episode set carried by a V4 envelope."""
    series_id = _safe(series_id, "series_id")
    contract = _validate_v4_contract(v4, require_all_approved=require_all_approved)
    declared_id = str((contract.get("series_bible") or {}).get("series_id") or "")
    if declared_id != series_id:
        raise ValueError(f"series_id mismatch: facade={series_id}, V4={declared_id or '<missing>'}")
    contracts = contract.get("episode_contracts") or {}
    outline = contract.get("season_outline") or []
    expected_ids = [str(item.get("episode_id") or "") for item in outline]
    if set(contracts) != set(expected_ids):
        raise ValueError("episode_contracts must contain exactly the season_outline episode IDs")

    # Persist the complete current V4 envelope first.  Because only the
    # structural V4 copy participates in series_contract_hash, adding V3
    # contracts/approvals preserves an already reviewed season approval.
    prepared = prepare_series_contract(series_id, contract)
    if prepared["series"]["status"] != "approved":
        raise RuntimeError("series must be approved before V4 episode registration")

    episodes: list[dict[str, Any]] = []
    for number, item in enumerate(outline, 1):
        episode_id = str(item["episode_id"])
        episode = _copy(contracts[episode_id])
        episode["episode_number"] = number
        # V4 episode IDs repeat in every season; task_store IDs must be global.
        episode["ep_id"] = f"{series_id}_{episode_id}"
        episode["series_episode_id"] = episode_id
        episode["series_sha256"] = contract["series_sha256"]
        episode["continuity_state_in"] = _copy(item.get("continuity_state_in") or {})
        episode["continuity_state_out"] = _copy(item.get("continuity_state_out") or {})
        episode["series_context"] = {
            **(episode.get("series_context") or {}),
            "series_episode_id": episode_id,
            "series_sha256": contract["series_sha256"],
            "continuity_state_in": _copy(item.get("continuity_state_in") or {}),
            "continuity_state_out": _copy(item.get("continuity_state_out") or {}),
            "season_outline": _copy(item),
        }
        episodes.append(episode)
    return register_episodes(series_id, episodes)


def _run_prepare_shared_assets_unlocked(
    series_id: str, *, character_generator=None, scene_generator=None,
) -> dict[str, Any]:
    """Generate only missing shared assets; injectable generators keep tests offline."""
    store = default_series_store()
    series = store.get_series(_safe(series_id, "series_id"))
    if not series or series["status"] != "approved":
        raise RuntimeError("series must be approved before shared asset generation")
    spec = series["spec"]
    project = series_project_dir(series_id)
    visual = spec.get("visual_bible") or {}
    source_map = {}
    for asset_type, collection, id_key in (
        ("character", spec.get("character_bible") or [], "character_id"),
        ("scene", spec.get("scene_bible") or [], "scene_id"),
    ):
        for item in collection:
            source_map[(asset_type, str(item.get(id_key) or item.get("id") or ""))] = item
    updated = []
    for asset in store.list_assets(series_id):
        if asset["status"] == "succeeded" and asset.get("reference_images"):
            continue
        source = source_map[(asset["asset_type"], asset["source_id"])]
        if asset["asset_type"] == "character":
            if character_generator is None:
                from generate_character_ref import generate_character_assets
                character_generator = generate_character_assets
            manifest = character_generator(source, visual, story_hash=series["contract_hash"])
            folder = project / "shared_assets" / "characters"
        else:
            if scene_generator is None:
                from scene_asset import generate_scene_asset
                scene_generator = generate_scene_asset
            manifest = scene_generator(source, visual, story_hash=series["contract_hash"])
            folder = project / "shared_assets" / "scenes"
        folder.mkdir(parents=True, exist_ok=True)
        persisted = []
        for index, value in enumerate(manifest.get("reference_images") or [], 1):
            src = Path(str(value))
            if not src.is_file():
                raise FileNotFoundError(src)
            destination = folder / f"{asset['source_id']}_{index:02d}{src.suffix.lower()}"
            if src.resolve() != destination.resolve():
                shutil.copy2(src, destination)
            persisted.append(str(destination.resolve()))
        if not persisted:
            raise RuntimeError(f"shared generator returned no references: {asset['asset_id']}")
        manifest_path = folder / f"{asset['source_id']}.manifest.json"
        _atomic_json(manifest_path, {**manifest, "reference_images": persisted})
        content_hash = _file_bundle_hash(persisted, project)
        store.update_asset(
            asset["asset_id"], status="succeeded", approved=False, content_hash=content_hash,
            reference_images=persisted, manifest_path=str(manifest_path), prompt_id=manifest.get("prompt_id"), error=None,
        )
        source["reference_images"] = persisted
        source["asset_status"] = "succeeded"
        source["asset_hash"] = content_hash
        source["asset_manifest_path"] = str(manifest_path)
        updated.append(asset["asset_id"])
    store.update_series(series_id, spec=spec, shared_assets_status="ready_for_approval", shared_assets_hash=None)
    _atomic_json(project / "series.json", spec)
    snapshot = status_series(series_id, reconcile=False)
    snapshot["shared_assets_updated"] = updated
    return snapshot


def run_prepare_shared_assets(
    series_id: str, *, character_generator=None, scene_generator=None,
) -> dict[str, Any]:
    store = default_series_store()
    if not store.acquire_worker(series_id):
        return {"series_id": series_id, "started": False, "reason": "series_worker_already_running"}
    try:
        return _run_prepare_shared_assets_unlocked(
            series_id, character_generator=character_generator, scene_generator=scene_generator,
        )
    finally:
        store.release_worker(series_id)


def _sync_shared_assets_to_episodes(series: Mapping[str, Any]) -> None:
    store = default_series_store()
    for record in store.list_episodes(series["series_id"]):
        episode_path = Path(task_store.project_snapshot(record["ep_id"])["project_dir"]) / "episode.json"
        if not episode_path.is_file():
            continue
        raw = json.loads(episode_path.read_text(encoding="utf-8"))
        episode = _materialize_episode(series, raw, int(record["episode_number"]))
        episode["series_context"]["predecessor_ep_id"] = record.get("predecessor_ep_id")
        episode["series_context"]["shared_assets_hash"] = series["shared_assets_hash"]
        snapshot = task_store.prepare_contract(record["ep_id"], episode)
        task_store.approve_contract(record["ep_id"], expected_hash=snapshot["pipeline"]["contract_hash"])
        task_store.approve_assets(record["ep_id"])
        episode_status = record["status"] if snapshot["jobs"] and all(
            job["status"] == "succeeded" for job in snapshot["jobs"]
        ) else "registered"
        store.update_episode(
            series["series_id"], record["episode_number"],
            contract_hash=task_store.contract_hash(episode), status=episode_status,
            continuity_state_out=record.get("continuity_state_out") if episode_status in {"succeeded", "exported"} else {},
            last_clip_path=record.get("last_clip_path") if episode_status in {"succeeded", "exported"} else None,
        )


def approve_shared_assets(series_id: str, *, expected_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    store = default_series_store()
    series = store.get_series(_safe(series_id, "series_id"))
    if not series or series["status"] != "approved":
        raise RuntimeError("series must be approved before shared assets")
    assets = store.list_assets(series_id)
    expected_hashes = dict(expected_hashes or {})
    for asset in assets:
        actual = _file_bundle_hash(asset.get("reference_images") or [], series_project_dir(series_id))
        if asset["status"] != "succeeded" or not actual or actual != asset.get("content_hash"):
            raise RuntimeError(f"shared asset is not ready: {asset['asset_id']}")
        if asset["asset_id"] in expected_hashes and expected_hashes[asset["asset_id"]] != actual:
            raise RuntimeError(f"shared asset changed since review: {asset['asset_id']}")
    bundle_hash = _json_hash([
        {"asset_type": item["asset_type"], "source_id": item["source_id"], "prompt_hash": item["prompt_hash"], "content_hash": item["content_hash"]}
        for item in assets
    ])
    for asset in assets:
        store.update_asset(asset["asset_id"], approved=True)
    series = store.update_series(series_id, shared_assets_status="approved", shared_assets_hash=bundle_hash)
    _sync_shared_assets_to_episodes(series)
    return status_series(series_id)


def reject_shared_asset(
    series_id: str, asset_id: str | None = None, *, asset_type: str | None = None,
    source_id: str | None = None, reason: str = "rejected by reviewer",
) -> dict[str, Any]:
    """Reject one season-level asset and invalidate only dependent episode jobs."""
    store = default_series_store()
    series = store.get_series(_safe(series_id, "series_id"))
    if not series:
        raise KeyError(series_id)
    assets = store.list_assets(series_id)
    asset = next((
        item for item in assets
        if (asset_id and item["asset_id"] == asset_id)
        or (not asset_id and item["asset_type"] == asset_type and item["source_id"] == source_id)
    ), None)
    if not asset:
        raise KeyError(f"unknown shared asset: {asset_id or f'{asset_type}/{source_id}'}")
    now = _now()
    metadata = dict(asset.get("metadata") or {})
    metadata["rejection_audit"] = [*(metadata.get("rejection_audit") or []), {
        "at": now, "reason": reason, "content_hash": asset.get("content_hash"),
        "reference_images": asset.get("reference_images") or [], "manifest_path": asset.get("manifest_path"),
    }]
    spec = series["spec"]
    collection = "character_bible" if asset["asset_type"] == "character" else "scene_bible"
    id_key = "character_id" if asset["asset_type"] == "character" else "scene_id"
    source = next(item for item in spec.get(collection) or [] if str(item.get(id_key) or item.get("id") or "") == asset["source_id"])
    source["asset_rejection_history"] = [*(source.get("asset_rejection_history") or []), metadata["rejection_audit"][-1]]
    source["reference_images"] = []
    source["asset_status"] = "queued"
    source["asset_hash"] = None
    source["asset_approval"] = {"state": "rejected", "reason": reason, "at": now}
    store.update_asset(
        asset["asset_id"], status="queued", approved=False, content_hash=None,
        reference_images=[], manifest_path=None, prompt_id=None, error=f"rejected: {reason}", metadata=metadata,
    )
    store.update_series(
        series_id, spec=spec, shared_assets_status="pending", shared_assets_hash=None,
    )
    _atomic_json(series_project_dir(series_id) / "series.json", spec)
    for episode in store.list_episodes(series_id):
        try:
            task_store.reject_asset(
                episode["ep_id"], asset_type=asset["asset_type"], source_id=asset["source_id"],
                reason=f"shared asset rejected: {reason}",
            )
        except KeyError:
            continue
        store.update_episode(series_id, episode["episode_number"], status="registered", error=None)
    return status_series(series_id)


def retry_shared_asset(
    series_id: str, asset_id: str | None = None, *, asset_type: str | None = None,
    source_id: str | None = None, reason: str = "manual retry",
) -> dict[str, Any]:
    return reject_shared_asset(
        series_id, asset_id, asset_type=asset_type, source_id=source_id, reason=reason,
    )


def _reconcile_episode(series_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    store = default_series_store()
    series = store.get_series(series_id) or {}
    snapshot = task_store.project_snapshot(record["ep_id"])
    jobs = snapshot["jobs"]
    if jobs and all(job["status"] == "succeeded" and job.get("output_path") and Path(job["output_path"]).is_file() for job in jobs):
        last = jobs[-1]
        outline_item = _v4_outline_item(series, int(record["episode_number"]))
        state_out = {
            **(record.get("continuity_state_out") or {}),
            **({"contract_state": _copy(outline_item.get("continuity_state_out") or {})} if outline_item else {}),
            "series_id": series_id, "episode_number": record["episode_number"],
            "last_job_id": last["job_id"], "last_panel_name": last["panel_name"],
            "last_clip_path": last["output_path"],
            "last_artifact_hash": (last.get("metadata") or {}).get("artifact_sha256"),
            "completed_at": last.get("completed_at"),
            "shared_assets_hash": series.get("shared_assets_hash"),
            "series_sha256": (series.get("spec") or {}).get("v4_series_sha256"),
            "episode_contract_hash": record.get("contract_hash"),
        }
        return store.update_episode(
            series_id, record["episode_number"],
            status="exported" if record.get("status") == "exported" else "succeeded",
            continuity_state_out=state_out, last_clip_path=last["output_path"], error=None,
        )
    if any(job["status"] == "failed" for job in jobs):
        return store.update_episode(series_id, record["episode_number"], status="failed", error="one or more panel jobs failed")
    if any(job["status"] in {"submitted", "running"} for job in jobs):
        return store.update_episode(series_id, record["episode_number"], status="running")
    if jobs and all(job["status"] == "cancelled" for job in jobs):
        return store.update_episode(series_id, record["episode_number"], status="cancelled")
    return dict(record)


def _hard_gate(series_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    store = default_series_store()
    series = store.get_series(_safe(series_id, "series_id"))
    if not series or series["status"] != "approved":
        raise RuntimeError("series_not_approved")
    episodes = store.list_episodes(series_id)
    if len(episodes) != int(series["episode_count"]):
        raise RuntimeError("exact_episode_count_not_registered")
    if [item["episode_number"] for item in episodes] != list(range(1, int(series["episode_count"]) + 1)):
        raise RuntimeError("episode_sequence_is_not_contiguous")
    v4 = _stored_v4(series)
    outline: list[dict[str, Any]] = []
    if v4:
        try:
            _validate_v4_contract(v4, require_all_approved=True)
        except ValueError as exc:
            raise RuntimeError(f"v4_contract_gate_failed:{exc}") from exc
        if series_contract_hash(_series_spec_from_v4(v4)) != series.get("contract_hash"):
            raise RuntimeError("v4_structural_hash_changed_since_approval")
        if str(v4.get("series_sha256") or "") != str((series.get("spec") or {}).get("v4_series_sha256") or ""):
            raise RuntimeError("v4_shared_contract_hash_mismatch")
        outline = v4.get("season_outline") or []
    for index, episode in enumerate(episodes):
        expected_predecessor = None if index == 0 else episodes[index - 1]["ep_id"]
        if episode.get("predecessor_ep_id") != expected_predecessor:
            raise RuntimeError(f"cross_episode_continuity_chain_broken:{episode['episode_number']}")
        if outline:
            expected_in = outline[index].get("continuity_state_in") or {}
            persisted_in = episode.get("continuity_state_in") or {}
            if persisted_in.get("contract_state") != expected_in:
                raise RuntimeError(f"cross_episode_contract_state_in_drift:{episode['episode_number']}")
    assets = store.list_assets(series_id)
    if series["shared_assets_status"] != "approved" or any(not item["approved"] for item in assets):
        raise RuntimeError("shared_assets_not_approved")
    for asset in assets:
        actual = _file_bundle_hash(asset.get("reference_images") or [], series_project_dir(series_id))
        if not actual or actual != asset.get("content_hash"):
            raise RuntimeError(f"shared_asset_hash_changed:{asset['asset_id']}")
    return series, episodes


def _tail_frame(previous: Mapping[str, Any], last_clip: Path) -> Path:
    state = previous.get("continuity_state_out") or {}
    existing = Path(str(state.get("last_frame_path") or ""))
    if existing.is_file():
        return existing.resolve()
    destination = series_project_dir(previous["series_id"]) / "continuity" / f"ep_{int(previous['episode_number']):03d}_tail.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    subprocess.run([
        ffmpeg_executable(), "-y", "-sseof", "-0.08", "-i", str(last_clip),
        "-frames:v", "1", "-update", "1", str(temporary),
    ], check=True, capture_output=True, text=True)
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError(f"failed to extract previous episode tail: {last_clip}")
    temporary.replace(destination)
    return destination


def _bind_previous_episode(series: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    if int(record["episode_number"]) == 1:
        return
    store = default_series_store()
    previous = store.get_episode(series["series_id"], int(record["episode_number"]) - 1)
    if not previous or previous["status"] not in {"succeeded", "exported"}:
        raise RuntimeError(f"previous_episode_not_succeeded:{int(record['episode_number']) - 1}")
    state = previous.get("continuity_state_out") or {}
    last_clip = Path(str(state.get("last_clip_path") or previous.get("last_clip_path") or ""))
    if not state or not last_clip.is_file():
        raise RuntimeError("previous_episode_continuity_state_is_incomplete")
    if state.get("shared_assets_hash") != series.get("shared_assets_hash"):
        raise RuntimeError("previous_episode_continuity_hash_mismatch")
    outline_item = _v4_outline_item(series, int(record["episode_number"]))
    if outline_item:
        if state.get("contract_state") != (outline_item.get("continuity_state_in") or {}):
            raise RuntimeError("previous_episode_contract_state_mismatch")
        if state.get("series_sha256") != (series.get("spec") or {}).get("v4_series_sha256"):
            raise RuntimeError("previous_episode_series_hash_mismatch")
    last_frame = _tail_frame(previous, last_clip)
    state = {**state, "last_frame_path": str(last_frame)}
    store.update_episode(
        series["series_id"], previous["episode_number"], continuity_state_out=state,
    )
    episode_path = Path(task_store.project_snapshot(record["ep_id"])["project_dir"]) / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    first_panel = episode["panels"][0]
    first_panel["previous_episode_last_clip"] = str(last_clip.resolve())
    first_panel["series_continuity_state_in"] = state
    first_panel["first_frame_path"] = str(last_frame)
    first_panel["reference_images"] = list(dict.fromkeys([
        *(first_panel.get("reference_images") or []), str(last_frame),
    ]))
    episode.setdefault("series_context", {})["continuity_state_in"] = state
    snapshot = task_store.prepare_contract(record["ep_id"], episode)
    task_store.approve_contract(record["ep_id"], expected_hash=snapshot["pipeline"]["contract_hash"])
    task_store.approve_assets(record["ep_id"])
    store.update_episode(
        series["series_id"], record["episode_number"],
        continuity_state_in=state, status="registered",
    )


def run_series_production(
    series_id: str, *, episode_numbers: Iterable[int] | None = None,
    episode_runner=worker.run_worker, timeout: float = 2400.0,
) -> dict[str, Any]:
    """Run selected episodes synchronously in a background process, in order."""
    store = default_series_store()
    if not store.acquire_worker(series_id):
        return {"series_id": series_id, "started": False, "reason": "series_worker_already_running"}
    try:
        series, episodes = _hard_gate(series_id)
        wanted = set(int(value) for value in episode_numbers) if episode_numbers else set(range(1, len(episodes) + 1))
        for record in episodes:
            number = int(record["episode_number"])
            if number not in wanted:
                continue
            record = _reconcile_episode(series_id, record)
            if record["status"] in {"succeeded", "exported"}:
                continue
            _bind_previous_episode(series, record)
            store.update_episode(series_id, number, status="running", error=None)
            result = episode_runner(record["ep_id"], timeout=timeout)
            if not result.get("started", True):
                store.update_episode(series_id, number, status="failed", error=result.get("reason"))
                break
            reconciled = _reconcile_episode(series_id, store.get_episode(series_id, number) or record)
            if reconciled["status"] != "succeeded":
                break
            store.heartbeat(series_id)
        return {"series_id": series_id, "started": True, "snapshot": status_series(series_id)}
    finally:
        store.release_worker(series_id)


def _start_process(series_id: str, *, episode_numbers: Iterable[int] | None, timeout: float, shared_assets_only: bool) -> dict[str, Any]:
    active = default_series_store().worker_info(series_id)
    if active and active["active"]:
        return {"series_id": series_id, "started": False, "reason": "series_worker_already_running", "pid": active["pid"]}
    logs = project_root() / "logs" / "series_workers"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{series_id}.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--series-id", series_id, "--timeout", str(timeout)]
    if shared_assets_only:
        command.append("--shared-assets-only")
    elif episode_numbers:
        command.extend(["--episodes", ",".join(str(value) for value in episode_numbers)])
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=str(project_root()), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, creationflags=flags, close_fds=True, env=env,
        )
    return {"series_id": series_id, "started": True, "pid": process.pid, "log_path": str(log_path), "command": command}


def prepare_shared_assets(series_id: str, *, timeout: float = 1800.0) -> dict[str, Any]:
    return _start_process(series_id, episode_numbers=None, timeout=timeout, shared_assets_only=True)


def start_episode(series_id: str, episode_number: int, *, timeout: float = 2400.0) -> dict[str, Any]:
    series, episodes = _hard_gate(series_id)
    number = int(episode_number)
    if number < 1 or number > len(episodes):
        raise KeyError(f"{series_id}/{number}")
    if number > 1:
        previous = _reconcile_episode(series_id, episodes[number - 2])
        state = previous.get("continuity_state_out") or {}
        clip = Path(str(state.get("last_clip_path") or previous.get("last_clip_path") or ""))
        if previous["status"] not in {"succeeded", "exported"} or not state or not clip.is_file():
            raise RuntimeError(f"previous_episode_not_succeeded:{number - 1}")
        if state.get("shared_assets_hash") != series.get("shared_assets_hash"):
            raise RuntimeError("previous_episode_continuity_hash_mismatch")
        outline_item = _v4_outline_item(series, number)
        if outline_item and state.get("contract_state") != (outline_item.get("continuity_state_in") or {}):
            raise RuntimeError("previous_episode_contract_state_mismatch")
    return _start_process(series_id, episode_numbers=[number], timeout=timeout, shared_assets_only=False)


def start_series(series_id: str, *, episode_numbers: Iterable[int] | None = None, timeout: float = 2400.0) -> dict[str, Any]:
    _hard_gate(series_id)
    return _start_process(series_id, episode_numbers=episode_numbers, timeout=timeout, shared_assets_only=False)


def resume_episode(series_id: str, episode_number: int, *, start: bool = False, timeout: float = 2400.0) -> dict[str, Any]:
    record = default_series_store().get_episode(series_id, episode_number)
    if not record:
        raise KeyError(f"{series_id}/{episode_number}")
    task_store.resume_jobs(record["ep_id"], statuses=("pending", "failed", "cancelled"))
    default_series_store().update_episode(series_id, episode_number, status="registered", error=None)
    return start_episode(series_id, episode_number, timeout=timeout) if start else status_series(series_id)


def resume_series(series_id: str, *, start: bool = False, timeout: float = 2400.0) -> dict[str, Any]:
    for record in default_series_store().list_episodes(series_id):
        if record["status"] not in {"succeeded", "exported"}:
            resume_episode(series_id, record["episode_number"], start=False)
    return start_series(series_id, timeout=timeout) if start else status_series(series_id)


def retry_episode(series_id: str, episode_number: int, *, start: bool = False, timeout: float = 2400.0) -> dict[str, Any]:
    record = default_series_store().get_episode(series_id, episode_number)
    if not record:
        raise KeyError(f"{series_id}/{episode_number}")
    for job in task_store.list_jobs(record["ep_id"]):
        if job["status"] in {"failed", "cancelled"}:
            task_store.retry_job(record["ep_id"], job["job_id"])
    default_series_store().update_episode(series_id, episode_number, status="registered", error=None)
    return start_episode(series_id, episode_number, timeout=timeout) if start else status_series(series_id)


def retry_series(series_id: str, *, start: bool = False, timeout: float = 2400.0) -> dict[str, Any]:
    for record in default_series_store().list_episodes(series_id):
        if record["status"] not in {"succeeded", "exported"}:
            retry_episode(series_id, record["episode_number"], start=False)
    return start_series(series_id, timeout=timeout) if start else status_series(series_id)


def cancel_episode(
    series_id: str, episode_number: int, *, interrupt_running: bool = False,
) -> dict[str, Any]:
    record = default_series_store().get_episode(series_id, episode_number)
    if not record:
        raise KeyError(f"{series_id}/{episode_number}")
    store = task_store.default_store()
    for job in task_store.list_jobs(record["ep_id"]):
        if job["status"] not in {"succeeded", "cancelled"}:
            if job.get("prompt_id"):
                from render_video_h3 import cancel_render_job
                cancel_render_job(job["job_id"], store=store, interrupt_running=interrupt_running)
            else:
                store.update_job(job["job_id"], status="cancelled", error="cancelled by series controller")
    default_series_store().update_episode(series_id, episode_number, status="cancelled", error="cancelled by user")
    return status_series(series_id)


def cancel_series(series_id: str, *, interrupt_running: bool = False) -> dict[str, Any]:
    for record in default_series_store().list_episodes(series_id):
        if record["status"] not in {"succeeded", "exported"}:
            cancel_episode(series_id, record["episode_number"], interrupt_running=interrupt_running)
    return status_series(series_id)


def status_series(series_id: str, *, reconcile: bool = True) -> dict[str, Any]:
    store = default_series_store()
    series = store.get_series(_safe(series_id, "series_id"))
    if not series:
        raise KeyError(series_id)
    episodes = store.list_episodes(series_id)
    if reconcile:
        episodes = [_reconcile_episode(series_id, record) for record in episodes]
    assets = store.list_assets(series_id)
    complete = sum(item["status"] in {"succeeded", "exported"} for item in episodes)
    v4 = _stored_v4(series)
    v4_errors: list[str] = []
    if v4:
        try:
            _validate_v4_contract(v4, require_all_approved=False)
        except ValueError as exc:
            v4_errors.append(str(exc))
    return {
        "series_id": series_id, "series": series, "episodes": episodes, "shared_assets": assets,
        "series_contract_v4": v4,
        "v4_validation_errors": v4_errors,
        "counts": {"expected": series["episode_count"], "registered": len(episodes), "complete": complete},
        "ready": bool(
            series["status"] == "approved" and series["shared_assets_status"] == "approved"
            and len(episodes) == series["episode_count"] and not v4_errors
        ),
    }


def export_episode(series_id: str, episode_number: int, preset: str, **kwargs: Any) -> dict[str, Any]:
    store = default_series_store()
    record = store.get_episode(series_id, episode_number)
    if not record:
        raise KeyError(f"{series_id}/{episode_number}")
    record = _reconcile_episode(series_id, record)
    if record["status"] not in {"succeeded", "exported"}:
        raise RuntimeError("episode must succeed before export")
    manifest = video_delivery.export_episode(record["ep_id"], preset, **kwargs)
    store.update_episode(series_id, episode_number, status="exported", delivery_manifest=manifest)
    return manifest


def export_season(
    series_id: str, preset: str, *, export_missing: bool = True,
    episode_exporter=export_episode,
) -> dict[str, Any]:
    snapshot = status_series(series_id)
    deliveries = []
    for record in snapshot["episodes"]:
        manifest = record.get("delivery_manifest") or {}
        if not manifest and export_missing:
            manifest = episode_exporter(series_id, record["episode_number"], preset)
        if not manifest or not Path(str(manifest.get("output_path") or "")).is_file():
            raise RuntimeError(f"episode {record['episode_number']} has no complete delivery")
        deliveries.append({"episode_number": record["episode_number"], "ep_id": record["ep_id"], **manifest})
    if len(deliveries) != int(snapshot["series"]["episode_count"]):
        raise RuntimeError("season package requires every episode delivery")
    root = series_project_dir(series_id) / "exports"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / f"{series_id}_{preset}.season.json"
    package_path = root / f"{series_id}_{preset}.season.zip"
    season_manifest = {
        "schema_version": 1, "series_id": series_id, "preset": preset,
        "episode_count": snapshot["series"]["episode_count"],
        "episode_seconds": snapshot["series"]["episode_seconds"],
        "shared_assets_hash": snapshot["series"]["shared_assets_hash"],
        "series_sha256": (snapshot.get("series_contract_v4") or {}).get("series_sha256"),
        "v4_shared_contract_hash": (snapshot["series"].get("spec") or {}).get("v4_shared_contract_hash"),
        "episodes": deliveries, "package_path": str(package_path),
    }
    _atomic_json(manifest_path, season_manifest)
    temporary = package_path.with_suffix(package_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(manifest_path, "season_manifest.json")
        series_json = series_project_dir(series_id) / "series.json"
        if series_json.is_file():
            bundle.write(series_json, "series.json")
        for delivery in deliveries:
            number = int(delivery["episode_number"])
            bundle.write(delivery["output_path"], f"episodes/{number:03d}/final.mp4")
            for key in ("manifest_path", "subtitle_vtt"):
                path = Path(str(delivery.get(key) or ""))
                if path.is_file():
                    bundle.write(path, f"episodes/{number:03d}/{path.name}")
    temporary.replace(package_path)
    return {**season_manifest, "manifest_path": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="AI manga series worker")
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--episodes", default="")
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--shared-assets-only", action="store_true")
    args = parser.parse_args()
    if args.shared_assets_only:
        result = run_prepare_shared_assets(args.series_id)
    else:
        numbers = [int(value) for value in args.episodes.split(",") if value.strip()] or None
        result = run_series_production(args.series_id, episode_numbers=numbers, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "prepare_series", "approve_series", "register_episodes", "prepare_shared_assets",
    "run_prepare_shared_assets", "approve_shared_assets", "start_episode", "start_series",
    "reject_shared_asset", "retry_shared_asset",
    "run_series_production", "resume_episode", "resume_series", "retry_episode", "retry_series",
    "cancel_episode", "cancel_series", "status_series", "export_episode", "export_season",
]
