"""Durable render-job store shared by CLI workers and the Streamlit UI.

SQLite is used instead of Streamlit session state or a rewritten JSONL file so
multiple processes can safely register a complete episode before GPU work
starts.  Public module-level functions intentionally return plain dictionaries
for easy use from Streamlit.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
import hashlib
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from runtime_config import projects_dir, render_job_db
from atomic_io import write_json_atomic as _write_json_atomic
from h3_profiles import (
    DEFAULT_PRODUCTION_STRATEGY,
    H3_RENDER_PROFILE_CONTRACT,
    apply_render_profile,
)
from video_quality import validate_edit_selection


JOB_STATUSES = {
    "queued", "submitted", "running", "succeeded", "failed", "cancelled",
}
RESUMABLE_STATUSES = {"queued", "failed", "cancelled", "pending", "timed_out"}
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_JSON_FIELDS = {"reference_images", "metadata", "probe", "dialogue_cues", "audio_cues"}
_ASSET_JSON_FIELDS = {"reference_images", "metadata"}
RENDER_INPUT_SCHEMA_VERSION = "render-job/v5-source-edit-selection"
SOURCE_GENERATION_DURATION_SECONDS = 10.125
MIN_EDIT_DURATION_SECONDS = 1.5
MAX_EDIT_DURATION_SECONDS = 4.0


def _utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, separators=(",", ":"))


def _safe_id(value: str, label: str = "id") -> str:
    text = str(value).strip()
    if (
        not text or text in {".", ".."}
        or "/" in text or "\\" in text
        or Path(text).is_absolute()
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", text)
    ):
        raise ValueError(f"{label} must contain only letters, digits, dot, underscore or hyphen")
    return text


def _safe_panel_name(panel: Mapping[str, Any], panel_index: int) -> str:
    raw = str(panel.get("panel_id") or panel.get("name") or f"panel_{panel_index:03d}")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return (name or f"panel_{panel_index:03d}")[:96]


def _validate_job_edit_selection(job: Mapping[str, Any], metadata: Mapping[str, Any], artifact_sha256: str) -> None:
    """Revalidate a selection against the durable job dialogue before review/release."""
    selection = metadata.get("edit_selection") if isinstance(metadata.get("edit_selection"), Mapping) else {}
    # Pre-v5 selections were approved before selector identity/probe persistence
    # existed.  Keep them readable for historical proof promotion; any native
    # dialogue rebase is never grandfathered and always takes the strict path.
    if not selection.get("dialogue_audio_alignment") and not (selection.get("selector") or {}):
        return
    settings = metadata.get("settings") if isinstance(metadata.get("settings"), Mapping) else {}
    shot_plan = (
        (metadata.get("inputs") or {}).get("shot_plan")
        if isinstance(metadata.get("inputs"), Mapping) else {}
    ) or {}
    requested = shot_plan.get("edit_duration_seconds") or settings.get("edit_duration_seconds")
    if requested is None:
        raise RuntimeError("approved edit duration is missing")
    check = validate_edit_selection(
        selection,
        source_artifact_sha256=artifact_sha256,
        requested_duration_seconds=float(requested),
        source_duration_seconds=float((job.get("probe") or {}).get("duration_seconds") or 0),
        current_dialogue_cues=[
            cue for cue in (job.get("dialogue_cues") or []) if isinstance(cue, Mapping)
        ],
    )
    if not check["valid"]:
        raise RuntimeError("edit selection is stale or invalid: " + ",".join(check["errors"]))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _contract_payload(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Return the creative contract without generated/runtime asset fields."""
    payload = json.loads(json.dumps(dict(episode), ensure_ascii=False))
    for collection in ("character_bible", "scene_bible"):
        for item in payload.get(collection) or []:
            if isinstance(item, dict):
                for key in (
                    "reference_images", "asset_status", "asset_manifest", "asset_hash",
                    "asset_manifest_path", "asset_approval", "approved", "approved_at",
                    "asset_rejection_history", "prompt_id", "error",
                ):
                    item.pop(key, None)
    for key in ("pipeline", "jobs", "assets", "deliveries"):
        payload.pop(key, None)
    return payload


def contract_hash(episode: Mapping[str, Any]) -> str:
    return _sha256_json(_contract_payload(episode))


def _contract_errors(episode: Mapping[str, Any]) -> list[str]:
    """Validate production-critical cross references without silently rebinding."""
    errors: list[str] = []
    panels = episode.get("panels") or []
    scenes = episode.get("scene_bible") or []
    characters = episode.get("character_bible") or []
    render_settings = episode.get("render_settings") or {}
    target_edit_duration = (
        render_settings.get("target_edit_duration_seconds")
        if isinstance(render_settings, Mapping) else None
    )
    strict_edit_plan = target_edit_duration is not None or any(
        isinstance(panel, Mapping) and panel.get("edit_duration_seconds") is not None
        for panel in panels
    )
    edit_total = 0.0
    panel_ids: list[str] = []
    for index, panel in enumerate(panels, 1):
        if not isinstance(panel, Mapping):
            errors.append(f"panels[{index - 1}] is not an object")
            continue
        panel_id = str(panel.get("panel_id") or panel.get("name") or f"panel_{index:03d}")
        panel_ids.append(panel_id)
        if strict_edit_plan:
            try:
                source_duration = float(panel.get("source_generation_duration_seconds"))
            except (TypeError, ValueError):
                source_duration = 0.0
            try:
                edit_duration = float(panel.get("edit_duration_seconds"))
            except (TypeError, ValueError):
                edit_duration = 0.0
            if abs(source_duration - SOURCE_GENERATION_DURATION_SECONDS) > 1e-6:
                errors.append(
                    f"panel {panel_id} source_generation_duration_seconds must equal "
                    f"{SOURCE_GENERATION_DURATION_SECONDS}"
                )
            if not MIN_EDIT_DURATION_SECONDS <= edit_duration <= MAX_EDIT_DURATION_SECONDS:
                errors.append(
                    f"panel {panel_id} edit_duration_seconds must be between "
                    f"{MIN_EDIT_DURATION_SECONDS} and {MAX_EDIT_DURATION_SECONDS}"
                )
            edit_total += edit_duration
            for field in ("shot_role", "story_beat_id", "visible_action", "first_state", "final_state"):
                if not str(panel.get(field) or "").strip():
                    errors.append(f"panel {panel_id} requires {field} in strict edit plan")
            camera_plan = panel.get("camera_plan")
            if not isinstance(camera_plan, Mapping) or not str(camera_plan.get("movement") or "").strip():
                errors.append(f"panel {panel_id} requires camera_plan.movement in strict edit plan")
    if len(panel_ids) != len(set(panel_ids)):
        errors.append("panel_id values must be unique")
    if strict_edit_plan:
        try:
            target_value = float(target_edit_duration)
        except (TypeError, ValueError):
            target_value = 0.0
        if target_value <= 0:
            errors.append("render_settings.target_edit_duration_seconds must be positive")
        elif abs(edit_total - target_value) > (1.0 / 24.0):
            errors.append(
                "sum(panel.edit_duration_seconds) must equal "
                "render_settings.target_edit_duration_seconds"
            )

    scene_ids = [
        str(scene.get("scene_id") or "") for scene in scenes if isinstance(scene, Mapping)
    ]
    if len(scene_ids) != len(scenes):
        errors.append("every scene_bible entry must be an object with scene_id")
    if any(not re.fullmatch(r"scene_[a-z0-9_]+", scene_id) for scene_id in scene_ids):
        errors.append("scene_id must match scene_[a-z0-9_]+")
    if len(scene_ids) != len(set(scene_ids)):
        errors.append("scene_id values must be unique")
    scene_set = set(scene_ids)

    character_ids = [
        str(card.get("character_id") or card.get("id") or "")
        for card in characters if isinstance(card, Mapping)
    ]
    if len(character_ids) != len(characters) or any(not value for value in character_ids):
        errors.append("every character_bible entry must have character_id")
    if len(character_ids) != len(set(character_ids)):
        errors.append("character_id values must be unique")
    character_set = set(character_ids)

    referenced_by_scene: dict[str, set[str]] = {scene_id: set() for scene_id in scene_ids}
    for index, panel in enumerate(panels, 1):
        if not isinstance(panel, Mapping):
            continue
        panel_id = str(panel.get("panel_id") or panel.get("name") or f"panel_{index:03d}")
        scene_id = str(panel.get("scene_id") or "")
        if scenes and scene_id not in scene_set:
            errors.append(f"panel {panel_id} references unknown scene_id {scene_id or '<empty>'}")
        elif scene_id:
            referenced_by_scene.setdefault(scene_id, set()).add(panel_id)
        package = panel.get("prompt_package") or {}
        if isinstance(package, Mapping) and package.get("scene_id") and str(package["scene_id"]) != scene_id:
            errors.append(f"panel {panel_id} prompt_package.scene_id does not match panel.scene_id")
        for character_id in panel.get("character_ids") or []:
            if str(character_id) not in character_set:
                errors.append(f"panel {panel_id} references unknown character_id {character_id}")

    panel_set = set(panel_ids)
    for scene in scenes:
        if not isinstance(scene, Mapping):
            continue
        scene_id = str(scene.get("scene_id") or "")
        declared = {str(value) for value in scene.get("panel_ids") or []}
        unknown = declared - panel_set
        if unknown:
            errors.append(f"scene {scene_id} panel_ids contain unknown panels: {sorted(unknown)}")
        if declared and declared != referenced_by_scene.get(scene_id, set()):
            errors.append(f"scene {scene_id} panel_ids are not symmetric with panel.scene_id")
    if str(episode.get("schema_version") or "") == "ai-manga.prompt-package/v3":
        # V3 carries dialogue, prompt-package and continuity constraints that
        # are production-critical. The durable gate must enforce the same
        # validator shown in Web; UI warnings alone are not a safety boundary.
        for index, panel in enumerate(panels, 1):
            if isinstance(panel, Mapping) and not panel.get("character_ids"):
                panel_id = str(panel.get("panel_id") or panel.get("name") or f"panel_{index:03d}")
                errors.append(f"panel {panel_id} must reference at least one visible character")
        from story_splitter import validate_episode_contract
        errors.extend(validate_episode_contract(dict(episode)))
    return list(dict.fromkeys(errors))


def _resolve_reference(value: Any, project: Path) -> Optional[Path]:
    if isinstance(value, Mapping):
        value = value.get("source_path") or value.get("path") or value.get("staged_name")
    if not value:
        return None
    raw = Path(str(value))
    candidates = [raw] if raw.is_absolute() else [project / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _reference_bundle_hash(values: Iterable[Any], project: Path) -> Optional[str]:
    records = []
    for value in values:
        path = _resolve_reference(value, project)
        if not path:
            return None
        records.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return _sha256_json(records) if records else None


def _canonical_prompt_text(value: Any) -> str:
    """Normalize semantically duplicate prompt tags introduced by contract re-enrichment."""
    seen: set[str] = set()
    result: list[str] = []
    for token in re.split(r"[,;\n]+", str(value or "")):
        cleaned = re.sub(r"\s+", " ", token).strip(" .")
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return ", ".join(result)


def _canonical_prompt_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_prompt_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        result = []
        seen: set[str] = set()
        for item in value:
            normalized = _canonical_prompt_value(item)
            marker = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if marker not in seen:
                seen.add(marker)
                result.append(normalized)
        return result
    if isinstance(value, str):
        return _canonical_prompt_text(value)
    return value


def _render_hash_value(value: Any, field_name: str = "") -> Any:
    """Canonicalize prompt tag lanes without changing dialogue/timeline text."""
    if isinstance(value, Mapping):
        return {
            str(key): _render_hash_value(item, str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_render_hash_value(item, field_name) for item in value]
    if isinstance(value, str) and field_name in {"negative_prompt", "global_negative_prompt"}:
        return _canonical_prompt_text(value)
    return value


def _prompt_package_hash_source(package: Any) -> Any:
    normalized = _render_hash_value(package)
    if not isinstance(normalized, dict):
        return normalized
    structured_fields = {
        "first_frame_prompt", "last_frame_prompt", "camera_timeline",
        "character_prompts", "spoken_dialogue_timeline", "sound_timeline",
    }
    if normalized.get("schema_version") and structured_fields.intersection(normalized):
        # V3 positive_prompt is a derived concatenation of the structured
        # fields plus scene/character asset prompts. Re-enrichment may repeat
        # those inherited tags even though no panel model input changed.
        normalized.pop("positive_prompt", None)
    return normalized


def _asset_prompt_source(
    asset_type: str, source_id: str, source: Mapping[str, Any], visual: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile only fields actually consumed by the character/scene asset model.

    Panel membership, voice/performance notes, approvals and other episode
    bookkeeping deliberately do not participate in an asset identity hash.
    """
    if asset_type == "character":
        from prompt_contracts import build_character_reference_prompt

        compiled = build_character_reference_prompt(dict(source), dict(visual), view="anchor")
        return {
            "schema": "asset-prompt/v2-character-model-input",
            "asset_type": asset_type,
            "source_id": source_id,
            "positive_prompt": _canonical_prompt_text(compiled.get("positive_prompt")),
            "negative_prompt": _canonical_prompt_text(compiled.get("negative_prompt")),
            # These fields also drive stable_character_seed before graph build.
            "seed_identity": _canonical_prompt_text(source.get("identity_prompt")),
            "seed_wardrobe": _canonical_prompt_value(
                source.get("wardrobe_prompt") or source.get("wardrobe_lock") or ""
            ),
        }
    scene_fields = {
        key: _canonical_prompt_value(source.get(key))
        for key in (
            "description", "positive_prompt", "negative_prompt", "model_prompt_en",
            "model_environment_tags_en", "continuity_lock", "palette",
        )
        if source.get(key) not in (None, "", [], {})
    }
    visual_fields = {
        key: _canonical_prompt_value(visual.get(key))
        for key in ("aspect_ratio", "style_prompt", "global_negative_prompt")
        if visual.get(key) not in (None, "", [], {})
    }
    return {
        "schema": "asset-prompt/v2-scene-model-input",
        "asset_type": asset_type,
        "source_id": source_id,
        "source": scene_fields,
        "visual": visual_fields,
    }


def _stored_asset_prompt_hash(metadata: Mapping[str, Any]) -> Optional[str]:
    """Translate a legacy stored prompt_source into the semantic v2 hash."""
    prompt_source = metadata.get("prompt_source")
    if not isinstance(prompt_source, Mapping):
        return None
    if str(metadata.get("prompt_schema") or "").startswith("asset-prompt/v2"):
        return _sha256_json(prompt_source)
    asset_type = str(prompt_source.get("asset_type") or "")
    source = prompt_source.get("source")
    visual = prompt_source.get("visual_bible")
    if asset_type not in {"character", "scene"} or not isinstance(source, Mapping):
        return None
    source_id = str(
        source.get("character_id" if asset_type == "character" else "scene_id") or ""
    )
    try:
        compiled = _asset_prompt_source(
            asset_type, source_id, source, visual if isinstance(visual, Mapping) else {},
        )
    except (TypeError, ValueError):
        return None
    return _sha256_json(compiled)


def _asset_specs(ep_id: str, episode: Mapping[str, Any], project: Path) -> list[dict[str, Any]]:
    visual = episode.get("visual_bible") or {}
    specs: list[dict[str, Any]] = []
    for asset_type, collection, id_key in (
        ("character", episode.get("character_bible") or [], "character_id"),
        ("scene", episode.get("scene_bible") or [], "scene_id"),
    ):
        for index, source in enumerate(collection, 1):
            if not isinstance(source, Mapping):
                continue
            source_id = str(source.get(id_key) or f"{asset_type}_{index:02d}")
            refs = list(source.get("reference_images") or [])
            prompt_source = _asset_prompt_source(asset_type, source_id, source, visual)
            specs.append({
                "asset_id": f"{ep_id}:{asset_type}:{source_id}",
                "asset_type": asset_type,
                "source_id": source_id,
                "prompt_hash": _sha256_json(prompt_source),
                "content_hash": _reference_bundle_hash(refs, project),
                "reference_images": refs,
                "metadata": {
                    "prompt_schema": str(prompt_source["schema"]),
                    "prompt_source": prompt_source,
                },
            })
    return specs


class RenderJobStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or render_job_db()).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS render_jobs (
                    job_id TEXT PRIMARY KEY,
                    ep_id TEXT NOT NULL,
                    panel_index INTEGER NOT NULL,
                    panel_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0,
                    prompt_id TEXT,
                    reference_images TEXT NOT NULL DEFAULT '[]',
                    output_path TEXT,
                    preview_path TEXT,
                    comfy_output_path TEXT,
                    graph_path TEXT,
                    timing_path TEXT,
                    error TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 2,
                    input_hash TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    probe TEXT NOT NULL DEFAULT '{}',
                    dialogue_cues TEXT NOT NULL DEFAULT '[]',
                    audio_cues TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    submitted_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_render_jobs_episode
                    ON render_jobs(ep_id, panel_index);
                CREATE INDEX IF NOT EXISTS idx_render_jobs_status
                    ON render_jobs(ep_id, status);
                CREATE TABLE IF NOT EXISTS workers (
                    ep_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    heartbeat REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episode_pipeline (
                    ep_id TEXT PRIMARY KEY,
                    contract_hash TEXT NOT NULL,
                    contract_status TEXT NOT NULL DEFAULT 'draft',
                    contract_approved_at TEXT,
                    assets_hash TEXT,
                    assets_status TEXT NOT NULL DEFAULT 'pending',
                    assets_approved_at TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS production_assets (
                    asset_id TEXT PRIMARY KEY,
                    ep_id TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    approved INTEGER NOT NULL DEFAULT 0,
                    prompt_hash TEXT NOT NULL,
                    content_hash TEXT,
                    reference_images TEXT NOT NULL DEFAULT '[]',
                    manifest_path TEXT,
                    prompt_id TEXT,
                    error TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 2,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    approved_at TEXT,
                    UNIQUE(ep_id, asset_type, source_id)
                );
                CREATE INDEX IF NOT EXISTS idx_production_assets_episode
                    ON production_assets(ep_id, asset_type, source_id);
                """
            )

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        result = dict(row)
        for field in _JSON_FIELDS:
            raw = result.get(field)
            try:
                result[field] = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                result[field] = [] if field.endswith("_cues") or field == "reference_images" else {}
        result["progress"] = max(0.0, min(1.0, float(result.get("progress") or 0.0)))
        return result

    def register_jobs(
        self,
        ep_id: str,
        jobs: Iterable[Mapping[str, Any]],
        *,
        prune_missing: bool = True,
    ) -> list[dict[str, Any]]:
        """Register jobs atomically.

        ``prune_missing`` is only appropriate when the caller owns the complete
        episode job set.  A single-panel runtime submission must upsert that job
        without deleting its queued siblings.
        """
        now = _utc_now()
        rows = list(jobs)
        active_ids = [
            str(item.get("job_id") or f"{ep_id}:{int(item.get('panel_index', index)):04d}:{item.get('panel_name') or f'panel_{index:03d}'}")
            for index, item in enumerate(rows, 1)
        ]
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for index, item in enumerate(rows, 1):
                panel_index = int(item.get("panel_index", index))
                panel_name = str(item.get("panel_name") or f"panel_{panel_index:03d}")
                job_id = str(item.get("job_id") or f"{ep_id}:{panel_index:04d}:{panel_name}")
                input_hash = item.get("input_hash")
                existing = conn.execute(
                    "SELECT * FROM render_jobs WHERE job_id=?", (job_id,)
                ).fetchone()
                same_input = bool(existing and input_hash and existing["input_hash"] == input_hash)
                keep_success = bool(
                    same_input and existing["status"] == "succeeded"
                    and existing["output_path"] and Path(existing["output_path"]).exists()
                )
                preserve_state = bool(same_input and (existing["status"] != "succeeded" or keep_success))
                incoming_metadata = dict(item.get("metadata", {}))
                if preserve_state and existing and existing["metadata"]:
                    try:
                        existing_metadata = json.loads(existing["metadata"])
                    except json.JSONDecodeError:
                        existing_metadata = {}
                    incoming_metadata = {**existing_metadata, **incoming_metadata}
                elif existing and not same_input:
                    # QA and release decisions are artifact/input-bound. Keep
                    # their audit history but never carry an acceptance onto a
                    # changed panel contract.
                    try:
                        stale_metadata = json.loads(existing["metadata"] or "{}")
                    except json.JSONDecodeError:
                        stale_metadata = {}
                    for audit_key in ("editorial_review_history", "release_history"):
                        if stale_metadata.get(audit_key):
                            incoming_metadata[audit_key] = stale_metadata[audit_key]
                    stale_review = stale_metadata.get("editorial_review")
                    stale_release = stale_metadata.get("release")
                    if stale_review:
                        incoming_metadata.setdefault("editorial_review_history", []).append(stale_review)
                    if stale_release:
                        incoming_metadata.setdefault("release_history", []).append(stale_release)
                    # A storyboard correction changes the input hash but must
                    # not erase the human-QA evidence that required it. Keep
                    # audit/feedback as provenance while intentionally dropping
                    # any old candidate or approved anchor bound to stale bytes.
                    for audit_key in (
                        "qa_rejection_audit", "qa_rejection_classification",
                        "qa_retry_feedback", "group_anchor_rejection_audit",
                    ):
                        if stale_metadata.get(audit_key):
                            incoming_metadata[audit_key] = stale_metadata[audit_key]
                    incoming_metadata["editorial_review"] = {
                        "status": "pending", "reason": "job input changed",
                    }
                    incoming_metadata["release"] = {
                        "status": "pending", "reason": "job input changed",
                    }
                    incoming_metadata.pop("content_qa", None)
                    incoming_metadata.pop("edit_selection", None)
                status = str(existing["status"]) if preserve_state else str(item.get("status", "queued"))
                if status == "pending":
                    status = "queued"
                if status not in JOB_STATUSES:
                    raise ValueError(f"invalid render status: {status}")
                values = {
                    "job_id": job_id,
                    "ep_id": ep_id,
                    "panel_index": panel_index,
                    "panel_name": panel_name,
                    "status": status,
                    "progress": float(existing["progress"]) if preserve_state else float(item.get("progress", 0.0)),
                    "prompt_id": existing["prompt_id"] if preserve_state else item.get("prompt_id"),
                    "reference_images": _json(item.get("reference_images", [])),
                    "output_path": str(item.get("output_path")) if item.get("output_path") else None,
                    "preview_path": str(item.get("preview_path")) if item.get("preview_path") else None,
                    "comfy_output_path": existing["comfy_output_path"] if preserve_state else (str(item.get("comfy_output_path")) if item.get("comfy_output_path") else None),
                    "graph_path": (
                        str(item.get("graph_path")) if item.get("graph_path")
                        else (existing["graph_path"] if preserve_state else None)
                    ),
                    "timing_path": (
                        str(item.get("timing_path")) if item.get("timing_path")
                        else (existing["timing_path"] if preserve_state else None)
                    ),
                    "error": existing["error"] if preserve_state else item.get("error"),
                    "retry_count": int(existing["retry_count"]) if preserve_state else int(item.get("retry_count", 0)),
                    "max_retries": int(item.get("max_retries", 2)),
                    "input_hash": input_hash,
                    "metadata": _json(incoming_metadata),
                    "probe": existing["probe"] if preserve_state else _json(item.get("probe", {})),
                    "dialogue_cues": _json(item.get("dialogue_cues", [])),
                    "audio_cues": _json(item.get("audio_cues", [])),
                    "created_at": now,
                    "updated_at": now,
                    "submitted_at": existing["submitted_at"] if preserve_state else item.get("submitted_at"),
                    "completed_at": existing["completed_at"] if preserve_state else item.get("completed_at"),
                }
                conn.execute(
                    """
                    INSERT INTO render_jobs (
                        job_id, ep_id, panel_index, panel_name, status, progress,
                        prompt_id, reference_images, output_path, preview_path,
                        comfy_output_path, graph_path, timing_path, error,
                        retry_count, max_retries, input_hash, metadata, probe,
                        dialogue_cues, audio_cues, created_at, updated_at,
                        submitted_at, completed_at
                    ) VALUES (
                        :job_id, :ep_id, :panel_index, :panel_name, :status, :progress,
                        :prompt_id, :reference_images, :output_path, :preview_path,
                        :comfy_output_path, :graph_path, :timing_path, :error,
                        :retry_count, :max_retries, :input_hash, :metadata, :probe,
                        :dialogue_cues, :audio_cues, :created_at, :updated_at,
                        :submitted_at, :completed_at
                    )
                    ON CONFLICT(job_id) DO UPDATE SET
                        panel_index=excluded.panel_index,
                        panel_name=excluded.panel_name,
                        reference_images=excluded.reference_images,
                        output_path=COALESCE(excluded.output_path, render_jobs.output_path),
                        preview_path=COALESCE(excluded.preview_path, render_jobs.preview_path),
                        graph_path=excluded.graph_path,
                        timing_path=excluded.timing_path,
                        input_hash=COALESCE(excluded.input_hash, render_jobs.input_hash),
                        metadata=excluded.metadata,
                        prompt_id=excluded.prompt_id,
                        comfy_output_path=excluded.comfy_output_path,
                        error=excluded.error,
                        retry_count=excluded.retry_count,
                        probe=excluded.probe,
                        submitted_at=excluded.submitted_at,
                        completed_at=excluded.completed_at,
                        dialogue_cues=excluded.dialogue_cues,
                        audio_cues=excluded.audio_cues,
                        status=excluded.status,
                        progress=excluded.progress,
                        updated_at=excluded.updated_at
                    """,
                    values,
                )
            if prune_missing:
                if active_ids:
                    placeholders = ",".join("?" for _ in active_ids)
                    conn.execute(
                        f"DELETE FROM render_jobs WHERE ep_id=? AND job_id NOT IN ({placeholders})",
                        (ep_id, *active_ids),
                    )
                else:
                    conn.execute("DELETE FROM render_jobs WHERE ep_id=?", (ep_id,))
            conn.commit()
        return self.list_jobs(ep_id)

    def get_job(self, job_id: str, ep_id: str | None = None) -> Optional[dict[str, Any]]:
        sql = "SELECT * FROM render_jobs WHERE job_id=?"
        args: tuple[Any, ...] = (job_id,)
        if ep_id is not None:
            sql += " AND ep_id=?"
            args = (job_id, ep_id)
        with self._connection() as conn:
            return self._decode(conn.execute(sql, args).fetchone())

    def list_jobs(self, ep_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM render_jobs WHERE ep_id=? ORDER BY panel_index, created_at", (ep_id,)
            ).fetchall()
        return [self._decode(row) for row in rows if row is not None]

    @staticmethod
    def _decode_asset(row: sqlite3.Row | None) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        result = dict(row)
        for field in _ASSET_JSON_FIELDS:
            raw = result.get(field)
            try:
                result[field] = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except json.JSONDecodeError:
                result[field] = [] if field == "reference_images" else {}
        result["approved"] = bool(result.get("approved"))
        return result

    def get_pipeline(self, ep_id: str) -> Optional[dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM episode_pipeline WHERE ep_id=?", (ep_id,)).fetchone()
        return dict(row) if row else None

    def update_pipeline(self, ep_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {
            "contract_hash", "contract_status", "contract_approved_at",
            "assets_hash", "assets_status", "assets_approved_at",
        }
        if set(changes) - allowed:
            raise ValueError(f"unsupported pipeline fields: {sorted(set(changes) - allowed)}")
        changes["updated_at"] = _utc_now()
        with self._connection() as conn:
            existing = conn.execute("SELECT ep_id FROM episode_pipeline WHERE ep_id=?", (ep_id,)).fetchone()
            if not existing:
                if not changes.get("contract_hash"):
                    raise KeyError(f"pipeline is not prepared: {ep_id}")
                conn.execute(
                    """INSERT INTO episode_pipeline(
                           ep_id,contract_hash,contract_status,contract_approved_at,
                           assets_hash,assets_status,assets_approved_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        ep_id, changes["contract_hash"], changes.get("contract_status", "draft"),
                        changes.get("contract_approved_at"), changes.get("assets_hash"),
                        changes.get("assets_status", "pending"), changes.get("assets_approved_at"),
                        changes["updated_at"],
                    ),
                )
            else:
                assignments = ", ".join(f"{name}=?" for name in changes)
                conn.execute(
                    f"UPDATE episode_pipeline SET {assignments} WHERE ep_id=?",
                    (*changes.values(), ep_id),
                )
        result = self.get_pipeline(ep_id)
        assert result is not None
        return result

    def register_assets(self, ep_id: str, assets: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        now = _utc_now()
        rows = list(assets)
        active_ids = [str(item.get("asset_id") or f"{ep_id}:{item['asset_type']}:{item['source_id']}") for item in rows]
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for item in rows:
                asset_type = str(item["asset_type"])
                source_id = str(item["source_id"])
                asset_id = str(item.get("asset_id") or f"{ep_id}:{asset_type}:{source_id}")
                prompt_hash = str(item["prompt_hash"])
                content_hash = item.get("content_hash")
                existing = conn.execute("SELECT * FROM production_assets WHERE asset_id=?", (asset_id,)).fetchone()
                existing_metadata: dict[str, Any] = {}
                if existing and existing["metadata"]:
                    try:
                        existing_metadata = json.loads(existing["metadata"])
                    except json.JSONDecodeError:
                        existing_metadata = {}
                stored_semantic_hash = _stored_asset_prompt_hash(existing_metadata)
                same_prompt = bool(
                    existing and (
                        existing["prompt_hash"] == prompt_hash
                        or stored_semantic_hash == prompt_hash
                    )
                )
                same_content = bool(existing and existing["content_hash"] == content_hash)
                provided_ready = bool(content_hash and item.get("reference_images"))
                preserve_pending_runtime = bool(
                    existing
                    and same_prompt
                    and existing["status"] in {"queued", "running", "failed", "cancelled"}
                    and not existing["content_hash"]
                )
                if preserve_pending_runtime:
                    # The episode JSON may still carry the last generated
                    # paths.  A failed/queued database record is authoritative:
                    # keep its diagnostics but never revive those stale files.
                    provided_ready = False
                preserve_runtime = bool(same_prompt and (same_content or preserve_pending_runtime))
                if preserve_runtime:
                    status = existing["status"]
                    approved = existing["approved"]
                elif existing and not same_prompt:
                    # Creative source changed: old files may still exist in the
                    # episode JSON but cannot satisfy the new asset contract.
                    status = "queued"
                    approved = 0
                    provided_ready = False
                elif (
                    existing
                    and same_prompt
                    and existing["status"] in {"queued", "running", "failed", "cancelled"}
                    and not existing["content_hash"]
                ):
                    # Another asset stage may persist the episode while this
                    # asset is still queued.  Its JSON can still contain the
                    # previous reference paths; those stale paths must not
                    # resurrect an invalidated asset as succeeded.
                    status = existing["status"]
                    approved = 0
                    provided_ready = False
                else:
                    status = "succeeded" if provided_ready else "queued"
                    approved = 0
                incoming_metadata = dict(item.get("metadata", {}))
                if existing_metadata:
                    for audit_key in ("rejection_audit", "selection_audit"):
                        if existing_metadata.get(audit_key):
                            incoming_metadata[audit_key] = existing_metadata[audit_key]
                values = {
                    "asset_id": asset_id, "ep_id": ep_id, "asset_type": asset_type,
                    "source_id": source_id, "status": status, "approved": approved,
                    "prompt_hash": prompt_hash, "content_hash": content_hash if provided_ready else None,
                    "reference_images": _json(item.get("reference_images", []) if provided_ready else []),
                    "manifest_path": existing["manifest_path"] if preserve_runtime else item.get("manifest_path"),
                    "prompt_id": existing["prompt_id"] if preserve_runtime else None,
                    "error": existing["error"] if preserve_runtime else None,
                    "retry_count": existing["retry_count"] if preserve_runtime else 0,
                    "max_retries": int(item.get("max_retries", 2)),
                    "metadata": _json(incoming_metadata),
                    "created_at": now, "updated_at": now,
                    "completed_at": existing["completed_at"] if preserve_runtime else (now if provided_ready else None),
                    "approved_at": existing["approved_at"] if preserve_runtime else None,
                }
                conn.execute(
                    """INSERT INTO production_assets(
                           asset_id,ep_id,asset_type,source_id,status,approved,prompt_hash,
                           content_hash,reference_images,manifest_path,prompt_id,error,
                           retry_count,max_retries,metadata,created_at,updated_at,completed_at,approved_at
                       ) VALUES(
                           :asset_id,:ep_id,:asset_type,:source_id,:status,:approved,:prompt_hash,
                           :content_hash,:reference_images,:manifest_path,:prompt_id,:error,
                           :retry_count,:max_retries,:metadata,:created_at,:updated_at,:completed_at,:approved_at
                       ) ON CONFLICT(asset_id) DO UPDATE SET
                           status=excluded.status,approved=excluded.approved,prompt_hash=excluded.prompt_hash,
                           content_hash=excluded.content_hash,reference_images=excluded.reference_images,
                           manifest_path=excluded.manifest_path,prompt_id=excluded.prompt_id,error=excluded.error,
                           retry_count=excluded.retry_count,max_retries=excluded.max_retries,
                           metadata=excluded.metadata,updated_at=excluded.updated_at,
                           completed_at=excluded.completed_at,approved_at=excluded.approved_at""",
                    values,
                )
            if active_ids:
                placeholders = ",".join("?" for _ in active_ids)
                conn.execute(
                    f"DELETE FROM production_assets WHERE ep_id=? AND asset_id NOT IN ({placeholders})",
                    (ep_id, *active_ids),
                )
            else:
                conn.execute("DELETE FROM production_assets WHERE ep_id=?", (ep_id,))
        return self.list_assets(ep_id)

    def list_assets(self, ep_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM production_assets WHERE ep_id=? ORDER BY asset_type, source_id", (ep_id,)
            ).fetchall()
        return [self._decode_asset(row) for row in rows if row is not None]

    def get_asset(self, asset_id: str, ep_id: str | None = None) -> Optional[dict[str, Any]]:
        sql = "SELECT * FROM production_assets WHERE asset_id=?"
        args: tuple[Any, ...] = (asset_id,)
        if ep_id:
            sql += " AND ep_id=?"
            args = (asset_id, ep_id)
        with self._connection() as conn:
            row = conn.execute(sql, args).fetchone()
        return self._decode_asset(row)

    def update_asset(self, asset_id: str, **changes: Any) -> dict[str, Any]:
        allowed = {
            "status", "approved", "content_hash", "reference_images", "manifest_path",
            "prompt_id", "error", "retry_count", "max_retries", "metadata",
            "completed_at", "approved_at",
        }
        if set(changes) - allowed:
            raise ValueError(f"unsupported asset fields: {sorted(set(changes) - allowed)}")
        for field in _ASSET_JSON_FIELDS:
            if field in changes:
                changes[field] = _json(changes[field])
        if "approved" in changes:
            changes["approved"] = int(bool(changes["approved"]))
        changes["updated_at"] = _utc_now()
        assignments = ", ".join(f"{name}=?" for name in changes)
        with self._connection() as conn:
            cur = conn.execute(
                f"UPDATE production_assets SET {assignments} WHERE asset_id=?",
                (*changes.values(), asset_id),
            )
            if cur.rowcount != 1:
                raise KeyError(asset_id)
        result = self.get_asset(asset_id)
        assert result is not None
        return result

    @staticmethod
    def _normalize_job_changes(changes: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(changes)
        allowed = {
            "status", "progress", "prompt_id", "reference_images", "output_path",
            "preview_path", "comfy_output_path", "graph_path", "timing_path",
            "error", "retry_count", "max_retries", "input_hash", "metadata",
            "probe", "dialogue_cues", "audio_cues", "submitted_at", "completed_at",
        }
        unknown = set(normalized) - allowed
        if unknown:
            raise ValueError(f"unsupported job fields: {sorted(unknown)}")
        if "status" in normalized:
            status = "queued" if normalized["status"] == "pending" else normalized["status"]
            if status not in JOB_STATUSES:
                raise ValueError(f"invalid render status: {status}")
            normalized["status"] = status
            if status == "succeeded":
                normalized.setdefault("progress", 1.0)
                normalized.setdefault("completed_at", _utc_now())
        if "progress" in normalized:
            normalized["progress"] = max(0.0, min(1.0, float(normalized["progress"])))
        for field in _JSON_FIELDS:
            if field in normalized:
                normalized[field] = _json(normalized[field])
        normalized["updated_at"] = _utc_now()
        return normalized

    def update_jobs_atomic(
        self, updates: Iterable[tuple[str, Mapping[str, Any]]]
    ) -> list[dict[str, Any]]:
        """Apply related job transitions in one SQLite transaction."""
        prepared = [
            (str(job_id), self._normalize_job_changes(changes))
            for job_id, changes in updates
        ]
        if not prepared:
            return []
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for job_id, changes in prepared:
                assignments = ", ".join(f"{name}=?" for name in changes)
                cur = conn.execute(
                    f"UPDATE render_jobs SET {assignments} WHERE job_id=?",
                    (*changes.values(), job_id),
                )
                if cur.rowcount != 1:
                    raise KeyError(job_id)
        results: list[dict[str, Any]] = []
        for job_id, _changes in prepared:
            job = self.get_job(job_id)
            assert job is not None
            results.append(job)
        return results

    def update_job(self, job_id: str, **changes: Any) -> dict[str, Any]:
        if not changes:
            job = self.get_job(job_id)
            if not job:
                raise KeyError(job_id)
            return job
        return self.update_jobs_atomic([(job_id, changes)])[0]

    def compare_and_update_job(
        self,
        job_id: str,
        *,
        expected: Mapping[str, Any],
        **changes: Any,
    ) -> Optional[dict[str, Any]]:
        """Apply one transition only while its persisted ownership fields match.

        Reconciliation uses this small CAS boundary so concurrent controllers
        cannot both claim the same failed remote prompt or both consume one
        retry authorization.  JSON metadata is deliberately not accepted as
        an expectation; prompt/status/attempt identity is the durable owner.
        """
        allowed_expected = {"status", "prompt_id", "retry_count", "input_hash"}
        unknown = set(expected) - allowed_expected
        if unknown:
            raise ValueError(f"unsupported job expectations: {sorted(unknown)}")
        if not expected:
            raise ValueError("at least one expected job field is required")
        normalized = self._normalize_job_changes(changes)
        conditions = ["job_id=?"]
        condition_values: list[Any] = [job_id]
        for field, value in expected.items():
            if value is None:
                conditions.append(f"{field} IS NULL")
            else:
                conditions.append(f"{field}=?")
                condition_values.append(value)
        assignments = ", ".join(f"{name}=?" for name in normalized)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                f"UPDATE render_jobs SET {assignments} WHERE {' AND '.join(conditions)}",
                (*normalized.values(), *condition_values),
            )
            changed = cursor.rowcount == 1
        return self.get_job(job_id) if changed else None

    def next_runnable(self, ep_id: str) -> Optional[dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM render_jobs
                 WHERE ep_id=? AND status='queued' AND retry_count <= max_retries
                 ORDER BY panel_index LIMIT 1
                """,
                (ep_id,),
            ).fetchone()
        return self._decode(row)

    def reserve_worker_launch(self, ep_id: str, launch_token: str, stale_after: float = 120.0) -> bool:
        """Atomically reserve one background-process launch for an episode."""
        if not launch_token:
            raise ValueError("launch_token is required")
        now = time.time()
        owner = f"launch:{launch_token}"
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM workers WHERE ep_id=?", (ep_id,)).fetchone()
            if row and now - float(row["heartbeat"]) <= stale_after:
                return False
            conn.execute(
                "INSERT OR REPLACE INTO workers(ep_id, owner, pid, heartbeat) VALUES(?,?,?,?)",
                (ep_id, owner, 0, now),
            )
        return True

    def set_worker_launch_pid(self, ep_id: str, launch_token: str, pid: int) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE workers SET pid=?, heartbeat=? WHERE ep_id=? AND owner=?",
                (int(pid), time.time(), ep_id, f"launch:{launch_token}"),
            )

    def release_worker_launch(self, ep_id: str, launch_token: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM workers WHERE ep_id=? AND owner=?",
                (ep_id, f"launch:{launch_token}"),
            )

    def acquire_worker(
        self, ep_id: str, stale_after: float = 120.0, *, launch_token: str | None = None,
    ) -> bool:
        now = time.time()
        owner = f"{socket.gethostname()}:{os.getpid()}"
        reservation = f"launch:{launch_token}" if launch_token else None
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM workers WHERE ep_id=?", (ep_id,)).fetchone()
            if (
                row and now - float(row["heartbeat"]) <= stale_after
                and row["owner"] != owner and row["owner"] != reservation
            ):
                return False
            conn.execute(
                "INSERT OR REPLACE INTO workers(ep_id, owner, pid, heartbeat) VALUES(?,?,?,?)",
                (ep_id, owner, os.getpid(), now),
            )
            conn.commit()
        return True

    def worker_info(self, ep_id: str, stale_after: float = 120.0) -> Optional[dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM workers WHERE ep_id=?", (ep_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["active"] = time.time() - float(result["heartbeat"]) <= stale_after
        return result

    def active_workers(self, stale_after: float = 120.0) -> list[dict[str, Any]]:
        """Return every live worker/launch reservation without mutating leases."""
        now = time.time()
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM workers ORDER BY ep_id").fetchall()
        active: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["active"] = now - float(item["heartbeat"]) <= stale_after
            if item["active"]:
                active.append(item)
        return active

    def heartbeat_worker(self, ep_id: str) -> None:
        with self._connection() as conn:
            conn.execute("UPDATE workers SET heartbeat=? WHERE ep_id=? AND pid=?", (time.time(), ep_id, os.getpid()))

    def release_worker(self, ep_id: str) -> None:
        with self._connection() as conn:
            conn.execute("DELETE FROM workers WHERE ep_id=? AND pid=?", (ep_id, os.getpid()))


_default_store: RenderJobStore | None = None


def default_store() -> RenderJobStore:
    global _default_store
    expected = render_job_db().resolve()
    if _default_store is None or _default_store.path != expected:
        _default_store = RenderJobStore(expected)
    return _default_store


def list_jobs(ep_id: str) -> list[dict[str, Any]]:
    _safe_id(ep_id, "ep_id")
    return default_store().list_jobs(ep_id)


def _prepare_episode_jobs(ep_id: str, episode: Mapping[str, Any]) -> dict[str, Any]:
    """Persist an episode and register every panel without starting ComfyUI/GPU work.

    The episode file is atomically replaced only after the complete job set has
    been committed to SQLite.  Repeating this call preserves a succeeded job
    when its output file still exists.
    """
    ep_id = _safe_id(ep_id, "ep_id")
    if not isinstance(episode, Mapping):
        raise TypeError("episode must be a mapping")
    panels = episode.get("panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("episode.panels must be a non-empty list")
    payload = dict(episode)
    payload["ep_id"] = ep_id
    project = projects_dir() / ep_id
    videos_dir = project / "videos"
    previews_dir = project / "previews"
    videos_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    store = default_store()
    registered_assets = store.list_assets(ep_id)
    asset_by_key = {
        (asset["asset_type"], asset["source_id"]): asset for asset in registered_assets
    }

    episode_settings = {
        "aspect_ratio": payload.get("aspect_ratio", "16:9"),
        "duration_seconds": payload.get("duration_seconds", 10.0),
        "use_lora": payload.get("use_lora", True),
        "lora_strength": payload.get("lora_strength", 1.0),
        "reference_fidelity": payload.get("reference_fidelity", "fast"),
        "sage_attention": payload.get("sage_attention", payload.get("sage_mode", "auto")),
    }
    if isinstance(payload.get("render_settings"), Mapping):
        episode_settings.update(payload["render_settings"])
    canonical = _contract_payload(payload)
    story_context = canonical.get("story_bible") or {}
    scene_contexts = {
        str(scene.get("scene_id")): scene
        for scene in canonical.get("scene_bible") or [] if isinstance(scene, Mapping)
    }
    jobs: list[dict[str, Any]] = []
    prepared_by_panel: dict[str, dict[str, Any]] = {}
    last_by_group: dict[str, str] = {}
    existing_jobs = {job["job_id"]: job for job in store.list_jobs(ep_id)}
    for index, panel in enumerate(panels, 1):
        if not isinstance(panel, Mapping):
            raise ValueError(f"episode.panels[{index - 1}] must be an object")
        panel_name = _safe_panel_name(panel, index)
        job_id = f"{ep_id}:{index:04d}:{panel_name}"
        raw_panel_id = str(panel.get("panel_id") or panel.get("name") or panel_name)
        continuity_group = str(panel.get("continuity_group") or "")
        previous_panel_id = str(panel.get("previous_panel_id") or "")
        if not previous_panel_id and continuity_group:
            previous_panel_id = last_by_group.get(continuity_group, "")
        strict_continuity = bool(
            previous_panel_id and (
                panel.get("strict_continuity", True)
                or episode_settings.get("continuity_mode") == "strict"
            )
        )
        continuity_dependency: dict[str, Any] = {}
        if previous_panel_id:
            previous_job = prepared_by_panel.get(previous_panel_id)
            if not previous_job:
                raise ValueError(f"panel {raw_panel_id} references unknown/forward previous_panel_id {previous_panel_id}")
            previous_runtime = existing_jobs.get(previous_job["job_id"], {})
            continuity_dependency = {
                "continuity_group": continuity_group or None,
                "previous_panel_id": previous_panel_id,
                "previous_job_id": previous_job["job_id"],
                "previous_input_hash": previous_job["input_hash"],
                "previous_artifact_hash": (previous_runtime.get("metadata") or {}).get("artifact_sha256"),
                "strict": strict_continuity,
                "first_frame_source": "previous_tail" if strict_continuity else "optional",
            }
        char_ids = panel.get("character_ids") or []
        package = panel.get("prompt_package") or {}
        scene_id = str(
            panel.get("scene_id")
            or (package.get("scene_id") if isinstance(package, Mapping) else "")
            or ""
        )
        references: list[str] = [str(path) for path in panel.get("reference_images", []) if path]
        reference_inputs: list[dict[str, Any]] = [
            {"role": "panel_reference", "source_id": panel_name, "path": str(path)}
            for path in references
        ]
        asset_dependencies: list[dict[str, Any]] = []
        scene_asset = asset_by_key.get(("scene", scene_id)) if scene_id else None
        if scene_asset:
            asset_dependencies.append({
                key: scene_asset.get(key) for key in (
                    "asset_id", "asset_type", "source_id", "prompt_hash", "content_hash", "approved"
                )
            })
            if scene_asset["status"] == "succeeded":
                for path in scene_asset.get("reference_images") or []:
                    references.append(str(path))
                    reference_inputs.append({"role": "scene_reference", "source_id": scene_id, "path": str(path)})
        for char_id in char_ids:
            char_asset = asset_by_key.get(("character", str(char_id)))
            if not char_asset:
                continue
            asset_dependencies.append({
                key: char_asset.get(key) for key in (
                    "asset_id", "asset_type", "source_id", "prompt_hash", "content_hash", "approved"
                )
            })
            if char_asset["status"] == "succeeded":
                for path in char_asset.get("reference_images") or []:
                    references.append(str(path))
                    reference_inputs.append({"role": "character_reference", "source_id": str(char_id), "path": str(path)})
        references = list(dict.fromkeys(references))
        reference_inputs = list({(item["role"], item["source_id"], item["path"]): item for item in reference_inputs}.values())
        package = package or {
            "positive_prompt": panel.get("positive_prompt") or panel.get("prompt") or "",
            "negative_prompt": panel.get("negative_prompt") or "",
            "character_ids": list(char_ids),
            "scene_id": panel.get("scene_id"),
        }
        panel_settings = dict(episode_settings)
        if isinstance(package, Mapping) and isinstance(package.get("render_settings"), Mapping):
            panel_settings.update(package["render_settings"])
        for key in episode_settings:
            if panel.get(key) is not None:
                panel_settings[key] = panel[key]
        # Normalize the V2 UI/LLM vocabulary to the renderer vocabulary while
        # retaining the original fields for audit/debug display.
        if panel_settings.get("ref_image_size") in {"match", "max"}:
            panel_settings["reference_fidelity"] = (
                "identity" if panel_settings["ref_image_size"] == "max" else "fast"
            )
        if panel_settings.get("sage_mode"):
            panel_settings["sage_attention"] = panel_settings["sage_mode"]
        existing_job = existing_jobs.get(job_id) or {}
        existing_metadata = existing_job.get("metadata") or {}
        panel_settings.setdefault("production_strategy", DEFAULT_PRODUCTION_STRATEGY)
        panel_settings = apply_render_profile(panel_settings, metadata=existing_metadata)
        source_generation_duration = float(
            panel_settings.get("source_generation_duration_seconds")
            or panel_settings.get("duration_seconds")
            or SOURCE_GENERATION_DURATION_SECONDS
        )
        edit_duration = panel.get("edit_duration_seconds")
        panel_settings["source_generation_duration_seconds"] = source_generation_duration
        panel_settings["duration_seconds"] = source_generation_duration
        if edit_duration is not None:
            panel_settings["edit_duration_seconds"] = float(edit_duration)
        shot_plan = {
            key: panel.get(key) for key in (
                "source_generation_duration_seconds", "edit_duration_seconds", "shot_role",
                "story_beat_id", "visible_action", "first_state", "final_state", "cause",
                "next_hook", "camera_plan", "transition", "edit_hint", "priority",
                "group_shot_reason",
            )
        }
        inputs = {
            "reference_images": references,
            "prompt_package": package,
            "settings": panel_settings,
            "character_ids": list(char_ids),
            "scene_id": scene_id,
            "story_context": story_context,
            "scene_context": scene_contexts.get(scene_id, {}),
            "reference_inputs": reference_inputs,
            "asset_dependencies": asset_dependencies,
            "continuity_dependency": continuity_dependency,
            "shot_plan": shot_plan,
        }
        panel_reference_hashes = []
        for item in reference_inputs:
            if item["role"] != "panel_reference":
                continue
            resolved = _resolve_reference(item["path"], project)
            panel_reference_hashes.append({
                "source_id": item["source_id"],
                "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved else None,
            })
        hash_inputs = {
            "render_input_schema": RENDER_INPUT_SCHEMA_VERSION,
            "prompt_package": _prompt_package_hash_source(package),
            "settings": panel_settings,
            "character_ids": list(char_ids),
            "scene_id": scene_id,
            "story_context": story_context,
            "scene_model_context": _asset_prompt_source(
                "scene", scene_id, scene_contexts.get(scene_id, {}),
                canonical.get("visual_bible") or {},
            ) if scene_id else {},
            "panel_reference_hashes": panel_reference_hashes,
            "asset_dependencies": [
                {key: dependency.get(key) for key in ("asset_id", "prompt_hash", "content_hash")}
                for dependency in asset_dependencies
            ],
            "continuity_dependency": continuity_dependency,
            "shot_plan": shot_plan,
        }
        material = json.dumps(hash_inputs, ensure_ascii=False, sort_keys=True).encode("utf-8")
        output_dir = videos_dir if panel_settings.get("delivery_eligible") else previews_dir
        preserved_runtime = {
            key: existing_metadata[key]
            for key in ("preview_promotion", "preview_history")
            if existing_metadata.get(key)
        }
        prepared_job = {
            "job_id": job_id,
            "panel_index": index,
            "panel_name": panel_name,
            "status": "queued",
            "reference_images": references,
            "output_path": str((output_dir / f"{panel_name}.mp4").resolve()),
            "preview_path": str((output_dir / f"{panel_name}.mp4").resolve()),
            "input_hash": hashlib.sha256(material).hexdigest(),
            "dialogue_cues": panel.get("spoken_dialogue") or panel.get("dialogue") or [],
            "audio_cues": panel.get("audio_cues") or panel.get("sfx") or [],
            "metadata": {
                **preserved_runtime,
                "inputs": inputs,
                "settings": panel_settings,
                "render_profile_contract": H3_RENDER_PROFILE_CONTRACT,
            },
        }
        jobs.append(prepared_job)
        prepared_by_panel[raw_panel_id] = prepared_job
        prepared_by_panel[panel_name] = prepared_job
        if continuity_group:
            last_by_group[continuity_group] = raw_panel_id

    store.register_jobs(ep_id, jobs)
    _write_json_atomic(project / "episode.json", payload)
    return project_snapshot(ep_id)


def prepare_contract(ep_id: str, episode: Mapping[str, Any]) -> dict[str, Any]:
    """Persist a draft contract, all asset dependencies and all panel jobs.

    This operation is CPU/filesystem only.  A changed contract must be approved
    again, while unchanged approved assets retain approval independently.
    """
    ep_id = _safe_id(ep_id, "ep_id")
    if not isinstance(episode, Mapping):
        raise TypeError("episode must be a mapping")
    payload = json.loads(json.dumps(dict(episode), ensure_ascii=False))
    payload["ep_id"] = ep_id
    current_hash = contract_hash(payload)
    store = default_store()
    pipeline = store.get_pipeline(ep_id)
    same_contract = bool(pipeline and pipeline["contract_hash"] == current_hash)
    store.update_pipeline(
        ep_id,
        contract_hash=current_hash,
        contract_status=pipeline["contract_status"] if same_contract else "draft",
        contract_approved_at=pipeline.get("contract_approved_at") if same_contract else None,
        assets_hash=pipeline.get("assets_hash") if pipeline else None,
        assets_status=pipeline.get("assets_status", "pending") if pipeline else "pending",
        assets_approved_at=pipeline.get("assets_approved_at") if pipeline else None,
    )
    project = projects_dir() / ep_id
    assets = store.register_assets(ep_id, _asset_specs(ep_id, payload, project))
    asset_map = {(asset["asset_type"], asset["source_id"]): asset for asset in assets}
    for asset_type, collection, id_key in (
        ("character", payload.get("character_bible") or [], "character_id"),
        ("scene", payload.get("scene_bible") or [], "scene_id"),
    ):
        for item in collection:
            if not isinstance(item, dict):
                continue
            asset = asset_map.get((asset_type, str(item.get(id_key) or item.get("id") or "")))
            if not asset:
                continue
            item["asset_status"] = asset["status"]
            item["asset_hash"] = asset.get("content_hash")
            item["asset_manifest_path"] = asset.get("manifest_path")
            item["asset_approval"] = {
                "state": "approved" if asset["approved"] else "pending",
                "content_hash": asset.get("content_hash"),
                "approved_at": asset.get("approved_at"),
            }
    if not assets or all(asset["status"] == "succeeded" for asset in assets):
        assets_status = "approved" if all(asset["approved"] for asset in assets) else "ready_for_approval"
    else:
        assets_status = "pending"
    store.update_pipeline(ep_id, assets_status=assets_status)
    return _prepare_episode_jobs(ep_id, payload)


def prepare_episode(ep_id: str, episode: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility alias for the Phase-2 contract preparation gate."""
    return prepare_contract(ep_id, episode)


def approve_contract(ep_id: str, expected_hash: str | None = None) -> dict[str, Any]:
    ep_id = _safe_id(ep_id, "ep_id")
    store = default_store()
    pipeline = store.get_pipeline(ep_id)
    if not pipeline:
        raise KeyError(f"contract is not prepared: {ep_id}")
    if expected_hash and expected_hash != pipeline["contract_hash"]:
        raise RuntimeError("contract changed since it was reviewed")
    episode_path = projects_dir() / ep_id / "episode.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode contract is missing: {episode_path}")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    errors = _contract_errors(episode)
    if errors:
        raise ValueError("contract validation failed: " + "; ".join(errors))
    if contract_hash(episode) != pipeline["contract_hash"]:
        raise RuntimeError("persisted contract changed since preparation")
    store.update_pipeline(
        ep_id, contract_status="approved", contract_approved_at=_utc_now()
    )
    return project_snapshot(ep_id)


def list_assets(ep_id: str) -> list[dict[str, Any]]:
    _safe_id(ep_id, "ep_id")
    return default_store().list_assets(ep_id)


def approve_assets(ep_id: str, expected_hashes: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    ep_id = _safe_id(ep_id, "ep_id")
    store = default_store()
    pipeline = store.get_pipeline(ep_id)
    if not pipeline or pipeline["contract_status"] != "approved":
        raise RuntimeError("contract must be approved before assets")
    episode_path = projects_dir() / ep_id / "episode.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode contract is missing: {episode_path}")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    errors = _contract_errors(episode)
    if errors:
        raise ValueError("contract validation failed before asset approval: " + "; ".join(errors))
    assets = store.list_assets(ep_id)
    incomplete = [asset["asset_id"] for asset in assets if asset["status"] != "succeeded" or not asset.get("content_hash")]
    if incomplete:
        raise RuntimeError(f"assets are not ready for approval: {incomplete}")
    expected_hashes = dict(expected_hashes or {})
    project = projects_dir() / ep_id
    for asset in assets:
        actual_hash = _reference_bundle_hash(asset.get("reference_images") or [], project)
        if actual_hash != asset["content_hash"]:
            raise RuntimeError(f"asset files changed since preparation: {asset['asset_id']}")
        if asset["asset_id"] in expected_hashes and expected_hashes[asset["asset_id"]] != asset["content_hash"]:
            raise RuntimeError(f"asset changed since review: {asset['asset_id']}")
    approved_at = _utc_now()
    for asset in assets:
        store.update_asset(asset["asset_id"], approved=True, approved_at=approved_at)
    bundle_hash = _sha256_json([
        {"asset_id": asset["asset_id"], "prompt_hash": asset["prompt_hash"], "content_hash": asset["content_hash"]}
        for asset in assets
    ])
    store.update_pipeline(
        ep_id, assets_status="approved", assets_hash=bundle_hash, assets_approved_at=approved_at
    )
    if episode_path.is_file():
        approved_map = {(asset["asset_type"], asset["source_id"]): asset for asset in store.list_assets(ep_id)}
        for asset_type, collection, id_key in (
            ("character", episode.get("character_bible") or [], "character_id"),
            ("scene", episode.get("scene_bible") or [], "scene_id"),
        ):
            for item in collection:
                if not isinstance(item, dict):
                    continue
                asset = approved_map.get((asset_type, str(item.get(id_key) or item.get("id") or "")))
                if asset:
                    item["asset_status"] = asset["status"]
                    item["asset_hash"] = asset.get("content_hash")
                    item["asset_manifest_path"] = asset.get("manifest_path")
                    item["asset_approval"] = {
                        "state": "approved", "content_hash": asset.get("content_hash"),
                        "approved_at": approved_at,
                    }
        _write_json_atomic(episode_path, episode)
    return project_snapshot(ep_id)


def _select_asset(
    ep_id: str, asset_id: str | None, asset_type: str | None, source_id: str | None
) -> dict[str, Any]:
    store = default_store()
    if asset_id:
        asset = store.get_asset(str(asset_id), ep_id=ep_id)
    elif asset_type and source_id:
        asset = next((
            item for item in store.list_assets(ep_id)
            if item["asset_type"] == str(asset_type) and item["source_id"] == str(source_id)
        ), None)
    else:
        raise ValueError("asset_id or both asset_type and source_id are required")
    if not asset:
        raise KeyError(f"unknown asset for {ep_id}: {asset_id or f'{asset_type}/{source_id}'}")
    return asset


def _queue_asset_regeneration(
    ep_id: str,
    asset_id: str | None,
    *,
    asset_type: str | None,
    source_id: str | None,
    reason: str,
    action: str,
) -> dict[str, Any]:
    ep_id = _safe_id(ep_id, "ep_id")
    store = default_store()
    asset = _select_asset(ep_id, asset_id, asset_type, source_id)
    now = _utc_now()
    audit_entry = {
        "action": action, "reason": str(reason or action), "at": now,
        "content_hash": asset.get("content_hash"),
        "reference_images": list(asset.get("reference_images") or []),
        "manifest_path": asset.get("manifest_path"), "prompt_id": asset.get("prompt_id"),
    }
    metadata = dict(asset.get("metadata") or {})
    metadata["rejection_audit"] = [*(metadata.get("rejection_audit") or []), audit_entry]

    episode_path = projects_dir() / ep_id / "episode.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode contract is missing: {episode_path}")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    collection_name = "character_bible" if asset["asset_type"] == "character" else "scene_bible"
    id_key = "character_id" if asset["asset_type"] == "character" else "scene_id"
    source = next((
        item for item in episode.get(collection_name) or []
        if isinstance(item, dict) and str(item.get(id_key) or item.get("id") or "") == asset["source_id"]
    ), None)
    if source is None:
        raise RuntimeError(f"asset source is missing from episode contract: {asset['source_id']}")
    source["asset_rejection_history"] = [*(source.get("asset_rejection_history") or []), audit_entry]
    source["reference_images"] = []
    source["asset_status"] = "queued"
    source["asset_hash"] = None
    source["asset_manifest_path"] = None
    source["asset_approval"] = {"state": "rejected", "reason": str(reason or action), "at": now}
    _write_json_atomic(episode_path, episode)

    store.update_asset(
        asset["asset_id"], status="queued", approved=False, content_hash=None,
        reference_images=[], manifest_path=None, prompt_id=None,
        error=f"{action}: {reason or action}", retry_count=0,
        metadata=metadata, completed_at=None, approved_at=None,
    )
    store.update_pipeline(
        ep_id, assets_status="pending", assets_hash=None, assets_approved_at=None,
    )
    snapshot = prepare_contract(ep_id, episode)
    snapshot["asset_action"] = {
        "action": action, "asset_id": asset["asset_id"], "source_id": asset["source_id"],
        "status": "queued", "reason": str(reason or action),
    }
    return snapshot


def reject_asset(
    ep_id: str,
    asset_id: str | None = None,
    *,
    asset_type: str | None = None,
    source_id: str | None = None,
    reason: str = "rejected by reviewer",
) -> dict[str, Any]:
    """Reject one generated asset and queue only its dependent regeneration."""
    return _queue_asset_regeneration(
        ep_id, asset_id, asset_type=asset_type, source_id=source_id,
        reason=reason, action="rejected",
    )


def retry_asset(
    ep_id: str,
    asset_id: str | None = None,
    *,
    asset_type: str | None = None,
    source_id: str | None = None,
    reason: str = "manual retry",
) -> dict[str, Any]:
    """Clear a failed/rejected asset result so the asset worker regenerates it."""
    asset = _select_asset(_safe_id(ep_id, "ep_id"), asset_id, asset_type, source_id)
    if asset["status"] == "succeeded" and asset["approved"]:
        raise RuntimeError("approved asset must be rejected before retry")
    return _queue_asset_regeneration(
        ep_id, asset["asset_id"], asset_type=None, source_id=None,
        reason=reason, action="retry_requested",
    )


def select_asset_references(
    ep_id: str,
    asset_id: str,
    reference_images: Iterable[str],
    *,
    reason: str = "selected existing asset in Web review",
) -> dict[str, Any]:
    """Replace one asset's media without rewriting the creative contract.

    This is the production-safe path for choosing a better historical or
    externally supplied reference.  It preserves unrelated assets and the
    approved creative hash, but revokes approval for the selected content and
    rebuilds dependent render-job inputs.
    """
    ep_id = _safe_id(ep_id, "ep_id")
    asset = _select_asset(ep_id, asset_id, None, None)
    project = projects_dir() / ep_id
    resolved: list[Path] = []
    for value in reference_images:
        path = _resolve_reference(value, project)
        if not path:
            raise FileNotFoundError(f"asset reference does not exist: {value}")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"asset reference must be an image: {path}")
        if path not in resolved:
            resolved.append(path)
    if not resolved:
        raise ValueError("at least one existing reference image is required")

    episode_path = project / "episode.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode contract is missing: {episode_path}")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    before = (default_store().get_pipeline(ep_id) or {}).get("contract_hash")
    collection_name = "character_bible" if asset["asset_type"] == "character" else "scene_bible"
    id_key = "character_id" if asset["asset_type"] == "character" else "scene_id"
    source = next((
        item for item in episode.get(collection_name) or []
        if isinstance(item, dict) and str(item.get(id_key) or item.get("id") or "") == asset["source_id"]
    ), None)
    if source is None:
        raise RuntimeError(f"asset source is missing from episode contract: {asset['source_id']}")
    selected_hash = _reference_bundle_hash(resolved, project)
    if not selected_hash:
        raise RuntimeError("selected asset references could not be hashed")
    metadata = dict(asset.get("metadata") or {})
    metadata["selection_audit"] = [
        *(metadata.get("selection_audit") or []),
        {
            "action": "selected_existing_reference", "reason": str(reason),
            "at": _utc_now(), "reference_images": [str(path) for path in resolved],
            "content_hash": selected_hash,
        },
    ]
    # This user action is an explicit content choice. Register it durably
    # before prepare_contract so the anti-stale branch cannot reinterpret the
    # just-selected references as old files from a queued generation request.
    default_store().update_asset(
        asset_id, status="succeeded", approved=False,
        content_hash=selected_hash, reference_images=[str(path) for path in resolved],
        manifest_path=None, prompt_id=None, error=None, retry_count=0,
        metadata=metadata, completed_at=_utc_now(), approved_at=None,
    )
    source["reference_images"] = [str(path) for path in resolved]
    source["asset_status"] = "succeeded"
    source["asset_hash"] = selected_hash
    source["asset_manifest_path"] = None
    source["asset_approval"] = {"state": "pending", "content_hash": source["asset_hash"]}

    snapshot = prepare_contract(ep_id, episode)
    after = snapshot.get("pipeline", {}).get("contract_hash")
    if before and after != before:
        raise RuntimeError("asset-only selection changed the creative contract hash")
    selected = default_store().get_asset(asset_id, ep_id=ep_id)
    if not selected or selected["status"] != "succeeded":
        raise RuntimeError("selected asset reference was not registered as succeeded")
    return project_snapshot(ep_id)


def production_gate(ep_id: str) -> dict[str, Any]:
    store = default_store()
    pipeline = store.get_pipeline(ep_id) or {}
    assets = store.list_assets(ep_id)
    reasons = []
    if pipeline.get("contract_status") != "approved":
        reasons.append("contract_not_approved")
    if pipeline.get("assets_status") != "approved":
        reasons.append("assets_not_approved")
    if any(asset["status"] != "succeeded" or not asset["approved"] for asset in assets):
        reasons.append("asset_dependency_not_approved")
    contract_errors: list[str] = []
    episode_path = projects_dir() / ep_id / "episode.json"
    if episode_path.is_file():
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        contract_errors = _contract_errors(episode)
    else:
        contract_errors = ["episode contract is missing"]
    if contract_errors:
        reasons.append("contract_invalid")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "pipeline": pipeline,
        "assets": assets,
        "contract_errors": contract_errors,
    }


def _strict_continuity_descendants(
    jobs: Iterable[Mapping[str, Any]], root_job_id: str,
) -> list[dict[str, Any]]:
    """Return only jobs transitively bound to ``root_job_id`` by strict continuity."""
    job_list = [dict(item) for item in jobs]
    children: dict[str, list[dict[str, Any]]] = {}
    for job in job_list:
        dependency = (
            ((job.get("metadata") or {}).get("inputs") or {}).get("continuity_dependency") or {}
        )
        previous_job_id = str(dependency.get("previous_job_id") or "")
        if previous_job_id and bool(dependency.get("strict")):
            children.setdefault(previous_job_id, []).append(job)
    descendants: list[dict[str, Any]] = []
    pending = [root_job_id]
    seen = {root_job_id}
    while pending:
        parent_id = pending.pop(0)
        for child in sorted(children.get(parent_id, []), key=lambda item: int(item["panel_index"])):
            child_id = str(child["job_id"])
            if child_id in seen:
                continue
            seen.add(child_id)
            descendants.append(child)
            pending.append(child_id)
    return descendants


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_job_files(
    ep_id: str, job: Mapping[str, Any], *, stamp: str,
) -> tuple[dict[str, Any], list[tuple[Path, Path]]]:
    """Move local delivery artifacts into a reviewer-visible, recoverable folder."""
    project = (projects_dir() / ep_id).resolve()
    job_token = hashlib.sha256(str(job["job_id"]).encode("utf-8")).hexdigest()[:8]
    destination = project / "rejected" / str(job["panel_name"]) / f"{stamp}_{job_token}"
    candidates: list[tuple[str, Path]] = []
    for field in ("output_path", "preview_path", "graph_path", "timing_path"):
        value = job.get(field)
        if value:
            candidates.append((field, Path(str(value)).resolve()))
    output_value = job.get("output_path")
    if output_value:
        output = Path(str(output_value)).resolve()
        candidates.extend([
            ("artifact", output.with_suffix(".artifact.json")),
            ("cues", output.with_suffix(".cues.json")),
        ])

    archived: dict[str, Any] = {}
    moves: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for role, source in candidates:
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        try:
            source.relative_to(project)
        except ValueError as exc:
            raise ValueError(
                f"refusing to move {role} outside episode project: {source}"
            ) from exc
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        suffix = 1
        while target.exists():
            target = destination / f"{source.stem}_{suffix}{source.suffix}"
            suffix += 1
        digest = _file_sha256(source)
        shutil.move(str(source), str(target))
        moves.append((source, target))
        archived[role] = {"path": str(target), "sha256": digest}
    return archived, moves


def reject_job(
    ep_id: str,
    job_id: str,
    *,
    reason: str,
    rejection_category: str = "other",
    cancel_job: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Reject exactly one panel and invalidate only its strict continuity descendants.

    Active target/descendant prompts must be cancelled through ``cancel_job``
    before durable state changes. Successful local files are moved under the
    episode's ``rejected/`` folder and retained in the QA audit metadata.
    """
    ep_id = _safe_id(ep_id, "ep_id")
    job_id = str(job_id).strip()
    if not job_id:
        raise ValueError("job_id is required")
    reason = str(reason).strip()
    if not reason:
        raise ValueError("QA rejection reason is required")
    rejection_category = str(rejection_category or "").strip()
    allowed_rejection_categories = {
        "action_timing_or_edit_window",
        "identity_or_character",
        "composition_or_scene",
        "continuity_or_state",
        "other",
    }
    if rejection_category not in allowed_rejection_categories:
        raise ValueError(f"unsupported QA rejection category: {rejection_category}")
    store = default_store()
    target = store.get_job(job_id, ep_id=ep_id)
    if not target:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    jobs = store.list_jobs(ep_id)
    descendants = _strict_continuity_descendants(jobs, job_id)
    affected = [target, *descendants]
    active = [job for job in affected if job["status"] in {"submitted", "running"}]
    active_job_ids = {str(job["job_id"]) for job in active}
    if active and cancel_job is None:
        raise RuntimeError(
            "active continuity jobs must be cancelled before rejection: "
            + ", ".join(str(job["job_id"]) for job in active)
        )
    for job in active:
        assert cancel_job is not None
        cancel_job(dict(job))

    now = _utc_now()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    moved: list[tuple[Path, Path]] = []
    updates: list[tuple[str, Mapping[str, Any]]] = []
    archived_by_job: dict[str, dict[str, Any]] = {}
    try:
        for job in affected:
            archived, job_moves = _archive_job_files(ep_id, job, stamp=stamp)
            archived_by_job[str(job["job_id"])] = archived
            moved.extend(job_moves)

        target_metadata = json.loads(json.dumps(target.get("metadata") or {}, ensure_ascii=False))
        target_audit = list(target_metadata.get("qa_rejection_audit") or [])
        target_audit.append({
            "action": "job_rejected",
            "reason": reason,
            "category": rejection_category,
            "at": now,
            "previous_status": target["status"],
            "prompt_id": target.get("prompt_id"),
            "output_path": target.get("output_path"),
            "preview_path": target.get("preview_path"),
            "comfy_output_path": target.get("comfy_output_path"),
            "probe": target.get("probe") or {},
            "artifact_sha256": target_metadata.get("artifact_sha256"),
            "archived_files": archived_by_job.get(job_id, {}),
        })
        target_metadata["qa_rejection_audit"] = target_audit
        feedback_payload = {
            "reason": reason,
            "category": rejection_category,
            "at": now,
            "source": "human_qa",
        }
        feedback_payload["sha256"] = hashlib.sha256(
            json.dumps(
                feedback_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        target_metadata["qa_retry_feedback"] = feedback_payload
        target_metadata["qa_retry_paths"] = {
            "output_path": target.get("output_path") or str(
                projects_dir() / ep_id / "videos" / f"{target['panel_name']}.mp4"
            ),
            "preview_path": target.get("preview_path") or target.get("output_path"),
        }
        max_retries = max(1, int(target.get("max_retries") or 0))
        retry_count = min(
            int(target.get("retry_count") or 0),
            max_retries - 1,
        )
        updates.append((job_id, {
            "status": "failed", "progress": 0.0,
            "prompt_id": None, "output_path": None, "preview_path": None,
            "comfy_output_path": None, "graph_path": None, "timing_path": None,
            "probe": {}, "error": f"QA rejected: {reason}",
            "retry_count": retry_count, "max_retries": max_retries,
            "metadata": target_metadata,
            "submitted_at": None, "completed_at": None,
        }))

        for descendant in descendants:
            descendant_id = str(descendant["job_id"])
            was_active = descendant_id in active_job_ids
            descendant_metadata = json.loads(json.dumps(descendant.get("metadata") or {}, ensure_ascii=False))
            inputs = descendant_metadata.setdefault("inputs", {})
            dependency = dict(inputs.get("continuity_dependency") or {})
            dependency["previous_input_hash"] = None
            dependency["previous_artifact_hash"] = None
            dependency["first_frame_source"] = "previous_tail_pending"
            inputs["continuity_dependency"] = dependency
            audit = list(descendant_metadata.get("qa_invalidation_audit") or [])
            audit.append({
                "action": "invalidated_by_rejected_predecessor",
                "reason": reason,
                "at": now,
                "rejected_job_id": job_id,
                "previous_status": descendant["status"],
                "final_status": "failed" if was_active else "queued",
                "prompt_id": descendant.get("prompt_id"),
                "archived_files": archived_by_job.get(str(descendant["job_id"]), {}),
            })
            descendant_metadata["qa_invalidation_audit"] = audit
            descendant_metadata["qa_retry_paths"] = {
                "output_path": descendant.get("output_path") or str(
                    projects_dir() / ep_id / "videos" / f"{descendant['panel_name']}.mp4"
                ),
                "preview_path": descendant.get("preview_path") or descendant.get("output_path"),
            }
            descendant_max_retries = max(1, int(descendant.get("max_retries") or 0))
            descendant_retry_count = min(
                int(descendant.get("retry_count") or 0),
                descendant_max_retries - 1,
            )
            updates.append((descendant_id, {
                # A prompt that was active has already been cancelled above.
                # Keep it non-runnable until an explicit resume instead of
                # overwriting ``cancelled`` with ``queued`` from this stale
                # pre-cancellation snapshot. Inactive descendants remain
                # queued behind the rejected predecessor.
                "status": "failed" if was_active else "queued",
                "progress": 0.0, "prompt_id": None,
                "output_path": None, "preview_path": None, "comfy_output_path": None,
                "graph_path": None, "timing_path": None, "probe": {},
                "error": (
                    f"cancelled after QA rejection of predecessor {job_id}; retry required"
                    if was_active else None
                ),
                "retry_count": descendant_retry_count if was_active else 0,
                "max_retries": descendant_max_retries,
                "input_hash": None, "metadata": descendant_metadata,
                "submitted_at": None, "completed_at": None,
            }))
        store.update_jobs_atomic(updates)
    except Exception:
        for original, archived in reversed(moved):
            if archived.exists() and not original.exists():
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(archived), str(original))
        raise

    return {
        "ep_id": ep_id,
        "rejected_job_id": job_id,
        "reason": reason,
        "cancelled_job_ids": [str(job["job_id"]) for job in active],
        "invalidated_job_ids": [str(job["job_id"]) for job in descendants],
        "job": store.get_job(job_id, ep_id=ep_id),
        "jobs": store.list_jobs(ep_id),
    }


def classify_job_rejection(
    ep_id: str, job_id: str, *, rejection_category: str,
) -> dict[str, Any]:
    """Attach an explicit reviewer category to the latest legacy rejection audit."""
    ep_id = _safe_id(ep_id, "ep_id")
    job_id = str(job_id).strip()
    allowed = {
        "action_timing_or_edit_window",
        "identity_or_character",
        "composition_or_scene",
        "continuity_or_state",
        "other",
    }
    category = str(rejection_category or "").strip()
    if category not in allowed:
        raise ValueError(f"unsupported QA rejection category: {category}")
    store = default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
    rejection_audit = metadata.get("qa_rejection_audit")
    if not isinstance(rejection_audit, list) or not rejection_audit:
        raise RuntimeError(f"job {job_id} has no QA rejection to classify")
    latest = rejection_audit[-1] if isinstance(rejection_audit[-1], dict) else {}
    classification = {
        "action": "job_rejection_classified",
        "category": category,
        "classified_at": _utc_now(),
        "rejection_at": latest.get("at"),
        "rejection_reason": latest.get("reason"),
    }
    history = list(metadata.get("qa_rejection_classification_audit") or [])
    history.append(classification)
    metadata["qa_rejection_classification_audit"] = history
    metadata["qa_rejection_classification"] = classification
    store.update_job(job_id, metadata=metadata)
    return {
        "ep_id": ep_id,
        "job": store.get_job(job_id, ep_id=ep_id),
        "jobs": store.list_jobs(ep_id),
    }


def authorize_additional_job_retry(
    ep_id: str, job_id: str, *, reason: str,
) -> dict[str, Any]:
    """Allow exactly one more failed-job retry after explicit reviewer authorization."""
    ep_id = _safe_id(ep_id, "ep_id")
    job_id = str(job_id).strip()
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("additional retry authorization reason is required")
    store = default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    if str(job.get("status") or "") != "failed":
        raise RuntimeError("additional retry authorization requires a failed job")
    retry_count = int(job.get("retry_count") or 0)
    max_retries = int(job.get("max_retries") or 0)
    if retry_count < max_retries:
        raise RuntimeError("job still has an unused retry; extra authorization is not required")
    metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
    audit = list(metadata.get("additional_retry_authorization_audit") or [])
    audit.append({
        "action": "additional_job_retry_authorized",
        "reason": reason,
        "at": _utc_now(),
        "retry_count": retry_count,
        "previous_max_retries": max_retries,
        "new_max_retries": max_retries + 1,
        "failure": job.get("error"),
    })
    metadata["additional_retry_authorization_audit"] = audit
    store.update_job(job_id, max_retries=max_retries + 1, metadata=metadata)
    return {
        "ep_id": ep_id,
        "job": store.get_job(job_id, ep_id=ep_id),
        "jobs": store.list_jobs(ep_id),
    }


def _has_remote_retry_authorization(job: Mapping[str, Any]) -> bool:
    """Return whether an explicit Comfy error authorized this exact prompt retry."""
    prompt_id = str(job.get("prompt_id") or "").strip()
    authorization = (job.get("metadata") or {}).get("remote_retry_authorization") or {}
    return bool(
        prompt_id
        and authorization.get("disposition") == "safe_to_retry"
        and str(authorization.get("prompt_id") or "") == prompt_id
    )


def resume_jobs(ep_id: str, statuses: Iterable[str] = ("pending", "failed")) -> dict[str, Any]:
    _safe_id(ep_id, "ep_id")
    wanted = {"queued" if s == "pending" else s for s in statuses}
    resumed: list[str] = []
    skipped: list[str] = []
    store = default_store()
    for job in store.list_jobs(ep_id):
        if job["status"] not in wanted:
            skipped.append(job["job_id"])
            continue
        dependency_blocked = bool(
            job["status"] == "failed"
            and str(job.get("error") or "").startswith(
                "strict continuity predecessor is not succeeded:"
            )
            and not str(job.get("prompt_id") or "").strip()
        )
        if (
            job["status"] == "failed"
            and not dependency_blocked
            and job["retry_count"] >= job["max_retries"]
        ):
            skipped.append(job["job_id"])
            continue
        # A strict-predecessor block submitted no GPU prompt and is not a real
        # attempt.  Older workers recorded it as failed; restore its retry
        # budget instead of consuming another attempt while unblocking it.
        retry_count = 0 if dependency_blocked else (
            job["retry_count"] + (1 if job["status"] == "failed" else 0)
        )
        changes: dict[str, Any] = {
            "status": "queued", "progress": 0.0,
            "retry_count": retry_count, "completed_at": None,
        }
        if dependency_blocked:
            changes["error"] = None
        if job["status"] == "failed":
            retry_paths = (job.get("metadata") or {}).get("qa_retry_paths") or {}
            if retry_paths:
                changes.update({
                    "output_path": retry_paths.get("output_path"),
                    "preview_path": retry_paths.get("preview_path"),
                    "prompt_id": None, "comfy_output_path": None,
                    "probe": {}, "submitted_at": None, "error": None,
                })
        store.update_job(job["job_id"], **changes)
        resumed.append(job["job_id"])
    return {"ep_id": ep_id, "resumed": len(resumed), "job_ids": resumed, "skipped": skipped}


def retry_job(ep_id: str, job_id: str) -> dict[str, Any]:
    _safe_id(ep_id, "ep_id")
    store = default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    if job["status"] == "succeeded" and job.get("output_path") and Path(job["output_path"]).exists():
        return job
    if job["retry_count"] >= job["max_retries"]:
        raise RuntimeError(f"retry limit reached for {job_id}")
    metadata = dict(job.get("metadata") or {})
    rejection_audit = metadata.get("qa_rejection_audit")
    latest_rejection = (
        rejection_audit[-1]
        if isinstance(rejection_audit, list) and rejection_audit
        and isinstance(rejection_audit[-1], dict)
        else {}
    )
    if latest_rejection.get("reason"):
        feedback_payload = {
            "reason": str(latest_rejection["reason"]),
            "category": str(latest_rejection.get("category") or "other"),
            "at": str(latest_rejection.get("at") or ""),
            "source": "human_qa",
        }
        feedback_payload["sha256"] = hashlib.sha256(
            json.dumps(
                feedback_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        metadata["qa_retry_feedback"] = feedback_payload
    prompt_id = str(job.get("prompt_id") or "").strip()
    if prompt_id and not _has_remote_retry_authorization(job):
        raise RuntimeError(
            f"remote prompt {prompt_id} must be reconciled before retrying {job_id}"
        )
    if prompt_id and job.get("status") != "failed":
        raise RuntimeError(f"remote retry requires failed status for {job_id}")
    authorization = metadata.pop("remote_retry_authorization", None)
    if authorization:
        history = list(metadata.get("remote_reconciliation_history") or [])
        history.append({**authorization, "consumed_at": _utc_now()})
        metadata["remote_reconciliation_history"] = history[-50:]
    retry_paths = metadata.get("qa_retry_paths") or {}
    updated = store.compare_and_update_job(
        job_id,
        expected={
            "status": job["status"], "prompt_id": job.get("prompt_id"),
            "retry_count": job["retry_count"],
        },
        status="queued", progress=0.0, error=None,
        retry_count=job["retry_count"] + 1, completed_at=None,
        output_path=retry_paths.get("output_path") or job.get("output_path"),
        preview_path=retry_paths.get("preview_path") or job.get("preview_path"),
        prompt_id=None, comfy_output_path=None, probe={}, submitted_at=None,
        metadata=metadata,
    )
    if updated is None:
        raise RuntimeError(f"job changed concurrently before retry: {job_id}")
    return updated


def approve_job_review(
    ep_id: str,
    job_id: str,
    *,
    expected_artifact_sha256: str,
    expected_edit_selection_sha256: str,
    reviewed_by: str = "reviewer",
    reason: str = "editorial content approved",
) -> dict[str, Any]:
    """Approve one content-QA-passed artifact by its immutable bytes.

    This is editorial acceptance, not the episode release switch.  Both are
    required by delivery so an old technical ``succeeded`` row cannot ship.
    """
    ep_id = _safe_id(ep_id, "ep_id")
    store = default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    if job.get("status") != "succeeded":
        raise RuntimeError("editorial review requires a technically succeeded job")
    output = Path(str(job.get("output_path") or ""))
    if not output.is_file():
        raise FileNotFoundError(f"job output is missing: {output}")
    actual_artifact = _file_sha256(output)
    expected = str(expected_artifact_sha256 or "").lower().strip()
    if not expected or actual_artifact != expected:
        raise RuntimeError("editorial review artifact hash is stale or mismatched")
    metadata = dict(job.get("metadata") or {})
    settings = dict(metadata.get("settings") or {})
    if settings.get("delivery_eligible") is not True or settings.get("render_profile") != "production":
        raise RuntimeError(
            "proof renders cannot receive release approval; approve and promote the preview first"
        )
    _validate_job_edit_selection(job, metadata, actual_artifact)
    content_qa = dict(metadata.get("content_qa") or {})
    analysis = dict(content_qa.get("analysis") or {})
    visual_hash = str(analysis.get("decoded_visual_sha256") or "")
    if not content_qa.get("passed") or not visual_hash:
        raise RuntimeError("editorial review requires persisted passing content QA")
    edit_selection = dict(metadata.get("edit_selection") or {})
    selection_hash = str(edit_selection.get("selection_sha256") or "")
    if not selection_hash or selection_hash != str(expected_edit_selection_sha256 or "").strip():
        raise RuntimeError("editorial review edit selection hash is missing, stale or mismatched")
    if edit_selection.get("source_artifact_sha256") != actual_artifact:
        raise RuntimeError("editorial review edit selection belongs to another source artifact")
    reviewed_by = str(reviewed_by).strip()
    reason = str(reason).strip()
    if not reviewed_by or not reason:
        raise ValueError("editorial review requires reviewed_by and reason")
    previous = dict(metadata.get("editorial_review") or {})
    history = list(metadata.get("editorial_review_history") or [])
    if previous:
        history.append(previous)
    review = {
        "status": "approved", "artifact_sha256": actual_artifact,
        "decoded_visual_sha256": visual_hash, "approved_at": _utc_now(),
        "edit_selection_sha256": selection_hash,
        "approved_by": reviewed_by, "reason": reason,
    }
    metadata["editorial_review"] = review
    metadata["editorial_review_history"] = history[-50:]
    metadata["release"] = {"status": "pending", "reason": "episode release not approved"}
    return store.update_job(job_id, metadata=metadata)


def approve_preview_and_promote(
    ep_id: str,
    job_id: str,
    *,
    expected_artifact_sha256: str,
    expected_edit_selection_sha256: str,
    reviewed_by: str = "reviewer",
    reason: str = "proof prompt, references, motion and continuity approved",
) -> dict[str, Any]:
    """Bind a passing proof artifact and queue the same panel for production.

    Promotion never relabels the low-cost proof as a deliverable.  Its file,
    graph, prompt/reference hashes and decoded-video QA stay under ``previews``
    as immutable audit evidence; re-preparing the unchanged creative contract
    then creates a fresh production input/output path for the same panel job.
    """
    ep_id = _safe_id(ep_id, "ep_id")
    store = default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    if job.get("status") != "succeeded":
        raise RuntimeError("preview promotion requires a technically succeeded proof")
    metadata = dict(job.get("metadata") or {})
    settings = dict(metadata.get("settings") or {})
    if settings.get("render_profile") != "proof" or settings.get("delivery_eligible") is not False:
        raise RuntimeError("only a non-deliverable proof profile can be promoted")
    output = Path(str(job.get("output_path") or ""))
    project = (projects_dir() / ep_id).resolve()
    if not output.is_file():
        raise FileNotFoundError(f"proof output is missing: {output}")
    try:
        output.resolve().relative_to((project / "previews").resolve())
    except ValueError as exc:
        raise RuntimeError("proof output must be stored under the episode previews directory") from exc
    actual_artifact = _file_sha256(output)
    if actual_artifact != str(expected_artifact_sha256 or "").lower().strip():
        raise RuntimeError("preview promotion artifact hash is stale or mismatched")
    content_qa = dict(metadata.get("content_qa") or {})
    visual_hash = str((content_qa.get("analysis") or {}).get("decoded_visual_sha256") or "")
    if not content_qa.get("passed") or not visual_hash:
        raise RuntimeError("preview promotion requires persisted passing content QA")
    _validate_job_edit_selection(job, metadata, actual_artifact)
    selection = dict(metadata.get("edit_selection") or {})
    selection_hash = str(selection.get("selection_sha256") or "")
    if (
        not selection_hash
        or selection_hash != str(expected_edit_selection_sha256 or "").strip()
        or selection.get("source_artifact_sha256") != actual_artifact
    ):
        raise RuntimeError("preview promotion edit selection is missing, stale or mismatched")
    prompt_sha256 = str(metadata.get("prompt_sha256") or settings.get("prompt_sha256") or "")
    reference_bundle_sha256 = str(
        metadata.get("reference_bundle_sha256")
        or settings.get("reference_bundle_sha256") or ""
    )
    if not prompt_sha256 or not reference_bundle_sha256:
        raise RuntimeError("preview promotion requires prompt and reference bundle hashes")
    reviewed_by = str(reviewed_by).strip()
    reason = str(reason).strip()
    if not reviewed_by or not reason:
        raise ValueError("preview promotion requires reviewed_by and reason")
    promotion = {
        "status": "approved",
        "artifact_sha256": actual_artifact,
        "decoded_visual_sha256": visual_hash,
        "edit_selection_sha256": selection_hash,
        "prompt_sha256": prompt_sha256,
        "reference_bundle_sha256": reference_bundle_sha256,
        "profile_id": settings.get("render_profile_id"),
        "profile_contract": settings.get("render_profile_contract") or H3_RENDER_PROFILE_CONTRACT,
        "director_skill_version": metadata.get("director_skill_version") or settings.get("director_skill_version"),
        "output_path": str(output.resolve()),
        "graph_path": job.get("graph_path"),
        "timing_path": job.get("timing_path"),
        "approved_at": _utc_now(),
        "approved_by": reviewed_by,
        "reason": reason,
    }
    history = list(metadata.get("preview_history") or [])
    history.append(promotion)
    metadata["preview_promotion"] = promotion
    metadata["preview_history"] = history[-50:]
    store.update_job(job_id, metadata=metadata)

    episode_path = project / "episode.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode contract is missing: {episode_path}")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    snapshot = _prepare_episode_jobs(ep_id, episode)
    promoted = next(
        (item for item in snapshot.get("jobs") or [] if item.get("job_id") == job_id), None,
    )
    promoted_settings = ((promoted or {}).get("metadata") or {}).get("settings") or {}
    if (
        not promoted
        or promoted.get("status") != "queued"
        or promoted_settings.get("render_profile") != "production"
        or promoted_settings.get("delivery_eligible") is not True
    ):
        raise RuntimeError("preview evidence was saved but production job preparation failed closed")
    return snapshot


def approve_episode_release(
    ep_id: str,
    *,
    expected_artifact_hashes: Mapping[str, str],
    expected_edit_selection_hashes: Mapping[str, str],
    approved_by: str = "reviewer",
    reason: str = "episode editorial release approved",
) -> dict[str, Any]:
    """Release an episode only when every reviewed artifact hash still matches."""
    ep_id = _safe_id(ep_id, "ep_id")
    store = default_store()
    jobs = store.list_jobs(ep_id)
    if not jobs:
        raise RuntimeError("episode has no registered panel jobs")
    provided = {str(key): str(value).lower().strip() for key, value in expected_artifact_hashes.items()}
    provided_selections = {
        str(key): str(value).lower().strip() for key, value in expected_edit_selection_hashes.items()
    }
    if set(provided) != {str(job["job_id"]) for job in jobs}:
        raise RuntimeError("episode release requires exact hashes for every registered panel job")
    if set(provided_selections) != {str(job["job_id"]) for job in jobs}:
        raise RuntimeError("episode release requires exact edit selection hashes for every panel job")
    approved_by = str(approved_by).strip()
    reason = str(reason).strip()
    if not approved_by or not reason:
        raise ValueError("episode release requires approved_by and reason")
    release_at = _utc_now()
    updates: list[tuple[str, Mapping[str, Any]]] = []
    for job in jobs:
        if job.get("status") != "succeeded":
            raise RuntimeError(f"episode release blocked by non-succeeded job {job['job_id']}")
        output = Path(str(job.get("output_path") or ""))
        if not output.is_file():
            raise FileNotFoundError(f"job output is missing: {output}")
        actual = _file_sha256(output)
        if actual != provided[str(job["job_id"])]:
            raise RuntimeError(f"episode release artifact hash mismatch: {job['job_id']}")
        metadata = dict(job.get("metadata") or {})
        settings = dict(metadata.get("settings") or {})
        if settings.get("delivery_eligible") is not True or settings.get("render_profile") != "production":
            raise RuntimeError(f"episode release blocked by non-production render {job['job_id']}")
        content_qa = dict(metadata.get("content_qa") or {})
        review = dict(metadata.get("editorial_review") or {})
        visual_hash = str((content_qa.get("analysis") or {}).get("decoded_visual_sha256") or "")
        selection = dict(metadata.get("edit_selection") or {})
        _validate_job_edit_selection(job, metadata, actual)
        selection_hash = str(selection.get("selection_sha256") or "")
        if not content_qa.get("passed") or not visual_hash:
            raise RuntimeError(f"episode release content QA missing/failed: {job['job_id']}")
        if (
            not selection_hash
            or selection_hash != provided_selections[str(job["job_id"])]
            or selection.get("source_artifact_sha256") != actual
        ):
            raise RuntimeError(f"episode release edit selection stale/missing: {job['job_id']}")
        if (
            review.get("status") != "approved"
            or review.get("artifact_sha256") != actual
            or review.get("decoded_visual_sha256") != visual_hash
            or review.get("edit_selection_sha256") != selection_hash
        ):
            raise RuntimeError(f"episode release editorial review stale/missing: {job['job_id']}")
        history = list(metadata.get("release_history") or [])
        current = dict(metadata.get("release") or {})
        if current:
            history.append(current)
        metadata["release"] = {
            "status": "approved", "artifact_sha256": actual,
            "decoded_visual_sha256": visual_hash, "approved_at": release_at,
            "edit_selection_sha256": selection_hash,
            "approved_by": approved_by, "reason": reason,
        }
        metadata["release_history"] = history[-50:]
        updates.append((str(job["job_id"]), {"metadata": metadata}))
    store.update_jobs_atomic(updates)
    return project_snapshot(ep_id)


def revoke_release(
    ep_id: str,
    *,
    reason: str,
    revoked_by: str = "reviewer",
) -> dict[str, Any]:
    """Revoke release eligibility without deleting or moving any artifact."""
    ep_id = _safe_id(ep_id, "ep_id")
    reason = str(reason).strip()
    revoked_by = str(revoked_by).strip()
    if not reason or not revoked_by:
        raise ValueError("release revocation requires reason and revoked_by")
    store = default_store()
    now = _utc_now()
    updates: list[tuple[str, Mapping[str, Any]]] = []
    for job in store.list_jobs(ep_id):
        metadata = dict(job.get("metadata") or {})
        history = list(metadata.get("release_history") or [])
        current = dict(metadata.get("release") or {})
        if current:
            history.append(current)
        metadata["release"] = {
            "status": "revoked", "revoked_at": now,
            "revoked_by": revoked_by, "reason": reason,
        }
        metadata["release_history"] = history[-50:]
        updates.append((str(job["job_id"]), {"metadata": metadata}))
    if not updates:
        raise RuntimeError("episode has no registered panel jobs")
    store.update_jobs_atomic(updates)
    return project_snapshot(ep_id)


def project_snapshot(ep_id: str) -> dict[str, Any]:
    ep_id = _safe_id(ep_id, "ep_id")
    project = projects_dir() / ep_id
    jobs = list_jobs(ep_id)
    store = default_store()
    asset_records = store.list_assets(ep_id)
    pipeline = store.get_pipeline(ep_id) or {}
    succeeded = sum(job["status"] == "succeeded" for job in jobs)
    failed = sum(job["status"] == "failed" for job in jobs)
    progress = sum(job["progress"] for job in jobs) / len(jobs) if jobs else 0.0
    episode: dict[str, Any] = {}
    episode_path = project / "episode.json"
    if episode_path.exists():
        try:
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            episode = {}

    def files(folder: str, suffixes: tuple[str, ...]) -> list[str]:
        base = project / folder
        if not base.exists():
            return []
        return [str(p) for p in sorted(base.rglob("*")) if p.is_file() and p.suffix.lower() in suffixes]

    images = files("charrefs", (".png", ".jpg", ".jpeg", ".webp")) \
             + files("scenerefs", (".png", ".jpg", ".jpeg", ".webp")) \
             + files("images", (".png", ".jpg", ".jpeg", ".webp"))
    videos = files("videos", (".mp4", ".mov", ".webm"))
    deliveries = files("exports", (".json", ".zip", ".mp4", ".vtt", ".srt", ".ass"))
    delivery_reports: list[dict[str, Any]] = []
    for manifest_file in sorted((project / "exports").glob("*.manifest.json")) if (project / "exports").exists() else []:
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        delivery_reports.append({
            "manifest_path": str(manifest_file),
            "output_path": manifest.get("output_path"),
            "preset": manifest.get("preset"),
            "release_status": manifest.get("release_status") or "legacy_unapproved",
            "approved_artifact_hashes": manifest.get("approved_artifact_hashes") or {},
            "approved_visual_hashes": manifest.get("approved_visual_hashes") or {},
            "approved_edit_selection_hashes": manifest.get("approved_edit_selection_hashes") or {},
            "qa_report_path": manifest.get("qa_report_path"),
            "qa_report_sha256": manifest.get("qa_report_sha256"),
        })
    assets = {"images": images, "videos": videos, "items": asset_records}
    qa_items = []
    release_states = []
    for job in jobs:
        metadata = job.get("metadata") or {}
        qa = dict(metadata.get("content_qa") or {})
        review = dict(metadata.get("editorial_review") or {})
        release = dict(metadata.get("release") or {})
        release_states.append(str(release.get("status") or "pending"))
        qa_items.append({
            "job_id": job["job_id"], "passed": bool(qa.get("passed")),
            "content_qa": qa, "editorial_review": review, "release": release,
        })
    release_status = (
        "revoked" if "revoked" in release_states else
        "approved" if release_states and all(state == "approved" for state in release_states) else
        "pending"
    )
    pipeline_view = {
        **pipeline, "contract_errors": _contract_errors(episode) if episode else [],
        "release_status": release_status,
    }
    return {
        "ep_id": ep_id,
        "project_dir": str(project),
        "episode": episode,
        "pipeline": pipeline_view,
        "content_qa": {
            "passed": bool(qa_items) and all(item["passed"] for item in qa_items),
            "release_status": release_status, "jobs": qa_items,
        },
        "assets": assets,
        "deliveries": deliveries,
        "delivery_reports": delivery_reports,
        "characters": episode.get("characters") or episode.get("character_bible") or [],
        "images": images,
        "videos": videos,
        "jobs": jobs,
        "progress": progress,
        "counts": {"total": len(jobs), "succeeded": succeeded, "failed": failed},
    }
