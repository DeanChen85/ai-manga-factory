# -*- coding: utf-8 -*-
"""Pure presentation helpers for the Streamlit production console."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any, Iterable


TERMINAL_SUCCESS = {"completed", "finalized", "succeeded", "success", "delivered"}
TERMINAL_FAILURE = {"failed", "error", "cancelled"}
ACTIVE = {"queued", "pending", "submitted", "running", "rendering", "processing"}

CREATIVE_APPROVAL_KEYS = ("story", "characters", "storyboard")

REJECTION_CATEGORIES_REQUIRING_GROUP_ANCHOR = {
    "identity_or_character",
    "composition_or_scene",
    "continuity_or_state",
    "other",
}


def rejection_requires_group_anchor(metadata: dict[str, Any]) -> bool:
    """Fail closed for legacy/unclassified visual QA rejects, except timing-only rejects."""
    audit = metadata.get("qa_rejection_audit") if isinstance(metadata, dict) else None
    if not isinstance(audit, list) or not audit:
        return False
    latest = audit[-1] if isinstance(audit[-1], dict) else {}
    classification = metadata.get("qa_rejection_classification")
    category = str(latest.get("category") or "").strip()
    if not category and isinstance(classification, dict):
        classification_matches = bool(
            classification.get("rejection_at") == latest.get("at")
            and classification.get("rejection_reason") == latest.get("reason")
        )
        if classification_matches:
            category = str(classification.get("category") or "").strip()
    category = category or "legacy_unclassified"
    if category == "action_timing_or_edit_window":
        return False
    return category in REJECTION_CATEGORIES_REQUIRING_GROUP_ANCHOR or category == "legacy_unclassified"


def generation_wait_notice(
    started_at: float, timeout_seconds: float, planned_calls: int = 1,
) -> dict[str, Any]:
    """Return truthful copy for an explicit synchronous LLM call plan."""
    timeout = float(timeout_seconds)
    calls = max(1, int(planned_calls))
    headline = (
        f"MiniMax 计划 {calls} 次调用；每次最长等待 {timeout:g} 秒"
        if calls > 1 else f"MiniMax 请求已开始；最长等待 {timeout:g} 秒"
    )
    return {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(started_at))),
        "timeout_seconds": timeout,
        "planned_calls": calls,
        "headline": headline,
        "stop_help": (
            "等待期间可使用 Streamlit 页面右上角 Stop 停止本页执行。"
            "Stop 不是远端取消确认：已发出的请求可能仍在服务端处理，请勿立刻连续重复点击，以免重复计费。"
        ),
        "failure_help": (
            "若超时或 API 报错，当前创作输入会保留，且不会保存半成品合同；"
            "系统不会自动重试付费请求，可由你确认后安全重试。"
        ),
    }


def generation_input_signature(
    ep_id: str, story_text: str, settings: dict[str, Any],
) -> str:
    """Hash only the non-secret inputs that bind a resumable Stage-1 draft."""
    forbidden = {"api_key", "progress_cb", "stage1_checkpoint_hash", "checkpoint_hash"}
    clean_settings = {
        str(key): copy.deepcopy(value)
        for key, value in dict(settings or {}).items()
        if str(key) not in forbidden
    }
    payload = {
        "ep_id": str(ep_id or "").strip(),
        "story_text": str(story_text or ""),
        "settings": clean_settings,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def updated_incomplete_stage2_checkpoint(
    before: Iterable[dict[str, Any]], after: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Pick only a checkpoint created or updated by the just-finished attempt."""
    before_by_hash = {
        str(item.get("checkpoint_sha256") or ""): item
        for item in before if isinstance(item, dict)
    }
    for item in after:
        if not isinstance(item, dict):
            continue
        digest = str(item.get("checkpoint_sha256") or "")
        if len(digest) != 64 or item.get("stage1_status") != "validated":
            continue
        if item.get("stage2_status") not in {"pending", "failed"}:
            continue
        previous = before_by_hash.get(digest)
        if previous is None or any(
            item.get(key) != previous.get(key)
            for key in ("updated_at", "stage2_status", "stage2_attempt_count")
        ):
            return copy.deepcopy(item)
    return {}


def stage2_resume_eligibility(
    checkpoint: dict[str, Any], *, saved_ep_id: str, current_ep_id: str,
    saved_input_signature: str = "", current_input_signature: str = "",
    protocol: str, model: str,
) -> dict[str, Any]:
    """Fail closed when a Web resume candidate no longer matches visible inputs."""
    if not checkpoint:
        return {"ready": False, "reason": "没有可续跑的阶段 1 草稿"}
    if str(saved_ep_id or "") != str(current_ep_id or ""):
        return {"ready": False, "reason": "项目 ID 已变化，不能复用原草稿"}
    if (saved_input_signature or current_input_signature) and (
        not saved_input_signature or saved_input_signature != current_input_signature
    ):
        return {"ready": False, "reason": "创作输入或生成设置已变化，阶段 2 续跑已禁用"}
    if checkpoint.get("stage1_status") != "validated":
        return {"ready": False, "reason": "阶段 1 草稿尚未通过硬校验"}
    if checkpoint.get("stage2_status") not in {"pending", "failed"}:
        return {"ready": False, "reason": "该草稿已完成或当前不可续跑"}
    if str(checkpoint.get("protocol") or "").casefold() != str(protocol or "").casefold():
        return {"ready": False, "reason": "MiniMax 协议已变化，阶段 2 续跑已禁用"}
    if str(checkpoint.get("model") or "") != str(model or ""):
        return {"ready": False, "reason": "MiniMax 模型已变化，阶段 2 续跑已禁用"}
    return {"ready": True, "reason": "阶段 1 已保存，阶段 2 未完成"}


def resume_stage2_via_facade(
    handler, story_text: str, *, ep_id: str, checkpoint_hash: str,
    settings: dict[str, Any], api_key: str | None = None, progress_cb=None,
):
    """Call the public Stage-2-only API with the exact current generation settings."""
    forwarded = copy.deepcopy(dict(settings or {}))
    reserved = {
        "ep_id", "checkpoint_hash", "stage1_checkpoint_hash", "api_key", "progress_cb",
    }
    collision = sorted(reserved.intersection(forwarded))
    if collision:
        raise ValueError(f"reserved Stage-2 resume settings: {', '.join(collision)}")
    return handler(
        story_text,
        ep_id=ep_id,
        checkpoint_hash=checkpoint_hash,
        api_key=api_key,
        progress_cb=progress_cb,
        **forwarded,
    )


def with_series_episode_approval(
    series: dict[str, Any], episode_id: str, approved: bool
) -> dict[str, Any]:
    """Return a V4 copy with one generated V3 episode approved/revoked."""
    known = {item.get("episode_id") for item in series.get("season_outline") or []}
    if episode_id not in known:
        raise KeyError(f"unknown series episode: {episode_id}")
    if approved and episode_id not in (series.get("episode_contracts") or {}):
        raise ValueError("cannot approve an episode before its V3 contract is generated")
    updated = copy.deepcopy(series)
    updated.setdefault("episode_approvals", {})[episode_id] = bool(approved)
    return updated


def series_episode_counts(series: dict[str, Any]) -> dict[str, int]:
    outline = series.get("season_outline") or []
    contracts = series.get("episode_contracts") or {}
    approvals = series.get("episode_approvals") or {}
    return {
        "total": len(outline),
        "generated": sum(item.get("episode_id") in contracts for item in outline),
        "approved": sum(bool(approvals.get(item.get("episode_id"))) for item in outline),
    }


def _series_structural_contract(series: dict[str, Any]) -> dict[str, Any]:
    structural = copy.deepcopy(series)
    for key in (
        "episode_contracts", "episode_approvals", "season_approved",
        "quality_warnings", "backend_status",
    ):
        structural.pop(key, None)
    runtime_asset_fields = {
        "reference_images", "asset_status", "asset_hash", "asset_manifest_path",
        "asset_approval", "asset_rejection_history", "approved", "approved_at", "error",
    }
    for collection in ("shared_character_bible", "shared_scene_bible"):
        for item in structural.get(collection) or []:
            for key in runtime_asset_fields:
                item.pop(key, None)
    return structural


def series_service_spec(series: dict[str, Any]) -> dict[str, Any]:
    """Adapt a V4 contract to the public series_service durable spec."""
    bible = series.get("series_bible") or {}
    brief = series.get("creative_brief") or {}
    title = str(bible.get("title") or brief.get("topic") or "").strip()
    theme = str(brief.get("topic") or ", ".join(bible.get("themes") or []) or title).strip()
    synopsis = str(bible.get("premise") or brief.get("synopsis") or "").strip()
    if not title or not theme or not synopsis:
        raise ValueError("V4 series title/theme/synopsis are required for durable preparation")
    structural = _series_structural_contract(series)
    return {
        "schema_version": "ai-manga.series-service-spec/v1",
        "title": title,
        "theme": theme,
        "synopsis": synopsis,
        "episode_count": int(series.get("episode_count") or 0),
        "episode_seconds": float(series.get("seconds_per_episode") or 0),
        "story_bible": copy.deepcopy(bible),
        "visual_bible": copy.deepcopy(series.get("visual_bible") or {}),
        "world_bible": copy.deepcopy(series.get("world_bible") or {}),
        "character_bible": copy.deepcopy(series.get("shared_character_bible") or []),
        "scene_bible": copy.deepcopy(series.get("shared_scene_bible") or []),
        "season_outline": copy.deepcopy(series.get("season_outline") or []),
        "v4_contract": structural,
        # series_store.canonical_series deliberately removes runtime, so
        # per-episode generation/approval can persist without invalidating the
        # approved shared season contract.
        "runtime": {"v4_contract": copy.deepcopy(series)},
    }


def series_from_service_snapshot(
    snapshot: dict[str, Any], fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Restore V4 UI state and hydrate backend-owned shared asset fields."""
    row = snapshot.get("series") if isinstance(snapshot.get("series"), dict) else {}
    spec = row.get("spec") if isinstance(row.get("spec"), dict) else {}
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    raw = runtime.get("v4_contract") or spec.get("v4_contract") or fallback or {}
    if not isinstance(raw, dict) or not raw:
        return copy.deepcopy(fallback or {})
    result = copy.deepcopy(raw)
    if isinstance(spec.get("character_bible"), list):
        result["shared_character_bible"] = copy.deepcopy(spec["character_bible"])
    if isinstance(spec.get("scene_bible"), list):
        result["shared_scene_bible"] = copy.deepcopy(spec["scene_bible"])
    result["season_approved"] = row.get("status") == "approved"
    result["backend_status"] = {
        "series_id": snapshot.get("series_id") or row.get("series_id"),
        "status": row.get("status"),
        "contract_hash": row.get("contract_hash"),
        "shared_assets_status": row.get("shared_assets_status"),
        "shared_assets_hash": row.get("shared_assets_hash"),
        "counts": copy.deepcopy(snapshot.get("counts") or {}),
        "ready": bool(snapshot.get("ready")),
    }
    return result


def merge_series_backend_assets(
    series: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, Any]:
    """Overlay backend-owned shared asset runtime fields on newer local V4 state."""
    updated = copy.deepcopy(series)
    row = snapshot.get("series") if isinstance(snapshot.get("series"), dict) else {}
    spec = row.get("spec") if isinstance(row.get("spec"), dict) else {}
    if isinstance(spec.get("character_bible"), list):
        updated["shared_character_bible"] = copy.deepcopy(spec["character_bible"])
    if isinstance(spec.get("scene_bible"), list):
        updated["shared_scene_bible"] = copy.deepcopy(spec["scene_bible"])
    return updated


def series_registration_payloads(series: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact ordered V3 contracts only when every V4 episode is approved."""
    outline = series.get("season_outline") or []
    contracts = series.get("episode_contracts") or {}
    approvals = series.get("episode_approvals") or {}
    expected = int(series.get("episode_count") or 0)
    if len(outline) != expected:
        raise ValueError(f"exactly {expected} outline episodes are required")
    series_id = str((series.get("series_bible") or {}).get("series_id") or "series")
    payloads: list[dict[str, Any]] = []
    for number, item in enumerate(outline, 1):
        episode_id = str(item.get("episode_id") or "")
        contract = contracts.get(episode_id)
        if not isinstance(contract, dict) or not approvals.get(episode_id):
            raise ValueError(f"{episode_id or number} must be generated and approved before registration")
        payload = copy.deepcopy(contract)
        payload["episode_number"] = number
        payload["ep_id"] = f"{series_id}_ep_{number:03d}"
        payloads.append(payload)
    return payloads


def prepare_series_via_facade(
    service: Any,
    series: dict[str, Any],
    current_snapshot: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist V4 through the public facade and return backend-hydrated UI state."""
    handler = getattr(service, "prepare_series", None)
    if not callable(handler):
        raise RuntimeError("series_service.prepare_series public facade is unavailable")
    series_id = str((series.get("series_bible") or {}).get("series_id") or "").strip()
    if not series_id:
        raise ValueError("V4 series_bible.series_id is required")
    merged = (
        merge_series_backend_assets(series, current_snapshot)
        if current_snapshot
        else copy.deepcopy(series)
    )
    snapshot = handler(series_id, series_service_spec(merged))
    return series_from_service_snapshot(snapshot, merged), snapshot


def register_series_episodes_via_facade(
    service: Any, series: dict[str, Any]
) -> dict[str, Any]:
    """Register the exact approved V3 season through the public facade only."""
    handler = getattr(service, "register_episodes", None)
    if not callable(handler):
        raise RuntimeError("series_service.register_episodes public facade is unavailable")
    series_id = str((series.get("series_bible") or {}).get("series_id") or "").strip()
    if not series_id:
        raise ValueError("V4 series_bible.series_id is required")
    return handler(series_id, series_registration_payloads(series))


def normalize_jobs(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("jobs", value.get("items", []))
    if not isinstance(value, list):
        return []
    jobs = []
    for index, item in enumerate(value, 1):
        if not isinstance(item, dict):
            continue
        job = copy.deepcopy(item)
        job.setdefault("job_id", job.get("panel_id") or f"job_{index:02d}")
        job.setdefault("panel_id", job.get("panel_name") or job.get("scene_name") or job["job_id"])
        job["status"] = str(job.get("status") or "pending").lower()
        job.setdefault("attempt", 0)
        job.setdefault("inputs", {})
        job.setdefault("outputs", {})
        jobs.append(job)
    return jobs


def job_counts(jobs: Iterable[dict[str, Any]]) -> dict[str, int]:
    result = {"total": 0, "success": 0, "active": 0, "failed": 0, "other": 0}
    for job in jobs:
        result["total"] += 1
        status = str(job.get("status") or "pending").lower()
        if status in TERMINAL_SUCCESS:
            result["success"] += 1
        elif status in TERMINAL_FAILURE:
            result["failed"] += 1
        elif status in ACTIVE:
            result["active"] += 1
        else:
            result["other"] += 1
    return result


def _readiness_row(
    key: str, label: str, state: str, detail: str,
) -> dict[str, Any]:
    """Build one auditable readiness row without collapsing unknown into ready."""
    if state not in {"ready", "pending", "blocked"}:
        raise ValueError(f"unsupported readiness state: {state}")
    return {
        "key": key,
        "label": label,
        "state": state,
        "ready": state == "ready",
        "blocking": state == "blocked",
        "detail": detail,
    }


def _subtitle_timeline_matches_dialogue(panel: dict[str, Any]) -> tuple[bool, str]:
    spoken = [item for item in panel.get("spoken_dialogue") or [] if isinstance(item, dict)]
    subtitles = [item for item in panel.get("subtitle_timeline") or [] if isinstance(item, dict)]
    if len(spoken) != len(subtitles):
        return False, f"口播 {len(spoken)} 条，但字幕 {len(subtitles)} 条"
    required = ("speaker_id", "text", "start_s", "end_s")
    for index, (line, subtitle) in enumerate(zip(spoken, subtitles), 1):
        for key in required:
            left = line.get(key)
            right = subtitle.get(key)
            if key in {"start_s", "end_s"}:
                try:
                    matches = abs(float(left) - float(right)) <= 0.001
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = bool(str(left or "").strip()) and str(left).strip() == str(right or "").strip()
            if not matches:
                return False, f"第 {index} 条字幕的 {key} 与批准口播不一致或缺失"
    warnings = [str(value).strip() for value in panel.get("subtitle_warnings") or [] if str(value).strip()]
    if warnings:
        return False, f"仍有 {len(warnings)} 条字幕 mismatch 警告"
    return True, f"{len(spoken)} 条字幕逐项绑定批准口播" if spoken else "本镜无口播，字幕轨为空"


def _job_provider_label(job: dict[str, Any], snapshot: dict[str, Any]) -> str:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    inputs = metadata.get("inputs") if isinstance(metadata.get("inputs"), dict) else {}
    settings = inputs.get("settings") if isinstance(inputs.get("settings"), dict) else {}
    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
    sources = (job, metadata, settings, pipeline)
    for key in ("provider", "render_provider", "renderer", "engine", "model", "render_mode"):
        for source in sources:
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return ""


def shot_readiness_rows(
    panel: dict[str, Any], episode: dict[str, Any], snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a fail-closed, human-readable production checklist for one shot.

    This intentionally treats missing backend proof as blocked rather than
    inferring readiness from a pretty prompt or a terminal status alone.
    """
    panel_id = str(panel.get("panel_id") or panel.get("name") or "").strip()
    scene_id = str(panel.get("scene_id") or "").strip()
    rows: list[dict[str, Any]] = []
    identity_missing = [name for name, value in (("panel_id", panel_id), ("scene_id", scene_id)) if not value]
    rows.append(_readiness_row(
        "identity", "基础 ID / 场景", "blocked" if identity_missing else "ready",
        "缺少 " + ", ".join(identity_missing) if identity_missing else f"{panel_id} → {scene_id}",
    ))

    action = str(panel.get("visible_action") or "").strip()
    components = panel.get("action_components") if isinstance(panel.get("action_components"), dict) else {}
    missing_action = [key for key in ("sub", "verb", "obj", "res") if not str(components.get(key) or "").strip()]
    action_ready = bool(action) and not missing_action
    action_detail = (
        f"单一可见动作已结构化：{action}"
        if action_ready else
        "缺少 visible_action" if not action else
        "action_components 缺少 " + ", ".join(missing_action)
    )
    rows.append(_readiness_row(
        "action", "结构化可见动作", "ready" if action_ready else "blocked", action_detail,
    ))

    camera = panel.get("camera_plan") if isinstance(panel.get("camera_plan"), dict) else {}
    missing_camera = [
        key for key in ("shot_size", "angle", "movement", "composition")
        if not str(camera.get(key) or "").strip()
    ]
    rows.append(_readiness_row(
        "camera", "镜头语言四项", "blocked" if missing_camera else "ready",
        "缺少 " + ", ".join(missing_camera) if missing_camera else
        "shot_size / angle / movement / composition 均已声明",
    ))

    first_state = str(panel.get("first_state") or "").strip()
    final_state = str(panel.get("final_state") or "").strip()
    first_frame = str(panel.get("first_frame") or "").strip()
    last_frame = str(panel.get("last_frame") or "").strip()
    state_ready = bool(
        first_state and final_state and first_state.casefold() != final_state.casefold()
        and first_frame and last_frame
    )
    rows.append(_readiness_row(
        "first_final", "First / Final 状态与帧", "ready" if state_ready else "blocked",
        "首尾状态有可见变化，且首帧/尾帧均已声明" if state_ready else
        "必须提供不同的 first_state/final_state，并同时提供 first_frame/last_frame",
    ))

    characters = {
        str(item.get("character_id") or "").strip(): item
        for item in episode.get("character_bible") or [] if isinstance(item, dict)
    }
    scenes = {
        str(item.get("scene_id") or "").strip(): item
        for item in episode.get("scene_bible") or [] if isinstance(item, dict)
    }
    character_ids = [str(value).strip() for value in panel.get("character_ids") or [] if str(value).strip()]
    unknown_characters = [value for value in character_ids if value not in characters]
    spoken_speakers = [
        str(item.get("speaker_id") or "").strip()
        for item in panel.get("spoken_dialogue") or [] if isinstance(item, dict)
    ]
    invalid_speakers = [value for value in spoken_speakers if value not in character_ids]
    references_ready = bool(
        character_ids and scene_id in scenes and not unknown_characters and not invalid_speakers
    )
    reference_problems: list[str] = []
    if not character_ids:
        reference_problems.append("character_ids 为空")
    if unknown_characters:
        reference_problems.append("未知人物 " + ", ".join(unknown_characters))
    if scene_id not in scenes:
        reference_problems.append(f"未知场景 {scene_id or '-'}")
    if invalid_speakers:
        reference_problems.append("说话人未在本镜可见角色中：" + ", ".join(invalid_speakers))
    rows.append(_readiness_row(
        "references", "人物 / 场景引用", "ready" if references_ready else "blocked",
        f"{len(character_ids)} 位人物与场景 {scene_id} 均命中 bible"
        if references_ready else "; ".join(reference_problems),
    ))

    subtitles_ready, subtitle_detail = _subtitle_timeline_matches_dialogue(panel)
    rows.append(_readiness_row(
        "dialogue_subtitles", "对白 / 字幕一致", "ready" if subtitles_ready else "blocked",
        subtitle_detail,
    ))

    assets_section = snapshot.get("assets") if isinstance(snapshot.get("assets"), dict) else {}
    asset_items = assets_section.get("items") if isinstance(assets_section.get("items"), list) else []
    asset_map = {
        (str(item.get("asset_type") or ""), str(item.get("source_id") or "")): item
        for item in asset_items if isinstance(item, dict)
    }
    asset_problems: list[str] = []
    required_assets = [("character", value) for value in character_ids]
    if scene_id:
        required_assets.append(("scene", scene_id))
    for asset_type, source_id in required_assets:
        bible_item = characters.get(source_id) if asset_type == "character" else scenes.get(source_id)
        if not bible_item or not list(bible_item.get("reference_images") or []):
            asset_problems.append(f"{asset_type}:{source_id} 缺少 reference_images")
        record = asset_map.get((asset_type, source_id))
        if not record:
            asset_problems.append(f"{asset_type}:{source_id} 缺少后端资产记录")
        elif not (
            str(record.get("status") or "").lower() == "succeeded"
            and bool(record.get("approved"))
            and str(record.get("content_hash") or "").strip()
            and list(record.get("reference_images") or [])
        ):
            asset_problems.append(f"{asset_type}:{source_id} 尚未成功并绑定已批准 content_hash")
        elif bible_item:
            displayed_refs = {str(value).strip() for value in bible_item.get("reference_images") or []}
            backend_refs = {str(value).strip() for value in record.get("reference_images") or []}
            displayed_hash = str(bible_item.get("asset_hash") or "").strip()
            backend_hash = str(record.get("content_hash") or "").strip()
            if displayed_refs != backend_refs:
                asset_problems.append(f"{asset_type}:{source_id} 页面引用与后端批准引用不一致")
            if displayed_hash and displayed_hash != backend_hash:
                asset_problems.append(f"{asset_type}:{source_id} 页面 asset_hash 已过期")
    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
    if str(pipeline.get("assets_status") or "").lower() != "approved":
        asset_problems.append("整集 assets_status 尚未 approved")
    rows.append(_readiness_row(
        "asset_refs", "批准资产与引用", "blocked" if asset_problems else "ready",
        "; ".join(asset_problems) if asset_problems else f"{len(required_assets)} 项资产均有批准 hash 与引用",
    ))

    job = next((
        item for item in normalize_jobs(snapshot.get("jobs") or [])
        if panel_id and panel_id in {
            str(item.get("panel_id") or ""), str(item.get("panel_name") or "")
        }
    ), {})
    if not job:
        rows.append(_readiness_row(
            "job_provider", "任务 / 渲染方状态", "blocked",
            "任务尚未注册，渲染方与执行状态均不可审计",
        ))
    else:
        status = str(job.get("status") or "").lower()
        provider = _job_provider_label(job, snapshot)
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        artifact = str(metadata.get("artifact_sha256") or "").strip()
        output = str(job.get("output_path") or "").strip()
        if not provider:
            state = "blocked"
            detail = f"job={job.get('job_id')} 已注册，但 provider/model/render_mode 未上报"
        elif status in TERMINAL_FAILURE:
            state = "blocked"
            detail = f"{provider} · {status}：{job.get('error') or '任务失败，需处理'}"
        elif status in ACTIVE:
            state = "pending"
            detail = f"{provider} · {status}，尚未产生可验收产物"
        elif status in TERMINAL_SUCCESS and artifact and output:
            state = "ready"
            detail = f"{provider} · {status} · artifact {artifact[:12]}"
        elif status in TERMINAL_SUCCESS:
            state = "blocked"
            detail = f"{provider} · {status}，但缺少 artifact_sha256 或 output_path"
        else:
            state = "blocked"
            detail = f"{provider} · 未知任务状态 {status or '-'}"
        rows.append(_readiness_row("job_provider", "任务 / 渲染方状态", state, detail))
    return rows


def _review_record(
    snapshot: dict[str, Any], job: dict[str, Any], section: str,
) -> dict[str, Any]:
    """Return a per-artifact QA/review record without treating aggregates as proof."""
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    local_keys = {
        "content_qa": ("content_qa", "automated_qa", "quality_gate"),
        "editorial_review": ("editorial_review", "human_review", "content_review"),
    }[section]
    for key in local_keys:
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    top_keys = {
        "content_qa": ("content_qa", "automated_qa"),
        "editorial_review": ("editorial_reviews", "human_reviews"),
    }[section]
    job_id = str(job.get("job_id") or "")
    panel_id = str(job.get("panel_id") or job.get("panel_name") or "")
    for key in top_keys:
        top = snapshot.get(key)
        if not isinstance(top, dict):
            continue
        records = top.get("items") or top.get("jobs") or top.get("panels") or {}
        if isinstance(records, dict):
            value = records.get(job_id) or records.get(panel_id)
            if isinstance(value, dict):
                return value
        if isinstance(records, list):
            for value in records:
                if not isinstance(value, dict):
                    continue
                record_id = str(value.get("job_id") or value.get("panel_id") or "")
                if record_id in {job_id, panel_id}:
                    return value
    return {}


def _artifact_bound_pass(record: dict[str, Any], artifact_sha256: str) -> bool:
    status = str(record.get("status") or record.get("result") or "").lower()
    record_hash = str(
        record.get("artifact_sha256")
        or record.get("expected_artifact_sha256")
        or record.get("artifact_hash")
        or ""
    )
    return bool(
        artifact_sha256
        and record_hash == artifact_sha256
        and status in {"passed", "pass", "approved", "accepted"}
    )


def _content_qa_passes_current_job(record: dict[str, Any], job: dict[str, Any]) -> bool:
    """Accept backend QA only with decoded visual evidence for this output path."""
    passed = bool(record.get("passed")) or str(
        record.get("status") or record.get("result") or ""
    ).lower() in {"passed", "pass"}
    analysis = record.get("analysis") if isinstance(record.get("analysis"), dict) else {}
    visual_hash = str(analysis.get("decoded_visual_sha256") or record.get("decoded_visual_sha256") or "")
    source = str(analysis.get("source_path") or record.get("source_path") or "")
    output = str(job.get("output_path") or "")
    if not passed or not visual_hash or not source or not output:
        return False
    try:
        return Path(source).resolve() == Path(output).resolve()
    except OSError:
        return False


def content_review_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Summarize four independent delivery gates, failing closed on stale proof."""
    jobs = normalize_jobs(snapshot.get("jobs") or [])
    summary: dict[str, Any] = {
        "total": len(jobs), "technical_complete": 0, "automated_qa_passed": 0,
        "human_approved": 0, "rejected": 0, "release_approved": False,
        "same_anchor_safe_count": 0, "same_anchor_sha256": None,
    }
    anchors: list[str] = []
    artifact_hashes: dict[str, str] = {}
    edit_selection_hashes: dict[str, str] = {}
    release_bound_count = 0
    for job in jobs:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        artifact = str(metadata.get("artifact_sha256") or "")
        job_id = str(job.get("job_id") or "")
        if artifact:
            artifact_hashes[job_id] = artifact
        selection = metadata.get("edit_selection") if isinstance(metadata.get("edit_selection"), dict) else {}
        selection_hash = str(selection.get("selection_sha256") or "")
        selection_valid = bool(
            artifact and selection_hash
            and str(selection.get("source_artifact_sha256") or "") == artifact
            and float(selection.get("out_seconds") or 0) > float(selection.get("in_seconds") or 0)
            and 1.5 <= float(selection.get("duration_seconds") or 0) <= 4.0
            and abs(
                (float(selection.get("out_seconds") or 0) - float(selection.get("in_seconds") or 0))
                - float(selection.get("duration_seconds") or 0)
            ) <= 0.01
        )
        if selection_valid:
            edit_selection_hashes[job_id] = selection_hash
        technical = str(job.get("status") or "").lower() in TERMINAL_SUCCESS
        if technical:
            summary["technical_complete"] += 1
        qa = _review_record(snapshot, job, "content_qa")
        review = _review_record(snapshot, job, "editorial_review")
        if technical and _content_qa_passes_current_job(qa, job):
            summary["automated_qa_passed"] += 1
        if (
            technical and selection_valid and _artifact_bound_pass(review, artifact)
            and str(review.get("edit_selection_sha256") or "") == selection_hash
        ):
            summary["human_approved"] += 1
        rejection_statuses = {
            str(qa.get("status") or qa.get("result") or "").lower(),
            str(review.get("status") or review.get("result") or "").lower(),
        }
        current_failed = str(job.get("status") or "").lower() in TERMINAL_FAILURE
        if rejection_statuses & {"failed", "rejected", "reject", "revoked"} or (
            current_failed and (
                metadata.get("qa_rejection_audit")
                or metadata.get("qa_invalidation_audit")
                or "qa rejected" in str(job.get("error") or "").lower()
            )
        ):
            summary["rejected"] += 1
        safe = metadata.get("continuity_safe") if isinstance(metadata.get("continuity_safe"), dict) else {}
        if metadata.get("render_mode") == "continuity_safe" and safe.get("source_anchor_sha256"):
            anchors.append(str(safe["source_anchor_sha256"]))
        release_record = metadata.get("release") if isinstance(metadata.get("release"), dict) else {}
        qa_analysis = qa.get("analysis") if isinstance(qa.get("analysis"), dict) else {}
        if (
            str(release_record.get("status") or "").lower() in {"approved", "released"}
            and artifact
            and str(release_record.get("artifact_sha256") or "") == artifact
            and str(qa_analysis.get("decoded_visual_sha256") or "")
            == str(release_record.get("decoded_visual_sha256") or "")
            and selection_valid
            and str(release_record.get("edit_selection_sha256") or "") == selection_hash
        ):
            release_bound_count += 1
    if len(anchors) >= 2 and len(set(anchors)) == 1:
        summary["same_anchor_safe_count"] = len(anchors)
        summary["same_anchor_sha256"] = anchors[0]

    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
    release = snapshot.get("release") if isinstance(snapshot.get("release"), dict) else {}
    release_status = str(
        pipeline.get("release_status")
        or release.get("status")
        or snapshot.get("release_status")
        or ""
    ).lower()
    approved_hashes = release.get("approved_artifact_hashes") or pipeline.get("approved_artifact_hashes")
    approved_selections = (
        release.get("approved_edit_selection_hashes")
        or pipeline.get("approved_edit_selection_hashes")
    )
    hashes_bound = False
    if isinstance(approved_hashes, dict):
        hashes_bound = bool(artifact_hashes) and approved_hashes == artifact_hashes
    elif isinstance(approved_hashes, list):
        hashes_bound = bool(artifact_hashes) and sorted(map(str, approved_hashes)) == sorted(artifact_hashes.values())
    per_job_release_bound = bool(jobs) and release_bound_count == len(jobs)
    selections_bound = bool(edit_selection_hashes) and isinstance(approved_selections, dict) and (
        approved_selections == edit_selection_hashes
    )
    summary["release_approved"] = release_status in {"approved", "released"} and (
        (hashes_bound and selections_bound) or per_job_release_bound
    )
    summary["release_status"] = release_status or "missing"
    summary["artifact_hashes"] = artifact_hashes
    summary["edit_selection_hashes"] = edit_selection_hashes
    summary["ready_for_export"] = bool(
        jobs
        and summary["technical_complete"] == len(jobs)
        and summary["automated_qa_passed"] == len(jobs)
        and summary["human_approved"] == len(jobs)
        and summary["release_approved"]
    )
    return summary


def _current_edit_selection(
    job: dict[str, Any], artifact_sha256: str,
) -> tuple[bool, str]:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    selection = metadata.get("edit_selection") if isinstance(metadata.get("edit_selection"), dict) else {}
    selection_hash = str(selection.get("selection_sha256") or "").strip()
    try:
        in_seconds = float(selection.get("in_seconds"))
        out_seconds = float(selection.get("out_seconds"))
        duration = float(selection.get("duration_seconds"))
    except (TypeError, ValueError):
        return False, selection_hash
    valid = bool(
        artifact_sha256 and selection_hash
        and str(selection.get("source_artifact_sha256") or "") == artifact_sha256
        and out_seconds > in_seconds
        and 1.5 <= duration <= 4.0
        and abs((out_seconds - in_seconds) - duration) <= 0.01
    )
    return valid, selection_hash


def _job_release_matches_current(
    snapshot: dict[str, Any], job: dict[str, Any], artifact_sha256: str,
    selection_sha256: str, qa: dict[str, Any],
) -> bool:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    release_record = metadata.get("release") if isinstance(metadata.get("release"), dict) else {}
    qa_analysis = qa.get("analysis") if isinstance(qa.get("analysis"), dict) else {}
    visual_sha256 = str(qa_analysis.get("decoded_visual_sha256") or qa.get("decoded_visual_sha256") or "")
    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
    release = snapshot.get("release") if isinstance(snapshot.get("release"), dict) else {}
    release_status = str(
        pipeline.get("release_status") or release.get("status") or snapshot.get("release_status") or ""
    ).lower()
    if release_status not in {"approved", "released"}:
        return False
    per_job_bound = bool(
        str(release_record.get("status") or "").lower() in {"approved", "released"}
        and artifact_sha256
        and str(release_record.get("artifact_sha256") or "") == artifact_sha256
        and selection_sha256
        and str(release_record.get("edit_selection_sha256") or "") == selection_sha256
        and visual_sha256
        and str(release_record.get("decoded_visual_sha256") or "") == visual_sha256
    )
    if per_job_bound:
        return True

    job_id = str(job.get("job_id") or "")
    approved_artifacts = release.get("approved_artifact_hashes") or pipeline.get("approved_artifact_hashes")
    approved_selections = (
        release.get("approved_edit_selection_hashes")
        or pipeline.get("approved_edit_selection_hashes")
    )
    if isinstance(approved_artifacts, dict):
        artifact_bound = str(approved_artifacts.get(job_id) or "") == artifact_sha256
    elif isinstance(approved_artifacts, list):
        artifact_bound = artifact_sha256 in {str(value) for value in approved_artifacts}
    else:
        artifact_bound = False
    return bool(
        artifact_bound and isinstance(approved_selections, dict)
        and str(approved_selections.get(job_id) or "") == selection_sha256
    )


def classify_shot_worklist(
    jobs: Iterable[dict[str, Any]], snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify shots for operators; only current release-bound proof can pass."""
    snapshot = snapshot or {}
    normalized = normalize_jobs(list(jobs))
    buckets: dict[str, list[dict[str, Any]]] = {
        "needs_attention": [], "active": [], "awaiting_review": [], "passed": [],
    }
    all_jobs: list[dict[str, Any]] = []
    for job in normalized:
        classified = copy.deepcopy(job)
        status = str(job.get("status") or "").lower()
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        artifact = str(metadata.get("artifact_sha256") or "").strip()
        qa = _review_record(snapshot, job, "content_qa")
        review = _review_record(snapshot, job, "editorial_review")
        qa_status = str(qa.get("status") or qa.get("result") or "").lower()
        review_status = str(review.get("status") or review.get("result") or "").lower()

        if status in ACTIVE:
            bucket = "active"
            reason = f"任务状态 {status}，尚未产生当前可发布证据"
        elif status in TERMINAL_FAILURE:
            bucket = "needs_attention"
            reason = f"任务状态 {status}，需要重试或人工处理"
        elif status not in TERMINAL_SUCCESS:
            bucket = "needs_attention"
            reason = f"未知或不可交付任务状态 {status or '-'}"
        elif qa_status in {"failed", "rejected", "reject", "revoked"}:
            bucket = "needs_attention"
            reason = "当前内容 QA 已拒收或撤销"
        elif not _content_qa_passes_current_job(qa, job):
            bucket = "needs_attention"
            reason = "技术产物缺少绑定当前 output_path 的 decoded-visual QA 通过证据"
        else:
            selection_valid, selection_sha256 = _current_edit_selection(job, artifact)
            if not artifact or not selection_valid:
                bucket = "needs_attention"
                reason = "当前 artifact 或 1.5-4.0 秒 edit-selection 缺失/失配"
            else:
                review_hash = str(
                    review.get("artifact_sha256")
                    or review.get("expected_artifact_sha256")
                    or review.get("artifact_hash")
                    or ""
                )
                review_rejects_current = bool(
                    review_status in {"failed", "rejected", "reject", "revoked"}
                    and (not review_hash or review_hash == artifact)
                )
                human_current = bool(
                    _artifact_bound_pass(review, artifact)
                    and str(review.get("edit_selection_sha256") or "") == selection_sha256
                )
                if review_rejects_current:
                    bucket = "needs_attention"
                    reason = "当前 artifact 已被人工拒收或撤销"
                elif not human_current:
                    bucket = "awaiting_review"
                    reason = "自动 QA 已通过，等待人工批准当前 artifact 与 edit-selection"
                elif not _job_release_matches_current(
                    snapshot, job, artifact, selection_sha256, qa,
                ):
                    bucket = "awaiting_review"
                    reason = "逐镜人工批准有效，等待整集 release 绑定当前 hashes"
                else:
                    bucket = "passed"
                    reason = "当前 QA、人工批准与 release hashes 全部绑定"
        classified["worklist_state"] = bucket
        classified["worklist_reason"] = reason
        buckets[bucket].append(classified)
        all_jobs.append(classified)

    return {
        **buckets,
        "all": all_jobs,
        "counts": {
            "needs_attention": len(buckets["needs_attention"]),
            "active": len(buckets["active"]),
            "awaiting_review": len(buckets["awaiting_review"]),
            "passed": len(buckets["passed"]),
            "all": len(all_jobs),
        },
    }


def job_review_evidence(
    snapshot: dict[str, Any], job: dict[str, Any], roots: Iterable[Path] = (),
) -> dict[str, Any]:
    """Resolve backend-generated first/middle/last evidence for human review."""
    record = _review_record(snapshot, job, "content_qa")
    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    if not evidence and isinstance(metadata.get("qa_evidence"), dict):
        evidence = metadata["qa_evidence"]
    analysis = record.get("analysis") if isinstance(record.get("analysis"), dict) else {}
    source_analysis = (
        record.get("source_analysis")
        if isinstance(record.get("source_analysis"), dict) else {}
    )
    edit_selection = metadata.get("edit_selection") if isinstance(metadata.get("edit_selection"), dict) else {}
    artifact_sha256 = str(metadata.get("artifact_sha256") or "")
    try:
        edit_in = float(edit_selection.get("in_seconds"))
        edit_out = float(edit_selection.get("out_seconds"))
        selected_duration = float(edit_selection.get("duration_seconds"))
    except (TypeError, ValueError):
        edit_in = edit_out = selected_duration = 0.0
    selection_current = bool(
        str(edit_selection.get("selection_sha256") or "")
        and str(edit_selection.get("source_artifact_sha256") or "") == artifact_sha256
        and edit_in >= 0 and edit_out > edit_in
        and abs((edit_out - edit_in) - selected_duration) <= 1e-4
    )
    # A current selection is the review subject.  The full-source analysis is
    # helpful context only; it must never replace a manually selected action
    # window in the accept/reject evidence.
    source_hashes = list(source_analysis.get("sample_frame_sha256") or [])
    selected_hashes = list(analysis.get("sample_frame_sha256") or [])
    review_analysis = analysis if selection_current else (
        source_analysis if (source_hashes or source_analysis.get("source_path")) else analysis
    )
    paths: dict[str, Path | None] = {}
    aliases = {
        "first": ("first_frame_path", "first_frame", "start_frame_path"),
        "middle": ("middle_frame_path", "middle_frame", "mid_frame_path"),
        "last": ("last_frame_path", "last_frame", "end_frame_path"),
    }
    for slot, keys in aliases.items():
        # Legacy frame files have no selection-hash binding.  Do not present
        # them as selected-window evidence after a human re-edit.
        value = None if selection_current else next(
            (evidence.get(key) for key in keys if evidence.get(key)), None
        )
        resolved = existing_media_paths([value], roots)
        paths[slot] = resolved[0] if resolved else None
    if not any(paths.values()):
        sampled = existing_media_paths(
            review_analysis.get("sample_frame_paths") or record.get("sample_frame_paths") or [], roots,
        )
        if sampled:
            paths = {
                "first": sampled[0],
                "middle": sampled[len(sampled) // 2],
                "last": sampled[-1],
            }
    output_media = existing_media_paths([job.get("output_path")], roots)
    video_path = output_media[0] if output_media else None
    sample_hashes = list(review_analysis.get("sample_frame_sha256") or [])
    source_duration = float((job.get("probe") or {}).get("duration_seconds") or 0)
    review_duration = selected_duration if selection_current else source_duration
    analysis_media = existing_media_paths([review_analysis.get("source_path")], roots)
    analysis_matches_video = bool(
        video_path and analysis_media
        and analysis_media[0].resolve() == video_path.resolve()
    )
    selection_hash_bound_video = bool(
        selection_current and video_path
        and str(edit_selection.get("source_artifact_sha256") or "") == artifact_sha256
    )
    effective_visual_hash = str(
        review_analysis.get("decoded_visual_sha256")
        or (edit_selection.get("source_decoded_visual_sha256") if selection_current else "")
        or ""
    )
    video_sampling_complete = bool(
        video_path and review_duration > 0 and (len(sample_hashes) >= 3 or selection_current)
        and (analysis_matches_video or selection_hash_bound_video)
        and effective_visual_hash
    )
    if evidence.get("action") or evidence.get("action_evidence"):
        action = evidence.get("action") or evidence.get("action_evidence")
        action_source = "content_qa_evidence"
    elif record.get("action_evidence"):
        action = record.get("action_evidence")
        action_source = "content_qa_record"
    else:
        action = edit_selection.get("reason")
        action_source = "edit_selection_fallback" if action else None
    if evidence.get("first_last") or evidence.get("state_transition") or record.get("state_transition"):
        first_last = (
            evidence.get("first_last") or evidence.get("state_transition")
            or record.get("state_transition")
        )
        first_last_source = "content_qa_state_transition"
    else:
        first_last = (review_analysis.get("metrics") or {}).get("first_last_luma_change")
        first_last_source = "decoded_visual_luma_metric" if first_last is not None else None
    return {
        "paths": paths,
        "video_path": video_path,
        "video_timestamps": {
            "first": round(edit_in if selection_current else 0.0, 3),
            "middle": round((edit_in if selection_current else 0.0) + review_duration / 2.0, 3),
            "last": round((edit_in if selection_current else 0.0) + max(0.0, review_duration - min(0.10, review_duration / 4.0)), 3),
        },
        "review_window": {
            "source_start_seconds": round(edit_in if selection_current else 0.0, 6),
            "source_end_seconds": round(edit_out if selection_current else source_duration, 6),
            "duration_seconds": round(review_duration, 6),
            "selection_sha256": str(edit_selection.get("selection_sha256") or "") if selection_current else "",
            "binding": "current_edit_selection" if selection_current else "full_source_context",
        },
        "sample_frame_sha256": sample_hashes,
        "complete": all(paths.values()) or video_sampling_complete,
        "evidence_method": (
            "selection_hash_bound_timestamp_sampling"
            if selection_current and len(sample_hashes) < 3 else "decoded_sample_frames"
        ),
        "action": action,
        "action_source": action_source,
        "first_last": first_last,
        "first_last_source": first_last_source,
        "record": record,
    }


def approve_job_review_via_facade(
    service: Any, ep_id: str, job_id: str, artifact_sha256: str,
    edit_selection_sha256: str,
) -> Any:
    """Approve only the current artifact and its exact editorial selection."""
    handler = next((
        getattr(service, name, None)
        for name in ("approve_job_review", "approve_panel_review", "approve_job")
        if callable(getattr(service, name, None))
    ), None)
    if not callable(handler):
        raise RuntimeError("render_service.approve_job_review public facade is unavailable")
    artifact_sha256 = str(artifact_sha256).strip()
    if not artifact_sha256:
        raise ValueError("human approval requires the current artifact_sha256")
    edit_selection_sha256 = str(edit_selection_sha256).strip()
    if not edit_selection_sha256:
        raise ValueError("human approval requires the current edit_selection_sha256")
    parameters = inspect.signature(handler).parameters
    hash_parameter = next((
        name for name in (
            "expected_artifact_sha256", "artifact_sha256", "expected_artifact_hash",
        ) if name in parameters
    ), None)
    if hash_parameter is None:
        raise RuntimeError("job review facade must bind approval to expected_artifact_sha256")
    selection_parameter = next((
        name for name in ("expected_edit_selection_sha256", "edit_selection_sha256")
        if name in parameters
    ), None)
    if selection_parameter is None:
        raise RuntimeError("job review facade must bind approval to expected_edit_selection_sha256")
    return handler(ep_id, job_id, **{
        hash_parameter: artifact_sha256,
        selection_parameter: edit_selection_sha256,
    })


def approve_preview_and_promote_via_facade(
    service: Any, ep_id: str, job_id: str, artifact_sha256: str,
    edit_selection_sha256: str,
) -> Any:
    """Promote only a passing proof bound to the current bytes and selection."""
    handler = getattr(service, "approve_preview_and_promote", None)
    if not callable(handler):
        raise RuntimeError("render_service.approve_preview_and_promote public facade is unavailable")
    artifact_sha256 = str(artifact_sha256).strip()
    edit_selection_sha256 = str(edit_selection_sha256).strip()
    if not artifact_sha256 or not edit_selection_sha256:
        raise ValueError("preview promotion requires current artifact and edit-selection hashes")
    return handler(
        ep_id, job_id,
        expected_artifact_sha256=artifact_sha256,
        expected_edit_selection_sha256=edit_selection_sha256,
    )


def approve_release_via_facade(
    service: Any, ep_id: str, artifact_hashes: dict[str, str],
    edit_selection_hashes: dict[str, str], *, qa_report_hash: str = "",
) -> Any:
    """Approve release only for the exact reviewed artifact set."""
    handler = next((
        getattr(service, name, None)
        for name in ("approve_episode_release", "approve_release")
        if callable(getattr(service, name, None))
    ), None)
    if not callable(handler):
        raise RuntimeError("render_service.approve_episode_release public facade is unavailable")
    if not artifact_hashes or any(not str(value).strip() for value in artifact_hashes.values()):
        raise ValueError("release approval requires every current artifact hash")
    if set(edit_selection_hashes) != set(artifact_hashes) or any(
        not str(value).strip() for value in edit_selection_hashes.values()
    ):
        raise ValueError("release approval requires every current edit selection hash")
    parameters = inspect.signature(handler).parameters
    hash_parameter = next((
        name for name in ("expected_artifact_hashes", "approved_artifact_hashes", "artifact_hashes")
        if name in parameters
    ), None)
    if hash_parameter is None:
        raise RuntimeError("release facade must bind approval to expected_artifact_hashes")
    kwargs: dict[str, Any] = {hash_parameter: dict(artifact_hashes)}
    selection_parameter = next((
        name for name in ("expected_edit_selection_hashes", "approved_edit_selection_hashes")
        if name in parameters
    ), None)
    if selection_parameter is None:
        raise RuntimeError("release facade must bind approval to expected_edit_selection_hashes")
    kwargs[selection_parameter] = dict(edit_selection_hashes)
    if "qa_report_hash" in parameters and qa_report_hash:
        kwargs["qa_report_hash"] = qa_report_hash
    return handler(ep_id, **kwargs)


def snapshot_episode(snapshot: Any, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(snapshot, dict) and isinstance(snapshot.get("episode"), dict):
        return copy.deepcopy(snapshot["episode"])
    return copy.deepcopy(fallback or {})


def merge_episode_asset_review_state(
    latest: dict[str, Any],
    local: dict[str, Any] | None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge fresh worker assets without losing valid browser review choices.

    Asset approvals are browser review state until the final backend approval
    gate is submitted. Preserve them only while the reviewed content hash and
    reference bundle are unchanged; regenerated assets automatically revoke
    the old choice.
    """
    result = copy.deepcopy(latest or {})
    local = local or {}
    snapshot = snapshot or {}
    if not result:
        return result
    local_state = approval_state(local)
    merged_state = approval_state(result)
    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
    backend_gate_approved = pipeline.get("assets_status") == "approved"
    asset_section = snapshot.get("assets") if isinstance(snapshot.get("assets"), dict) else {}
    backend_records = asset_section.get("items") if isinstance(asset_section.get("items"), list) else []

    def reference_bundle(value: Any) -> tuple[str, ...]:
        normalized: list[str] = []
        for raw in value or []:
            text = str(raw).strip()
            if not text:
                continue
            if "://" in text:
                normalized.append(text)
            else:
                normalized.append(str(Path(text).resolve()).casefold())
        return tuple(sorted(normalized))

    backend_by_source = {
        (str(record.get("asset_type") or ""), str(record.get("source_id") or "")): record
        for record in backend_records
        if isinstance(record, dict) and record.get("source_id")
    }
    for kind, collection, id_key, state_key in (
        ("character", "character_bible", "character_id", "character_ids"),
        ("scene", "scene_bible", "scene_id", "scene_ids"),
    ):
        old_items = {
            str(item.get(id_key)): item
            for item in local.get(collection, [])
            if isinstance(item, dict) and item.get(id_key)
        }
        new_items = {
            str(item.get(id_key)): item
            for item in result.get(collection, [])
            if isinstance(item, dict) and item.get(id_key)
        }
        kept: list[str] = []
        # Surface durable worker truth on every poll.  Previously a failed
        # subprocess looked like an endless pending card to Web users.
        for item_id, new in new_items.items():
            record = backend_by_source.get((kind, item_id)) or {}
            if record:
                backend_status = str(record.get("status") or "pending")
                new["asset_status"] = backend_status
                new["asset_error"] = str(record.get("error") or "")
                if backend_status.lower() not in TERMINAL_SUCCESS:
                    # Keep old files on disk for audit, but never display them
                    # as the current review candidate after a prompt change,
                    # rejection or retry.
                    new["reference_images"] = []
                    new["asset_hash"] = None
                    new["asset_manifest_path"] = None
        for item_id in local_state["assets"][state_key]:
            old = old_items.get(item_id) or {}
            new = new_items.get(item_id) or {}
            old_hash = str(old.get("asset_hash") or old.get("content_hash") or "")
            new_hash = str(new.get("asset_hash") or new.get("content_hash") or "")
            old_refs = tuple(str(path) for path in old.get("reference_images") or [])
            new_refs = tuple(str(path) for path in new.get("reference_images") or [])
            same_content = bool(new_refs) and old_refs == new_refs and (
                old_hash == new_hash if old_hash or new_hash else True
            )
            if same_content:
                kept.append(item_id)
                new["asset_review_status"] = "approved"
        if backend_gate_approved:
            for item_id, new in new_items.items():
                record = backend_by_source.get((kind, item_id)) or {}
                card_hash = str(new.get("asset_hash") or new.get("content_hash") or "")
                backend_hash = str(record.get("content_hash") or "")
                card_refs = reference_bundle(new.get("reference_images") or [])
                backend_refs = reference_bundle(record.get("reference_images") or [])
                matches_approved_backend = (
                    bool(record.get("approved"))
                    and str(record.get("status") or "").lower() in TERMINAL_SUCCESS
                    and bool(card_hash)
                    and card_hash == backend_hash
                    and bool(card_refs)
                    and card_refs == backend_refs
                )
                if matches_approved_backend and item_id not in kept:
                    kept.append(item_id)
                    new["asset_review_status"] = "approved"
        merged_state["assets"][state_key] = kept
    result["approval_state"] = merged_state
    return result


def _delivery_release_binding_sha(
    artifact_hashes: dict[str, str], selection_hashes: dict[str, str], visual_hashes: dict[str, str],
) -> str:
    return hashlib.sha256(json.dumps({
        "artifact_hashes": artifact_hashes,
        "selection_hashes": selection_hashes,
        "visual_hashes": visual_hashes,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _manifest_matches_current_release(snapshot: dict[str, Any], manifest: dict[str, Any]) -> bool:
    """A delivery file is downloadable only while its exact release set survives."""
    review = content_review_summary(snapshot)
    artifacts = {str(k): str(v) for k, v in (review.get("artifact_hashes") or {}).items()}
    selections = {str(k): str(v) for k, v in (review.get("edit_selection_hashes") or {}).items()}
    visuals: dict[str, str] = {}
    for job in normalize_jobs(snapshot.get("jobs") or []):
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        analysis = ((metadata.get("content_qa") or {}).get("analysis") or {})
        visual = str(analysis.get("decoded_visual_sha256") or "")
        if visual:
            visuals[str(job.get("job_id") or "")] = visual
    manifest_artifacts = {str(k): str(v) for k, v in (manifest.get("approved_artifact_hashes") or {}).items()}
    manifest_selections = {str(k): str(v) for k, v in (manifest.get("approved_edit_selection_hashes") or {}).items()}
    manifest_visuals = {str(k): str(v) for k, v in (manifest.get("approved_visual_hashes") or {}).items()}
    if not (
        review.get("ready_for_export")
        and manifest.get("release_status") in {"approved", "released"}
        and artifacts == manifest_artifacts
        and selections == manifest_selections
        and visuals == manifest_visuals
        and str(manifest.get("release_binding_sha256") or "")
        == _delivery_release_binding_sha(artifacts, selections, visuals)
    ):
        return False
    qa_path = Path(str(manifest.get("qa_report_path") or ""))
    try:
        return qa_path.is_file() and hashlib.sha256(qa_path.read_bytes()).hexdigest() == str(
            manifest.get("qa_report_sha256") or ""
        )
    except OSError:
        return False


def persisted_delivery_manifests(
    snapshot: dict[str, Any], roots: Iterable[Path] = (),
) -> list[dict[str, Any]]:
    """Load durable delivery manifests advertised by a project snapshot."""
    roots = [Path(root).resolve() for root in roots]

    def iter_paths(value: Any) -> Iterable[str]:
        if isinstance(value, (str, Path)):
            yield str(value)
        elif isinstance(value, dict):
            for nested in value.values():
                yield from iter_paths(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from iter_paths(nested)

    def resolve_file(value: Any, search_roots: Iterable[Path]) -> Path | None:
        if not value:
            return None
        candidate = Path(str(value))
        candidates = [candidate] if candidate.is_absolute() else [
            candidate, *(Path(root) / candidate for root in search_roots)
        ]
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved.is_file():
                return resolved
        return None

    manifest_paths: list[Path] = []
    for raw in iter_paths(snapshot.get("deliveries") or []):
        if not raw.lower().endswith(".manifest.json"):
            continue
        resolved = resolve_file(raw, roots)
        if resolved is not None:
            manifest_paths.append(resolved)
    records: list[tuple[float, dict[str, Any]]] = []
    for manifest_path in manifest_paths:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        manifest = copy.deepcopy(manifest)
        manifest["manifest_path"] = str(manifest_path.resolve())
        search_roots = [manifest_path.parent, *roots]
        for field in ("output_path", "package_path"):
            resolved = resolve_file(manifest.get(field), search_roots)
            manifest[field] = str(resolved) if resolved is not None else None
        output = Path(manifest["output_path"]) if manifest.get("output_path") else None
        if output is None or output.suffix.lower() != ".mp4":
            continue
        if _manifest_matches_current_release(snapshot, manifest):
            records.append((manifest_path.stat().st_mtime, manifest))
    records.sort(key=lambda item: item[0], reverse=True)
    return [manifest for _, manifest in records]


def reviewed_asset_hashes(
    episode: dict[str, Any], snapshot: dict[str, Any]
) -> dict[str, str]:
    """Return hashes for assets the browser actually displayed and approved.

    A hash is eligible only when the local review state approves its source,
    the displayed episode card has reference images and the displayed hash
    exactly matches the latest backend asset record.  This makes the final
    backend approval a compare-and-swap instead of a blind approval.
    """
    state = approval_state(episode)["assets"]
    approved_sources = {
        *(('character', value) for value in state["character_ids"]),
        *(('scene', value) for value in state["scene_ids"]),
    }
    displayed: dict[tuple[str, str], dict[str, Any]] = {}
    for asset_type, collection, id_key in (
        ("character", episode.get("character_bible") or [], "character_id"),
        ("scene", episode.get("scene_bible") or [], "scene_id"),
    ):
        for item in collection:
            if isinstance(item, dict) and item.get(id_key):
                displayed[(asset_type, str(item[id_key]))] = item
    assets = snapshot.get("assets") if isinstance(snapshot.get("assets"), dict) else {}
    records = assets.get("items") if isinstance(assets.get("items"), list) else []
    result: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        source = (str(record.get("asset_type") or ""), str(record.get("source_id") or ""))
        card = displayed.get(source) or {}
        backend_hash = str(record.get("content_hash") or "")
        displayed_hash = str(card.get("asset_hash") or card.get("content_hash") or "")
        if (
            source in approved_sources
            and record.get("asset_id")
            and backend_hash
            and displayed_hash == backend_hash
            and bool(card.get("reference_images"))
        ):
            result[str(record["asset_id"])] = backend_hash
    return result


def approval_state(episode: dict[str, Any]) -> dict[str, Any]:
    raw = episode.get("approval_state") if isinstance(episode.get("approval_state"), dict) else {}
    creative_raw = raw.get("creative") if isinstance(raw.get("creative"), dict) else {}
    assets_raw = raw.get("assets") if isinstance(raw.get("assets"), dict) else {}
    return {
        "creative": {key: bool(creative_raw.get(key)) for key in CREATIVE_APPROVAL_KEYS},
        "assets": {
            "character_ids": list(dict.fromkeys(str(value) for value in assets_raw.get("character_ids", []) if value)),
            "scene_ids": list(dict.fromkeys(str(value) for value in assets_raw.get("scene_ids", []) if value)),
        },
    }


def with_creative_approval(episode: dict[str, Any], section: str, approved: bool) -> dict[str, Any]:
    if section not in CREATIVE_APPROVAL_KEYS:
        raise ValueError(f"unknown creative approval section: {section}")
    updated = copy.deepcopy(episode)
    state = approval_state(updated)
    state["creative"][section] = bool(approved)
    if not approved:
        state["assets"] = {"character_ids": [], "scene_ids": []}
    updated["approval_state"] = state
    return updated


def with_asset_approval(episode: dict[str, Any], kind: str, item_id: str, approved: bool) -> dict[str, Any]:
    key = {"character": "character_ids", "scene": "scene_ids"}.get(kind)
    if not key:
        raise ValueError("asset kind must be character or scene")
    updated = copy.deepcopy(episode)
    state = approval_state(updated)
    values = list(state["assets"][key])
    if approved and item_id not in values:
        values.append(item_id)
    elif not approved:
        values = [value for value in values if value != item_id]
    state["assets"][key] = values
    updated["approval_state"] = state
    collection = "character_bible" if kind == "character" else "scene_bible"
    id_key = "character_id" if kind == "character" else "scene_id"
    for item in updated.get(collection, []):
        if str(item.get(id_key)) == item_id:
            item["asset_review_status"] = "approved" if approved else "ready_for_review"
            break
    return updated


def with_asset_review_status(
    episode: dict[str, Any],
    kind: str,
    item_id: str,
    status: str,
    *,
    reason: str = "",
) -> dict[str, Any]:
    """Mark a single asset rejected/regenerating and always revoke approval."""
    if status not in {"rejected", "regenerating", "ready_for_review"}:
        raise ValueError("asset review status must be rejected, regenerating or ready_for_review")
    updated = with_asset_approval(episode, kind, item_id, False)
    collection = "character_bible" if kind == "character" else "scene_bible"
    id_key = "character_id" if kind == "character" else "scene_id"
    found = False
    for item in updated.get(collection, []):
        if str(item.get(id_key)) == item_id:
            item["asset_review_status"] = status
            item["asset_rejection_reason"] = reason.strip() if status == "rejected" else ""
            found = True
            break
    if not found:
        raise KeyError(f"unknown {kind} asset: {item_id}")
    return updated


def creative_gate_ready(episode: dict[str, Any]) -> bool:
    state = approval_state(episode)
    return bool(episode) and all(state["creative"].values())


def asset_readiness(episode: dict[str, Any]) -> dict[str, dict[str, bool]]:
    def ready(item: dict[str, Any]) -> bool:
        if str(item.get("asset_review_status") or "").lower() in {"rejected", "regenerating"}:
            return False
        worker_status = str(item.get("asset_status") or "").lower()
        if worker_status and worker_status not in {"succeeded", "success", "completed", "ready_for_approval"}:
            return False
        return bool(item.get("reference_images"))

    return {
        "characters": {
            str(item.get("character_id")): ready(item)
            for item in episode.get("character_bible", []) if isinstance(item, dict) and item.get("character_id")
        },
        "scenes": {
            str(item.get("scene_id")): ready(item)
            for item in episode.get("scene_bible", []) if isinstance(item, dict) and item.get("scene_id")
        },
    }


def asset_gate_ready(episode: dict[str, Any]) -> bool:
    if not creative_gate_ready(episode):
        return False
    state = approval_state(episode)
    readiness = asset_readiness(episode)
    if not readiness["characters"] or not readiness["scenes"]:
        return False
    return (
        all(readiness["characters"].values())
        and all(readiness["scenes"].values())
        and set(state["assets"]["character_ids"]) == set(readiness["characters"])
        and set(state["assets"]["scene_ids"]) == set(readiness["scenes"])
    )


def derive_project_stage(
    episode: dict[str, Any],
    snapshot: dict[str, Any] | None = None,
    *,
    contract_persisted: bool = False,
    production_started: bool = False,
) -> str:
    snapshot = snapshot or {}
    if not episode:
        return "creative_brief"
    if not creative_gate_ready(episode):
        return "creative_review"
    if not contract_persisted:
        return "creative_approved"
    readiness = asset_readiness(episode)
    if not readiness["characters"] or not all(readiness["characters"].values()):
        return "character_assets"
    if not readiness["scenes"] or not all(readiness["scenes"].values()):
        return "scene_assets"
    if not asset_gate_ready(episode):
        return "asset_review"
    jobs = normalize_jobs(snapshot.get("jobs") or [])
    counts = job_counts(jobs)
    if snapshot.get("deliveries"):
        return "delivered" if content_review_summary(snapshot)["ready_for_export"] else "release_revoked"
    if jobs and counts["success"] == counts["total"]:
        review = content_review_summary(snapshot)
        if review["automated_qa_passed"] != review["total"]:
            return "content_qa"
        if review["human_approved"] != review["total"]:
            return "content_review"
        if not review["release_approved"]:
            return "release_review"
        return "ready_to_export"
    if production_started or jobs:
        return "video_production"
    return "ready_for_video"


def attach_reference_images(
    episode: dict[str, Any],
    character_id: str,
    paths: Iterable[str],
) -> dict[str, Any]:
    """Return a copy with persisted reference paths on the matching character."""
    updated = copy.deepcopy(episode)
    for character in updated.get("character_bible", []):
        if character.get("character_id") == character_id:
            existing = list(character.get("reference_images") or [])
            for path in paths:
                value = str(path)
                if value and value not in existing:
                    existing.append(value)
            character["reference_images"] = existing
            break
    return updated


def attach_scene_reference_images(
    episode: dict[str, Any],
    scene_id: str,
    paths: Iterable[str],
) -> dict[str, Any]:
    """Attach scene assets and inject them into every panel using that scene."""
    updated = copy.deepcopy(episode)
    additions = [str(path).strip() for path in paths if str(path).strip()]
    scene_refs: list[str] = []
    for scene in updated.get("scene_bible", []):
        if scene.get("scene_id") == scene_id:
            scene_refs = list(scene.get("reference_images") or [])
            for value in additions:
                if value not in scene_refs:
                    scene_refs.append(value)
            scene["reference_images"] = scene_refs
            break
    for panel in updated.get("panels", []):
        if panel.get("scene_id") != scene_id:
            continue
        references = list(panel.get("reference_images") or [])
        for value in scene_refs:
            if value not in references:
                references.append(value)
        panel["reference_images"] = references
    return updated


def iter_media_paths(value: Any) -> Iterable[str]:
    """Yield image/video paths from nested task snapshots without guessing keys."""
    if isinstance(value, str):
        suffix = Path(value).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov"}:
            yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_media_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_media_paths(item)


def existing_media_paths(value: Any, roots: Iterable[Path] = ()) -> list[Path]:
    roots = [Path(root) for root in roots]
    result: list[Path] = []
    seen: set[str] = set()
    for raw in iter_media_paths(value):
        candidate = Path(raw)
        candidates = [candidate] if candidate.is_absolute() else [candidate, *[root / candidate for root in roots]]
        for path in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            key = str(resolved).casefold()
            if resolved.exists() and resolved.is_file() and key not in seen:
                result.append(resolved)
                seen.add(key)
                break
    return result


def runtime_prompt_audit(
    job: dict[str, Any], roots: Iterable[Path],
) -> dict[str, Any]:
    """Load the immutable H3 graph snapshot for human-facing prompt review.

    Only project-owned roots are readable.  The returned reference rows expose
    role, model label, basename and content hash, never arbitrary file content.
    """
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    settings = metadata.get("settings") if isinstance(metadata.get("settings"), dict) else {}
    expected_prompt_sha = str(
        metadata.get("prompt_sha256") or settings.get("prompt_sha256") or ""
    )
    expected_reference_sha = str(
        metadata.get("reference_bundle_sha256")
        or settings.get("reference_bundle_sha256") or ""
    )
    result = {
        "available": False,
        "error": "",
        "prompt": "",
        "prompt_sha256": expected_prompt_sha,
        "prompt_hash_matches": False,
        "reference_bundle_sha256": expected_reference_sha,
        "director_skill_version": str(
            metadata.get("director_skill_version")
            or settings.get("director_skill_version") or ""
        ),
        "official_prompt_shape": str(settings.get("official_prompt_shape") or ""),
        "runtime_prompt_contract": str(settings.get("runtime_prompt_contract") or ""),
        "references": [],
    }
    raw_graph_path = str(job.get("graph_path") or "").strip()
    if not raw_graph_path:
        result["error"] = "该任务尚无不可变 graph 快照。"
        return result
    try:
        graph_path = Path(raw_graph_path).resolve()
        allowed_roots = [Path(root).resolve() for root in roots]
    except OSError as exc:
        result["error"] = f"graph 路径不可解析：{exc}"
        return result
    if not allowed_roots or not any(graph_path.is_relative_to(root) for root in allowed_roots):
        result["error"] = "graph 快照不在当前项目允许目录内，已拒绝读取。"
        return result
    if not graph_path.is_file():
        result["error"] = "graph 快照尚未落盘。"
        return result
    try:
        snapshot = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result["error"] = f"graph 快照不可读：{exc}"
        return result
    prompt = str(snapshot.get("prompt") or "")
    if not prompt:
        result["error"] = "graph 快照缺少最终 H3 prompt。"
        return result
    actual_prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    prompt_audit = snapshot.get("prompt_audit") if isinstance(snapshot.get("prompt_audit"), dict) else {}
    snapshot_expected_sha = str(prompt_audit.get("prompt_sha256") or expected_prompt_sha)
    snapshot_settings = snapshot.get("settings") if isinstance(snapshot.get("settings"), dict) else {}
    bindings = snapshot_settings.get("reference_bindings") or []
    references = snapshot.get("reference_images") or []
    rows: list[dict[str, str]] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            continue
        binding = bindings[index] if index < len(bindings) and isinstance(bindings[index], dict) else {}
        source = str(reference.get("source_path") or reference.get("filename") or "")
        rows.append({
            "model_label": str(binding.get("model_label") or f"<Picture {index + 1}>"),
            "role": str(reference.get("role") or binding.get("role") or "reference"),
            "file": Path(source).name,
            "sha256": str(reference.get("sha256") or ""),
        })
    result.update({
        "available": True,
        "prompt": prompt,
        "prompt_sha256": actual_prompt_sha,
        "prompt_hash_matches": bool(snapshot_expected_sha and actual_prompt_sha == snapshot_expected_sha),
        "reference_bundle_sha256": str(
            prompt_audit.get("reference_bundle_sha256") or expected_reference_sha
        ),
        "director_skill_version": str(
            prompt_audit.get("skill_version")
            or result["director_skill_version"]
        ),
        "official_prompt_shape": str(
            prompt_audit.get("official_prompt_shape")
            or result["official_prompt_shape"]
        ),
        "runtime_prompt_contract": str(
            prompt_audit.get("runtime_prompt_contract")
            or result["runtime_prompt_contract"]
        ),
        "references": rows,
    })
    if not result["prompt_hash_matches"]:
        result["error"] = "最终 prompt 哈希与任务快照不一致，禁止批准。"
    return result


def prioritize_preview_media(paths: Iterable[Path]) -> list[Path]:
    """Put playable videos before reference images without losing stable order."""
    unique: list[Path] = []
    seen: set[str] = set()
    for value in paths:
        path = Path(value)
        key = str(path.resolve()).casefold()
        if key not in seen:
            unique.append(path)
            seen.add(key)
    video_suffixes = {".mp4", ".webm", ".mov"}
    videos = [path for path in unique if path.suffix.lower() in video_suffixes]
    images = [path for path in unique if path.suffix.lower() not in video_suffixes]
    return videos[:1] + images


def job_media_for_review(
    job: dict[str, Any], roots: Iterable[Path] = ()
) -> dict[str, Any]:
    """Separate current job output from QA-rejected archival media.

    QA audit metadata intentionally retains rejected clips.  It must never be
    traversed as an ordinary preview source, especially while the job is
    failed, cancelled, queued or running after a retry.
    """
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    rejection_audit = metadata.get("qa_rejection_audit") or []
    invalidation_audit = metadata.get("qa_invalidation_audit") or []
    if not isinstance(rejection_audit, list):
        rejection_audit = []
    if not isinstance(invalidation_audit, list):
        invalidation_audit = []
    error = str(job.get("error") or "").lower()
    qa_invalidated = bool(
        rejection_audit
        or invalidation_audit
        or "qa rejected" in error
        or "qa rejection" in error
        or "rejected predecessor" in error
    )
    status = str(job.get("status") or "pending").lower()
    current_source = {
        key: job.get(key)
        for key in ("output_path", "preview_path", "comfy_output_path", "outputs")
        if job.get(key)
    }
    current = existing_media_paths(current_source, roots)
    if qa_invalidated and status not in TERMINAL_SUCCESS:
        current = []

    audit_sources: list[Any] = []
    for audit in [*rejection_audit, *invalidation_audit]:
        if isinstance(audit, dict) and isinstance(audit.get("archived_files"), dict):
            audit_sources.append(audit["archived_files"])
    audit = existing_media_paths(audit_sources, roots)
    return {
        "qa_invalidated": qa_invalidated,
        "current": prioritize_preview_media(current),
        "audit": prioritize_preview_media(audit),
    }


def earliest_qa_rejected_failed_job(
    jobs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return the earliest failed QA rejection as the strict-chain safe root."""
    candidates: list[dict[str, Any]] = []
    for job in jobs:
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        error = str(job.get("error") or "").lower()
        if (
            str(job.get("status") or "").lower() in {"failed", "error"}
            and (metadata.get("qa_rejection_audit") or "qa rejected" in error)
        ):
            candidates.append(job)
    if not candidates:
        return {}
    return min(
        candidates,
        key=lambda item: (
            int(item.get("panel_index") or 10**9),
            str(item.get("job_id") or ""),
        ),
    )


def continuity_anchor_candidates(
    job: dict[str, Any],
    episode: dict[str, Any],
    jobs: Iterable[dict[str, Any]],
    roots: Iterable[Path] = (),
) -> list[dict[str, str]]:
    """Find existing image anchors in production-authority priority order."""
    roots = [Path(root).resolve() for root in roots]
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    inputs = metadata.get("inputs") if isinstance(metadata.get("inputs"), dict) else {}
    dependency = (
        inputs.get("continuity_dependency")
        if isinstance(inputs.get("continuity_dependency"), dict)
        else {}
    )
    panel_name = str(job.get("panel_id") or job.get("panel_name") or "")
    panel = next((
        item for item in episode.get("panels") or []
        if isinstance(item, dict)
        and panel_name in {str(item.get("panel_id") or ""), str(item.get("name") or "")}
    ), {})
    by_job_id = {
        str(item.get("job_id")): item for item in jobs
        if isinstance(item, dict) and item.get("job_id")
    }
    previous = by_job_id.get(str(dependency.get("previous_job_id") or ""), {})
    previous_metadata = (
        previous.get("metadata") if isinstance(previous.get("metadata"), dict) else {}
    )
    previous_inputs = (
        previous_metadata.get("inputs")
        if isinstance(previous_metadata.get("inputs"), dict)
        else {}
    )
    previous_panel_name = str(previous.get("panel_id") or previous.get("panel_name") or "")
    previous_panel = next((
        item for item in episode.get("panels") or []
        if isinstance(item, dict)
        and previous_panel_name in {str(item.get("panel_id") or ""), str(item.get("name") or "")}
    ), {})

    def values(source: dict[str, Any]) -> list[Any]:
        keys = (
            "continuity_tail_path", "tail_frame_path", "previous_tail_path",
            "continuity_anchor_path", "source_anchor", "first_frame_path", "last_frame_path",
        )
        return [source.get(key) for key in keys if source.get(key)]

    sources: list[tuple[str, str, list[Any]]] = [
        ("continuity_tail", "任务连续尾帧", [*values(metadata), *values(inputs), *values(dependency)]),
        ("panel_last_frame", "本镜批准末帧", [panel.get("last_frame_path")]),
    ]
    approved_group_anchor = (
        metadata.get("approved_group_anchor")
        if isinstance(metadata.get("approved_group_anchor"), dict) else {}
    )
    if approved_group_anchor.get("status") == "approved":
        sources.insert(0, (
            "approved_group_anchor_final",
            "已批准 H3 终态构图锚",
            [approved_group_anchor.get("last_path")],
        ))
        sources.insert(1, (
            "approved_group_anchor_first",
            "已批准 H3 首态构图锚",
            [approved_group_anchor.get("path")],
        ))
    if str(previous.get("status") or "").lower() in TERMINAL_SUCCESS:
        sources.append((
            "previous_succeeded_tail",
            "前序成功镜头尾帧",
            [*values(previous_metadata), *values(previous_inputs), previous_panel.get("last_frame_path")],
        ))
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, label, raw_values in sources:
        for path in existing_media_paths(raw_values, roots):
            if path.suffix.lower() not in image_suffixes:
                continue
            if roots:
                try:
                    path.resolve().relative_to(roots[0])
                except ValueError:
                    # The backend binds approvals only to files inside this
                    # episode project. Do not advertise an unusable anchor.
                    continue
            key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append({"source": source, "label": label, "path": str(path)})
    return result


def start_continuity_safe_via_facade(
    service: Any,
    ep_id: str,
    job_id: str,
    source_anchor: str,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Approve the exact anchor, then launch the explicit safe mode."""
    approve = getattr(service, "approve_continuity_anchor", None)
    start = getattr(service, "start_continuity_safe", None)
    if not callable(approve):
        raise RuntimeError("render_service.approve_continuity_anchor public facade is unavailable")
    if not callable(start):
        raise RuntimeError("render_service.start_continuity_safe public facade is unavailable")
    approval = approve(ep_id, job_id, source_anchor, reason=reason)
    launch = start(
        ep_id,
        job_id,
        preferred_voice="Microsoft Huihui Desktop",
        motion="slow_push",
        # Panel previews keep deterministic subtitle sidecars.  Burn exactly
        # once during platform delivery so resizing never creates duplicate
        # glyphs or mismatched safe margins.
        burn_subtitles=False,
    )
    return approval, launch
