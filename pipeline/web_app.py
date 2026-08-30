# -*- coding: utf-8 -*-
"""Streamlit creative approval and production console.

The page owns user intent and review state. Durable registration, background
workers and delivery are invoked only through public service facades.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

st.set_page_config(page_title="AI 漫剧工厂", page_icon="🎬", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE))

from runtime_config import comfyui_root, projects_dir  # noqa: E402
from h3_profiles import (  # noqa: E402
    DIRECT_PRODUCTION,
    PROOF_THEN_PRODUCTION,
    profile_cost_summary,
)
from story_splitter import (  # noqa: E402
    AMBIENCE_PRESETS,
    MUSIC_PRESETS,
    MissingMiniMaxAPIKey,
    MiniMaxRequestTimeout,
    generate_series_episode,
    regenerate_contract_item,
    resume_story_stage2,
    split_story_checkpoint_inputs,
    split_series,
    split_story,
    minimax_configuration_status,
    minimax_request_timeout_seconds,
    update_series_outline_episode,
    update_contract_item,
    validate_episode_contract,
)
from prompt_contracts import (  # noqa: E402
    MODERN_URBAN_STYLE_PROMPT,
    auto_episode_shot_count,
    repair_episode_character_references,
    shot_count_bounds,
    shot_plan_cost_summary,
    validate_series_contract,
)
from ui_helpers import (  # noqa: E402
    approval_state,
    approve_job_review_via_facade,
    approve_preview_and_promote_via_facade,
    approve_release_via_facade,
    asset_gate_ready,
    asset_readiness,
    attach_scene_reference_images,
    continuity_anchor_candidates,
    classify_shot_worklist,
    content_review_summary,
    creative_gate_ready,
    derive_project_stage,
    earliest_qa_rejected_failed_job,
    existing_media_paths,
    job_counts,
    job_media_for_review,
    job_review_evidence,
    generation_wait_notice,
    generation_input_signature,
    merge_series_backend_assets,
    normalize_jobs,
    prioritize_preview_media,
    merge_episode_asset_review_state,
    prepare_series_via_facade,
    persisted_delivery_manifests,
    register_series_episodes_via_facade,
    reviewed_asset_hashes,
    runtime_prompt_audit,
    series_episode_counts,
    series_from_service_snapshot,
    snapshot_episode,
    shot_readiness_rows,
    start_continuity_safe_via_facade,
    stage2_resume_eligibility,
    resume_stage2_via_facade,
    updated_incomplete_stage2_checkpoint,
    with_asset_approval,
    with_asset_review_status,
    with_creative_approval,
    with_series_episode_approval,
)
from generation_drafts import list_stage1_checkpoints, match_stage1_checkpoint  # noqa: E402
from shot_group_anchor import requires_approved_group_anchor, requires_paired_state_anchor  # noqa: E402

PROJECTS_DIR = projects_dir()
COMFYUI_ROOT = comfyui_root()
COMFYUI_INPUT = Path(os.environ.get("COMFYUI_INPUT_DIR", COMFYUI_ROOT / "input")).resolve()
COMFYUI_OUTPUT = Path(os.environ.get("COMFYUI_OUTPUT_DIR", COMFYUI_ROOT / "output")).resolve()

try:
    import render_service  # type: ignore  # noqa: E402
    if not all(hasattr(render_service, name) for name in (
        "select_asset_references", "classify_job_rejection", "authorize_additional_job_retry",
    )):
        render_service = importlib.reload(render_service)
    RENDER_SERVICE_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    render_service = None
    RENDER_SERVICE_ERROR = str(exc)

try:
    import series_service  # type: ignore  # noqa: E402
    SERIES_SERVICE_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    series_service = None
    SERIES_SERVICE_ERROR = str(exc)

STYLE_PRESETS = {
    "日系动画": "premium Japanese 2D animation, clean line art, painterly backgrounds, expressive acting",
    "现代都市": MODERN_URBAN_STYLE_PROMPT,
    "武侠仙侠": "Chinese wuxia/xianxia animation, elegant ink texture, jade-ivory-cinnabar palette",
    "科幻赛博": "premium cyberpunk science-fiction animation, controlled neon palette, volumetric atmosphere",
    "美式漫画": "premium American comic-book animation, bold ink, halftone texture, graphic shadows",
}
PLATFORM_DEFAULTS = {
    "抖音 / TikTok": "9:16",
    "小红书 / Reels": "9:16",
    "Bilibili / YouTube": "16:9",
    "方形信息流": "1:1",
}
LANGUAGE_MAP = {"中文": ("cn", "Chinese"), "English": ("en", "English"), "日本語": ("jp", "Japanese")}
SAGE_MODE_MAP = {
    "自动": "auto",
    "关闭": "disabled",
    "Sage 2": "sageattn_qk_int8_pv_fp16_cuda",
    "Sage 3": "sageattn3",
}
REF_SIZE_MAP = {"快速（match）": "match", "身份优先（max，更慢）": "max"}
PRODUCTION_STRATEGY_MAP = {
    "低成本预演 → 人工批准 → 正式生产（推荐）": PROOF_THEN_PRODUCTION,
    "直接正式生产（跳过预演，不推荐）": DIRECT_PRODUCTION,
}
DELIVERY_PRESET_OPTIONS = {
    "竖屏720p｜抖音（720×1280）": "douyin",
    "竖屏720p｜TikTok（720×1280）": "tiktok",
    "竖屏720p｜YouTube Shorts（720×1280）": "youtube_shorts",
    "横屏720p｜Bilibili（1280×720）": "bilibili",
    "横屏720p｜YouTube（1280×720）": "youtube",
    "方形720p（720×720）": "square_1_1",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov"}
STAGE_LABELS = {
    "creative_brief": "① 填写创作简报",
    "creative_review": "② 审核故事 / 人物 / 分镜",
    "creative_approved": "③ 创作已批准，待注册",
    "character_assets": "④ 生成人物资产",
    "scene_assets": "⑤ 补齐场景资产",
    "asset_review": "⑥ 审核人物 / 场景资产",
    "ready_for_video": "⑦ 可启动视频生产",
    "video_production": "⑧ 视频生产中",
    "content_qa": "⑨ 技术生成完成，自动内容 QA 未通过（禁止导出）",
    "content_review": "⑨ 自动 QA 已通过，待逐镜人工批准（禁止导出）",
    "release_review": "⑨ 逐镜已批准，待整集发布批准（禁止导出）",
    "release_revoked": "⛔ 历史技术导出，发布资格缺失或已撤销",
    "ready_to_export": "⑨ 内容验收通过，可导出",
    "delivered": "⑩ 已交付",
}


def _init_state() -> None:
    defaults = {
        "episode": {},
        "series_contract": {},
        "last_series_snapshot": {},
        "loaded_series_id": "",
        "ep_id": f"ep_{int(time.time())}",
        "project_id_input": "",
        "pending_project_id": "",
        "loaded_ep_id": "",
        "last_snapshot": {},
        "contract_persisted": False,
        "contract_approved": False,
        "assets_approved": False,
        "production_started": False,
        "local_dirty": False,
        "flash_success": "",
        "stage2_resume_checkpoint": {},
        "stage2_resume_ep_id": "",
        "stage2_resume_input_signature": "",
        "stage2_resume_flash": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if not st.session_state.get("project_id_input"):
        st.session_state["project_id_input"] = st.session_state["ep_id"]


def _service(name: str):
    handler = getattr(render_service, name, None) if render_service is not None else None
    return handler if callable(handler) else None


def _series_service(name: str):
    handler = getattr(series_service, name, None) if series_service is not None else None
    return handler if callable(handler) else None


def _series_snapshot(series_id: str) -> dict[str, Any]:
    handler = _series_service("status_series")
    if handler is None:
        return {}
    try:
        return handler(series_id)
    except (KeyError, ValueError):
        return {}


def _persist_series_contract(series: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if series_service is None or _series_service("prepare_series") is None:
        raise RuntimeError("series_service.prepare_series 公共接口不可用")
    series_id = str((series.get("series_bible") or {}).get("series_id") or "").strip()
    if not series_id:
        raise ValueError("V4 series_bible.series_id 缺失")
    current = _series_snapshot(series_id)
    restored, snapshot = prepare_series_via_facade(series_service, series, current)
    st.session_state["last_series_snapshot"] = snapshot
    st.session_state["series_contract"] = restored
    return restored, snapshot


def _snapshot(ep_id: str) -> dict[str, Any]:
    handler = _service("project_snapshot") or _service("status")
    return handler(ep_id) if handler else {}


def _call_service(handler, ep_id: str, episode: dict[str, Any] | None = None):
    parameters = inspect.signature(handler).parameters
    kwargs: dict[str, Any] = {}
    if episode is not None and "episode" in parameters:
        kwargs["episode"] = episode
    if episode is not None and "episode_data" in parameters:
        kwargs["episode_data"] = episode
    if "ep_id" in parameters:
        return handler(ep_id=ep_id, **kwargs)
    return handler(ep_id, **kwargs)


def _call_asset_service(
    handler, ep_id: str, asset_id: str, *, reason: str | None = None,
):
    """Call the public single-asset facade without assuming positional style."""
    parameters = inspect.signature(handler).parameters
    kwargs: dict[str, Any] = {}
    if "ep_id" in parameters:
        kwargs["ep_id"] = ep_id
    if "asset_id" in parameters:
        kwargs["asset_id"] = asset_id
    if reason is not None and "reason" in parameters:
        kwargs["reason"] = reason
    if "ep_id" in parameters and "asset_id" in parameters:
        return handler(**kwargs)
    if reason is not None and "reason" in parameters:
        return handler(ep_id, asset_id, reason=reason)
    return handler(ep_id, asset_id)


def _set_episode(episode: dict[str, Any], *, persisted: bool, dirty: bool = True) -> None:
    st.session_state["episode"] = episode
    st.session_state["contract_persisted"] = persisted
    st.session_state["local_dirty"] = dirty
    if not persisted:
        st.session_state["production_started"] = False
        st.session_state["contract_approved"] = False
        st.session_state["assets_approved"] = False


def _persist(ep_id: str, episode: dict[str, Any]) -> dict[str, Any]:
    handler = _service("prepare_contract") or _service("prepare_episode")
    if handler is None:
        raise RuntimeError("render_service.prepare_contract 公共接口不可用")
    return _call_service(handler, ep_id, episode)


def _render_media(paths: list[Path], *, max_items: int = 8) -> None:
    for path in prioritize_preview_media(paths)[:max_items]:
        if path.suffix.lower() in IMAGE_SUFFIXES:
            st.image(str(path), caption=path.name, use_container_width=True)
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            st.video(str(path))
            st.caption(path.name)


def _minimax_available(api_key: str) -> bool:
    """Use an environment credential without ever echoing it into the page."""
    return bool((api_key or "").strip() or os.environ.get("MiniMax_API_KEY", "").strip())


def _asset_sync_signature(episode: dict[str, Any]) -> str:
    payload: list[dict[str, Any]] = []
    for collection, id_key in (("character_bible", "character_id"), ("scene_bible", "scene_id")):
        for item in episode.get(collection) or []:
            if not isinstance(item, dict):
                continue
            payload.append({
                "id": str(item.get(id_key) or ""),
                "status": str(item.get("asset_status") or ""),
                "hash": str(item.get("asset_hash") or item.get("content_hash") or ""),
                "refs": [str(path) for path in item.get("reference_images") or []],
            })
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _item_editor(
    episode: dict[str, Any],
    item_type: str,
    item_id: str,
    item: dict[str, Any],
    api_key: str,
) -> None:
    label = {"story": "故事", "visual": "视觉风格", "character": "人物", "scene": "场景", "panel": "分镜"}[item_type]
    runtime_only_fields = {
        "reference_images", "asset_status", "asset_hash", "content_hash",
        "asset_manifest", "asset_manifest_path", "asset_approval",
        "asset_review_status", "asset_rejection_history", "asset_error",
        "approved", "approved_at", "prompt_id", "error", "jobs", "pipeline",
        "deliveries", "output_path", "preview_path", "qa", "review",
    }
    editable_item = {
        key: value for key, value in item.items() if key not in runtime_only_fields
    }
    with st.expander(f"高级：JSON 编辑 / 重生{label}", expanded=False):
        edited = st.text_area(
            "高级 JSON 编辑",
            value=json.dumps(editable_item, ensure_ascii=False, indent=2),
            height=260,
            key=f"edit_{item_type}_{item_id or 'root'}",
        )
        instruction = st.text_input(
            "单项重生要求",
            placeholder="例如：保留身份与情节，只加强冲突和镜头可执行性",
            key=f"regen_note_{item_type}_{item_id or 'root'}",
        )
        edit_col, regen_col = st.columns(2)
        if edit_col.button("应用编辑", key=f"apply_{item_type}_{item_id or 'root'}", use_container_width=True):
            try:
                replacement = json.loads(edited)
                updated = update_contract_item(episode, item_type, item_id, replacement)
                _set_episode(updated, persisted=False)
                st.rerun()
            except Exception as exc:
                st.error(f"编辑未应用：{exc}")
        regen_label = {
            "character": "MiniMax 参考图提示词大师重生",
            "scene": "MiniMax 场景提示词大师重生",
            "panel": "MiniMax H3 提示词大师重生本镜",
        }.get(item_type, "MiniMax 重生单项")
        if regen_col.button(
            regen_label,
            key=f"regen_{item_type}_{item_id or 'root'}",
            disabled=item_type == "visual" or not _minimax_available(api_key),
            use_container_width=True,
        ):
            try:
                with st.spinner(f"正在重生{label}，其他已批准事实保持不变…"):
                    updated = regenerate_contract_item(
                        episode, item_type, item_id, instruction, api_key=api_key or None
                    )
                _set_episode(updated, persisted=False)
                st.rerun()
            except Exception as exc:
                st.error(f"单项重生失败：{exc}")


def _approval_button(episode: dict[str, Any], section: str, label: str) -> None:
    approved = approval_state(episode)["creative"][section]
    text = f"撤回{label}批准" if approved else f"批准{label}"
    if st.button(text, key=f"approve_{section}", type="secondary", use_container_width=True):
        updated = with_creative_approval(episode, section, not approved)
        _set_episode(updated, persisted=False)
        st.rerun()
    st.caption("✅ 已批准" if approved else "⏳ 待批准")


def _show_story_review(episode: dict[str, Any], api_key: str) -> None:
    story = episode.get("story_bible") or {}
    st.subheader("故事圣经")
    st.markdown(f"### {story.get('title') or episode.get('title') or '未命名'}")
    st.write(story.get("logline") or "")
    st.write(story.get("synopsis") or "")
    meta = st.columns(3)
    meta[0].metric("类型", story.get("genre") or "-")
    meta[1].metric("受众", story.get("target_audience") or "-")
    meta[2].metric("镜数", len(episode.get("panels") or []))
    _item_editor(episode, "story", "", story, api_key)
    visual = episode.get("visual_bible") or {}
    with st.expander("视觉风格圣经 / 模型全局提示词", expanded=False):
        st.caption("以下两项会实际进入人物、场景与 H3 提示词；必须是可打印 ASCII 英文。")
        st.code(visual.get("style_prompt") or "", language="text")
        st.code(visual.get("global_negative_prompt") or "", language="text")
        _item_editor(episode, "visual", "", visual, api_key)
    _approval_button(episode, "story", "故事")


def _show_character_cards(episode: dict[str, Any], snapshot: dict[str, Any], api_key: str) -> None:
    st.subheader("人物卡 · Character Bible")
    characters = episode.get("character_bible") or []
    if not characters:
        st.error("人物圣经为空，不能进入生产。")
        return
    columns = st.columns(min(3, len(characters)))
    for index, character in enumerate(characters):
        char_id = str(character.get("character_id"))
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(f"### {character.get('name') or char_id}")
                st.caption(f"`{char_id}` · {character.get('role', '')}")
                refs = existing_media_paths(
                    character.get("reference_images") or [],
                    [PROJECTS_DIR, COMFYUI_INPUT, COMFYUI_OUTPUT],
                )
                _render_media(refs, max_items=4)
                if not refs:
                    st.warning("尚无人物参考资产")
                asset_status = character.get("asset_review_status") or character.get("asset_status") or "pending"
                st.caption(f"资产状态：{asset_status}")
                st.write(character.get("story_function") or "")
                with st.expander("实际人物 Prompt / 声音", expanded=False):
                    st.markdown("**审阅描述**")
                    st.write(character.get("editorial_identity_description") or character.get("identity_prompt") or "")
                    st.write(character.get("editorial_wardrobe_description") or character.get("wardrobe_prompt") or "")
                    st.markdown("**模型英文标签（实际用于资产/H3）**")
                    st.code(", ".join(character.get("model_identity_tags_en") or []), language="text")
                    st.code(", ".join(character.get("model_wardrobe_tags_en") or []), language="text")
                    st.code(character.get("negative_prompt") or "", language="text")
                    for warning in character.get("model_prompt_warnings") or []:
                        st.warning(warning)
                    st.json(character.get("voice_profile") or {}, expanded=False)
                _item_editor(episode, "character", char_id, character, api_key)
    _approval_button(episode, "characters", "人物")


def _show_scene_cards(episode: dict[str, Any], api_key: str) -> None:
    st.subheader("场景卡 · Scene Bible")
    scenes = episode.get("scene_bible") or []
    if not scenes:
        st.error("场景圣经为空，不能进入生产。")
        return
    columns = st.columns(min(3, len(scenes)))
    for index, scene in enumerate(scenes):
        scene_id = str(scene.get("scene_id"))
        with columns[index % len(columns)]:
            with st.container(border=True):
                st.markdown(f"### {scene.get('name') or scene_id}")
                st.caption(f"`{scene_id}`")
                refs = existing_media_paths(
                    scene.get("reference_images") or [],
                    [PROJECTS_DIR, COMFYUI_INPUT, COMFYUI_OUTPUT],
                )
                _render_media(refs, max_items=3)
                if not refs:
                    st.warning("尚无场景参考资产")
                asset_status = scene.get("asset_review_status") or scene.get("asset_status") or "pending"
                st.caption(f"资产状态：{asset_status}")
                st.write(scene.get("description") or "")
                with st.expander("实际场景 Prompt", expanded=False):
                    st.code(scene.get("model_prompt_en") or scene.get("asset_prompt") or "", language="text")
                    st.code(scene.get("negative_prompt") or "", language="text")
                    for warning in scene.get("model_prompt_warnings") or []:
                        st.warning(warning)
                path_text = st.text_input(
                    "选择已有场景参考图路径",
                    key=f"scene_ref_{scene_id}",
                    placeholder="多个路径用换行或 ; 分隔",
                )
                select_handler = _service("select_asset_references")
                if st.button(
                    "使用这些图片替换当前场景资产",
                    key=f"attach_scene_{scene_id}",
                    disabled=(not path_text.strip() or select_handler is None or st.session_state.get("local_dirty")),
                ):
                    paths = [value.strip() for value in path_text.replace(";", "\n").splitlines() if value.strip()]
                    try:
                        selected = select_handler(
                            ep_id=str(episode.get("ep_id") or st.session_state.get("project_id") or ""),
                            asset_id=f"{episode.get('ep_id')}:scene:{scene_id}",
                            reference_images=paths,
                        )
                        _set_episode(snapshot_episode(selected, episode), persisted=True, dirty=False)
                        st.session_state["flash_success"] = "场景资产已替换；创作合同保持不变，请重新审核该场景。"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"场景资产替换失败：{exc}")
                if select_handler is None:
                    st.caption("当前公共服务没有 select_asset_references，不能安全替换已注册场景。")
                _item_editor(episode, "scene", scene_id, scene, api_key)


def _job_for_panel(snapshot: dict[str, Any], panel_id: str) -> dict[str, Any]:
    for job in normalize_jobs(snapshot.get("jobs") or []):
        if panel_id in {str(job.get("panel_id")), str(job.get("panel_name"))}:
            return job
    return {}


def _show_storyboard(episode: dict[str, Any], snapshot: dict[str, Any], api_key: str) -> None:
    st.subheader("逐镜分镜、连续性与实际 Prompt")
    warnings = validate_episode_contract(episode)
    repaired_preview = repair_episode_character_references(episode)
    current_character_refs = [
        (
            str(panel.get("panel_id") or panel.get("name") or ""),
            tuple(str(value) for value in panel.get("character_ids") or []),
            tuple(str(value) for value in (panel.get("prompt_package") or {}).get("character_ids") or []),
        )
        for panel in episode.get("panels") or []
    ]
    repaired_character_refs = [
        (
            str(panel.get("panel_id") or panel.get("name") or ""),
            tuple(str(value) for value in panel.get("character_ids") or []),
            tuple(str(value) for value in (panel.get("prompt_package") or {}).get("character_ids") or []),
        )
        for panel in repaired_preview.get("panels") or []
    ]
    reference_repair_needed = current_character_refs != repaired_character_refs
    if warnings:
        st.error(f"合同有 {len(warnings)} 条生产级错误，已禁止注册资产和视频任务。")
        with st.expander("查看合同错误", expanded=True):
            for warning in warnings:
                st.write(f"- {warning}")
        character_reference_errors = [
            warning for warning in warnings
            if "speaker_id must reference a visible character" in warning
            or "prompt_package.character_ids missing" in warning
            or "character_ids unknown" in warning
            or "character_ids must be a list" in warning
        ]
        if character_reference_errors and st.button(
            "自动修复人物引用并重新审核分镜",
            type="primary",
            use_container_width=True,
        ):
            repaired = repair_episode_character_references(episode)
            remaining = validate_episode_contract(repaired)
            _set_episode(repaired, persisted=False, dirty=True)
            if remaining:
                st.warning(f"已修复可确定的人物引用，仍有 {len(remaining)} 条错误需要编辑。")
            else:
                st.success("人物、说话人与 Prompt 引用已统一；请重新批准分镜并保存合同。")
            st.rerun()
    if reference_repair_needed:
        st.warning(
            "检测到群像镜头没有完整绑定同场人物。若继续使用旧合同，视频参考图会只带说话人，"
            "其他角色将自由漂移；必须先修复并重新审核分镜。"
        )
        if st.button(
            "检测并修复群像镜头人物覆盖",
            type="primary",
            use_container_width=True,
        ):
            _set_episode(repaired_preview, persisted=False, dirty=True)
            st.success("群像镜头已绑定全部同场人物；请检查每镜角色列表，重新批准分镜并保存合同。")
            st.rerun()
    for index, panel in enumerate(episode.get("panels") or [], 1):
        panel_id = str(panel.get("panel_id") or panel.get("name"))
        package = panel.get("prompt_package") or {}
        job = _job_for_panel(snapshot, panel_id)
        with st.expander(f"{index:02d} · {panel_id} · {panel.get('scene_id')}", expanded=index == 1):
            shot_metrics = st.columns(4)
            shot_metrics[0].metric("镜头角色", panel.get("shot_role") or "-")
            shot_metrics[1].metric("成片选段", f"{float(panel.get('edit_duration_seconds') or 0):g}s")
            shot_metrics[2].metric("H3 源素材", f"{float(panel.get('source_generation_duration_seconds') or panel.get('duration_seconds') or 0):g}s")
            shot_metrics[3].metric("优先级", panel.get("priority") or "-")
            readiness = shot_readiness_rows(panel, episode, snapshot)
            ready_count = sum(row["ready"] for row in readiness)
            blocked_count = sum(row["blocking"] for row in readiness)
            pending_count = sum(row["state"] == "pending" for row in readiness)
            with st.expander(
                f"逐镜生产就绪清单 · {ready_count}/{len(readiness)} 就绪"
                f" · {blocked_count} 阻断 · {pending_count} 处理中",
                expanded=bool(blocked_count),
            ):
                st.caption("未知、未注册或缺少当前后端证据的项目不会显示为就绪。")
                for row in readiness:
                    message = f"{row['label']}：{row['detail']}"
                    if row["state"] == "ready":
                        st.success(message)
                    elif row["state"] == "pending":
                        st.warning(message)
                    else:
                        st.error(message)
            st.markdown(f"**可见动作：** {panel.get('visible_action') or '-'}")
            st.caption(
                f"状态：{panel.get('first_state') or '-'} → {panel.get('final_state') or '-'} · "
                f"因果：{panel.get('cause') or '-'} · 下一钩子：{panel.get('next_hook') or '-'}"
            )
            continuity = st.columns(3)
            continuity[0].caption(f"连续组：{panel.get('continuity_group') or '-'}")
            continuity[1].caption(f"上一镜：{panel.get('previous_panel_id') or '链首'}")
            continuity[2].caption("角色：" + ", ".join(panel.get("character_ids") or []))
            st.markdown(f"**首帧：** {panel.get('first_frame') or '-'}")
            st.markdown(f"**尾帧：** {panel.get('last_frame') or '-'}")
            tabs = st.tabs(["预览", "正向 Prompt", "负向 Prompt", "对白 / 字幕 / 声音", "连续状态", "高级：原始 JSON"])
            with tabs[0]:
                media_state = job_media_for_review(job, [PROJECTS_DIR, COMFYUI_OUTPUT])
                if media_state["qa_invalidated"] and str(job.get("status") or "").lower() not in {
                    "completed", "finalized", "succeeded", "success", "delivered",
                }:
                    st.warning("已拒收，旧片仅归档供审计，不参与合片；当前镜头等待重新生成并验收。")
                _render_media(media_state["current"], max_items=4)
                if not media_state["current"]:
                    st.info("该镜尚无图片或视频预览。")
                if media_state["audit"]:
                    with st.expander("拒收审计片（历史版本，不参与合片）", expanded=False):
                        _render_media(media_state["audit"], max_items=4)
            with tabs[1]:
                st.code(package.get("positive_prompt") or panel.get("positive_prompt") or "", language="text")
            with tabs[2]:
                st.code(package.get("negative_prompt") or panel.get("negative_prompt") or "", language="text")
            with tabs[3]:
                st.markdown("**批准口播**")
                st.json(panel.get("spoken_dialogue") or [], expanded=False)
                st.markdown("**派生字幕（仅后期，不进入 H3）**")
                st.json(panel.get("subtitle_timeline") or [], expanded=False)
                for warning in panel.get("subtitle_warnings") or []:
                    st.warning(warning)
                st.markdown("**声音提示**")
                st.json(panel.get("audio_cues") or [], expanded=False)
                st.caption("H3 visible text policy: forbidden")
            with tabs[4]:
                st.json({"state_in": panel.get("continuity_state_in"), "state_out": panel.get("continuity_state_out")})
            with tabs[5]:
                st.json(panel, expanded=False)
            _item_editor(episode, "panel", panel_id, panel, api_key)
    for warning in episode.get("continuity_warnings") or []:
        st.error(f"连续性断链：{warning}")
    _approval_button(episode, "storyboard", "分镜")


def _creative_gate(ep_id: str, episode: dict[str, Any]) -> None:
    st.subheader("门禁一 · 批准创作合同后才能保存 / 注册")
    state = approval_state(episode)["creative"]
    cols = st.columns(3)
    for column, key, label in zip(cols, ("story", "characters", "storyboard"), ("故事", "人物", "分镜")):
        column.metric(label, "已批准" if state[key] else "待批准")
    ready = creative_gate_ready(episode)
    contract_errors = validate_episode_contract(episode)
    if st.button(
        "保存合同并注册任务",
        type="primary",
        disabled=(
            not ready
            or bool(contract_errors)
            or episode.get("is_demo")
            or (_service("prepare_contract") or _service("prepare_episode")) is None
        ),
        use_container_width=True,
    ):
        try:
            snapshot = _persist(ep_id, episode)
            approve_handler = _service("approve_contract")
            if approve_handler is None:
                raise RuntimeError("render_service.approve_contract 公共接口不可用")
            contract_hash = (snapshot.get("pipeline") or {}).get("contract_hash")
            snapshot = approve_handler(ep_id, expected_hash=contract_hash)
            st.session_state["last_snapshot"] = snapshot
            _set_episode(snapshot_episode(snapshot, episode), persisted=True, dirty=False)
            st.session_state["contract_approved"] = True
            st.session_state["assets_approved"] = False
            st.session_state["flash_success"] = "创作合同已保存、任务已注册；尚未启动任何 GPU worker。"
            st.rerun()
        except Exception as exc:
            st.error(f"保存 / 注册失败：{exc}")
    if not ready:
        st.info("请分别批准故事、人物和分镜。任何编辑或单项重生都会撤回受影响的批准。")
    elif contract_errors:
        st.error("创作审批虽已勾选，但合同校验未通过；必须先修复上方错误，不能保存或进入资产阶段。")


def _asset_approval_button(
    ep_id: str,
    episode: dict[str, Any],
    kind: str,
    item_id: str,
    ready: bool,
    *,
    contract_valid: bool,
) -> None:
    state = approval_state(episode)["assets"]
    key = "character_ids" if kind == "character" else "scene_ids"
    approved = item_id in state[key]
    collection = episode.get("character_bible" if kind == "character" else "scene_bible") or []
    id_key = "character_id" if kind == "character" else "scene_id"
    item = next((value for value in collection if str(value.get(id_key)) == item_id), {})
    review_status = str(item.get("asset_review_status") or item.get("asset_status") or "pending").lower()
    is_technical_retry = review_status == "failed"
    prompt_refiner_ready = _minimax_available(api_key)
    feedback = st.text_area(
        "重试备注（可选）" if is_technical_retry else "拒收反馈（必填，会进入下一轮生成提示词）",
        key=f"asset_reject_reason_{kind}_{item_id}",
        placeholder=(
            "例如：ComfyUI 显存中断，保持当前创意合同重试"
            if is_technical_retry else
            "具体写出画面错误与期望修正，例如：室内不应下雨；公益箱必须位于收银台上"
        ),
        height=72,
    ).strip()
    approve_col, reject_col = st.columns(2)
    if approve_col.button(
        "撤回批准" if approved else "批准",
        key=f"asset_approve_{kind}_{item_id}",
        # Revocation remains available, but an invalid creative contract can
        # never gain a new asset approval.
        disabled=not ready or (not approved and not contract_valid),
        use_container_width=True,
    ):
        updated = with_asset_approval(episode, kind, item_id, not approved)
        _set_episode(
            updated,
            persisted=st.session_state.get("contract_persisted", False),
            dirty=False,
        )
        st.rerun()
    reject_handler = _service("reject_asset")
    retry_handler = _service("retry_asset")
    can_reject = bool(item.get("reference_images")) or review_status in {"rejected", "failed"}
    retry_label = (
        "重试生成" if review_status == "failed" else
        "再次重生" if review_status == "rejected" else "拒绝并重生"
    )
    if reject_col.button(
        retry_label,
        key=f"asset_reject_{kind}_{item_id}",
        disabled=(
            not contract_valid
            or not can_reject
            or review_status == "regenerating"
            or (not is_technical_retry and not feedback)
            or (not is_technical_retry and not prompt_refiner_ready)
            or reject_handler is None
            or retry_handler is None
        ),
        use_container_width=True,
    ):
        rejection_reason = feedback or "技术失败后从网页人工重试"
        rejected = with_asset_review_status(
            episode, kind, item_id, "rejected", reason=rejection_reason
        )
        _set_episode(
            rejected,
            persisted=st.session_state.get("contract_persisted", False),
            dirty=False,
        )
        st.session_state["assets_approved"] = False
        try:
            action_handler = retry_handler if is_technical_retry else reject_handler
            _call_asset_service(
                action_handler, ep_id, f"{ep_id}:{kind}:{item_id}",
                reason=rejection_reason,
            )
            if not is_technical_retry:
                with st.spinner("MiniMax 提示词大师正在根据拒收反馈改写模型 Prompt…"):
                    refined = regenerate_contract_item(
                        rejected, kind, item_id, rejection_reason,
                        api_key=api_key or None,
                    )
                _set_episode(refined, persisted=False, dirty=True)
                st.session_state["flash_success"] = (
                    "已拒收当前资产，并由 MiniMax 按反馈改写模型提示词。"
                    "请在上方审阅新 Prompt、重新批准受影响的创作项并保存合同；"
                    "保存后再生成资产，旧 Prompt 不会被继续抽卡。"
                )
                st.rerun()
            worker_handler = _service("prepare_assets")
            if worker_handler is not None:
                _call_service(worker_handler, ep_id, rejected)
            refreshed = _snapshot(ep_id)
            st.session_state["last_snapshot"] = refreshed
            refreshed_episode = merge_episode_asset_review_state(
                snapshot_episode(refreshed, rejected), rejected, refreshed,
            )
            regenerating = with_asset_review_status(
                refreshed_episode, kind, item_id, "regenerating"
            )
            _set_episode(
                regenerating,
                persisted=st.session_state.get("contract_persisted", False),
                dirty=False,
            )
            if worker_handler is not None:
                st.success("该资产已拒绝、旧批准已撤销，并已重新排队且请求资产 worker 启动。视频门禁保持关闭。")
            else:
                st.warning("该资产已拒绝并重新排队，但 prepare_assets 不可用，需稍后手动启动资产生产。")
            st.rerun()
        except Exception as exc:
            st.error(f"资产已在本地标记拒绝，但公共拒绝/重生服务调用失败：{exc}")
    if reject_handler is None or retry_handler is None:
        st.caption("单项拒绝/重生接口尚未就绪")
    elif not is_technical_retry and not prompt_refiner_ready:
        st.caption("配置 MiniMax API Key 后，拒收反馈才会由提示词大师改写为下一轮模型 Prompt。")
    status_label = (
        "🔄 重生中" if review_status == "regenerating" else
        "⛔ 已拒绝" if review_status == "rejected" else
        "✅ 资产已批准" if approved else
        "⏳ 待批准" if ready else "❌ 资产未就绪"
    )
    st.caption(status_label)


@st.fragment(run_every="2s")
def _asset_gate(ep_id: str, episode: dict[str, Any]) -> None:
    st.subheader("门禁二 · 批准人物与场景资产后才能启动视频")
    snapshot: dict[str, Any] = {}
    try:
        snapshot = _snapshot(ep_id)
        latest_episode = snapshot_episode(snapshot, episode)
        local_episode = st.session_state.get("episode") or episode
        # A polling fragment must never replace an unsaved storyboard edit
        # with the older durable contract.  The user must remain in control of
        # creative changes until they explicitly press Save/Register.
        if st.session_state.get("local_dirty"):
            episode = local_episode
        elif latest_episode:
            before_sync = _asset_sync_signature(local_episode)
            episode = merge_episode_asset_review_state(latest_episode, local_episode, snapshot)
            st.session_state["episode"] = episode
            st.session_state["last_snapshot"] = snapshot
            # A fragment can update the lower asset gate while the character
            # cards above remain visually stale.  Trigger one app rerun only
            # when the persisted asset bundle actually changes.
            if _asset_sync_signature(episode) != before_sync:
                st.rerun()
    except Exception as exc:
        st.warning(f"资产状态刷新失败，将自动重试：{exc}")
    readiness = asset_readiness(episode)
    contract_errors = validate_episode_contract(episode)
    state = approval_state(episode)["assets"]
    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
    backend_status = str(pipeline.get("assets_status") or "pending")
    ready_count = sum(readiness["characters"].values()) + sum(readiness["scenes"].values())
    total_count = len(readiness["characters"]) + len(readiness["scenes"])
    if backend_status == "approved":
        st.success("人物与场景资产已通过后台门禁，可以继续启动视频生产。")
    elif total_count and ready_count == total_count:
        st.success("参考图片已全部生成并回传。请逐项预览；全部批准后即可进入视频生产。")
    else:
        st.info(f"资产生成/回传进度：{ready_count}/{total_count}。本区域每 2 秒自动刷新，无需手动刷新网页。")
    assets_service = _service("prepare_assets")
    if st.button(
        "生成人物 + 场景资产",
        disabled=(
            bool(contract_errors)
            or not st.session_state.get("contract_approved")
            or assets_service is None
        ),
        use_container_width=True,
    ):
        try:
            result = _call_service(assets_service, ep_id, episode)
            st.success("人物与场景资产后台 worker 已请求启动；图片完成后会自动出现在下方。")
            st.json(result, expanded=False)
        except Exception as exc:
            st.error(f"资产任务失败：{exc}")
    if assets_service is None:
        st.warning("当前公共服务没有 prepare_assets；可附加已有参考图，但仍需后端资产审批接口。")
    if contract_errors:
        st.error("当前合同校验失败：禁止重新生成资产、批准资产或启动视频。请回到分镜错误区修复并重新保存。")

    reviewable_assets = [
        *(f"{ep_id}:character:{item_id}" for item_id, ready in readiness["characters"].items() if ready),
        *(f"{ep_id}:scene:{item_id}" for item_id, ready in readiness["scenes"].items() if ready),
    ]
    reject_handler = _service("reject_asset")
    batch_rejection_reason = st.text_area(
        "整批拒收反馈（使用整批重生时必填）",
        key=f"batch_asset_reject_reason_{ep_id}",
        placeholder="具体说明共同问题，例如：三张人物画风不一致；场景出现文字和品牌标识",
        height=72,
    ).strip()
    if st.button(
        "整批拒绝并重生（画风 / 人物 / 场景不一致）",
        disabled=(
            bool(contract_errors)
            or not reviewable_assets
            or not batch_rejection_reason
            or reject_handler is None
            or assets_service is None
        ),
        use_container_width=True,
    ):
        try:
            for asset_id in reviewable_assets:
                _call_asset_service(
                    reject_handler, ep_id, asset_id, reason=batch_rejection_reason,
                )
            result = _call_service(assets_service, ep_id, episode)
            st.session_state["assets_approved"] = False
            st.success(
                f"已从网页拒绝 {len(reviewable_assets)} 项当前资产，并只启动一个后台 worker。"
                "图片完成后会在本页自动替换，视频门禁保持关闭。"
            )
            st.json(result, expanded=False)
            st.rerun()
        except Exception as exc:
            st.error(f"整批拒绝 / 重生失败：{exc}")

    st.markdown("#### 人物资产审核")
    char_cols = st.columns(max(1, min(4, len(readiness["characters"]))))
    for index, (item_id, ready) in enumerate(readiness["characters"].items()):
        with char_cols[index % len(char_cols)]:
            st.markdown(f"`{item_id}`")
            character = next(
                (item for item in episode.get("character_bible", []) if str(item.get("character_id")) == item_id),
                {},
            )
            refs = existing_media_paths(character.get("reference_images") or [], [PROJECTS_DIR, COMFYUI_INPUT])
            if refs:
                _render_media(refs, max_items=3)
            else:
                st.caption(f"生成状态：{character.get('asset_status') or 'pending'} · 暂无可预览图片")
            if character.get("asset_error"):
                st.error(f"生成失败：{character['asset_error']}")
            _asset_approval_button(
                ep_id, episode, "character", item_id, ready,
                contract_valid=not bool(contract_errors),
            )
    st.markdown("#### 场景资产审核")
    scene_cols = st.columns(max(1, min(4, len(readiness["scenes"]))))
    for index, (item_id, ready) in enumerate(readiness["scenes"].items()):
        with scene_cols[index % len(scene_cols)]:
            st.markdown(f"`{item_id}`")
            scene = next(
                (item for item in episode.get("scene_bible", []) if str(item.get("scene_id")) == item_id),
                {},
            )
            refs = existing_media_paths(scene.get("reference_images") or [], [PROJECTS_DIR, COMFYUI_INPUT])
            if refs:
                _render_media(refs, max_items=3)
            else:
                st.caption(f"生成状态：{scene.get('asset_status') or 'pending'} · 暂无可预览图片")
            if scene.get("asset_error"):
                st.error(f"生成失败：{scene['asset_error']}")
            _asset_approval_button(
                ep_id, episode, "scene", item_id, ready,
                contract_valid=not bool(contract_errors),
            )

    start_handler = _service("start_production") or _service("start_worker")
    gate_ready = asset_gate_ready(episode)
    expected_hashes = reviewed_asset_hashes(episode, snapshot)
    reviewed_hashes_complete = total_count > 0 and len(expected_hashes) == total_count
    st.caption(
        f"人物批准 {len(state['character_ids'])}/{len(readiness['characters'])} · "
        f"场景批准 {len(state['scene_ids'])}/{len(readiness['scenes'])}"
    )
    if st.button(
        "启动视频生产",
        type="primary",
        disabled=(
            bool(contract_errors)
            or not gate_ready
            or not reviewed_hashes_complete
            or not st.session_state.get("contract_approved")
            or start_handler is None
        ),
        use_container_width=True,
    ):
        try:
            approve_handler = _service("approve_assets")
            if approve_handler is None:
                raise RuntimeError("render_service.approve_assets 公共接口不可用")
            snapshot = approve_handler(ep_id, expected_hashes=expected_hashes)
            st.session_state["last_snapshot"] = snapshot
            result = _call_service(start_handler, ep_id, episode)
            st.session_state["assets_approved"] = True
            st.session_state["production_started"] = True
            st.session_state["local_dirty"] = False
            st.success("视频生产 worker 已请求启动。")
            st.json(result, expanded=False)
        except Exception as exc:
            st.error(f"启动视频生产失败：{exc}")
    if not gate_ready:
        st.info("视频启动仍被门禁拦截：每个人物和场景都必须有参考资产，并逐项人工批准。")
    elif not reviewed_hashes_complete:
        st.warning("已批准资产与当前页面预览哈希不一致；等待自动刷新后请重新审核，禁止盲目批准。")


@st.fragment(run_every="2s")
def _live_jobs(ep_id: str, episode: dict[str, Any]) -> None:
    st.subheader("实时任务、继续与重试")
    if render_service is None:
        st.warning(f"生产服务不可用：{RENDER_SERVICE_ERROR}")
        return
    try:
        snapshot = _snapshot(ep_id)
        st.session_state["last_snapshot"] = snapshot
        local_episode = st.session_state.get("episode") or episode
        current_episode = (
            local_episode
            if st.session_state.get("local_dirty")
            else merge_episode_asset_review_state(
                snapshot_episode(snapshot, episode),
                local_episode,
                snapshot,
            )
        )
        if current_episode:
            # Keep regenerated character/scene references and worker status
            # available to the next full-page rerun without blocking here.
            st.session_state["episode"] = current_episode
        jobs = normalize_jobs(snapshot.get("jobs") or [])
    except Exception as exc:
        st.error(f"读取任务失败：{exc}")
        return
    counts = job_counts(jobs)
    review_summary = content_review_summary(snapshot)
    contract_errors = validate_episode_contract(current_episode)
    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
    backend_gate_ready = (
        pipeline.get("contract_status") == "approved"
        and pipeline.get("assets_status") == "approved"
    )
    cols = st.columns(4)
    cols[0].metric("技术生成完成", f"{review_summary['technical_complete']}/{review_summary['total']}")
    cols[1].metric("自动 QA 通过", f"{review_summary['automated_qa_passed']}/{review_summary['total']}")
    cols[2].metric("人工批准", f"{review_summary['human_approved']}/{review_summary['total']}")
    cols[3].metric("拒收", review_summary["rejected"])
    st.caption(
        f"运行 / 排队 {counts['active']} · 技术失败 {counts['failed']} · "
        "技术生成完成不代表内容合格。"
    )
    worklists = classify_shot_worklist(jobs, snapshot)
    worklist_labels = {
        f"需要处理 ({worklists['counts']['needs_attention']})": "needs_attention",
        f"运行中 ({worklists['counts']['active']})": "active",
        f"待人审 / 发布 ({worklists['counts']['awaiting_review']})": "awaiting_review",
        f"已通过 ({worklists['counts']['passed']})": "passed",
        f"全部 ({worklists['counts']['all']})": "all",
    }
    selected_worklist = st.selectbox(
        "镜头工作清单",
        list(worklist_labels),
        index=0,
        key=f"shot_worklist_filter_{ep_id}",
        help="默认只显示需要处理的镜头；‘已通过’必须绑定当前 QA、artifact、edit-selection 与 release 证据。",
    )
    visible_jobs = worklists[worklist_labels[selected_worklist]]
    st.caption(
        f"当前显示 {len(visible_jobs)}/{counts['total']} 镜；总计数与生产控制始终基于全部任务。"
    )
    if review_summary["same_anchor_safe_count"]:
        st.error(
            "内容 QA 失败：检测到 "
            f"{review_summary['same_anchor_safe_count']}/{review_summary['total']} 镜复用同一静态锚；"
            "仅对白、字幕或容器变化不能构成不同剧情分镜。"
        )
    progress = float(snapshot.get("progress") or (counts["success"] / counts["total"] if counts["total"] else 0))
    st.progress(max(0.0, min(1.0, progress)), text=f"整体进度 {progress:.0%}")
    resume_handler = _service("resume") or _service("resume_jobs")
    start_handler = _service("start_production") or _service("start_worker")
    if st.button(
        "继续未完成任务",
        disabled=(
            not jobs
            or resume_handler is None
            or bool(contract_errors)
            or not backend_gate_ready
        ),
    ):
        try:
            result = _call_service(resume_handler, ep_id, current_episode)
            if backend_gate_ready and start_handler is not None:
                _call_service(start_handler, ep_id, current_episode)
                st.success(f"已重新排队 {result.get('resumed', 0)} 个任务并请求 worker 启动。")
            else:
                st.warning("任务已重新排队，但资产门禁未满足或 worker 接口不可用，尚未启动执行。")
        except Exception as exc:
            st.error(f"继续失败：{exc}")

    safe_root = earliest_qa_rejected_failed_job(jobs)
    if safe_root:
        with st.container(border=True):
            st.error("同一镜头已经被人工 QA 拒收；继续盲目重抽可能重复人物漂移、群像缺失或随机文字。")
            st.markdown("#### 显式降级：连续性安全模式（低动态，严格人物与字幕）")
            st.write(
                "该模式使用已批准的群像锚与人物/场景参考，采用受控低动态镜头运动，"
                "对白与字幕仍由 approved spoken_dialogue 在后期确定性合成。"
                "它会牺牲大动作、快速运镜和夸张动态，以优先保证人物数量、身份、场景与禁字稳定。"
            )
            st.warning("这不是 H3 重抽或高动态生成，也不会自动启用；属于需要人工确认的质量降级路径。")
            st.caption(
                "严格连续链起点："
                + str(safe_root.get("panel_id") or safe_root.get("panel_name") or safe_root.get("job_id"))
                + f" · job_id={safe_root.get('job_id')}"
            )
            anchor_candidates = continuity_anchor_candidates(
                safe_root,
                current_episode,
                jobs,
                [PROJECTS_DIR / ep_id],
            )
            if anchor_candidates:
                options = {
                    f"{item['label']} · {item['path']}": item["path"]
                    for item in anchor_candidates
                }
                selected_anchor_label = st.selectbox(
                    "候选群像锚（仅列出当前存在的图片）",
                    list(options),
                    key=f"continuity_safe_anchor_candidate_{ep_id}_{safe_root.get('job_id')}",
                )
                default_anchor = options[selected_anchor_label]
            else:
                default_anchor = ""
                st.error("没有可用群像锚：未找到连续尾帧、本镜 last_frame_path 或前序成功镜头尾帧。")
            source_anchor = st.text_input(
                "确认具体锚路径",
                value=default_anchor,
                key=f"continuity_safe_anchor_path_{ep_id}_{safe_root.get('job_id')}",
                help="只能批准当前磁盘上存在的 PNG/JPG/WEBP 群像锚。",
            ).strip()
            resolved_anchor_paths = existing_media_paths(
                [source_anchor], [PROJECTS_DIR / ep_id]
            )
            approved_anchor = next((
                path for path in resolved_anchor_paths
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
                and path.resolve().is_relative_to((PROJECTS_DIR / ep_id).resolve())
            ), None)
            if approved_anchor:
                st.image(str(approved_anchor), caption="待批准群像锚", use_container_width=True)
            elif source_anchor:
                st.error("指定锚路径不是现有 PNG/JPG/WEBP 文件，安全模式保持禁用。")
            anchor_reason = st.text_area(
                "群像锚批准理由",
                value="人工确认该锚完整包含批准群像、人物身份、数量、服装与场景构图，可作为严格连续链权威。",
                key=f"continuity_safe_anchor_reason_{ep_id}_{safe_root.get('job_id')}",
            ).strip()
            safe_confirmed = st.checkbox(
                "我确认接受更低动态表现，以换取严格群像、人物/场景连续性和确定性对白字幕",
                value=False,
                key=f"continuity_safe_confirm_{ep_id}",
            )
            safe_handler = _service("start_continuity_safe")
            approve_anchor_handler = _service("approve_continuity_anchor")
            if safe_handler is None or approve_anchor_handler is None:
                st.error(
                    "连续性安全模式服务未就绪：需要公共 approve_continuity_anchor "
                    "和 start_continuity_safe 接口。"
                )
            if st.button(
                "连续性安全模式（低动态，严格人物与字幕）",
                key=f"start_continuity_safe_{ep_id}",
                type="primary",
                disabled=(
                    not safe_confirmed
                    or safe_handler is None
                    or approve_anchor_handler is None
                    or approved_anchor is None
                    or not anchor_reason
                    or bool(contract_errors)
                    or not backend_gate_ready
                ),
                use_container_width=True,
            ):
                try:
                    approval, result = start_continuity_safe_via_facade(
                        render_service,
                        ep_id,
                        str(safe_root["job_id"]),
                        str(approved_anchor),
                        anchor_reason,
                    )
                    st.success("已请求连续性安全模式：镜头阶段生成确定性字幕侧车，平台交付时只烧录一次。该模式是显式低动态降级，不会伪装成 H3 高动态重抽。")
                    st.json({"anchor_approval": approval, "launch": result}, expanded=False)
                except Exception as exc:
                    st.error(f"连续性安全模式启动失败：{exc}")
    if not visible_jobs:
        st.info("当前筛选没有镜头；可切换到其他工作清单查看。")
    for job in visible_jobs:
        panel_id = str(job.get("panel_id") or job.get("panel_name") or job.get("job_id"))
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        job_settings = metadata.get("settings") if isinstance(metadata.get("settings"), dict) else {}
        render_profile = str(job_settings.get("render_profile") or "production")
        is_proof = render_profile == "proof" or job_settings.get("delivery_eligible") is False
        inputs = metadata.get("inputs") if isinstance(metadata.get("inputs"), dict) else {}
        package = inputs.get("prompt_package") or metadata.get("prompt_package") or {}
        artifact_sha256 = str(metadata.get("artifact_sha256") or "")
        technical_label = (
            ("预演生成完成（待审核晋级，禁止交付）" if is_proof else "正式素材生成完成（待内容验收）")
            if str(job.get("status") or "").lower() in {"succeeded", "success", "completed", "finalized"}
            else str(job.get("status") or "pending")
        )
        panel = next((
            item for item in current_episode.get("panels") or []
            if str(item.get("panel_id") or item.get("name") or "") == panel_id
        ), {})
        evidence = job_review_evidence(
            snapshot, job, [PROJECTS_DIR / ep_id, COMFYUI_OUTPUT],
        )
        job_summary = content_review_summary({**snapshot, "jobs": [job]})
        with st.expander(f"{panel_id} · {technical_label} · retry {job.get('retry_count', 0)}"):
            st.caption(f"工作清单判定：{job.get('worklist_reason') or '缺少可审计判定'}")
            st.progress(float(job.get("progress") or 0))
            st.caption(f"prompt_id={job.get('prompt_id') or '-'} · {job.get('updated_at') or '-'}")
            remote_queue = metadata.get("remote_queue") if isinstance(metadata.get("remote_queue"), dict) else {}
            if job.get("prompt_id") and str(job.get("status") or "") in {"submitted", "running"}:
                remote_state = str(remote_queue.get("state") or "unknown")
                if remote_state == "running":
                    st.info("ComfyUI 执行中（GPU 正在处理本镜）。")
                elif remote_state == "pending":
                    position = max(1, int(remote_queue.get("position") or 1))
                    st.info(f"ComfyUI 排队中：当前第 {position} 位，前方 {position - 1} 个任务；尚未占用 GPU 执行本镜。")
                elif remote_state == "absent_or_history_pending":
                    st.info("ComfyUI 队列中已不可见，正在等待 history / 文件写盘确认；系统不会重复提交。")
                else:
                    st.warning("暂时无法读取 ComfyUI 远端队列；保留 prompt_id 并防止重复提交。")
            profile_cols = st.columns(4)
            profile_cols[0].metric("生产档位", "低成本预演" if is_proof else "正式生产")
            profile_cols[1].metric("H3时长", f"{float(job_settings.get('actual_duration_seconds') or job_settings.get('duration_seconds') or 0):.3f}s")
            profile_cols[2].metric("目标像素", f"{float(job_settings.get('megapixels') or 0):.1f}MP")
            profile_cols[3].metric("Turbo步数", str(job_settings.get("turbo_steps") or ((job_settings.get("sampling_contract") or {}).get("steps")) or "-"))
            if is_proof:
                st.warning("当前是不可交付预演：即使技术成功和自动 QA 通过，也不能合片、发布或导出；必须人工确认后晋级正式生产。")
            elif metadata.get("preview_promotion"):
                promotion = metadata["preview_promotion"]
                st.success(
                    "本镜已由预演晋级正式生产 · proof "
                    f"{str(promotion.get('artifact_sha256') or '')[:12]} · prompt "
                    f"{str(promotion.get('prompt_sha256') or '')[:12]}"
                )
            if metadata.get("render_mode") == "continuity_safe":
                st.error(
                    "⚠ continuity_safe：静态锚保底样片，非剧情镜头。"
                    "即使技术状态 succeeded，也必须通过动作/叙事 QA 和人工批准；不得绿标。"
                )
            contract_cols = st.columns(2)
            contract_cols[0].markdown("**合同动作 / Action**")
            contract_cols[0].write(
                "; ".join(str(cut.get("shot_description") or "") for cut in panel.get("cuts") or [])
                or package.get("camera_timeline")
                or "未声明"
            )
            contract_cols[1].markdown("**合同 First → Last**")
            contract_cols[1].write(
                f"{panel.get('first_frame') or package.get('first_frame_prompt') or '未声明'}"
                " → "
                f"{panel.get('last_frame') or package.get('last_frame_prompt') or '未声明'}"
            )
            st.markdown("**自动 QA 首 / 中 / 尾证据**")
            review_window = evidence.get("review_window") if isinstance(evidence.get("review_window"), dict) else {}
            if review_window.get("binding") == "current_edit_selection":
                st.caption(
                    "以下证据及播放器定位均绑定当前成片选段 "
                    f"{float(review_window.get('source_start_seconds') or 0):.2f}-"
                    f"{float(review_window.get('source_end_seconds') or 0):.2f}s · selection "
                    f"{str(review_window.get('selection_sha256') or '')[:12]}；整段源片仅作辅助。"
                )
            evidence_cols = st.columns(3)
            for index, slot in enumerate(("first", "middle", "last")):
                path = evidence["paths"][slot]
                if path:
                    evidence_cols[index].image(str(path), caption=slot, use_container_width=True)
                elif evidence.get("video_path"):
                    evidence_cols[index].video(
                        str(evidence["video_path"]),
                        start_time=float(evidence["video_timestamps"][slot]),
                    )
                    evidence_cols[index].caption(
                        f"{slot} · t={float(evidence['video_timestamps'][slot]):.2f}s · 自动QA抽样流"
                    )
                else:
                    evidence_cols[index].warning(f"{slot} 证据缺失")
            if evidence.get("action_source") == "edit_selection_fallback":
                st.write(f"动作线索（自动选段理由，仍需人工看视频）：{evidence.get('action') or '缺失'}")
            else:
                st.write(f"动作证据：{evidence.get('action') or '缺失'}")
            if evidence.get("first_last_source") == "decoded_visual_luma_metric":
                st.write(f"首尾变化指标（亮度差，仅作抽样线索）：{evidence.get('first_last')}")
            else:
                st.write(f"首尾状态变化证据：{evidence.get('first_last') or '缺失'}")
            prompt_audit = runtime_prompt_audit(job, [PROJECTS_DIR / ep_id])
            with st.expander("最终发送给 MiniMax H3 的提示词与参照审计", expanded=False):
                audit_cols = st.columns(3)
                audit_cols[0].metric("导演 Skill", prompt_audit["director_skill_version"] or "未记录")
                audit_cols[1].metric("官方结构", prompt_audit["official_prompt_shape"] or "未记录")
                audit_cols[2].metric(
                    "Prompt 哈希", (prompt_audit["prompt_sha256"] or "未记录")[:12],
                )
                st.caption(
                    f"prompt contract={prompt_audit['runtime_prompt_contract'] or '-'} · "
                    f"reference bundle={(prompt_audit['reference_bundle_sha256'] or '-')[:12]}"
                )
                if prompt_audit["available"]:
                    st.code(prompt_audit["prompt"], language="text")
                    if prompt_audit["references"]:
                        st.dataframe(prompt_audit["references"], hide_index=True, use_container_width=True)
                if prompt_audit["error"]:
                    st.error(prompt_audit["error"])
            with st.expander("上游画面提示词包（调试）", expanded=False):
                st.code(package.get("positive_prompt") or "", language="text")
                st.code(package.get("negative_prompt") or "", language="text")
            refs = existing_media_paths(job.get("reference_images") or [], [PROJECTS_DIR, COMFYUI_INPUT])
            media_state = job_media_for_review(job, [PROJECTS_DIR, COMFYUI_OUTPUT])
            if media_state["qa_invalidated"] and str(job.get("status") or "").lower() not in {
                "completed", "finalized", "succeeded", "success", "delivered",
            }:
                st.warning("已拒收，旧片仅归档供审计，不参与合片；重试期间不会作为当前预览。")
            _render_media(refs + media_state["current"], max_items=6)
            if media_state["audit"]:
                with st.expander("拒收审计片（历史版本，不参与合片）", expanded=False):
                    _render_media(media_state["audit"], max_items=4)
            if job.get("error"):
                st.error(job["error"])
            character_ids = [str(value) for value in panel.get("character_ids") or []]
            rejection_audit = metadata.get("qa_rejection_audit")
            latest_rejection = (
                rejection_audit[-1]
                if isinstance(rejection_audit, list) and rejection_audit
                and isinstance(rejection_audit[-1], dict) else {}
            )
            rejection_classification = metadata.get("qa_rejection_classification")
            rejection_is_classified = bool(
                latest_rejection.get("category")
                or (
                    isinstance(rejection_classification, dict)
                    and rejection_classification.get("rejection_at") == latest_rejection.get("at")
                    and rejection_classification.get("rejection_reason") == latest_rejection.get("reason")
                    and rejection_classification.get("category")
                )
            )
            if latest_rejection and not rejection_is_classified:
                st.warning("这条旧拒收记录尚未分类；请由审核人补录类别后，系统才会判断是否必须重做构图锚。")
                legacy_category_label = st.selectbox(
                    "补录拒收问题类别",
                    ["动作节奏 / 成片窗口", "人物身份", "构图 / 场景", "连续状态", "其他"],
                    key=f"legacy_reject_category_{job.get('job_id')}",
                )
                legacy_category = {
                    "动作节奏 / 成片窗口": "action_timing_or_edit_window",
                    "人物身份": "identity_or_character",
                    "构图 / 场景": "composition_or_scene",
                    "连续状态": "continuity_or_state",
                    "其他": "other",
                }[legacy_category_label]
                classify_handler = _service("classify_job_rejection")
                if st.button(
                    "确认拒收分类",
                    key=f"classify_reject_{job.get('job_id')}",
                    disabled=classify_handler is None,
                ):
                    try:
                        classified = classify_handler(
                            ep_id, job["job_id"], rejection_category=legacy_category,
                        )
                        st.session_state["last_snapshot"] = classified
                        st.success("拒收类别已绑定到原始拒收记录；重试门禁已按类别重新计算。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"拒收分类失败：{exc}")
            latest_rejection_category = str(latest_rejection.get("category") or "")
            archived_output_record = (
                ((latest_rejection.get("archived_files") or {}).get("output_path") or {})
                if latest_rejection else {}
            )
            archived_output_path = Path(str(archived_output_record.get("path") or ""))
            if (
                str(job.get("status") or "") == "failed"
                and latest_rejection_category == "action_timing_or_edit_window"
                and archived_output_path.is_file()
            ):
                with st.expander("不用再次生成：人工选择真实动作段并对齐对白 / 字幕", expanded=True):
                    st.info(
                        "该路径不会调用 H3。系统只恢复当前拒收片的不可变字节，按人工动作窗口重新解码 QA；"
                        "若批准对白在窗口外，可把同一 H3 原生对白音频迁移到动作段，禁止同时叠加 TTS 双声。"
                    )
                    st.video(str(archived_output_path))
                    archived_probe = latest_rejection.get("probe") or {}
                    source_duration = float(archived_probe.get("duration_seconds") or 0)
                    approved_edit_duration = float(
                        ((metadata.get("inputs") or {}).get("shot_plan") or {}).get("edit_duration_seconds")
                        or (metadata.get("settings") or {}).get("edit_duration_seconds") or 0
                    )
                    max_start = max(0.0, source_duration - approved_edit_duration)
                    default_start = min(max_start, 1.5)
                    manual_start = st.number_input(
                        "人工动作选段起点（秒）",
                        min_value=0.0, max_value=float(max_start), value=float(default_start),
                        step=0.05, format="%.2f",
                        key=f"manual_reedit_start_{job.get('job_id')}_{archived_output_record.get('sha256')}",
                    )
                    st.caption(
                        f"当前将选择 {manual_start:.2f}-{manual_start + approved_edit_duration:.2f}s · "
                        f"固定合同长度 {approved_edit_duration:.2f}s · archive {str(archived_output_record.get('sha256') or '')[:12]}"
                    )
                    st.caption(
                        "选段预览从所选入点开始播放；请只验收上述固定时长内的动作、终态、音画同步，"
                        "不要用整段源片中的后续动作替代。"
                    )
                    st.video(str(archived_output_path), start_time=float(manual_start))
                    relocate_dialogue = st.checkbox(
                        "把批准的 H3 原生对白音频迁移到该动作段，并让最终字幕使用同一时间线（不叠加 TTS）",
                        value=bool(job.get("dialogue_cues")),
                        key=f"manual_reedit_audio_{job.get('job_id')}_{archived_output_record.get('sha256')}",
                    )
                    def _cue_outside_selected_window(cue: dict[str, Any]) -> bool:
                        try:
                            cue_start = float(cue.get("start_seconds", cue.get("start_s", 0)) or 0)
                            cue_end = float(cue.get("end_seconds", cue.get("end_s", cue_start)) or cue_start)
                        except (TypeError, ValueError):
                            return True
                        return cue_start < float(manual_start) - 1e-6 or cue_end > float(manual_start) + approved_edit_duration + 1e-6

                    excluded_dialogue = [
                        str(index) for index, cue in enumerate(job.get("dialogue_cues") or [])
                        if isinstance(cue, dict) and _cue_outside_selected_window(cue)
                    ]
                    if excluded_dialogue and not relocate_dialogue:
                        st.error(
                            "当前选段不含已批准对白 cue " + ", ".join(excluded_dialogue)
                            + "；不能静音恢复后再假装可交付。必须启用并通过原生音频证据迁移，"
                            "或另行建立批准配音轨合同。"
                        )
                    manual_reason = st.text_input(
                        "人工选段依据",
                        value="逐帧确认该窗口包含合同动作完成与稳定终态；对白和字幕必须与该窗口重对齐",
                        key=f"manual_reedit_reason_{job.get('job_id')}_{archived_output_record.get('sha256')}",
                    )
                    manual_reedit_handler = _service("reopen_rejected_with_manual_edit")
                    if st.button(
                        "应用人工动作窗口并重新执行内容 QA",
                        key=f"manual_reedit_apply_{job.get('job_id')}_{archived_output_record.get('sha256')}",
                        disabled=(
                            manual_reedit_handler is None
                            or not str(archived_output_record.get("sha256") or "")
                            or not manual_reason.strip()
                            or approved_edit_duration <= 0
                            or (bool(excluded_dialogue) and not relocate_dialogue)
                        ),
                    ):
                        try:
                            reopened = manual_reedit_handler(
                                ep_id, job["job_id"],
                                expected_archive_sha256=str(archived_output_record.get("sha256") or ""),
                                in_seconds=float(manual_start),
                                reason=manual_reason.strip(),
                                relocate_approved_dialogue=bool(relocate_dialogue),
                            )
                            st.session_state["last_snapshot"] = reopened
                            st.success("拒收片已按新动作窗口恢复并重新 QA；仍需逐帧人工批准，未自动晋级。")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"人工重选段失败：{exc}")
            paired_state_required = requires_paired_state_anchor(panel, len(character_ids))
            group_anchor_required = requires_approved_group_anchor(
                panel, metadata, len(character_ids),
            )
            group_anchor_candidate = (
                metadata.get("group_anchor_candidate")
                if isinstance(metadata.get("group_anchor_candidate"), dict) else {}
            )
            approved_group_anchor = (
                metadata.get("approved_group_anchor")
                if isinstance(metadata.get("approved_group_anchor"), dict) else {}
            )
            group_anchor_path = Path(str(
                approved_group_anchor.get("path")
                or group_anchor_candidate.get("path")
                or ""
            ))
            group_anchor_last_path = Path(str(
                approved_group_anchor.get("last_path")
                or group_anchor_candidate.get("last_path")
                or ""
            ))
            group_anchor_approved = bool(
                approved_group_anchor.get("status") == "approved"
                and group_anchor_path.is_file()
                and approved_group_anchor.get("sha256")
                and (
                    not paired_state_required
                    or (
                        group_anchor_last_path.is_file()
                        and approved_group_anchor.get("last_sha256")
                    )
                )
            )
            if group_anchor_required or group_anchor_candidate or approved_group_anchor:
                st.markdown("**逐镜首态 / 终态构图锚**")
                if group_anchor_required and not group_anchor_approved:
                    st.error(
                        "本镜曾因构图/人物/状态连续性被拒收。禁止只改文字后重试；"
                        "必须先生成、预览并哈希批准满足当前人物、场景和首态/终态合同的构图锚。"
                    )
                if group_anchor_path.is_file():
                    st.image(
                        str(group_anchor_path), caption=(
                            "已批准的 H3 Picture 1 首态构图"
                            if group_anchor_approved else "待人工审核的首态构图候选"
                        ), use_container_width=True,
                    )
                    st.caption(
                        f"candidate sha256="
                        f"{str((approved_group_anchor or group_anchor_candidate).get('sha256') or '')[:16]}"
                    )
                if paired_state_required and group_anchor_last_path.is_file():
                    st.image(
                        str(group_anchor_last_path), caption=(
                            "已批准的 H3 Picture 2 终态构图"
                            if group_anchor_approved else "待人工审核的终态构图候选"
                        ), use_container_width=True,
                    )
                    st.caption(
                        f"final sha256="
                        f"{str((approved_group_anchor or group_anchor_candidate).get('last_sha256') or '')[:16]}"
                    )
                candidate_status = str(group_anchor_candidate.get("status") or "missing")
                candidate_reviewable = bool(
                    candidate_status == "succeeded"
                    and group_anchor_path.is_file()
                    and (
                        not paired_state_required
                        or (
                            group_anchor_last_path.is_file()
                            and group_anchor_candidate.get("last_sha256")
                        )
                    )
                )
                if candidate_status == "running":
                    st.info("逐镜构图锚正在后台生成；页面每 2 秒自动刷新，完成后会在这里直接预览。")
                elif candidate_status == "failed":
                    st.error(f"逐镜构图锚生成失败：{group_anchor_candidate.get('error') or '未知错误'}")
                elif candidate_status == "succeeded" and not candidate_reviewable:
                    st.error("旧构图锚缺少可审计终态；必须重新生成首态 / 终态双锚，不能直接复用。")
                if group_anchor_approved:
                    st.success(
                        "逐镜构图锚已绑定当前图片哈希；H3 将用 Picture 1 锁定首态，"
                        "并在状态变化镜头用 Picture 2 锁定终态。"
                    )
                    reject_group_anchor_handler = _service("reject_group_anchor")
                    revoke_group_anchor_reason = st.text_input(
                        "撤回当前构图锚的原因",
                        value="成片证明当前首态/终态锚仍存在动作接触或道具连续性问题，需要重新制作并审核",
                        key=(
                            f"revoke_group_anchor_reason_{job.get('job_id')}_"
                            f"{approved_group_anchor.get('sha256')}"
                        ),
                    )
                    if st.button(
                        "撤回并拒收当前 H3 首态 / 终态锚",
                        key=(
                            f"revoke_group_anchor_{job.get('job_id')}_"
                            f"{approved_group_anchor.get('sha256')}"
                        ),
                        disabled=(
                            reject_group_anchor_handler is None
                            or not revoke_group_anchor_reason.strip()
                        ),
                    ):
                        try:
                            reject_group_anchor_handler(
                                ep_id, job["job_id"], reason=revoke_group_anchor_reason.strip(),
                            )
                            st.success("当前 H3 首态 / 终态锚已按原哈希撤回并保留审计记录，可重新生成。")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"撤回群像构图锚失败：{exc}")
                elif candidate_reviewable:
                    approve_group_anchor_handler = _service("approve_group_anchor")
                    reject_group_anchor_handler = _service("reject_group_anchor")
                    group_anchor_confirmed = st.checkbox(
                        "我已确认首态 / 终态中的合同人物清晰可见、身份服装正确、动作方向与场景正确且无伪文字",
                        key=f"group_anchor_confirm_{job.get('job_id')}_{group_anchor_candidate.get('sha256')}",
                    )
                    group_anchor_reason = st.text_input(
                        "逐镜构图锚批准说明",
                        value="合同人物、首态/终态构图、动作方向、场景和无文字要求均通过人工检查",
                        key=f"group_anchor_reason_{job.get('job_id')}_{group_anchor_candidate.get('sha256')}",
                    )
                    if st.button(
                        "批准并绑定 H3 首态 / 终态",
                        key=f"approve_group_anchor_{job.get('job_id')}_{group_anchor_candidate.get('sha256')}",
                        disabled=(
                            approve_group_anchor_handler is None
                            or not group_anchor_confirmed
                            or not group_anchor_reason.strip()
                        ),
                    ):
                        try:
                            approve_group_anchor_handler(
                                ep_id, job["job_id"],
                                expected_sha256=str(group_anchor_candidate.get("sha256") or ""),
                                reason=group_anchor_reason.strip(),
                            )
                            st.success("逐镜首态 / 终态构图锚已按当前哈希批准并绑定。")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"群像构图锚批准失败：{exc}")
                    group_anchor_reject_reason = st.text_input(
                        "逐镜构图锚拒收原因",
                        value="人物数量、身份、动作方向、首尾构图、场景或无文字要求未通过",
                        key=f"reject_group_anchor_reason_{job.get('job_id')}_{group_anchor_candidate.get('sha256')}",
                    )
                    if st.button(
                        "拒收当前逐镜构图锚",
                        key=f"reject_group_anchor_{job.get('job_id')}_{group_anchor_candidate.get('sha256')}",
                        disabled=(
                            reject_group_anchor_handler is None
                            or not group_anchor_reject_reason.strip()
                        ),
                    ):
                        try:
                            reject_group_anchor_handler(
                                ep_id, job["job_id"], reason=group_anchor_reject_reason.strip(),
                            )
                            st.success("当前逐镜构图锚已按哈希归档为拒收，不会进入 H3。")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"群像构图锚拒收失败：{exc}")
                if not group_anchor_approved and candidate_status != "running":
                    start_group_anchor_handler = _service("start_group_anchor")
                    if st.button(
                        "生成逐镜首态 / 终态构图锚" if not group_anchor_candidate else "重新生成逐镜首态 / 终态构图锚",
                        key=f"start_group_anchor_{job.get('job_id')}_{candidate_status}",
                        disabled=start_group_anchor_handler is None or bool(contract_errors) or not backend_gate_ready,
                    ):
                        try:
                            launch = start_group_anchor_handler(ep_id, job["job_id"])
                            if isinstance(launch, dict) and not launch.get("started", True):
                                st.warning(f"未重复启动：{launch.get('reason') or 'worker 已运行'}")
                            else:
                                st.success("逐镜构图锚已交给后台 worker；完成后会自动显示首态 / 终态预览。")
                        except Exception as exc:
                            st.error(f"群像构图锚启动失败：{exc}")
            if job.get("status") == "succeeded":
                job_metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
                edit_selection = (
                    job_metadata.get("edit_selection")
                    if isinstance(job_metadata.get("edit_selection"), dict) else {}
                )
                edit_selection_sha256 = str(edit_selection.get("selection_sha256") or "")
                selection_current = bool(
                    edit_selection_sha256
                    and str(edit_selection.get("source_artifact_sha256") or "") == artifact_sha256
                    and float(edit_selection.get("out_seconds") or 0) > float(edit_selection.get("in_seconds") or 0)
                    and 1.5 <= float(edit_selection.get("duration_seconds") or 0) <= 4.0
                )
                if selection_current:
                    st.caption(
                        f"成片选段 {float(edit_selection.get('in_seconds')):.2f}-"
                        f"{float(edit_selection.get('out_seconds')):.2f}s · "
                        f"{float(edit_selection.get('duration_seconds')):.2f}s · selection {edit_selection_sha256[:12]}"
                    )
                else:
                    st.error("当前源片尚无有效 1.5-4.0 秒 edit_selection，禁止人工批准；请等待自动选段与内容 QA。")
                approve_review_handler = (
                    _service("approve_preview_and_promote") if is_proof else (
                        _service("approve_job_review")
                        or _service("approve_panel_review")
                        or _service("approve_job")
                    )
                )
                reject_handler = _service("reject_job")
                narrative_confirmed = st.checkbox(
                    "我已对照合同确认：主体动作、首尾状态变化和本镜叙事信息均在当前版本中可见",
                    value=False,
                    key=f"narrative_confirm_{job.get('job_id')}_{artifact_sha256}",
                )
                identity_confirmed = st.checkbox(
                    "我已逐帧抽查：人物身份、脸型、发型、服装、人数和场景连续，无增人、漏人、合并或替换",
                    value=False,
                    key=f"identity_confirm_{job.get('job_id')}_{artifact_sha256}",
                )
                clean_frame_confirmed = st.checkbox(
                    "我已确认生成画面无字幕、标题、标签、Logo、招牌、乱码字形；交付字幕只在最终后期添加",
                    value=False,
                    key=f"clean_frame_confirm_{job.get('job_id')}_{artifact_sha256}",
                )
                approval_cols = st.columns(2)
                if approval_cols[0].button(
                    "批准预演并晋级正式生产" if is_proof else "批准当前正式镜头版本",
                    key=f"approve_job_review_{job.get('job_id')}_{artifact_sha256}",
                    disabled=(
                        approve_review_handler is None
                        or not artifact_sha256
                        or not prompt_audit["prompt_hash_matches"]
                        or not selection_current
                        or not evidence["complete"]
                        or job_summary["automated_qa_passed"] != 1
                        or not narrative_confirmed
                        or not identity_confirmed
                        or not clean_frame_confirmed
                    ),
                    help="必须自动 QA 通过、首中尾证据齐全，且人工分别确认叙事、身份连续和无模型伪文字；批准绑定当前 artifact_sha256。",
                ):
                    try:
                        if is_proof:
                            approved = approve_preview_and_promote_via_facade(
                                render_service, ep_id, str(job["job_id"]), artifact_sha256,
                                edit_selection_sha256,
                            )
                        else:
                            approved = approve_job_review_via_facade(
                                render_service, ep_id, str(job["job_id"]), artifact_sha256,
                                edit_selection_sha256,
                            )
                        st.session_state["last_snapshot"] = approved if isinstance(approved, dict) else snapshot
                        st.success(
                            "预演证据已绑定并排入正式生产；请先完成全部镜头预演审核，再点击继续未完成任务。"
                            if is_proof else
                            "已批准当前正式 artifact hash；产物变化会自动撤销本次批准。"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"镜头批准失败：{exc}")
                if approve_review_handler is None:
                    st.warning(
                        "预演晋级服务未就绪：缺少公共 approve_preview_and_promote facade。"
                        if is_proof else
                        "逐镜批准服务未就绪：缺少公共 approve_job_review facade，保持禁止导出。"
                    )
                rejection_category_label = st.selectbox(
                    "拒收问题类别",
                    ["动作节奏 / 成片窗口", "人物身份", "构图 / 场景", "连续状态", "其他"],
                    key=f"reject_job_category_{job.get('job_id')}",
                    help="只有人物、构图/场景、连续状态或未分类问题会强制重做逐镜构图锚；纯动作节奏问题仍须重跑并重新人工验收，但复用当前已批准人物/场景资产。",
                )
                rejection_category = {
                    "动作节奏 / 成片窗口": "action_timing_or_edit_window",
                    "人物身份": "identity_or_character",
                    "构图 / 场景": "composition_or_scene",
                    "连续状态": "continuity_or_state",
                    "其他": "other",
                }[rejection_category_label]
                reject_reason = st.text_input(
                    "镜头拒收原因",
                    value="人物、动作、随机文字或连续性未通过人工验收",
                    key=f"reject_job_reason_{job.get('job_id')}",
                )
                if approval_cols[1].button(
                    "拒收本镜并重置后续连续镜头",
                    key=f"reject_job_{job.get('job_id')}",
                    disabled=reject_handler is None or not reject_reason.strip(),
                ):
                    try:
                        rejected = reject_handler(
                            ep_id,
                            job["job_id"],
                            reason=reject_reason.strip(),
                            rejection_category=rejection_category,
                            interrupt_running=True,
                        )
                        st.session_state["last_snapshot"] = rejected
                        st.success("本镜已归档为拒收样片；严格连续链下游已清除旧尾帧并等待重跑。")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"镜头拒收失败：{exc}")
            if job.get("status") in {"queued", "failed", "error", "cancelled"}:
                preview_promotion = (
                    metadata.get("preview_promotion")
                    if isinstance(metadata.get("preview_promotion"), dict) else {}
                )
                approved_proof_path = Path(str(preview_promotion.get("output_path") or ""))
                stable_promotion_ready = bool(
                    preview_promotion.get("status") == "approved"
                    and preview_promotion.get("artifact_sha256")
                    and preview_promotion.get("decoded_visual_sha256")
                    and preview_promotion.get("prompt_sha256")
                    and preview_promotion.get("reference_bundle_sha256")
                    and approved_proof_path.is_file()
                )
                if stable_promotion_ready:
                    with st.container(border=True):
                        st.markdown("**已批准预演 → 稳定 720p 生产母版**")
                        st.warning(
                            (
                                "此前通过人审的 proof 已排队等待正式重渲染；可在尚未提交 ComfyUI 前，"
                                "改用内容锁定的稳定晋级，避免随机重绘破坏人物、场景和动作。"
                                if job.get("status") == "queued" else
                                "正式 H3 重渲染已被人工拒收，但此前通过人审的 proof 仍可安全复用。"
                            )
                            + " proof 由 artifact、解码画面、Prompt 和参考图哈希绑定；此路径不会再次"
                            "调用模型，只做确定性 720p 缩放、24fps 和 48kHz 双声道规范化，再重新"
                            "执行解码 QA、选段和正式片人工审核。不会凭空增加高分辨率细节。"
                        )
                        stable_confirmed = st.checkbox(
                            "我确认优先保留已批准画面内容，接受确定性放大而不再随机重生成",
                            key=f"stable_promotion_confirm_{job.get('job_id')}_{preview_promotion.get('artifact_sha256')}",
                        )
                        stable_reason = st.text_input(
                            "稳定生产晋级说明",
                            value=(
                                "优先保留已人工批准的 proof 内容，跳过随机正式重渲染并确定性规范为 720p"
                                if job.get("status") == "queued" else
                                "正式长时重渲染发生视觉漂移；保留已人工批准的 proof 内容并确定性规范为 720p"
                            ),
                            key=f"stable_promotion_reason_{job.get('job_id')}_{preview_promotion.get('artifact_sha256')}",
                        )
                        stable_handler = _service("promote_approved_preview_master")
                        if st.button(
                            "生成稳定 720p 生产母版",
                            key=f"stable_promotion_{job.get('job_id')}_{preview_promotion.get('artifact_sha256')}",
                            disabled=(
                                stable_handler is None
                                or not stable_confirmed
                                or not stable_reason.strip()
                                or bool(contract_errors)
                                or not backend_gate_ready
                            ),
                        ):
                            try:
                                promoted = stable_handler(
                                    ep_id, job["job_id"], reason=stable_reason.strip(),
                                    confirmed=stable_confirmed,
                                )
                                st.session_state["last_snapshot"] = promoted
                                st.success(
                                    "已批准 proof 已生成新的 720p 生产母版；"
                                    "仍需在待人审清单独立验收这个 production artifact。"
                                )
                                st.rerun()
                            except Exception as exc:
                                st.error(f"稳定 720p 生产母版失败：{exc}")
                retry_handler = _service("retry_job")
                prompt_id = str(job.get("prompt_id") or "").strip()
                retry_authorization = (
                    (job.get("metadata") or {}).get("remote_retry_authorization") or {}
                )
                needs_restart_confirmation = bool(
                    prompt_id
                    and str(retry_authorization.get("prompt_id") or "") != prompt_id
                )
                restart_authorizer = _service("authorize_retry_after_comfy_restart")
                retry_count = int(job.get("retry_count") or 0)
                max_retries = int(job.get("max_retries") or 0)
                retry_exhausted = bool(
                    job.get("status") == "failed" and retry_count >= max_retries
                )
                if retry_exhausted:
                    st.warning(
                        f"本镜已达到自动重试上限 {retry_count}/{max_retries}；系统不会继续消耗算力。"
                        "如已完成原因分析与提示词修正，必须由审核人显式批准增加 1 次额度。"
                    )
                    extra_retry_reason = st.text_input(
                        "增加一次重试额度的审核理由",
                        key=f"extra_retry_reason_{job.get('job_id')}_{max_retries}",
                        placeholder="写清失败原因、已完成的修正与本次重试验收目标",
                    )
                    extra_retry_handler = _service("authorize_additional_job_retry")
                    if st.button(
                        "批准增加 1 次重试额度",
                        key=f"authorize_extra_retry_{job.get('job_id')}_{max_retries}",
                        disabled=extra_retry_handler is None or not extra_retry_reason.strip(),
                    ):
                        try:
                            authorized = extra_retry_handler(
                                ep_id, job["job_id"], reason=extra_retry_reason.strip(),
                            )
                            st.session_state["last_snapshot"] = authorized
                            st.success("已审计增加 1 次重试额度；请重新检查修正内容后再点击重试。")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"增加重试额度失败：{exc}")
                restart_confirmed = False
                if needs_restart_confirmation:
                    st.warning(
                        "旧 Comfy prompt 仍绑定在本镜，但远端 history 已丢失。"
                        "禁止普通重试；只有确认 ComfyUI 已重启且当前队列为空后，"
                        "才能审计释放一次基础设施重试。"
                    )
                    restart_confirmed = st.checkbox(
                        "我确认 ComfyUI 已重启，当前队列为空，旧 prompt 不会继续运行",
                        key=f"confirm_comfy_restart_{job.get('job_id')}",
                    )
                retry_label = (
                    "确认 ComfyUI 重启并重试本镜"
                    if needs_restart_confirmation else "重试本镜"
                )
                if st.button(
                    retry_label,
                    key=f"retry_{job.get('job_id')}",
                    disabled=(
                        retry_handler is None
                        or start_handler is None
                        or job.get("status") == "queued"
                        or retry_exhausted
                        or bool(contract_errors)
                        or not backend_gate_ready
                        or (group_anchor_required and not group_anchor_approved)
                        or (needs_restart_confirmation and (
                            restart_authorizer is None or not restart_confirmed
                        ))
                    ),
                ):
                    try:
                        if needs_restart_confirmation:
                            restart_authorizer(
                                ep_id, job["job_id"], confirmed=restart_confirmed,
                            )
                        retry_handler(ep_id, job["job_id"])
                        if resume_handler is not None:
                            _call_service(resume_handler, ep_id, current_episode)
                        worker_result = _call_service(start_handler, ep_id, current_episode)
                        if isinstance(worker_result, dict) and not worker_result.get("started", True):
                            reason = worker_result.get("reason") or "worker 已在运行"
                            st.success(f"本镜已重新排队；未重复启动 worker：{reason}。")
                        else:
                            st.success("本镜已重新排队并请求 worker 启动。")
                    except Exception as exc:
                        st.error(f"重试失败：{exc}")


def _render_delivery_manifest(
    manifest: dict[str, Any], ep_id: str, *, persisted: bool, releasable: bool,
) -> None:
    """Show one durable delivery with preview, downloads and audit facts."""
    output_path = Path(str(manifest.get("output_path") or ""))
    package_path = Path(str(manifest.get("package_path") or ""))
    manifest_path = Path(str(manifest.get("manifest_path") or ""))
    manifest_release_status = str(manifest.get("release_status") or "").lower()
    approved_delivery = releasable and manifest_release_status in {"approved", "released"}
    if not approved_delivery:
        st.error(
            "历史技术导出，不可发布：manifest 缺少有效 release_status，"
            "或当前自动 QA / 逐镜人工批准 / 整集发布批准未全部通过。"
        )
    if output_path.is_file():
        st.video(str(output_path))
        with output_path.open("rb") as handle:
            st.download_button(
                "下载最终 MP4" if approved_delivery else "审计用 MP4（发布禁用）",
                data=handle.read(),
                file_name=output_path.name,
                mime="video/mp4",
                key=f"delivery_mp4_{ep_id}_{output_path.name}_{persisted}",
                disabled=not approved_delivery,
                use_container_width=True,
            )
    if package_path.is_file():
        with package_path.open("rb") as handle:
            st.download_button(
                "下载 delivery.zip" if approved_delivery else "审计用 delivery.zip（发布禁用）",
                data=handle.read(),
                file_name=package_path.name,
                mime="application/zip",
                key=f"delivery_zip_{ep_id}_{package_path.name}_{persisted}",
                disabled=not approved_delivery,
                use_container_width=True,
            )
    subtitle_manifest = (
        manifest.get("subtitles") if isinstance(manifest.get("subtitles"), dict) else {}
    )
    preset_manifest = (
        manifest.get("preset") if isinstance(manifest.get("preset"), dict) else {}
    )
    probe = manifest.get("probe") if isinstance(manifest.get("probe"), dict) else {}
    st.markdown("#### 交付 Manifest 关键值")
    metrics = st.columns(4)
    metrics[0].metric("平台规格", preset_manifest.get("canonical_name") or preset_manifest.get("requested_name") or "-")
    metrics[1].metric("burned_in", "True" if subtitle_manifest.get("burned_in") else "False")
    metrics[2].metric("subtitle_strict", "True" if subtitle_manifest.get("strict") is True else "False")
    metrics[3].metric("时长", f"{float(probe.get('duration_seconds') or 0):.2f}s")
    st.json({
        "output_path": str(output_path) if output_path.is_file() else None,
        "package_path": str(package_path) if package_path.is_file() else None,
        "manifest_path": str(manifest_path) if manifest_path.is_file() else manifest.get("manifest_path"),
        "preset": preset_manifest,
        "resize_mode": manifest.get("resize_mode"),
        "probe": probe,
        "subtitles": subtitle_manifest,
        "source_clip_count": len(manifest.get("source_clips") or []),
        "release_status": manifest.get("release_status"),
        "qa_report_hash": manifest.get("qa_report_hash"),
        "approved_artifact_hashes": manifest.get("approved_artifact_hashes"),
    }, expanded=False)


@st.fragment(run_every="2s")
def _delivery_panel(ep_id: str) -> None:
    st.subheader("平台成片导出")
    try:
        snapshot = _snapshot(ep_id)
        st.session_state["last_snapshot"] = snapshot
    except Exception as exc:
        st.error(f"读取交付状态失败，将自动重试：{exc}")
        return
    jobs = normalize_jobs(snapshot.get("jobs") or [])
    counts = job_counts(jobs)
    review = content_review_summary(snapshot)
    ready = bool(review["ready_for_export"])
    persisted_manifests = persisted_delivery_manifests(snapshot, [PROJECTS_DIR / ep_id])
    if persisted_manifests:
        manifest_release = str(persisted_manifests[0].get("release_status") or "").lower()
        if ready and manifest_release in {"approved", "released"}:
            st.success("已恢复批准发布的平台交付；刷新页面不会丢失成片与下载入口。")
        else:
            st.error("已恢复历史技术导出，但内容发布资格缺失或已撤销；不可作为合格交付。")
        _render_delivery_manifest(
            persisted_manifests[0], ep_id, persisted=True, releasable=ready,
        )
    export_handler = _service("export")
    if export_handler is None:
        st.warning("交付服务不可用：render_service.export 缺失")
        return
    release_handler = _service("approve_episode_release") or _service("approve_release")
    pre_release_ready = bool(
        review["total"]
        and review["technical_complete"] == review["total"]
        and review["automated_qa_passed"] == review["total"]
        and review["human_approved"] == review["total"]
    )
    gate_cols = st.columns(4)
    gate_cols[0].metric("技术完成", f"{review['technical_complete']}/{review['total']}")
    gate_cols[1].metric("自动 QA", f"{review['automated_qa_passed']}/{review['total']}")
    gate_cols[2].metric("人工批准", f"{review['human_approved']}/{review['total']}")
    gate_cols[3].metric("发布批准", "通过" if review["release_approved"] else "未通过")
    if st.button(
        "批准整集发布（绑定当前全部 artifact hash）",
        disabled=not pre_release_ready or release_handler is None or review["release_approved"],
        key=f"approve_episode_release_{ep_id}",
        help="只有技术完成、自动 QA、逐镜人工批准全部通过后才能批准发布。",
    ):
        try:
            top_qa = snapshot.get("content_qa") if isinstance(snapshot.get("content_qa"), dict) else {}
            result = approve_release_via_facade(
                render_service,
                ep_id,
                review["artifact_hashes"],
                review["edit_selection_hashes"],
                qa_report_hash=str(top_qa.get("report_hash") or ""),
            )
            st.session_state["last_snapshot"] = result if isinstance(result, dict) else snapshot
            st.success("整集发布已绑定当前 artifact hashes；任一镜头变化后必须重新审批。")
            st.rerun()
        except Exception as exc:
            st.error(f"整集发布批准失败：{exc}")
    if release_handler is None:
        st.warning("整集发布批准服务未就绪：缺少公共 approve_episode_release facade，导出保持禁用。")
    preset_label = st.selectbox("平台规格", list(DELIVERY_PRESET_OPTIONS))
    preset = DELIVERY_PRESET_OPTIONS[preset_label]
    st.caption("统一720p交付：竖屏720×1280，横屏1280×720；不降低内容QA、人工审核和发布门禁。")
    resize_mode = st.selectbox("缩放方式", ["fit", "fill"], help="fit 不裁人物；fill 可能裁切画面")
    burn_subtitles = st.toggle(
        "烧录批准对白字幕",
        value=True,
        key=f"delivery_burn_subtitles_{ep_id}",
        help=(
            "字幕只由已批准 spoken_dialogue 唯一派生并严格校验；"
            "不会让 H3 在画面中生成随机文字。关闭后仍交付 SRT/VTT/ASS 字幕包。"
        ),
    )
    st.caption("字幕来源：approved spoken_dialogue → 确定性后期字幕；subtitle_strict 始终开启。")
    if st.button("导出并验证成片", type="primary", disabled=not ready):
        try:
            with st.spinner("正在通过公共交付服务合成、探测并打包…"):
                manifest = export_handler(
                    ep_id,
                    preset,
                    resize_mode=resize_mode,
                    burn_subtitles=burn_subtitles,
                    subtitle_strict=True,
                )
            st.success(f"已生成经发布批准的交付：{manifest['output_path']}")
            _render_delivery_manifest(manifest, ep_id, persisted=False, releasable=True)
        except Exception as exc:
            st.error(f"导出失败：{exc}")
    if not ready:
        st.error(
            "导出硬门未通过：必须后台内容 QA 全部 passed、每镜人工批准当前 artifact hash，"
            "且整集 release_status=approved；缺少任何证据均禁止导出。"
        )


def _remember_series_snapshot(
    snapshot: dict[str, Any], fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Keep the latest public snapshot and its recoverable V4 contract in UI state."""
    restored = series_from_service_snapshot(snapshot, fallback)
    st.session_state["last_series_snapshot"] = snapshot
    if restored:
        st.session_state["series_contract"] = restored
        st.session_state["loaded_series_id"] = str(snapshot.get("series_id") or "")
    return restored


def _shared_asset_console(series_id: str, snapshot: dict[str, Any], season_approved: bool) -> None:
    """Review shared identity/world assets without doing GPU work in Streamlit."""
    st.divider()
    st.header("3 · 全季共享资产门禁")
    assets = snapshot.get("shared_assets") if isinstance(snapshot.get("shared_assets"), list) else []
    series_row = snapshot.get("series") if isinstance(snapshot.get("series"), dict) else {}
    status = str(series_row.get("shared_assets_status") or "pending")
    succeeded = sum(asset.get("status") == "succeeded" for asset in assets)
    approved = sum(bool(asset.get("approved")) for asset in assets)
    asset_metrics = st.columns(4)
    asset_metrics[0].metric("共享资产", len(assets))
    asset_metrics[1].metric("生成成功", succeeded)
    asset_metrics[2].metric("已批准", approved)
    asset_metrics[3].metric("后端状态", status)

    controls = st.columns(2)
    prepare_assets = _series_service("prepare_shared_assets")
    if controls[0].button(
        "生成 / 继续共享人物与场景资产",
        key=f"series_prepare_assets_{series_id}",
        disabled=not season_approved or prepare_assets is None,
        use_container_width=True,
    ):
        try:
            result = prepare_assets(series_id)
            st.toast("共享资产后台 worker 已启动" if result.get("started") else f"未重复启动：{result.get('reason') or 'worker 已运行'}")
            st.rerun()
        except Exception as exc:
            st.error(f"共享资产启动失败：{exc}")

    all_reviewable = bool(assets) and all(
        asset.get("status") == "succeeded" and asset.get("content_hash") for asset in assets
    )
    approve_assets = _series_service("approve_shared_assets")
    if controls[1].button(
        "批准全部共享资产并开启视频门禁",
        key=f"series_approve_assets_{series_id}",
        type="primary",
        disabled=not all_reviewable or approve_assets is None,
        use_container_width=True,
    ):
        try:
            expected = {str(asset["asset_id"]): str(asset["content_hash"]) for asset in assets}
            _remember_series_snapshot(approve_assets(series_id, expected_hashes=expected))
            st.rerun()
        except Exception as exc:
            st.error(f"共享资产批准失败：{exc}")

    if not assets:
        st.info("全季批准后，后端会按共享人物/场景圣经登记资产；当前没有可审核资产。")
        return
    reject_handler = _series_service("reject_shared_asset")
    retry_handler = _series_service("retry_shared_asset")
    for asset in assets:
        asset_id = str(asset.get("asset_id") or "")
        title = f"{asset.get('asset_type')} · {asset.get('source_id')} · {asset.get('status')}"
        with st.expander(title, expanded=asset.get("status") in {"failed", "succeeded"}):
            st.caption(f"asset_id={asset_id} · approved={bool(asset.get('approved'))}")
            paths = existing_media_paths(
                asset.get("reference_images") or [],
                [PROJECTS_DIR, COMFYUI_INPUT, COMFYUI_OUTPUT],
            )
            _render_media(paths, max_items=6)
            if asset.get("error"):
                st.error(str(asset["error"]))
            reason = st.text_input(
                "拒绝 / 重生原因",
                value="身份、构图或世界连续性未通过验收",
                key=f"series_asset_reason_{asset_id}",
            )
            action_cols = st.columns(2)
            if action_cols[0].button(
                "拒绝并重生",
                key=f"series_asset_reject_{asset_id}",
                disabled=reject_handler is None or asset.get("status") != "succeeded",
                use_container_width=True,
            ):
                try:
                    _remember_series_snapshot(reject_handler(series_id, asset_id, reason=reason))
                    if prepare_assets is not None:
                        prepare_assets(series_id)
                    st.rerun()
                except Exception as exc:
                    st.error(f"共享资产拒绝失败：{exc}")
            if action_cols[1].button(
                "重试该资产",
                key=f"series_asset_retry_{asset_id}",
                disabled=retry_handler is None or asset.get("status") not in {"failed", "cancelled", "queued"},
                use_container_width=True,
            ):
                try:
                    _remember_series_snapshot(retry_handler(series_id, asset_id, reason=reason))
                    if prepare_assets is not None:
                        prepare_assets(series_id)
                    st.rerun()
                except Exception as exc:
                    st.error(f"共享资产重试失败：{exc}")


@st.fragment(run_every="2s")
def _live_series_console(series_id: str) -> None:
    """Poll only the public series facade and expose non-blocking controls."""
    try:
        snapshot = _series_snapshot(series_id)
    except Exception as exc:
        st.error(f"整季状态读取失败：{exc}")
        return
    if not snapshot:
        st.info("整季尚未持久化。")
        return
    _remember_series_snapshot(snapshot, st.session_state.get("series_contract") or {})
    row = snapshot.get("series") if isinstance(snapshot.get("series"), dict) else {}
    counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
    episodes = snapshot.get("episodes") if isinstance(snapshot.get("episodes"), list) else []
    metrics = st.columns(4)
    metrics[0].metric("阶段", str(row.get("status") or "draft"))
    metrics[1].metric("已注册", f"{counts.get('registered', 0)}/{counts.get('expected', 0)}")
    metrics[2].metric("已完成", f"{counts.get('complete', 0)}/{counts.get('expected', 0)}")
    metrics[3].metric("共享资产", str(row.get("shared_assets_status") or "pending"))

    ready = bool(snapshot.get("ready"))
    start_series = _series_service("start_series")
    resume_series = _series_service("resume_series")
    retry_series = _series_service("retry_series")
    cancel_series = _series_service("cancel_series")
    controls = st.columns(4)
    series_actions = (
        ("启动整季", start_series, {}, not ready),
        ("继续整季", resume_series, {"start": True}, not ready or not episodes),
        ("重试整季失败项", retry_series, {"start": True}, not ready or not episodes),
        ("取消未完成任务", cancel_series, {"interrupt_running": False}, not episodes),
    )
    for column, (label, handler, kwargs, disabled) in zip(controls, series_actions):
        if column.button(
            label,
            key=f"{label}_{series_id}",
            disabled=disabled or handler is None,
            use_container_width=True,
        ):
            try:
                result = handler(series_id, **kwargs)
                if isinstance(result, dict) and result.get("series"):
                    _remember_series_snapshot(result)
                st.toast(f"{label}命令已提交")
                st.rerun()
            except Exception as exc:
                st.error(f"{label}失败：{exc}")

    preset = st.selectbox(
        "整季交付规格",
        list(DELIVERY_PRESET_OPTIONS),
        key=f"series_export_preset_{series_id}",
    )
    preset = DELIVERY_PRESET_OPTIONS[preset]
    outline = {
        int(item.get("episode_index") or index): item
        for index, item in enumerate((st.session_state.get("series_contract") or {}).get("season_outline") or [], 1)
    }
    for record in episodes:
        number = int(record.get("episode_number") or 0)
        outline_item = outline.get(number) or {}
        with st.expander(
            f"第 {number:02d} 集 · {outline_item.get('title') or record.get('ep_id')} · {record.get('status')}",
            expanded=record.get("status") in {"running", "failed", "cancelled"},
        ):
            st.caption(f"ep_id={record.get('ep_id')} · predecessor={record.get('predecessor_ep_id') or '-'}")
            if record.get("error"):
                st.error(str(record["error"]))
            media = existing_media_paths(
                [record.get("last_clip_path"), record.get("delivery_manifest") or {}],
                [PROJECTS_DIR, COMFYUI_OUTPUT],
            )
            _render_media(media, max_items=4)
            start_episode = _series_service("start_episode")
            resume_episode = _series_service("resume_episode")
            retry_episode = _series_service("retry_episode")
            cancel_episode = _series_service("cancel_episode")
            export_episode = _series_service("export_episode")
            episode_controls = st.columns(5)
            actions = (
                ("启动", start_episode, {}, not ready or record.get("status") in {"succeeded", "exported"}),
                ("继续", resume_episode, {"start": True}, not ready or record.get("status") in {"succeeded", "exported"}),
                ("重试", retry_episode, {"start": True}, not ready or record.get("status") not in {"failed", "cancelled"}),
                ("取消", cancel_episode, {"interrupt_running": False}, record.get("status") in {"succeeded", "exported", "cancelled"}),
                ("导出", export_episode, {"preset": preset}, record.get("status") not in {"succeeded", "exported"}),
            )
            for column, (label, handler, kwargs, disabled) in zip(episode_controls, actions):
                if column.button(
                    label,
                    key=f"series_ep_{label}_{series_id}_{number}",
                    disabled=disabled or handler is None,
                    use_container_width=True,
                ):
                    try:
                        result = handler(series_id, number, **kwargs)
                        if isinstance(result, dict) and result.get("series"):
                            _remember_series_snapshot(result)
                        elif label == "导出":
                            st.success(f"单集交付完成：{result.get('output_path') or result}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"第 {number} 集{label}失败：{exc}")

    export_season = _series_service("export_season")
    expected = int(counts.get("expected") or 0)
    complete = int(counts.get("complete") or 0)
    if st.button(
        "导出完整整季包",
        key=f"series_export_season_{series_id}",
        type="primary",
        disabled=not expected or complete != expected or export_season is None,
        use_container_width=True,
    ):
        try:
            manifest = export_season(series_id, preset)
            st.success(f"整季交付完成：{manifest.get('package_path')}")
            st.json(manifest, expanded=False)
        except Exception as exc:
            st.error(f"整季导出失败：{exc}")


def _show_series_console(series: dict[str, Any], api_key: str) -> None:
    """Persist, review and operate a V4 season only through series_service."""
    st.divider()
    st.header("2 · V4 整季圣经与连续大纲审核")
    bible = series.get("series_bible") or {}
    series_id = str(bible.get("series_id") or "").strip()
    snapshot = st.session_state.get("last_series_snapshot") or {}
    if series_id and snapshot.get("series_id") != series_id:
        snapshot = _series_snapshot(series_id)
    if snapshot:
        series = merge_series_backend_assets(series, snapshot)
        row = snapshot.get("series") if isinstance(snapshot.get("series"), dict) else {}
        series["season_approved"] = row.get("status") == "approved"
        st.session_state["series_contract"] = series
    backend_counts = snapshot.get("counts") if isinstance(snapshot.get("counts"), dict) else {}
    counts = series_episode_counts(series)
    metrics = st.columns(4)
    metrics[0].metric("总集数", counts["total"])
    metrics[1].metric("已生成 V3", counts["generated"])
    metrics[2].metric("已批准", counts["approved"])
    metrics[3].metric("每集秒数", f"{float(series.get('seconds_per_episode') or 0):g}")
    errors = validate_series_contract(series)
    if errors:
        st.error(f"V4 合同有 {len(errors)} 条连续性/结构错误，不能派生新集。")
        with st.expander("查看 V4 错误", expanded=True):
            for error in errors:
                st.write(f"- {error}")

    st.markdown(f"### {bible.get('title') or '未命名整季'}")
    st.caption(f"series_id={series_id or '缺失'} · 后端状态={((snapshot.get('series') or {}).get('status') if snapshot else '未持久化')}")
    st.write(bible.get("premise") or "")
    st.info(f"全季弧线：{bible.get('season_arc') or '-'}")
    with st.expander("不可变系列事实 / 世界规则", expanded=False):
        st.json({
            "series_bible": bible,
            "world_bible": series.get("world_bible") or {},
            "visual_bible": series.get("visual_bible") or {},
        }, expanded=False)

    st.subheader("全季共享人物")
    characters = series.get("shared_character_bible") or []
    char_cols = st.columns(max(1, min(3, len(characters))))
    for index, character in enumerate(characters):
        with char_cols[index % len(char_cols)]:
            with st.container(border=True):
                st.markdown(f"**{character.get('name') or character.get('character_id')}**")
                st.caption(str(character.get("character_id") or ""))
                st.write(character.get("editorial_identity_description") or "")
                st.code(", ".join(character.get("model_identity_tags_en") or []), language="text")
                st.code(", ".join(character.get("model_wardrobe_tags_en") or []), language="text")
                _render_media(existing_media_paths(character.get("reference_images") or [], [PROJECTS_DIR, COMFYUI_INPUT]), max_items=4)
                st.json(character.get("voice_profile") or {}, expanded=False)

    st.subheader("全季共享场景")
    scenes = series.get("shared_scene_bible") or []
    scene_cols = st.columns(max(1, min(3, len(scenes))))
    for index, scene in enumerate(scenes):
        with scene_cols[index % len(scene_cols)]:
            with st.container(border=True):
                st.markdown(f"**{scene.get('name') or scene.get('scene_id')}**")
                st.caption(str(scene.get("scene_id") or ""))
                st.write(scene.get("description") or "")
                st.code(scene.get("model_prompt_en") or "", language="text")
                _render_media(existing_media_paths(scene.get("reference_images") or [], [PROJECTS_DIR, COMFYUI_INPUT]), max_items=4)

    season_approved = bool(series.get("season_approved"))
    approve_series = _series_service("approve_series")
    if st.button(
        "全季大纲与共享圣经已由后端批准" if season_approved else "批准全季大纲与共享圣经",
        type="primary",
        disabled=season_approved or bool(errors) or approve_series is None,
        use_container_width=True,
    ):
        try:
            persisted, prepared = _persist_series_contract(series)
            contract_hash = str((prepared.get("series") or {}).get("contract_hash") or "")
            approved_snapshot = approve_series(series_id, expected_hash=contract_hash)
            _remember_series_snapshot(approved_snapshot, persisted)
            st.rerun()
        except Exception as exc:
            st.error(f"全季批准失败：{exc}")
    if season_approved:
        st.info("结构性编辑会自动把后端状态退回 draft；后端不提供伪撤回按钮。")
    else:
        st.warning("先批准共享人物、世界、场景与整季大纲，之后才能生成逐集 V3 合同。")

    st.subheader("逐集连续大纲与 V3 审核卡")
    contracts = series.get("episode_contracts") or {}
    approvals = series.get("episode_approvals") or {}
    registered_count = int(backend_counts.get("registered") or 0)
    for item in series.get("season_outline") or []:
        episode_id = str(item.get("episode_id"))
        contract = contracts.get(episode_id)
        approved = bool(approvals.get(episode_id))
        label = f"{int(item.get('episode_index') or 0):02d} · {item.get('title')} · {'已批准' if approved else '已生成' if contract else '待生成'}"
        with st.expander(label, expanded=item.get("episode_index") == 1):
            st.write(item.get("logline") or "")
            st.caption(f"{float(item.get('duration_seconds') or 0):g} 秒 · {item.get('shot_count')} 镜 · {episode_id}")
            source_total = float(item.get("shot_count") or 0) * 10.125
            st.caption(
                f"最终成片 {float(item.get('duration_seconds') or 0):g}s / "
                f"预计源素材 {source_total:g}s / GPU 镜数 {item.get('shot_count')}"
            )
            state_cols = st.columns(2)
            state_cols[0].markdown("**继承 state_in**")
            state_cols[0].json(item.get("continuity_state_in") or {}, expanded=False)
            state_cols[1].markdown("**交付 state_out**")
            state_cols[1].json(item.get("continuity_state_out") or {}, expanded=False)
            st.markdown("**因果节拍**")
            st.json(item.get("beats") or [], expanded=False)
            if item.get("wardrobe_change_events"):
                st.warning("本集含显式换装事件")
                st.json(item["wardrobe_change_events"], expanded=False)
            if item.get("time_jump_event"):
                st.warning("本集含显式时间跳跃")
                st.json(item["time_jump_event"], expanded=False)

            with st.expander("编辑本集大纲（状态边界/时长/镜数锁定）", expanded=False):
                edited = st.text_area(
                    "大纲 JSON", value=json.dumps(item, ensure_ascii=False, indent=2),
                    height=260, key=f"series_outline_edit_{episode_id}",
                )
                if st.button("应用本集大纲编辑", key=f"series_outline_apply_{episode_id}"):
                    try:
                        updated = update_series_outline_episode(series, episode_id, json.loads(edited))
                        updated["season_approved"] = False
                        _persist_series_contract(updated)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"大纲编辑失败：{exc}")

            instruction = st.text_input(
                "本集生成 / 重生要求",
                placeholder="只能优化本集节奏、表演与镜头，不得改共享事实或前后状态",
                key=f"series_episode_instruction_{episode_id}",
            )
            action_cols = st.columns(2)
            if action_cols[0].button(
                "重生本集 V3" if contract else "生成本集 V3",
                key=f"series_generate_{episode_id}",
                disabled=not api_key or not season_approved or bool(errors) or registered_count > 0,
                use_container_width=True,
            ):
                try:
                    with st.spinner(f"MiniMax 正在锁定全季事实生成 {episode_id}…"):
                        updated = generate_series_episode(series, episode_id, instruction=instruction, api_key=api_key)
                    _persist_series_contract(updated)
                    st.rerun()
                except Exception as exc:
                    st.error(f"逐集生成失败：{exc}")
            if action_cols[1].button(
                "已批准（注册后锁定）" if approved and registered_count else "撤回本集批准" if approved else "批准本集",
                key=f"series_approve_{episode_id}",
                disabled=contract is None or bool(approved and registered_count),
                use_container_width=True,
            ):
                try:
                    updated = with_series_episode_approval(series, episode_id, not approved)
                    persisted, latest = _persist_series_contract(updated)
                    updated_counts = series_episode_counts(persisted)
                    if (
                        not approved
                        and updated_counts["generated"] == updated_counts["total"]
                        and updated_counts["approved"] == updated_counts["total"]
                    ):
                        registered = register_series_episodes_via_facade(series_service, persisted)
                        _remember_series_snapshot(registered, persisted)
                    else:
                        _remember_series_snapshot(latest, persisted)
                    st.rerun()
                except Exception as exc:
                    st.error(f"本集批准 / 注册失败：{exc}")
            if contract:
                with st.expander("查看该集完整 V3 合同 / 实际 Prompt", expanded=False):
                    st.json(contract, expanded=False)

    counts = series_episode_counts(series)
    register_handler = _series_service("register_episodes")
    can_register = (
        season_approved
        and counts["total"] > 0
        and counts["generated"] == counts["total"]
        and counts["approved"] == counts["total"]
        and registered_count != counts["total"]
    )
    if st.button(
        f"注册 exact {counts['total']} 集到整季生产后端",
        key=f"series_register_all_{series_id}",
        disabled=not can_register or register_handler is None,
        use_container_width=True,
    ):
        try:
            registered = register_series_episodes_via_facade(series_service, series)
            _remember_series_snapshot(registered, series)
            st.rerun()
        except Exception as exc:
            st.error(f"整季逐集注册失败：{exc}")

    if snapshot:
        _shared_asset_console(series_id, snapshot, season_approved)
        st.divider()
        st.header("4 · 整季生产、连续性状态与交付")
        _live_series_console(series_id)
    else:
        st.error("整季尚未写入 series_service；不能生成共享资产或启动视频。")


_init_state()
st.title("🎬 AI 漫剧工厂")
st.caption("创作简报 → 创作审批 → 合同注册 → 人物/场景资产审批 → 视频生产 → 平台交付")
flash_success = st.session_state.pop("flash_success", "")
if flash_success:
    st.success(flash_success)
stage2_resume_flash = st.session_state.pop("stage2_resume_flash", "")
if stage2_resume_flash:
    st.warning(stage2_resume_flash)

with st.sidebar:
    st.header("渲染设置")
    api_key = st.text_input("MiniMax API Key", type="password", help="仅用于当前页面的生成或单项重生")
    minimax_config = minimax_configuration_status()
    st.caption(
        f"MiniMax 协议：{minimax_config['protocol']} · {minimax_config['endpoint']}。"
        "生产默认 Anthropic Messages；V3 两阶段使用强制工具提交结构化合同。"
    )
    if minimax_config["deprecated"]:
        for config_warning in minimax_config["warnings"]:
            st.warning(f"MiniMax 配置已弃用：{config_warning}")
    prompt_mode = st.radio("提示模式", ["cinematic", "comic"], horizontal=True)
    use_lora = st.toggle("Turbo LoRA", value=True)
    lora_strength = st.slider("LoRA strength", 0.0, 1.5, 1.0, 0.05)
    sage_label = st.selectbox("Sage Attention", list(SAGE_MODE_MAP))
    ref_label = st.selectbox("参考图策略", list(REF_SIZE_MAP))
    production_strategy_label = st.selectbox(
        "视频生产策略",
        list(PRODUCTION_STRATEGY_MAP),
        help=(
            "推荐策略先生成约5.17秒/0.4MP/6步的不可交付预演；"
            "只有自动QA和人工审核都通过，才晋级约10.13秒/0.9MP/8步正式素材。"
        ),
    )
    if PRODUCTION_STRATEGY_MAP[production_strategy_label] == PROOF_THEN_PRODUCTION:
        st.caption("双阶段策略自动使用预演 `match`、正式 `max`；上方参考图策略仅供“直接正式生产”专家模式使用。")
    audio_labels = {"auto_contextual": "自动（MiniMax 分镜逐镜选择，编译器兜底）"}
    background_music = st.selectbox(
        "背景音乐", list(MUSIC_PRESETS), index=list(MUSIC_PRESETS).index("auto_contextual"),
        format_func=lambda value: audio_labels.get(value, value),
    )
    ambience = st.selectbox(
        "环境音", list(AMBIENCE_PRESETS), index=list(AMBIENCE_PRESETS).index("auto_contextual"),
        format_func=lambda value: audio_labels.get(value, value),
    )

st.header("1 · 创作简报")
pending_project_id = st.session_state.get("pending_project_id") or ""
if pending_project_id:
    st.session_state["project_id_input"] = pending_project_id
    st.session_state["ep_id"] = pending_project_id
    st.session_state["pending_project_id"] = ""
ep_id = st.text_input("项目 ID", key="project_id_input").strip() or st.session_state["ep_id"]
st.session_state["ep_id"] = ep_id
if st.session_state.get("loaded_ep_id") != ep_id:
    series_snapshot = _series_snapshot(ep_id) if series_service is not None else {}
    if series_snapshot:
        restored_series = _remember_series_snapshot(series_snapshot)
        st.session_state["loaded_ep_id"] = ep_id
        st.session_state["last_snapshot"] = {}
        _set_episode({}, persisted=False, dirty=False)
        snapshot = {}
        st.session_state["series_contract"] = restored_series
    else:
        snapshot = _snapshot(ep_id) if render_service is not None else {}
        loaded = merge_episode_asset_review_state(
            snapshot_episode(snapshot), {}, snapshot,
        )
        st.session_state["loaded_ep_id"] = ep_id
        st.session_state["loaded_series_id"] = ""
        st.session_state["last_series_snapshot"] = {}
        st.session_state["series_contract"] = {}
        st.session_state["last_snapshot"] = snapshot
        _set_episode(loaded, persisted=bool(loaded), dirty=False)
        pipeline_state = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), dict) else {}
        st.session_state["contract_approved"] = pipeline_state.get("contract_status") == "approved"
        st.session_state["assets_approved"] = pipeline_state.get("assets_status") == "approved"
else:
    snapshot = st.session_state.get("last_snapshot") or {}

creation_scope = st.radio("创作范围", ["单集 V3", "整季 V4"], horizontal=True)
if st.session_state.get("series_contract"):
    restored_id = str(
        ((st.session_state["series_contract"].get("series_bible") or {}).get("series_id"))
        or ep_id
    )
    st.caption(
        f"已恢复整季项目 {restored_id}。下方 V4 控制台是当前权威合同；"
        "上方简报仅用于创建新版本，除非再次点击生成，否则不会覆盖已恢复项目。"
    )
brief_cols = st.columns(2)
with brief_cols[0]:
    topic = st.text_input("主题", placeholder="一句话说明作品要讨论什么")
    synopsis = st.text_area("故事梗概", height=180, placeholder="写清人物、目标、阻碍、升级与结局")
    style_name = st.selectbox("风格", [*STYLE_PRESETS, "自定义"])
    style_enforcement = st.text_area("自定义风格", height=80) if style_name == "自定义" else STYLE_PRESETS[style_name]
with brief_cols[1]:
    target_audience = st.selectbox("目标受众", ["全年龄", "青少年 13+", "年轻成人 18-35", "家庭观众", "成熟观众 18+"])
    if creation_scope == "整季 V4":
        episode_count = st.number_input("总集数", min_value=2, max_value=100, value=12, step=1)
        seconds_per_episode = st.number_input("每集秒数", min_value=4, max_value=900, value=60, step=5)
        auto_shots = st.toggle("每集镜数由短剧剪辑密度自动规划", value=True)
        shots_per_episode = None if auto_shots else int(st.number_input(
            "每集镜数（高级）", min_value=1, max_value=400,
            value=auto_episode_shot_count(float(seconds_per_episode)), step=1
        ))
        preview_shots = shots_per_episode or auto_episode_shot_count(float(seconds_per_episode))
        total_duration = float(seconds_per_episode)
        shot_count = int(preview_shots)
    else:
        episode_count = 1
        total_duration = st.number_input("最终成片总时长（秒）", min_value=8, max_value=600, value=60, step=5)
        auto_shots = st.toggle("镜数由短剧剪辑密度自动规划", value=True)
        shot_count = (
            auto_episode_shot_count(float(total_duration))
            if auto_shots else int(st.number_input(
                "镜数（高级）", min_value=1, max_value=400,
                value=auto_episode_shot_count(float(total_duration)), step=1,
            ))
        )
        seconds_per_episode = float(total_duration)
        shots_per_episode = int(shot_count)
    language_label = st.selectbox("语言", list(LANGUAGE_MAP))
    platform = st.selectbox("平台", list(PLATFORM_DEFAULTS))
    aspect_ratio = st.selectbox("平台画幅", ["9:16", "16:9", "1:1"], index=["9:16", "16:9", "1:1"].index(PLATFORM_DEFAULTS[platform]))

shot_bounds = shot_count_bounds(float(total_duration))
shot_density_valid = shot_bounds["minimum"] <= int(shot_count) <= shot_bounds["maximum"]
cost = shot_plan_cost_summary(float(total_duration), int(shot_count))
if not shot_density_valid:
    st.error(
        f"当前 {float(total_duration):g} 秒成片必须使用 {shot_bounds['minimum']}-"
        f"{shot_bounds['maximum']} 镜，才能保证每镜剪辑 1.5-4.0 秒；低镜数会退化成静态长镜头。"
    )
cost_cols = st.columns(4)
cost_cols[0].metric("最终成片", f"{float(total_duration):g}s")
cost_cols[1].metric("H3 源素材", f"{cost['total_source_generation_duration_seconds']:g}s")
cost_cols[2].metric("预计 GPU 镜数", str(cost["gpu_generation_jobs"]))
cost_cols[3].metric("素材/成片比", f"{cost['source_to_edit_ratio']:.1f}x")
profile_cost = profile_cost_summary()
if PRODUCTION_STRATEGY_MAP[production_strategy_label] == PROOF_THEN_PRODUCTION:
    proof_profile = profile_cost["proof"]
    formal_profile = profile_cost["production"]
    st.info(
        f"低成本预演：{proof_profile['duration_seconds']:.3f}s · "
        f"{proof_profile['megapixels']:.1f}MP · Turbo {proof_profile['turbo_steps']}步 · match；"
        f"粗略算量约为正式档 {profile_cost['proof_relative_compute']:.0%}。"
        f"正式档：{formal_profile['duration_seconds']:.3f}s · "
        f"{formal_profile['megapixels']:.1f}MP · Turbo {formal_profile['turbo_steps']}步 · max。"
    )
    st.caption("预演只用于提示词、动作、身份、镜头和音画检查，永远不能合片或导出。")
else:
    st.warning("已选择跳过预演。正式渲染成本和失败返工风险更高，仍不会绕过内容 QA 与人工发布门。")
st.caption(
    "每镜 H3 固定生成 10.125 秒源素材，后端自动选取 1.5-4.0 秒进入成片。镜数即主要 GPU 作业数；"
    "实际成本取决于显卡速度、失败重试与质量门禁。"
)
seconds_per_shot = float(total_duration) / int(shot_count)
language_code, voice_language = LANGUAGE_MAP[language_label]
generate_col, demo_col = st.columns(2)
generate_clicked = generate_col.button(
    "生成 V4 整季圣经与连续大纲" if creation_scope == "整季 V4" else "生成 V3 故事 / 人物 / 场景 / 分镜合同",
    type="primary",
    disabled=not topic.strip() or not synopsis.strip() or not shot_density_valid,
    use_container_width=True,
)
demo_clicked = demo_col.button(
    "加载明确标注的单集 DEMO",
    disabled=creation_scope == "整季 V4",
    use_container_width=True,
)

settings = dict(
    topic=topic,
    synopsis=synopsis,
    target_audience=target_audience,
    total_duration_seconds=float(total_duration),
    shot_count=int(shot_count),
    platform=platform,
    min_panels=int(shot_count),
    max_panels=int(shot_count),
    prompt_mode=prompt_mode,
    style=prompt_mode,
    visual_style=style_name,
    style_enforcement=style_enforcement,
    aspect_ratio=aspect_ratio,
    language=language_code,
    voice_language=voice_language,
    duration_seconds=10.125,
    use_lora=use_lora,
    lora_strength=float(lora_strength),
    sage_mode=SAGE_MODE_MAP[sage_label],
    ref_image_size=REF_SIZE_MAP[ref_label],
    background_music=background_music,
    ambience=ambience,
    production_strategy=PRODUCTION_STRATEGY_MAP[production_strategy_label],
)
current_generation_signature = generation_input_signature(ep_id, synopsis, settings)

if int(episode_count) == 1:
    st.info(
        "单集 V3 使用明确的两阶段生成计划：阶段 1 生成故事/人物/场景/节拍圣经，"
        "校验通过后阶段 2 生成 exact-N 镜头合同；将调用 MiniMax 2 次，两次可能分别计费。"
    )
else:
    st.caption(
        "整季 V4 当前仍使用单次生成，尚未拆分；受 2048 completion token 限制，"
        "大体量整季若返回不完整会失败且不保存，也不会自动重试。"
    )

resume_checkpoint: dict[str, Any] = {}
resume_candidates: list[dict[str, Any]] = []
resume_probe_error = ""
if int(episode_count) == 1 and ep_id and synopsis.strip():
    try:
        resume_candidates = list_stage1_checkpoints(ep_id)
        checkpoint_inputs = split_story_checkpoint_inputs(synopsis, **settings)
        resume_checkpoint = match_stage1_checkpoint(
            ep_id,
            creative_brief=checkpoint_inputs["creative_brief"],
            settings=checkpoint_inputs["settings"],
            protocol=minimax_config["protocol"],
            model=minimax_config["model"],
        )
    except Exception as exc:
        # Resume discovery is operational state, not a cosmetic detail.  Keep
        # provider text and secrets out, but surface the local exception class
        # so a user is not left with a silent dead end after a refresh/crash.
        resume_probe_error = type(exc).__name__
        resume_checkpoint = {}
if int(episode_count) == 1 and resume_probe_error:
    st.error(
        "阶段 1 草稿恢复检查失败；不会重复调用 MiniMax。"
        f"本地诊断：{resume_probe_error}。"
    )
elif int(episode_count) == 1 and resume_candidates and not resume_checkpoint:
    latest_candidate = resume_candidates[0]
    st.warning(
        "检测到本项目的阶段 1 草稿，但它与当前简报/渲染设置不完全一致，"
        "或阶段 2 仍标记为运行中/已经完成；为防止串用人物与剧情，系统不会自动续跑。"
    )
    st.caption(
        f"最近草稿 {str(latest_candidate.get('checkpoint_sha256') or '')[:12]}… · "
        f"阶段 2={latest_candidate.get('stage2_status') or 'unknown'} · "
        f"尝试 {int(latest_candidate.get('stage2_attempt_count') or 0)} 次。"
    )
if int(episode_count) == 1 and resume_checkpoint:
    resume_gate = stage2_resume_eligibility(
        resume_checkpoint,
        saved_ep_id=str(resume_checkpoint.get("ep_id") or ""),
        current_ep_id=ep_id,
        protocol=minimax_config["protocol"],
        model=minimax_config["model"],
    )
    if resume_gate["ready"]:
        st.warning(
            "阶段 1 已保存，阶段 2 未完成。该草稿未注册、未批准，也不是可生产合同。"
        )
        st.caption(
            f"草稿 {str(resume_checkpoint.get('checkpoint_sha256') or '')[:12]}… · "
            f"阶段 2 已尝试 {int(resume_checkpoint.get('stage2_attempt_count') or 0)} 次。"
        )
    else:
        st.error(f"阶段 2 续跑已禁用：{resume_gate['reason']}。请按当前输入重新执行完整两阶段生成。")
    if st.button(
        "只重试阶段 2（将只调用 MiniMax 1 次，可能计费）",
        key=f"resume_stage2_{ep_id}",
        disabled=not resume_gate["ready"],
        use_container_width=True,
    ):
        stage_notice = st.empty()

        def show_resume_stage(_stage, message):
            stage_notice.info(message)

        try:
            with st.spinner("正在从已校验的阶段 1 草稿生成 exact-N 镜头；不会再次调用阶段 1…"):
                generated = resume_stage2_via_facade(
                    resume_story_stage2,
                    synopsis,
                    ep_id=ep_id,
                    checkpoint_hash=str(resume_checkpoint["checkpoint_sha256"]),
                    settings=settings,
                    api_key=api_key or None,
                    progress_cb=show_resume_stage,
                )
            st.session_state["series_contract"] = {}
            _set_episode(generated, persisted=False)
            st.session_state["stage2_resume_checkpoint"] = {}
            st.session_state["stage2_resume_ep_id"] = ""
            st.session_state["stage2_resume_input_signature"] = ""
            st.success("阶段 2 已完成，完整合同尚未保存或注册。请完成三项创作审批。")
        except Exception as exc:
            st.error(f"阶段 2 重试失败：{exc}")
            st.info("阶段 1 草稿仍保留；系统不会自动发起下一次付费请求。修正问题后可再次人工确认重试。")

if generate_clicked:
    request_started_at = time.time()
    request_timeout = minimax_request_timeout_seconds()
    planned_calls = 2 if int(episode_count) == 1 else 1
    wait_notice = generation_wait_notice(
        request_started_at, request_timeout, planned_calls=planned_calls,
    )
    st.info(
        f"{wait_notice['headline']} · 开始时间 {wait_notice['started_at']}。"
        "正在生成故事圣经、人物/场景圣经和短剧镜头合同。"
    )
    st.warning(wait_notice["stop_help"])
    st.caption(wait_notice["failure_help"])
    checkpoints_before: list[dict[str, Any]] = []
    if creation_scope == "单集 V3":
        try:
            checkpoints_before = list_stage1_checkpoints(ep_id)
        except Exception:
            checkpoints_before = []
    try:
        if creation_scope == "整季 V4":
            with st.spinner("MiniMax 剧集总编剧正在建立全季 V4 圣经与连续大纲…"):
                generated_series = split_series(
                    topic=topic,
                    synopsis=synopsis,
                    episode_count=int(episode_count),
                    seconds_per_episode=float(seconds_per_episode),
                    shots_per_episode=shots_per_episode,
                    target_audience=target_audience,
                    platform=platform,
                    visual_style=style_name,
                    style_enforcement=style_enforcement,
                    aspect_ratio=aspect_ratio,
                    language=language_code,
                    voice_language=voice_language,
                    prompt_mode=prompt_mode,
                    use_lora=use_lora,
                    lora_strength=float(lora_strength),
                    sage_mode=SAGE_MODE_MAP[sage_label],
                    ref_image_size=REF_SIZE_MAP[ref_label],
                    background_music=background_music,
                    ambience=ambience,
                    api_key=api_key or None,
                )
            persisted_series, persisted_snapshot = _persist_series_contract(generated_series)
            series_id = str((persisted_series.get("series_bible") or {}).get("series_id") or "")
            st.session_state["series_contract"] = persisted_series
            st.session_state["last_series_snapshot"] = persisted_snapshot
            st.session_state["loaded_series_id"] = series_id
            st.session_state["pending_project_id"] = series_id
            _set_episode({}, persisted=False)
            st.success("V4 全季合同已生成并持久化。先审核共享圣经和连续大纲，再逐集生成 V3。")
        else:
            stage_notice = st.empty()
            def show_generation_stage(_stage, message):
                stage_notice.info(message)
            with st.spinner("MiniMax 两阶段 V3 生成正在执行；当前阶段见下方状态…"):
                generated = split_story(
                    synopsis, api_key=api_key or None, demo_mode=False,
                    progress_cb=show_generation_stage, ep_id=ep_id, **settings,
                )
            st.session_state["series_contract"] = {}
            _set_episode(generated, persisted=False)
            st.session_state["stage2_resume_checkpoint"] = {}
            st.session_state["stage2_resume_ep_id"] = ""
            st.session_state["stage2_resume_input_signature"] = ""
            st.success("合同已生成，尚未保存或注册。请完成三项创作审批。")
    except MissingMiniMaxAPIKey as exc:
        st.error(str(exc))
    except MiniMaxRequestTimeout as exc:
        st.error(str(exc))
        st.info("主题、故事梗概、时长、镜数、语言和风格输入均保留在当前页面；确认后可再次点击生成。")
    except Exception as exc:
        st.error(f"生成失败：{exc}")
        resumable = {}
        if creation_scope == "单集 V3":
            try:
                checkpoint_inputs = split_story_checkpoint_inputs(synopsis, **settings)
                resumable = match_stage1_checkpoint(
                    ep_id,
                    creative_brief=checkpoint_inputs["creative_brief"],
                    settings=checkpoint_inputs["settings"],
                    protocol=minimax_config["protocol"],
                    model=minimax_config["model"],
                )
            except Exception:
                resumable = {}
        if resumable:
            st.session_state["stage2_resume_checkpoint"] = resumable
            st.session_state["stage2_resume_ep_id"] = ep_id
            st.session_state["stage2_resume_input_signature"] = current_generation_signature
            st.session_state["stage2_resume_flash"] = (
                "阶段 1 已保存，阶段 2 未完成；最终合同没有保存或注册。"
                "可在当前输入保持不变时使用“只重试阶段 2”按钮。上次错误："
                f"{exc}"
            )
            st.rerun()
        else:
            st.info(
                "未生成的合同没有被保存或注册；当前表单输入仍保留。"
                "为避免重复计费，系统不会自动重试，请检查错误后再手动点击生成。"
            )

if demo_clicked:
    generated = split_story("", demo_mode=True, **settings)
    st.session_state["series_contract"] = {}
    _set_episode(generated, persisted=False)
    st.warning(generated.get("demo_notice"))

episode = st.session_state.get("episode") or {}
if st.session_state.get("contract_persisted") and not st.session_state.get("local_dirty"):
    latest = _snapshot(ep_id) if render_service is not None else snapshot
    st.session_state["last_snapshot"] = latest
    persisted_episode = merge_episode_asset_review_state(
        snapshot_episode(latest),
        st.session_state.get("episode") or episode,
        latest,
    )
    if persisted_episode:
        episode = persisted_episode
        st.session_state["episode"] = episode
        pipeline_state = latest.get("pipeline") if isinstance(latest.get("pipeline"), dict) else {}
        st.session_state["contract_approved"] = pipeline_state.get("contract_status") == "approved"
        st.session_state["assets_approved"] = pipeline_state.get("assets_status") == "approved"
snapshot = st.session_state.get("last_snapshot") or snapshot
series_contract = st.session_state.get("series_contract") or {}

stage = derive_project_stage(
    episode,
    snapshot,
    contract_persisted=st.session_state.get("contract_approved", False),
    production_started=st.session_state.get("production_started", False),
)
if episode or not series_contract:
    st.info(f"当前项目阶段：{STAGE_LABELS.get(stage, stage)}")
if episode.get("is_demo"):
    st.error("DEMO DATA：仅用于检查界面和合同，禁止保存为真实项目或启动生产。")

if series_contract and not episode:
    _show_series_console(series_contract, api_key)
elif episode:
    st.divider()
    st.header("2 · 创作审核与门禁一")
    _show_story_review(episode, api_key)
    _show_character_cards(episode, snapshot, api_key)
    _show_scene_cards(episode, api_key)
    _show_storyboard(episode, snapshot, api_key)
    _creative_gate(ep_id, episode)

    st.divider()
    st.header("3 · 资产审核与门禁二")
    _asset_gate(ep_id, episode)

    st.divider()
    st.header("4 · 视频任务")
    _live_jobs(ep_id, episode)

    st.divider()
    st.header("5 · 平台交付")
    _delivery_panel(ep_id)
else:
    st.info("填写创作简报并生成单集 V3 或整季 V4 合同后，才会进入审核与生产流程。")
