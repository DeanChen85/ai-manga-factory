# -*- coding: utf-8 -*-
"""Episode orchestration shared by the CLI worker and Streamlit.

The durable task store is the source of truth.  Preparing an episode never
touches the GPU; character assets and H3 panels are produced only by a worker.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from generation_log import append_record, update_status
from atomic_io import write_json_atomic as _write_json_atomic
from h3_profiles import DEFAULT_PRODUCTION_STRATEGY, apply_render_profile
from render_video_h3 import (
    H3_LORA_ENABLED_DEFAULT,
    H3_LORA_STRENGTH,
    recover_render_job,
    release_comfy_resources,
    submit_render_job,
    wait_render_job,
)
from runtime_config import comfyui_root, ffmpeg_executable, projects_dir
from shot_group_anchor import (
    panel_anchor_contract_sha256,
    requires_approved_group_anchor,
    requires_paired_state_anchor,
)
from story_splitter import COMIC_EXAMPLE_HERO_KAIJU, split_story, validate_panels
from task_store import (
    RenderJobStore,
    default_store,
    prepare_episode,
    production_gate,
    project_snapshot,
    resume_jobs,
)
from video_delivery import export_episode


PROJECTS_DIR = projects_dir()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ep_id(ep_id: str) -> str:
    value = str(ep_id).strip()
    if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("ep_id must contain only letters, digits, dot, underscore or hyphen")
    return value


def _episode_path(ep_id: str) -> Path:
    return PROJECTS_DIR / _safe_ep_id(ep_id) / "episode.json"


def _load_episode(ep_id: str) -> dict[str, Any]:
    path = _episode_path(ep_id)
    if not path.exists():
        raise FileNotFoundError(f"episode.json missing: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"episode.json must contain an object: {path}")
    return result


def split_story_to_episode(
    story_text: str,
    *,
    ep_id: str,
    style: str = "comic",
    min_panels: int = 4,
    max_panels: int = 8,
    use_lora: bool = H3_LORA_ENABLED_DEFAULT,
    lora_strength: float = H3_LORA_STRENGTH,
    aspect_ratio: str = "16:9",
    language: str = "cn",
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    """Split a story and atomically register the complete episode job set."""
    ep_id = _safe_ep_id(ep_id)
    if progress_cb:
        progress_cb("split", f"M3 拆分中（{min_panels}-{max_panels} panels）")
    result = split_story(
        story_text,
        style=style,
        min_panels=min_panels,
        max_panels=max_panels,
        use_lora=use_lora,
        lora_strength=lora_strength,
        aspect_ratio=aspect_ratio,
        language=language,
        progress_cb=progress_cb,
    )
    panels = result.get("panels") or []
    errors = validate_panels(panels)
    if errors:
        raise ValueError("storyboard validation failed: " + "; ".join(errors[:12]))
    episode = {
        **result,
        "ep_id": ep_id,
        "style": result.get("style", style),
        "aspect_ratio": result.get("aspect_ratio", aspect_ratio),
        "use_lora": use_lora,
        "lora_strength": lora_strength,
        "source_story_hash": hashlib.sha256(story_text.encode("utf-8")).hexdigest(),
        "panels": panels,
    }
    snapshot = prepare_episode(ep_id, episode)
    for index, panel in enumerate(panels, 1):
        append_record({
            "project": ep_id,
            "scene_idx": index,
            "scene_name": panel.get("name") or panel.get("panel_id") or f"panel_{index:03d}",
            "status": "pending",
            "source": "prepare_episode",
        })
    if progress_cb:
        progress_cb("split", f"已原子登记 {len(snapshot['jobs'])} 个 render jobs；尚未启动 GPU")
    return episode


def _resolve_asset_path(value: str | Path | Mapping[str, Any], project: Path) -> Optional[Path]:
    if isinstance(value, Mapping):
        value = value.get("source_path") or value.get("path") or value.get("staged_name") or ""
    if not value:
        return None
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [project / raw, comfyui_root() / "input" / raw]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _copy_character_reference(source: Path, directory: Path, char_id: str, index: int) -> Path:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    safe_char = re.sub(r"[^A-Za-z0-9_-]+", "_", char_id).strip("_") or "character"
    suffix = source.suffix.lower() or ".png"
    destination = (directory / f"{safe_char}_{index:02d}_{digest}{suffix}").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination and not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        import shutil
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    return destination


def _reference_bundle_hash(paths: Iterable[Path]) -> str:
    return hashlib.sha256(json.dumps([
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    ], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _select_character_reference_paths(
    resolved_inputs: Iterable[Mapping[str, Any]],
) -> list[Path]:
    """Use all crops for one actor, but one canonical anchor per ensemble actor."""
    items = [item for item in resolved_inputs if item.get("role") == "character_reference"]
    source_ids = list(dict.fromkeys(
        str(item.get("source_id") or "") for item in items if item.get("source_id")
    ))
    if len(source_ids) <= 1:
        return [Path(item["resolved"]) for item in items]
    selected: list[Path] = []
    for source_id in source_ids:
        first = next(
            item["resolved"] for item in items
            if str(item.get("source_id")) == source_id
        )
        selected.append(Path(first))
    return selected


def prepare_character_assets(
    ep_id: str,
    *,
    generator: Optional[Callable[..., dict[str, Any]]] = None,
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    """Generate missing character assets inside the worker, then refresh jobs.

    This function is intentionally separate from :func:`prepare_episode`; Web
    should start a worker instead of invoking this GPU stage directly.
    """
    ep_id = _safe_ep_id(ep_id)
    episode = _load_episode(ep_id)
    store = default_store()
    pipeline = store.get_pipeline(ep_id) or {}
    if pipeline.get("contract_status") != "approved":
        raise RuntimeError("contract must be approved before character asset generation")
    characters = episode.get("character_bible") or []
    if not characters:
        return prepare_episode(ep_id, episode)
    if generator is None:
        from generate_character_ref import generate_character_assets
        generator = generate_character_assets
    project = PROJECTS_DIR / ep_id
    charrefs_dir = project / "charrefs"
    visual_bible = episode.get("visual_bible") or {}
    story_hash = str(episode.get("source_story_hash") or hashlib.sha256(
        json.dumps(episode.get("story_bible") or episode, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest())

    updated: list[str] = []
    failures: list[str] = []
    asset_map = {(asset["asset_type"], asset["source_id"]): asset for asset in store.list_assets(ep_id)}
    for index, card in enumerate(characters, 1):
        if not isinstance(card, dict):
            continue
        char_id = str(card.get("character_id") or card.get("id") or f"character_{index}")
        asset = asset_map.get(("character", char_id))
        if not asset:
            raise RuntimeError(f"character asset was not registered: {char_id}")
        if asset["status"] == "succeeded" and asset.get("reference_images"):
            card["reference_images"] = list(asset["reference_images"])
            continue
        if asset["status"] == "failed" and asset["retry_count"] >= asset["max_retries"]:
            failures.append(f"{char_id}: retry limit reached")
            continue
        # A queued asset means its prompt or source changed.  Existing files in
        # the episode are intentionally not reused until regenerated/approved.
        if progress_cb:
            progress_cb("character", f"生成角色资产：{char_id}")
        callback = (lambda message: progress_cb("character", message)) if progress_cb else None
        retry_count = asset["retry_count"] + (1 if asset["status"] == "failed" else 0)
        store.update_asset(asset["asset_id"], status="running", error=None, retry_count=retry_count)
        try:
            manifest = generator(
                card,
                visual_bible,
                story_hash=story_hash,
                progress_cb=callback,
            )
            sources = []
            for value in manifest.get("reference_images") or []:
                resolved = _resolve_asset_path(value, project)
                if resolved:
                    sources.append(resolved)
            if not sources:
                raise RuntimeError(f"character generator returned no usable reference images: {char_id}")
            persisted = [
                _copy_character_reference(source, charrefs_dir, char_id, ref_index)
                for ref_index, source in enumerate(sources, 1)
            ]
            card["reference_images"] = [str(path) for path in persisted]
            card["asset_status"] = "succeeded"
            manifest_path = charrefs_dir / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', char_id)}.manifest.json"
            _write_json_atomic(manifest_path, {**manifest, "reference_images": card["reference_images"]})
            store.update_asset(
                asset["asset_id"], status="succeeded", approved=False,
                content_hash=_reference_bundle_hash(persisted),
                reference_images=card["reference_images"], manifest_path=str(manifest_path),
                prompt_id=manifest.get("prompt_id"), completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            )
            updated.append(char_id)
        except Exception as exc:
            store.update_asset(asset["asset_id"], status="failed", error=str(exc), approved=False)
            failures.append(f"{char_id}: {exc}")

    episode["character_bible"] = characters
    _write_json_atomic(_episode_path(ep_id), episode)
    snapshot = prepare_episode(ep_id, episode)
    snapshot["character_assets_updated"] = updated
    if failures:
        raise RuntimeError("character asset failures: " + "; ".join(failures))
    return snapshot


def prepare_scene_assets(
    ep_id: str,
    *,
    generator: Optional[Callable[..., dict[str, Any]]] = None,
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    """Generate missing scene plates and persist prompt/content audit data."""
    ep_id = _safe_ep_id(ep_id)
    episode = _load_episode(ep_id)
    store = default_store()
    pipeline = store.get_pipeline(ep_id) or {}
    if pipeline.get("contract_status") != "approved":
        raise RuntimeError("contract must be approved before scene asset generation")
    using_default_generator = generator is None
    structural_checkpoint: str | None = None
    if generator is None:
        from scene_asset import STRUCTURAL_SCENE_CHECKPOINT, generate_scene_asset
        generator = generate_scene_asset
        structural_model = (
            comfyui_root() / "models" / "checkpoints" / STRUCTURAL_SCENE_CHECKPOINT
        )
        if structural_model.is_file():
            structural_checkpoint = STRUCTURAL_SCENE_CHECKPOINT
    scenes = episode.get("scene_bible") or []
    project = PROJECTS_DIR / ep_id
    scene_dir = project / "scenerefs"
    visual_bible = episode.get("visual_bible") or {}
    story_hash = str(episode.get("source_story_hash") or episode.get("story_bible", {}).get("source_sha256") or "")
    asset_map = {(asset["asset_type"], asset["source_id"]): asset for asset in store.list_assets(ep_id)}
    updated: list[str] = []
    failures: list[str] = []
    for index, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("scene_id") or f"scene_{index:02d}")
        asset = asset_map.get(("scene", scene_id))
        if not asset:
            raise RuntimeError(f"scene asset was not registered: {scene_id}")
        if asset["status"] == "succeeded" and asset.get("reference_images"):
            scene["reference_images"] = list(asset["reference_images"])
            continue
        if asset["status"] == "failed" and asset["retry_count"] >= asset["max_retries"]:
            failures.append(f"{scene_id}: retry limit reached")
            continue
        callback = (lambda message: progress_cb("scene", message)) if progress_cb else None
        retry_count = asset["retry_count"] + (1 if asset["status"] == "failed" else 0)
        store.update_asset(asset["asset_id"], status="running", error=None, retry_count=retry_count)
        try:
            generator_kwargs: dict[str, Any] = {
                "story_hash": story_hash,
                "progress_cb": callback,
            }
            # Accurate geography matters more than style during scene-plate
            # approval.  When RealVisXL is installed, the built-in generator
            # uses it for the structural first pass and Animagine for a low-
            # denoise style pass.  Custom generators keep their existing
            # callable contract, and clones without the optional checkpoint
            # fall back cleanly to the single-pass graph.
            if using_default_generator and structural_checkpoint:
                generator_kwargs["structural_checkpoint"] = structural_checkpoint
            manifest = generator(scene, visual_bible, **generator_kwargs)
            sources = [
                path for value in manifest.get("reference_images") or []
                if (path := _resolve_asset_path(value, project)) is not None
            ]
            if not sources:
                raise RuntimeError(f"scene generator returned no usable reference images: {scene_id}")
            persisted = [
                _copy_character_reference(source, scene_dir, scene_id, ref_index)
                for ref_index, source in enumerate(sources, 1)
            ]
            scene["reference_images"] = [str(path) for path in persisted]
            scene["asset_status"] = "succeeded"
            manifest_path = scene_dir / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', scene_id)}.manifest.json"
            _write_json_atomic(manifest_path, {**manifest, "reference_images": scene["reference_images"]})
            store.update_asset(
                asset["asset_id"], status="succeeded", approved=False,
                content_hash=_reference_bundle_hash(persisted),
                reference_images=scene["reference_images"], manifest_path=str(manifest_path),
                prompt_id=manifest.get("prompt_id"), completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            )
            updated.append(scene_id)
        except Exception as exc:
            store.update_asset(asset["asset_id"], status="failed", error=str(exc), approved=False)
            failures.append(f"{scene_id}: {exc}")
    episode["scene_bible"] = scenes
    _write_json_atomic(_episode_path(ep_id), episode)
    snapshot = prepare_episode(ep_id, episode)
    snapshot["scene_assets_updated"] = updated
    if failures:
        raise RuntimeError("scene asset failures: " + "; ".join(failures))
    return snapshot


def prepare_all_assets(
    ep_id: str,
    *,
    character_generator: Optional[Callable[..., dict[str, Any]]] = None,
    scene_generator: Optional[Callable[..., dict[str, Any]]] = None,
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    # Character anchors and empty-scene plates are independent approval
    # assets.  Always attempt both so one failure cannot strand the other in
    # an endless ``queued`` state.
    failures: list[str] = []
    try:
        prepare_character_assets(ep_id, generator=character_generator, progress_cb=progress_cb)
    except Exception as exc:
        failures.append(str(exc))
    try:
        snapshot = prepare_scene_assets(ep_id, generator=scene_generator, progress_cb=progress_cb)
    except Exception as exc:
        failures.append(str(exc))
        snapshot = prepare_episode(ep_id, _load_episode(ep_id))
    if failures:
        raise RuntimeError("asset generation failures: " + " | ".join(failures))
    return snapshot


def _character_context(episode: Mapping[str, Any], panel: Mapping[str, Any]) -> str:
    wanted = {str(value) for value in panel.get("character_ids") or []}
    descriptions = []
    for card in episode.get("character_bible") or []:
        if not isinstance(card, Mapping):
            continue
        char_id = str(card.get("character_id") or card.get("id") or "")
        if wanted and char_id not in wanted:
            continue
        wardrobe = card.get("wardrobe_prompt") or card.get("wardrobe_lock") or ""
        if isinstance(wardrobe, Mapping):
            wardrobe = ", ".join(f"{key}: {value}" for key, value in wardrobe.items())
        descriptions.append(
            f"{char_id} ({card.get('name', char_id)}): {card.get('identity_prompt') or card.get('description') or ''}; wardrobe: {wardrobe}"
        )
    return " | ".join(descriptions)


def _worker_settings(episode: Mapping[str, Any], panel: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "aspect_ratio": episode.get("aspect_ratio", "16:9"),
        "duration_seconds": episode.get("duration_seconds", 10.0),
        "use_lora": episode.get("use_lora", H3_LORA_ENABLED_DEFAULT),
        "lora_strength": episode.get("lora_strength", H3_LORA_STRENGTH),
    }
    if isinstance(episode.get("render_settings"), Mapping):
        settings.update(episode["render_settings"])
    metadata = job.get("metadata") or {}
    if isinstance(metadata.get("settings"), Mapping):
        settings.update(metadata["settings"])
    package = panel.get("prompt_package") or {}
    if isinstance(package, Mapping) and isinstance(package.get("render_settings"), Mapping):
        settings.update(package["render_settings"])
    settings.setdefault("production_strategy", DEFAULT_PRODUCTION_STRATEGY)
    settings = apply_render_profile(settings, metadata=metadata)
    if settings.get("ref_image_size") in {"match", "max"}:
        fidelity = "identity" if settings["ref_image_size"] == "max" else "fast"
    else:
        fidelity = settings.get("reference_fidelity", "fast")
    sage = settings.get("sage_mode", settings.get("sage_attention", "auto"))
    return {
        **settings,
        "duration_seconds": float(settings.get("duration_seconds") or 10.125),
        "reference_fidelity": fidelity,
        "sage_attention": sage,
    }


def _extract_tail_frame(video: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp.png")
    subprocess.run([
        ffmpeg_executable(), "-y", "-sseof", "-0.08", "-i", str(video),
        "-frames:v", "1", "-update", "1", str(temporary),
    ], check=True, capture_output=True, text=True)
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise RuntimeError(f"failed to extract continuity tail frame: {video}")
    temporary.replace(destination)
    return destination


def _has_validated_success_artifact(job: Mapping[str, Any] | None) -> bool:
    """True only after technical validation and fail-closed content QA."""
    if not job or job.get("status") != "succeeded" or not job.get("output_path"):
        return False
    try:
        output = Path(str(job["output_path"]))
        if not output.is_file() or output.stat().st_size <= 0:
            return False
    except OSError:
        return False
    probe = job.get("probe") or {}
    video = probe.get("video") or {}
    content_qa = ((job.get("metadata") or {}).get("content_qa") or {})
    settings = (job.get("metadata") or {}).get("settings") or {}
    selection = (job.get("metadata") or {}).get("edit_selection") or {}
    selection_ready = (
        True if settings.get("edit_duration_seconds") is None
        else bool(selection.get("selection_sha256") and selection.get("source_artifact_sha256"))
    )
    return bool(
        float(probe.get("duration_seconds") or 0) > 0
        and int(video.get("width") or 0) > 0
        and int(video.get("height") or 0) > 0
        and float(video.get("fps") or 0) > 0
        and content_qa.get("passed") is True
        and selection_ready
    )


def _requires_proof_human_gate(job: Mapping[str, Any] | None) -> bool:
    """Stop the worker after each non-deliverable proof until a reviewer acts."""
    if not job or job.get("status") != "succeeded":
        return False
    metadata = job.get("metadata") if isinstance(job.get("metadata"), Mapping) else {}
    settings = metadata.get("settings") if isinstance(metadata.get("settings"), Mapping) else {}
    profile = str(metadata.get("render_profile") or settings.get("render_profile") or "").lower()
    delivery_eligible = metadata.get("delivery_eligible")
    if delivery_eligible is None:
        delivery_eligible = settings.get("delivery_eligible")
    promotion = (
        metadata.get("preview_promotion")
        if isinstance(metadata.get("preview_promotion"), Mapping) else {}
    )
    return bool(
        profile == "proof"
        and delivery_eligible is False
        and promotion.get("status") != "approved"
    )


def _preserve_success_with_pipeline_warning(
    store: RenderJobStore, job: Mapping[str, Any], error: Exception,
) -> dict[str, Any]:
    """Record a post-render warning without rolling back a validated clip."""
    metadata = dict(job.get("metadata") or {})
    warning = {
        "stage": "post_render_episode_refresh",
        "error": str(error),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_preserved": True,
    }
    metadata["pipeline_warnings"] = [
        *(metadata.get("pipeline_warnings") or []), warning,
    ][-50:]
    store.update_job(
        str(job["job_id"]), status="succeeded", progress=1.0,
        error=None, metadata=metadata,
    )
    return warning


def run_episode_jobs(
    ep_id: str,
    *,
    statuses: Iterable[str] = ("pending", "failed"),
    ensure_character_assets: bool = True,
    character_generator: Optional[Callable[..., dict[str, Any]]] = None,
    timeout: float = 2400.0,
    poll_interval: float = 5.0,
    progress_cb: Optional[Callable[[str, str], None]] = None,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    """Resume/recover/submit eligible jobs; intended to run in a worker.

    ``max_jobs`` bounds jobs consumed by this invocation.  The default remains
    the historical whole-episode behavior; the night scheduler uses one so it
    can re-check resource and time gates between shots.
    """
    if max_jobs is not None and int(max_jobs) <= 0:
        raise ValueError("max_jobs must be positive when provided")
    ep_id = _safe_ep_id(ep_id)
    store = default_store()
    episode = _load_episode(ep_id)

    def callback(phase: str, message: str) -> None:
        store.heartbeat_worker(ep_id)
        if progress_cb:
            progress_cb(phase, message)

    del ensure_character_assets, character_generator
    prepare_episode(ep_id, episode)
    gate = production_gate(ep_id)
    if not gate["ready"]:
        raise RuntimeError("production gate blocked: " + ",".join(gate["reasons"]))
    resume_summary = resume_jobs(ep_id, statuses=statuses)
    panels = episode.get("panels") or []
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, Any]] = []
    jobs_consumed = 0

    def release_before_next_shot(job: Mapping[str, Any]) -> bool:
        release_result = release_comfy_resources()
        if release_result.get("released"):
            return True
        warning = {
            "job_id": str(job.get("job_id") or ""),
            "disposition": "stopped_before_next_shot_resource_release_failed",
            "error": str(release_result.get("reason") or "unknown resource release failure"),
        }
        warnings.append(warning)
        callback("warning", f"{job.get('panel_name')}: {warning['error']}")
        return False

    job_ids = [job["job_id"] for job in store.list_jobs(ep_id)]
    for current_job_id in job_ids:
        job = store.get_job(current_job_id)
        if not job:
            continue
        index = int(job["panel_index"])
        if index < 1 or index > len(panels):
            store.update_job(job["job_id"], status="failed", error="panel index is outside episode.panels")
            failures.append({"job_id": job["job_id"], "error": "invalid panel index"})
            continue
        panel = dict(panels[index - 1])
        if job["status"] == "succeeded" and job.get("output_path") and Path(job["output_path"]).exists():
            continue
        if job["status"] not in {"queued", "submitted", "running"}:
            continue
        if max_jobs is not None and jobs_consumed >= int(max_jobs):
            break
        jobs_consumed += 1
        try:
            current_gate = production_gate(ep_id)
            if not current_gate["ready"]:
                raise RuntimeError("production gate changed while worker was running: " + ",".join(current_gate["reasons"]))
            if job.get("prompt_id"):
                recovered = recover_render_job(job["job_id"], store=store)
                if recovered.get("status") == "succeeded":
                    update_status(ep_id, index, "finalized", final_path=recovered.get("output_path"))
                    prepare_episode(ep_id, _load_episode(ep_id))
                    if not release_before_next_shot(job):
                        break
                    finished = store.get_job(job["job_id"])
                    if _requires_proof_human_gate(finished):
                        callback("review", f"{job.get('panel_name')}: proof 已完成，等待人工验收后再继续")
                        break
                    continue
                if recovered.get("status") == "failed":
                    raise RuntimeError(
                        "persisted Comfy prompt failed; reconcile_job must explicitly "
                        "authorize retry before prompt_id can be cleared"
                    )
                else:
                    wait_render_job(
                        job["job_id"], store=store, timeout=timeout,
                        poll_interval=poll_interval, progress_cb=callback,
                    )
                    update_status(ep_id, index, "finalized", final_path=job.get("output_path"))
                    prepare_episode(ep_id, _load_episode(ep_id))
                    if not release_before_next_shot(job):
                        break
                    finished = store.get_job(job["job_id"])
                    if _requires_proof_human_gate(finished):
                        callback("review", f"{job.get('panel_name')}: proof 已完成，等待人工验收后再继续")
                        break
                    continue

            job = store.get_job(job["job_id"]) or job
            metadata = job.get("metadata") or {}
            inputs = metadata.get("inputs") or {}
            project = PROJECTS_DIR / ep_id
            current_assets = {asset["asset_id"]: asset for asset in store.list_assets(ep_id)}
            for dependency in inputs.get("asset_dependencies") or []:
                asset = current_assets.get(dependency.get("asset_id"))
                if not asset or not asset["approved"] or asset["status"] != "succeeded":
                    raise RuntimeError(f"asset dependency is not approved: {dependency.get('asset_id')}")
                if asset.get("content_hash") != dependency.get("content_hash"):
                    raise RuntimeError(f"asset dependency hash changed: {asset['asset_id']}")
            resolved_inputs = []
            for item in inputs.get("reference_inputs") or []:
                resolved = _resolve_asset_path(item, project)
                if resolved:
                    resolved_inputs.append({**item, "resolved": resolved})
            reference_source_ids = {
                str(Path(item["resolved"]).resolve()): str(item.get("source_id") or "")
                for item in resolved_inputs if item.get("source_id")
            }
            character_refs = _select_character_reference_paths(resolved_inputs)
            scene_refs = [item["resolved"] for item in resolved_inputs if item.get("role") == "scene_reference"]
            if panel.get("character_ids") and not character_refs:
                raise RuntimeError("panel has character_ids but no usable reference_images; character asset stage is incomplete")
            if panel.get("scene_id") and not scene_refs:
                raise RuntimeError("panel has scene_id but no usable approved scene reference")
            settings = _worker_settings(episode, panel, job)
            panel.update({
                "aspect_ratio": settings.get("aspect_ratio", "16:9"),
                "duration_seconds": float(settings.get("duration_seconds", 10.0)),
                "story_context": inputs.get("story_context") or {},
                "scene_context": inputs.get("scene_context") or {},
            })
            qa_retry_feedback = metadata.get("qa_retry_feedback")
            if isinstance(qa_retry_feedback, dict) and qa_retry_feedback.get("reason"):
                panel["qa_retry_feedback"] = dict(qa_retry_feedback)
            else:
                panel.pop("qa_retry_feedback", None)
            first_frame = _resolve_asset_path(panel["first_frame_path"], project) if panel.get("first_frame_path") else None
            last_frame = _resolve_asset_path(panel["last_frame_path"], project) if panel.get("last_frame_path") else None
            approved_group_anchor = (
                (job.get("metadata") or {}).get("approved_group_anchor") or {}
            )
            character_count = len([value for value in panel.get("character_ids") or [] if value])
            paired_state_required = requires_paired_state_anchor(panel, character_count)
            group_anchor_required = requires_approved_group_anchor(panel, metadata, character_count)
            if group_anchor_required and not approved_group_anchor:
                raise RuntimeError(
                    "approved group anchor required before H3 submission; generate, preview and hash-approve it in Web"
                )
            group_anchor_path: Path | None = None
            group_anchor_last_path: Path | None = None
            if approved_group_anchor:
                candidate = Path(str(approved_group_anchor.get("path") or "")).resolve()
                expected_hash = str(approved_group_anchor.get("sha256") or "")
                if (
                    approved_group_anchor.get("status") != "approved"
                    or not candidate.is_file()
                    or not candidate.is_relative_to(project.resolve())
                    or not expected_hash
                    or _sha256_file(candidate) != expected_hash
                    or list(approved_group_anchor.get("character_ids") or [])
                    != [str(value) for value in panel.get("character_ids") or []]
                    or str(approved_group_anchor.get("scene_id") or "")
                    != str(panel.get("scene_id") or "")
                    or str(approved_group_anchor.get("panel_contract_sha256") or "")
                    != panel_anchor_contract_sha256(panel)
                ):
                    raise RuntimeError("approved group anchor is stale, missing or hash-invalid")
                group_anchor_path = candidate
                first_frame = candidate
                if bool(approved_group_anchor.get("paired_state_required")) != paired_state_required:
                    raise RuntimeError("approved group anchor paired-state contract is stale")
                if paired_state_required:
                    final_candidate = Path(str(approved_group_anchor.get("last_path") or "")).resolve()
                    final_expected_hash = str(approved_group_anchor.get("last_sha256") or "")
                    if (
                        not final_candidate.is_file()
                        or not final_candidate.is_relative_to(project.resolve())
                        or not final_expected_hash
                        or _sha256_file(final_candidate) != final_expected_hash
                    ):
                        raise RuntimeError(
                            "approved paired group anchor is missing a valid final-state image"
                        )
                    group_anchor_last_path = final_candidate
                    last_frame = final_candidate
            continuity = inputs.get("continuity_dependency") or {}
            if continuity.get("strict"):
                previous = store.get_job(str(continuity.get("previous_job_id")))
                if not previous or previous["status"] != "succeeded" or not previous.get("output_path") or not Path(previous["output_path"]).is_file():
                    raise RuntimeError(f"strict continuity predecessor is not succeeded: {continuity.get('previous_job_id')}")
                if not first_frame:
                    tail_dir = project / "continuity"
                    first_frame = _extract_tail_frame(
                        Path(previous["output_path"]), tail_dir / f"{job['panel_name']}_from_previous.png"
                    )
                # The configured last-frame anchor may intentionally point at
                # the freshly extracted continuity tail. Resolve it again after
                # extraction so the composition-anchor policy sees both roles.
                if not last_frame and panel.get("last_frame_path"):
                    last_frame = _resolve_asset_path(panel["last_frame_path"], project)
            # A predecessor tail is temporal continuity evidence, not a
            # complete cast/scene identity bible.  Suppressing approved
            # character and scene refs made newly entering actors disappear
            # and duplicated the same tail into both H3 temporal slots.
            # Composition-only authority is therefore an explicit expert
            # opt-in for a deliberately authored group anchor.
            composition_anchor_first = bool(
                first_frame
                and (
                    group_anchor_path
                    or (
                        continuity.get("strict")
                        and str(panel.get("continuity_reference_policy") or "")
                        == "composition_anchor_first"
                    )
                )
            )
            if not first_frame and scene_refs:
                first_frame = scene_refs[0]
            anchor = character_refs[0] if character_refs else None
            extras = character_refs[1:] + ([] if first_frame in scene_refs else scene_refs)
            anchor_source_id = (
                reference_source_ids.get(str(Path(anchor).resolve())) if anchor else None
            )
            extra_source_ids = [
                reference_source_ids.get(str(Path(path).resolve())) for path in extras
            ]
            submitted = submit_render_job(
                panel,
                Path(job["output_path"]),
                ep_id=ep_id,
                panel_index=index,
                job_id=job["job_id"],
                character_desc=_character_context(episode, panel),
                char_refs=extras,
                duration_seconds=float(settings.get("duration_seconds", 10.0)),
                first_frame=first_frame,
                last_frame=last_frame,
                character_anchor=anchor,
                character_anchor_source_id=anchor_source_id,
                extra_reference_source_ids=extra_source_ids,
                use_lora=bool(settings.get("use_lora", True)),
                lora_strength=float(settings.get("lora_strength", 1.0)),
                aspect_ratio=str(settings.get("aspect_ratio", "16:9")),
                reference_fidelity=str(settings["reference_fidelity"]),
                sage_attention=str(settings["sage_attention"]),
                composition_anchor_first=composition_anchor_first,
                megapixels=float(settings.get("megapixels", 0.6)),
                turbo_steps=int(settings.get("turbo_steps", 8)),
                render_profile=str(settings.get("render_profile") or "production"),
                production_strategy=str(settings.get("production_strategy") or DEFAULT_PRODUCTION_STRATEGY),
                delivery_eligible=bool(settings.get("delivery_eligible", True)),
                store=store,
                progress_cb=callback,
            )
            update_status(ep_id, index, "submitted", comfy_path=submitted.get("prompt_id"))
            output = wait_render_job(
                submitted["job_id"], store=store, timeout=timeout,
                poll_interval=poll_interval, progress_cb=callback,
            )
            update_status(ep_id, index, "finalized", final_path=str(output))
            prepare_episode(ep_id, _load_episode(ep_id))
            if not release_before_next_shot(job):
                break
            finished = store.get_job(job["job_id"])
            if _requires_proof_human_gate(finished):
                callback("review", f"{job.get('panel_name')}: proof 已完成，等待人工验收后再继续")
                break
        except Exception as exc:
            current = store.get_job(job["job_id"])
            if str(exc).startswith("strict continuity predecessor is not succeeded:"):
                # This panel did not consume GPU work and is merely waiting for
                # its predecessor.  Keep it queued and continue scanning;
                # marking every descendant failed burns retry budgets and makes
                # an overnight run appear to stop after its first clips.
                warning = {
                    "job_id": job["job_id"],
                    "disposition": "blocked_by_strict_predecessor",
                    "error": str(exc),
                }
                warnings.append(warning)
                callback("blocked", f"{job['panel_name']}: {exc}")
                # A later panel may be independent and should still be able to
                # finish overnight. Strict descendants will remain queued when
                # their own predecessor gate is evaluated.
                continue
            if _has_validated_success_artifact(current):
                warning = _preserve_success_with_pipeline_warning(store, current, exc)
                warnings.append({"job_id": job["job_id"], **warning})
                callback(
                    "warning",
                    f"{job['panel_name']}: validated clip preserved; episode refresh deferred: {exc}",
                )
                continue
            invalidation_audit = (
                (current.get("metadata") or {}).get("qa_invalidation_audit") or []
                if current else []
            )
            qa_chain_invalidated = bool(
                invalidation_audit
                and invalidation_audit[-1].get("action") == "invalidated_by_rejected_predecessor"
            )
            if current and current["status"] not in {"failed", "cancelled"}:
                store.update_job(job["job_id"], status="failed", error=str(exc))
            update_status(ep_id, index, "failed", error=str(exc))
            failures.append({"job_id": job["job_id"], "error": str(exc)})
            callback("error", f"{job['panel_name']}: {exc}")
            if qa_chain_invalidated:
                # The reviewer invalidated the continuity chain while this
                # worker was waiting. Leave later descendants queued for the
                # next explicit resume instead of cascading synthetic failures.
                break
    return {
        "ep_id": ep_id, "resume": resume_summary,
        "failures": failures, "warnings": warnings,
        "jobs_consumed": jobs_consumed, "max_jobs": max_jobs,
        "snapshot": project_snapshot(ep_id),
    }


def render_episode(
    ep_id: str,
    *,
    character_anchor: Optional[Path] = None,
    use_lora: Optional[bool] = None,
    lora_strength: Optional[float] = None,
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> list[Path]:
    """Synchronous CLI compatibility wrapper around the durable worker flow."""
    episode = _load_episode(ep_id)
    if character_anchor:
        episode.setdefault("character_bible", [{"character_id": "legacy_character"}])
        episode["character_bible"][0]["reference_images"] = [str(character_anchor.resolve())]
    if use_lora is not None:
        episode["use_lora"] = use_lora
    if lora_strength is not None:
        episode["lora_strength"] = lora_strength
    prepare_episode(ep_id, episode)
    result = run_episode_jobs(ep_id, ensure_character_assets=not bool(character_anchor), progress_cb=progress_cb)
    return [
        Path(job["output_path"]) for job in result["snapshot"]["jobs"]
        if job["status"] == "succeeded" and job.get("output_path")
    ]


def assemble_episode(
    ep_id: str,
    *,
    preset: str = "landscape_16_9",
    resize_mode: str = "fit",
) -> Optional[Path]:
    """Export all validated clips and return the final path for compatibility."""
    try:
        manifest = export_episode(ep_id, preset, resize_mode=resize_mode)
    except ValueError as exc:
        if "no successful clips" in str(exc):
            return None
        raise
    return Path(manifest["output_path"])


def status(ep_id: str) -> dict[str, Any]:
    return project_snapshot(ep_id)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: orchestrator.py split|render|assemble|status|resume|export|example ...")
        return
    command = sys.argv[1]
    if command == "example":
        print(json.dumps(COMIC_EXAMPLE_HERO_KAIJU, ensure_ascii=False, indent=2))
        return
    if command == "split":
        if len(sys.argv) < 3:
            raise SystemExit("need story.txt")
        ep_id = next((arg.split("=", 1)[1] for arg in sys.argv[3:] if arg.startswith("--ep-id=")), f"ep_{int(time.time())}")
        language = "en" if "--en" in sys.argv[3:] else "cn"
        story = Path(sys.argv[2]).read_text(encoding="utf-8")
        result = split_story_to_episode(story, ep_id=ep_id, language=language)
        print(json.dumps({"ep_id": ep_id, "panels": len(result["panels"])}, ensure_ascii=False))
        return
    if len(sys.argv) < 3:
        raise SystemExit(f"{command} requires ep_id")
    ep_id = sys.argv[2]
    if command == "status":
        print(json.dumps(status(ep_id), ensure_ascii=False, indent=2))
    elif command == "resume":
        print(json.dumps(resume_jobs(ep_id), ensure_ascii=False, indent=2))
    elif command == "render":
        rendered = render_episode(ep_id, progress_cb=lambda phase, message: print(f"[{phase}] {message}"))
        print(json.dumps({"rendered": [str(path) for path in rendered]}, ensure_ascii=False, indent=2))
    elif command in {"assemble", "export"}:
        preset = next((arg.split("=", 1)[1] for arg in sys.argv[3:] if arg.startswith("--preset=")), "landscape_16_9")
        resize_mode = next((arg.split("=", 1)[1] for arg in sys.argv[3:] if arg.startswith("--resize-mode=")), "fit")
        output = assemble_episode(ep_id, preset=preset, resize_mode=resize_mode)
        print(json.dumps({"output_path": str(output) if output else None}, ensure_ascii=False))
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
