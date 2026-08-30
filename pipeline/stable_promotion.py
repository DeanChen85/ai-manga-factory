"""Deterministically promote an approved H3 proof into a 720p production master.

The approved visual content is preserved byte-for-byte at the decoded-story
level: FFmpeg only scales/pads, normalizes FPS/audio, and writes a new audited
artifact.  No model is called and no generative detail is invented.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from atomic_io import write_json_atomic
from runtime_config import ffmpeg_executable, ffprobe_executable, projects_dir
from task_store import RenderJobStore, default_store, prepare_episode, production_gate, project_snapshot
from video_delivery import probe_media, validate_probe
from video_quality import analyze_video, evaluate_content, select_edit_window, validate_edit_selection


STABLE_PROMOTION_CONTRACT = "approved-proof-stable-production/v1"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_content_qa_output_path(
    content_qa: Mapping[str, Any], output: Path,
) -> dict[str, Any]:
    """Bind decoded QA evidence to the atomically installed output path."""
    rebound = dict(content_qa)
    analysis = dict(rebound.get("analysis") or {})
    analysis["source_path"] = str(output)
    rebound["analysis"] = analysis
    source_analysis = dict(rebound.get("source_analysis") or {})
    source_analysis["source_path"] = str(output)
    rebound["source_analysis"] = source_analysis
    return rebound


def _project_file(value: str | Path, project: Path, *, require_file: bool = True) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(project.resolve()):
        raise RuntimeError(f"stable promotion path is outside the episode project: {path}")
    if require_file and (not path.is_file() or path.stat().st_size <= 0):
        raise FileNotFoundError(path)
    return path


def _approved_proof_contract(
    job: Mapping[str, Any], project: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = dict(job.get("metadata") or {})
    promotion = dict(metadata.get("preview_promotion") or {})
    if promotion.get("status") != "approved":
        raise RuntimeError("stable promotion requires an approved proof record")
    proof = _project_file(str(promotion.get("output_path") or ""), project)
    if proof.parent.name != "previews":
        raise RuntimeError("approved proof must come from the episode previews directory")
    if _sha256(proof) != str(promotion.get("artifact_sha256") or ""):
        raise RuntimeError("approved proof artifact hash changed after human review")
    artifact_path = proof.with_suffix(".artifact.json")
    artifact = json.loads(_project_file(artifact_path, project).read_text(encoding="utf-8"))
    if (
        str(artifact.get("job_id") or "") != str(job.get("job_id") or "")
        or artifact.get("artifact_sha256") != promotion.get("artifact_sha256")
        or not (artifact.get("content_qa") or {}).get("passed")
        or ((artifact.get("content_qa") or {}).get("analysis") or {}).get("decoded_visual_sha256")
        != promotion.get("decoded_visual_sha256")
        or (artifact.get("edit_selection") or {}).get("selection_sha256")
        != promotion.get("edit_selection_sha256")
    ):
        raise RuntimeError("approved proof artifact/QA/edit evidence no longer matches promotion")
    graph_path = _project_file(str(promotion.get("graph_path") or ""), project)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    settings = dict(graph.get("settings") or {})
    if (
        settings.get("render_profile") != "proof"
        or settings.get("delivery_eligible") is not False
        or settings.get("prompt_sha256") != promotion.get("prompt_sha256")
        or settings.get("reference_bundle_sha256") != promotion.get("reference_bundle_sha256")
    ):
        raise RuntimeError("approved proof prompt/reference contract is stale")
    return proof, artifact, graph, promotion


def promote_approved_preview_master(
    ep_id: str,
    job_id: str,
    *,
    reason: str,
    confirmed: bool,
    approved_by: str = "reviewer",
    store: RenderJobStore | None = None,
    runner: Callable[..., Any] = subprocess.run,
    probe_func: Callable[..., dict[str, Any]] = probe_media,
    quality_analyzer: Callable[..., dict[str, Any]] = analyze_video,
    quality_runner: Callable[..., Any] = subprocess.run,
    edit_selector: Callable[..., dict[str, Any]] = select_edit_window,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
) -> dict[str, Any]:
    """Create a delivery-eligible 720p derivative from the exact approved proof."""
    if not confirmed:
        raise ValueError("stable production promotion requires explicit human confirmation")
    reason = str(reason).strip()
    approved_by = str(approved_by).strip()
    if not reason or not approved_by:
        raise ValueError("stable production promotion requires reason and approved_by")
    store = store or default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    job_status = str(job.get("status") or "").lower()
    if job_status not in {"failed", "queued"}:
        raise RuntimeError(
            "stable production promotion requires a rejected formal render or an unsubmitted queued render"
        )
    if job_status == "queued":
        if str(job.get("prompt_id") or "").strip() or float(job.get("progress") or 0) > 0:
            raise RuntimeError("queued formal render was already submitted and cannot be replaced safely")
        if store.worker_info(ep_id):
            raise RuntimeError("episode worker is active; stop at the human gate before stable promotion")
    gate = production_gate(ep_id)
    if not gate.get("ready"):
        raise RuntimeError("stable production gate blocked: " + ",".join(gate.get("reasons") or []))
    project = (projects_dir() / ep_id).resolve()
    proof, proof_artifact, proof_graph, promotion = _approved_proof_contract(job, project)
    episode_path = _project_file(project / "episode.json", project)
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    aspect = str(episode.get("aspect_ratio") or "9:16")
    width, height = (1280, 720) if aspect in {"16:9", "landscape"} else (720, 1280)
    output_value = job.get("output_path") or project / "videos" / f"{job['panel_name']}.mp4"
    output = _project_file(output_value, project, require_file=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.stable.partial.mp4"
    manifest_dir = project / "stable_promotions" / str(job.get("panel_name") or "shot")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = ffmpeg or ffmpeg_executable()
    command = [
        ffmpeg_bin, "-y", "-i", str(proof),
        "-vf",
        (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=24"
        ),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(temporary),
    ]
    running_metadata = dict(job.get("metadata") or {})
    running_metadata["stable_promotion"] = {
        "contract": STABLE_PROMOTION_CONTRACT,
        "status": "running", "source_proof": str(proof),
        "source_proof_sha256": promotion["artifact_sha256"],
        "approved_by": approved_by, "reason": reason, "started_at": _utc_now(),
    }
    store.update_job(
        job_id, status="running", progress=0.1, prompt_id=None, error=None,
        metadata=running_metadata,
    )
    try:
        runner(command, check=True, capture_output=True, text=True)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("stable promotion FFmpeg produced no output")
        probe = probe_func(temporary, ffprobe=ffprobe or ffprobe_executable())
        validate_probe(
            probe, expected_width=width, expected_height=height, expected_fps=24,
            require_audio=True, expected_video_codec="h264", expected_pixel_format="yuv420p",
            expected_audio_codec="aac", expected_audio_rate=48000, expected_audio_channels=2,
        )
        source_analysis = quality_analyzer(
            temporary, ffmpeg=ffmpeg_bin, runner=quality_runner,
        )
        source_qa = evaluate_content(source_analysis, (), require_motion=True)
        if not source_qa.get("passed"):
            raise RuntimeError("stable production source QA failed: " + ",".join(source_qa.get("reasons") or []))
        shot_plan = ((job.get("metadata") or {}).get("inputs") or {}).get("shot_plan") or {}
        requested_duration = float(
            shot_plan.get("edit_duration_seconds")
            or (proof_artifact.get("edit_selection") or {}).get("duration_seconds")
            or 0
        )
        selection = edit_selector(
            source_analysis,
            source_duration_seconds=float(probe.get("duration_seconds") or 0),
            requested_duration_seconds=requested_duration,
            source_artifact_sha256=_sha256(temporary),
            edit_hint=shot_plan.get("edit_hint") or {},
        )
        selection_check = validate_edit_selection(
            selection, source_artifact_sha256=_sha256(temporary),
            requested_duration_seconds=requested_duration,
            source_duration_seconds=float(probe.get("duration_seconds") or 0),
        )
        if not selection_check["valid"]:
            raise RuntimeError("stable production edit selection failed: " + ",".join(selection_check["errors"]))
        selected_analysis = quality_analyzer(
            temporary, ffmpeg=ffmpeg_bin, runner=quality_runner,
            start_seconds=float(selection["in_seconds"]),
            duration_seconds=float(selection["duration_seconds"]),
        )
        content_qa = evaluate_content(selected_analysis, (), require_motion=True)
        content_qa.update({
            "source_analysis": source_analysis, "checked_at": _utc_now(),
            "stage": "stable_production_selected_window",
            "render_mode": "approved_preview_master",
            "edit_selection_sha256": selection["selection_sha256"],
        })
        if not content_qa.get("passed"):
            raise RuntimeError("stable production selected-window QA failed")
        temporary.replace(output)
        probe["path"] = str(output)
        artifact_sha = _sha256(output)
        # Rebind the selection to the atomically installed path bytes (the bytes
        # are unchanged by replace, but this assertion guards future edits).
        if artifact_sha != selection["source_artifact_sha256"]:
            raise RuntimeError("stable production artifact changed during atomic install")
        # The decoded bytes do not change during the atomic replace, but QA is
        # intentionally bound to the installed output path as well as its
        # decoded-visual hash.  Rebind both analyses so the UI and delivery
        # gates cannot mistake the now-removed temporary path for stale proof.
        content_qa = _bind_content_qa_output_path(content_qa, output)
        completed_at = _utc_now()
        stable_record = {
            **running_metadata["stable_promotion"],
            "status": "succeeded", "output_path": str(output),
            "output_sha256": artifact_sha, "probe": probe,
            "proof_graph_path": str(promotion.get("graph_path") or ""),
            "proof_prompt_sha256": promotion["prompt_sha256"],
            "proof_reference_bundle_sha256": promotion["reference_bundle_sha256"],
            "transform": "deterministic_scale_pad_audio_normalize_only",
            "ffmpeg_command": command, "completed_at": completed_at,
        }
        manifest = {
            "schema": STABLE_PROMOTION_CONTRACT, "ep_id": ep_id, "job_id": job_id,
            "stable_promotion": stable_record, "probe": probe,
            "content_qa": content_qa, "edit_selection": selection,
            "proof_artifact": {
                "path": str(proof), "sha256": promotion["artifact_sha256"],
                "decoded_visual_sha256": promotion["decoded_visual_sha256"],
                "human_approved_at": promotion.get("approved_at"),
            },
        }
        manifest_path = manifest_dir / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        artifact_path = output.with_suffix(".artifact.json")
        write_json_atomic(artifact_path, {
            "schema_version": 1, "job_id": job_id, "prompt_id": None,
            "source_path": str(proof), "output_path": str(output), "probe": probe,
            "artifact_sha256": artifact_sha, "content_qa": content_qa,
            "edit_selection": selection,
            "reference_images": proof_artifact.get("reference_images") or [],
            # The immutable H3 graph remains the authoritative prompt/reference
            # audit source.  The deterministic transform has its own manifest
            # and must not replace the graph used by the human approval gate.
            "graph_path": str(promotion.get("graph_path") or ""),
            "stable_manifest_path": str(manifest_path),
            "timing_path": promotion.get("timing_path"),
            "stable_promotion_contract": STABLE_PROMOTION_CONTRACT,
        })
        success_metadata = dict((store.get_job(job_id) or job).get("metadata") or {})
        success_metadata.update({
            "render_mode": "approved_preview_master",
            "render_profile": "production",
            "render_profile_id": "approved-proof-stable-master-v1",
            "delivery_eligible": True,
            "artifact_sha256": artifact_sha,
            "content_qa": content_qa,
            "edit_selection": selection,
            "editorial_review": {"status": "pending", "reason": "awaiting human review"},
            "release": {"status": "pending", "reason": "episode release not approved"},
            "stable_promotion": stable_record,
        })
        store.update_job(
            job_id, status="succeeded", progress=1.0, prompt_id=None,
            output_path=str(output), preview_path=str(output), comfy_output_path=None,
            graph_path=str(promotion.get("graph_path") or ""),
            timing_path=promotion.get("timing_path"),
            probe=probe, error=None, metadata=success_metadata,
            reference_images=proof_artifact.get("reference_images") or [],
        )
        try:
            prepare_episode(ep_id, episode)
        except Exception as exc:
            current = store.get_job(job_id) or {}
            current_metadata = dict(current.get("metadata") or {})
            warnings = list(current_metadata.get("pipeline_warnings") or [])
            warnings.append({"stage": "stable_promotion_refresh", "error": str(exc), "at": _utc_now()})
            current_metadata["pipeline_warnings"] = warnings[-20:]
            store.update_job(job_id, metadata=current_metadata)
        return project_snapshot(ep_id)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        current = store.get_job(job_id) or job
        failed_metadata = dict(current.get("metadata") or {})
        failed_metadata["stable_promotion"] = {
            **dict(failed_metadata.get("stable_promotion") or {}),
            "status": "failed", "error": str(exc), "failed_at": _utc_now(),
        }
        store.update_job(
            job_id, status="failed", progress=0.0, prompt_id=None,
            error=f"stable production promotion failed: {exc}", metadata=failed_metadata,
        )
        raise


__all__ = ["STABLE_PROMOTION_CONTRACT", "promote_approved_preview_master"]
