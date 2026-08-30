"""Stable public facade for Web/CLI production controls.

Import this module from Streamlit instead of reaching into renderer internals.
All preparation calls are durable; only ``start_worker``/``run_worker`` begin
the character or video GPU stages.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from continuity_safe import (
    DEFAULT_SAPI_VOICE,
    approve_continuity_anchor as _approve_continuity_anchor,
    run_continuity_safe_chain,
)
from atomic_io import write_json_atomic
from runtime_config import ffmpeg_executable, projects_dir
from task_store import (
    approve_assets as _approve_assets,
    approve_contract as _approve_contract,
    approve_episode_release as _approve_episode_release,
    approve_job_review as _approve_job_review,
    approve_preview_and_promote as _approve_preview_and_promote,
    authorize_additional_job_retry as _authorize_additional_job_retry,
    classify_job_rejection as _classify_job_rejection,
    default_store,
    list_jobs,
    prepare_contract as _prepare_contract,
    prepare_episode as _prepare_episode,
    project_snapshot,
    reject_asset as _reject_asset,
    reject_job as _reject_job,
    resume_jobs as _resume_jobs,
    retry_asset as _retry_asset,
    retry_job as _retry_job,
    revoke_release as _revoke_release,
    select_asset_references as _select_asset_references,
    _safe_id,
)
from render_video_h3 import (
    H3_RUNTIME_PROMPT_CONTRACT,
    authorize_retry_after_comfy_restart as _authorize_retry_after_comfy_restart,
    cancel_render_job,
    reconcile_render_job,
    recover_render_job,
)
from shot_group_anchor import approve_group_anchor as _approve_group_anchor
from shot_group_anchor import generate_group_anchor as _generate_group_anchor
from shot_group_anchor import reject_group_anchor as _reject_group_anchor
from stable_promotion import promote_approved_preview_master as _promote_approved_preview_master
from video_delivery import export_episode, probe_media
from video_quality import (
    analyze_native_dialogue_audio, analyze_video, build_manual_edit_selection, evaluate_content,
    validate_edit_selection,
)
from worker import (
    run_worker, start_assets, start_character_assets, start_continuity_safe,
    start_group_anchor, start_worker,
)


def prepare_contract(ep_id: str, episode: Mapping[str, Any]) -> dict[str, Any]:
    return _prepare_contract(ep_id, episode)


def prepare_episode(ep_id: str, episode: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible name for draft contract registration."""
    return _prepare_episode(ep_id, episode)


def approve_contract(ep_id: str, *, expected_hash: str | None = None) -> dict[str, Any]:
    return _approve_contract(ep_id, expected_hash=expected_hash)


def prepare_assets(ep_id: str, *, timeout: float = 1800.0) -> dict[str, Any]:
    """Start the complete character+scene asset stage in a hidden worker."""
    return start_assets(ep_id, timeout=timeout)


def approve_assets(ep_id: str, *, expected_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    return _approve_assets(ep_id, expected_hashes=expected_hashes)


def prepare_character_assets(ep_id: str, *, timeout: float = 1800.0) -> dict[str, Any]:
    """Queue the character-only stage; never run GPU work on the Web thread."""
    return start_character_assets(ep_id, timeout=timeout)


def promote_approved_preview_master(
    ep_id: str, job_id: str, *, reason: str, confirmed: bool,
    approved_by: str = "reviewer",
) -> dict[str, Any]:
    """Create an audited 720p production master without another model call."""
    return _promote_approved_preview_master(
        ep_id, job_id, reason=reason, confirmed=confirmed, approved_by=approved_by,
    )


def start_production(
    ep_id: str,
    *,
    statuses: Iterable[str] = ("pending", "failed"),
    ensure_character_assets: bool = True,
    timeout: float = 2400.0,
    max_jobs: int | None = None,
) -> dict[str, Any]:
    return start_worker(
        ep_id,
        statuses=statuses,
        ensure_character_assets=ensure_character_assets,
        timeout=timeout,
        max_jobs=max_jobs,
    )


def approve_continuity_anchor(
    ep_id: str,
    job_id: str,
    source_anchor: str | Path,
    *,
    reason: str,
    approved_by: str = "reviewer",
) -> dict[str, Any]:
    """Approve one group anchor before the explicit safe-render action."""
    return _approve_continuity_anchor(
        ep_id, job_id, source_anchor, reason=reason, approved_by=approved_by,
    )


def generate_group_anchor(
    ep_id: str, job_id: str, *, timeout: float = 900.0,
) -> dict[str, Any]:
    """Synchronous entry for tests/CLI; Web must call ``start_group_anchor``."""
    return _generate_group_anchor(ep_id, job_id, timeout=timeout)


def approve_group_anchor(
    ep_id: str, job_id: str, *, expected_sha256: str, reason: str,
    approved_by: str = "reviewer",
) -> dict[str, Any]:
    return _approve_group_anchor(
        ep_id, job_id, expected_sha256=expected_sha256,
        reason=reason, approved_by=approved_by,
    )


def reject_group_anchor(
    ep_id: str, job_id: str, *, reason: str, rejected_by: str = "reviewer",
) -> dict[str, Any]:
    return _reject_group_anchor(
        ep_id, job_id, reason=reason, rejected_by=rejected_by,
    )


def run_continuity_safe(
    ep_id: str,
    job_id: str,
    *,
    preferred_voice: str = DEFAULT_SAPI_VOICE,
    motion: str = "slow_push",
    burn_subtitles: bool | None = None,
    **runtime: Any,
) -> dict[str, Any]:
    """Synchronous worker/testing entry; Web should call start_continuity_safe.

    ``burn_subtitles`` is compatibility-only; panel artifacts remain clean and
    ``export_episode`` performs the optional final burn exactly once.
    """
    return run_continuity_safe_chain(
        ep_id, job_id, preferred_voice=preferred_voice,
        motion=motion, burn_subtitles=burn_subtitles, **runtime,
    )


def status(ep_id: str) -> dict[str, Any]:
    return project_snapshot(ep_id)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reopen_rejected_with_manual_edit(
    ep_id: str, job_id: str, *, expected_archive_sha256: str,
    in_seconds: float, reason: str, relocate_approved_dialogue: bool,
    reviewed_by: str = "reviewer",
) -> dict[str, Any]:
    """Reopen an immutable rejected H3 artifact with a human action window.

    The model is not called.  Only a rejection classified as action/window can
    use this path.  The archived bytes are hash checked, the new interval is
    decoded again, and any approved native H3 dialogue is moved exactly once
    from its original cue range onto the selected action interval.
    """
    ep_id = _safe_id(ep_id, "ep_id")
    job_id = str(job_id or "").strip()
    if not job_id:
        raise ValueError("job_id is required")
    store = default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    if job.get("status") != "failed":
        raise RuntimeError("manual re-edit requires a currently rejected/failed job")
    metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
    audit = metadata.get("qa_rejection_audit") or []
    latest = audit[-1] if audit and isinstance(audit[-1], dict) else {}
    if latest.get("category") != "action_timing_or_edit_window":
        raise RuntimeError("manual re-edit is allowed only for an action timing/window rejection")
    archived = latest.get("archived_files") or {}
    archived_output = Path(str((archived.get("output_path") or {}).get("path") or "")).resolve()
    expected = str(expected_archive_sha256 or "").strip().lower()
    if not archived_output.is_file() or not expected or _sha256_file(archived_output) != expected:
        raise RuntimeError("rejected artifact hash is missing, stale or mismatched")
    if str((archived.get("output_path") or {}).get("sha256") or "").lower() != expected:
        raise RuntimeError("rejection audit does not bind the requested artifact hash")
    projects_root = projects_dir().resolve()
    project = (projects_root / ep_id).resolve()
    try:
        project.relative_to(projects_root)
    except ValueError as exc:
        raise RuntimeError("episode project resolves outside the configured projects root") from exc
    try:
        archived_output.relative_to((project / "rejected").resolve())
    except ValueError as exc:
        raise RuntimeError("rejected artifact is outside the current episode") from exc
    target_raw = str(latest.get("output_path") or "").strip()
    if not target_raw:
        raise RuntimeError("rejection audit is missing the original live output path")
    target_output = Path(target_raw).resolve()
    try:
        target_output.relative_to(project)
    except ValueError as exc:
        raise RuntimeError("restored output must stay inside the current episode") from exc
    if target_output.exists():
        raise RuntimeError("manual re-edit target already exists; refusing to overwrite a live artifact")

    def archived_source(role: str) -> Path | None:
        record = archived.get(role) or {}
        source = Path(str(record.get("path") or "")).resolve()
        if not source.is_file():
            return None
        if _sha256_file(source) != str(record.get("sha256") or "").lower():
            raise RuntimeError(f"archived {role} hash is stale")
        try:
            source.relative_to((project / "rejected").resolve())
        except ValueError as exc:
            raise RuntimeError(f"archived {role} is outside the current episode rejection archive") from exc
        return source

    source_output = archived_source("output_path")
    if source_output is None:
        raise RuntimeError("rejected video could not be restored")
    source_graph = archived_source("graph_path")
    source_timing = archived_source("timing_path")

    artifact_sha256 = _sha256_file(source_output)
    probe = probe_media(source_output)
    source_duration = float(probe.get("duration_seconds") or 0)
    settings = metadata.get("settings") or {}
    shot_plan = ((metadata.get("inputs") or {}).get("shot_plan") or {})
    requested_duration = float(
        shot_plan.get("edit_duration_seconds")
        or settings.get("edit_duration_seconds") or 0
    )
    reason = str(reason or "").strip()
    reviewed_by = str(reviewed_by or "").strip()
    if not reason or not reviewed_by:
        raise ValueError("manual re-edit requires reviewer and reason")

    alignment: list[dict[str, Any]] = []
    dialogue = [dict(item) for item in job.get("dialogue_cues") or [] if isinstance(item, Mapping)]
    if dialogue and not relocate_approved_dialogue:
        excluded = []
        for cue_index, cue in enumerate(dialogue):
            start = float(cue.get("start_seconds", cue.get("start_s", 0)) or 0)
            end = float(cue.get("end_seconds", cue.get("end_s", start)) or start)
            if end <= start or start < float(in_seconds) - 1e-6 or end > float(in_seconds) + requested_duration + 1e-6:
                excluded.append(str(cue_index))
        if excluded:
            raise RuntimeError(
                "selected action window excludes approved native dialogue cues " + ",".join(excluded)
                + "; verified native relocation or a separately approved voice-track contract is required"
            )
    if relocate_approved_dialogue and dialogue:
        if not (probe.get("audio") or {}):
            raise RuntimeError("native H3 dialogue relocation requires an actual source audio stream")
        bounds: list[tuple[float, float]] = []
        for cue in dialogue:
            if not str(cue.get("text") or "").strip():
                raise RuntimeError("approved dialogue cue text is missing; cannot bind native audio relocation")
            start = float(cue.get("start_seconds", cue.get("start_s", 0)) or 0)
            end = float(cue.get("end_seconds", cue.get("end_s", start)) or start)
            if end <= start:
                raise RuntimeError("approved dialogue cue timing is invalid")
            if not (end <= float(in_seconds) + 1e-6 or start >= float(in_seconds) + requested_duration - 1e-6):
                raise RuntimeError("selected interval already overlaps approved dialogue; relocation would double the voice")
            bounds.append((start, end))
        origin = min(start for start, _ in bounds)
        for cue_index, (cue, (start, end)) in enumerate(zip(dialogue, bounds)):
            target_start = start - origin
            target_end = end - origin
            if target_end > requested_duration + 1e-6:
                raise RuntimeError("approved dialogue does not fit inside the selected action interval")
            audio_evidence = analyze_native_dialogue_audio(
                source_output, start_seconds=start, end_seconds=end,
                source_artifact_sha256=artifact_sha256, ffmpeg=ffmpeg_executable(),
            )
            if not audio_evidence.get("eligible_for_native_dialogue_rebase"):
                raise RuntimeError(
                    "native H3 dialogue audio is not audibly complete enough to relocate; "
                    "keep the clip rejected or use a separately approved voice-track contract"
                )
            alignment.append({
                "contract": "source-dialogue-rebase/v1",
                "cue_index": cue_index,
                "source_start_seconds": round(start, 6),
                "source_end_seconds": round(end, 6),
                "target_start_seconds": round(target_start, 6),
                "target_end_seconds": round(target_end, 6),
                "speaker_id": str(cue.get("speaker_id") or ""),
                "text_sha256": hashlib.sha256(str(cue.get("text") or "").encode("utf-8")).hexdigest(),
                "audio_authority": "relocated_native_h3_dialogue",
                "source_audio_evidence": audio_evidence,
            })

    selected_analysis = analyze_video(
        source_output, ffmpeg=ffmpeg_executable(),
        start_seconds=float(in_seconds), duration_seconds=requested_duration,
    )
    full_analysis = analyze_video(source_output, ffmpeg=ffmpeg_executable())
    selection_analysis = dict(selected_analysis)
    selection_analysis["decoded_visual_sha256"] = full_analysis.get("decoded_visual_sha256")
    selection = build_manual_edit_selection(
        selection_analysis, source_duration_seconds=source_duration,
        requested_duration_seconds=requested_duration,
        source_artifact_sha256=artifact_sha256, in_seconds=float(in_seconds),
        reason=reason, reviewed_by=reviewed_by,
        dialogue_audio_alignment=alignment,
        current_dialogue_cues=dialogue if alignment else None,
    )
    selection_check = validate_edit_selection(
        selection, source_artifact_sha256=artifact_sha256,
        requested_duration_seconds=requested_duration,
        source_duration_seconds=source_duration,
        current_dialogue_cues=dialogue if alignment else None,
    )
    if not selection_check["valid"]:
        raise RuntimeError("manual edit selection is invalid: " + ",".join(selection_check["errors"]))
    prior = []
    for other in store.list_jobs(ep_id):
        if int(other.get("panel_index") or 0) >= int(job.get("panel_index") or 0):
            continue
        analysis = (((other.get("metadata") or {}).get("content_qa") or {}).get("analysis") or {})
        if analysis.get("decoded_visual_sha256"):
            prior.append((str(other["job_id"]), analysis))
    content_qa = evaluate_content(selected_analysis, prior, require_motion=True)
    if not content_qa.get("passed"):
        raise RuntimeError(
            "manual selected-window content QA failed: "
            + ",".join(str(item) for item in content_qa.get("reasons") or ["unknown"])
        )
    content_qa.update({
        "source_analysis": full_analysis,
        "stage": "human_contract_selected_window",
        "render_mode": "h3",
        "edit_selection_sha256": selection["selection_sha256"],
    })
    # The immutable rejected bytes were used for staging QA.  Once committed,
    # both analyses must identify the restored live artifact so Web evidence
    # cannot silently fall back to an unrelated full-source review.
    content_qa["analysis"]["source_path"] = str(target_output)
    content_qa["analysis"]["archived_source_path"] = str(source_output)
    content_qa["source_analysis"]["source_path"] = str(target_output)
    content_qa["source_analysis"]["archived_source_path"] = str(source_output)
    previous_review = metadata.get("editorial_review") or {}
    history = list(metadata.get("editorial_review_history") or [])
    if previous_review:
        history.append(previous_review)
    previous_release = metadata.get("release") or {}
    release_history = list(metadata.get("release_history") or [])
    if previous_release:
        release_history.append(previous_release)
    metadata.update({
        "artifact_sha256": artifact_sha256,
        "content_qa": content_qa,
        "edit_selection": selection,
        "editorial_review": {"status": "pending", "reason": "manual action window awaiting review"},
        "editorial_review_history": history[-50:],
        "release": {"status": "pending", "reason": "manual edit selection requires review"},
        "release_history": release_history[-50:],
    })
    reedit_audit = list(metadata.get("manual_reedit_audit") or [])
    reedit_audit.append({
        "action": "reopen_rejected_with_manual_edit",
        "archive_sha256": expected,
        "selection_sha256": selection["selection_sha256"],
        "in_seconds": selection["in_seconds"],
        "out_seconds": selection["out_seconds"],
        "dialogue_audio_authority": (
            "relocated_native_h3_dialogue" if alignment else "selected_native_audio"
        ),
        "reviewed_by": reviewed_by,
        "reason": reason,
    })
    metadata["manual_reedit_audit"] = reedit_audit[-20:]

    # Nothing above mutates the live project.  Commit all restored files only
    # after hash/probe/audio/visual QA passed, and clean them back up if a
    # later local write or SQLite update fails.
    restore_targets: list[tuple[Path, Path]] = [(source_output, target_output)]
    if source_graph:
        restore_targets.append((source_graph, target_output.with_suffix(".graph.json")))
    if source_timing:
        restore_targets.append((source_timing, target_output.with_suffix(".cues.json")))
    for _, target in restore_targets:
        try:
            target.resolve().relative_to(project)
        except ValueError as exc:
            raise RuntimeError("restored companion target is outside the current episode") from exc
        if target.exists():
            raise RuntimeError("manual re-edit companion target already exists; refusing overwrite")
    artifact_path = target_output.with_suffix(".artifact.json")
    if artifact_path.exists():
        raise RuntimeError("manual re-edit artifact sidecar already exists; refusing overwrite")
    created: list[Path] = []
    temporaries: list[Path] = []
    try:
        for source, target in restore_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".restore.tmp")
            if temporary.exists():
                raise RuntimeError("manual re-edit temporary restore path already exists")
            temporaries.append(temporary)
            shutil.copy2(source, temporary)
            temporary.replace(target)
            temporaries.remove(temporary)
            created.append(target)
        restored_output = target_output
        restored_graph = target_output.with_suffix(".graph.json") if source_graph else None
        restored_timing = target_output.with_suffix(".cues.json") if source_timing else None
        probe["path"] = str(restored_output)
        write_json_atomic(artifact_path, {
        "schema_version": 1,
        "job_id": job_id,
        "prompt_id": latest.get("prompt_id"),
        "source_path": latest.get("comfy_output_path"),
        "output_path": str(restored_output),
        "probe": probe,
        "artifact_sha256": artifact_sha256,
        "content_qa": content_qa,
        "edit_selection": selection,
        "reference_images": job.get("reference_images") or [],
        "graph_path": str(restored_graph) if restored_graph else None,
        "timing_path": str(restored_timing) if restored_timing else None,
        })
        created.append(artifact_path)
        updated = store.compare_and_update_job(
            job_id,
            expected={
                "status": "failed", "prompt_id": job.get("prompt_id"),
                "retry_count": job.get("retry_count"), "input_hash": job.get("input_hash"),
            },
            status="succeeded", progress=1.0,
            prompt_id=latest.get("prompt_id"), output_path=str(restored_output),
            preview_path=str(restored_output),
            comfy_output_path=latest.get("comfy_output_path"),
            graph_path=str(restored_graph) if restored_graph else None,
            timing_path=str(restored_timing) if restored_timing else None,
            probe=probe, error=None, metadata=metadata,
        )
        if updated is None:
            raise RuntimeError("manual re-edit lost compare-and-swap ownership; no state was committed")
    except Exception:
        for temporary in reversed(temporaries):
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass
        for created_path in reversed(created):
            try:
                if created_path.exists():
                    created_path.unlink()
            except OSError:
                pass
        raise
    return project_snapshot(ep_id)


def approve_job_review(
    ep_id: str, job_id: str, *, expected_artifact_sha256: str,
    expected_edit_selection_sha256: str,
    reviewed_by: str = "reviewer", reason: str = "editorial content approved",
) -> dict[str, Any]:
    return _approve_job_review(
        ep_id, job_id, expected_artifact_sha256=expected_artifact_sha256,
        expected_edit_selection_sha256=expected_edit_selection_sha256,
        reviewed_by=reviewed_by, reason=reason,
    )


def approve_preview_and_promote(
    ep_id: str, job_id: str, *, expected_artifact_sha256: str,
    expected_edit_selection_sha256: str, reviewed_by: str = "reviewer",
    reason: str = "proof prompt, references, motion and continuity approved",
) -> dict[str, Any]:
    """Approve an immutable proof and queue its fresh production render."""
    return _approve_preview_and_promote(
        ep_id, job_id,
        expected_artifact_sha256=expected_artifact_sha256,
        expected_edit_selection_sha256=expected_edit_selection_sha256,
        reviewed_by=reviewed_by, reason=reason,
    )


def approve_episode_release(
    ep_id: str, *, expected_artifact_hashes: Mapping[str, str],
    expected_edit_selection_hashes: Mapping[str, str],
    approved_by: str = "reviewer", reason: str = "episode editorial release approved",
) -> dict[str, Any]:
    return _approve_episode_release(
        ep_id, expected_artifact_hashes=expected_artifact_hashes,
        expected_edit_selection_hashes=expected_edit_selection_hashes,
        approved_by=approved_by, reason=reason,
    )


def revoke_release(
    ep_id: str, *, reason: str, revoked_by: str = "reviewer",
) -> dict[str, Any]:
    """Revoke delivery eligibility without deleting any panel or export."""
    return _revoke_release(ep_id, reason=reason, revoked_by=revoked_by)


def retry(ep_id: str, job_id: str) -> dict[str, Any]:
    return _retry_job(ep_id, job_id)


def reconcile_job(ep_id: str, job_id: str) -> dict[str, Any]:
    """Reconcile one persisted Comfy prompt before any retry is permitted."""
    job = next((item for item in list_jobs(ep_id) if item["job_id"] == job_id), None)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    return reconcile_render_job(job_id, store=default_store())


def authorize_retry_after_comfy_restart(
    ep_id: str, job_id: str, *, confirmed: bool,
) -> dict[str, Any]:
    """Public Web facade for an audited recovery after Comfy history loss."""
    job = next((item for item in list_jobs(ep_id) if item["job_id"] == job_id), None)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    return _authorize_retry_after_comfy_restart(
        job_id, confirmed=confirmed, store=default_store(),
    )


def recover_job(
    ep_id: str,
    job_id: str,
    *,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recover an existing successful Comfy prompt without submitting GPU work.

    The local artifact supplies the exact prompt id; Comfy history remains the
    authority for marking the clip succeeded. Missing history restores the
    prior durable state instead of falsely accepting a local file.
    """
    snapshot = project_snapshot(ep_id)
    job = next((item for item in snapshot.get("jobs") or [] if item["job_id"] == job_id), None)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    project = Path(snapshot["project_dir"]).resolve()
    default_artifact = (
        Path(job["output_path"]).with_suffix(".artifact.json")
        if job.get("output_path") else project / "videos" / f"{job['panel_name']}.artifact.json"
    )
    artifact_file = Path(artifact_path or default_artifact).resolve()
    try:
        artifact_file.relative_to(project)
    except ValueError as exc:
        raise ValueError(f"artifact must be inside episode project: {artifact_file}") from exc
    if not artifact_file.is_file():
        raise FileNotFoundError(f"render artifact is missing: {artifact_file}")
    artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
    if str(artifact.get("job_id") or "") != job_id:
        raise ValueError("render artifact job_id does not match requested job")
    prompt_id = str(artifact.get("prompt_id") or "").strip()
    if not prompt_id:
        raise ValueError("render artifact has no prompt_id")

    store = default_store()
    previous = dict(job)
    store.update_job(
        job_id, status="submitted", prompt_id=prompt_id,
        output_path=job.get("output_path") or artifact.get("output_path"),
        graph_path=artifact.get("graph_path") or job.get("graph_path"),
        timing_path=artifact.get("timing_path") or job.get("timing_path"),
        error=None, completed_at=None,
    )
    recovered = recover_render_job(job_id, store=store)
    if recovered.get("status") != "succeeded":
        store.update_job(
            job_id, status=previous["status"], prompt_id=previous.get("prompt_id"),
            output_path=previous.get("output_path"), preview_path=previous.get("preview_path"),
            comfy_output_path=previous.get("comfy_output_path"),
            graph_path=previous.get("graph_path"), timing_path=previous.get("timing_path"),
            error=previous.get("error"), progress=previous.get("progress", 0.0),
            metadata=previous.get("metadata") or {}, probe=previous.get("probe") or {},
            submitted_at=previous.get("submitted_at"), completed_at=previous.get("completed_at"),
        )
        raise RuntimeError(f"Comfy history did not confirm successful prompt {prompt_id}")
    return recovered


def reject_job(
    ep_id: str,
    job_id: str,
    *,
    reason: str = "rejected by reviewer",
    rejection_category: str = "other",
    interrupt_running: bool = True,
) -> dict[str, Any]:
    """Reject one panel and cancel only its active strict-continuity descendants."""

    def cancel_affected(job: dict[str, Any]) -> dict[str, Any]:
        return cancel_render_job(
            str(job["job_id"]),
            interrupt_running=bool(interrupt_running and job.get("status") == "running"),
        )

    return _reject_job(
        ep_id, job_id, reason=reason, rejection_category=rejection_category,
        cancel_job=cancel_affected,
    )


def classify_job_rejection(
    ep_id: str, job_id: str, *, rejection_category: str,
) -> dict[str, Any]:
    """Public reviewer action for classifying a legacy/unclassified QA rejection."""
    return _classify_job_rejection(
        ep_id, job_id, rejection_category=rejection_category,
    )


def authorize_additional_job_retry(
    ep_id: str, job_id: str, *, reason: str,
) -> dict[str, Any]:
    """Public reviewer action that grants exactly one extra failed-job retry."""
    return _authorize_additional_job_retry(ep_id, job_id, reason=reason)


def reject_asset(
    ep_id: str, asset_id: str | None = None, *, asset_type: str | None = None,
    source_id: str | None = None, reason: str = "rejected by reviewer",
) -> dict[str, Any]:
    return _reject_asset(
        ep_id, asset_id, asset_type=asset_type, source_id=source_id, reason=reason,
    )


def retry_asset(
    ep_id: str, asset_id: str | None = None, *, asset_type: str | None = None,
    source_id: str | None = None, reason: str = "manual retry",
) -> dict[str, Any]:
    return _retry_asset(
        ep_id, asset_id, asset_type=asset_type, source_id=source_id, reason=reason,
    )


def select_asset_references(
    ep_id: str, asset_id: str, reference_images: Iterable[str],
    *, reason: str = "selected existing asset in Web review",
) -> dict[str, Any]:
    return _select_asset_references(
        ep_id, asset_id, reference_images, reason=reason,
    )


def resume(ep_id: str, statuses: Iterable[str] = ("pending", "failed")) -> dict[str, Any]:
    """Reconcile every failed remote prompt before bulk resume.

    A plain ``failed -> queued`` transition preserves ``prompt_id`` on purpose
    so recovery can inspect Comfy history.  The Web bulk-resume button must
    therefore perform that inspection first; otherwise an interrupted prompt
    is replayed from the same terminal history and instantly fails again.
    """
    repaired_dependency_blocks = _repair_legacy_strict_predecessor_blocks(ep_id)
    reconciled: list[dict[str, Any]] = []
    retried: list[str] = []
    for job in list_jobs(ep_id):
        if job.get("status") != "failed" or not str(job.get("prompt_id") or "").strip():
            continue
        result = reconcile_render_job(str(job["job_id"]), store=default_store())
        event = {
            "job_id": str(job["job_id"]),
            "disposition": str(result.get("disposition") or "submission_unknown"),
            "reason": str(result.get("reason") or ""),
        }
        reconciled.append(event)
        if event["disposition"] == "safe_to_retry":
            _renew_retry_budget_for_prompt_contract(ep_id, str(job["job_id"]))
            _retry_job(ep_id, str(job["job_id"]))
            retried.append(str(job["job_id"]))
    # Release jobs stuck in "submitted" after ambiguous reconciliation
    # (ComfyUI unreachable, no history, no queue confirmation).
    # Without this, _resume_jobs cannot see them because it only scans
    # "failed" and "pending" statuses.
    released: list[str] = []
    for job in list_jobs(ep_id):
        if job.get("status") == "submitted" and str(job.get("prompt_id") or "").strip():
            default_store().update_job(
                str(job["job_id"]),
                status="failed", progress=0.0,
                error="reconciliation released: submitted -> failed (ambiguous remote state)",
            )
            released.append(str(job["job_id"]))
    for job in list_jobs(ep_id):
        if job.get("status") == "failed" and not str(job.get("prompt_id") or "").strip():
            _renew_retry_budget_for_prompt_contract(ep_id, str(job["job_id"]))
    summary = _resume_jobs(ep_id, statuses=statuses)
    # After reconciliation is complete, clear stale prompt_ids so the next
    # submission creates a fresh Comfy prompt instead of trying to recover
    # a dead one.  _resume_jobs intentionally preserves prompt_id for the
    # low-level recovery path; the web resume must not.
    store = default_store()
    for job_id in summary.get("job_ids") or []:
        job = store.get_job(job_id, ep_id=ep_id)
        if job and str(job.get("prompt_id") or "").strip():
            store.update_job(job_id, prompt_id=None, submitted_at=None)
    summary["reconciled"] = reconciled
    summary["remote_retries"] = retried
    summary["repaired_dependency_blocks"] = repaired_dependency_blocks
    return summary


def _repair_legacy_strict_predecessor_blocks(ep_id: str) -> list[str]:
    """Repair retry budgets burned by pre-fix dependency-only failures."""
    store = default_store()
    repaired: list[str] = []
    prefix = "strict continuity predecessor is not succeeded:"
    for job in store.list_jobs(ep_id):
        if (
            job.get("status") not in {"failed", "queued"}
            or not str(job.get("error") or "").startswith(prefix)
            or str(job.get("prompt_id") or "").strip()
        ):
            continue
        metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
        audit = list(metadata.get("dependency_retry_budget_repair_audit") or [])
        audit.append({
            "previous_retry_count": int(job.get("retry_count") or 0),
            "reason": "legacy_strict_predecessor_block_consumed_no_gpu_prompt",
        })
        metadata["dependency_retry_budget_repair_audit"] = audit[-20:]
        store.update_job(
            str(job["job_id"]), retry_count=0, error=None, metadata=metadata,
        )
        repaired.append(str(job["job_id"]))
    return repaired


def _renew_retry_budget_for_prompt_contract(ep_id: str, job_id: str) -> bool:
    """Grant a fresh bounded attempt only when the runtime compiler changed.

    Retry limits protect overnight mode from an infinite loop.  They must not
    permanently brick a human-rejected shot after a newer deterministic prompt
    compiler fixes the rejected cause.  This revision reset is hash-audited and
    can occur only once per contract version because the new submission stores
    the current version back into job metadata.
    """
    store = default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job or job.get("status") != "failed":
        return False
    retry_count = int(job.get("retry_count") or 0)
    max_retries = max(1, int(job.get("max_retries") or 0))
    metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
    prior_contract = str((metadata.get("settings") or {}).get("runtime_prompt_contract") or "")
    if retry_count < max_retries or not prior_contract or prior_contract == H3_RUNTIME_PROMPT_CONTRACT:
        return False
    audit = list(metadata.get("prompt_contract_revision_audit") or [])
    audit.append({
        "from": prior_contract,
        "to": H3_RUNTIME_PROMPT_CONTRACT,
        "previous_retry_count": retry_count,
        "reason": "new_runtime_prompt_contract_after_failed_or_human_rejected_attempt",
    })
    metadata["prompt_contract_revision_audit"] = audit[-20:]
    store.update_job(job_id, retry_count=0, metadata=metadata)
    return True


def cancel(ep_id: str, job_id: str, *, interrupt_running: bool = False) -> dict[str, Any]:
    job = next((item for item in list_jobs(ep_id) if item["job_id"] == job_id), None)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    return cancel_render_job(job_id, interrupt_running=interrupt_running)


def export(ep_id: str, preset: str, **kwargs: Any) -> dict[str, Any]:
    return export_episode(ep_id, preset, **kwargs)


def run_overnight(
    ep_ids: Iterable[str], *, policy: Any = None, stop_at: Any = None, **runtime: Any,
) -> dict[str, Any]:
    """Run the fail-closed overnight scheduler through public facades only.

    Imports are local so normal Web startup does not pay for or initialize the
    night controller.  Optional runtime hooks exist for offline acceptance
    tests; production callers normally pass only ``ep_ids``, ``policy`` and
    ``stop_at``.
    """
    from overnight_ops import run_overnight_production

    runtime.setdefault("status_fn", status)
    runtime.setdefault("start_fn", start_production)
    runtime.setdefault("resume_fn", resume)
    runtime.setdefault("retry_fn", retry)
    runtime.setdefault("reconcile_fn", reconcile_job)
    return run_overnight_production(
        ep_ids, policy=policy, stop_at=stop_at, **runtime,
    )


# Existing import names remain usable by older Web builds.
resume_jobs = _resume_jobs
retry_job = _retry_job


__all__ = [
    "prepare_episode",
    "prepare_contract",
    "approve_contract",
    "prepare_assets",
    "approve_assets",
    "prepare_character_assets",
    "select_asset_references",
    "start_assets",
    "start_character_assets",
    "list_jobs",
    "resume_jobs",
    "retry_job",
    "project_snapshot",
    "start_worker",
    "start_production",
    "approve_continuity_anchor",
    "generate_group_anchor",
    "start_group_anchor",
    "approve_group_anchor",
    "reject_group_anchor",
    "start_continuity_safe",
    "run_continuity_safe",
    "run_worker",
    "status",
    "approve_job_review",
    "approve_preview_and_promote",
    "promote_approved_preview_master",
    "approve_episode_release",
    "revoke_release",
    "retry",
    "reconcile_job",
    "authorize_retry_after_comfy_restart",
    "recover_job",
    "reject_job",
    "reject_asset",
    "retry_asset",
    "resume",
    "cancel",
    "export",
    "run_overnight",
]
