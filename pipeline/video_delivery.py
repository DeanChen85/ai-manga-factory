"""Validated episode assembly and platform-safe video exports.

Exports deliberately use scale+pad by default so converting a landscape group
shot to a vertical canvas does not silently crop people.  `fill` is opt-in.
No platform logo or watermark is added.
"""
from __future__ import annotations

import json
import hashlib
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from runtime_config import ffmpeg_executable, ffprobe_executable, projects_dir
from atomic_io import write_json_atomic
from task_store import list_jobs
from video_quality import analyze_native_dialogue_audio, analyze_video, evaluate_content
from video_quality import validate_edit_selection
from subtitle_delivery import write_subtitle_bundle


TRANSITION_EVIDENCE_VERSION = "motivated-cut-evidence/v1"
CUT_ON_ACTION_MOTION_MIN = 0.0025
MATCH_CUT_SIMILARITY_MIN = 0.75


BASE_PRESETS: dict[str, dict[str, Any]] = {
    "vertical_9_16": {
        "width": 720, "height": 1280, "fps": 30, "video_codec": "libx264",
        "profile": "high", "pixel_format": "yuv420p", "video_bitrate": "5M",
        "buffer_size": "10M", "delivery_standard": "720p-v1", "orientation": "portrait",
        "audio_codec": "aac", "audio_rate": 48000, "audio_channels": 2,
        "audio_bitrate": "192k", "loudness_lufs": -14,
        "subtitle_margin_v": 256,
    },
    "landscape_16_9": {
        "width": 1280, "height": 720, "fps": 30, "video_codec": "libx264",
        "profile": "high", "pixel_format": "yuv420p", "video_bitrate": "5M",
        "buffer_size": "10M", "delivery_standard": "720p-v1", "orientation": "landscape",
        "audio_codec": "aac", "audio_rate": 48000, "audio_channels": 2,
        "audio_bitrate": "192k", "loudness_lufs": -14,
        "subtitle_margin_v": 72,
    },
    "square_1_1": {
        "width": 720, "height": 720, "fps": 30, "video_codec": "libx264",
        "profile": "high", "pixel_format": "yuv420p", "video_bitrate": "4M",
        "buffer_size": "8M", "delivery_standard": "720p-v1", "orientation": "square",
        "audio_codec": "aac", "audio_rate": 48000, "audio_channels": 2,
        "audio_bitrate": "192k", "loudness_lufs": -14,
        "subtitle_margin_v": 72,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_binding_sha256(
    artifact_hashes: Mapping[str, str], selection_hashes: Mapping[str, str],
    visual_hashes: Mapping[str, str],
) -> str:
    """Stable identity for the exact release approval set in a delivery."""
    payload = {
        "artifact_hashes": {str(k): str(v) for k, v in artifact_hashes.items()},
        "selection_hashes": {str(k): str(v) for k, v in selection_hashes.items()},
        "visual_hashes": {str(k): str(v) for k, v in visual_hashes.items()},
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

PRESET_ALIASES = {
    "tiktok": "vertical_9_16",
    "reels": "vertical_9_16",
    "youtube_shorts": "vertical_9_16",
    "douyin": "vertical_9_16",
    "bilibili": "landscape_16_9",
    "youtube": "landscape_16_9",
    "master_16_9": "landscape_16_9",
    "master_9_16": "vertical_9_16",
    "master_1_1": "square_1_1",
}


def preset_spec(name: str) -> dict[str, Any]:
    canonical = PRESET_ALIASES.get(name, name)
    if canonical not in BASE_PRESETS:
        raise ValueError(f"unknown export preset: {name}; choose from {sorted(set(BASE_PRESETS) | set(PRESET_ALIASES))}")
    return {"requested_name": name, "canonical_name": canonical, **BASE_PRESETS[canonical]}


def probe_media(path: str | Path, *, ffprobe: str | None = None) -> dict[str, Any]:
    media = Path(path).resolve()
    if not media.exists() or media.stat().st_size <= 0:
        raise FileNotFoundError(f"media output is missing or empty: {media}")
    cmd = [
        ffprobe or ffprobe_executable(), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(media),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
    raw = json.loads(proc.stdout)
    streams = raw.get("streams", [])
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video = video_streams[0] if video_streams else None
    audio = audio_streams[0] if audio_streams else None
    if not video:
        raise ValueError(f"no video stream in {media}")
    duration = float(raw.get("format", {}).get("duration") or video.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError(f"invalid duration in {media}: {duration}")
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    try:
        num, den = rate.split("/", 1)
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    video_duration = float(video.get("duration") or duration)
    audio_duration = float(audio.get("duration") or duration) if audio else None
    return {
        "path": str(media),
        "size_bytes": media.stat().st_size,
        "duration_seconds": duration,
        "stream_counts": {"video": len(video_streams), "audio": len(audio_streams)},
        "video": {
            "codec": video.get("codec_name"), "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0), "fps": fps,
            "pixel_format": video.get("pix_fmt"), "duration_seconds": video_duration,
        },
        "audio": None if not audio else {
            "codec": audio.get("codec_name"), "sample_rate": int(audio.get("sample_rate") or 0),
            "channels": int(audio.get("channels") or 0), "duration_seconds": audio_duration,
        },
    }


def validate_probe(
    probe: dict[str, Any], *, expected_width: int | None = None,
    expected_height: int | None = None, expected_fps: float | None = None,
    require_audio: bool = True, expected_video_codec: str | None = None,
    expected_pixel_format: str | None = None, expected_audio_codec: str | None = None,
    expected_audio_rate: int | None = None, expected_audio_channels: int | None = None,
    expected_video_streams: int = 1, expected_audio_streams: int = 1,
    max_av_duration_delta_seconds: float = 0.1,
) -> None:
    video = probe["video"]
    if expected_width and video["width"] != expected_width:
        raise ValueError(f"width mismatch: {video['width']} != {expected_width}")
    if expected_height and video["height"] != expected_height:
        raise ValueError(f"height mismatch: {video['height']} != {expected_height}")
    if expected_fps and abs(video["fps"] - expected_fps) > 0.05:
        raise ValueError(f"fps mismatch: {video['fps']} != {expected_fps}")
    if expected_video_codec and video.get("codec") != expected_video_codec:
        raise ValueError(f"video codec mismatch: {video.get('codec')} != {expected_video_codec}")
    if expected_pixel_format and video.get("pixel_format") != expected_pixel_format:
        raise ValueError(f"pixel format mismatch: {video.get('pixel_format')} != {expected_pixel_format}")
    if require_audio and not probe.get("audio"):
        raise ValueError("delivery output has no audio stream")
    audio = probe.get("audio") or {}
    if expected_audio_codec and audio.get("codec") != expected_audio_codec:
        raise ValueError(f"audio codec mismatch: {audio.get('codec')} != {expected_audio_codec}")
    if expected_audio_rate and audio.get("sample_rate") != expected_audio_rate:
        raise ValueError(f"audio sample rate mismatch: {audio.get('sample_rate')} != {expected_audio_rate}")
    if expected_audio_channels and audio.get("channels") != expected_audio_channels:
        raise ValueError(f"audio channels mismatch: {audio.get('channels')} != {expected_audio_channels}")
    counts = probe.get("stream_counts") or {}
    if counts and int(counts.get("video") or 0) != int(expected_video_streams):
        raise ValueError(
            f"video stream count mismatch: {counts.get('video')} != {expected_video_streams}"
        )
    if counts and int(counts.get("audio") or 0) != int(expected_audio_streams):
        raise ValueError(
            f"audio stream count mismatch: {counts.get('audio')} != {expected_audio_streams}"
        )
    video_duration = video.get("duration_seconds")
    audio_duration = audio.get("duration_seconds")
    if video_duration is not None and audio_duration is not None:
        delta = abs(float(video_duration) - float(audio_duration))
        if delta > float(max_av_duration_delta_seconds):
            raise ValueError(
                f"audio/video duration mismatch: delta {delta:.6f}s exceeds "
                f"{max_av_duration_delta_seconds:.6f}s"
            )


def _video_filter(spec: dict[str, Any], resize_mode: str) -> str:
    width, height = spec["width"], spec["height"]
    if resize_mode == "fit":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease," 
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black," 
            f"setsar=1,fps={spec['fps']}"
        )
    if resize_mode == "fill":
        return (
            f"scale={width}:{height}:force_original_aspect_ratio=increase," 
            f"crop={width}:{height},setsar=1,fps={spec['fps']}"
        )
    raise ValueError("resize_mode must be 'fit' (safe default) or 'fill' (crop)")


def _trim_concat_filter(
    spec: Mapping[str, Any], resize_mode: str,
    selections: Iterable[Mapping[str, Any]],
) -> tuple[str, str, str]:
    resize = _video_filter(dict(spec), resize_mode)
    segments = list(selections)
    filters: list[str] = []
    concat_inputs: list[str] = []
    for index, selection in enumerate(segments):
        edit_in = float(selection["in_seconds"])
        edit_out = float(selection["out_seconds"])
        duration = float(selection["duration_seconds"])
        filters.append(
            f"[{index}:v]trim=start={edit_in:.6f}:end={edit_out:.6f},setpts=PTS-STARTPTS,{resize}[v{index}]"
        )
        alignments = [
            dict(item) for item in (selection.get("dialogue_audio_alignment") or [])
            if isinstance(item, Mapping)
        ]
        if not alignments:
            filters.append(
                f"[{index}:a]atrim=start={edit_in:.6f}:end={edit_out:.6f},asetpts=PTS-STARTPTS[a{index}]"
            )
        else:
            # A rebased dialogue lane is the sole audio authority.  Keeping
            # even quiet selected-source audio here risks duplicate speech;
            # ambience separation needs its own approved contract and is not
            # implemented in this delivery path.
            aligned_labels: list[str] = []
            for cue_index, alignment in enumerate(alignments):
                source_start = float(alignment["source_start_seconds"])
                source_end = float(alignment["source_end_seconds"])
                target_start = float(alignment["target_start_seconds"])
                delay_ms = max(0, round(target_start * 1000))
                label = f"adialogue{index}_{cue_index}"
                filters.append(
                    f"[{index}:a]atrim=start={source_start:.6f}:end={source_end:.6f},"
                    f"asetpts=PTS-STARTPTS,adelay={delay_ms}:all=1,apad,"
                    f"atrim=0:{duration:.6f}[{label}]"
                )
                aligned_labels.append(f"[{label}]")
            if len(aligned_labels) == 1:
                filters.append(f"{aligned_labels[0]}anull[a{index}]")
            else:
                filters.append(
                    "".join(aligned_labels)
                    + f"amix=inputs={len(aligned_labels)}:duration=longest:normalize=0,"
                    f"apad,atrim=0:{duration:.6f}[a{index}]"
                )
        concat_inputs.append(f"[v{index}][a{index}]")
    filters.append("".join(concat_inputs) + f"concat=n={len(segments)}:v=1:a=1[vcat][acat]")
    filters.append(
        f"[acat]loudnorm=I={spec['loudness_lufs']}:TP=-1.5:LRA=11,"
        f"aresample={spec['audio_rate']}[aout]"
    )
    return ";".join(filters), "[vcat]", "[aout]"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(path, payload)


def _archive_stem(job: dict[str, Any]) -> str:
    raw = str(job.get("panel_name") or f"panel_{int(job.get('panel_index') or 0):02d}").strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "panel"
    return stem


def _artifact_archive_name(job: dict[str, Any], field: str, artifact: Path) -> str:
    stem = _archive_stem(job)
    if field == "graph_path":
        suffix = ".manifest.json" if artifact.name.casefold() == "manifest.json" else ".graph.json"
        return f"graphs/{stem}{suffix}"
    suffixes = "".join(artifact.suffixes).lower() or ".bin"
    if suffixes == ".json":
        suffixes = ".cues.json"
    return f"cues/{stem}{suffixes}"


def _zip_write_unique(bundle: zipfile.ZipFile, source: Path, archive_name: str, names: set[str]) -> None:
    normalized = archive_name.replace("\\", "/")
    if normalized in names:
        raise RuntimeError(f"duplicate delivery archive entry: {normalized}")
    bundle.write(source, normalized)
    names.add(normalized)


def _validate_delivery_package(
    package_path: str | Path, *, manifest_path: str | Path,
    final_path: str | Path, required_members: Iterable[str],
) -> dict[str, Any]:
    """Fail closed when the just-written delivery archive is corrupt or incomplete."""
    package = Path(package_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    final_file = Path(final_path).resolve()
    expected_manifest = manifest_file.read_bytes()
    required = {str(name).replace("\\", "/") for name in required_members}
    with zipfile.ZipFile(package, "r") as bundle:
        corrupt_member = bundle.testzip()
        if corrupt_member:
            raise RuntimeError(f"delivery archive CRC failure: {corrupt_member}")
        names = set(bundle.namelist())
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"delivery archive missing required members: {missing}")
        archived_manifest = bundle.read("manifest.json")
        if archived_manifest != expected_manifest:
            raise RuntimeError("delivery archive manifest does not match the verified sidecar")
        archived = json.loads(archived_manifest.decode("utf-8"))
        if archived.get("release_status") != "approved":
            raise RuntimeError("delivery archive manifest is not release-approved")
        final_info = bundle.getinfo("final.mp4")
        if final_info.file_size != final_file.stat().st_size:
            raise RuntimeError("delivery archive final.mp4 size does not match verified output")
    return {
        "status": "passed",
        "required_members": sorted(required),
        "member_count": len(names),
        "manifest_sha256": hashlib.sha256(expected_manifest).hexdigest(),
        "final_size_bytes": final_file.stat().st_size,
    }


def _job_report_row(job: dict[str, Any]) -> dict[str, Any]:
    metadata = job.get("metadata") or {}
    qa_rejections = list(metadata.get("qa_rejection_audit") or [])
    qa_invalidations = list(metadata.get("qa_invalidation_audit") or [])
    latest_rejection = qa_rejections[-1] if qa_rejections else {}
    retry_count = int(job.get("retry_count") or 0)
    return {
        "job_id": str(job.get("job_id") or ""),
        "panel_index": int(job.get("panel_index") or 0),
        "panel_name": str(job.get("panel_name") or ""),
        "status": str(job.get("status") or "unknown"),
        "render_mode": str(metadata.get("render_mode") or "h3"),
        "attempt": max(1, retry_count + 1),
        "retry_count": retry_count,
        "qa": {
            "rejection_count": len(qa_rejections),
            "invalidation_count": len(qa_invalidations),
            "latest_rejection_reason": str(
                latest_rejection.get("reason") or latest_rejection.get("error") or ""
            ),
            "continuity_safe_fallback": metadata.get("render_mode") == "continuity_safe",
        },
        "duration_seconds": float((job.get("probe") or {}).get("duration_seconds") or 0),
        "source_duration_seconds": float((job.get("probe") or {}).get("duration_seconds") or 0),
        "selected_duration_seconds": float((metadata.get("edit_selection") or {}).get("duration_seconds") or 0),
        "edit_selection": metadata.get("edit_selection") or {},
        "discarded": bool((metadata.get("edit_selection") or {}).get("status") == "deadletter"),
        "probe": job.get("probe") or {},
        "output_path": job.get("output_path"),
        "error": job.get("error"),
    }


def _transition_directive(panel: Mapping[str, Any], stored_plan: Mapping[str, Any]) -> dict[str, Any]:
    raw = panel.get("transition")
    if raw is None:
        raw = panel.get("tr")
    if raw is None:
        raw = stored_plan.get("transition", stored_plan.get("tr"))
    if isinstance(raw, Mapping):
        result = dict(raw)
    elif isinstance(raw, (list, tuple)):
        result = {
            "type": raw[0] if raw else "",
            "motivation": raw[1] if len(raw) > 1 else "",
        }
    elif raw:
        result = {"type": str(raw)}
    else:
        result = {}
    if not result.get("sound_bridge"):
        result["sound_bridge"] = panel.get("sound_bridge", stored_plan.get("sound_bridge"))
    return result


def _perceptual_similarity(left: str, right: str) -> float:
    if not left or len(left) != len(right):
        return 0.0
    try:
        distance = (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 0.0
    return 1.0 - distance / float(len(left) * 4)


def _validate_motivated_transition_plan(
    episode: Mapping[str, Any], jobs: list[dict[str, Any]],
    release_qa: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind a requested cut to selected-window visual evidence.

    Only cuts are executed.  Dissolves/crossfades are rejected, and sound
    bridges are fail-closed until audio overlap has its own approved edit hash.
    """
    panels = list(episode.get("panels") or [])
    qa_by_job = {str(item.get("job_id") or ""): item for item in release_qa}
    rows = []
    errors = []
    normalized_types = {
        "": "legacy_hard_cut", "hard_cut": "motivated_hard_cut",
        "motivated_hard_cut": "motivated_hard_cut", "reaction_cut": "motivated_hard_cut",
        "cold_open": "motivated_hard_cut", "close": "terminal_close",
        "end": "terminal_close", "cut_to_black": "terminal_close",
        "none": "terminal_close", "terminal": "terminal_close",
        "cut_on_action": "cut_on_action", "action_cut": "cut_on_action",
        "match_cut": "match_cut",
    }
    for index, job in enumerate(jobs):
        panel_index = int(job.get("panel_index") or index + 1)
        panel = panels[panel_index - 1] if 0 < panel_index <= len(panels) else {}
        panel = panel if isinstance(panel, Mapping) else {}
        stored = (((job.get("metadata") or {}).get("inputs") or {}).get("shot_plan") or {})
        stored = stored if isinstance(stored, Mapping) else {}
        directive = _transition_directive(panel, stored)
        requested = str(directive.get("type") or "").strip().lower().replace("-", "_")
        normalized = normalized_types.get(requested)
        motivation = str(directive.get("motivation") or "").strip()
        sound_bridge = directive.get("sound_bridge")
        shot_role = str(panel.get("shot_role") or stored.get("shot_role") or "").strip().lower()
        current = qa_by_job.get(str(job.get("job_id") or ""), {})
        next_job = jobs[index + 1] if index + 1 < len(jobs) else None
        following = qa_by_job.get(str((next_job or {}).get("job_id") or ""), {})
        current_analysis = current.get("reanalysis") or {}
        following_analysis = following.get("reanalysis") or {}
        current_changes = list((current_analysis.get("metrics") or {}).get("adjacent_luma_changes") or [])
        following_changes = list((following_analysis.get("metrics") or {}).get("adjacent_luma_changes") or [])
        current_hashes = [str(value) for value in current_analysis.get("perceptual_hashes") or []]
        following_hashes = [str(value) for value in following_analysis.get("perceptual_hashes") or []]
        evidence = {
            "outgoing_selection_sha256": str((current.get("edit_selection") or {}).get("selection_sha256") or ""),
            "incoming_selection_sha256": str((following.get("edit_selection") or {}).get("selection_sha256") or ""),
            "outgoing_end_motion": float(current_changes[-1]) if current_changes else None,
            "incoming_start_motion": float(following_changes[0]) if following_changes else None,
            "boundary_perceptual_similarity": (
                _perceptual_similarity(current_hashes[-1], following_hashes[0])
                if current_hashes and following_hashes else None
            ),
        }
        row_errors = []
        if requested in {"dissolve", "crossfade", "cross_dissolve", "fade"}:
            row_errors.append("dissolve_or_crossfade_forbidden")
        elif normalized is None:
            row_errors.append(f"unsupported_transition_type:{requested}")
        elif normalized != "legacy_hard_cut" and not motivation and normalized != "terminal_close":
            row_errors.append("structured_transition_requires_motivation")
        if next_job is None:
            # A final close/end has no outgoing boundary.  Legacy generators
            # often left `hard_cut` on every panel, so an explicitly approved
            # close role converts that otherwise motivated cut into a terminal
            # instruction.  Boundary-dependent requests remain fail-closed.
            terminal_semantics = normalized == "terminal_close" or (
                normalized in {"legacy_hard_cut", "motivated_hard_cut"}
                and shot_role in {"close", "cliffhanger", "end"}
            )
            if terminal_semantics:
                normalized = "terminal_close"
            elif normalized in {"cut_on_action", "match_cut", "motivated_hard_cut"}:
                row_errors.append("requested_transition_has_no_following_shot")
        if normalized == "cut_on_action" and next_job is not None:
            if (
                evidence["outgoing_end_motion"] is None
                or evidence["incoming_start_motion"] is None
                or evidence["outgoing_end_motion"] <= CUT_ON_ACTION_MOTION_MIN
                or evidence["incoming_start_motion"] <= CUT_ON_ACTION_MOTION_MIN
            ):
                row_errors.append("cut_on_action_motion_evidence_insufficient")
        if normalized == "match_cut" and next_job is not None:
            similarity = evidence["boundary_perceptual_similarity"]
            if similarity is None or similarity < MATCH_CUT_SIMILARITY_MIN:
                row_errors.append("match_cut_visual_evidence_insufficient")
        if sound_bridge:
            row_errors.append("sound_bridge_requires_approved_audio_overlap_contract")
        status = "blocked" if row_errors else ("terminal" if not next_job else "validated")
        rows.append({
            "from_job_id": str(job.get("job_id") or ""),
            "to_job_id": str((next_job or {}).get("job_id") or "") or None,
            "requested_type": requested or "unspecified",
            "normalized_type": normalized,
            "motivation": motivation,
            "shot_role": shot_role,
            "execution": "hard_cut" if next_job else "none_terminal",
            "status": status, "evidence": evidence,
            "thresholds": {
                "cut_on_action_motion_min": CUT_ON_ACTION_MOTION_MIN,
                "match_cut_similarity_min": MATCH_CUT_SIMILARITY_MIN,
            },
            "errors": row_errors,
        })
        errors.extend(f"{job.get('job_id')}:{error}" for error in row_errors)
    return {
        "schema_version": TRANSITION_EVIDENCE_VERSION,
        "passed": not errors,
        "execution_policy": "hard_cut_only_no_dissolve",
        "boundaries": rows,
        "errors": errors,
    }


def _director_delivery_plan(
    episode: dict[str, Any], jobs: list[dict[str, Any]],
    transition_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit requested editorial intent while keeping export deterministic.

    The reliable delivery implementation intentionally remains hard cuts.  A
    requested match cut or sound bridge is recorded for a future editor; it is
    never approximated with a dissolve that could conceal continuity drift.
    """
    panels = list(episode.get("panels") or [])
    shots = []
    for job in jobs:
        panel_index = int(job.get("panel_index") or 0)
        panel = panels[panel_index - 1] if 0 < panel_index <= len(panels) else {}
        panel = panel if isinstance(panel, dict) else {}
        stored_plan = (((job.get("metadata") or {}).get("inputs") or {}).get("shot_plan") or {})
        stored_plan = stored_plan if isinstance(stored_plan, dict) else {}

        def field(name: str, default: Any = None) -> Any:
            return panel.get(name) if panel.get(name) is not None else stored_plan.get(name, default)

        camera = field("camera_plan", {})
        camera = camera if isinstance(camera, dict) else {}
        transition = _transition_directive(panel, stored_plan)
        requested_type = str(transition.get("type") or "hard_cut").strip()
        sound_bridge = field("sound_bridge", transition.get("sound_bridge"))
        shots.append({
            "job_id": str(job.get("job_id") or ""),
            "panel_id": str(panel.get("panel_id") or panel.get("name") or job.get("panel_name") or ""),
            "story_function": field("story_function", ""),
            "blocking": field("blocking", {}),
            "screen_direction": field("screen_direction", ""),
            "axis": field("axis", ""),
            "eyeline": field("eyeline", ""),
            "dominant_camera_move": field("dominant_camera_move", camera.get("movement") or ""),
            "keyframe_strategy": {
                "first": field("first_keyframe_strategy", ""),
                "last": field("last_keyframe_strategy", ""),
                "combined": field("keyframe_strategy", {}),
            },
            "transition": {
                "requested_type": requested_type,
                "motivation": str(transition.get("motivation") or ""),
                "match_cut": transition.get("match_cut") or field("match_cut", {}),
                "executed_type": "hard_cut",
            },
            "sound_bridge": {
                "request": sound_bridge or {},
                "execution": "blocked_without_approved_audio_overlap_contract" if sound_bridge else "not_requested",
            },
            "risk": field("risk", field("risk_code", "")),
            "failure_code": field("failure_code", field("failure_codes", "")),
        })
    return {
        "schema_version": "director-delivery-plan/v1",
        "execution_policy": {
            "video_transition": "hard_cut_only",
            "requested_transitions_are_metadata_only": True,
            "sound_bridges_are_metadata_only": True,
            "dissolve_forbidden": True,
            "reason": "do not conceal content or continuity drift",
        },
        "shots": shots,
        "transition_validation": dict(transition_validation or {}),
    }


def _write_morning_report(
    *,
    ep_id: str,
    jobs: list[dict[str, Any]],
    project: Path,
    json_path: Path,
    markdown_path: Path,
    delivery: dict[str, Any],
) -> dict[str, Any]:
    try:
        disk = shutil.disk_usage(project)
        disk_report: dict[str, Any] = {
            "path": str(project.resolve()), "total_bytes": disk.total,
            "used_bytes": disk.used, "free_bytes": disk.free,
            "free_gib": round(disk.free / (1024 ** 3), 2),
        }
    except OSError as exc:
        disk_report = {"path": str(project.resolve()), "error": str(exc)}
    rows = [_job_report_row(job) for job in jobs]
    report = {
        "schema_version": 1,
        "report_type": "episode_morning_report",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ep_id": ep_id,
        "summary": {
            "job_count": len(rows),
            "status_counts": {
                status: sum(row["status"] == status for row in rows)
                for status in sorted({row["status"] for row in rows})
            },
            "qa_rejection_count": sum(row["qa"]["rejection_count"] for row in rows),
            "continuity_safe_count": sum(row["qa"]["continuity_safe_fallback"] for row in rows),
            "total_validated_duration_seconds": round(sum(row["duration_seconds"] for row in rows), 3),
            "total_source_duration_seconds": round(sum(row["source_duration_seconds"] for row in rows), 3),
            "total_selected_duration_seconds": round(sum(row["selected_duration_seconds"] for row in rows), 3),
            "target_edit_duration_seconds": delivery.get("target_edit_duration_seconds"),
            "discarded_count": sum(row["discarded"] for row in rows),
        },
        "panels": rows,
        "delivery": delivery,
        "disk": disk_report,
    }
    _write_json_atomic(json_path, report)
    lines = [
        f"# Morning Report — {ep_id}", "",
        f"Generated: {report['generated_at']}",
        f"Delivery: {delivery.get('output_path')}",
        f"Disk free: {disk_report.get('free_gib', 'unknown')} GiB", "",
        "| # | Panel | Status | Mode | Attempt | QA rejects | Duration |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['panel_index']} | {row['panel_name']} | {row['status']} | "
            f"{row['render_mode']} | {row['attempt']} | {row['qa']['rejection_count']} | "
            f"{row['duration_seconds']:.3f}s |"
        )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _concat_entry(path: Path) -> str:
    escaped = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped}'\n"


def _subtitle_filter_path(path: Path) -> str:
    """Escape a path for FFmpeg's subtitles filter, including Windows drives."""
    return path.resolve().as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def write_dialogue_vtt(cues: Iterable[dict[str, Any]], path: str | Path) -> Path:
    def ts(seconds: float) -> str:
        ms = max(0, round(float(seconds) * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for index, cue in enumerate(cues, 1):
        text = str(cue.get("text") or "").strip()
        if not text:
            continue
        lines.extend([
            str(index), f"{ts(cue.get('start_seconds', 0))} --> {ts(cue.get('end_seconds', 0))}",
            text, "",
        ])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def export_episode(
    ep_id: str,
    preset: str,
    *,
    clip_paths: Optional[Iterable[str | Path]] = None,
    output_path: str | Path | None = None,
    resize_mode: str = "fit",
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    probe_func: Callable[..., dict[str, Any]] = probe_media,
    quality_analyzer: Callable[..., dict[str, Any]] = analyze_video,
    quality_runner: Callable[..., Any] = subprocess.run,
    create_package: bool = True,
    burn_subtitles: bool = True,
    subtitle_strict: bool = True,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Export only content-QA-passed and editorially released panel artifacts."""
    spec = preset_spec(preset)
    project = projects_dir() / ep_id
    exports = project / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    jobs = list_jobs(ep_id)
    episode_path = project / "episode.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode contract is missing: {episode_path}")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    expected_panels = len(episode.get("panels") or [])
    if require_complete:
        incomplete = [
            {"job_id": job["job_id"], "status": job["status"]}
            for job in jobs if job["status"] != "succeeded"
        ]
        if expected_panels != len(jobs):
            raise RuntimeError(
                f"delivery gate blocked: expected {expected_panels} panel jobs, found {len(jobs)}"
            )
        if incomplete:
            raise RuntimeError(f"delivery gate blocked: incomplete panel jobs: {incomplete}")
        unvalidated = []
        for job in jobs:
            clip_probe = job.get("probe") or {}
            video_probe = clip_probe.get("video") or {}
            if (
                float(clip_probe.get("duration_seconds") or 0) <= 0
                or int(video_probe.get("width") or 0) <= 0
                or int(video_probe.get("height") or 0) <= 0
                or float(video_probe.get("fps") or 0) <= 0
            ):
                unvalidated.append(job["job_id"])
        if unvalidated:
            raise RuntimeError(f"delivery gate blocked: clips have no validated probe: {unvalidated}")
    if clip_paths is None:
        selected = [
            Path(job["output_path"]) for job in jobs
            if job["status"] == "succeeded" and job.get("output_path")
        ]
    else:
        selected = [Path(p) for p in clip_paths]
        if require_complete:
            registered = {
                Path(job["output_path"]).resolve() for job in jobs if job.get("output_path")
            }
            requested = {path.resolve() for path in selected}
            if requested != registered or len(selected) != len(jobs):
                raise RuntimeError("delivery gate blocked: explicit clip_paths must match every registered panel job")
    if not selected:
        raise ValueError(f"no successful clips available for episode {ep_id}")
    missing = [str(path) for path in selected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"delivery clips are missing: {missing}")

    # Re-decode every current artifact. A container checksum cannot detect that
    # six different audio tracks share the same failed/static visual stream.
    selected_by_path = {path.resolve(): path for path in selected}
    release_qa: list[dict[str, Any]] = []
    prior_analyses: list[tuple[str, Mapping[str, Any]]] = []
    gate_errors: list[str] = []
    for job in jobs:
        output = Path(str(job.get("output_path") or "")).resolve()
        if output not in selected_by_path:
            continue
        metadata = job.get("metadata") or {}
        stored_qa = dict(metadata.get("content_qa") or {})
        stored_analysis = dict(stored_qa.get("analysis") or {})
        review = dict(metadata.get("editorial_review") or {})
        release = dict(metadata.get("release") or {})
        selection = dict(metadata.get("edit_selection") or {})
        requested_edit_duration = (
            ((metadata.get("inputs") or {}).get("shot_plan") or {}).get("edit_duration_seconds")
            or (metadata.get("settings") or {}).get("edit_duration_seconds")
        )
        try:
            if requested_edit_duration is None:
                raise RuntimeError("approved edit_duration_seconds is missing")
            selection_check = validate_edit_selection(
                selection, source_artifact_sha256=_sha256_file(output),
                requested_duration_seconds=float(requested_edit_duration),
                source_duration_seconds=float((job.get("probe") or {}).get("duration_seconds") or 0),
                current_dialogue_cues=[
                    cue for cue in (job.get("dialogue_cues") or []) if isinstance(cue, Mapping)
                ],
            )
            if not selection_check["valid"]:
                raise RuntimeError("invalid edit selection:" + ",".join(selection_check["errors"]))
            for alignment_index, alignment in enumerate(selection.get("dialogue_audio_alignment") or []):
                if not isinstance(alignment, Mapping):
                    raise RuntimeError(f"dialogue audio alignment {alignment_index} is not an object")
                observed_evidence = analyze_native_dialogue_audio(
                    output,
                    start_seconds=float(alignment["source_start_seconds"]),
                    end_seconds=float(alignment["source_end_seconds"]),
                    source_artifact_sha256=_sha256_file(output),
                    ffmpeg=ffmpeg or ffmpeg_executable(), runner=runner,
                )
                if (
                    not observed_evidence.get("eligible_for_native_dialogue_rebase")
                    or observed_evidence != alignment.get("source_audio_evidence")
                ):
                    raise RuntimeError(
                        f"dialogue audio alignment {alignment_index} lacks current audibility evidence"
                    )
            current = quality_analyzer(
                output, ffmpeg=ffmpeg or ffmpeg_executable(), runner=quality_runner,
                start_seconds=float(selection["in_seconds"]),
                duration_seconds=float(selection["duration_seconds"]),
            )
            evaluated = evaluate_content(current, prior_analyses, require_motion=True)
        except Exception as exc:
            gate_errors.append(f"{job['job_id']}:content_analysis_error:{exc}")
            continue
        current_visual = str(current.get("decoded_visual_sha256") or "")
        current_artifact = _sha256_file(output)
        selection_hash = str(selection.get("selection_sha256") or "")
        if not stored_qa.get("passed") or not stored_analysis.get("decoded_visual_sha256"):
            gate_errors.append(f"{job['job_id']}:persisted_content_qa_missing_or_failed")
        elif stored_analysis.get("decoded_visual_sha256") != current_visual:
            gate_errors.append(f"{job['job_id']}:content_qa_visual_hash_stale")
        if not evaluated.get("passed"):
            gate_errors.extend(f"{job['job_id']}:{reason}" for reason in evaluated.get("reasons") or [])
        if (
            review.get("status") != "approved"
            or review.get("artifact_sha256") != current_artifact
            or review.get("decoded_visual_sha256") != current_visual
            or review.get("edit_selection_sha256") != selection_hash
        ):
            gate_errors.append(f"{job['job_id']}:editorial_review_missing_or_stale")
        if (
            release.get("status") != "approved"
            or release.get("artifact_sha256") != current_artifact
            or release.get("decoded_visual_sha256") != current_visual
            or release.get("edit_selection_sha256") != selection_hash
        ):
            gate_errors.append(f"{job['job_id']}:release_not_approved_or_stale")
        prior_analyses.append((str(job["job_id"]), current))
        release_qa.append({
            "job_id": job["job_id"], "artifact_sha256": current_artifact,
            "stored": stored_qa, "reanalysis": current, "evaluation": evaluated,
            "editorial_review": review, "release": release,
            "edit_selection": selection,
        })
    if len(release_qa) != len(selected):
        gate_errors.append("selected_clip_job_mapping_incomplete")
    transition_validation = _validate_motivated_transition_plan(episode, jobs, release_qa)
    gate_errors.extend(transition_validation["errors"])
    if gate_errors:
        raise RuntimeError("delivery content/release gate blocked: " + ";".join(gate_errors))

    final = Path(output_path) if output_path else exports / f"{ep_id}_{spec['canonical_name']}.mp4"
    final = final.resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    concat_file = exports / f"{ep_id}_{spec['canonical_name']}_concat.txt"
    concat_lines = []
    for item in release_qa:
        path = Path(next(job["output_path"] for job in jobs if job["job_id"] == item["job_id"]))
        escaped = path.resolve().as_posix().replace("'", "'\\''")
        selection = item["edit_selection"]
        concat_lines.extend([
            f"file '{escaped}'\n",
            f"inpoint {float(selection['in_seconds']):.6f}\n",
            f"outpoint {float(selection['out_seconds']):.6f}\n",
        ])
    concat_file.write_text("".join(concat_lines), encoding="utf-8")
    temp_output = final.with_suffix(".partial.mp4")
    subtitle_bundle = write_subtitle_bundle(
        episode, jobs, final, strict=subtitle_strict,
        play_res_x=spec["width"], play_res_y=spec["height"],
        margin_v=spec["subtitle_margin_v"],
    )
    legacy_preburned_jobs = []
    for job in jobs:
        metadata = job.get("metadata") or {}
        safe_metadata = metadata.get("continuity_safe") or {}
        subtitle_metadata = safe_metadata.get("subtitle_paths") or {}
        if bool(subtitle_metadata.get("burned_in")):
            legacy_preburned_jobs.append(str(job.get("job_id") or job.get("panel_name") or "unknown"))
    if burn_subtitles and subtitle_bundle["cues"] and legacy_preburned_jobs:
        raise RuntimeError(
            "delivery subtitle gate blocked: source clips already contain legacy burned-in "
            "subtitles; export with burn_subtitles=False or regenerate those continuity-safe "
            "clips as clean masters: " + ",".join(legacy_preburned_jobs)
        )
    filter_complex, video_label, audio_label = _trim_concat_filter(
        spec, resize_mode, [item["edit_selection"] for item in release_qa],
    )
    if burn_subtitles and subtitle_bundle["cues"]:
        filter_complex += (
            f";{video_label}subtitles='{_subtitle_filter_path(Path(subtitle_bundle['ass_path']))}'[vout]"
        )
        video_label = "[vout]"
    command = [ffmpeg or ffmpeg_executable(), "-y"]
    for path in selected:
        command.extend(["-i", str(path.resolve())])
    command.extend([
        "-filter_complex", filter_complex, "-map", video_label, "-map", audio_label,
        "-c:v", spec["video_codec"], "-profile:v", spec["profile"],
        "-pix_fmt", spec["pixel_format"], "-b:v", spec["video_bitrate"],
        "-maxrate", spec["video_bitrate"], "-bufsize", spec["buffer_size"],
        "-c:a", spec["audio_codec"], "-b:a", spec["audio_bitrate"],
        "-ar", str(spec["audio_rate"]), "-ac", str(spec["audio_channels"]),
        "-movflags", "+faststart", str(temp_output),
    ])
    runner(command, check=True, capture_output=True, text=True)
    if not temp_output.exists() or temp_output.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg did not produce a valid temporary output: {temp_output}")
    probe = probe_func(temp_output, ffprobe=ffprobe)
    validate_probe(
        probe, expected_width=spec["width"], expected_height=spec["height"],
        expected_fps=spec["fps"], require_audio=True,
        expected_video_codec="h264", expected_pixel_format=spec["pixel_format"],
        expected_audio_codec="aac", expected_audio_rate=spec["audio_rate"],
        expected_audio_channels=spec["audio_channels"],
    )
    selected_total_duration = sum(
        float(item["edit_selection"]["duration_seconds"]) for item in release_qa
    )
    render_settings = episode.get("render_settings") or {}
    target_edit_duration = float(
        render_settings.get("target_edit_duration_seconds") or selected_total_duration
    )
    if abs(selected_total_duration - target_edit_duration) > (1.0 / 24.0):
        raise RuntimeError(
            f"delivery edit duration gate blocked: selected={selected_total_duration:.6f}, "
            f"target={target_edit_duration:.6f}"
        )
    if abs(float(probe.get("duration_seconds") or 0) - target_edit_duration) > 0.075:
        raise RuntimeError(
            f"delivery output duration gate blocked: output={probe.get('duration_seconds')}, "
            f"target={target_edit_duration:.6f}"
        )
    temp_output.replace(final)
    probe["path"] = str(final)

    dialogue_cues = list(subtitle_bundle["cues"])
    audio_cues = []
    offset = 0.0
    for job in jobs:
        if job["status"] != "succeeded":
            continue
        selection = (job.get("metadata") or {}).get("edit_selection") or {}
        edit_in = float(selection.get("in_seconds") or 0)
        edit_out = float(selection.get("out_seconds") or 0)
        alignments = [
            item for item in (selection.get("dialogue_audio_alignment") or [])
            if isinstance(item, Mapping)
        ]
        if alignments:
            dialogue = [item for item in (job.get("dialogue_cues") or []) if isinstance(item, Mapping)]
            for alignment in alignments:
                cue_index = int(alignment.get("cue_index"))
                if cue_index < 0 or cue_index >= len(dialogue):
                    raise RuntimeError("validated dialogue alignment cue index is outside job dialogue")
                cue = dict(dialogue[cue_index])
                if hashlib.sha256(str(cue.get("text") or "").encode("utf-8")).hexdigest() != str(alignment.get("text_sha256") or ""):
                    raise RuntimeError("validated dialogue alignment text is stale")
                cue.update({
                    "start_seconds": float(alignment["target_start_seconds"]) + offset,
                    "end_seconds": float(alignment["target_end_seconds"]) + offset,
                    "source_start_seconds": float(alignment["source_start_seconds"]),
                    "source_end_seconds": float(alignment["source_end_seconds"]),
                    "audio_authority": "relocated_native_h3_dialogue",
                })
                audio_cues.append(cue)
        else:
            for cue in job.get("audio_cues", []):
                start = float(cue.get("start_seconds", cue.get("start_s", 0)) or 0)
                end = float(cue.get("end_seconds", cue.get("end_s", start)) or start)
                if end <= edit_in or start >= edit_out:
                    continue
                shifted = dict(cue)
                shifted["start_seconds"] = max(start, edit_in) - edit_in + offset
                shifted["end_seconds"] = min(end, edit_out) - edit_in + offset
                shifted["audio_authority"] = "selected_native_audio"
                audio_cues.append(shifted)
        offset += float(selection.get("duration_seconds") or 0)
    manifest = {
        "schema_version": 1,
        "ep_id": ep_id,
        "preset": spec,
        "resize_mode": resize_mode,
        "watermark_added": False,
        "source_clips": [str(path.resolve()) for path in selected],
        "edit_selections": {
            str(item["job_id"]): item["edit_selection"] for item in release_qa
        },
        "source_duration_seconds": round(sum(
            float((job.get("probe") or {}).get("duration_seconds") or 0) for job in jobs
        ), 6),
        "selected_duration_seconds": round(selected_total_duration, 6),
        "target_edit_duration_seconds": round(target_edit_duration, 6),
        "output_path": str(final),
        "probe": probe,
        "dialogue_cues": dialogue_cues,
        "audio_cues": audio_cues,
        "director_delivery_plan": _director_delivery_plan(
            episode, jobs, transition_validation,
        ),
        "subtitles": {
            "source": "approved_spoken_dialogue",
            "burned_in": bool((burn_subtitles and dialogue_cues) or legacy_preburned_jobs),
            "burn_stage": "final_delivery" if burn_subtitles and dialogue_cues else (
                "legacy_source_clip" if legacy_preburned_jobs else "none"
            ),
            "legacy_preburned_job_ids": legacy_preburned_jobs,
            "strict": subtitle_strict,
            "warnings": subtitle_bundle["warnings"],
            "canvas": subtitle_bundle["subtitle_canvas"],
            "safe_margin_bottom_px": subtitle_bundle["subtitle_canvas"]["safe_margin_bottom_px"],
            "srt_path": subtitle_bundle["srt_path"],
            "vtt_path": subtitle_bundle["vtt_path"],
            "ass_path": subtitle_bundle["ass_path"],
        },
        "subtitle_vtt": subtitle_bundle["vtt_path"],
        "ffmpeg_command": command,
        "content_qa": {
            "policy": "fail_closed_redecode_every_export",
            "passed": True,
            "shots": release_qa,
        },
        "release_status": "approved",
        "approved_artifact_hashes": {
            str(item["job_id"]): str(item["artifact_sha256"]) for item in release_qa
        },
        "approved_visual_hashes": {
            str(item["job_id"]): str(item["reanalysis"]["decoded_visual_sha256"])
            for item in release_qa
        },
        "approved_edit_selection_hashes": {
            str(item["job_id"]): str(item["edit_selection"]["selection_sha256"])
            for item in release_qa
        },
    }
    manifest_path = final.with_suffix(".manifest.json")
    manifest["release_binding_sha256"] = release_binding_sha256(
        manifest["approved_artifact_hashes"],
        manifest["approved_edit_selection_hashes"],
        manifest["approved_visual_hashes"],
    )
    qa_report_payload = {
        "algorithm": "delivery-content-release-gate/v1",
        "ep_id": ep_id, "release_status": "approved", "shots": release_qa,
    }
    qa_report_path = final.with_suffix(".qa-report.json")
    _write_json_atomic(qa_report_path, qa_report_payload)
    qa_report_hash = hashlib.sha256(qa_report_path.read_bytes()).hexdigest()
    manifest["qa_report_path"] = str(qa_report_path)
    manifest["qa_report_sha256"] = qa_report_hash
    package_path = final.with_suffix(".delivery.zip") if create_package else None
    manifest["package_path"] = str(package_path) if package_path else None
    morning_json = final.with_suffix(".morning-report.json")
    morning_markdown = final.with_suffix(".morning-report.md")
    delivery_report = {
        "preset": spec,
        "resize_mode": resize_mode,
        "output_path": str(final),
        "manifest_path": str(manifest_path),
        "package_path": str(package_path) if package_path else None,
        "probe": probe,
        "release_status": "approved",
        "approved_artifact_hashes": manifest["approved_artifact_hashes"],
        "approved_visual_hashes": manifest["approved_visual_hashes"],
        "approved_edit_selection_hashes": manifest["approved_edit_selection_hashes"],
        "source_duration_seconds": manifest["source_duration_seconds"],
        "selected_duration_seconds": manifest["selected_duration_seconds"],
        "target_edit_duration_seconds": manifest["target_edit_duration_seconds"],
        "qa_report_path": str(qa_report_path),
        "qa_report_sha256": qa_report_hash,
        "subtitles": {
            "burned_in": manifest["subtitles"]["burned_in"],
            "burn_stage": manifest["subtitles"]["burn_stage"],
        },
    }
    _write_morning_report(
        ep_id=ep_id, jobs=jobs, project=project,
        json_path=morning_json, markdown_path=morning_markdown,
        delivery=delivery_report,
    )
    manifest["morning_report"] = {
        "json_path": str(morning_json), "markdown_path": str(morning_markdown),
    }
    _write_json_atomic(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    if package_path:
        temp_package = package_path.with_suffix(package_path.suffix + ".tmp")
        with zipfile.ZipFile(temp_package, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            archive_names: set[str] = set()
            _zip_write_unique(bundle, final, "final.mp4", archive_names)
            _zip_write_unique(bundle, manifest_path, "manifest.json", archive_names)
            if episode_path.exists():
                _zip_write_unique(bundle, episode_path, "episode.json", archive_names)
            _zip_write_unique(bundle, morning_json, "reports/morning-report.json", archive_names)
            _zip_write_unique(bundle, morning_markdown, "reports/morning-report.md", archive_names)
            _zip_write_unique(bundle, qa_report_path, "reports/content-qa.json", archive_names)
            for key in ("srt_path", "vtt_path", "ass_path"):
                subtitle_path = Path(subtitle_bundle[key])
                if subtitle_path.exists():
                    _zip_write_unique(
                        bundle, subtitle_path, f"subtitles/{subtitle_path.name}", archive_names,
                    )
            for job in jobs:
                for field in ("graph_path", "timing_path"):
                    artifact = Path(job[field]) if job.get(field) else None
                    if artifact and artifact.exists():
                        _zip_write_unique(
                            bundle, artifact,
                            _artifact_archive_name(job, field, artifact), archive_names,
                        )
        required_members = {
            "final.mp4", "manifest.json", "reports/morning-report.json",
            "reports/morning-report.md", "reports/content-qa.json",
        }
        if episode_path.exists():
            required_members.add("episode.json")
        for key in ("srt_path", "vtt_path", "ass_path"):
            subtitle_path = Path(subtitle_bundle[key])
            if subtitle_path.exists():
                required_members.add(f"subtitles/{subtitle_path.name}")
        try:
            package_validation = _validate_delivery_package(
                temp_package, manifest_path=manifest_path, final_path=final,
                required_members=required_members,
            )
        except Exception:
            temp_package.unlink(missing_ok=True)
            raise
        temp_package.replace(package_path)
        manifest["package_validation"] = package_validation
    return manifest
