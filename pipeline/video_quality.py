"""Offline, audio-independent content QA for rendered panel videos.

The container checksum is deliberately not used as a visual identity: two MP4s
with identical pictures and different audio must compare as the same shot.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from runtime_config import ffmpeg_executable


ALGORITHM_NAME = "ai-manga-decoded-visual-qa"
ALGORITHM_VERSION = "1.0.0"
SAMPLE_WIDTH = 160
SAMPLE_HEIGHT = 160
STATIC_MEAN_CHANGE_MAX = 0.0025
STATIC_FIRST_LAST_MAX = 0.0100
NEAR_DUPLICATE_SIMILARITY_MIN = 0.985
EDIT_SELECTOR_NAME = "ai-manga-motion-window-selector"
EDIT_SELECTOR_VERSION = "1.0.0"
MANUAL_SELECTOR_NAME = "ai-manga-human-contract-window"
MANUAL_SELECTOR_VERSION = "1.0.0"
MIN_EDIT_DURATION_SECONDS = 1.5
MAX_EDIT_DURATION_SECONDS = 4.0
AUDIO_EVIDENCE_CONTRACT = "native-h3-audible-dialogue-window/v1"
MAX_DIALOGUE_SILENCE_SECONDS = 0.25
MIN_DIALOGUE_AUDIBLE_FRACTION = 0.70


def _run(runner: Callable[..., Any], command: list[str], *, text: bool) -> Any:
    result = runner(command, check=True, capture_output=True, text=text)
    return result.stdout


def _merged_process_log(result: Any) -> str:
    """Return both process channels without assuming a specific runner type."""
    return "\n".join(
        str(value or "") for value in (
            getattr(result, "stdout", ""), getattr(result, "stderr", ""),
        )
    )


def _merged_ranges(ranges: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(a), float(b)) for a, b in ranges if b > a):
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def analyze_native_dialogue_audio(
    path: str | Path, *, start_seconds: float, end_seconds: float,
    source_artifact_sha256: str, ffmpeg: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Prove an H3 dialogue interval is audibly usable before rebasing it.

    Cue metadata alone is not audio evidence.  We run ``silencedetect`` over
    the exact source interval and fail closed when a long silent gap would
    create an apparently aligned subtitle with no native speech underneath.
    This is deliberately an *audibility* proof, not a speech-recognition
    claim; deployments that need semantic ASR verification need a separate,
    approved provider contract.
    """
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"audio evidence input is missing or empty: {source}")
    start = float(start_seconds)
    end = float(end_seconds)
    duration = end - start
    if start < 0 or duration <= 0:
        raise ValueError("dialogue audio evidence range is invalid")
    executable = ffmpeg or ffmpeg_executable()
    command = [
        executable, "-hide_banner", "-nostats", "-v", "info", "-i", str(source),
        "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-map", "0:a:0",
        "-af", "asetpts=PTS-STARTPTS,silencedetect=n=-35dB:d=0.08", "-f", "null", "-",
    ]
    result = runner(command, check=True, capture_output=True, text=True)
    log = _merged_process_log(result)
    if re.search(r"(?:matches no streams|Stream map .*matches no streams|does not contain any stream)", log, re.I):
        raise RuntimeError("native H3 source has no usable audio stream for dialogue relocation")
    starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", log)]
    ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", log)]
    silent: list[tuple[float, float]] = []
    for index, silence_start in enumerate(starts):
        silence_end = ends[index] if index < len(ends) else duration
        silent.append((max(0.0, silence_start), min(duration, silence_end)))
    merged = _merged_ranges(silent)
    silent_seconds = sum(end_value - start_value for start_value, end_value in merged)
    audible_seconds = max(0.0, duration - silent_seconds)
    max_silence = max((end_value - start_value for start_value, end_value in merged), default=0.0)
    audible_fraction = audible_seconds / duration
    eligible = bool(
        audible_fraction >= MIN_DIALOGUE_AUDIBLE_FRACTION
        and max_silence <= MAX_DIALOGUE_SILENCE_SECONDS
    )
    return {
        "contract": AUDIO_EVIDENCE_CONTRACT,
        "source_artifact_sha256": str(source_artifact_sha256),
        "source_start_seconds": round(start, 6),
        "source_end_seconds": round(end, 6),
        "duration_seconds": round(duration, 6),
        "silence_threshold_db": -35,
        "minimum_silence_seconds": 0.08,
        "silent_ranges_seconds": [[round(a, 6), round(b, 6)] for a, b in merged],
        "silent_seconds": round(silent_seconds, 6),
        "audible_seconds": round(audible_seconds, 6),
        "audible_fraction": round(audible_fraction, 6),
        "max_silence_seconds": round(max_silence, 6),
        "eligible_for_native_dialogue_rebase": eligible,
    }


def _decoded_hash(
    path: Path, *, ffmpeg: str, runner: Callable[..., Any],
    start_seconds: float = 0.0, duration_seconds: float | None = None,
) -> str:
    command = [ffmpeg, "-v", "error"]
    if start_seconds > 0:
        command.extend(["-ss", f"{start_seconds:.6f}"])
    command.extend(["-i", str(path)])
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.6f}"])
    command.extend(["-map", "0:v:0",
        "-f", "hash", "-hash", "sha256", "-",
    ])
    stdout = str(_run(runner, command, text=True) or "")
    for line in stdout.splitlines():
        if line.upper().startswith("SHA256="):
            value = line.split("=", 1)[1].strip().lower()
            if len(value) == 64:
                return value
    raise RuntimeError(f"ffmpeg did not return a decoded video hash for {path}")


def _mean_absolute_change(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("sample frames have incompatible dimensions")
    return sum(abs(a - b) for a, b in zip(left, right)) / (len(left) * 255.0)


def _perceptual_hash(frame: bytes) -> str:
    """Return a deterministic 16x16 block-mean hash for a 160x160 gray frame."""
    if len(frame) != SAMPLE_WIDTH * SAMPLE_HEIGHT:
        raise ValueError("invalid sampled frame size")
    block = 10
    values: list[float] = []
    for by in range(16):
        for bx in range(16):
            total = 0
            for y in range(by * block, (by + 1) * block):
                start = y * SAMPLE_WIDTH + bx * block
                total += sum(frame[start:start + block])
            values.append(total / float(block * block))
    threshold = sum(values) / len(values)
    bits = 0
    for value in values:
        bits = (bits << 1) | int(value >= threshold)
    return f"{bits:064x}"


def analyze_video(
    path: str | Path,
    *,
    ffmpeg: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    sample_fps: float = 1.0,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Decode a clip and return audio-independent visual QA evidence."""
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"video QA input is missing or empty: {source}")
    executable = ffmpeg or ffmpeg_executable()
    command = [executable, "-v", "error"]
    if start_seconds > 0:
        command.extend(["-ss", f"{float(start_seconds):.6f}"])
    command.extend(["-i", str(source)])
    if duration_seconds is not None:
        command.extend(["-t", f"{float(duration_seconds):.6f}"])
    command.extend(["-map", "0:v:0",
        "-vf", f"fps={float(sample_fps):g},scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT}:flags=area,format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ])
    raw = _run(runner, command, text=False)
    if not isinstance(raw, (bytes, bytearray)):
        raise RuntimeError("ffmpeg sampled-frame output was not binary")
    frame_size = SAMPLE_WIDTH * SAMPLE_HEIGHT
    if len(raw) < frame_size or len(raw) % frame_size:
        raise RuntimeError(f"ffmpeg returned an incomplete sampled frame stream for {source}")
    frames = [bytes(raw[offset:offset + frame_size]) for offset in range(0, len(raw), frame_size)]
    changes = [_mean_absolute_change(frames[index - 1], frames[index]) for index in range(1, len(frames))]
    first_last = _mean_absolute_change(frames[0], frames[-1]) if len(frames) > 1 else 0.0
    mean_change = sum(changes) / len(changes) if changes else 0.0
    perceptual = [_perceptual_hash(frame) for frame in frames]
    static = mean_change <= STATIC_MEAN_CHANGE_MAX and first_last <= STATIC_FIRST_LAST_MAX
    return {
        "algorithm": {
            "name": ALGORITHM_NAME,
            "version": ALGORITHM_VERSION,
            "sample_fps": float(sample_fps),
            "sample_size": [SAMPLE_WIDTH, SAMPLE_HEIGHT],
            "start_seconds": float(start_seconds),
            "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
            "static_mean_change_max": STATIC_MEAN_CHANGE_MAX,
            "static_first_last_max": STATIC_FIRST_LAST_MAX,
            "near_duplicate_similarity_min": NEAR_DUPLICATE_SIMILARITY_MIN,
        },
        "source_path": str(source),
        "decoded_visual_sha256": _decoded_hash(
            source, ffmpeg=executable, runner=runner,
            start_seconds=float(start_seconds), duration_seconds=duration_seconds,
        ),
        "sample_stream_sha256": hashlib.sha256(bytes(raw)).hexdigest(),
        "sample_frame_sha256": [hashlib.sha256(frame).hexdigest() for frame in frames],
        "perceptual_hashes": perceptual,
        "metrics": {
            "sample_count": len(frames),
            "mean_adjacent_luma_change": mean_change,
            "max_adjacent_luma_change": max(changes) if changes else 0.0,
            "first_last_luma_change": first_last,
            "adjacent_luma_changes": changes,
        },
        "static": static,
    }


def _hash_similarity(left: str, right: str) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    distance = (int(left, 16) ^ int(right, 16)).bit_count()
    return 1.0 - distance / (len(left) * 4.0)


def compare_analyses(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    left_hash = str(left.get("decoded_visual_sha256") or "")
    right_hash = str(right.get("decoded_visual_sha256") or "")
    left_frames = [str(item) for item in left.get("perceptual_hashes") or []]
    right_frames = [str(item) for item in right.get("perceptual_hashes") or []]
    count = min(len(left_frames), len(right_frames))
    similarity = (
        sum(_hash_similarity(left_frames[index], right_frames[index]) for index in range(count)) / count
        if count else 0.0
    )
    exact = bool(left_hash and left_hash == right_hash)
    return {
        "exact_duplicate": exact,
        "near_duplicate": bool(exact or similarity >= NEAR_DUPLICATE_SIMILARITY_MIN),
        "perceptual_similarity": similarity,
        "compared_samples": count,
    }


def evaluate_content(
    analysis: Mapping[str, Any],
    prior: Iterable[tuple[str, Mapping[str, Any]]] = (),
    *,
    require_motion: bool = True,
) -> dict[str, Any]:
    reasons: list[str] = []
    comparisons: list[dict[str, Any]] = []
    if require_motion and bool(analysis.get("static")):
        reasons.append("static_or_ineffective_motion")
    for job_id, other in prior:
        comparison = {"job_id": str(job_id), **compare_analyses(analysis, other)}
        comparisons.append(comparison)
        if comparison["exact_duplicate"]:
            reasons.append(f"exact_visual_duplicate:{job_id}")
        elif comparison["near_duplicate"]:
            reasons.append(f"near_visual_duplicate:{job_id}")
    return {
        "passed": not reasons,
        "algorithm": {"name": ALGORITHM_NAME, "version": ALGORITHM_VERSION},
        "analysis": dict(analysis),
        "comparisons": comparisons,
        "reasons": reasons,
    }


def _selection_hash(selection: Mapping[str, Any]) -> str:
    payload = {
        key: selection.get(key) for key in (
            "in_seconds", "out_seconds", "duration_seconds", "reason", "metrics",
            "selector", "source_artifact_sha256", "source_decoded_visual_sha256",
        )
    }
    # Optional fields are included only when present so selections produced by
    # older releases keep their existing hashes.  New manual selections bind
    # the reviewer identity and any source-dialogue relocation instructions.
    for key in ("manual_review", "dialogue_audio_alignment", "dialogue_cues_sha256"):
        if key in selection:
            payload[key] = selection.get(key)
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def select_edit_window(
    analysis: Mapping[str, Any], *, source_duration_seconds: float,
    requested_duration_seconds: float, source_artifact_sha256: str,
    edit_hint: Mapping[str, Any] | None = None,
    protected_ranges: Iterable[tuple[float, float]] = (),
) -> dict[str, Any]:
    """Select the strongest sampled motion window; missing evidence fails closed."""
    source_duration = float(source_duration_seconds)
    requested = float(requested_duration_seconds)
    if not MIN_EDIT_DURATION_SECONDS <= requested <= MAX_EDIT_DURATION_SECONDS:
        raise ValueError(
            f"edit duration must be between {MIN_EDIT_DURATION_SECONDS} and {MAX_EDIT_DURATION_SECONDS} seconds"
        )
    if source_duration + 1e-6 < requested:
        raise ValueError("source clip is shorter than requested edit duration")
    metrics = analysis.get("metrics") or {}
    interval_changes = [float(value) for value in metrics.get("adjacent_luma_changes") or []]
    sample_fps = float((analysis.get("algorithm") or {}).get("sample_fps") or 0.0)
    if sample_fps <= 0 or not interval_changes:
        raise RuntimeError("edit selector has insufficient sampled motion evidence")
    span = max(1, round(requested * sample_fps))
    if len(interval_changes) < span:
        raise RuntimeError("edit selector has too few sampled intervals for requested duration")
    candidates = []
    for start_index in range(0, len(interval_changes) - span + 1):
        window = interval_changes[start_index:start_index + span]
        candidates.append({
            "start_index": start_index,
            "mean_change": sum(window) / len(window),
            "min_change": min(window), "max_change": max(window),
        })
    protected = [(float(start), float(end)) for start, end in protected_ranges]
    viable = []
    for item in candidates:
        edit_in = item["start_index"] / sample_fps
        edit_out = edit_in + requested
        if item["mean_change"] <= STATIC_MEAN_CHANGE_MAX:
            continue
        if protected and not all(edit_in <= start + 1e-6 and edit_out >= end - 1e-6 for start, end in protected):
            continue
        viable.append(item)
    if not viable:
        raise RuntimeError("no non-static edit window satisfies the requested duration")
    best = max(viable, key=lambda item: (item["mean_change"], item["min_change"]))
    edit_in = best["start_index"] / sample_fps
    edit_out = edit_in + requested
    if edit_out > source_duration + 1e-6:
        edit_out = source_duration
        edit_in = edit_out - requested
    hint = dict(edit_hint or {})
    selection = {
        "in_seconds": round(edit_in, 6), "out_seconds": round(edit_out, 6),
        "duration_seconds": round(requested, 6),
        "reason": str(hint.get("preferred_moment") or "highest validated motion window").strip(),
        "metrics": {
            "window_mean_luma_change": best["mean_change"],
            "window_min_luma_change": best["min_change"],
            "window_max_luma_change": best["max_change"],
            "candidate_count": len(candidates), "viable_candidate_count": len(viable),
            "protected_range_count": len(protected),
        },
        "selector": {"name": EDIT_SELECTOR_NAME, "version": EDIT_SELECTOR_VERSION},
        "source_artifact_sha256": str(source_artifact_sha256),
        "source_decoded_visual_sha256": str(analysis.get("decoded_visual_sha256") or ""),
    }
    selection["selection_sha256"] = _selection_hash(selection)
    return selection


def _normalized_dialogue_cues(cues: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cue in cues:
        start = float(cue.get("start_seconds", cue.get("start_s", 0)) or 0)
        end = float(cue.get("end_seconds", cue.get("end_s", start)) or start)
        result.append({
            "speaker_id": str(cue.get("speaker_id") or cue.get("speaker") or ""),
            "text": str(cue.get("text") or ""),
            "start_seconds": round(start, 6), "end_seconds": round(end, 6),
        })
    return result


def dialogue_cues_sha256(cues: Iterable[Mapping[str, Any]]) -> str:
    """Hash exact cue identity/timing for a re-based native dialogue lane."""
    return hashlib.sha256(json.dumps(
        _normalized_dialogue_cues(cues), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def build_manual_edit_selection(
    analysis: Mapping[str, Any], *, source_duration_seconds: float,
    requested_duration_seconds: float, source_artifact_sha256: str,
    in_seconds: float, reason: str, reviewed_by: str,
    dialogue_audio_alignment: Iterable[Mapping[str, Any]] = (),
    current_dialogue_cues: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a hash-bound reviewer window without pretending it was automatic.

    This is intentionally separate from :func:`select_edit_window`: a human may
    choose a lower-motion interval because it contains the contract action.
    The selected bytes are still decoded and content-QA checked by the caller.
    """
    source_duration = float(source_duration_seconds)
    requested = float(requested_duration_seconds)
    edit_in = round(float(in_seconds), 6)
    edit_out = round(edit_in + requested, 6)
    reason = str(reason or "").strip()
    reviewed_by = str(reviewed_by or "").strip()
    if not MIN_EDIT_DURATION_SECONDS <= requested <= MAX_EDIT_DURATION_SECONDS:
        raise ValueError(
            f"edit duration must be between {MIN_EDIT_DURATION_SECONDS} and {MAX_EDIT_DURATION_SECONDS} seconds"
        )
    if edit_in < 0 or edit_out > source_duration + 1e-6:
        raise ValueError("manual edit range is outside source duration")
    if not reason or not reviewed_by:
        raise ValueError("manual edit selection requires reviewer and reason")
    metrics = analysis.get("metrics") or {}
    selection: dict[str, Any] = {
        "in_seconds": edit_in,
        "out_seconds": edit_out,
        "duration_seconds": round(requested, 6),
        "reason": reason,
        "metrics": {
            "manual_contract_review": True,
            "sample_count": int(metrics.get("sample_count") or 0),
            "mean_adjacent_luma_change": float(metrics.get("mean_adjacent_luma_change") or 0),
            "first_last_luma_change": float(metrics.get("first_last_luma_change") or 0),
        },
        "selector": {"name": MANUAL_SELECTOR_NAME, "version": MANUAL_SELECTOR_VERSION},
        "source_artifact_sha256": str(source_artifact_sha256),
        "source_decoded_visual_sha256": str(analysis.get("decoded_visual_sha256") or ""),
        "manual_review": {"reviewed_by": reviewed_by, "reason": reason},
    }
    alignment = [dict(item) for item in dialogue_audio_alignment]
    if alignment:
        selection["dialogue_audio_alignment"] = alignment
        if current_dialogue_cues is not None:
            selection["dialogue_cues_sha256"] = dialogue_cues_sha256(current_dialogue_cues)
    selection["selection_sha256"] = _selection_hash(selection)
    return selection


def validate_edit_selection(
    selection: Mapping[str, Any], *, source_artifact_sha256: str,
    requested_duration_seconds: float, source_duration_seconds: float,
    current_dialogue_cues: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    current = dict(selection or {})
    expected_hash = _selection_hash(current)
    errors = []
    try:
        edit_in = float(current.get("in_seconds"))
        edit_out = float(current.get("out_seconds"))
        duration = float(current.get("duration_seconds"))
    except (TypeError, ValueError):
        edit_in = edit_out = duration = -1.0
        errors.append("selection timing is missing or invalid")
    if edit_in < 0 or edit_out <= edit_in or edit_out > float(source_duration_seconds) + 1e-6:
        errors.append("selection range is outside source duration")
    if abs(duration - (edit_out - edit_in)) > 1e-4:
        errors.append("selection duration does not equal out-in")
    if abs(duration - float(requested_duration_seconds)) > 1e-4:
        errors.append("selection duration does not equal approved edit duration")
    if current.get("source_artifact_sha256") != str(source_artifact_sha256):
        errors.append("selection source artifact hash is stale")
    if current.get("selection_sha256") != expected_hash:
        errors.append("selection hash is missing or stale")
    selector = current.get("selector") or {}
    if not selector.get("name") or not selector.get("version"):
        errors.append("selection algorithm identity is missing")
    alignments = [item for item in (current.get("dialogue_audio_alignment") or []) if isinstance(item, Mapping)]
    if len(alignments) != len(current.get("dialogue_audio_alignment") or []):
        errors.append("dialogue audio alignment contains a non-object entry")
    seen_cue_indexes: set[int] = set()
    source_ranges: list[tuple[float, float]] = []
    target_ranges: list[tuple[float, float]] = []
    for index, item in enumerate(alignments):
        try:
            cue_index = int(item.get("cue_index"))
            source_start = float(item.get("source_start_seconds"))
            source_end = float(item.get("source_end_seconds"))
            target_start = float(item.get("target_start_seconds"))
            target_end = float(item.get("target_end_seconds"))
        except (AttributeError, TypeError, ValueError):
            errors.append(f"dialogue alignment {index} timing is invalid")
            continue
        if source_start < 0 or source_end <= source_start or source_end > float(source_duration_seconds) + 1e-6:
            errors.append(f"dialogue alignment {index} source range is invalid")
        if source_start < edit_out - 1e-6 and source_end > edit_in + 1e-6:
            errors.append(f"dialogue alignment {index} would duplicate native speech already inside selection")
        if target_start < 0 or target_end <= target_start or target_end > duration + 1e-6:
            errors.append(f"dialogue alignment {index} target range is invalid")
        if abs((source_end - source_start) - (target_end - target_start)) > 1e-4:
            errors.append(f"dialogue alignment {index} cannot time-stretch approved speech")
        source_ranges.append((source_start, source_end))
        target_ranges.append((target_start, target_end))
        if str(item.get("contract") or "") != "source-dialogue-rebase/v1":
            errors.append(f"dialogue alignment {index} contract is missing")
        if cue_index < 0 or cue_index in seen_cue_indexes:
            errors.append(f"dialogue alignment {index} cue index is invalid or duplicated")
        seen_cue_indexes.add(cue_index)
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("text_sha256") or "").lower()):
            errors.append(f"dialogue alignment {index} text binding is missing")
        if str(item.get("audio_authority") or "") != "relocated_native_h3_dialogue":
            errors.append(f"dialogue alignment {index} audio authority is invalid")
        evidence = item.get("source_audio_evidence")
        if not isinstance(evidence, Mapping):
            errors.append(f"dialogue alignment {index} native audio evidence is missing")
            continue
        try:
            evidence_start = float(evidence.get("source_start_seconds"))
            evidence_end = float(evidence.get("source_end_seconds"))
            audible_fraction = float(evidence.get("audible_fraction"))
            max_silence = float(evidence.get("max_silence_seconds"))
        except (TypeError, ValueError):
            errors.append(f"dialogue alignment {index} native audio evidence is invalid")
            continue
        if str(evidence.get("contract") or "") != AUDIO_EVIDENCE_CONTRACT:
            errors.append(f"dialogue alignment {index} native audio evidence contract is invalid")
        if str(evidence.get("source_artifact_sha256") or "") != str(source_artifact_sha256):
            errors.append(f"dialogue alignment {index} native audio evidence artifact is stale")
        if abs(evidence_start - source_start) > 1e-4 or abs(evidence_end - source_end) > 1e-4:
            errors.append(f"dialogue alignment {index} native audio evidence range is stale")
        if evidence.get("eligible_for_native_dialogue_rebase") is not True:
            errors.append(f"dialogue alignment {index} native audio is not eligible for relocation")
        if audible_fraction < MIN_DIALOGUE_AUDIBLE_FRACTION or max_silence > MAX_DIALOGUE_SILENCE_SECONDS:
            errors.append(f"dialogue alignment {index} native audio evidence fails audibility policy")
    for label, ranges in (("source", source_ranges), ("target", target_ranges)):
        for left_index, (left_start, left_end) in enumerate(ranges):
            for right_start, right_end in ranges[left_index + 1:]:
                if left_start < right_end - 1e-6 and right_start < left_end - 1e-6:
                    errors.append(f"dialogue alignment {label} intervals overlap")
                    break
    if current_dialogue_cues is not None and alignments:
        try:
            cues = _normalized_dialogue_cues(current_dialogue_cues)
        except (AttributeError, TypeError, ValueError) as exc:
            errors.append(f"current dialogue cues are invalid: {exc}")
            cues = []
        expected_digest = dialogue_cues_sha256(cues)
        if current.get("dialogue_cues_sha256") != expected_digest:
            errors.append("dialogue cue digest is missing or stale")
        if len(alignments) != len(cues) or seen_cue_indexes != set(range(len(cues))):
            errors.append("dialogue alignment cue coverage does not match current dialogue")
        for alignment in alignments:
            try:
                cue_index = int(alignment.get("cue_index"))
                cue = cues[cue_index]
            except (IndexError, TypeError, ValueError):
                continue
            if (
                str(alignment.get("text_sha256") or "").lower()
                != hashlib.sha256(cue["text"].encode("utf-8")).hexdigest()
                or str(alignment.get("speaker_id") or "") != cue["speaker_id"]
                or abs(float(alignment.get("source_start_seconds")) - cue["start_seconds"]) > 1e-4
                or abs(float(alignment.get("source_end_seconds")) - cue["end_seconds"]) > 1e-4
            ):
                errors.append(f"dialogue alignment {cue_index} does not match current dialogue cue")
    return {"valid": not errors, "errors": errors, "selection_sha256": expected_hash}


__all__ = [
    "ALGORITHM_NAME", "ALGORITHM_VERSION", "analyze_video", "compare_analyses",
    "evaluate_content", "select_edit_window", "build_manual_edit_selection",
    "analyze_native_dialogue_audio", "AUDIO_EVIDENCE_CONTRACT", "dialogue_cues_sha256",
    "validate_edit_selection",
]
