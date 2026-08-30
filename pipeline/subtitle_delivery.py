"""Deterministic subtitle derivation and SRT/VTT/ASS delivery.

The approved ``spoken_dialogue`` lane is the source of truth.  An optional
``subtitle_timeline`` may mirror it for editorial review, but conflicting text
or timing is rejected instead of silently changing approved dialogue.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


FPS = 24
DEFAULT_ASS_PLAY_RES_X = 1280
DEFAULT_ASS_PLAY_RES_Y = 720
DEFAULT_LANDSCAPE_MARGIN_V = 72
DEFAULT_PORTRAIT_MARGIN_V = 256


def _seconds(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _time_range(value: Any) -> tuple[float, float] | None:
    match = re.fullmatch(
        r"\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-–—~至]\s*(\d+(?:\.\d+)?)\s*(?:s|秒)?\s*",
        str(value or ""),
        re.IGNORECASE,
    )
    return (float(match.group(1)), float(match.group(2))) if match else None


def normalize_cue(cue: Mapping[str, Any], *, fps: int = FPS) -> dict[str, Any]:
    parsed = _time_range(cue.get("time_range") or cue.get("time"))
    start = _seconds(cue.get("start_seconds", cue.get("start_s")), parsed[0] if parsed else 0.0)
    end = _seconds(cue.get("end_seconds", cue.get("end_s")), parsed[1] if parsed else start)
    start_frame = max(0, round(start * fps))
    end_frame = max(start_frame + 1, round(end * fps))
    return {
        "speaker_id": str(cue.get("speaker_id") or cue.get("speaker") or "").strip(),
        "text": str(cue.get("text") or cue.get("line") or "").strip(),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_seconds": start_frame / fps,
        "end_seconds": end_frame / fps,
    }


def panel_subtitle_cues(
    panel: Mapping[str, Any], *, fps: int = FPS, strict: bool = True
) -> tuple[list[dict[str, Any]], list[str]]:
    package = panel.get("prompt_package") or {}
    spoken_raw = panel.get("spoken_dialogue")
    if spoken_raw is None and isinstance(package, Mapping):
        spoken_raw = package.get("spoken_dialogue_timeline")
    spoken = [normalize_cue(cue, fps=fps) for cue in (spoken_raw or []) if isinstance(cue, Mapping)]
    spoken = [cue for cue in spoken if cue["text"]]

    explicit_raw = panel.get("subtitle_timeline")
    if explicit_raw is None and isinstance(package, Mapping):
        explicit_raw = package.get("subtitle_timeline")
    warnings: list[str] = []
    if explicit_raw is not None:
        explicit = [normalize_cue(cue, fps=fps) for cue in explicit_raw if isinstance(cue, Mapping)]
        explicit = [cue for cue in explicit if cue["text"]]
        comparable_spoken = [
            (cue["text"], cue["start_frame"], cue["end_frame"]) for cue in spoken
        ]
        comparable_explicit = [
            (cue["text"], cue["start_frame"], cue["end_frame"]) for cue in explicit
        ]
        if comparable_explicit != comparable_spoken:
            message = "subtitle_timeline conflicts with approved spoken_dialogue"
            if strict:
                raise ValueError(message)
            warnings.append(message)
    return spoken, warnings


def build_episode_cues(
    episode: Mapping[str, Any], jobs: Iterable[Mapping[str, Any]], *, fps: int = FPS,
    strict: bool = True,
) -> dict[str, Any]:
    panels = episode.get("panels") or []
    ordered_jobs = sorted(jobs, key=lambda item: int(item.get("panel_index") or 0))
    offset = 0.0
    result: list[dict[str, Any]] = []
    warnings: list[str] = []
    for job in ordered_jobs:
        panel_index = int(job.get("panel_index") or 0)
        if panel_index < 1 or panel_index > len(panels):
            raise ValueError(f"subtitle job panel_index is outside episode.panels: {panel_index}")
        panel = panels[panel_index - 1]
        cues, panel_warnings = panel_subtitle_cues(panel, fps=fps, strict=strict)
        warnings.extend(f"panel {panel_index}: {message}" for message in panel_warnings)
        metadata = job.get("metadata") or {}
        selection = metadata.get("edit_selection") if isinstance(metadata.get("edit_selection"), Mapping) else {}
        selection_valid = False
        try:
            edit_in = float(selection.get("in_seconds"))
            edit_out = float(selection.get("out_seconds"))
            selected_duration = float(selection.get("duration_seconds"))
            selection_valid = bool(
                edit_in >= 0 and edit_out > edit_in
                and abs(selected_duration - (edit_out - edit_in)) <= 1e-4
            )
        except (TypeError, ValueError):
            edit_in = edit_out = selected_duration = 0.0
        alignment_by_index = {
            int(item.get("cue_index")): item
            for item in (selection.get("dialogue_audio_alignment") or [])
            if isinstance(item, Mapping) and str(item.get("cue_index", "")).isdigit()
        }
        for cue_index, cue in enumerate(cues):
            shifted = dict(cue)
            shifted["panel_index"] = panel_index
            local_start = float(cue["start_seconds"])
            local_end = float(cue["end_seconds"])
            alignment = alignment_by_index.get(cue_index)
            if selection_valid and alignment:
                try:
                    local_start = float(alignment.get("target_start_seconds"))
                    local_end = float(alignment.get("target_end_seconds"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"dialogue audio alignment {cue_index} timing is invalid") from exc
                expected_text_hash = hashlib.sha256(cue["text"].encode("utf-8")).hexdigest()
                if (
                    str(alignment.get("contract") or "") != "source-dialogue-rebase/v1"
                    or str(alignment.get("audio_authority") or "") != "relocated_native_h3_dialogue"
                    or str(alignment.get("text_sha256") or "").lower() != expected_text_hash
                    or (str(alignment.get("speaker_id") or "") and str(alignment.get("speaker_id")) != cue["speaker_id"])
                    or local_start < -1e-6 or local_end <= local_start
                    or local_end > selected_duration + 1e-6
                ):
                    message = f"dialogue audio alignment {cue_index} does not bind this approved subtitle cue"
                    if strict:
                        raise ValueError(message)
                    warnings.append(f"panel {panel_index}: {message}")
                    continue
                shifted["delivery_alignment"] = "relocated_native_dialogue"
            elif selection_valid:
                if local_start < edit_in - 1e-6 or local_end > edit_out + 1e-6:
                    message = (
                        f"approved spoken_dialogue cue {cue_index} falls outside edit_selection "
                        "without an approved audio alignment"
                    )
                    if strict:
                        raise ValueError(message)
                    warnings.append(f"panel {panel_index}: {message}")
                    continue
                local_start -= edit_in
                local_end -= edit_in
                shifted["delivery_alignment"] = "selected_native_audio"
            shifted["source_start_seconds"] = cue["start_seconds"]
            shifted["source_end_seconds"] = cue["end_seconds"]
            shifted["start_seconds"] = local_start + offset
            shifted["end_seconds"] = local_end + offset
            result.append(shifted)
        probe = job.get("probe") or {}
        settings = metadata.get("settings") or {}
        duration = selected_duration if selection_valid else _seconds(
            probe.get("duration_seconds"),
            _seconds(metadata.get("actual_duration_seconds"), _seconds(settings.get("duration_seconds"), 0.0)),
        )
        if duration <= 0:
            raise ValueError(f"panel {panel_index} has no validated duration for subtitle offset")
        offset += duration
    return {"fps": fps, "duration_seconds": offset, "cues": result, "warnings": warnings}


def _timestamp(seconds: float, separator: str) -> str:
    millis = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _write_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8-sig")
    temporary.replace(path)
    return path


def write_srt(cues: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    lines: list[str] = []
    for index, cue in enumerate(cues, 1):
        lines.extend([
            str(index),
            f"{_timestamp(cue['start_seconds'], ',')} --> {_timestamp(cue['end_seconds'], ',')}",
            str(cue["text"]),
            "",
        ])
    return _write_atomic(Path(path), "\n".join(lines))


def write_vtt(cues: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    lines = ["WEBVTT", ""]
    for index, cue in enumerate(cues, 1):
        lines.extend([
            str(index),
            f"{_timestamp(cue['start_seconds'], '.')} --> {_timestamp(cue['end_seconds'], '.')}",
            str(cue["text"]),
            "",
        ])
    return _write_atomic(Path(path), "\n".join(lines))


def _ass_timestamp(seconds: float) -> str:
    centis = max(0, round(float(seconds) * 100))
    hours, remainder = divmod(centis, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_canvas(
    play_res_x: int = DEFAULT_ASS_PLAY_RES_X,
    play_res_y: int = DEFAULT_ASS_PLAY_RES_Y,
    margin_v: int | None = None,
) -> dict[str, int]:
    """Resolve a validated ASS canvas for the unified 720p delivery family."""
    width = int(play_res_x)
    height = int(play_res_y)
    if width <= 0 or height <= 0:
        raise ValueError("ASS play resolution must use positive dimensions")
    if margin_v is None:
        safe_margin = DEFAULT_PORTRAIT_MARGIN_V if height > width else DEFAULT_LANDSCAPE_MARGIN_V
    else:
        safe_margin = int(margin_v)
    if safe_margin < 0 or safe_margin >= height:
        raise ValueError(f"ASS MarginV must be in the range 0..{height - 1}")
    return {
        "play_res_x": width,
        "play_res_y": height,
        "safe_margin_bottom_px": safe_margin,
    }


def write_ass(
    cues: Iterable[Mapping[str, Any]], path: str | Path, *,
    play_res_x: int = DEFAULT_ASS_PLAY_RES_X,
    play_res_y: int = DEFAULT_ASS_PLAY_RES_Y,
    margin_v: int | None = None,
) -> Path:
    canvas = _ass_canvas(play_res_x, play_res_y, margin_v)
    short_edge = min(canvas["play_res_x"], canvas["play_res_y"])
    font_size = max(24, round(short_edge * 0.05))
    side_margin = max(24, round(canvas["play_res_x"] * 0.05))
    outline = max(1, round(short_edge / 360))
    shadow = max(1, round(short_edge / 720))
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {canvas['play_res_x']}
PlayResY: {canvas['play_res_y']}
WrapStyle: 0

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Microsoft YaHei,{font_size},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,{side_margin},{side_margin},{canvas['safe_margin_bottom_px']},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = [header.rstrip()]
    for cue in cues:
        text = str(cue["text"]).replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")
        speaker = str(cue.get("speaker_id") or "").replace(",", " ")
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(cue['start_seconds'])},{_ass_timestamp(cue['end_seconds'])},Default,{speaker},0,0,0,,{text}"
        )
    return _write_atomic(Path(path), "\n".join(lines) + "\n")


def write_subtitle_bundle(
    episode: Mapping[str, Any], jobs: Iterable[Mapping[str, Any]], base_path: str | Path,
    *, strict: bool = True,
    play_res_x: int = DEFAULT_ASS_PLAY_RES_X,
    play_res_y: int = DEFAULT_ASS_PLAY_RES_Y,
    margin_v: int | None = None,
) -> dict[str, Any]:
    timeline = build_episode_cues(episode, jobs, strict=strict)
    base = Path(base_path)
    cues = timeline["cues"]
    canvas = _ass_canvas(play_res_x, play_res_y, margin_v)
    return {
        **timeline,
        "subtitle_canvas": canvas,
        "srt_path": str(write_srt(cues, base.with_suffix(".srt"))),
        "vtt_path": str(write_vtt(cues, base.with_suffix(".vtt"))),
        "ass_path": str(write_ass(
            cues, base.with_suffix(".ass"),
            play_res_x=canvas["play_res_x"], play_res_y=canvas["play_res_y"],
            margin_v=canvas["safe_margin_bottom_px"],
        )),
    }


__all__ = [
    "normalize_cue", "panel_subtitle_cues", "build_episode_cues",
    "write_srt", "write_vtt", "write_ass", "write_subtitle_bundle",
]
