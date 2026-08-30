"""Explicit, deterministic continuity-safe rendering for rejected H3 shots.

This is a reviewer-selected fallback, never an automatic quality downgrade.  A
human-approved group-composition anchor is animated with a very small camera
move; approved dialogue is synthesized through an audited Windows SAPI voice.
Deterministic SRT/VTT/ASS sidecars are produced beside the clip, but subtitles
are never burned into a panel artifact.  The final delivery export owns the one
and only subtitle burn.  A clip is committed as ``succeeded`` only after
ffprobe validates the encoded artifact.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from PIL import Image

from atomic_io import write_json_atomic
from runtime_config import ffmpeg_executable, ffprobe_executable, projects_dir
from subtitle_delivery import panel_subtitle_cues, write_ass, write_srt, write_vtt
from task_store import (
    RenderJobStore, default_store, prepare_episode, production_gate, project_snapshot,
)
from video_delivery import probe_media, validate_probe
from video_quality import analyze_video, evaluate_content, select_edit_window, validate_edit_selection


SAFE_FPS = 24
SAFE_FRAME_COUNT = 243
SAFE_DURATION_SECONDS = SAFE_FRAME_COUNT / SAFE_FPS
DEFAULT_SAPI_VOICE = "Microsoft Huihui Desktop"
SAFE_MOTIONS = {"slow_push", "locked"}


def _utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(ep_id: str) -> Path:
    return Path(project_snapshot(ep_id)["project_dir"]).resolve()


def _inside_project(path: str | Path, project: Path, *, require_file: bool = True) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = project / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(project)
    except ValueError as exc:
        raise ValueError(f"continuity-safe path must be inside episode project: {candidate}") from exc
    if require_file and (not candidate.is_file() or candidate.stat().st_size <= 0):
        raise FileNotFoundError(f"continuity-safe file is missing or empty: {candidate}")
    return candidate


def approve_continuity_anchor(
    ep_id: str,
    job_id: str,
    source_anchor: str | Path,
    *,
    reason: str,
    approved_by: str = "reviewer",
    store: RenderJobStore | None = None,
) -> dict[str, Any]:
    """Persist an independent visual-state anchor approval for exactly one job."""
    store = store or default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    if job["status"] != "failed":
        raise RuntimeError("continuity anchor can only be approved for an explicitly failed job")
    reason = str(reason).strip()
    approved_by = str(approved_by).strip()
    if not reason or not approved_by:
        raise ValueError("continuity anchor approval requires reason and approved_by")
    project = _project_path(ep_id)
    anchor = _inside_project(source_anchor, project)
    if anchor.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"continuity anchor must be an image: {anchor}")
    # Synchronize current continuity dependency/artifact hashes before binding
    # the approval. An earlier QA failure may predate the accepted predecessor
    # artifact being written to the dependency hash.
    episode_path = project / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    previous_metadata = dict(job.get("metadata") or {})
    previous_error = job.get("error")
    prepare_episode(ep_id, episode)
    refreshed = store.get_job(job_id, ep_id=ep_id)
    if not refreshed:
        raise RuntimeError("continuity-safe job disappeared during registration")
    if refreshed["status"] != "failed":
        refreshed_metadata = {**previous_metadata, **dict(refreshed.get("metadata") or {})}
        refreshed = store.update_job(
            job_id, status="failed", progress=0.0, prompt_id=None,
            error=previous_error or "awaiting explicit continuity-safe render",
            metadata=refreshed_metadata, probe={}, completed_at=None,
        )
    job = refreshed
    panels = episode.get("panels") or []
    panel_index = int(job.get("panel_index") or 0)
    if panel_index < 1 or panel_index > len(panels):
        raise RuntimeError("continuity-safe panel is missing from the approved contract")
    action, final_state = _panel_visual_intent(panels[panel_index - 1])
    if not action or not final_state:
        raise RuntimeError(
            "continuity-safe approval requires explicit panel action and approved final_state"
        )
    anchor_sha = _sha256_file(anchor)
    for other in store.list_jobs(ep_id):
        if str(other.get("job_id")) == job_id:
            continue
        other_approval = ((other.get("metadata") or {}).get("continuity_safe_anchor_approval") or {})
        if other_approval.get("approved") and other_approval.get("sha256") == anchor_sha:
            raise RuntimeError(
                "continuity-safe visual-state anchor was already approved for another panel"
            )
    approval = {
        "approved": True,
        "source_anchor": str(anchor),
        "sha256": anchor_sha,
        "approved_at": _utc_now(),
        "approved_by": approved_by,
        "reason": reason,
        "approval_scope": "single_panel_visual_state_anchor",
        "approved_job_id": job_id,
        "panel_action": action,
        "approved_final_state": final_state,
        "job_input_hash": job.get("input_hash"),
    }
    metadata = dict(job.get("metadata") or {})
    history = list(metadata.get("continuity_safe_anchor_history") or [])
    history.append({**approval, "action": "approved"})
    metadata["continuity_safe_anchor_history"] = history[-50:]
    metadata["continuity_safe_anchor_approval"] = approval
    updated = store.update_job(job_id, metadata=metadata)
    return {"ep_id": ep_id, "job": updated, "anchor_approval": approval}


def _panel_visual_intent(panel: Mapping[str, Any]) -> tuple[str, str]:
    package = panel.get("prompt_package") if isinstance(panel.get("prompt_package"), Mapping) else {}
    action = str(
        panel.get("action") or panel.get("motion") or package.get("action") or ""
    ).strip()
    final_state = str(
        panel.get("final_state") or panel.get("approved_final_state")
        or package.get("final_state") or ""
    ).strip()
    return action, final_state


_SAPI_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$request = Get-Content -LiteralPath $env:AI_MANGA_TTS_REQUEST -Raw -Encoding UTF8 | ConvertFrom-Json
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
  $voices = @($synth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo })
  if ($voices.Count -eq 0) { throw 'No enabled Windows SAPI voices are installed.' }
  $selected = $null
  if ($request.preferred_voice) {
    $selected = $voices | Where-Object { $_.Name -eq [string]$request.preferred_voice } | Select-Object -First 1
  }
  if (-not $selected) {
    $selected = $voices | Where-Object { $_.Name -eq 'Microsoft Huihui Desktop' } | Select-Object -First 1
  }
  if (-not $selected) {
    $selected = $voices | Where-Object { $_.Culture.Name -eq 'zh-CN' } | Select-Object -First 1
  }
  if (-not $selected) { $selected = $voices | Select-Object -First 1 }
  $synth.SelectVoice($selected.Name)
  $synth.Rate = [int]$request.rate
  $synth.Volume = 100
  $synth.SetOutputToWaveFile([string]$request.output_path)
  $synth.Speak([string]$request.text)
  $synth.SetOutputToNull()
  $audit = [ordered]@{
    engine = 'windows_sapi'
    selected_voice = $selected.Name
    selected_culture = $selected.Culture.Name
    requested_voice = [string]$request.preferred_voice
    fallback_used = ([string]$request.preferred_voice -ne $selected.Name)
    available_voices = @($voices | ForEach-Object { [ordered]@{ name = $_.Name; culture = $_.Culture.Name } })
    voice_fidelity = 'single_system_voice_not_character_voice_cloning'
  }
  [System.IO.File]::WriteAllText(
    $env:AI_MANGA_TTS_RESPONSE,
    ($audit | ConvertTo-Json -Depth 5 -Compress),
    (New-Object System.Text.UTF8Encoding($false))
  )
}
finally {
  $synth.Dispose()
}
"""


def synthesize_windows_sapi(
    text: str,
    output_path: str | Path,
    *,
    preferred_voice: str = DEFAULT_SAPI_VOICE,
    rate: int = 8,
    powershell: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Synthesize one cue and return the exact installed-voice audit."""
    line = str(text).strip()
    if not line:
        raise ValueError("TTS text cannot be empty")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    request_path = output.parent / f".{output.name}.{token}.request.json"
    response_path = output.parent / f".{output.name}.{token}.response.json"
    write_json_atomic(request_path, {
        "text": line,
        "output_path": str(output),
        "preferred_voice": str(preferred_voice or DEFAULT_SAPI_VOICE),
        "rate": max(-10, min(10, int(rate))),
    })
    executable = powershell or shutil.which("powershell.exe") or shutil.which("powershell")
    if not executable:
        request_path.unlink(missing_ok=True)
        raise FileNotFoundError("Windows PowerShell is required for the audited SAPI TTS engine")
    encoded = base64.b64encode(_SAPI_SCRIPT.encode("utf-16le")).decode("ascii")
    env = os.environ.copy()
    env["AI_MANGA_TTS_REQUEST"] = str(request_path)
    env["AI_MANGA_TTS_RESPONSE"] = str(response_path)
    try:
        runner(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            check=True, capture_output=True, text=True, env=env,
        )
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("Windows SAPI did not produce a non-empty WAV file")
        if not response_path.is_file():
            raise RuntimeError("Windows SAPI did not produce a voice audit")
        audit = json.loads(response_path.read_text(encoding="utf-8-sig"))
        if audit.get("engine") != "windows_sapi" or not audit.get("selected_voice"):
            raise RuntimeError("Windows SAPI returned an incomplete voice audit")
        return audit
    finally:
        request_path.unlink(missing_ok=True)
        response_path.unlink(missing_ok=True)


def probe_audio_duration(
    path: str | Path,
    *,
    ffprobe: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> float:
    audio = Path(path).resolve()
    proc = runner(
        [
            ffprobe or ffprobe_executable(), "-v", "error",
            "-show_entries", "format=duration", "-of", "json", str(audio),
        ],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout)
    duration = float((payload.get("format") or {}).get("duration") or 0)
    if duration <= 0:
        raise ValueError(f"TTS WAV has invalid duration: {audio}")
    return duration


def _strict_chain(jobs: Iterable[Mapping[str, Any]], root_job_id: str) -> list[dict[str, Any]]:
    job_list = [dict(job) for job in jobs]
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for job in job_list:
        dependency = (((job.get("metadata") or {}).get("inputs") or {}).get("continuity_dependency") or {})
        if dependency.get("strict") and dependency.get("previous_job_id"):
            by_parent.setdefault(str(dependency["previous_job_id"]), []).append(job)
    result: list[dict[str, Any]] = []
    pending = [root_job_id]
    seen = {root_job_id}
    while pending:
        parent = pending.pop(0)
        for child in sorted(by_parent.get(parent, []), key=lambda item: int(item["panel_index"])):
            child_id = str(child["job_id"])
            if child_id in seen:
                continue
            seen.add(child_id)
            result.append(child)
            pending.append(child_id)
    return result


def _output_spec(anchor: Path) -> dict[str, int]:
    """Keep the approved anchor's native even canvas (for example 608x1056)."""
    with Image.open(anchor) as image:
        width, height = image.size
    width -= width % 2
    height -= height % 2
    if width < 64 or height < 64:
        raise ValueError(f"approved continuity anchor is too small: {width}x{height}")
    if height > width:
        margin = max(64, round(200 * height / 1920))
    else:
        margin = max(32, round(64 * height / 1080))
    return {"width": width, "height": height, "margin_v": min(margin, height - 1)}


def _atempo_chain(ratio: float) -> list[str]:
    factors: list[str] = []
    remaining = max(1.0, float(ratio))
    while remaining > 2.0:
        factors.append("atempo=2.0")
        remaining /= 2.0
    if remaining > 1.0001:
        factors.append(f"atempo={remaining:.6f}")
    return factors


def _build_ffmpeg_command(
    *,
    anchor: Path,
    temporary_output: Path,
    cue_audio: list[dict[str, Any]],
    width: int,
    height: int,
    motion: str,
    ffmpeg: str,
) -> list[str]:
    if motion not in SAFE_MOTIONS:
        raise ValueError(f"unsupported continuity-safe motion: {motion}")
    command = [ffmpeg, "-y", "-loop", "1", "-framerate", str(SAFE_FPS), "-i", str(anchor)]
    for cue in cue_audio:
        command.extend(["-i", str(cue["path"])])
    ambient_index = len(cue_audio) + 1
    command.extend([
        "-f", "lavfi", "-t", f"{SAFE_DURATION_SECONDS:.6f}",
        "-i", "anoisesrc=color=pink:amplitude=0.012:r=48000",
    ])

    base_video = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )
    if motion == "slow_push":
        base_video += (
            f",zoompan=z='min(zoom+0.000062,1.015)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={SAFE_FPS}"
        )
    else:
        base_video += f",fps={SAFE_FPS}"
    base_video += f",trim=duration={SAFE_DURATION_SECONDS:.6f},setpts=PTS-STARTPTS[vbase]"
    filters = [base_video]
    # Panel artifacts stay text-free.  ASS is generated as an auditable sidecar
    # and consumed only by ``video_delivery.export_episode`` after concat/resize,
    # so a Web caller cannot accidentally burn the same subtitle twice.
    filters.append("[vbase]null[vout]")

    cue_labels: list[str] = []
    for index, cue in enumerate(cue_audio, 1):
        cue_duration = float(cue["end_seconds"]) - float(cue["start_seconds"])
        if cue_duration <= 0:
            raise ValueError("continuity-safe dialogue cue has non-positive duration")
        source_duration = float(cue["source_duration"])
        audio_filters = [f"[{index}:a]aresample=48000"]
        audio_filters.extend(_atempo_chain(source_duration / cue_duration))
        delay_ms = max(0, round(float(cue["start_seconds"]) * 1000))
        audio_filters.extend([
            f"apad", f"atrim=0:{cue_duration:.6f}",
            f"adelay={delay_ms}:all=1[cue_{index}]",
        ])
        filters.append(",".join(audio_filters))
        cue_labels.append(f"[cue_{index}]")
    filters.append(
        f"[{ambient_index}:a]volume=0.16,atrim=0:{SAFE_DURATION_SECONDS:.6f}[ambient]"
    )
    if cue_labels:
        filters.append(
            "[ambient]" + "".join(cue_labels)
            + f"amix=inputs={len(cue_labels) + 1}:duration=longest:normalize=0,"
            + f"apad,atrim=0:{SAFE_DURATION_SECONDS:.6f},"
            + "loudnorm=I=-16:TP=-1.5:LRA=11,aformat=sample_rates=48000:channel_layouts=stereo[aout]"
        )
    else:
        filters.append(
            "[ambient]loudnorm=I=-24:TP=-3:LRA=7,"
            "aformat=sample_rates=48000:channel_layouts=stereo[aout]"
        )
    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[vout]", "-map", "[aout]", "-t", f"{SAFE_DURATION_SECONDS:.6f}",
        "-r", str(SAFE_FPS), "-c:v", "libx264", "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(temporary_output),
    ])
    return command


def _replace_with_retry(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.025 * (2 ** attempt), 0.25))


def run_continuity_safe_chain(
    ep_id: str,
    job_id: str,
    *,
    preferred_voice: str = DEFAULT_SAPI_VOICE,
    motion: str = "slow_push",
    burn_subtitles: bool | None = None,
    store: RenderJobStore | None = None,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    probe_func: Callable[..., dict[str, Any]] = probe_media,
    tts_func: Callable[..., dict[str, Any]] = synthesize_windows_sapi,
    tts_runner: Callable[..., Any] = subprocess.run,
    audio_probe_func: Callable[..., float] = probe_audio_duration,
    quality_analyzer: Callable[..., dict[str, Any]] = analyze_video,
    quality_evaluator: Callable[..., dict[str, Any]] = evaluate_content,
    edit_selector: Callable[..., dict[str, Any]] = select_edit_window,
    quality_runner: Callable[..., Any] = subprocess.run,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Render exactly one failed job from its independently approved anchor.

    ``burn_subtitles`` is retained only so older Web/facade callers do not
    break.  Its value is intentionally ignored: continuity-safe panel clips are
    always clean masters and final delivery performs the optional single burn.
    """
    if motion not in SAFE_MOTIONS:
        raise ValueError(f"unsupported continuity-safe motion: {motion}")
    store = store or default_store()
    target = store.get_job(job_id, ep_id=ep_id)
    if not target:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    if target["status"] != "failed":
        raise RuntimeError("continuity-safe mode must be explicitly started from a failed job")
    gate = production_gate(ep_id)
    if not gate["ready"]:
        raise RuntimeError("continuity-safe production gate blocked: " + ",".join(gate["reasons"]))
    metadata = dict(target.get("metadata") or {})
    approval = dict(metadata.get("continuity_safe_anchor_approval") or {})
    if not approval.get("approved"):
        raise RuntimeError("continuity-safe mode requires an explicitly approved visual-state anchor")
    project = _project_path(ep_id)
    anchor = _inside_project(str(approval.get("source_anchor") or ""), project)
    if _sha256_file(anchor) != approval.get("sha256"):
        raise RuntimeError("approved continuity anchor bytes changed after review")
    episode_path = project / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    if approval.get("job_input_hash") != target.get("input_hash"):
        raise RuntimeError("continuity anchor approval is stale for the current job input")
    # Rebuild semantic hashes before the fallback. Rejected descendants may
    # intentionally have ``input_hash=None`` until this registration pass.
    prepare_episode(ep_id, episode)
    target = store.get_job(job_id, ep_id=ep_id)
    if not target or target["status"] != "failed":
        raise RuntimeError("failed continuity-safe target changed during registration")
    if target.get("input_hash") != approval.get("job_input_hash"):
        raise RuntimeError("continuity anchor approval became stale during registration")
    panels = episode.get("panels") or []
    chain = [target]

    completed: list[str] = []
    failure: dict[str, Any] | None = None
    for original in chain:
        current = store.get_job(str(original["job_id"]), ep_id=ep_id) or original
        panel_index = int(current["panel_index"])
        if panel_index < 1 or panel_index > len(panels):
            failure = {"job_id": current["job_id"], "error": "panel index outside episode contract"}
            break
        panel = panels[panel_index - 1]
        job_metadata = dict(current.get("metadata") or {})
        action, final_state = _panel_visual_intent(panel)
        if not action or not final_state:
            failure = {
                "job_id": current["job_id"],
                "error": "continuity-safe requires explicit panel action and approved final_state",
            }
            break
        if (
            approval.get("approved_job_id") != current["job_id"]
            or approval.get("panel_action") != action
            or approval.get("approved_final_state") != final_state
        ):
            failure = {"job_id": current["job_id"], "error": "visual-state anchor approval is stale"}
            break
        job_metadata["continuity_safe_anchor_approval"] = approval
        job_metadata["render_mode"] = "continuity_safe"
        job_metadata["continuity_safe"] = {
            "status": "running",
            "source_anchor": str(anchor),
            "source_anchor_sha256": approval["sha256"],
            "motion": motion,
            "duration_seconds": SAFE_DURATION_SECONDS,
            "frame_count": SAFE_FRAME_COUNT,
            "visual_text_policy": "no_model_text; approved_subtitles_postproduction_only",
            "panel_action": action,
            "approved_final_state": final_state,
            "started_at": _utc_now(),
        }
        store.update_job(
            str(current["job_id"]), status="running", progress=0.05,
            prompt_id=None, error=None, metadata=job_metadata,
        )
        safe_dir = project / "continuity_safe" / str(current["panel_name"])
        safe_dir.mkdir(parents=True, exist_ok=True)
        output_value = (
            current.get("output_path")
            or (job_metadata.get("qa_retry_paths") or {}).get("output_path")
            or project / "videos" / f"{current['panel_name']}.mp4"
        )
        output = _inside_project(output_value, project, require_file=False)
        temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.partial.mp4"
        content_qa: dict[str, Any] | None = None
        try:
            cues, subtitle_warnings = panel_subtitle_cues(panel, strict=True)
            if any(float(cue["end_seconds"]) > SAFE_DURATION_SECONDS for cue in cues):
                raise ValueError("approved dialogue cue exceeds continuity-safe duration")
            subtitle_base = safe_dir / str(current["panel_name"])
            spec = _output_spec(anchor)
            srt_path = write_srt(cues, subtitle_base.with_suffix(".srt"))
            vtt_path = write_vtt(cues, subtitle_base.with_suffix(".vtt"))
            ass_path = write_ass(
                cues, subtitle_base.with_suffix(".ass"),
                play_res_x=spec["width"], play_res_y=spec["height"],
                margin_v=spec["margin_v"],
            )
            cue_audio: list[dict[str, Any]] = []
            voice_audits: list[dict[str, Any]] = []
            selected_voice = preferred_voice
            for cue_index, cue in enumerate(cues, 1):
                wave = safe_dir / f"cue_{cue_index:03d}.wav"
                audit = tts_func(
                    str(cue["text"]), wave,
                    preferred_voice=selected_voice, runner=tts_runner,
                )
                selected_voice = str(audit.get("selected_voice") or selected_voice)
                source_duration = audio_probe_func(wave, ffprobe=ffprobe, runner=tts_runner)
                cue_audio.append({**cue, "path": str(wave), "source_duration": source_duration})
                voice_audits.append(audit)
            command = _build_ffmpeg_command(
                anchor=anchor,
                temporary_output=temporary,
                cue_audio=cue_audio,
                width=spec["width"], height=spec["height"],
                motion=motion,
                ffmpeg=ffmpeg or ffmpeg_executable(),
            )
            runner(
                command, check=True, capture_output=True, text=True,
                timeout=max(1.0, float(timeout)),
            )
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError("continuity-safe FFmpeg did not produce a non-empty clip")
            probe = probe_func(temporary, ffprobe=ffprobe)
            validate_probe(
                probe,
                expected_width=spec["width"], expected_height=spec["height"],
                expected_fps=SAFE_FPS, require_audio=True,
                expected_video_codec="h264", expected_pixel_format="yuv420p",
                expected_audio_codec="aac", expected_audio_rate=48000,
                expected_audio_channels=2,
            )
            if abs(float(probe.get("duration_seconds") or 0) - SAFE_DURATION_SECONDS) > 0.15:
                raise ValueError(
                    f"continuity-safe duration mismatch: {probe.get('duration_seconds')} != {SAFE_DURATION_SECONDS}"
                )
            analysis = quality_analyzer(
                temporary, ffmpeg=ffmpeg or ffmpeg_executable(), runner=quality_runner,
            )
            prior = []
            for other in store.list_jobs(ep_id):
                if str(other.get("job_id")) == str(current["job_id"]):
                    continue
                other_qa = ((other.get("metadata") or {}).get("content_qa") or {})
                other_analysis = other_qa.get("analysis") or {}
                if other.get("status") == "succeeded" and other.get("output_path"):
                    if not other_analysis:
                        other_path = Path(str(other["output_path"]))
                        if not other_path.is_file():
                            raise RuntimeError(
                                f"prior succeeded clip is missing for content QA: {other['job_id']}"
                            )
                        other_analysis = quality_analyzer(
                            other_path, ffmpeg=ffmpeg or ffmpeg_executable(), runner=quality_runner,
                        )
                    prior.append((str(other["job_id"]), other_analysis))
            content_qa = quality_evaluator(analysis, prior, require_motion=True)
            content_qa.update({"checked_at": _utc_now(), "stage": "pre_success"})
            if not content_qa.get("passed"):
                raise RuntimeError(
                    "continuity-safe content QA failed: "
                    + ",".join(str(item) for item in content_qa.get("reasons") or ["unknown"])
                )
            shot_plan = ((current.get("metadata") or {}).get("inputs") or {}).get("shot_plan") or {}
            requested_edit_duration = shot_plan.get("edit_duration_seconds")
            edit_selection = None
            if requested_edit_duration is not None:
                source_artifact_sha = _sha256_file(temporary)
                protected_ranges = [
                    (float(cue["start_seconds"]), float(cue["end_seconds"])) for cue in cues
                ]
                try:
                    edit_selection = edit_selector(
                        analysis, source_duration_seconds=float(probe.get("duration_seconds") or 0),
                        requested_duration_seconds=float(requested_edit_duration),
                        source_artifact_sha256=source_artifact_sha,
                        edit_hint=shot_plan.get("edit_hint") or {},
                        protected_ranges=protected_ranges,
                    )
                    selection_check = validate_edit_selection(
                        edit_selection, source_artifact_sha256=source_artifact_sha,
                        requested_duration_seconds=float(requested_edit_duration),
                        source_duration_seconds=float(probe.get("duration_seconds") or 0),
                    )
                    if not selection_check["valid"]:
                        raise RuntimeError(",".join(selection_check["errors"]))
                except Exception as exc:
                    edit_selection = {
                        "status": "deadletter", "reason": str(exc),
                        "source_artifact_sha256": source_artifact_sha,
                        "requested_duration_seconds": float(requested_edit_duration),
                    }
                    raise RuntimeError(f"continuity-safe edit selection deadletter: {exc}") from exc
            _replace_with_retry(temporary, output)
            probe["path"] = str(output)
            manifest = {
                "schema_version": 1,
                "ep_id": ep_id,
                "job_id": current["job_id"],
                "render_mode": "continuity_safe",
                "source_anchor": str(anchor),
                "source_anchor_sha256": approval["sha256"],
                "anchor_approval": approval,
                "panel_action": action,
                "approved_final_state": final_state,
                "motion": motion,
                "duration_seconds": SAFE_DURATION_SECONDS,
                "frame_count": SAFE_FRAME_COUNT,
                "fps": SAFE_FPS,
                "tts_engine": "windows_sapi" if cues else "none_no_dialogue",
                "tts_voice": selected_voice if cues else None,
                "voice_audits": voice_audits,
                "voice_fidelity": "single_system_voice_not_character_voice_cloning",
                "ambient_bed": "deterministic_ffmpeg_pink_noise",
                "subtitles": {
                    "source": "approved_spoken_dialogue",
                    "burned_in": False,
                    "burn_policy": "delivery_only",
                    "legacy_burn_request_ignored": bool(burn_subtitles),
                    "srt_path": str(srt_path), "vtt_path": str(vtt_path),
                    "ass_path": str(ass_path), "warnings": subtitle_warnings,
                },
                "probe": probe,
                "content_qa": content_qa,
                "edit_selection": edit_selection,
                "ffmpeg_command": command,
                "completed_at": _utc_now(),
            }
            manifest_path = safe_dir / "manifest.json"
            write_json_atomic(manifest_path, manifest)
            success_metadata = dict((store.get_job(str(current["job_id"])) or current).get("metadata") or {})
            success_metadata.update({
                "render_mode": "continuity_safe",
                "artifact_sha256": _sha256_file(output),
                "content_qa": content_qa,
                "editorial_review": {"status": "pending", "reason": "awaiting human review"},
                "release": {"status": "pending", "reason": "episode release not approved"},
                "continuity_safe": {
                    **job_metadata["continuity_safe"],
                    "status": "succeeded",
                    "source_anchor": str(anchor),
                    "source_anchor_sha256": approval["sha256"],
                    "tts_engine": manifest["tts_engine"],
                    "tts_voice": manifest["tts_voice"],
                    "tts_voice_fidelity": manifest["voice_fidelity"],
                    "voice_fallback_used": any(bool(item.get("fallback_used")) for item in voice_audits),
                    "subtitle_paths": manifest["subtitles"],
                    "manifest_path": str(manifest_path),
                    "completed_at": manifest["completed_at"],
                },
            })
            if edit_selection:
                success_metadata["edit_selection"] = edit_selection
            store.update_job(
                str(current["job_id"]), status="succeeded", progress=1.0,
                output_path=str(output), preview_path=str(output),
                prompt_id=None, comfy_output_path=None,
                graph_path=str(manifest_path), timing_path=str(ass_path),
                probe=probe, error=None, metadata=success_metadata,
                reference_images=[{
                    "role": "approved_group_composition_anchor",
                    "source_path": str(anchor), "sha256": approval["sha256"],
                }],
            )
            completed.append(str(current["job_id"]))
            try:
                # Refresh the next strict dependency from this newly validated
                # artifact hash. This prevents a later Web save from treating
                # the safe clip as stale merely because reject_job cleared a
                # descendant input hash.
                prepare_episode(ep_id, episode)
            except Exception as refresh_error:
                preserved = store.get_job(str(current["job_id"])) or current
                preserved_metadata = dict(preserved.get("metadata") or {})
                pipeline_warnings = list(preserved_metadata.get("pipeline_warnings") or [])
                pipeline_warnings.append({
                    "stage": "continuity_safe_post_render_registration",
                    "error": str(refresh_error), "artifact_preserved": True,
                    "recorded_at": _utc_now(),
                })
                preserved_metadata["pipeline_warnings"] = pipeline_warnings[-50:]
                store.update_job(
                    str(current["job_id"]), status="succeeded", progress=1.0,
                    error=None, metadata=preserved_metadata,
                )
                failure = {
                    "job_id": str(current["job_id"]),
                    "stage": "post_render_registration",
                    "error": str(refresh_error),
                    "artifact_preserved": True,
                }
                break
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            failed_current = store.get_job(str(current["job_id"])) or current
            failed_metadata = dict(failed_current.get("metadata") or {})
            safe_audit = dict(failed_metadata.get("continuity_safe") or {})
            safe_audit.update({
                "status": "failed", "error": str(exc), "failed_at": _utc_now(),
                "source_anchor": str(anchor), "source_anchor_sha256": approval["sha256"],
            })
            failed_metadata.update({"render_mode": "continuity_safe", "continuity_safe": safe_audit})
            if content_qa is not None:
                failed_metadata["content_qa"] = content_qa
                failed_metadata["editorial_review"] = {
                    "status": "blocked", "reason": "content QA failed",
                }
                failed_metadata["release"] = {
                    "status": "revoked", "reason": "content QA failed",
                }
            if "edit_selection" in locals() and edit_selection:
                failed_metadata["edit_selection"] = edit_selection
            store.update_job(
                str(current["job_id"]), status="failed", progress=0.0,
                prompt_id=None, error=str(exc), metadata=failed_metadata,
                probe={}, completed_at=None,
            )
            failure = {"job_id": str(current["job_id"]), "error": str(exc)}
            break

    return {
        "ep_id": ep_id,
        "render_mode": "continuity_safe",
        "source_job_id": job_id,
        "chain_job_ids": [str(job["job_id"]) for job in chain],
        "completed_job_ids": completed,
        "failure": failure,
        "stopped": failure is not None,
        "snapshot": project_snapshot(ep_id),
    }


__all__ = [
    "SAFE_FPS", "SAFE_FRAME_COUNT", "SAFE_DURATION_SECONDS", "DEFAULT_SAPI_VOICE",
    "approve_continuity_anchor", "synthesize_windows_sapi", "probe_audio_duration",
    "run_continuity_safe_chain",
]
