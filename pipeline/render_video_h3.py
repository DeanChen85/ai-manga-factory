"""Batch MiniMax H3 ref2va renderer — renders storyboard panels into animated video clips with native speech.

Uses the ref2va (reference-to-video) model which:
- Takes first frame + last frame as reference images (ref_image_0 / ref_image_1)
- Optionally takes up to 7 additional character reference portraits (ref_image_2..8)
- Generates video transitioning between the two frames
- Generates NATIVE stereo audio including voice/speech, sound effects, ambient sound, background music
- Uses the prompt to determine what dialogue/voice is spoken

This eliminates the need for separate TTS — H3 generates speech directly from the prompt.

Workflow is rebuilt from the official ComfyUI H3 template (video_minimax_h3_r2v.json).
"""
from __future__ import annotations

import json
import hashlib
import math
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from action_catalog import compile_panel_action
from h3_director import (
    H3_DIRECTOR_SKILL_VERSION,
    H3_OFFICIAL_PROMPT_SHAPE,
    H3_PROMPT_MAX_ENGLISH_WORDS,
    compile_h3_director_prompt,
)
from h3_profiles import H3_RENDER_PROFILE_CONTRACT
from runtime_config import comfyui_root, comfyui_server, project_root, ffmpeg_executable
from task_store import RenderJobStore, default_store

# ── Persistent generation ledger (best-effort import; never blocks rendering) ──
try:
    import generation_log
    _GEN_LOG_AVAILABLE = True
except ImportError:
    _GEN_LOG_AVAILABLE = False

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = project_root()
COMFY = comfyui_root()
SERVER = comfyui_server()

# ── H3 ref2va Model files ─────────────────────────────────────────────────────
H3_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

# ── Larryvrh MiniMax-H3 Turbo LoRA ───────────────────────────────────────────
# Upstream recommends v4 step-600 EMA at 6-8 steps.  Existing installations may
# still carry the older v1/ckpt500 names, so graph construction resolves the best
# installed compatible file rather than emitting a filename ComfyUI cannot load.
H3_LORA_RECOMMENDED = "minimax_h3_turbo_v4_step600_ema.safetensors"
H3_LORA_CANDIDATES = (
    H3_LORA_RECOMMENDED,
    "minimax_h3_turbo_v4_step600.safetensors",
    "minimax_h3_turbo_ema_ckpt500.safetensors",
    "minimax_h3_turbo_4step_ema_ckpt850_pruned_comfyui.safetensors",
    "minimax_h3_turbo_4step_ckpt850_pruned_comfyui.safetensors",
)
# Compatibility alias retained for callers that imported the old constant.
H3_LORA = H3_LORA_RECOMMENDED
H3_LORA_STRENGTH = 1.0
H3_INFERENCE_STEPS = 8       # Upstream useful range 4-8; default to best-quality 8.
H3_LORA_ENABLED_DEFAULT = True  # off only if explicitly disabled
H3_TURBO_SCHEDULER = "simple"
H3_VIDEO_FLOW_SHIFT = 12.0
H3_AUDIO_FLOW_SHIFT = 3.0
H3_AUDIO_SAMPLE_RATE = 32000
H3_AUDIO_CHANNELS = 2
H3_PROMPT_BODY_MIN_ENGLISH_WORDS = 80
H3_PROMPT_BODY_MAX_ENGLISH_WORDS = H3_PROMPT_MAX_ENGLISH_WORDS
H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS = H3_PROMPT_MAX_ENGLISH_WORDS
H3_RUNTIME_PROMPT_CONTRACT = "h3-runtime/v9-complete-fragments"

# ── Default video parameters ─────────────────────────────────────────────────
# 16:9 landscape, 0.6 megapixels, multiples of 32 — H3 VAE constraint
# Default per-clip duration = 10s → 17*14+5 = 243 frames @ 24fps (≈10.13s)
VIDEO_FPS = 24
DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_LENGTH_FRAMES = 243  # 17*14+5, the closest valid 10s frame count
DEFAULT_MEGAPIXELS = 0.6  # 提高分辨率，避免模糊
MAX_CHAR_REFS = 6            # ref_image_3..8; first/last/anchor occupy slots 0/1/2
SAGE_ATTENTION_MODE = "auto"
EXTRA_CHAR_REF_NODE_IDS = ("170", "171", "172", "173", "174", "175")
ASPECT_RATIO_CHOICES = {
    "16:9": "16:9 (Widescreen)",
    "9:16": "9:16 (Portrait Widescreen)",
    "1:1": "1:1 (Square)",
}
SAGE_ATTENTION_CHOICES = {
    "off": "disabled", "disabled": "disabled", "auto": "auto",
    "sage2": "sageattn_qk_int8_pv_fp16_cuda", "sage3": "sageattn3",
}


# ── 6 background music + 6 ambience presets (AI selects per panel) ─────────────
MUSIC_PRESETS = {
    "soft_piano":     "sparse soft piano notes at a slow tempo, gradually decreasing in volume",
    "string_orch":    "sustained strings at a moderate tempo, rising gently before a short final fade",
    "urban_electronic":"a restrained electronic beat with a soft synthesizer pulse at a moderate tempo",
    "chinese_folk":   "measured guzheng plucks and bamboo-flute phrases at a slow tempo with a clean fade",
    "suspense_dark":  "low cello pulses and sustained bass drones at a slow tempo, stopping on the final action",
    "epic_brass":     "short brass phrases and controlled percussion at a moderate tempo, ending with one final accent",
}

AMBIENCE_PRESETS = {
    "rain_city":      "ambient city rain, distant traffic, wet pavement",
    "rain_night_city":"nighttime city rain, wet asphalt reflections, distant traffic, soft thunder and water dripping from awnings",
    "rain_outside_glass":"muffled exterior water taps the awning and closed glass doors over a quiet refrigerator hum; dry interior room tone remains continuous",
    "office_quiet":   "quiet indoor office, soft HVAC hum, paper rustle",
    "street_bustle":  "busy street ambience, footsteps, car horns, distant chatter",
    "home_intimate":  "intimate indoor room, faint clock tick, kettle whistle",
    "night_empty":    "empty night street, far away dog bark, single passing car",
    "nature_wind":    "open-air wind, leaves rustling, distant birds",
    "forest_morning": "calm morning forest, gentle leaf rustle, close birdsong, distant stream and soft breeze",
    "subway_crowd":   "busy underground station, layered crowd murmur, footsteps, train brakes and platform announcements in the distance",
    "storm_thunder":  "heavy storm ambience, rolling thunder, driving rain, strong wind gusts and occasional close cracks",
    "silence":        "near silence with only subtle room tone and no prominent environmental sound",
}

# ── STYLE_PRESETS — maps Chinese style tags to rich H3 aesthetic descriptions ──────────────
# Used by `_build_ref2va_prompt` to inject a fully-formed cinematic preamble
# even when the panel only has a short style tag like "吉卜力" or "国风".
STYLE_PRESETS = {
    # 卡通/吉卜力 ─────────────────────────────────────────────────────────
    "ghibli": (
        "Studio Ghibli hand-drawn anime aesthetic, soft watercolor textures, "
        "warm natural lighting, gentle wind-blown atmosphere, lush background "
        "detail in the style of Hayao Miyazaki, cel-shaded characters with "
        "expressive eyes, painterly sky and clouds, storybook warmth"
    ),
    "吉卜力": (
        "Studio Ghibli hand-drawn anime aesthetic, soft watercolor textures, "
        "warm natural lighting, gentle wind-blown atmosphere, lush background "
        "detail in the style of Hayao Miyazaki, cel-shaded characters with "
        "expressive eyes, painterly sky and clouds, storybook warmth"
    ),
    "cartoon": (
        "high-quality 2D animation, expressive character acting, saturated "
        "colors, clean linework, storybook illustration style, soft cel shading"
    ),
    "卡通": (
        "high-quality 2D animation, expressive character acting, saturated "
        "colors, clean linework, storybook illustration style, soft cel shading"
    ),
    "anime": (
        "modern anime aesthetic, sharp character design, vivid colors, "
        "cinematic anime keyframes, dynamic camera, expressive eyes"
    ),
    "动漫": (
        "modern anime aesthetic, sharp character design, vivid colors, "
        "cinematic anime keyframes, dynamic camera, expressive eyes"
    ),

    # 拟人/国风/现代写实 ──────────────────────────────────────────────────
    "realistic": (
        "cinematic photorealistic, modern Chinese urban aesthetic, "
        "natural skin texture, realistic eye reflections, volumetric lighting, "
        "shallow depth of field, anamorphic lens flare, 35mm film grain, "
        "IMAX-quality color grading"
    ),
    "拟人": (
        "cinematic photorealistic, modern Chinese urban aesthetic, "
        "natural skin texture, realistic eye reflections, volumetric lighting, "
        "shallow depth of field, anamorphic lens flare, 35mm film grain, "
        "IMAX-quality color grading"
    ),
    "国风": (
        "modern Chinese guofeng aesthetic, photorealistic East Asian actors, "
        "traditional Chinese costume textures (silk, brocade, jade ornaments), "
        "warm wood and paper-lantern lighting, restrained elegant color palette "
        "of ivory, cinnabar, ink-black and jade-green, cinematic 35mm shallow "
        "depth of field, subtle film grain"
    ),
    "都市": (
        "modern Chinese urban drama, photorealistic, glass-and-steel office "
        "towers, neon reflections on wet asphalt, cool blue-warm amber lighting "
        "contrast, cinematic shallow depth of field, 35mm anamorphic, "
        "naturalistic East Asian actors"
    ),
    "都市反转短剧": (
        "modern Chinese urban drama, photorealistic, glass-and-steel office "
        "towers, neon reflections on wet asphalt, cool blue-warm amber lighting "
        "contrast, cinematic shallow depth of field, 35mm anamorphic, "
        "naturalistic East Asian actors, suspenseful grading"
    ),

    # 古风/仙侠 ──────────────────────────────────────────────────────────
    "古风": (
        "Chinese xianxia ink-wash painting aesthetic, flowing silk hanfu robes, "
        "bamboo and pine forests, soft moonlight and lantern glow, "
        "traditional architecture, restrained color palette of jade, ivory "
        "and ink, painterly mist and clouds, cinematic 2.39:1 framing"
    ),
    "仙侠": (
        "Chinese xianxia ink-wash painting aesthetic, flowing silk hanfu robes, "
        "bamboo and pine forests, soft moonlight and lantern glow, sword glow "
        "and floating talismans, restrained color palette of jade, ivory and "
        "ink, painterly mist and clouds, cinematic 2.39:1 framing"
    ),
    "水墨": (
        "traditional Chinese ink-wash shuimo painting, sumi-e brush strokes, "
        "rice paper texture, negative space (留白), calligraphic composition, "
        "restrained black-grey-cinnabar palette, poetic stillness"
    ),

    # 科幻 ──────────────────────────────────────────────────────────────
    "科幻": (
        "cyberpunk sci-fi aesthetic, photorealistic, holographic interfaces, "
        "neon teal-and-magenta lighting, rain-slick reflective surfaces, "
        "atmospheric fog, volumetric god-rays, futuristic East Asian cityscape"
    ),
    "未来": (
        "near-future sci-fi aesthetic, photorealistic, clean white-and-silver "
        "futuristic interiors, soft ambient LED panels, holographic displays, "
        "minimalist East Asian architecture, volumetric lighting"
    ),

    # 悬疑/惊悚 ──────────────────────────────────────────────────────────
    "悬疑": (
        "cinematic noir thriller aesthetic, photorealistic, deep shadows, "
        "single hard key-light source, desaturated teal-and-amber grading, "
        "35mm anamorphic, naturalistic East Asian actors, suspenseful framing"
    ),
    "惊悚": (
        "cinematic horror aesthetic, photorealistic, harsh under-lighting, "
        "desaturated cold color palette, deep crushing blacks, fog, "
        "naturalistic East Asian actors, unsettling framing"
    ),
}

# Default fallback for unrecognized style tags
STYLE_DEFAULT = (
    "cinematic photorealistic, modern Chinese urban aesthetic, "
    "naturalistic East Asian actors, 35mm shallow depth of field, "
    "soft volumetric lighting, consistent film grain"
)


def resolve_h3_turbo_lora(
    requested: str | None = None,
    *,
    comfy_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve a Turbo LoRA to an actually installed file, preferring v4-600.

    No download or mutation occurs here.  Returning the selection reason makes
    legacy fallback visible in graph snapshots instead of silently pretending
    the recommended checkpoint is present.
    """
    root = Path(comfy_root or COMFY)
    lora_root = root / "models" / "loras"
    installed: dict[str, str] = {}
    if lora_root.is_dir():
        for path in lora_root.rglob("*.safetensors"):
            relative = path.relative_to(lora_root).as_posix()
            installed.setdefault(path.name.casefold(), relative)

    if requested:
        selected = installed.get(Path(requested).name.casefold())
        if not selected:
            raise FileNotFoundError(
                f"requested MiniMax-H3 Turbo LoRA is not installed under {lora_root}: {requested}"
            )
        reason = "explicit"
    else:
        selected = next(
            (installed.get(candidate.casefold()) for candidate in H3_LORA_CANDIDATES
             if installed.get(candidate.casefold())),
            None,
        )
        if not selected:
            raise FileNotFoundError(
                "no compatible MiniMax-H3 Turbo LoRA is installed; expected one of: "
                + ", ".join(H3_LORA_CANDIDATES)
                + f" under {lora_root}"
            )
        reason = "recommended" if Path(selected).name == H3_LORA_RECOMMENDED else "legacy_fallback"
    return {
        "lora_name": selected,
        "selection": reason,
        "recommended_name": H3_LORA_RECOMMENDED,
        "recommended_available": H3_LORA_RECOMMENDED.casefold() in installed,
        "lora_root": str(lora_root.resolve()),
    }


def build_h3_sampling_contract(
    *,
    use_lora: bool = H3_LORA_ENABLED_DEFAULT,
    lora_name: str | None = None,
    lora_strength: float = H3_LORA_STRENGTH,
    turbo_steps: int = H3_INFERENCE_STEPS,
    comfy_root: Path | None = None,
) -> dict[str, Any]:
    """Return the auditable AV sampling contract used by the actual graph."""
    steps = int(turbo_steps if use_lora else 20)
    if use_lora and not 4 <= steps <= 8:
        raise ValueError("MiniMax-H3 Turbo steps must stay in the upstream-supported 4-8 range")
    lora = resolve_h3_turbo_lora(lora_name, comfy_root=comfy_root) if use_lora else {
        "lora_name": None,
        "selection": "disabled",
        "recommended_name": H3_LORA_RECOMMENDED,
        "recommended_available": None,
        "lora_root": str((Path(comfy_root or COMFY) / "models" / "loras").resolve()),
    }
    return {
        "mode": "turbo" if use_lora else "base",
        "steps": steps,
        "scheduler": H3_TURBO_SCHEDULER,
        "sampler_node": "MiniMaxH3TurboSampler",
        "lora_name": lora["lora_name"],
        "lora_strength": float(lora_strength) if use_lora else 0.0,
        "lora_selection": lora["selection"],
        "recommended_lora": lora["recommended_name"],
        "recommended_lora_available": lora["recommended_available"],
        "low_vram": False,
        "video_clock": {"fps": VIDEO_FPS, "flow_shift": H3_VIDEO_FLOW_SHIFT},
        "audio_clock": {
            "sample_rate_hz": H3_AUDIO_SAMPLE_RATE,
            "channels": H3_AUDIO_CHANNELS,
            "flow_shift": H3_AUDIO_FLOW_SHIFT,
        },
        "clock_policy": (
            "MiniMaxH3TurboSampler auto-detects native ModelSamplingAV; otherwise it steps "
            "video shift 12 and audio shift 3 on separate clocks"
        ),
    }


def align_length(n: int) -> int:
    """Snap frame count to H3's 17k+5 grid (5, 22, 39, ..., 362, 379, ...).
    Matches the ComfyMathExpression in the official template."""
    n = max(5, int(n))
    return n + (5 - (n % 17)) % 17


def duration_to_length(seconds: float) -> int:
    """Convert desired duration (seconds) to a valid H3 frame count (17k+5 grid)."""
    raw = max(5, round(seconds * VIDEO_FPS))
    return align_length(raw)


_TIME_RANGE_RE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-~至]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*$",
    re.IGNORECASE,
)


def _cue_range(cue: dict[str, Any], duration: float) -> tuple[float, float]:
    if any(key in cue for key in ("start_seconds", "end_seconds", "start_s", "end_s")):
        start = float(cue.get("start_seconds", cue.get("start_s", 0.0)))
        end = float(cue.get("end_seconds", cue.get("end_s", duration)))
    else:
        match = _TIME_RANGE_RE.match(str(cue.get("time_range") or ""))
        if not match:
            raise ValueError(f"invalid cue time_range: {cue.get('time_range')!r}")
        start, end = float(match.group("start")), float(match.group("end"))
    if end <= start:
        raise ValueError(f"cue end must be after start: {start}-{end}")
    if start < 0 or start >= duration:
        raise ValueError(f"cue start outside clip: {start}s (duration {duration}s)")
    return start, min(end, duration)


def _normalize_cues(
    cues: list[dict[str, Any]], duration: float, kind: str, warnings: list[str]
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, original in enumerate(cues):
        cue = dict(original)
        try:
            start, end = _cue_range(cue, duration)
        except (TypeError, ValueError) as exc:
            warnings.append(f"{kind}[{index}] ignored: {exc}")
            continue
        start_frame = max(0, min(round(start * VIDEO_FPS), round(duration * VIDEO_FPS)))
        end_frame = max(start_frame + 1, min(round(end * VIDEO_FPS), round(duration * VIDEO_FPS)))
        cue.update({
            "kind": kind,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_seconds": start_frame / VIDEO_FPS,
            "end_seconds": end_frame / VIDEO_FPS,
        })
        cue.pop("time_range", None)
        normalized.append(cue)
    normalized.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
    return normalized


def build_timing_contract(panel: dict[str, Any], duration_seconds: float = DEFAULT_DURATION_SECONDS) -> dict[str, Any]:
    """Build an auditable 24fps cue sheet separated by semantic purpose.

    H3 native audio remains generative, so this contract improves prompt timing
    and validation but does not claim sample-accurate lip sync.  The same cues
    are suitable for a later deterministic TTS/subtitle mixing stage.
    """
    frame_count = duration_to_length(duration_seconds)
    actual_duration = frame_count / VIDEO_FPS
    warnings: list[str] = []

    spoken_source = panel.get("spoken_dialogue")
    if spoken_source is None:
        legacy = panel.get("dialogue")
        spoken_source = legacy if isinstance(legacy, list) else []
    spoken = _normalize_cues(
        [dict(item) for item in spoken_source or [] if isinstance(item, dict)],
        actual_duration, "spoken_dialogue", warnings,
    )

    text_source = panel.get("on_screen_text")
    if text_source is None:
        text_source = panel.get("dialogue_bubbles", [])
    on_screen = _normalize_cues(
        [dict(item) for item in text_source or [] if isinstance(item, dict)],
        actual_duration, "on_screen_text", warnings,
    )

    audio_source = [dict(item) for item in panel.get("audio_cues", []) if isinstance(item, dict)]
    for sfx in panel.get("sfx", []):
        if isinstance(sfx, dict):
            audio_source.append({**sfx, "cue_type": "sfx", "text": sfx.get("tag", "")})
    typed_audio_source: list[dict[str, Any]] = []
    for index, cue in enumerate(audio_source):
        cue_type = str(cue.get("cue_type") or cue.get("type") or "sfx").strip().lower()
        if cue_type not in {"sfx", "ambience", "music"}:
            warnings.append(
                f"audio_cue[{index}] ignored: cue_type must be sfx, ambience, or music"
            )
            continue
        normalized_cue = {**cue, "cue_type": cue_type}
        typed_audio_source.append(normalized_cue)
        duck_db = normalized_cue.get("duck_dialogue_db")
        if duck_db not in (None, "", 0, 0.0, "0", "0.0"):
            warnings.append(
                f"audio_cue[{index}].duck_dialogue_db is advisory for native H3 audio; "
                "deterministic ducking requires postproduction mixing"
            )
    audio = _normalize_cues(typed_audio_source, actual_duration, "audio_cue", warnings)

    previous_end = -1
    for index, cue in enumerate(spoken):
        if cue["start_frame"] < previous_end:
            warnings.append(f"spoken_dialogue[{index}] overlaps the previous spoken cue")
        previous_end = max(previous_end, cue["end_frame"])
        text = str(cue.get("text") or cue.get("line") or "").strip()
        seconds = max(1 / VIDEO_FPS, cue["end_seconds"] - cue["start_seconds"])
        latin_words = len(re.findall(r"[A-Za-z0-9]+", text))
        cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
        rate = (cjk_chars if cjk_chars else latin_words) / seconds
        limit = 6.0 if cjk_chars else 3.5
        cue["speech_units_per_second"] = round(rate, 2)
        if rate > limit:
            warnings.append(
                f"spoken_dialogue[{index}] may be too fast ({rate:.2f} units/s > {limit:.1f})"
            )
        if not cue.get("speaker_id"):
            warnings.append(f"spoken_dialogue[{index}] has no stable speaker_id")

    return {
        "schema_version": 1,
        "fps": VIDEO_FPS,
        "requested_duration_seconds": float(duration_seconds),
        "frame_count": frame_count,
        "actual_duration_seconds": actual_duration,
        "spoken_dialogue": spoken,
        "on_screen_text": on_screen,
        "audio_cues": audio,
        "warnings": warnings,
        "native_audio_mode": True,
        "native_audio_schedule": {
            "sample_rate_hz": H3_AUDIO_SAMPLE_RATE,
            "channels": H3_AUDIO_CHANNELS,
            "video_flow_shift": H3_VIDEO_FLOW_SHIFT,
            "audio_flow_shift": H3_AUDIO_FLOW_SHIFT,
            "synchronization": "joint AV latent sampled by MiniMaxH3TurboSampler",
        },
        "deterministic_external_audio_ready": True,
    }


def _api(path: str, payload=None):
    """Make a request to the ComfyUI API."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            body = r.read().decode("utf-8", errors="replace").strip()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API error {e.code}: {body[:500]}")


def release_comfy_resources(
    *, api_func: Callable[[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Unload H3 weights between shots, but never while any queue item exists."""
    api_func = api_func or _api
    try:
        queue = api_func("/queue", None)
        if queue.get("queue_running") or queue.get("queue_pending"):
            return {
                "released": False,
                "reason": "ComfyUI queue is not empty; resource release skipped",
            }
        api_func("/free", {"unload_models": True, "free_memory": True})
        return {"released": True, "reason": "models unloaded and cache released"}
    except Exception as exc:
        return {"released": False, "reason": f"ComfyUI resource release failed: {exc}"}


def build_h3_reference_bindings(
    *,
    first_frame_filename: str | None = None,
    last_frame_filename: str | None = None,
    character_anchor_filename: str | None = None,
    character_anchor_source_id: str | None = None,
    extra_reference_filenames: Optional[list[str]] = None,
    extra_reference_roles: Optional[list[str]] = None,
    extra_reference_source_ids: Optional[list[str | None]] = None,
) -> list[dict[str, Any]]:
    """Map internal autogrow sockets to H3's actual ``<Picture N>`` labels.

    MiniMaxH3ReferenceToVideo ignores gaps in ref_image_N and presents only the
    connected images to Qwen in insertion order.  Therefore internal socket 2 is
    not necessarily <Picture 3>; the ordinal must be computed from connected
    references, as done here.
    """
    extras = list(extra_reference_filenames or [])[:MAX_CHAR_REFS]
    roles = list(extra_reference_roles or [])
    source_ids = list(extra_reference_source_ids or [])
    specs: list[tuple[str, str, str, str, str | None]] = []
    if first_frame_filename:
        specs.append(("ref_images.ref_image_0", "137", "first_frame", first_frame_filename, None))
    if last_frame_filename:
        specs.append(("ref_images.ref_image_1", "139", "last_frame", last_frame_filename, None))
    if character_anchor_filename:
        specs.append((
            "ref_images.ref_image_2", "153", "character_anchor",
            character_anchor_filename, character_anchor_source_id,
        ))
    for index, filename in enumerate(extras):
        role = roles[index] if index < len(roles) else "character_reference"
        specs.append((
            f"ref_images.ref_image_{3 + index}",
            EXTRA_CHAR_REF_NODE_IDS[index],
            role,
            filename,
            source_ids[index] if index < len(source_ids) else None,
        ))
    bindings: list[dict[str, Any]] = []
    for picture_index, (slot, node_id, role, filename, source_id) in enumerate(specs, 1):
        binding = {
            "slot": slot,
            "node_id": node_id,
            "role": role,
            "filename": filename,
            "model_label": f"<Picture {picture_index}>",
        }
        if source_id:
            binding["source_id"] = str(source_id)
        bindings.append(binding)
    return bindings


def _prompt_with_reference_map(
    prompt: str,
    bindings: list[dict[str, Any]],
    *,
    reference_policy: str = "standard",
    composition_anchor_cast_count: int | None = None,
) -> str:
    if not bindings:
        _assert_h3_prompt_budget(prompt, H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS, "total prompt")
        return prompt
    if prompt.lstrip().startswith("subject_definitions:"):
        missing_labels = [
            str(item.get("model_label") or "") for item in bindings
            if str(item.get("model_label") or "") not in prompt
        ]
        if missing_labels:
            raise ValueError(
                "official Ref2VA prompt is missing reference labels: "
                + ", ".join(missing_labels)
            )
        _assert_h3_prompt_budget(prompt, H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS, "official Ref2VA prompt")
        return prompt
    composition_anchor_mode = reference_policy == "composition_anchor_first"
    role_text = {
        "first_frame": (
            "opening group composition, identities, and environment authority"
            if composition_anchor_mode else "opening frame and scene authority"
        ),
        "last_frame": (
            "final group composition, identities, and environment authority"
            if composition_anchor_mode else "final frame authority"
        ),
        "character_anchor": "primary character identity authority",
        "character_reference": "character identity authority",
        "scene_reference": "scene layout and lighting authority",
    }
    # Every following line already carries the exact model ordinal and role.
    # Keep the heading structural so scarce H3 prompt words are spent on the
    # opening/final/cast/scene constraints rather than repeating authority.
    lines = ["[REFERENCE MAP]"]
    for item in bindings:
        lines.append(
            f"{item['model_label']} = {role_text.get(item['role'], item['role'].replace('_', ' '))}."
        )
    character_labels = [
        item["model_label"] for item in bindings
        if item["role"] in {"character_anchor", "character_reference"}
    ]
    if composition_anchor_mode:
        lines.append(
            "[GROUP ANCHOR POLICY] Group anchors exclusively control composition, identities, wardrobe, "
            "devices, and environment; never cut to individual portraits or alternate scenes."
        )
        if composition_anchor_cast_count:
            lines.append(
                f"[CAST] Exactly {int(composition_anchor_cast_count)} distinct people throughout every frame; "
                "never omit, merge, replace, or duplicate."
            )
    elif character_labels:
        if any(item["role"] == "scene_reference" for item in bindings):
            lines.append(
                f"[CAST] Exactly {len(character_labels)} distinct people throughout every frame, "
                "one per character-reference; never omit, merge, replace, or duplicate."
            )
        else:
            lines.append(
                f"[CAST] Show exactly {len(character_labels)} distinct people, one per character-reference; "
                "never omit, merge, replace, or duplicate."
            )
    if any(item["role"] == "first_frame" for item in bindings):
        label = next(item["model_label"] for item in bindings if item["role"] == "first_frame")
        lines.append(f"Begin with the composition and environment established by {label}.")
    if any(item["role"] == "last_frame" for item in bindings):
        label = next(item["model_label"] for item in bindings if item["role"] == "last_frame")
        lines.append(f"End with the composition established by {label}.")
    lines.append("Never render picture labels, filenames, subtitles, or reference annotations.")
    result = prompt.rstrip() + "\n\n" + "\n".join(lines)
    _assert_h3_prompt_budget(result, H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS, "prompt plus reference map")
    return result


def build_h3_ref2va_graph(
    prompt: str,
    seed: int,
    char_ref_filenames: Optional[list[str]] = None,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    first_frame_filename: Optional[str] = None,
    last_frame_filename: Optional[str] = None,
    character_anchor_filename: Optional[str] = None,
    use_lora: bool = H3_LORA_ENABLED_DEFAULT,
    lora_strength: float = H3_LORA_STRENGTH,
    ep_id: Optional[str] = None,
    name_prefix: Optional[str] = None,
    aspect_ratio: str = "16:9",
    reference_fidelity: str = "fast",
    sage_attention: str = SAGE_ATTENTION_MODE,
    lora_name: str | None = None,
    turbo_steps: int = H3_INFERENCE_STEPS,
    extra_reference_roles: Optional[list[str]] = None,
    reference_policy: str = "standard",
    composition_anchor_cast_count: int | None = None,
    megapixels: float = DEFAULT_MEGAPIXELS,
) -> dict:
    """Build the complete MiniMax H3 ref2va workflow graph (official node IDs).

    Node layout (matches the installed MiniMax-H3 Turbo extension):
      115: ResolutionSelector (9:16, 0.9MP, multiple=32)
      119: VAELoader (video VAE)
      120: VAELoader (audio VAE)
      121: VAEDecodeAudio
      122: VAEDecode (video frames)
      124: BasicScheduler (simple, 4-8 steps with Turbo LoRA, 20 without)
      125: SamplerCustomAdvanced
      126: BasicGuider
      127: UNETLoader (H3 ref2va INT8 pruned)
      128: CLIPLoader (Qwen3-VL-32B INT8, minimax)
      129: RandomNoise
      130: CreateVideo
      131: ComfyMathExpression — length = align(round(a*24))
      132: PrimitiveFloat — duration seconds (default 10)
      136: MiniMaxH3ReferenceToVideo
      137: LoadImage (first frame) → ref_images.ref_image_0
      138: PrimitiveStringMultiline (prompt)
      139: LoadImage (last frame) → ref_images.ref_image_1
      141: PathchSageAttentionKJ
      153: LoadImage (character face anchor) → ref_images.ref_image_2
      164: MiniMaxH3TurboLoRA (recommended v4-600 when installed, strength=1)
      165: MiniMaxH3TurboSampler (native/legacy dual audio-video clock adapter)
      92 : SaveVideo
      140/142/144..: optional extra LoadImage nodes for additional character references (ref_image_3..8)

    Args:
        first_frame_filename: Filename of first frame image (already in ComfyUI/input/).
        last_frame_filename: Filename of last frame image.
        character_anchor_filename: Optional face/character anchor → ref_image_2 (preserves
            character identity across the clip; conventionally a clear headshot).
        char_ref_filenames: Up to 6 additional approved references (characters
            or scene), connected after first/last/anchor in model picture order.
        prompt: Full H3 prompt (style, scene, dialogue, music, ambience).
        seed: Random seed.
        duration_seconds: Target clip duration. Snapped to 17k+5 frame grid.
        use_lora: Whether to apply the Turbo LoRA. If False, steps=20 and the
            LoRA loader is omitted entirely.
        lora_strength: Strength passed to MiniMaxH3TurboLoRA (upstream: 1.0).

    Returns:
        ComfyUI prompt graph dict.
    """
    char_refs = list(char_ref_filenames or [])[:MAX_CHAR_REFS]
    if aspect_ratio not in ASPECT_RATIO_CHOICES:
        raise ValueError(f"unsupported H3 aspect ratio: {aspect_ratio}")
    if reference_fidelity not in {"fast", "identity"}:
        raise ValueError("reference_fidelity must be 'fast' or 'identity'")
    megapixels = float(megapixels)
    if not 0.1 <= megapixels <= 16.0:
        raise ValueError("H3 megapixels must stay within ResolutionSelector's 0.1-16.0 range")
    sage_value = SAGE_ATTENTION_CHOICES.get(sage_attention, sage_attention)
    sampling = build_h3_sampling_contract(
        use_lora=use_lora,
        lora_name=lora_name,
        lora_strength=lora_strength,
        turbo_steps=turbo_steps,
    )

    # ── Reference image inputs ──────────────────────────────────────────────
    bindings = build_h3_reference_bindings(
        first_frame_filename=first_frame_filename,
        last_frame_filename=last_frame_filename,
        character_anchor_filename=character_anchor_filename,
        extra_reference_filenames=char_refs,
        extra_reference_roles=extra_reference_roles,
    )
    ref_image_inputs = {item["slot"]: [item["node_id"], 0] for item in bindings}
    char_ref_nodes = {
        item["node_id"]: {
            "class_type": "LoadImage",
            "inputs": {"image": item["filename"], "upload": "image"},
        }
        for item in bindings
    }
    for item in bindings:
        print(f"  [graph] {item['slot']} -> {item['model_label']} ({item['role']}) = {item['filename']}")
    prompt = _prompt_with_reference_map(
        prompt,
        bindings,
        reference_policy=reference_policy,
        composition_anchor_cast_count=composition_anchor_cast_count,
    )
    sampling_model = ["164", 0] if use_lora else ["144", 0]

    graph = {
        # ── Resolution & length calc nodes ──
        "115": {
            "class_type": "ResolutionSelector",
            "inputs": {
                "aspect_ratio": ASPECT_RATIO_CHOICES[aspect_ratio],
                "megapixels": megapixels,
                "multiple": 32,
            },
        },
        "132": {
            "class_type": "PrimitiveFloat",
            "inputs": {
                "value": float(duration_seconds),
            },
        },
        "131": {
            "class_type": "ComfyMathExpression",
            "inputs": {
                "expression": "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17",
                "values.a": ["132", 0],
            },
        },

        # ── Loaders ──
        "119": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": H3_VIDEO_VAE},
        },
        "120": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": H3_AUDIO_VAE},
        },
        "127": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": H3_UNET,
                "weight_dtype": "default",
            },
        },
        "128": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": H3_CLIP,
                "type": "minimax",
                "device": "default",
            },
        },

        # ── Prompt ──
        "138": {
            "class_type": "PrimitiveStringMultiline",
            "inputs": {"value": prompt},
        },

        # ── SageAttention optimization (KJNodes) ──
        # Node 144: UNET → SageAttention
        "144": {
            "class_type": "PathchSageAttentionKJ",
            "inputs": {
                "sage_attention": sage_value,
                "allow_compile": True,
                "model": ["127", 0],
            },
        },

        # ── Turbo LoRA ──
        # Model flow: 127 → 144 → 164; omitted entirely when Turbo is disabled.
        **({
            "164": {
                "class_type": "MiniMaxH3TurboLoRA",
                "inputs": {
                    "lora_name": sampling["lora_name"],
                    "strength": sampling["lora_strength"],
                    "low_vram": sampling["low_vram"],
                    "model": ["144", 0],
                },
            },
        } if use_lora else {}),
        
        # ── Adaptive joint AV Turbo sampler ──
        # The installed plugin lacks core ModelSamplingAV, so this node applies
        # separate video/audio flow shifts on the legacy compatibility path.
        "165": {
            "class_type": "MiniMaxH3TurboSampler",
            "inputs": {},
        },

        # ── Reference-to-video conditioning + joint AV latent ──
        "136": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "prompt": ["138", 0],
                "width": ["115", 0],
                "height": ["115", 1],
                "length": ["131", 1],
                "ref_image_size": "max" if reference_fidelity == "identity" else "match",
                "clip": ["128", 0],
                "vae": ["119", 0],
                "audio_vae": ["120", 0],
                **ref_image_inputs,
            },
        },

        # ── Sampling setup ──
        "129": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(seed)},
        },
        # Scheduler drives sigmas; the adaptive Turbo sampler drives both the
        # video and audio clocks.  Do not wire the unrelated KSamplerSelect.
        "124": {
            "class_type": "BasicScheduler",
            "inputs": {
                "scheduler": sampling["scheduler"],
                "steps": sampling["steps"],
                "denoise": 1,
                "model": sampling_model,
            },
        },
        "126": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": sampling_model,
                "conditioning": ["136", 0],
            },
        },

        # ── Joint AV sampling ──
        "125": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["129", 0],
                "guider": ["126", 0],
                "sampler": ["165", 0],
                "sigmas": ["124", 0],
                "latent_image": ["136", 1],
            },
        },

        # ── Decode video + audio ──
        "122": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["125", 0],
                "vae": ["119", 0],
            },
        },
        "121": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["125", 0],
                "vae": ["120", 0],
            },
        },

        # ── CreateVideo (images + audio) → SaveVideo ──
        "130": {
            "class_type": "CreateVideo",
            "inputs": {
                "fps": VIDEO_FPS,
                "bit_depth": 8,
                "images": ["122", 0],
                "audio": ["121", 0],
            },
        },
        "92": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename_prefix": (
                    f"video/{ep_id}/{name_prefix or 'panel'}"
                    if ep_id
                    else f"video/{name_prefix or 'panel'}"
                ),
                "format": "auto",
                "codec": "auto",
                "video-preview": "",
                "video": ["130", 0],
            },
        },
    }

    # Inject character reference LoadImage nodes
    graph.update(char_ref_nodes)
    return graph


def _copy_image_to_comfy(src_path: Path) -> str:
    """Content-address a reference image in ComfyUI/input and return its name."""
    source = Path(src_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"reference image does not exist: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._") or "reference"
    staged_name = f"ref_{digest}_{safe_stem[:64]}{source.suffix.lower()}"
    dst = (COMFY / "input" / staged_name).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if source != dst and not dst.exists():
        temp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(source, temp)
        temp.replace(dst)
    return staged_name


def _find_video_in_outputs(outputs: dict) -> Optional[Path]:
    """Find the generated video file from ComfyUI outputs.
    
    ComfyUI may output video in 'videos', 'gifs', or 'images' (with animated=True) fields.
    """
    for node_data in outputs.values():
        # Check 'videos' field
        for vid in node_data.get("videos", []):
            subfolder = vid.get("subfolder", "")
            filename = vid["filename"]
            path = COMFY / "output" / subfolder / filename
            if path.exists():
                return path
        # Check 'gifs' field
        for vid in node_data.get("gifs", []):
            subfolder = vid.get("subfolder", "")
            filename = vid["filename"]
            path = COMFY / "output" / subfolder / filename
            if path.exists():
                return path
        # Check 'images' field - ComfyUI sometimes puts videos here with animated=True
        for img in node_data.get("images", []):
            filename = img.get("filename", "")
            if filename.endswith(('.mp4', '.webm', '.mov', '.avi', '.gif')):
                subfolder = img.get("subfolder", "")
                path = COMFY / "output" / subfolder / filename
                if path.exists():
                    return path
    return None


def _poll_completion(prompt_id: str, timeout: int = 1800) -> dict:
    """Poll /history until the prompt completes or times out."""
    start = time.time()
    while time.time() - start < timeout:
        h = _api("/history/" + prompt_id)
        if prompt_id in h:
            result = h[prompt_id]
            status = result.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI error: {status}")
            return result
        time.sleep(5)
    raise TimeoutError(f"ComfyUI did not complete within {timeout}s")


def _normalize_dialogue(dialogue: str) -> tuple[str, str]:
    """Strip narrator marker. Returns (cleaned_text, kind) where kind is
    'dialogue', 'narration', or 'none'."""
    if not dialogue:
        return "", "none"
    d = dialogue.strip()
    if d.startswith("（旁白）") or d.startswith("(旁白)") or d.startswith("旁白："):
        for prefix in ("（旁白）", "(旁白)", "旁白："):
            if d.startswith(prefix):
                d = d[len(prefix):].strip()
                break
        return d, "narration"
    return d, "dialogue"


_H3_ENGLISH_WORD_RE = re.compile(r"\b[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*\b")
_H3_PACKAGE_TAG_RE = re.compile(r"^\[([A-Z0-9_ -]+)\]\s*(.*)$", re.MULTILINE)


def count_h3_english_words(prompt: str) -> int:
    """Count the English-style words governed by the H3 prompt budget."""
    return len(_H3_ENGLISH_WORD_RE.findall(str(prompt or "")))


def _assert_h3_prompt_budget(prompt: str, maximum: int, label: str) -> None:
    count = count_h3_english_words(prompt)
    if count > maximum:
        raise ValueError(f"H3 {label} exceeds hard budget: {count}>{maximum} English words")


def _compact_fragment(value: Any, max_english_words: int, *, max_chars: int = 420) -> str:
    """Collapse one approved field without copying its surrounding contract."""
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ;,.\n\t")
    if not text:
        return ""
    matches = list(_H3_ENGLISH_WORD_RE.finditer(text))
    truncated = False
    if len(matches) > max_english_words:
        text = text[:matches[max_english_words - 1].end()].rstrip(" ;,.")
        truncated = True
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ;,.")
        truncated = True
    if truncated:
        # A hard word cutoff used to emit fragments such as "above the." and
        # "no other;" into the director prompt. Back off trailing connector
        # words until the retained fragment ends on a concrete content word.
        dangling = re.compile(
            r"(?:\s|[,;:])+\b(?:a|an|the|and|or|but|with|while|above|below|in|on|at|to|from|of|no|other)\b$",
            re.I,
        )
        while dangling.search(text):
            text = dangling.sub("", text).rstrip(" ;,.")
    return text


def _package_tags(panel: dict[str, Any]) -> dict[str, str]:
    package = panel.get("prompt_package") or {}
    positive = str(package.get("positive_prompt") or "") if isinstance(package, dict) else ""
    return {
        match.group(1).strip().replace(" ", "_"): match.group(2).strip()
        for match in _H3_PACKAGE_TAG_RE.finditer(positive)
    }


def _short_untagged_package_canon(panel: dict[str, Any]) -> str:
    """Keep only a genuinely short untagged canon note; tagged packages are parsed."""
    package = panel.get("prompt_package") or {}
    positive = str(package.get("positive_prompt") or "").strip() if isinstance(package, dict) else ""
    if not positive or _H3_PACKAGE_TAG_RE.search(positive):
        return ""
    return _compact_fragment(positive, 24, max_chars=220)


def _director_shot_plan(panel: dict[str, Any]) -> dict[str, Any]:
    """Return optional director fields without copying the full prompt package."""
    package = panel.get("prompt_package") or {}
    packaged = package.get("shot_plan") if isinstance(package, dict) else {}
    plan = dict(packaged) if isinstance(packaged, dict) else {}
    for key in (
        "story_function", "blocking", "screen_direction", "axis", "eyeline",
        "dominant_camera_move", "camera_plan", "first_state", "final_state",
        "first_keyframe_strategy", "last_keyframe_strategy", "keyframe_strategy",
        "transition", "tr", "sound_bridge", "risk", "risk_code",
        "failure_code", "failure_codes",
    ):
        if panel.get(key) is not None:
            plan[key] = panel[key]
    return plan


def _compiled_action_authority(panel: dict[str, Any]) -> dict[str, Any] | None:
    """Compile the canonical action contract and return its renderer audit record.

    Old projects without an action contract keep the legacy prompt lane.  Once
    any canonical/compatibility action field is present, however, compilation
    is fail-closed and display prose such as ``visible_action`` is not trusted.
    """
    has_contract = (
        isinstance(panel.get("action_spec"), dict)
        or isinstance(panel.get("action_components"), dict)
        or bool(str(panel.get("action_code") or "").strip())
    )
    if not has_contract:
        return None
    compiled = compile_panel_action(panel, allow_legacy=True)
    compiled_text = str(compiled["h3_action_en"])
    return {
        "catalog_version": compiled["catalog_version"],
        "actor_id": compiled["actor_id"],
        "action_code": compiled["action_code"],
        "target": compiled["target"],
        "spec_sha256": compiled["spec_sha256"],
        "compiled_h3_sha256": hashlib.sha256(compiled_text.encode("utf-8")).hexdigest(),
        "source": (
            "panel.action_spec"
            if isinstance(panel.get("action_spec"), dict)
            else "panel.action_components.exact_legacy"
        ),
        "h3_action_en": compiled_text,
        "start_state": compiled["start_state"],
        "end_state": compiled["end_state"],
    }


def _reviewer_correction_audit(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the latest human-QA rejection into bounded English H3 constraints.

    Reviewer prose remains audit evidence and is never copied verbatim into the
    visual prompt.  The category plus canonical action contract produce a short,
    model-facing correction so a retry is meaningfully different from a blind
    seed reroll while retaining the official Ref2VA prompt shape.
    """
    feedback = panel.get("qa_retry_feedback")
    if not isinstance(feedback, Mapping):
        return {}
    reason = re.sub(r"\s+", " ", str(feedback.get("reason") or "")).strip()
    if not reason:
        return {}
    category = str(feedback.get("category") or "other").strip().casefold()
    action_spec = panel.get("action_spec")
    if not isinstance(action_spec, Mapping):
        action_spec = {}
    target = _compact_fragment(action_spec.get("target") or "action target", 8, max_chars=90)
    target = re.sub(r"^(?:(?:the\s+)?only|the|one|a|an)\s+", "", target, flags=re.I)
    final_state = _compact_fragment(
        action_spec.get("end_state") or panel.get("final_state") or "the approved final state",
        14,
        max_chars=170,
    )
    if category == "continuity_or_state":
        directive = (
            f"QA correction: exactly one {target} exists in every frame. Never duplicate, split, teleport, pre-place or "
            f"replace it. Transfer it once by direct hand contact; receiver takes it and prior holder releases it. Reach "
            f"{final_state} before the deadline and hold."
        )
    elif category == "action_timing_or_edit_window":
        directive = (
            f"QA correction: start the {target} action at frame zero without pause. Reach {final_state} before the stated "
            "deadline, then hold that state through the delivery edit window."
        )
    elif category == "identity_or_cast":
        directive = (
            "QA correction: preserve the exact approved cast count, identity, face, hair, wardrobe, "
            "body and spatial role in every frame; never add, omit, merge, duplicate or replace a person."
        )
    elif category == "composition_or_scene":
        directive = (
            "QA correction: preserve the approved Picture composition, scene geography, camera side, "
            "lighting, weather boundary and persistent props; never invent a new location, prop or layout."
        )
    else:
        directive = (
            f"QA correction: execute only the approved action involving {target}, preserve every approved "
            "identity, prop and scene constraint, and reach the approved final state before the delivery edit deadline."
        )
    directive += " Never render correction notes, labels, captions or letters."
    source_payload = {
        "reason": reason,
        "category": category,
        "at": str(feedback.get("at") or ""),
    }
    return {
        "category": category,
        "source_sha256": hashlib.sha256(
            json.dumps(source_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "directive": directive,
    }


def _runtime_action_and_camera(panel: dict[str, Any], tags: dict[str, str]) -> tuple[str, str]:
    director = _director_shot_plan(panel)
    action_authority = _compiled_action_authority(panel)
    cuts = panel.get("cuts") or []
    cut = cuts[0] if cuts and isinstance(cuts[0], dict) else {}
    blocking = director.get("blocking") if isinstance(director.get("blocking"), dict) else {}
    if action_authority:
        candidate = action_authority["h3_action_en"]
        # Canonical action text deliberately repeats the opening and final
        # states so it can be hashed and audited as a standalone contract.
        # The H3 director prompt already owns dedicated opening/final lanes;
        # copying a verbose canonical sentence into its action lane a second
        # time can push an otherwise valid prompt past the 512-word limit.
        # Keep short contracts byte-for-byte compatible, but for verbose ones
        # extract only the authoritative actor/verb/target motion.  The full
        # canonical sentence remains in the episode contract; graph/metadata
        # retain both the full sentence and its hash as immutable audit data.
        if count_h3_english_words(candidate) > 30:
            actor_id = str(action_authority.get("actor_id") or "").strip()
            marker = f", {actor_id} " if actor_id else ""
            ending_marker = ", ending with "
            start_at = candidate.find(marker) if marker else -1
            end_at = candidate.rfind(ending_marker)
            if start_at >= 0 and end_at > start_at:
                motion = candidate[start_at + 2:end_at]
            else:
                motion = candidate
            candidate = _compact_fragment(motion, 16, max_chars=180)
    else:
        candidate = _compact_fragment(
            panel.get("visible_action")
            or blocking.get("motion")
            or blocking.get("action")
            or panel.get("action")
            or panel.get("motion")
            or cut.get("editorial_shot_description")
            or cut.get("shot_description")
            or tags.get("SHOT_TIMELINE")
            or "natural continuous character action",
            22,
            max_chars=260,
        )
    camera_plan = director.get("camera_plan") if isinstance(director.get("camera_plan"), dict) else {}
    camera_parts: list[str] = []
    shot_size = str(camera_plan.get("shot_size") or camera_plan.get("size") or "").replace("_", " ").strip()
    angle = str(camera_plan.get("angle") or "").replace("_", " ").strip()
    composition = str(camera_plan.get("composition") or camera_plan.get("comp") or "").replace("_", " ").strip()
    movement = str(
        director.get("dominant_camera_move")
        or camera_plan.get("dominant_move")
        or camera_plan.get("movement")
        or camera_plan.get("move")
        or panel.get("camera_movement")
        or ""
    ).replace("_", " ").strip()
    if shot_size:
        camera_parts.append(f"{shot_size} shot")
    if angle:
        camera_parts.append(f"{angle} angle")
    if composition:
        camera_parts.append(f"{composition} composition")
    if movement:
        camera_parts.append(movement)
    camera = _compact_fragment("; ".join(camera_parts), 20, max_chars=230)
    action = candidate
    if not action_authority and not camera and "," in candidate:
        prefix, remainder = (part.strip() for part in candidate.split(",", 1))
        if re.search(r"\b(?:shot|camera|view|pan|tilt|dolly|track|zoom|crane|handheld)\b", prefix, re.I):
            camera, action = prefix, remainder
    if not camera:
        camera = "one stable eye-level camera path with a gentle forward drift"

    semantic = _panel_semantic_text(panel)
    is_indoor_store = bool(
        re.search(r"\b(?:convenience store|store interior|inside store)\b|便利店|店内", semantic)
    )
    enters_store = bool(re.search(r"\b(?:enter|enters|entering)\b|进入|进店", semantic))
    if is_indoor_store and not enters_store and shot_size and "close" not in shot_size.casefold():
        camera = _compact_fragment(
            f"{camera}; frame spans head level to checkout counter; "
            "only two people, plain glass and checkout counter",
            28,
            max_chars=320,
        )

    # Keep ensemble/device staging explicit without spending the runtime budget
    # on several near-identical clauses from the approved contract.  This is
    # intentionally conditional: ordinary shots must not gain a device prop.
    character_ids = [item for item in (panel.get("character_ids") or []) if item]
    ensemble_source = " ".join(
        str(value or "")
        for value in (
            panel.get("action"),
            panel.get("final_state"),
            panel.get("end_state"),
            tags.get("LAST_FRAME"),
        )
    )
    if not action_authority and (
        len(character_ids) == 5
        and re.search(r"\b(?:device|devices|phone|phones|smartphone|smartphones)\b", ensemble_source, re.I)
        and re.search(r"\b(?:each|own)\b", ensemble_source, re.I)
    ):
        action = "exactly 5 seated friends operate one device each continuously"

    # H3 responds better to one unambiguous camera path than to a verbose shot
    # description.  Preserve the approved panel-2 intent as a compact command.
    if all(re.search(pattern, camera, re.I) for pattern in (r"\bunbroken\b", r"\bmedium-wide\b", r"\bslow\b", r"\bpush\b")):
        camera = "one unbroken medium-wide slow push"
    runtime_action = action if action_authority else _compact_fragment(action, 18, max_chars=220)
    return runtime_action, _compact_fragment(camera, 32, max_chars=340)


def _compile_scene_continuity_lock(raw_lock: Any) -> str:
    """Preserve structured scene constraints in the runtime H3 prompt.

    Scene contracts use a mapping because weather, geography and persistent
    props have different authority.  Treating that mapping as prose used to
    drop it completely; in a rainy-night interior this let H3 move the rain
    through the glass and into the room.  Weather is deliberately first so the
    physical inside/outside boundary survives even when the prompt is compact.
    """
    if isinstance(raw_lock, str):
        return _compact_fragment(raw_lock, 12, max_chars=150)
    if not isinstance(raw_lock, Mapping):
        return ""
    parts: list[str] = []
    weather_detail = _compact_fragment(raw_lock.get("weather_boundary"), 18, max_chars=220)
    geography_detail = _compact_fragment(raw_lock.get("geography"), 14, max_chars=180)
    is_rainy_interior = bool(
        re.search(r"\b(?:rain|rainy|storm)\b|雨|暴雨", weather_detail, re.I)
        and re.search(r"\b(?:inside|interior|indoor|store|room)\b|室内|店内", geography_detail, re.I)
    )
    if is_rainy_interior:
        parts.append(
            "weather boundary: the deep-blue wet exterior stays behind closed glass while indoor air, counter, "
            "floor, and every visible interior surface remain uniformly dry and clear"
        )
    elif weather_detail:
        parts.append(f"weather boundary: {weather_detail}")
    for label, value in (
        ("geography", geography_detail),
        ("persistent hero props", _compact_fragment(raw_lock.get("hero_props"), 14, max_chars=180)),
    ):
        if value:
            parts.append(f"{label}: {value}")
    text_surface = _compact_fragment(raw_lock.get("text_surface_lock"), 16, max_chars=200)
    if text_surface:
        # H3 receives one positive multimodal prompt rather than a separate
        # negative prompt.  Listing forbidden object names here made the model
        # draw those exact concepts.  Preserve the contract as an affirmative
        # visual state instead of echoing its negative-token inventory.
        parts.append(
            "visible-surface treatment: uniform blank geometric color fields on every visible prop, package, wall, and fixture"
        )
    if is_rainy_interior:
        parts.append(
            "upper background treatment: uninterrupted plain deep-blue glass and blank wall fields with simple straight architectural lines"
        )
    return "; ".join(parts)


def _panel_semantic_text(panel: Mapping[str, Any]) -> str:
    semantic_fields = {
        key: panel.get(key)
        for key in (
            "scene_description", "scene_context", "story_context", "visible_action",
            "action", "first_state", "final_state", "end_state", "audio_cues",
        )
        if panel.get(key) not in (None, "", [], {})
    }
    return json.dumps(semantic_fields, ensure_ascii=False, sort_keys=True).casefold()


def resolve_panel_audio(panel: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve obvious semantic contradictions without rewriting approved cues.

    The Web form historically defaulted every story to office room tone and
    epic brass.  The LLM may still explicitly choose those presets, so the
    compiler only intervenes for two high-confidence contradictions and records
    each intervention in the immutable graph snapshot.
    """
    requested_music = str(panel.get("background_music") or "soft_piano").strip()
    requested_ambience = str(panel.get("ambience") or "office_quiet").strip()
    resolved_music = requested_music
    resolved_ambience = requested_ambience
    semantic = _panel_semantic_text(panel)
    overrides: list[dict[str, str]] = []

    has_rain = bool(re.search(r"\b(?:rain|rainy|storm|wet asphalt|wet pavement)\b|雨|暴雨|雷雨", semantic))
    is_office = bool(re.search(r"\b(?:office|workplace|boardroom)\b|办公室|写字楼", semantic))
    is_indoor = bool(re.search(r"\b(?:inside|interior|indoor|store|room)\b|室内|店内", semantic))
    if requested_ambience == "auto_contextual":
        if has_rain:
            resolved_ambience = "rain_outside_glass" if is_indoor else "rain_night_city"
        elif re.search(r"\b(?:forest|woods|mountain|nature)\b|森林|山林|自然", semantic):
            resolved_ambience = "forest_morning"
        elif re.search(r"\b(?:subway|metro|underground station|train platform)\b|地铁|站台", semantic):
            resolved_ambience = "subway_crowd"
        elif re.search(r"\b(?:street|road|market|city crowd)\b|街道|市集|人群", semantic):
            resolved_ambience = "street_bustle"
        elif re.search(r"\b(?:home|bedroom|kitchen|apartment)\b|家中|卧室|厨房|公寓", semantic):
            resolved_ambience = "home_intimate"
        elif re.search(r"\b(?:night|midnight|empty)\b|深夜|午夜|空旷", semantic):
            resolved_ambience = "night_empty"
        else:
            resolved_ambience = "silence"
        overrides.append({
            "field": "ambience", "from": requested_ambience, "to": resolved_ambience,
            "reason": "contextual_audio_director_selection",
        })
    elif requested_ambience == "office_quiet" and has_rain and not is_office:
        resolved_ambience = "rain_outside_glass" if is_indoor else "rain_night_city"
        overrides.append({
            "field": "ambience",
            "from": requested_ambience,
            "to": resolved_ambience,
            "reason": "rain_or_storm_scene_conflicts_with_office_room_tone",
        })

    gentle_signals = re.findall(
        r"\b(?:charity|kindness|gentle|warm cup|wallet|relief|thank|donation|compassion)\b|"
        r"公益|善意|温暖|热饮|钱包|感谢|捐赠",
        semantic,
    )
    if requested_music == "auto_contextual":
        if re.search(r"\b(?:battle|war|victory|heroic|charge|explosion)\b|战斗|战争|胜利|冲锋|爆炸", semantic):
            resolved_music = "epic_brass"
        elif re.search(r"\b(?:suspense|threat|mystery|stalk|danger|tense)\b|悬疑|威胁|危险|紧张", semantic):
            resolved_music = "suspense_dark"
        elif re.search(r"\b(?:ancient china|wuxia|jianghu|guzheng)\b|古风|武侠|江湖|古代", semantic):
            resolved_music = "chinese_folk"
        elif re.search(r"\b(?:cyber|technology|urban|city|electronic)\b|赛博|科技|都市|电子", semantic):
            resolved_music = "urban_electronic"
        elif gentle_signals:
            resolved_music = "soft_piano"
        else:
            resolved_music = "string_orch"
        overrides.append({
            "field": "background_music", "from": requested_music, "to": resolved_music,
            "reason": "contextual_audio_director_selection",
        })
    elif requested_music == "epic_brass" and gentle_signals:
        resolved_music = "soft_piano"
        overrides.append({
            "field": "background_music",
            "from": requested_music,
            "to": resolved_music,
            "reason": "gentle_human_story_conflicts_with_epic_brass",
        })

    return {
        "requested": {
            "background_music": requested_music,
            "ambience": requested_ambience,
        },
        "resolved": {
            "background_music": resolved_music,
            "ambience": resolved_ambience,
        },
        "overrides": overrides,
    }


def compile_h3_runtime_prompt(
    panel: dict[str, Any],
    character_desc: str = "",
    reference_bindings: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Compile an approved panel into one concise H3-native runtime prompt.

    Approved image references, not repeated character/scene prose, are the
    identity and layout authority.  Contract payloads are reduced to one action,
    three chronological beats, one camera path, look/lighting, final state, and
    the exact spoken/audio schedule.  ``character_desc`` is deliberately not
    copied because it duplicates the approved character images.
    """
    del character_desc
    duration = float(panel.get("duration_seconds") or DEFAULT_DURATION_SECONDS)
    timing = build_timing_contract(panel, duration)
    tags = _package_tags(panel)
    director = _director_shot_plan(panel)
    action_authority = _compiled_action_authority(panel)
    action, camera = _runtime_action_and_camera(panel, tags)
    aspect = str(panel.get("aspect_ratio") or "16:9")

    raw_style = str(panel.get("style") or panel.get("style_header") or "").strip()
    expanded_style = STYLE_PRESETS.get(raw_style.lower()) or STYLE_PRESETS.get(raw_style)
    style = _compact_fragment(
        tags.get("VISUAL_BIBLE") or expanded_style or raw_style
        or "cinematic hand-drawn animation with coherent anatomy and clean linework",
        10,
        max_chars=140,
    )
    scene_context = panel.get("scene_context") or {}
    scene = ""
    continuity_lock = ""
    if isinstance(scene_context, dict):
        scene = _compact_fragment(
            scene_context.get("model_prompt_en")
            or scene_context.get("positive_prompt")
            or scene_context.get("description"),
            18,
            max_chars=220,
        )
        continuity_lock = _compile_scene_continuity_lock(scene_context.get("continuity_lock"))
    scene = scene or _compact_fragment(tags.get("SCENE_LOCK") or panel.get("scene_description"), 18, max_chars=220)
    # Avoid strong retail priors that commonly hallucinate overhead safety
    # graphics.  The approved scene image remains the layout authority; this
    # text only supplies the neutral environment class.
    scene = re.sub(
        r"\b(?:anime\s+)?convenience\s+store\s+interior\b",
        "small neighborhood retail interior",
        scene,
        flags=re.I,
    )

    story_context = panel.get("story_context") or {}
    story = ""
    if isinstance(story_context, dict):
        story = _compact_fragment(
            story_context.get("logline") or story_context.get("summary") or story_context.get("synopsis"),
            18,
            max_chars=220,
        )
        # Editorial CJK prose was observed being copied verbatim into H3 frames
        # as an unwanted subtitle. Visual semantics already live in the scene,
        # action and beat lanes; only timed dialogue may remain non-English.
        if story and not story.isascii():
            story = ""
    canon = _short_untagged_package_canon(panel)
    first_state = _compact_fragment(
        (action_authority or {}).get("start_state") or director.get("first_state") or panel.get("first_state")
        or "approved opening",
        10,
        max_chars=120,
    )
    final_state = _compact_fragment(
        (action_authority or {}).get("end_state") or director.get("final_state")
        or panel.get("final_state") or panel.get("end_state") or tags.get("LAST_FRAME")
        or "the completed action in the approved scene composition",
        16,
        max_chars=190,
    )
    blocking = director.get("blocking")
    has_structured_blocking = isinstance(blocking, dict) and any(
        str(blocking.get(key) or "").strip() for key in ("start", "motion", "action", "end")
    )
    if action_authority:
        blocking_start, blocking_motion, blocking_end = first_state, action, final_state
    elif has_structured_blocking:
        blocking_start = _compact_fragment(blocking.get("start") or first_state, 12, max_chars=150)
        blocking_motion = _compact_fragment(
            blocking.get("motion") or blocking.get("action") or action, 16, max_chars=190,
        )
        blocking_end = _compact_fragment(blocking.get("end") or final_state, 14, max_chars=170)
    else:
        blocking_start, blocking_motion, blocking_end = first_state, action, final_state

    actual_duration = float(timing["actual_duration_seconds"])
    geography = []
    for label, key in (("screen direction", "screen_direction"), ("axis", "axis"), ("eyeline", "eyeline")):
        value = _compact_fragment(director.get(key), 6, max_chars=70)
        if value:
            geography.append(f"{label}: {value}")
    camera_instruction = (
        f"{camera}; one dominant path"
        + (f"; {'; '.join(geography)}" if geography else "")
    )
    if str(panel.get("continuity_reference_policy") or "") == "composition_anchor_first":
        continuity_instruction = (
            "Prior tail locks identity, cast, props, lighting, time of day, weather boundary, and state only; "
            "new framing/action starts immediately; never replay, freeze, or linger"
        )
    else:
        continuity_instruction = (
            "Preserve identity, cast count, direction, props, lighting, time of day, weather boundary, and prior state"
        )
    continuity_parts = [continuity_instruction]
    if continuity_lock:
        continuity_parts.append(continuity_lock)
    if canon:
        continuity_parts.append(canon)
    reviewer_correction = _reviewer_correction_audit(panel)
    if reviewer_correction:
        continuity_parts.append(str(reviewer_correction["directive"]))

    audio_resolution = resolve_panel_audio(panel)
    bgm_key = str(audio_resolution["resolved"]["background_music"])
    ambience_key = str(audio_resolution["resolved"]["ambience"])
    bgm = _compact_fragment(MUSIC_PRESETS.get(bgm_key, bgm_key), 18, max_chars=180)
    ambience = _compact_fragment(AMBIENCE_PRESETS.get(ambience_key, ambience_key), 16, max_chars=180)
    result = compile_h3_director_prompt(
        style=f"{style}; practical motivated cinematic lighting; coherent anatomy and stable framing",
        aspect_ratio=aspect,
        duration_seconds=actual_duration,
        narrative_duration_seconds=float(
            panel["edit_duration_seconds"]
            if panel.get("edit_duration_seconds") is not None else actual_duration
        ),
        scene=scene or "the approved scene reference with locked layout, lighting, palette, weather, and persistent props",
        action=action,
        first_state=blocking_start,
        final_state=blocking_end,
        camera=camera_instruction,
        continuity="; ".join(part for part in continuity_parts if part),
        cast_count=len([value for value in panel.get("character_ids") or [] if value]),
        spoken_dialogue=timing["spoken_dialogue"],
        audio_cues=timing["audio_cues"],
        ambience=ambience,
        music=bgm,
        bindings=reference_bindings or [],
    )
    _assert_h3_prompt_budget(result["prompt"], H3_PROMPT_BODY_MAX_ENGLISH_WORDS, "runtime body")
    return str(result["prompt"])


def _build_ref2va_prompt(panel: dict, character_desc: str = "") -> str:
    """Build the H3 ref2va prompt following MiniMax H3 best practices.

    Structure (following fal.ai / Krea.ai / HF docs guides):
      1. STYLE HEADER           — aesthetic + camera + format (locks visual look)
      2. CHARACTER ANCHOR        — full description of all on-screen characters
                                   (locks identity, wardrobe, hair — copy verbatim
                                    across every shot in the same episode)
      3. SCENE ANCHOR            — environment, lighting, time of day
                                   (repeated identically across shots to lock setting)
      4. CAST & COUNT            — explicit number + spatial relation
      5. CONTINUITY CONSTRAINTS  — explicit "do NOT change" rules to fight drift
      6. OPENING SHOT            — what's on screen at 0.00s
      7. CAMERA MOTION           — before action, so it frames the beat
      8. TIMELINE / ACTION       — beat-by-beat with onset → development → result
      9. DIALOGUE                — speaker, line (verbatim), tone, language
     10. EMOTION                 — single word/phrase for the whole beat
     11. AUDIO                   — music + ambience + transition
     12. TECHNICAL FOOTER        — duration, aspect, fps, style lock
    """
    style = panel.get("style", "").strip()
    voice_language = panel.get("voice_language", "Chinese").strip()
    background_music_key = panel.get("background_music", "soft_piano").strip()
    ambience_key = panel.get("ambience", "office_quiet").strip()
    transition = panel.get("transition", "smooth cut").strip()

    # New detailed format fields
    scene_description = panel.get("scene_description", "").strip()
    timeline = panel.get("timeline", [])
    camera_movement = panel.get("camera_movement", "").strip()
    emotion = panel.get("emotion", "").strip()
    duration = float(panel.get("duration_seconds") or DEFAULT_DURATION_SECONDS)
    timing = build_timing_contract(panel, duration)
    spoken_dialogue = timing["spoken_dialogue"]
    on_screen_text = timing["on_screen_text"]
    audio_cues = timing["audio_cues"]
    aspect_ratio = str(panel.get("aspect_ratio") or "16:9")

    # Legacy format fallback
    motion = panel.get("motion", "smooth cinematic transition").strip()
    legacy_dialogue = panel.get("dialogue", "") if isinstance(panel.get("dialogue"), str) else ""
    legacy_speaker = panel.get("speaker", "").strip()

    bgm = MUSIC_PRESETS.get(background_music_key, background_music_key)
    ambience = AMBIENCE_PRESETS.get(ambience_key, ambience_key)

    # ---- 1. STYLE HEADER ---------------------------------------------------
    # Locks aesthetic, framing, and shot grammar so every shot looks like the
    # same film. If the panel's `style` is a short Chinese tag like "吉卜力" or
    # "国风", STYLE_PRESETS expands it into a rich H3-grade cinematic preamble.
    # Otherwise the raw style string is passed through verbatim.
    raw_style = style.strip()
    style_expanded = STYLE_PRESETS.get(raw_style.lower()) if raw_style else None
    if style_expanded is None and raw_style:
        style_expanded = STYLE_PRESETS.get(raw_style)
    if style_expanded:
        style_header = f"[STYLE] {style_expanded}. {panel.get('aspect_ratio', '16:9')} frame, 24fps, cinematic shallow depth of field, consistent film grain throughout, locked aesthetic for the entire episode."
    elif raw_style:
        style_header = f"[STYLE] {raw_style}. {panel.get('aspect_ratio', '16:9')} frame, 24fps, cinematic shallow depth of field, consistent film grain throughout."
    else:
        style_header = f"[STYLE] {STYLE_DEFAULT}. {panel.get('aspect_ratio', '16:9')} frame, 24fps, cinematic shallow depth of field, consistent film grain throughout."

    # ---- 2. CHARACTER ANCHOR ----------------------------------------------
    # Full description of every on-screen character. Copied verbatim into
    # every prompt so H3 never re-imagines their face / wardrobe / hair.
    char_anchor = ""
    if character_desc:
        char_anchor = (
            "[CHARACTERS — DO NOT CHANGE] "
            + character_desc
            + " Lock these appearances EXACTLY: same face, same hair color and style, same wardrobe colors and cuts, same body proportions, same age, same distinguishing features (scars, accessories, makeup). Do not invent new clothing, do not change hairstyle, do not change skin tone, do not swap actors between shots."
        )

    # ---- 3. SCENE ANCHOR --------------------------------------------------
    # Same environment copy-pasted across shots of the same scene so the
    # background doesn't drift from shot 1 to shot 12.
    scene_anchor = ""
    if scene_description:
        scene_anchor = f"[SCENE — KEEP CONSISTENT] {scene_description}. This is the SAME location as previous shots — do not change furniture, walls, lighting fixtures, props, color palette, or weather."

    # ---- 4–5. CAST/COUNT + DRIFT CONSTRAINTS -----------------------------
    # H3 tends to "add" people. Force the exact count and lock the wardrobe.
    continuity_rules = (
        "[CONTINUITY — DO NOT VIOLATE] "
        "(a) On-screen cast count is exactly as described below — do not add bystanders, extras, or background characters that speak or interact. "
        "(b) Wardrobe, hair, makeup, accessories, and props described above are PERMANENT for this character across every shot of the episode. "
        "(c) Do NOT change room layout, furniture position, or background props between shots. "
        "(d) Do NOT change lighting direction or color temperature between shots in the same scene. "
        "(e) Maintain the SAME film grain, color grading, lens focal length, and depth of field as the reference images. "
        f"(f) {aspect_ratio} composition throughout — subjects remain at roughly the same screen position across the timeline unless the camera explicitly moves."
    )

    # ---- 6–8. OPENING SHOT + CAMERA + ACTION ------------------------------
    # For the timeline we build the opening-shot anchor from the first beat,
    # then list camera + action for every subsequent beat. This matches the
    # official MiniMax "first-frame anchor → action onset → continuous
    # development → result or reaction" structure.
    #
    # Each beat is rendered with a SECOND-BY-SECOND prefix (SECOND 0-1, etc.)
    # so H3 can follow the action beat-by-beat rather than averaging the whole
    # 10 seconds into a single motion.
    timeline_block = ""
    if timeline:
        n = len(timeline)
        timeline_block = "[OPENING SHOT — at 0.00s the frame MUST match the first reference image exactly]\n"
        timeline_block += "Lock the initial composition: character position, clothing folds, hair, prop placement, lighting direction, background.\n\n"
        timeline_block += "[SECOND-BY-SECOND ACTION MAP]\n"
        for i, item in enumerate(timeline, 1):
            t_range = (item.get("time") or "").strip()
            action = (item.get("action") or "").strip()
            camera = (item.get("camera") or "").strip()
            beat_label = f"[Beat {i} — t={t_range}]"
            timeline_block += f"{beat_label}\n"
            if t_range.startswith("0") and ("1s" in t_range or "0-1" in t_range or i == 1):
                timeline_block += "  FIRST 0.Xs: hold the opening composition. No body movement yet — only subtle ambient motion (wind, breath, fabric micro-settle).\n"
            if camera:
                timeline_block += f"  Camera: {camera}\n"
            if action:
                timeline_block += f"  Action: {action}\n"
            # Add a hint for the final beat to lock the last frame
            if i == n:
                timeline_block += "  LAST 0.Xs: settle into the LAST reference image composition. Match expression, pose, prop position.\n"
            timeline_block += "\n"
    elif motion:
        timeline_block = f"[CAMERA] {motion}\n[ACTION] Natural cinematic motion."

    # ---- 7. CAMERA MOVEMENT (top-level override) -------------------------
    camera_block = ""
    if camera_movement:
        camera_block = f"[CAMERA — OVERALL] {camera_movement}."

    # ---- 9. DIALOGUE ------------------------------------------------------
    # Detect language per-line, never translate the line itself, mark speaker
    # and tone — this is what H3 uses to pick the voice timbre and emotion.
    dialogue_block = ""
    if spoken_dialogue:
        lines = []
        for item in spoken_dialogue:
            speaker = str(item.get("speaker_id") or item.get("speaker") or "character").strip()
            line = str(item.get("text") or item.get("line") or "").strip()
            tone = str(item.get("delivery_style") or item.get("tone") or "neutral").strip()
            if not line:
                continue
            has_zh = bool(re.search(r'[\u4e00-\u9fff]', line))
            has_en = bool(re.search(r'[a-zA-Z]{3,}', line))
            if has_zh and not has_en:
                actual_lang = "Chinese"
            elif has_en and not has_zh:
                actual_lang = "English"
            else:
                actual_lang = voice_language
            lines.append(
                f"[SPOKEN DIALOGUE {item['start_seconds']:.3f}-{item['end_seconds']:.3f}s] "
                f"{speaker} speaks in {actual_lang} with {tone} tone, says verbatim: \"{line}\". "
                "Generate natural voice matching this speaker's timbre and the described tone."
            )
        if lines:
            dialogue_block = " ".join(lines)
    elif legacy_dialogue:
        dialogue, kind = _normalize_dialogue(legacy_dialogue)
        if kind == "dialogue" and dialogue:
            who = legacy_speaker or "the character"
            dialogue_block = (
                f'[DIALOGUE] {who} speaks aloud in {voice_language}, says verbatim: "{dialogue}". '
                f"Generate natural {voice_language} voice with matching emotion."
            )
        elif kind == "narration" and dialogue:
            dialogue_block = (
                f'[DIALOGUE] Narrator voiceover in {voice_language}, says verbatim: "{dialogue}". '
                f"Generate warm {voice_language} narrator voice."
            )

    if not dialogue_block:
        dialogue_block = "[DIALOGUE] No spoken dialogue this shot. Only ambient sound and music."

    # Model-rendered glyphs are non-deterministic and must never be used for
    # subtitles or contractual screen copy.  The timing contract still keeps
    # on_screen_text for deterministic post-production overlays.
    text_block = (
        "[TEXT POLICY] Do not render subtitles, captions, speech bubbles, "
        "titles, signs, logos, watermarks, letters, numbers, or random glyphs. "
        "All approved text is added deterministically in post-production."
    )

    # ---- 10. EMOTION ------------------------------------------------------
    emotion_block = f"[EMOTION] {emotion}." if emotion else "[EMOTION] Natural cinematic emotion."

    # ---- 11. AUDIO --------------------------------------------------------
    audio_block = (
        f"[AUDIO] Background music: {bgm}. Ambient sound: {ambience}. "
        f"Transition to next shot: {transition}."
    )
    if audio_cues:
        audio_block += " " + " ".join(
            f"[AUDIO CUE {cue['start_seconds']:.3f}-{cue['end_seconds']:.3f}s] "
            f"{cue.get('prompt') or cue.get('text') or cue.get('cue_type') or 'sound effect'}."
            for cue in audio_cues
        )

    # ---- 12. TECHNICAL FOOTER --------------------------------------------
    technical_footer = (
        "[TECHNICAL] "
        f"Target duration ~{duration:.0f}s ({timing['frame_count']} frames on H3's 24fps grid). {aspect_ratio} frame, 24fps, 0.6MP. "
        "Generate native stereo audio (voice + music + ambience + SFX). "
        "Use only the exact <Picture N> roles supplied in the attached REFERENCE MAP; internal ref_image socket names are not model labels."
    )

    parts = [
        style_header,
        char_anchor,
        scene_anchor,
        continuity_rules,
        camera_block,
        timeline_block,
        dialogue_block,
        text_block,
        emotion_block,
        audio_block,
        technical_footer,
    ]
    parts = [p for p in parts if p]  # drop empties
    return "\n\n".join(parts)


# =============================================================================
# Comic-book style prompt builder (matches video_minimax_h3_r2v_sage_lora.json)
# =============================================================================
# Triggered when panel["prompt_mode"] == "comic" (or style starts with "comic")
# Format mirrors the official Sage+LoRA workflow prompt the user attached:
#   [STYLE HEADER]
#   [0-4.5s CUT1 <name>] EXTREME/POWERFUL/AGGRESSIVE shot description
#   [4.5-5.2s TRANSITION <name>] transition description
#   [5.2-10s CUT2 <name>] second shot description
#   [DIALOGUE] word-by-word timed text bubbles
#   [SFX] onomatopoeia (SWOOSH! / RAAAAWR!!!)
#   [CAMERA] / [EMOTION] / [AUDIO] / [TECHNICAL]
#   --ar 16:9 --duration 10 --style raw
# =============================================================================

COMIC_DEFAULT_STYLE_HEADER = (
    # Dean 2026-08-09: 去掉"暗霓虹雨夜城市"硬编码，改为通用描述
    # 让 H3 从 scene_description 推断环境，而不是强制拉到"雨夜城市"
    "10-second dynamic comic-book animated short, bold hand-inked comic-book art, "
    "thick uneven black ink outlines, halftone print grain, "
    "lots of ink splatter, radiating speed-line effects, punchy pop-comic aesthetic."
    # 注意：去掉了 "limited red-and-blue-black color palette, dark neon-lit rainy night city, wet reflective rooftop concrete"
    # 这些会强制把所有场景拉到"雨夜城市"，导致背景不匹配
)


def _build_comic_style_prompt(panel: dict, character_desc: str = "") -> str:
    """Build a comic-book style H3 prompt (matches the user's attached workflow).

    Panel schema for comic mode (all optional, sensible defaults applied):
      style_header:     e.g. "10-second dynamic comic-book animated short, ..."
                        (default = COMIC_DEFAULT_STYLE_HEADER)
      aspect_ratio:     e.g. "16:9" / "9:16" (default "9:16")
      style_tag:        e.g. "raw" / "comic" / "cinematic" (default "raw")
      scene_description: 环境/时间/光线/背景描述（>= 30 词）
      cuts:             list of {time_range, name, shot_description, intensity}
                        e.g. [{"time_range": "0-4.5s", "name": "CUT1 Top-down rooftop shot",
                               "shot_description": "Extreme top-down...", "intensity": "EXTREME"}]
      transitions:      list of {time_range, name, transition_description}
      dialogue_bubbles: list of {time_range, speaker, text, position}
                        e.g. word-by-word timed comic title text synced to voice
      sfx:              list of onomatopoeia strings, e.g. ["SWOOSH!", "RAAAAWR!!!"]
      camera_movement:  overall camera direction
      emotion:          one-word emotional anchor
      background_music: key into MUSIC_PRESETS
      ambience:         key into AMBIENCE_PRESETS
      voice_language:   e.g. "Chinese" / "English"
      duration_seconds: target duration (default 10)
    """
    voice_language = panel.get("voice_language", "Chinese").strip()
    background_music_key = panel.get("background_music", "soft_piano").strip()
    ambience_key = panel.get("ambience", "office_quiet").strip()
    bgm = MUSIC_PRESETS.get(background_music_key, background_music_key)
    ambience = AMBIENCE_PRESETS.get(ambience_key, ambience_key)

    style_header = (panel.get("style_header") or COMIC_DEFAULT_STYLE_HEADER).strip()
    aspect = (panel.get("aspect_ratio") or "16:9").strip()  # 改为16:9横屏，适合多人场景
    style_tag = (panel.get("style_tag") or "raw").strip()

    # 新增：scene_description（锁背景不漂移）
    scene_description = (panel.get("scene_description") or "").strip()

    cuts = panel.get("cuts", []) or []
    transitions = panel.get("transitions", []) or []
    duration = float(panel.get("duration_seconds") or 10.0)
    timing = build_timing_contract(panel, duration)
    spoken_dialogue = timing["spoken_dialogue"]
    on_screen_text = timing["on_screen_text"]
    sfx = panel.get("sfx", []) or []
    camera_movement = (panel.get("camera_movement") or "").strip()
    emotion = (panel.get("emotion") or "").strip()

    sections = []

    # 1. STYLE HEADER (verbatim from user prompt)
    sections.append("[STYLE HEADER]")
    sections.append(style_header)

    # 2. CHARACTER ANCHOR (when supplied) -- locks face/wardrobe for comic consistency
    if character_desc:
        sections.append(
            "[CHARACTER ANCHOR -- DO NOT CHANGE] "
            + character_desc
            + " Lock face, hair, body proportions, wardrobe colors, distinguishing features, accessories, and age verbatim across every cut. Do not re-imagine character design between shots."
        )

    # 3. SCENE ANCHOR (新增：锁背景不漂移)
    if scene_description:
        sections.append(
            "[SCENE -- KEEP CONSISTENT] "
            + scene_description
            + " This is the SAME location as previous shots -- do not change furniture, walls, lighting fixtures, props, color palette, or weather. Maintain the same environment throughout the entire clip."
        )

    # 4. CONTINUITY RULES (新增：防漂移六规则)
    sections.append(
        "[CONTINUITY -- DO NOT VIOLATE] "
        "(a) On-screen cast count is exactly as described -- do not add bystanders or extras. "
        "(b) Wardrobe, hair, makeup, accessories are PERMANENT for this character across every shot of the episode. "
        "(c) Do NOT change room layout, furniture position, or background props between shots. "
        "(d) Do NOT change lighting direction or color temperature between shots in the same scene. "
        "(e) Maintain the SAME film grain, color grading, lens focal length, and depth of field as the reference images. "
        "(f) Use the SAME character design as the reference images -- do not re-imagine or alter the character's appearance."
    )

    # 5. CUTS -- each rendered as [time_range CUT name] INTENSITY shot_description
    if cuts:
        for cut in cuts:
            t_range = (cut.get("time_range") or "").strip()
            name = (cut.get("name") or "").strip()
            shot = (cut.get("shot_description") or "").strip()
            intensity = (cut.get("intensity") or "").strip().upper()
            intensity_prefix = (intensity + " ") if intensity else ""
            header = f"[{t_range} {name}]".strip()
            sections.append(header)
            if shot:
                sections.append(intensity_prefix + shot)
    else:
        # Fallback: use a single implicit cut from scene_description
        scene_description = (panel.get("scene_description") or "").strip()
        if scene_description:
            sections.append(f"[0-{duration:.0f}s CUT1 Main shot]")
            sections.append("EXTREME " + scene_description)

    # 4. TRANSITIONS
    for tr in transitions:
        t_range = (tr.get("time_range") or "").strip()
        name = (tr.get("name") or "").strip()
        desc = (tr.get("transition_description") or tr.get("description") or "").strip()
        header = f"[{t_range} TRANSITION {name}]".strip()
        sections.append(header)
        if desc:
            sections.append(desc)

    # Spoken lines remain an audio contract. Graphic text is never passed to
    # the video model; it is rendered deterministically in post-production.
    if spoken_dialogue:
        spoken_lines = []
        for cue in spoken_dialogue:
            text = str(cue.get("text") or cue.get("line") or "").strip()
            if not text:
                continue
            spoken_lines.append(
                f"  - [{cue['start_seconds']:.3f}-{cue['end_seconds']:.3f}s] "
                f"speaker_id={cue.get('speaker_id', 'UNASSIGNED')} says exactly: \"{text}\""
            )
        if spoken_lines:
            sections.append("[SPOKEN DIALOGUE -- voice only; obey the 24fps cue boundaries]")
            sections.append("\n".join(spoken_lines))

    sections.append(
        "[TEXT POLICY] Do not render subtitles, captions, speech bubbles, "
        "onomatopoeia, titles, signs, logos, watermarks, letters, numbers, "
        "or random glyphs. Approved text and SFX lettering are added in post-production."
    )

    # 7. CAMERA (overall)
    if camera_movement:
        sections.append(f"[CAMERA] {camera_movement}.")

    # 8. EMOTION
    if emotion:
        sections.append(f"[EMOTION] {emotion}.")

    # 9. AUDIO -- music + ambience (native stereo)
    sections.append(
        f"[AUDIO] Background music: {bgm}. Ambient sound: {ambience}. "
        f"Voice language: {voice_language}. Generate native stereo audio with music, ambience, and SFX; "
        "generate voice only for entries in SPOKEN DIALOGUE."
    )
    audio_cues = timing["audio_cues"]
    if audio_cues:
        sections.append("[AUDIO SCHEDULE -- native joint AV clock; obey the 24fps cue boundaries]")
        sections.append("\n".join(
            f"  - [{cue['start_seconds']:.3f}-{cue['end_seconds']:.3f}s] "
            f"{cue.get('prompt') or cue.get('text') or cue.get('cue_type') or 'sound effect'}"
            for cue in audio_cues
        ))

    # 10. TECHNICAL + technical footer (--ar/--duration/--style raw)
    sections.append(
        "[TECHNICAL] "
        f"Target duration {timing['actual_duration_seconds']:.3f}s on a {timing['frame_count']}-frame/24fps grid. Aspect ratio {aspect}. "
        "Use only the exact <Picture N> roles supplied in the attached REFERENCE MAP. "
        "Sampling steps and the synchronized video/audio clocks are controlled by the graph, not by prompt text."
    )

    # Compose with blank lines, then trailing technical flags footer
    body = "\n\n".join(sections)
    footer = f"\n\n--ar {aspect} --duration {duration:.0f} --style {style_tag}"
    return body + footer


def build_panel_prompt(
    panel: dict,
    character_desc: str = "",
    reference_bindings: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Compile the sole runtime prompt consumed by H3.

    The older verbose cinematic/comic builders remain import-compatible for
    offline comparison, but production never concatenates their output with the
    prompt package or whole story/scene contracts.
    """
    return compile_h3_runtime_prompt(panel, character_desc, reference_bindings)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_panel_name(panel: dict[str, Any], panel_index: int) -> str:
    raw = str(panel.get("name") or f"panel_{panel_index:03d}")
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return safe[:96] or f"panel_{panel_index:03d}"


def _copy_video_atomic(source: Path, destination: Path) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination:
        return destination
    temporary = destination.with_suffix(".partial" + destination.suffix)
    shutil.copy2(source, temporary)
    if temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copied render is empty: {source}")
    temporary.replace(destination)
    return destination


def _stage_reference(role: str, source: Path) -> tuple[str, dict[str, Any]]:
    source = Path(source).resolve()
    staged = _copy_image_to_comfy(source)
    return staged, {
        "role": role,
        "source_kind": _reference_source_kind(source),
        "source_path": str(source),
        "staged_name": staged,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _reference_source_kind(source: Path) -> str:
    """Classify staged project references without changing their graph slot."""
    path = Path(source)
    lowered_parts = {part.casefold() for part in path.parts}
    stem = path.stem.casefold()
    if "scenerefs" in lowered_parts or stem.startswith("scene_"):
        return "scene"
    return "character"


def select_h3_reference_sources(
    *,
    first_frame: Path | None,
    last_frame: Path | None,
    character_anchor: Path | None,
    extra_references: Optional[list[Path]] = None,
    composition_anchor_first: bool = False,
) -> dict[str, Any]:
    """Select the model-visible H3 references without staging or I/O writes.

    Strict continuity shots can use the predecessor tail/first frame plus an
    optional approved last frame as the sole group-composition authority.
    Individual portraits and the empty
    scene reference are deliberately suppressed in that mode because H3 may
    interpret them as alternate shots.  Standard/chain-opening calls preserve
    the existing character-plus-scene reference behaviour.

    The same source file may remain in both first and last roles: the two graph
    slots carry different temporal semantics even when their bytes match.
    """
    first = Path(first_frame).resolve() if first_frame else None
    last = Path(last_frame).resolve() if last_frame else None
    anchor = Path(character_anchor).resolve() if character_anchor else None
    extras = [Path(path).resolve() for path in (extra_references or [])]

    def usable(path: Path | None) -> bool:
        return bool(path and path.is_file() and path.stat().st_size > 0)

    use_composition_anchors = bool(composition_anchor_first and usable(first))
    if use_composition_anchors:
        suppressed = [path for path in [anchor, *extras] if path]
        explicit_last = last if usable(last) else None
        synthetic_last_from_first = explicit_last is None
        return {
            "policy": "composition_anchor_first",
            "first_frame": first,
            # H3 is materially more stable when both temporal image slots are
            # bound.  A strict continuation without an approved final frame
            # therefore reuses the predecessor group anchor as a synthetic
            # final authority while preserving distinct opening/final roles.
            "last_frame": first if synthetic_last_from_first else explicit_last,
            "synthetic_last_from_first": synthetic_last_from_first,
            "character_anchor": None,
            "extra_references": [],
            "suppressed_references": suppressed,
        }
    return {
        "policy": "standard",
        "first_frame": first,
        "last_frame": last,
        "synthetic_last_from_first": False,
        "character_anchor": anchor,
        "extra_references": extras,
        "suppressed_references": [],
    }


def submit_render_job(
    panel: dict[str, Any],
    output_path: Path,
    *,
    ep_id: str,
    panel_index: int = 1,
    job_id: str | None = None,
    character_desc: str = "",
    char_refs: Optional[list[Path]] = None,
    seed: Optional[int] = None,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    progress_cb: Optional[Callable] = None,
    first_frame: Optional[Path] = None,
    last_frame: Optional[Path] = None,
    character_anchor: Optional[Path] = None,
    character_anchor_source_id: str | None = None,
    extra_reference_source_ids: Optional[list[str | None]] = None,
    use_lora: bool = True,
    lora_strength: float = 1.0,
    lora_name: str | None = None,
    turbo_steps: int = H3_INFERENCE_STEPS,
    aspect_ratio: str | None = None,
    reference_fidelity: str = "fast",
    sage_attention: str = SAGE_ATTENTION_MODE,
    composition_anchor_first: bool = False,
    megapixels: float = DEFAULT_MEGAPIXELS,
    render_profile: str = "production",
    production_strategy: str = "direct_production",
    delivery_eligible: bool = True,
    store: RenderJobStore | None = None,
    api_func: Callable[[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Stage references, snapshot the exact graph, submit once, and return a job."""
    store = store or default_store()
    api_func = api_func or _api
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    panel_name = _safe_panel_name(panel, panel_index)
    job_id = job_id or f"{ep_id}:{panel_index:04d}:{panel_name}"
    existing = store.get_job(job_id, ep_id=ep_id)
    if existing and existing["status"] == "succeeded" and existing.get("output_path") \
            and Path(existing["output_path"]).exists():
        return existing
    seed = int(seed if seed is not None else time.time_ns() % (2**32))
    timing = build_timing_contract(panel, duration_seconds)
    actual_duration = timing["actual_duration_seconds"]
    reference_selection = select_h3_reference_sources(
        first_frame=first_frame,
        last_frame=last_frame,
        character_anchor=character_anchor,
        extra_references=char_refs,
        composition_anchor_first=composition_anchor_first,
    )
    first_frame = reference_selection["first_frame"]
    last_frame = reference_selection["last_frame"]
    character_anchor = reference_selection["character_anchor"]
    char_refs = reference_selection["extra_references"]
    reference_policy = str(reference_selection["policy"])
    synthetic_last_from_first = bool(reference_selection["synthetic_last_from_first"])
    composition_anchor_cast_count = (
        len({str(value) for value in panel.get("character_ids") or [] if value})
        if reference_policy == "composition_anchor_first" else None
    )
    references: list[dict[str, Any]] = []

    def stage(role: str, path: Optional[Path], source_id: str | None = None) -> Optional[str]:
        if not path:
            return None
        staged, record = _stage_reference(role, path)
        if source_id:
            record["source_id"] = str(source_id)
        references.append(record)
        return staged

    first_name = stage("first_frame", first_frame)
    last_name = stage("last_frame", last_frame)
    anchor_name = stage("character_anchor", character_anchor, character_anchor_source_id)
    if (
        reference_policy != "composition_anchor_first"
        and not anchor_name
        and panel.get("character_anchor")
    ):
        anchor_name = str(panel["character_anchor"])
        staged_path = COMFY / "input" / anchor_name
        if not staged_path.exists():
            raise FileNotFoundError(f"pre-staged character anchor is missing: {staged_path}")
        references.append({
            "role": "character_anchor", "source_path": str(staged_path),
            "staged_name": anchor_name,
            "sha256": hashlib.sha256(staged_path.read_bytes()).hexdigest(),
            **({"source_id": str(character_anchor_source_id)} if character_anchor_source_id else {}),
        })

    extra_names: list[str] = []
    extra_roles: list[str] = []
    selected_extra_source_ids: list[str | None] = []
    requested_extra_source_ids = list(extra_reference_source_ids or [])
    panel_scene_id = str(panel.get("scene_id") or "").strip()
    seen = {name for name in (first_name, last_name, anchor_name) if name}
    for source_index, path in enumerate(char_refs or []):
        source_id = (
            requested_extra_source_ids[source_index]
            if source_index < len(requested_extra_source_ids) else None
        )
        source_kind = _reference_source_kind(path)
        role = (
            "scene_reference"
            if (source_id and str(source_id) == panel_scene_id) or source_kind == "scene"
            else "character_reference"
        )
        staged, record = _stage_reference(role, path)
        if staged in seen:
            continue
        if len(extra_names) >= MAX_CHAR_REFS:
            if role == "scene_reference":
                raise ValueError(
                    "H3 extra-reference capacity exceeded: "
                    f"{MAX_CHAR_REFS} slots are available after first/last/anchor images; "
                    "submitting this panel would omit its approved scene reference"
                )
            raise ValueError(
                "H3 extra-reference capacity exceeded: "
                f"{MAX_CHAR_REFS} slots are available after first/last/anchor images; "
                "refusing to silently omit an approved reference"
            )
        if source_id:
            record["source_id"] = str(source_id)
        seen.add(staged)
        extra_names.append(staged)
        extra_roles.append(role)
        selected_extra_source_ids.append(source_id)
        references.append(record)

    reference_bindings = build_h3_reference_bindings(
        first_frame_filename=first_name,
        last_frame_filename=last_name,
        character_anchor_filename=anchor_name,
        character_anchor_source_id=character_anchor_source_id,
        extra_reference_filenames=extra_names,
        extra_reference_roles=extra_roles,
        extra_reference_source_ids=selected_extra_source_ids,
    )
    prompt = build_panel_prompt({
        **panel,
        "duration_seconds": duration_seconds,
        "continuity_reference_policy": reference_policy,
    }, character_desc, reference_bindings)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    reference_bundle_payload = [
        {
            "role": record.get("role"),
            "sha256": record.get("sha256"),
            "model_label": reference_bindings[index].get("model_label")
            if index < len(reference_bindings) else None,
        }
        for index, record in enumerate(references)
    ]
    reference_bundle_sha256 = hashlib.sha256(
        json.dumps(
            reference_bundle_payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    selected_aspect = aspect_ratio or str(panel.get("aspect_ratio") or "16:9")
    sampling = build_h3_sampling_contract(
        use_lora=use_lora,
        lora_name=lora_name,
        lora_strength=lora_strength,
        turbo_steps=turbo_steps,
    )
    graph = build_h3_ref2va_graph(
        prompt=prompt,
        seed=seed,
        char_ref_filenames=extra_names,
        duration_seconds=duration_seconds,
        first_frame_filename=first_name,
        last_frame_filename=last_name,
        character_anchor_filename=anchor_name,
        use_lora=use_lora,
        lora_strength=lora_strength,
        lora_name=sampling["lora_name"],
        turbo_steps=sampling["steps"],
        ep_id=ep_id,
        name_prefix=panel_name,
        aspect_ratio=selected_aspect,
        reference_fidelity=reference_fidelity,
        sage_attention=sage_attention,
        extra_reference_roles=extra_roles,
        reference_policy=reference_policy,
        composition_anchor_cast_count=composition_anchor_cast_count,
        megapixels=megapixels,
    )
    graph_path = output.with_suffix(".graph.json")
    timing_path = output.with_suffix(".cues.json")
    action_authority = _compiled_action_authority(panel)
    audio_resolution = resolve_panel_audio(panel)
    reviewer_correction = _reviewer_correction_audit(panel)
    action_contract_audit = (
        {
            key: action_authority[key]
            for key in (
                "catalog_version", "action_code", "spec_sha256",
                "compiled_h3_sha256", "source", "h3_action_en",
            )
        }
        if action_authority else None
    )
    snapshot = {
        "schema_version": 1,
        "ep_id": ep_id,
        "job_id": job_id,
        "panel_index": panel_index,
        "panel_name": panel_name,
        "prompt": prompt,
        "prompt_audit": {
            "prompt_sha256": prompt_sha256,
            "reference_bundle_sha256": reference_bundle_sha256,
            "skill_version": H3_DIRECTOR_SKILL_VERSION,
            "official_prompt_shape": H3_OFFICIAL_PROMPT_SHAPE,
            "runtime_prompt_contract": H3_RUNTIME_PROMPT_CONTRACT,
            "audio_resolution": audio_resolution,
            "reviewer_correction": reviewer_correction or None,
        },
        "graph": graph,
        "reference_images": references,
        "action_contract": action_contract_audit,
        "director_shot_plan": {
            "schema_version": "h3-director-shot-plan/v1",
            "approved_fields": _director_shot_plan(panel),
            "runtime_consumed": [
                "blocking", "screen_direction", "axis", "eyeline",
                "dominant_camera_move", "camera_plan", "first_state", "final_state",
            ],
            "delivery_only": [
                "story_function", "transition", "tr", "sound_bridge",
                "risk", "risk_code", "failure_code", "failure_codes",
                "first_keyframe_strategy", "last_keyframe_strategy", "keyframe_strategy",
            ],
        },
        "settings": {
            "seed": seed, "fps": VIDEO_FPS, "frame_count": timing["frame_count"],
            "requested_duration_seconds": duration_seconds,
            "actual_duration_seconds": actual_duration,
            "aspect_ratio": selected_aspect, "use_lora": use_lora,
            "lora_strength": lora_strength, "reference_fidelity": reference_fidelity,
            "megapixels": float(megapixels),
            "turbo_steps": int(sampling["steps"]),
            "render_profile": str(render_profile),
            "production_strategy": str(production_strategy),
            "delivery_eligible": bool(delivery_eligible),
            "render_profile_contract": H3_RENDER_PROFILE_CONTRACT,
            "director_skill_version": H3_DIRECTOR_SKILL_VERSION,
            "official_prompt_shape": H3_OFFICIAL_PROMPT_SHAPE,
            "prompt_sha256": prompt_sha256,
            "reference_bundle_sha256": reference_bundle_sha256,
            "sage_attention": SAGE_ATTENTION_CHOICES.get(sage_attention, sage_attention),
            "sampling_contract": sampling,
            "reference_bindings": reference_bindings,
            "reference_policy": reference_policy,
            "suppressed_reference_sources": [
                str(path) for path in reference_selection["suppressed_references"]
            ],
            "composition_anchor_cast_count": composition_anchor_cast_count,
            "synthetic_last_from_first": synthetic_last_from_first,
            "runtime_prompt_contract": H3_RUNTIME_PROMPT_CONTRACT,
            "audio_resolution": audio_resolution,
            "reviewer_correction": reviewer_correction or None,
        },
    }
    _write_json_atomic(graph_path, snapshot)
    _write_json_atomic(timing_path, timing)
    # A prepared job already owns the semantic input hash.  Keep it stable
    # across submission: the generated seed and graph snapshot are audit data,
    # not a reason for a later Web refresh to invalidate the resulting clip.
    input_hash = (existing or {}).get("input_hash") or hashlib.sha256(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    prior_metadata = dict((existing or {}).get("metadata") or {})
    store.register_jobs(ep_id, [{
        "job_id": job_id, "panel_index": panel_index, "panel_name": panel_name,
        "status": "queued", "output_path": str(output), "preview_path": str(output),
        "reference_images": references, "graph_path": str(graph_path),
        "timing_path": str(timing_path), "input_hash": input_hash,
        "dialogue_cues": timing["spoken_dialogue"], "audio_cues": timing["audio_cues"],
        "metadata": {
            **prior_metadata,
            "settings": snapshot["settings"],
            "prompt_sha256": prompt_sha256,
            "reference_bundle_sha256": reference_bundle_sha256,
            "director_skill_version": H3_DIRECTOR_SKILL_VERSION,
            "timing_warnings": timing["warnings"],
            "action_contract": action_contract_audit,
        },
    }], prune_missing=False)
    if progress_cb:
        progress_cb("submit", f"提交 {panel_name}: {len(references)} 张参考图，{timing['frame_count']} 帧")
    try:
        response = api_func("/prompt", {"prompt": graph, "client_id": str(uuid.uuid4())})
        prompt_id = response["prompt_id"]
    except Exception as exc:
        store.update_job(job_id, status="failed", error=f"submit failed: {exc}", progress=0.0)
        raise
    return store.update_job(
        job_id, status="submitted", prompt_id=prompt_id, progress=0.05,
        submitted_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), error=None,
    )


def _complete_job_from_history(
    job: dict[str, Any], result: dict[str, Any], store: RenderJobStore,
    probe_func: Callable[[Path], dict[str, Any]], progress_cb: Optional[Callable] = None,
    quality_analyzer: Optional[Callable[..., dict[str, Any]]] = None,
    quality_runner: Callable[..., Any] = subprocess.run,
    edit_selector: Optional[Callable[..., dict[str, Any]]] = None,
) -> Path:
    status = result.get("status", {})
    if status.get("status_str") == "error":
        message = f"ComfyUI error: {status.get('messages', status)}"
        store.update_job(job["job_id"], status="failed", error=message, progress=0.0)
        raise RuntimeError(message)
    source = _find_video_in_outputs(result.get("outputs", {}))
    if not source:
        raise FileNotFoundError("ComfyUI history completed without a video output")
    output = _copy_video_atomic(source, Path(job["output_path"]))
    probe = probe_func(output)
    if not probe.get("audio"):
        raise ValueError(f"H3 output has no audio stream: {output}")
    from video_quality import analyze_video, evaluate_content, select_edit_window, validate_edit_selection
    analyzer = quality_analyzer or analyze_video
    analysis = analyzer(output, ffmpeg=ffmpeg_executable(), runner=quality_runner)
    prior = []
    for other in store.list_jobs(str(job["ep_id"])):
        if str(other.get("job_id")) == str(job["job_id"]):
            continue
        other_qa = ((other.get("metadata") or {}).get("content_qa") or {})
        other_analysis = other_qa.get("analysis") or {}
        if other.get("status") == "succeeded" and other.get("output_path"):
            if not other_analysis:
                other_path = Path(str(other["output_path"]))
                if not other_path.is_file():
                    raise RuntimeError(f"prior succeeded clip is missing for content QA: {other['job_id']}")
                other_analysis = analyzer(
                    other_path, ffmpeg=ffmpeg_executable(), runner=quality_runner,
                )
            prior.append((str(other["job_id"]), other_analysis))
    content_qa = evaluate_content(analysis, (), require_motion=True)
    content_qa.update({
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": "pre_success", "render_mode": "h3",
    })
    qa_metadata = dict(job.get("metadata") or {})
    qa_metadata["content_qa"] = content_qa
    if not content_qa.get("passed"):
        qa_metadata["editorial_review"] = {
            "status": "blocked", "reason": "content QA failed",
        }
        qa_metadata["release"] = {"status": "revoked", "reason": "content QA failed"}
        store.update_job(job["job_id"], metadata=qa_metadata)
        raise RuntimeError(
            "H3 content QA failed: "
            + ",".join(str(reason) for reason in content_qa.get("reasons") or ["unknown"])
        )
    artifact_path = output.with_suffix(".artifact.json")
    artifact_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    job_inputs = (job.get("metadata") or {}).get("inputs") or {}
    shot_plan = job_inputs.get("shot_plan") or {}
    settings = job_inputs.get("settings") or (job.get("metadata") or {}).get("settings") or {}
    requested_edit_duration = shot_plan.get("edit_duration_seconds", settings.get("edit_duration_seconds"))
    edit_selection = None
    if requested_edit_duration is not None:
        selector = edit_selector or select_edit_window
        protected_ranges = []
        for cue in job.get("dialogue_cues") or []:
            try:
                start = float(cue.get("start_seconds", cue.get("start_s")))
                end = float(cue.get("end_seconds", cue.get("end_s")))
            except (TypeError, ValueError):
                continue
            if end > start:
                protected_ranges.append((start, end))
        try:
            edit_selection = selector(
                analysis,
                source_duration_seconds=float(probe.get("duration_seconds") or 0),
                requested_duration_seconds=float(requested_edit_duration),
                source_artifact_sha256=artifact_sha256,
                edit_hint=shot_plan.get("edit_hint") if isinstance(shot_plan, dict) else None,
                protected_ranges=protected_ranges,
            )
            selection_check = validate_edit_selection(
                edit_selection, source_artifact_sha256=artifact_sha256,
                requested_duration_seconds=float(requested_edit_duration),
                source_duration_seconds=float(probe.get("duration_seconds") or 0),
            )
            if not selection_check["valid"]:
                raise RuntimeError("invalid selection: " + ",".join(selection_check["errors"]))
        except Exception as exc:
            failed_metadata = dict(job.get("metadata") or {})
            failed_metadata["content_qa"] = content_qa
            failed_metadata["edit_selection"] = {
                "status": "deadletter", "reason": str(exc),
                "source_artifact_sha256": artifact_sha256,
                "requested_duration_seconds": float(requested_edit_duration),
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            failed_metadata["editorial_review"] = {
                "status": "blocked", "reason": "edit selection failed",
            }
            failed_metadata["release"] = {"status": "revoked", "reason": "edit selection failed"}
            store.update_job(job["job_id"], metadata=failed_metadata)
            raise RuntimeError(f"H3 edit selection deadletter: {exc}") from exc
        selected_analysis = analyzer(
            output, ffmpeg=ffmpeg_executable(), runner=quality_runner,
            start_seconds=float(edit_selection["in_seconds"]),
            duration_seconds=float(edit_selection["duration_seconds"]),
        )
        selected_qa = evaluate_content(selected_analysis, prior, require_motion=True)
        content_qa = {
            **selected_qa,
            "source_analysis": analysis,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": "pre_success_selected_window", "render_mode": "h3",
            "edit_selection_sha256": edit_selection["selection_sha256"],
        }
        qa_metadata["content_qa"] = content_qa
        if not content_qa.get("passed"):
            failed_metadata = dict(job.get("metadata") or {})
            failed_metadata.update({
                "content_qa": content_qa, "edit_selection": edit_selection,
                "editorial_review": {"status": "blocked", "reason": "selected-window QA failed"},
                "release": {"status": "revoked", "reason": "selected-window QA failed"},
            })
            store.update_job(job["job_id"], metadata=failed_metadata)
            raise RuntimeError(
                "H3 selected-window content QA failed: "
                + ",".join(str(reason) for reason in content_qa.get("reasons") or ["unknown"])
            )
    artifact = {
        "schema_version": 1, "job_id": job["job_id"], "prompt_id": job["prompt_id"],
        "source_path": str(source), "output_path": str(output), "probe": probe,
        "artifact_sha256": artifact_sha256,
        "content_qa": content_qa,
        "edit_selection": edit_selection,
        "reference_images": job.get("reference_images", []),
        "graph_path": job.get("graph_path"), "timing_path": job.get("timing_path"),
    }
    _write_json_atomic(artifact_path, artifact)
    metadata = qa_metadata
    metadata["artifact_sha256"] = artifact_sha256
    metadata["content_qa"] = content_qa
    if edit_selection:
        metadata["edit_selection"] = edit_selection
    metadata["editorial_review"] = {"status": "pending", "reason": "awaiting human review"}
    metadata["release"] = {"status": "pending", "reason": "episode release not approved"}
    store.update_job(
        job["job_id"], status="succeeded", progress=1.0,
        comfy_output_path=str(source), output_path=str(output), preview_path=str(output),
        probe=probe, error=None, metadata=metadata,
    )
    if progress_cb:
        progress_cb("done", f"已验证成片: {output.name} ({probe['duration_seconds']:.3f}s)")
    return output


def wait_render_job(
    job_id: str,
    *,
    store: RenderJobStore | None = None,
    timeout: float = 2400.0,
    poll_interval: float = 5.0,
    progress_cb: Optional[Callable] = None,
    api_func: Callable[[str, Any], dict[str, Any]] | None = None,
    probe_func: Callable[[Path], dict[str, Any]] | None = None,
    quality_analyzer: Optional[Callable[..., dict[str, Any]]] = None,
    quality_runner: Callable[..., Any] = subprocess.run,
    edit_selector: Optional[Callable[..., dict[str, Any]]] = None,
) -> Path:
    """Wait for one already-submitted job with a finite deadline."""
    from video_delivery import probe_media
    store = store or default_store()
    api_func = api_func or _api
    probe_func = probe_func or probe_media
    job = store.get_job(job_id)
    if not job:
        raise KeyError(job_id)
    if job["status"] == "succeeded" and job.get("output_path") and Path(job["output_path"]).exists():
        return Path(job["output_path"])
    if not job.get("prompt_id"):
        raise ValueError(f"job has not been submitted: {job_id}")
    expected_prompt_id = str(job["prompt_id"])

    def current_waitable_job() -> dict[str, Any]:
        current = store.get_job(job_id)
        if not current:
            raise RuntimeError(f"render job disappeared while waiting: {job_id}")
        if current["status"] == "cancelled":
            raise RuntimeError(f"render job was cancelled: {job_id}")
        if (
            current["status"] not in {"submitted", "running"}
            or str(current.get("prompt_id") or "") != expected_prompt_id
        ):
            raise RuntimeError(
                f"render job was invalidated while waiting: {job_id} "
                f"(status={current['status']}, prompt_id={current.get('prompt_id')})"
            )
        return current

    started = time.monotonic()
    store.update_job(job_id, status="running", progress=max(0.05, job["progress"]))
    while True:
        job = current_waitable_job()
        history = api_func(f"/history/{expected_prompt_id}", None)
        # Re-read after every potentially blocking API call. A concurrent QA
        # rejection may have cancelled and invalidated the prompt meanwhile;
        # stale history must never resurrect it or write ``running`` again.
        job = current_waitable_job()
        if expected_prompt_id in history:
            try:
                return _complete_job_from_history(
                    job, history[expected_prompt_id], store, probe_func, progress_cb,
                    quality_analyzer=quality_analyzer, quality_runner=quality_runner,
                    edit_selector=edit_selector,
                )
            except Exception as exc:
                store.update_job(job_id, status="failed", error=str(exc), progress=0.0)
                raise
        try:
            remote_queue = comfy_queue_state(expected_prompt_id, api_func("/queue", None))
        except Exception as exc:
            remote_queue = {
                "state": "unknown", "position": None,
                "pending_total": None, "error": str(exc)[:240],
            }
        remote_queue.update({
            "prompt_id": expected_prompt_id,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        job = current_waitable_job()
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            message = f"render timed out after {timeout:.1f}s; prompt_id retained for recovery"
            store.update_job(job_id, status="failed", error=message, progress=min(job["progress"], 0.95))
            raise TimeoutError(message)
        progress = min(0.95, max(job["progress"], 0.05 + 0.85 * elapsed / max(timeout, 1.0)))
        current_waitable_job()
        wait_metadata = dict(job.get("metadata") or {})
        wait_metadata["remote_queue"] = remote_queue
        store.update_job(
            job_id, status="running", progress=progress, metadata=wait_metadata,
        )
        if progress_cb:
            if remote_queue["state"] == "pending":
                ahead = max(0, int(remote_queue.get("position") or 1) - 1)
                message = f"{job['panel_name']} ComfyUI 排队中（前方 {ahead} 个，{elapsed:.0f}s）"
            elif remote_queue["state"] == "running":
                message = f"{job['panel_name']} ComfyUI GPU 执行中（{elapsed:.0f}s）"
            elif remote_queue["state"] == "absent_or_history_pending":
                message = f"{job['panel_name']} 等待 ComfyUI history / 写盘确认（{elapsed:.0f}s）"
            else:
                message = f"{job['panel_name']} 无法读取远端队列，保持防重提交等待（{elapsed:.0f}s）"
            progress_cb("running", message)
        time.sleep(max(0.01, poll_interval))


def recover_render_job(
    job_id: str, *, store: RenderJobStore | None = None,
    api_func: Callable[[str, Any], dict[str, Any]] | None = None,
    probe_func: Callable[[Path], dict[str, Any]] | None = None,
    quality_analyzer: Optional[Callable[..., dict[str, Any]]] = None,
    quality_runner: Callable[..., Any] = subprocess.run,
    edit_selector: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Reconcile one persisted prompt_id with ComfyUI history without resubmitting."""
    from video_delivery import probe_media
    store = store or default_store()
    api_func = api_func or _api
    probe_func = probe_func or probe_media
    job = store.get_job(job_id)
    if not job or not job.get("prompt_id"):
        return job or {}
    history = api_func(f"/history/{job['prompt_id']}", None)
    if job["prompt_id"] not in history:
        return job
    try:
        _complete_job_from_history(
            job, history[job["prompt_id"]], store, probe_func,
            quality_analyzer=quality_analyzer, quality_runner=quality_runner,
            edit_selector=edit_selector,
        )
    except Exception:
        pass
    return store.get_job(job_id) or {}


def comfy_queue_state(prompt_id: str, queue: Any) -> dict[str, Any]:
    """Describe one prompt's documented Comfy queue state without mutating it."""
    expected = str(prompt_id or "").strip()
    if not isinstance(queue, dict):
        return {
            "state": "unknown", "position": None,
            "pending_total": None, "error": "queue response is not an object",
        }
    for state, bucket in (("running", "queue_running"), ("pending", "queue_pending")):
        rows = queue.get(bucket)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows, 1):
            value: Any = None
            if isinstance(row, dict):
                value = row.get("prompt_id") or row.get("id")
            elif isinstance(row, (list, tuple)) and len(row) > 1:
                value = row[1]
            if str(value or "").strip() == expected:
                return {
                    "state": state,
                    "position": 0 if state == "running" else index,
                    "pending_total": len(queue.get("queue_pending") or []),
                    "error": None,
                }
    return {
        "state": "absent_or_history_pending", "position": None,
        "pending_total": len(queue.get("queue_pending") or []), "error": None,
    }


def _queued_prompt_ids(queue: Any) -> set[str]:
    """Extract prompt IDs from the documented Comfy queue response shape."""
    if not isinstance(queue, dict):
        return set()
    prompt_ids: set[str] = set()
    for bucket in ("queue_running", "queue_pending"):
        rows = queue.get(bucket)
        if not isinstance(rows, list):
            continue
        for row in rows:
            value: Any = None
            if isinstance(row, dict):
                value = row.get("prompt_id") or row.get("id")
            elif isinstance(row, (list, tuple)) and len(row) > 1:
                # Native Comfy rows are [queue_number, prompt_id, graph, ...].
                value = row[1]
            if value is not None and str(value).strip():
                prompt_ids.add(str(value).strip())
    return prompt_ids


def _reconciliation_disposition(job: dict[str, Any], prompt_id: str) -> str:
    if job.get("status") == "succeeded":
        return "recovered"
    authorization = (job.get("metadata") or {}).get("remote_retry_authorization") or {}
    if (
        job.get("status") == "failed"
        and authorization.get("disposition") == "safe_to_retry"
        and str(authorization.get("prompt_id") or "") == prompt_id
    ):
        return "safe_to_retry"
    if job.get("status") in {"submitted", "running"} and str(job.get("prompt_id") or "") == prompt_id:
        return "remote_active"
    return "submission_unknown"


def reconcile_render_job(
    job_id: str,
    *,
    store: RenderJobStore | None = None,
    api_func: Callable[[str, Any], dict[str, Any]] | None = None,
    probe_func: Callable[[Path], dict[str, Any]] | None = None,
    quality_analyzer: Optional[Callable[..., dict[str, Any]]] = None,
    quality_runner: Callable[..., Any] = subprocess.run,
    edit_selector: Optional[Callable[..., dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Fail-closed reconciliation for a failed job that still owns a prompt.

    One caller atomically claims ``failed -> submitted``.  A second caller
    observes that durable claim and cannot query/finalize/authorize the same
    failure concurrently.  Only an explicit Comfy history ``error`` creates a
    prompt-bound retry authorization; missing history, an empty queue, malformed
    responses and network failures remain ambiguous and keep the prompt.
    """
    from video_delivery import probe_media

    store = store or default_store()
    api_func = api_func or _api
    probe_func = probe_func or probe_media
    original = store.get_job(job_id)
    if not original:
        raise KeyError(job_id)
    prompt_id = str(original.get("prompt_id") or "").strip()
    if not prompt_id:
        return {
            "disposition": "submission_unknown", "reason": "persisted_prompt_id_missing",
            "prompt_id": None, "job": original,
        }
    if original.get("status") != "failed":
        return {
            "disposition": _reconciliation_disposition(original, prompt_id),
            "reason": "job_not_claimable_for_reconciliation",
            "prompt_id": prompt_id, "job": original,
        }

    claimed = store.compare_and_update_job(
        job_id,
        expected={"status": "failed", "prompt_id": prompt_id},
        status="submitted", completed_at=None,
    )
    if claimed is None:
        current = store.get_job(job_id) or {}
        return {
            "disposition": _reconciliation_disposition(current, prompt_id),
            "reason": "reconciliation_already_claimed_or_job_changed",
            "prompt_id": prompt_id, "job": current,
        }

    checked_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    history_error: Exception | None = None
    history: dict[str, Any] = {}
    try:
        response = api_func(f"/history/{prompt_id}", None)
        if not isinstance(response, dict):
            raise ValueError("Comfy history response is not an object")
        history = response
    except Exception as exc:
        history_error = exc

    entry = history.get(prompt_id) if isinstance(history, dict) else None
    if isinstance(entry, dict):
        remote_status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
        if str(remote_status.get("status_str") or "").lower() == "error":
            metadata = dict(claimed.get("metadata") or {})
            authorization = {
                "schema": "comfy-retry-authorization/v1",
                "disposition": "safe_to_retry", "prompt_id": prompt_id,
                "checked_at": checked_at, "remote_status": "error",
                "remote_messages": remote_status.get("messages") or [],
            }
            metadata["remote_retry_authorization"] = authorization
            failed = store.compare_and_update_job(
                job_id,
                expected={"status": "submitted", "prompt_id": prompt_id},
                status="failed", progress=0.0,
                error=f"ComfyUI error: {remote_status.get('messages', remote_status)}",
                metadata=metadata,
            )
            current = failed or store.get_job(job_id) or {}
            return {
                "disposition": _reconciliation_disposition(current, prompt_id),
                "reason": "explicit_comfy_history_error",
                "prompt_id": prompt_id, "job": current,
            }

        try:
            _complete_job_from_history(
                claimed, entry, store, probe_func,
                quality_analyzer=quality_analyzer, quality_runner=quality_runner,
                edit_selector=edit_selector,
            )
        except Exception as exc:
            current = store.get_job(job_id) or {}
            content_qa = (current.get("metadata") or {}).get("content_qa") or {}
            selection = (current.get("metadata") or {}).get("edit_selection") or {}
            if content_qa.get("passed") is False or selection.get("status") == "deadletter":
                failed = store.compare_and_update_job(
                    job_id,
                    expected={"status": "submitted", "prompt_id": prompt_id},
                    status="failed", error=str(exc),
                )
                current = failed or store.get_job(job_id) or current
                return {
                    "disposition": "content_failed", "reason": str(exc),
                    "prompt_id": prompt_id, "job": current,
                }
            # Remote completion is known, but local copy/probe/finalization is
            # not safe to turn into another billable generation.
            return {
                "disposition": "submission_unknown",
                "reason": f"remote_completion_finalize_failed:{type(exc).__name__}:{exc}",
                "prompt_id": prompt_id, "job": current,
            }
        current = store.get_job(job_id) or {}
        return {
            "disposition": "recovered", "reason": "comfy_history_success",
            "prompt_id": prompt_id, "job": current,
        }

    queue_error: Exception | None = None
    queue: dict[str, Any] = {}
    try:
        response = api_func("/queue", None)
        if not isinstance(response, dict):
            raise ValueError("Comfy queue response is not an object")
        queue = response
    except Exception as exc:
        queue_error = exc
    if prompt_id in _queued_prompt_ids(queue):
        return {
            "disposition": "remote_active", "reason": "prompt_present_in_comfy_queue",
            "prompt_id": prompt_id, "job": store.get_job(job_id) or claimed,
        }
    errors = [
        f"history:{type(history_error).__name__}:{history_error}" if history_error else "history:missing",
        f"queue:{type(queue_error).__name__}:{queue_error}" if queue_error else "queue:prompt_absent",
    ]
    return {
        "disposition": "submission_unknown", "reason": ";".join(errors),
        "prompt_id": prompt_id, "job": store.get_job(job_id) or claimed,
    }


def authorize_retry_after_comfy_restart(
    job_id: str,
    *,
    confirmed: bool,
    store: RenderJobStore | None = None,
    api_func: Callable[[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit an operator-confirmed Comfy restart before releasing a lost prompt.

    Comfy history is process-local in some installations.  After a deliberate
    backend restart, a terminal prompt can therefore disappear from both
    history and queue.  Missing history alone remains fail-closed; this escape
    hatch additionally requires an explicit operator confirmation, a reachable
    empty queue, and an absent exact prompt before granting one bounded retry.
    """
    if not confirmed:
        raise RuntimeError("explicit ComfyUI restart confirmation is required")
    store = store or default_store()
    api_func = api_func or _api
    job = store.get_job(job_id)
    if not job:
        raise KeyError(job_id)
    prompt_id = str(job.get("prompt_id") or "").strip()
    if job.get("status") != "failed" or not prompt_id:
        raise RuntimeError("restart recovery requires a failed job with a persisted prompt_id")

    history = api_func(f"/history/{prompt_id}", None)
    if not isinstance(history, dict):
        raise RuntimeError("ComfyUI history response is not an object")
    if prompt_id in history:
        raise RuntimeError("prompt still exists in ComfyUI history; reconcile it instead")
    queue = api_func("/queue", None)
    if not isinstance(queue, dict):
        raise RuntimeError("ComfyUI queue response is not an object")
    if prompt_id in _queued_prompt_ids(queue):
        raise RuntimeError("prompt is still active in the ComfyUI queue")
    if (queue.get("queue_running") or []) or (queue.get("queue_pending") or []):
        raise RuntimeError("ComfyUI queue must be empty for restart recovery")

    checked_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
    authorization = {
        "schema": "comfy-retry-authorization/v1",
        "disposition": "safe_to_retry",
        "prompt_id": prompt_id,
        "checked_at": checked_at,
        "remote_status": "history_lost_after_confirmed_restart",
        "source": "operator_restart_attestation",
    }
    metadata["remote_retry_authorization"] = authorization
    restart_audit = list(metadata.get("comfy_restart_recovery_audit") or [])
    restart_audit.append(authorization)
    metadata["comfy_restart_recovery_audit"] = restart_audit[-20:]
    retry_count = int(job.get("retry_count") or 0)
    max_retries = max(1, int(job.get("max_retries") or 0))
    # Infrastructure failure consumed no usable artifact.  Restore exactly one
    # bounded attempt; retry_job increments it back to max_retries and consumes
    # this prompt-bound authorization.
    renewed_retry_count = min(retry_count, max_retries - 1)
    updated = store.compare_and_update_job(
        job_id,
        expected={"status": "failed", "prompt_id": prompt_id, "retry_count": retry_count},
        retry_count=renewed_retry_count,
        error="ComfyUI restart confirmed; old prompt absent and queue empty; one retry authorized",
        metadata=metadata,
    )
    if updated is None:
        raise RuntimeError(f"job changed concurrently before restart recovery: {job_id}")
    return {
        "disposition": "safe_to_retry",
        "reason": "operator_confirmed_comfy_restart_and_empty_queue",
        "prompt_id": prompt_id,
        "job": updated,
    }


def cancel_render_job(
    job_id: str, *, store: RenderJobStore | None = None,
    api_func: Callable[[str, Any], dict[str, Any]] | None = None,
    interrupt_running: bool = False,
) -> dict[str, Any]:
    """Cancel a queued prompt; global running-queue interruption is opt-in."""
    store = store or default_store()
    api_func = api_func or _api
    job = store.get_job(job_id)
    if not job:
        raise KeyError(job_id)
    prompt_id = job.get("prompt_id")
    if prompt_id:
        api_func("/queue", {"delete": [prompt_id]})
        if interrupt_running:
            api_func("/interrupt", {})
    return store.update_job(job_id, status="cancelled", error="cancelled by user", progress=job["progress"])


def render_panel(
    panel: dict,
    output_path: Path,
    character_desc: str = "",
    char_refs: Optional[list[Path]] = None,
    seed: Optional[int] = None,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    progress_cb: Optional[Callable] = None,
    first_frame: Optional[Path] = None,
    last_frame: Optional[Path] = None,
    character_anchor: Optional[Path] = None,
    use_lora: bool = True,
    lora_strength: float = 1.0,
    lora_name: str | None = None,
    turbo_steps: int = H3_INFERENCE_STEPS,
    ep_id: Optional[str] = None,
    wait_for_completion: bool = True,
    *,
    panel_index: int = 1,
    job_id: str | None = None,
    aspect_ratio: str | None = None,
    reference_fidelity: str = "fast",
    sage_attention: str = SAGE_ATTENTION_MODE,
    store: RenderJobStore | None = None,
    timeout: float = 2400.0,
    poll_interval: float = 5.0,
    api_func: Callable[[str, Any], dict[str, Any]] | None = None,
    probe_func: Callable[[Path], dict[str, Any]] | None = None,
) -> Path | dict[str, Any]:
    """Compatibility wrapper: submit only or submit then wait.

    New code should call :func:`submit_render_job` and :func:`wait_render_job`
    directly.  When ``wait_for_completion`` is false this returns a job dict,
    never a fake output path.
    """
    episode = ep_id or "standalone"
    job = submit_render_job(
        panel, output_path, ep_id=episode, panel_index=panel_index, job_id=job_id,
        character_desc=character_desc, char_refs=char_refs, seed=seed,
        duration_seconds=duration_seconds, progress_cb=progress_cb,
        first_frame=first_frame, last_frame=last_frame, character_anchor=character_anchor,
        use_lora=use_lora, lora_strength=lora_strength,
        lora_name=lora_name, turbo_steps=turbo_steps, aspect_ratio=aspect_ratio,
        reference_fidelity=reference_fidelity, sage_attention=sage_attention,
        store=store, api_func=api_func,
    )
    if not wait_for_completion:
        return job
    return wait_render_job(
        job["job_id"], store=store, timeout=timeout, poll_interval=poll_interval,
        progress_cb=progress_cb, api_func=api_func, probe_func=probe_func,
    )


def sync_videos_from_comfyui(
    ep_id: str,
    episodes_path: Optional[Path] = None,
    progress_cb: Optional[Callable] = None,
) -> int:
    """Sync videos from ComfyUI output directory to project directory.
    
    This is useful when the rendering session disconnected and videos
    were generated but not copied to the project directory.
    
    Args:
        ep_id: Episode ID
        episodes_path: Path to episodes.json (defaults to ROOT/episodes.json)
        progress_cb: Optional callback for progress updates
    
    Returns:
        Number of videos synced
    """
    # Durable recovery is keyed by prompt_id.  Never guess ownership by file
    # modification time, because that can attach another project's render.
    del episodes_path
    store = default_store()
    recovered = 0
    for job in store.list_jobs(ep_id):
        before = job.get("status")
        current = recover_render_job(job["job_id"], store=store)
        if before != "succeeded" and current.get("status") == "succeeded":
            recovered += 1
            if progress_cb:
                progress_cb("sync", f"history 恢复成功: {job['panel_name']}")
    return recovered

    if episodes_path is None:
        episodes_path = ROOT / "episodes.json"
    
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    if ep_id not in episodes:
        raise ValueError(f"Episode {ep_id} not in episodes.json")
    
    ep = episodes[ep_id]
    panels = ep["panels"]
    
    videos_dir = ROOT / f"{ep_id}_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # 只从项目子目录读取视频（不再向后兼容旧格式）
    # 格式：output/video/{ep_id}/MiniMax_H3_*.mp4
    comfy_output_dir = COMFY / "output" / "video" / ep_id
    if not comfy_output_dir.exists():
        if progress_cb:
            progress_cb("sync", f"项目目录不存在: {comfy_output_dir}")
            progress_cb("sync", "请先渲染视频，或检查 ep_id 是否正确")
        return 0
    
    # Get all recent videos from ComfyUI project directory (sorted by modification time)
    comfy_videos = sorted(
        comfy_output_dir.glob("MiniMax_H3_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not comfy_videos:
        if progress_cb:
            progress_cb("sync", "No videos found in ComfyUI output")
        return 0
    
    synced = 0
    total_panels = len(panels)
    
    # Match videos to panels by order (newest videos = last panels)
    # Reverse to get chronological order
    comfy_videos_chrono = list(reversed(comfy_videos))
    
    for i, panel in enumerate(panels):
        if isinstance(panel, dict):
            name = panel["name"]
        else:
            name = panel[0]
        
        out_path = videos_dir / f"{name}.mp4"
        
        # Skip if already exists
        if out_path.exists():
            continue
        
        # Find the corresponding video from ComfyUI
        if i < len(comfy_videos_chrono):
            comfy_vid = comfy_videos_chrono[i]
            try:
                shutil.copy2(comfy_vid, out_path)
                synced += 1
                if progress_cb:
                    progress_cb("sync", f"[{i+1}/{total_panels}] 同步: {comfy_vid.name} -> {name}.mp4")
            except Exception as e:
                if progress_cb:
                    progress_cb("sync", f"[{i+1}/{total_panels}] 同步失败: {e}")
    
    if progress_cb:
        progress_cb("sync", f"同步完成: {synced}/{total_panels} 个视频")
    
    return synced


def get_comfyui_output_dir() -> Path:
    """Get ComfyUI output directory for videos."""
    return COMFY / "output" / "video"


def find_latest_videos_in_comfyui(count: int) -> list[Path]:
    """Find the latest N videos in ComfyUI output directory.
    
    Args:
        count: Number of videos to find
    
    Returns:
        List of video paths, sorted by modification time (newest first)
    """
    comfy_dir = get_comfyui_output_dir()
    if not comfy_dir.exists():
        return []
    
    videos = sorted(
        comfy_dir.glob("MiniMax_H3_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    return videos[:count]


def copy_videos_to_project(
    ep_id: str,
    video_paths: list[Path],
    progress_cb: Optional[Callable] = None,
    episodes_path: Optional[Path] = None,
) -> int:
    """Copy videos from ComfyUI output to project directory.
    
    Args:
        ep_id: Episode ID
        video_paths: List of video paths to copy (in order)
        progress_cb: Optional callback for progress updates
    
    Returns:
        Number of videos copied
    """
    del episodes_path
    jobs = default_store().list_jobs(ep_id)
    if not jobs:
        raise ValueError(f"episode has no registered render jobs: {ep_id}")
    copied = 0
    for job, source in zip(jobs, video_paths):
        try:
            output = _copy_video_atomic(Path(source), Path(job["output_path"]))
            default_store().update_job(
                job["job_id"], status="succeeded", progress=1.0,
                output_path=str(output), preview_path=str(output),
            )
            copied += 1
            if progress_cb:
                progress_cb("copy", f"[{copied}/{min(len(jobs), len(video_paths))}] {output.name}")
        except Exception as exc:
            if progress_cb:
                progress_cb("copy", f"复制失败: {exc}")
    return copied

    if episodes_path is None:
        episodes_path = ROOT / "episodes.json"
    
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    if ep_id not in episodes:
        raise ValueError(f"Episode {ep_id} not in episodes.json")
    
    ep = episodes[ep_id]
    panels = ep["panels"]
    
    videos_dir = ROOT / f"{ep_id}_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    total = min(len(video_paths), len(panels))
    
    # Copy videos in reverse order (oldest first = first panels)
    for i in range(total - 1, -1, -1):
        video_path = video_paths[i]
        panel = panels[total - 1 - i]
        
        if isinstance(panel, dict):
            name = panel["name"]
        else:
            name = panel[0]
        
        out_path = videos_dir / f"{name}.mp4"
        
        try:
            shutil.copy2(video_path, out_path)
            copied += 1
            if progress_cb:
                progress_cb("copy", f"[{copied}/{total}] 复制: {video_path.name} -> {name}.mp4")
        except Exception as e:
            if progress_cb:
                progress_cb("copy", f"[{copied}/{total}] 复制失败: {e}")
    
    if progress_cb:
        progress_cb("copy", f"复制完成: {copied}/{total} 个视频")
    
    return copied


def render_episode_batch(
    ep_id: str,
    episodes_path: Optional[Path] = None,
    seed_base: int = 2026080400,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    progress_cb: Optional[Callable] = None,
) -> list[Path]:
    """Render all panels for an episode into H3 ref2va video+audio clips.

    Each panel needs two images: {name}_first.png and {name}_last.png
    in the {ep_id}_panels/ directory.

    All character reference portraits in {ep_id}_panels_charref/ are
    automatically passed to H3 as additional ref_images for character
    consistency across all video clips (up to MAX_CHAR_REFS = 7).

    Args:
        ep_id: Episode ID (must exist in episodes.json)
        episodes_path: Path to episodes.json
        seed_base: Base random seed
        duration_seconds: Per-clip duration in seconds (default 15.0)
        progress_cb: Optional callback(phase, message)

    Returns:
        List of generated video paths (each with native audio)
    """
    # Legacy entry point now delegates to the same durable worker contract used
    # by Web/CLI.  ``episodes_path`` and ``seed_base`` are retained only for API
    # compatibility; episode.json/task_store are the source of truth.
    del episodes_path, seed_base, duration_seconds
    from orchestrator import run_episode_jobs
    result = run_episode_jobs(ep_id, progress_cb=progress_cb)
    return [
        Path(job["output_path"])
        for job in result["snapshot"]["jobs"]
        if job["status"] == "succeeded" and job.get("output_path")
    ]

    if episodes_path is None:
        episodes_path = ROOT / "episodes.json"

    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    if ep_id not in episodes:
        raise ValueError(f"Episode {ep_id} not in episodes.json")

    ep = episodes[ep_id]
    panels = ep["panels"]

    # Character description for consistency
    female_char = ep.get("female_character", "")
    male_char = ep.get("male_character", "")
    character_desc = f"{female_char} {male_char}".strip()

    # Locate panel images
    panels_dir = ROOT / f"{ep_id}_panels"
    if not panels_dir.exists():
        raise FileNotFoundError(f"Panel images directory not found: {panels_dir}")

    # Find ALL character reference portraits (cap at MAX_CHAR_REFS).
    # H3 ref2va accepts up to 9 ref_images; we use 2 for first/last frames
    # and up to 7 for character refs. Naming convention: {role}_{angle}.png.
    charref_dir = ROOT / f"{ep_id}_panels_charref"
    char_refs = []
    if charref_dir.exists():
        all_refs = sorted(charref_dir.glob("*.png"))
        # Prioritize: female first, then male, then extras
        def _role_key(p: Path):
            n = p.stem.lower()
            if "female" in n or n.startswith("f"):
                return (0, n)
            if "male" in n or n.startswith("m"):
                return (1, n)
            return (2, n)
        all_refs.sort(key=_role_key)
        char_refs = all_refs[:MAX_CHAR_REFS]
        if char_refs and progress_cb:
            progress_cb(
                "video",
                f"找到 {len(char_refs)}/{len(all_refs)} 张角色定妆照（多角度），将作为 H3 额外参考图",
            )

    # Output directory for video clips (项目目录)
    videos_dir = ROOT / f"{ep_id}_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    total = len(panels)
    # 记录所有生成的视频路径（ComfyUI 输出目录中的路径）
    rendered_in_comfy = []  # ComfyUI 输出目录中的视频
    rendered_in_project = []  # 已经复制到项目目录的视频

    for i, panel in enumerate(panels):
        # Support both new dict format and legacy tuple format
        if isinstance(panel, dict):
            name = panel["name"]
        else:
            name = panel[0]
            panel = {"name": name, "first_frame_prompt": panel[1], "last_frame_prompt": panel[1],
                     "dialogue": "", "motion": "slow cinematic motion"}

        first_img = panels_dir / f"{name}_first.png"
        last_img = panels_dir / f"{name}_last.png"

        # Fallback: if _first/_last not found, try just name.png for both
        if not first_img.exists():
            fallback = panels_dir / f"{name}.png"
            if fallback.exists():
                first_img = fallback
                last_img = fallback
            else:
                raise FileNotFoundError(f"Missing panel images: {first_img} and {last_img}")

        if not last_img.exists():
            last_img = first_img  # Use first frame as last frame if no last frame

        out_path = videos_dir / f"{name}.mp4"
        
        # 先检查项目目录里是否已经有视频（已经手动复制过的情况）
        if out_path.exists():
            if progress_cb:
                progress_cb("video", f"[{i+1}/{total}] 已存在，跳过: {out_path.name}")
            rendered_in_project.append(out_path)
            continue
        
        # 检查 ComfyUI 输出目录是否有该视频（可能是上次中断后遗留的）
        comfy_video = COMFY / "output" / "video" / f"MiniMax_H3_*.mp4"
        comfy_videos_all = sorted(
            COMFY.glob("output/video/MiniMax_H3_*.mp4") if COMFY.joinpath("output/video").exists() else [],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # 如果 ComfyUI 里有最近 2 小时生成的视频，认为它可能属于这个 panel
        # 但这个匹配不准确，所以还是重新渲染更可靠
        # 这里直接渲染，不尝试恢复（之前的恢复逻辑太复杂且不可靠）
        
        # 渲染视频
        if progress_cb:
            progress_cb("video", f"[{i+1}/{total}] 渲染 H3 ref2va: {name}")
        
        # 2026-08-09 Dean: 传入 character_anchor（角色定妆照 → ref_image_2）
        # 从 charref_dir 自动查找，优先用 front 视角
        character_anchor_for_panel = None
        if char_refs:
            # 优先用 front 视角作为 character_anchor
            for ref in char_refs:
                if "front" in ref.stem.lower():
                    character_anchor_for_panel = ref
                    break
            # 如果没有 front，用第一张
            if not character_anchor_for_panel:
                character_anchor_for_panel = char_refs[0]
        
        try:
            # render_panel 返回 ComfyUI 输出目录的视频路径
            comfy_video_path = render_panel(
                first_frame=first_img,
                last_frame=last_img,
                panel=panel,
                output_path=out_path,
                character_desc=character_desc,
                char_refs=char_refs if char_refs else None,
                seed=seed_base + i,
                duration_seconds=duration_seconds,
                progress_cb=progress_cb,
                ep_id=ep_id,
                character_anchor=character_anchor_for_panel,  # 新增：传入角色锚点
            )
            rendered_in_comfy.append(comfy_video_path)

            # ── Persistent generation log: every successful render gets recorded ──
            if _GEN_LOG_AVAILABLE:
                try:
                    _size = comfy_video_path.stat().st_size if comfy_video_path.exists() else None
                    generation_log.update_status(
                        ep_id, i + 1, generation_log.RAW_RENDERED,
                        comfy_path=str(comfy_video_path),
                        size_bytes=_size,
                    )
                except Exception as _log_err:
                    if progress_cb:
                        progress_cb("video", f"[{i+1}/{total}] log 写入失败（不影响渲染）: {_log_err}")

            # ── INLINE COPY: copy ComfyUI output to project dir RIGHT NOW ──
            # No more two-phase "render all then copy all". Each panel is
            # atomic: render → copy → log. Crash anywhere leaves the prior
            # panels already in auto_008_videos/ and the log.
            try:
                if not out_path.exists():
                    shutil.copy2(comfy_video_path, out_path)
                if _GEN_LOG_AVAILABLE:
                    _fsize = out_path.stat().st_size if out_path.exists() else None
                    generation_log.update_status(
                        ep_id, i + 1, generation_log.FINALIZED,
                        final_path=str(out_path),
                        size_bytes=_fsize,
                    )
                if progress_cb:
                    progress_cb("video", f"[{i+1}/{total}] 复制+log完成: {out_path.name}")
            except Exception as copy_err:
                if progress_cb:
                    progress_cb("video", f"[{i+1}/{total}] 复制失败 {name}: {copy_err}")
                # Don't re-raise — the raw render succeeded; copy is best-effort
                # and can be retried via recover_from_filesystem()
        except Exception as e:
            if progress_cb:
                progress_cb("video", f"[{i+1}/{total}] 失败: {name} - {e}")
            raise

    # 全部渲染完成后，统一复制到项目目录
    if rendered_in_comfy and progress_cb:
        progress_cb("video", "")
        progress_cb("video", "=" * 50)
        progress_cb("video", "全部渲染完成，开始统一复制到项目目录...")
        progress_cb("video", "=" * 50)
    
    copied_count = 0
    # 按生成的顺序（oldest first = first panels）复制
    for i, panel in enumerate(panels):
        if isinstance(panel, dict):
            name = panel["name"]
        else:
            name = panel[0]
        
        out_path = videos_dir / f"{name}.mp4"
        if out_path.exists():
            continue  # 已经存在，跳过
        
        # 找到对应的 ComfyUI 视频
        # 按面板顺序（oldest to newest）匹配
        # rendered_in_comfy 是按生成顺序的（第一个生成的 = panel 0）
        if i < len(rendered_in_comfy):
            comfy_video = rendered_in_comfy[i]
            try:
                shutil.copy2(comfy_video, out_path)
                copied_count += 1
                if progress_cb:
                    progress_cb("video", f"[{copied_count}/{len(rendered_in_comfy)}] 复制: {comfy_video.name} -> {name}.mp4")

                # ── Persistent generation log: record successful copy to project ──
                if _GEN_LOG_AVAILABLE:
                    try:
                        _size = out_path.stat().st_size if out_path.exists() else None
                        generation_log.update_status(
                            ep_id, i + 1, generation_log.FINALIZED,
                            final_path=str(out_path),
                            size_bytes=_size,
                        )
                    except Exception as _log_err:
                        if progress_cb:
                            progress_cb("video", f"log 写入失败（不影响复制）: {_log_err}")
            except Exception as e:
                if progress_cb:
                    progress_cb("video", f"复制失败 {name}: {e}")
    
    if progress_cb:
        progress_cb("video", f"全部完成：渲染 {len(rendered_in_comfy)} 个 + 复制 {copied_count} 个")
        progress_cb("video", f"项目目录: {videos_dir}")

    return rendered_in_project + [videos_dir / f"{p.get('name', '') if isinstance(p, dict) else p[0]}.mp4" for p in panels[:len(rendered_in_comfy)]]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: render_video_h3.py <ep_id> [duration_seconds]")
        sys.exit(1)

    ep_id = sys.argv[1]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0

    def _progress(phase, msg):
        print(f"[{phase}] {msg}", flush=True)

    videos = render_episode_batch(ep_id, duration_seconds=dur, progress_cb=_progress)
    for v in videos:
        print(f"  {v}")
