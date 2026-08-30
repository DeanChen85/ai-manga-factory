# -*- coding: utf-8 -*-
"""
story_splitter.py
=================

UTF-8 clean rewrite for AI 漫剧工厂 (comic-book mode).

Calls MiniMax M3 (or compatible OpenAI-style chat-completions) to split a story
into structured storyboard panels. Comic mode outputs the rich
[STYLE HEADER] / [cuts] / [transitions] / [dialogue_bubbles] / [sfx] / --ar X --duration Y --style Z
schema consumed by render_video_h3.build_panel_prompt().

Schema is enforced by the model via SYSTEM_PROMPT_COMIC_CN, validated by
JSON parsing + post-parse schema check, and emitted as panel dicts ready for
render_video_h3.render_panel().

Public API:
    split_story(story_text, style="comic", min_panels=4, max_panels=10,
                use_lora=True, lora_strength=1.0, aspect_ratio="16:9")
        -> {"title": ..., "panels": [...], ...}

    validate_panels(panels) -> list[str]  (empty list = OK)

    panels_to_episode_dict(...)  -> episode dict consumable by web_app.py
"""
from __future__ import annotations

import json
import os
import re
import copy
import socket
from typing import Any, Callable, Optional
from urllib.parse import urlsplit, urlunsplit

from runtime_config import load_project_env
from generation_drafts import (
    load_stage1_checkpoint,
    record_stage2_status,
    save_stage1_checkpoint,
)
from action_catalog import (
    ACTION_CODES,
    ActionContractError,
    compile_action_spec,
    derived_action_components,
)

load_project_env()

# ── Config ───────────────────────────────────────────────────────────────
M3_API_KEY = os.environ.get("MiniMax_API_KEY", "")
M3_BASE_URL = os.environ.get("MiniMax_BASE_URL", "https://api.minimaxi.com/anthropic")
M3_MODEL = os.environ.get("MiniMax_MODEL", "MiniMax-M2.7")
M3_PROTOCOL = os.environ.get("MiniMax_PROTOCOL", "anthropic")
DEFAULT_M3_REQUEST_TIMEOUT_SECONDS = 180.0
MAX_M3_COMPLETION_TOKENS = 2048
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192
MAX_ANTHROPIC_MAX_TOKENS = 32768
DEPRECATED_MINIMAX_HOSTS = {"api.minimax.chat"}
DEPRECATED_MINIMAX_MODELS = {"abab6.5s-chat"}

DEFAULT_STYLE = "comic"
DEFAULT_MIN_PANELS = 4
DEFAULT_MAX_PANELS = 10
DEFAULT_ASPECT_RATIO = "16:9"
DEFAULT_DURATION_SECONDS = 10.0

# ── Import comic prompt templates from sibling module ───────────────────
from comic_prompts import (  # noqa: E402
    COMIC_EXAMPLE_HERO_KAIJU,
    SYSTEM_PROMPT_COMIC_CN,
    SYSTEM_PROMPT_COMIC_EN,
    H3_PROMPT_MASTER_RULES,
    build_series_system_prompt,
    build_storyboard_system_prompt,
)
from prompt_contracts import (  # noqa: E402
    SHOT_PLAN_VERSION,
    SOURCE_GENERATION_DURATION_SECONDS,
    PROMPT_SCHEMA_VERSION,
    SERIES_SCHEMA_VERSION,
    _text,
    allocate_edit_durations,
    auto_episode_shot_count,
    continuity_chain_warnings,
    enrich_episode_contract,
    normalize_series_contract,
    series_episode_context,
    shot_count_bounds,
    subtitle_mismatch_warnings,
    validate_platform_shot_plan,
    validate_series_contract,
)


class MissingMiniMaxAPIKey(RuntimeError):
    """Raised when live story generation is requested without a MiniMax key."""


class MiniMaxRequestTimeout(TimeoutError):
    """Raised when one user-triggered MiniMax request exceeds its wait limit."""


class MiniMaxOutputTruncated(RuntimeError):
    """Raised when MiniMax explicitly reports a completion length cutoff."""


class MiniMaxGenerationStageError(ValueError):
    """Fail-closed error for one explicit stage of the two-call V3 plan."""

    def __init__(self, stage: int, calls_started: int, detail: str):
        self.stage = int(stage)
        self.calls_started = int(calls_started)
        super().__init__(
            f"V3 两阶段生成在阶段 {self.stage}/2 失败（计划调用 2 次，已发起 "
            f"{self.calls_started} 次；已发起的调用可能计费）：{detail}。"
            "最终合同未保存，系统不会自动再次调用或重试；请检查后由用户手动重新生成。"
        )


def minimax_request_timeout_seconds(value: Any = None) -> float:
    """Resolve the configurable synchronous request timeout, defaulting to 180s."""
    raw = value
    if raw in (None, ""):
        raw = (
            os.environ.get("AI_MANGA_MINIMAX_TIMEOUT_SECONDS")
            or os.environ.get("MiniMax_TIMEOUT_SECONDS")
            or DEFAULT_M3_REQUEST_TIMEOUT_SECONDS
        )
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("MiniMax timeout must be a number of seconds") from exc
    if not 10 <= timeout <= 600:
        raise ValueError("MiniMax timeout must be between 10 and 600 seconds")
    return timeout


def minimax_chat_completions_url(base_url: Optional[str] = None) -> str:
    """Build one OpenAI-compatible endpoint without duplicating ``/v1``."""
    raw = (base_url or os.environ.get("MiniMax_BASE_URL") or M3_BASE_URL).strip()
    if not raw:
        raw = "https://api.minimaxi.com/v1"
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MiniMax_BASE_URL must be an absolute http(s) URL")
    path = re.sub(r"(?:/v1){2,}", "/v1", parsed.path.rstrip("/"))
    if path.endswith("/chat/completions"):
        final_path = path
    elif not path:
        final_path = "/v1/chat/completions"
    else:
        final_path = f"{path}/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, final_path, "", ""))


def minimax_anthropic_messages_url(base_url: Optional[str] = None) -> str:
    """Migrate a MiniMax host/base URL to its Anthropic Messages endpoint."""
    raw = (base_url or os.environ.get("MiniMax_BASE_URL") or M3_BASE_URL).strip()
    if not raw:
        raw = "https://api.minimaxi.com/anthropic"
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MiniMax_BASE_URL must be an absolute http(s) URL")
    migrated_netloc = "api.minimaxi.com" if (parsed.hostname or "").lower() in DEPRECATED_MINIMAX_HOSTS else parsed.netloc
    path = parsed.path.rstrip("/")
    path = re.sub(r"/(?:v1(?:/chat/completions)?|chat/completions)$", "", path)
    path = re.sub(r"/anthropic(?:/v1/messages)?$", "", path)
    final_path = f"{path}/anthropic/v1/messages" if path else "/anthropic/v1/messages"
    return urlunsplit((parsed.scheme, migrated_netloc, final_path, "", ""))


def minimax_protocol(value: Optional[str] = None) -> str:
    protocol = (value or os.environ.get("MiniMax_PROTOCOL") or M3_PROTOCOL).strip().lower()
    aliases = {"anthropic": "anthropic", "messages": "anthropic", "openai": "openai"}
    if protocol not in aliases:
        raise ValueError("MiniMax_PROTOCOL must be 'anthropic' or explicit 'openai'")
    return aliases[protocol]


def minimax_configuration_status(
    base_url: Optional[str] = None, model: Optional[str] = None, protocol: Optional[str] = None,
) -> dict[str, Any]:
    """Return resolved non-secret configuration and actionable deprecation warnings."""
    resolved_protocol = minimax_protocol(protocol)
    resolved_url = (
        minimax_anthropic_messages_url(base_url)
        if resolved_protocol == "anthropic" else minimax_chat_completions_url(base_url)
    )
    resolved_model = (model or os.environ.get("MiniMax_MODEL") or M3_MODEL).strip()
    configured_raw = (base_url or os.environ.get("MiniMax_BASE_URL") or M3_BASE_URL).strip()
    host = (urlsplit(configured_raw).hostname or "").lower()
    warnings: list[str] = []
    if host in DEPRECATED_MINIMAX_HOSTS:
        warnings.append(
            "MiniMax_BASE_URL 正在使用旧域名 api.minimax.chat；建议迁移到国内 "
            "https://api.minimaxi.com/v1，国际用户可显式使用 https://api.minimax.io/v1。"
        )
    if resolved_model in DEPRECATED_MINIMAX_MODELS or resolved_model.lower().startswith("abab"):
        warnings.append(
            f"MiniMax_MODEL={resolved_model} 属于旧模型配置；当前默认模型为 MiniMax-M2.7。"
        )
    return {
        "endpoint": resolved_url,
        "protocol": resolved_protocol,
        "model": resolved_model,
        "deprecated": bool(warnings),
        "warnings": warnings,
    }


# ── Music + ambience preset keys (must match render_video_h3.MUSIC_PRESETS / AMBIENCE_PRESETS) ──
MUSIC_PRESETS = (
    "auto_contextual",
    "soft_piano",
    "string_orch",
    "urban_electronic",
    "chinese_folk",
    "suspense_dark",
    "epic_brass",
)
AMBIENCE_PRESETS = (
    "auto_contextual",
    "rain_night_city",
    "office_quiet",
    "forest_morning",
    "subway_crowd",
    "storm_thunder",
    "silence",
)

INTENSITY_VALUES = {"EXTREME", "POWERFUL", "AGGRESSIVE", "SMOOTH", "TENSE"}


# ── JSON repair helpers ──────────────────────────────────────────────────
def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` fences, <think>...</think> blocks, and surrounding prose."""
    text = text.strip()
    # Remove <think>...</think> blocks (MiniMax reasoning) - handle unclosed tags
    # 先去掉闭合的 
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 再去掉未闭合的 
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    # 去掉残留标签
    text = re.sub(r'<think>', '', text)
    text = re.sub(r'</think>', '', text)
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
    # Extract the JSON object from the text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        return match.group(0)
    return text


def _extract_json_object(text: str) -> str:
    """Find the first balanced {...} in text (LLMs sometimes wrap JSON in prose)."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _parse_json_lenient(text: str) -> dict:
    """Parse JSON even when LLM produces slight syntax errors."""
    cleaned = _strip_markdown_fences(text)
    cleaned = _extract_json_object(cleaned)
    # Try plain parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Repair common issues: trailing commas, single quotes
    cleaned2 = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned2 = cleaned2.replace("'", '"')
    try:
        return json.loads(cleaned2)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output is not valid JSON: {e}\n---raw---\n{text[:500]}") from e


# ── Schema validation ────────────────────────────────────────────────────
def validate_panel(panel: dict) -> list[str]:
    """Return list of validation errors (empty = OK)."""
    errors = []
    if "name" not in panel or not re.match(r"^[a-z0-9_]+$", panel["name"]):
        errors.append("panel.name must be lowercase english slug (e.g. ep01_panel01_hero_rooftop)")
    if panel.get("prompt_mode", "comic") not in {"comic", "cinematic"}:
        errors.append("panel.prompt_mode must be 'comic' or 'cinematic'")
    duration = float(panel.get("duration_seconds", 10))
    lane_duration = float(
        panel.get("edit_duration_seconds")
        if panel.get("shot_plan_version") == SHOT_PLAN_VERSION
        else duration
    )
    if not 2 <= duration <= 15:
        errors.append("panel.duration_seconds must be between 2 and 15")
    if not panel.get("scene_id"):
        errors.append("panel.scene_id missing")
    if not panel.get("first_frame"):
        errors.append("panel.first_frame missing")
    if not panel.get("last_frame"):
        errors.append("panel.last_frame missing")
    if not panel.get("continuity_group"):
        errors.append("panel.continuity_group missing")
    if not isinstance(panel.get("continuity_state_in"), dict):
        errors.append("panel.continuity_state_in must be an object")
    if not isinstance(panel.get("continuity_state_out"), dict):
        errors.append("panel.continuity_state_out must be an object")
    if not isinstance(panel.get("character_ids"), list):
        errors.append("panel.character_ids must be a list of character IDs")
    # cuts validation
    cuts = panel.get("cuts", [])
    if not cuts:
        errors.append("panel.cuts must be non-empty (>=1 cut)")
    for i, cut in enumerate(cuts):
        if "time_range" not in cut:
            errors.append(f"panel.cuts[{i}].time_range missing")
        description = str(cut.get("shot_description") or "")
        # V3 supports Chinese/Japanese as first-class output languages, where
        # whitespace word counts are meaningless. Require useful visual detail
        # without rejecting non-Latin scripts.
        if len(re.sub(r"\s+", "", description)) < 30:
            errors.append(
                f"panel.cuts[{i}].shot_description must contain >= 30 non-whitespace characters"
            )
        intensity = (cut.get("intensity") or "").upper()
        if intensity not in INTENSITY_VALUES:
            errors.append(f"panel.cuts[{i}].intensity must be one of {sorted(INTENSITY_VALUES)}, got '{intensity}'")
    # Audio cues are an optional V3 lane. A deliberately quiet shot is valid;
    # do not require a legacy onomatopoeia merely to pass creative review.
    if panel.get("on_screen_text") or panel.get("dialogue_bubbles"):
        errors.append("panel renderer-facing visible text must be empty; use postproduction timeline")
    for lane_name in ("spoken_dialogue", "subtitle_timeline", "postproduction_on_screen_text", "audio_cues"):
        previous_end = 0.0
        for i, item in enumerate(panel.get(lane_name, [])):
            start = item.get("start_s")
            end = item.get("end_s")
            if start is None or end is None:
                errors.append(f"panel.{lane_name}[{i}] requires parseable start/end times")
                continue
            if not (0 <= float(start) < float(end) <= lane_duration):
                errors.append(f"panel.{lane_name}[{i}] outside panel duration")
            if lane_name != "audio_cues" and float(start) < previous_end:
                errors.append(f"panel.{lane_name}[{i}] overlaps previous item")
            previous_end = max(previous_end, float(end))
            if lane_name == "spoken_dialogue":
                speaker_id = item.get("speaker_id")
                if speaker_id not in panel.get("character_ids", []):
                    errors.append(f"panel.spoken_dialogue[{i}].speaker_id must reference a visible character")
                max_chars = int(item.get("max_chars") or max(1, round((float(end) - float(start)) * 6)))
                if len(item.get("text", "")) > max_chars:
                    errors.append(f"panel.spoken_dialogue[{i}] exceeds max_chars={max_chars}")
    errors.extend(
        f"panel.{warning}"
        for warning in subtitle_mismatch_warnings(
            panel.get("spoken_dialogue", []), panel.get("subtitle_timeline", [])
        )
    )
    # music/ambience key validation
    bgm = panel.get("background_music", "")
    if bgm and bgm not in MUSIC_PRESETS:
        errors.append(f"panel.background_music '{bgm}' not in {MUSIC_PRESETS}")
    amb = panel.get("ambience", "")
    if amb and amb not in AMBIENCE_PRESETS:
        errors.append(f"panel.ambience '{amb}' not in {AMBIENCE_PRESETS}")
    package = panel.get("prompt_package")
    if not isinstance(package, dict):
        errors.append("panel.prompt_package missing")
    else:
        for key in ("positive_prompt", "negative_prompt", "character_ids", "scene_id"):
            if package.get(key) in (None, "", []):
                errors.append(f"panel.prompt_package.{key} missing")
    return errors


def validate_panels(panels: list[dict]) -> list[str]:
    all_errors = []
    for i, panel in enumerate(panels):
        for e in validate_panel(panel):
            all_errors.append(f"panel[{i}] {e}")
    # TEMPORARILY DISABLED: continuity chain validation is too strict for initial generation.
    # The LLM doesn't maintain perfect continuity between panels, which blocks generation.
    # We'll iterate on improving this once the system is working end-to-end.
    # all_errors.extend(f"continuity {warning}" for warning in continuity_chain_warnings(panels))
    return all_errors


def validate_episode_contract(episode: dict[str, Any]) -> list[str]:
    """Validate the V3 bibles and panel references after normalization."""
    errors: list[str] = []
    for key in ("story_bible", "character_bible", "scene_bible", "panels"):
        if not episode.get(key):
            errors.append(f"episode.{key} missing or empty")
    character_ids = {item.get("character_id") for item in episode.get("character_bible", [])}
    scene_ids = {item.get("scene_id") for item in episode.get("scene_bible", [])}
    visual = episode.get("visual_bible") if isinstance(episode.get("visual_bible"), dict) else {}
    for field in ("style_prompt", "global_negative_prompt"):
        value = _text(visual.get(field))
        if not value:
            errors.append(f"visual_bible.{field} missing")
        elif not value.isascii() or not value.isprintable():
            errors.append(f"visual_bible.{field} must be printable ASCII English")
    for index, character in enumerate(episode.get("character_bible", [])):
        voice = character.get("voice_profile")
        if not isinstance(voice, dict) or not all(voice.get(key) for key in ("language", "age", "timbre", "pace")):
            errors.append(f"character_bible[{index}].voice_profile incomplete")
        identity_tags = character.get("model_identity_tags_en")
        wardrobe_tags = character.get("model_wardrobe_tags_en")
        if not isinstance(identity_tags, list) or not identity_tags:
            errors.append(f"character_bible[{index}].model_identity_tags_en missing")
        if not isinstance(wardrobe_tags, list) or not wardrobe_tags:
            errors.append(f"character_bible[{index}].model_wardrobe_tags_en missing")
        wardrobe_text = " | ".join(_text(tag).casefold() for tag in wardrobe_tags or [])
        if wardrobe_text and not re.search(
            r"\b(?:shoes?|sneakers?|boots?|loafers?|heels?|sandals?)\b", wardrobe_text
        ):
            errors.append(
                f"character_bible[{index}].model_wardrobe_tags_en requires explicit footwear"
            )
        if any(re.search(r"[\u3400-\u9fff]", str(tag)) for tag in (identity_tags or []) + (wardrobe_tags or [])):
            errors.append(f"character_bible[{index}] model-facing tags must be English")
        for warning in character.get("model_prompt_warnings") or []:
            errors.append(f"character_bible[{index}] {warning}")
    for index, scene in enumerate(episode.get("scene_bible", [])):
        model_prompt = str(scene.get("model_prompt_en") or "")
        if not model_prompt:
            errors.append(f"scene_bible[{index}].model_prompt_en missing")
        elif re.search(r"[\u3400-\u9fff]", model_prompt):
            errors.append(f"scene_bible[{index}].model_prompt_en must be English")
        for warning in scene.get("model_prompt_warnings") or []:
            errors.append(f"scene_bible[{index}] {warning}")
    for index, panel in enumerate(episode.get("panels", [])):
        if panel.get("scene_id") not in scene_ids:
            errors.append(f"panels[{index}].scene_id does not reference scene_bible")
        unknown = set(panel.get("character_ids") or []) - character_ids
        if unknown:
            errors.append(f"panels[{index}].character_ids unknown: {sorted(unknown)}")
    errors.extend(validate_panels(episode.get("panels") or []))
    if (episode.get("shot_plan") or {}).get("version") == SHOT_PLAN_VERSION:
        target = float(
            (episode.get("shot_plan") or {}).get("target_edit_duration_seconds")
            or (episode.get("render_settings") or {}).get("target_edit_duration_seconds")
            or 0
        )
        errors.extend(validate_platform_shot_plan(episode.get("panels") or [], target))
    return errors


_V3_STAGE1_KEYS = (
    "story_bible", "character_bible", "visual_bible", "scene_bible", "story_beats",
)

_V3_STAGE1_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sb", "cb", "vb", "sc", "beats"],
    "properties": {
        "sb": {
            "type": "object", "additionalProperties": False,
            "required": ["t", "l", "s", "th", "cr"],
            "properties": {
                "t": {"type": "string"}, "l": {"type": "string"}, "s": {"type": "string"},
                "th": {"type": "array", "items": {"type": "string"}},
                "cr": {"type": "array", "items": {"type": "string"}},
            },
        },
        "cb": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "n", "desc", "hc", "hs", "ec", "it", "wt", "v"],
                "properties": {
                    "id": {"type": "string"}, "n": {"type": "string"}, "desc": {"type": "string"},
                    "hc": {
                        "type": "string",
                        "enum": ["black", "brown", "white", "silver", "blonde", "green"],
                        "description": "One explicit locked hair color for identity continuity.",
                    },
                    "hs": {
                        "type": "string", "minLength": 2, "pattern": "^[\\x20-\\x7E]+$",
                        "description": "One explicit printable-ASCII hair style, such as short bob or wet swept-back hair.",
                    },
                    "ec": {
                        "type": "string", "enum": ["brown", "black", "blue", "green", "gray"],
                        "description": "One explicit locked eye color for identity continuity.",
                    },
                    "it": {
                        "type": "array", "minItems": 4,
                        "description": "Model-facing identity tags. Every item must be printable ASCII English, never Chinese editorial prose.",
                        "examples": [["1boy", "adult male", "Chinese", "short black hair", "brown eyes"]],
                        "items": {"type": "string", "minLength": 2, "pattern": "^[\\x20-\\x7E]+$"},
                    },
                    "wt": {
                        "type": "array", "minItems": 2,
                        "description": "Model-facing wardrobe tags. Every item must be printable ASCII English with colors and garment nouns; one item MUST lock colored footwear (shoes, sneakers, boots, loafers, heels or sandals).",
                        "examples": [["navy blue rain jacket", "yellow courier shoulder bag", "white sneakers"]],
                        "items": {"type": "string", "minLength": 2, "pattern": "^[\\x20-\\x7E]+$"},
                    },
                    "v": {
                        "type": "object", "additionalProperties": False,
                        "required": ["lang", "age", "tone", "pace"],
                        "properties": {
                            "lang": {"type": "string"}, "age": {"type": "string"},
                            "tone": {"type": "string"}, "pace": {"type": "string"},
                        },
                    },
                },
            },
        },
        "vb": {
            "type": "object", "additionalProperties": False,
            "required": ["sp", "neg"],
            "properties": {
                "sp": {
                    "type": "string", "minLength": 8, "pattern": "^[\\x20-\\x7E]+$",
                    "description": "Printable ASCII English visual style tags for image/video models.",
                },
                "neg": {
                    "type": "string", "minLength": 8, "pattern": "^[\\x20-\\x7E]+$",
                    "description": "Printable ASCII English global negative tags; never Chinese or mojibake.",
                },
            },
        },
        "sc": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "desc", "mp"],
                "properties": {
                    "id": {"type": "string"}, "desc": {"type": "string"},
                    "mp": {
                        "type": "string", "minLength": 8, "pattern": "^[\\x20-\\x7E]+$",
                        "description": "Model-facing scene prompt in printable ASCII English only.",
                        "examples": ["grounded modern Chinese courier station, rainy night, practical fluorescent lighting"],
                    },
                },
            },
        },
        "beats": {
            "type": "array", "minItems": 5,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["id", "r", "q", "proof", "pay"],
                "properties": {
                    "id": {"type": "string"},
                    "r": {"type": "string", "enum": ["hook", "setup", "escalation", "reversal", "cliffhanger", "close"]},
                    "q": {"type": "string"}, "proof": {"type": "string"}, "pay": {"type": "string"},
                },
            },
        },
    },
}


def _v3_stage1_tool_schema(
    requested_character_count: Optional[int] = None,
    single_scene: bool = False,
) -> dict[str, Any]:
    """Build deterministic stage-1 slots; only unknown/other cast sizes use an array."""
    if requested_character_count is not None and not 1 <= int(requested_character_count) <= 20:
        raise ValueError("requested_character_count must be between 1 and 20")
    base = _V3_STAGE1_TOOL_SCHEMA["properties"]
    character = copy.deepcopy(base["cb"]["items"])
    scene = copy.deepcopy(base["sc"]["items"])
    beat = copy.deepcopy(base["beats"]["items"])
    properties: dict[str, Any] = {
        "sb": copy.deepcopy(base["sb"]),
        "vb": copy.deepcopy(base["vb"]),
        "s1": scene,
    }
    required = ["sb", "vb", "s1"]
    count = int(requested_character_count) if requested_character_count is not None else None
    if count == 2:
        properties["c1"] = copy.deepcopy(character)
        properties["c2"] = copy.deepcopy(character)
        required.extend(["c1", "c2"])
    else:
        cast = {"type": "array", "minItems": 1, "items": copy.deepcopy(character)}
        if count is not None:
            cast["minItems"] = count
            cast["maxItems"] = count
        properties["cb"] = cast
        required.append("cb")
    if not single_scene:
        properties["sx"] = {
            "type": "array", "minItems": 1,
            "description": "Optional additional scenes after required primary scene s1.",
            "items": copy.deepcopy(scene),
        }
    for slot, roles in (
        ("h", ["hook"]), ("setup", ["setup"]),
        ("escalation", ["escalation"]), ("reversal", ["reversal"]),
        ("end", ["cliffhanger", "close"]),
    ):
        slot_schema = copy.deepcopy(beat)
        slot_schema["properties"]["r"]["enum"] = roles
        properties[slot] = slot_schema
        required.append(slot)
    return {
        "type": "object", "additionalProperties": False,
        "required": required, "properties": properties,
    }

_V3_STAGE2_SHOT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "r", "b", "d", "c", "s", "act", "f", "l", "why", "next", "cam", "tr", "edit", "pri", "g", "si", "so", "dlg", "aud"],
    "properties": {
        "id": {"type": "string"},
        "r": {"type": "string", "enum": ["hook", "setup", "escalation", "reversal", "cliffhanger", "close"]},
        "b": {"type": "string"}, "d": {"type": "number", "minimum": 1.5, "maximum": 4.0},
        "c": {"type": "array", "items": {"type": "string"}}, "s": {"type": "string"},
        "act": {
            "type": "object", "additionalProperties": False,
            "required": ["sub", "code", "obj"],
            "description": "One visible action decomposed into subject, one stable action code and physical target. Shot f/l are its canonical start/end states.",
            "properties": {
                "sub": {"type": "string", "minLength": 1, "description": "One visible subject ID or name."},
                "code": {
                    "type": "string",
                    "enum": list(ACTION_CODES),
                    "description": "Choose exactly one stable physical action code. Never invent a code or return a natural-language verb.",
                },
                "obj": {
                    "type": "string", "minLength": 1,
                    "description": (
                        "One visible physical object or target. For DROP_OBJECT this MUST name both "
                        "the moving object and destination, e.g. 'coins into charity box' or "
                        "'wallet onto floor'; never return only the container/destination."
                    ),
                },
            },
        },
        "f": {"type": "string"}, "l": {"type": "string"},
        "why": {"type": "string"}, "next": {"type": "string"},
        "cam": {
            "type": "object", "additionalProperties": False,
            "required": ["size", "angle", "move", "comp"],
            "properties": {key: {"type": "string", "minLength": 1} for key in ("size", "angle", "move", "comp")},
        },
        "tr": {
            "type": "object", "additionalProperties": False,
            "required": ["type", "motivation"],
            "properties": {"type": {"type": "string", "minLength": 1}, "motivation": {"type": "string", "minLength": 1}},
        },
        "edit": {
            "type": "object", "additionalProperties": False,
            "required": ["moment", "in", "out"],
            "properties": {key: {"type": "string", "minLength": 1} for key in ("moment", "in", "out")},
        },
        "pri": {"type": "string", "enum": ["must_have", "important", "optional"]},
        "g": {"type": "string"}, "si": {"type": "object"}, "so": {"type": "object"},
        "dlg": {"type": "array"}, "aud": {"type": "array"},
        "bgm": {"type": "string", "enum": [value for value in MUSIC_PRESETS if value != "auto_contextual"]},
        "amb": {"type": "string", "enum": [value for value in AMBIENCE_PRESETS if value != "auto_contextual"]},
    },
}


def _v3_stage2_tool_schema(
    shot_count: int,
    *,
    character_ids: Optional[list[str]] = None,
    scene_ids: Optional[list[str]] = None,
    beat_ids: Optional[list[str]] = None,
) -> dict[str, Any]:
    count = int(shot_count)
    if not 1 <= count <= 400:
        raise ValueError("shot_count must be between 1 and 400")
    slots = [f"p{index:02d}" for index in range(1, count + 1)]
    shot_schema = copy.deepcopy(_V3_STAGE2_SHOT_SCHEMA)
    shot_schema["properties"]["c"]["uniqueItems"] = True
    shot_schema["properties"]["c"]["minItems"] = 1
    clean_character_ids = [_text(value) for value in character_ids or [] if _text(value)]
    clean_scene_ids = [_text(value) for value in scene_ids or [] if _text(value)]
    clean_beat_ids = [_text(value) for value in beat_ids or [] if _text(value)]
    if clean_character_ids:
        shot_schema["properties"]["c"]["items"]["enum"] = clean_character_ids
        shot_schema["properties"]["c"]["maxItems"] = len(clean_character_ids)
        shot_schema["properties"]["act"]["properties"]["sub"]["enum"] = clean_character_ids
    if clean_scene_ids:
        shot_schema["properties"]["s"]["enum"] = clean_scene_ids
    if clean_beat_ids:
        shot_schema["properties"]["b"]["enum"] = clean_beat_ids
    return {
        "type": "object", "additionalProperties": False, "required": slots,
        "properties": {slot: copy.deepcopy(shot_schema) for slot in slots},
    }


def explicit_requested_character_count(text: str) -> Optional[int]:
    """Return only an explicitly stated core-character count; never infer from names."""
    source = _text(text)
    number_token = r"(?:\d{1,2}|[一二三四五六七八九十两]{1,3})"
    patterns = (
        rf"(?<!\d)({number_token})\s*(?:位|个|名)\s*(?:核心|主要|主角|主角团)\s*(?:人物|角色|演员)?",
        rf"(?:核心|主要|主角|主角团)\s*(?:人物|角色|演员)?\s*(?:共|一共|总共|为|有|:|：)?\s*({number_token})\s*(?:位|个|名)?",
        r"(?<!\w)(\d{1,2})\s+(?:core|main|principal)\s+(?:characters?|actors?)(?!\w)",
    )

    def parse_count(value: str) -> int:
        token = _text(value)
        if token.isdigit():
            return int(token)
        digits = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                  "六": 6, "七": 7, "八": 8, "九": 9}
        if token == "十":
            return 10
        if "十" in token:
            left, right = token.split("十", 1)
            return (digits.get(left, 1) * 10) + digits.get(right, 0)
        return digits.get(token, 0)

    values = {
        parse_count(match.group(1))
        for pattern in patterns
        for match in re.finditer(pattern, source, flags=re.IGNORECASE)
        if 1 <= parse_count(match.group(1)) <= 20
    }
    return next(iter(values)) if len(values) == 1 else None


def explicit_single_scene(text: str) -> bool:
    """Recognize an explicit single-location production constraint, not story inference."""
    source = _text(text)
    return bool(re.search(
        r"(?:单一|唯一|仅一个|只有一个|全片同一|全剧同一)\s*(?:主要)?(?:场景|地点|空间)|"
        r"(?:全片|全剧|故事)?\s*只(?:发生|出现)?\s*在\s*同一(?:家|个|处|间)?"
        r"[^，。；\n]{0,16}(?:便利店|咖啡馆|餐厅|办公室|仓库|店|馆|室|房|屋|车|船)|"
        r"\b(?:single|one)\s+(?:primary\s+)?(?:scene|location)\b",
        source,
        flags=re.IGNORECASE,
    ))


def explicit_identity_equivalence_hints(text: str) -> list[str]:
    """Expose explicit time/version language as hints, without resolving names or merging actors."""
    source = _text(text)
    hints: list[str] = []
    signals = (
        (r"[一二三四五六七八九十百两\d]+\s*(?:分钟|小时|天|年)后", "an explicitly stated later-time version appears"),
        (r"(?:多年后|十年后|未来的|过去的|年轻时的|老年时的|少年时的)", "an explicit age/time version appears"),
        (r"(?:换装|伪装|乔装|回忆中的)", "an explicit wardrobe/disguise/memory version appears"),
    )
    for pattern, hint in signals:
        if re.search(pattern, source):
            hints.append(hint + "; preserve the same character_id unless a separate actor is explicit")
    return hints


def split_story_checkpoint_inputs(story_text: str, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Build the exact non-secret binding inputs used by Web checkpoint matching."""
    synopsis = str(kwargs.get("synopsis") or "").strip()
    resolved_story = synopsis or str(story_text or "").strip()
    topic = str(kwargs.get("topic") or kwargs.get("title") or "").strip() or resolved_story[:80]
    target_audience = str(kwargs.get("target_audience") or "general audience").strip()
    platform = str(kwargs.get("platform") or "custom").strip()
    duration_seconds = float(kwargs.get("duration_seconds", DEFAULT_DURATION_SECONDS))
    max_panels = int(kwargs.get("max_panels", DEFAULT_MAX_PANELS))
    total = float(kwargs.get("total_duration_seconds") or (duration_seconds * max_panels))
    shot_count = int(kwargs.get("shot_count") or auto_episode_shot_count(total))
    voice_language = kwargs.get("voice_language")
    language = str(kwargs.get("language") or "cn")
    if voice_language is None:
        voice_language = "Chinese" if language == "cn" else "English"
    resolved_mode = str(kwargs.get("prompt_mode") or kwargs.get("style") or "comic").strip().lower()
    visual_style = str(kwargs.get("visual_style") or "premium comic-book animation")
    style_enforcement = str(kwargs.get("style_enforcement") or visual_style)
    aspect_ratio = str(kwargs.get("aspect_ratio") or DEFAULT_ASPECT_RATIO)
    character_brief = str(kwargs.get("character_brief") or "")
    creative_brief = {
        "topic": topic, "synopsis": resolved_story,
        "visual_style": style_enforcement, "target_audience": target_audience,
        "total_duration_seconds": total, "shot_count": shot_count,
        "shot_plan_version": SHOT_PLAN_VERSION, "language": voice_language,
        "platform": platform, "aspect_ratio": aspect_ratio,
    }
    render_settings = {
        "prompt_mode": resolved_mode, "visual_style": visual_style,
        "style_enforcement": style_enforcement, "aspect_ratio": aspect_ratio,
        "duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
        "source_generation_duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
        "target_edit_duration_seconds": total, "shot_plan_version": SHOT_PLAN_VERSION,
        "use_lora": bool(kwargs.get("use_lora", True)),
        "lora_strength": float(kwargs.get("lora_strength", 1.0)),
        "sage_mode": str(kwargs.get("sage_mode") or "auto"),
        "ref_image_size": str(kwargs.get("ref_image_size") or "match"),
        "background_music": str(kwargs.get("background_music") or "epic_brass"),
        "ambience": str(kwargs.get("ambience") or "office_quiet"),
        "voice_language": voice_language, "platform": platform,
        "total_duration_seconds": total, "shot_count": shot_count,
        "creative_brief": creative_brief,
    }
    count = explicit_requested_character_count("\n".join(filter(None, [topic, resolved_story, character_brief])))
    hints = explicit_identity_equivalence_hints("\n".join(filter(None, [resolved_story, character_brief])))
    single_scene = explicit_single_scene("\n".join(filter(None, [topic, resolved_story, character_brief])))
    return {
        "creative_brief": creative_brief,
        "settings": {
            "requested_title": str(kwargs.get("title") or ""),
            "panel_range": {"min": shot_count, "max": shot_count, "exact": shot_count},
            "render_settings": render_settings,
            "character_brief_from_user": character_brief,
            "requested_character_count": count,
            "identity_equivalence_hints": hints,
            "single_scene": single_scene,
        },
    }


def _stage1_system_prompt(
    language: str, requested_character_count: Optional[int] = None,
    identity_equivalence_hints: Optional[list[str]] = None,
    single_scene: bool = False,
) -> str:
    output_language = "Simplified Chinese editorial prose" if language == "cn" else "English editorial prose"
    count_rule = (
        f"The user explicitly requires exactly {int(requested_character_count)} core characters; fill exactly the required character slots."
        if requested_character_count is not None
        else "Character count is not explicit; include only separately declared actors and do not invent identity duplicates."
    )
    equivalence = "; ".join(identity_equivalence_hints or []) or "No extra explicit identity-equivalence hint."
    wire_rule = (
        "Use required c1 and c2 objects; do not return cb. "
        if requested_character_count == 2 else "Use required non-empty cb array. "
    ) + (
        "Use required s1 only; this is an explicit single-scene story. "
        if single_scene else "Use required primary scene s1; use optional sx only for real additional scenes. "
    ) + "Use required beat object slots h, setup, escalation, reversal and end. Do not return sc or beats arrays."
    return f"""You are an elite series head writer and continuity supervisor.
This is STAGE 1 OF 2 of an explicitly planned V3 generation, not a retry.
Return one compact minified JSON object only, using {output_language}. Never emit shots or panels.
{count_rule}
{wire_rule}
An older/younger self, future/past appearance, disguise, wardrobe change, time skip, memory image,
or suspected identity variant MUST reuse the same character_id unless the user explicitly declares
a clone, double, twin or separate actor. Never split one person into a new actor merely because their
name, age, outfit, title or time version differs. Explicit identity-equivalence hints: {equivalence}
Every character must lock hc hair color, hs hair style and ec eye color; never leave appearance generic.
Every wt wardrobe list must lock at least one colored footwear item (shoes, sneakers, boots, loafers, heels or sandals); barefoot-by-omission is forbidden.
All it/wt array items, vb.sp, vb.neg and every scene mp string MUST be printable ASCII English only. Chinese is allowed
in editorial desc/story fields, never in model-facing it/wt/vb.sp/vb.neg/mp. Do not translate or repair these fields later.
The forced tool schema is authoritative. Every required slot must contain original story content;
never omit a slot and never emit an empty object/array as a placeholder.
Identity/wardrobe tags and scene mp must be ASCII English. Keep every string terse."""


def _stage2_system_prompt(language: str, shot_count: int, target_seconds: float) -> str:
    return f"""You are an elite short-drama storyboard director.
This is STAGE 2 OF 2 of an explicitly planned V3 generation, not a retry.
The validated stage-1 bibles in the user message are immutable. Return minified JSON only.
Fill every required fixed shot slot p01 through p{shot_count:02d}; do not return a shots array.
The d values across all {shot_count} slots must sum exactly to {target_seconds:g}.
Use only stage-1 character IDs, scene IDs and beat IDs. Every shot has one filmable visible action,
a visible state change, cause and next hook. Use 1-2 visible characters by default; group shots only
for hook/setup/close with a reason and <=2.5s. Adjacent camera compositions must differ.
The c array lists only characters actually visible in that shot. Each character ID may appear at most once;
never duplicate an ID to represent prominence, framing, dialogue or multiple body parts.
For every act object: sub is one visible character ID from that shot; code is exactly one uppercase enum value from the forced tool schema; obj is one visible physical object/target. Shot f and l are the canonical observable start and end states. The action code, not natural-language verb wording, is authoritative. Never return verb/res fields and never invent a code. For DROP_OBJECT, obj MUST name the moving object plus its destination (for example "coins into charity box"), never only the box/container.
**VALID examples:** sub="char_01", code="OPEN_OBJECT", obj="office door", f="door closed", l="door fully open"; sub="char_02", code="PRESS_CONTROL", obj="alarm button", f="button untouched", l="button remains depressed and indicator lit"; sub="char_01", code="DROP_OBJECT", obj="coins into charity box", f="coins rest in open palm", l="coins settle at bottom of charity box".
Use one code only. Psychology, abstract intent and chained actions are forbidden.
cam/tr/edit are named objects exactly as the forced tool schema specifies.
dlg items are [speaker_id,text,start_s,end_s,delivery_style,max_chars]; aud items are
[type,prompt,start_s,end_s]. When bgm/amb are present, choose one schema preset per shot from its
story emotion and physical location; never copy one mood blindly across the whole episode.
Do not create subtitles or visible/on-screen text. Keep prose concise."""


def _stage_error(stage: int, calls_started: int, detail: str, cause: Exception | None = None):
    error = MiniMaxGenerationStageError(stage, calls_started, detail)
    if cause is not None:
        raise error from cause
    raise error


def _parse_stage_response(raw: str, stage: int) -> dict[str, Any]:
    try:
        parsed = _parse_json_lenient(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        text = str(raw or "")
        shape = (
            f"响应正文 {len(text)} 字符；包含左花括号={'是' if '{' in text else '否'}；"
            f"包含右花括号={'是' if '}' in text else '否'}"
        )
        _stage_error(stage, stage, f"MiniMax 已返回，但该阶段 JSON 无法解析（{shape}）", exc)
    if not isinstance(parsed, dict):
        _stage_error(stage, stage, "该阶段必须返回一个 JSON object")
    return parsed


def _wire_role(value: Any) -> str:
    """Normalize only explicit, documented role spellings; never infer order."""
    normalized = re.sub(r"[\s-]+", "_", _text(value).casefold())
    aliases = {
        "hook": "hook", "opening_hook": "hook", "cold_open_hook": "hook",
        "setup": "setup", "story_setup": "setup",
        "escalation": "escalation", "rising_action": "escalation",
        "reversal": "reversal", "turning_point": "reversal",
        "cliffhanger": "cliffhanger", "ending_cliffhanger": "cliffhanger",
        "close": "close", "closing": "close", "resolution": "close",
    }
    return aliases.get(normalized, "")


def _expand_wire_beats(raw_beats: Any) -> list[dict[str, Any]]:
    """Accept explicit list roles or canonical role-keyed objects, never positions."""
    entries: list[tuple[Any, Any]]
    if isinstance(raw_beats, list):
        entries = [(None, item) for item in raw_beats]
    elif isinstance(raw_beats, dict):
        entries = list(raw_beats.items())
    else:
        return []
    beats: list[dict[str, Any]] = []
    for role_key, item in entries:
        if not isinstance(item, dict):
            continue
        role_from_item = _wire_role(item.get("r") or item.get("role"))
        role_from_key = _wire_role(role_key)
        if role_from_item and role_from_key and role_from_item != role_from_key:
            role = ""
        else:
            role = role_from_item or role_from_key
        beats.append({
            "beat_id": item.get("id") or item.get("beat_id"),
            "role": role,
            "dramatic_question": item.get("q") or item.get("dramatic_question"),
            "visible_proof": item.get("proof") or item.get("visible_proof"),
            "payoff_or_hook": item.get("pay") or item.get("payoff_or_hook"),
        })
    return beats


def _safe_stage1_wire_shape(payload: dict[str, Any]) -> str:
    """Describe schema shape without logging any response values."""
    fixed_slots = [key for key in ("h", "setup", "escalation", "reversal", "end") if key in payload]
    raw_beats = payload.get("beats")
    if fixed_slots:
        raw_beats = [payload.get(key) for key in fixed_slots]
    beat_items = (
        raw_beats if isinstance(raw_beats, list)
        else list(raw_beats.values()) if isinstance(raw_beats, dict)
        else []
    )
    field_names = sorted({
        str(key)
        for item in beat_items if isinstance(item, dict)
        for key in item.keys()
    })
    return (
        f"top_level_keys={sorted(str(key) for key in payload.keys())}; "
        f"fixed_beat_slots={sorted(fixed_slots)}; "
        f"beats_type={type(raw_beats).__name__}; beats_count={len(beat_items)}; "
        f"beat_field_names={field_names}"
    )


def expand_v3_stage1(payload: dict[str, Any]) -> dict[str, Any]:
    """Expand MiniMax's compact stage-1 wire JSON without weakening validation."""
    if any(payload.get(key) for key in _V3_STAGE1_KEYS):
        return copy.deepcopy(payload)
    sb = payload.get("sb") if isinstance(payload.get("sb"), dict) else {}
    vb = payload.get("vb") if isinstance(payload.get("vb"), dict) else {}
    if isinstance(payload.get("c1"), dict) or isinstance(payload.get("c2"), dict):
        raw_characters = [payload.get("c1"), payload.get("c2")]
    else:
        raw_characters = payload.get("cb") if isinstance(payload.get("cb"), list) else []
    characters = []
    for item in raw_characters:
        if not isinstance(item, dict):
            continue
        voice = item.get("v") if isinstance(item.get("v"), dict) else {}
        ascii_identity_tags = [
            _text(tag) for tag in item.get("it") or []
            if _text(tag) and _text(tag).isascii() and _text(tag).isprintable()
        ] if isinstance(item.get("it"), list) else []
        ascii_wardrobe_tags = [
            _text(tag) for tag in item.get("wt") or []
            if _text(tag) and _text(tag).isascii() and _text(tag).isprintable()
        ] if isinstance(item.get("wt"), list) else []
        hair_style = _text(item.get("hs"))
        characters.append({
            "character_id": item.get("id"), "name": item.get("n"),
            "identity_prompt": item.get("desc"),
            "model_identity_tags_en": [
                *([f"{_text(item.get('hc'))} hair"] if _text(item.get("hc")) else []),
                *([hair_style] if hair_style and hair_style.isascii() and hair_style.isprintable() else []),
                *([f"{_text(item.get('ec'))} eyes"] if _text(item.get("ec")) else []),
                *ascii_identity_tags,
            ],
            "model_wardrobe_tags_en": ascii_wardrobe_tags,
            "wardrobe_lock": ascii_wardrobe_tags,
            "voice_profile": {
                "language": voice.get("lang"), "age": voice.get("age"),
                "timbre": voice.get("tone"), "pace": voice.get("pace"),
            },
        })
    raw_scenes = []
    if isinstance(payload.get("s1"), dict):
        raw_scenes.append(payload["s1"])
        raw_scenes.extend(payload.get("sx") if isinstance(payload.get("sx"), list) else [])
    else:
        raw_scenes.extend(payload.get("sc") if isinstance(payload.get("sc"), list) else [])
    scenes = [
        {"scene_id": item.get("id"), "description": item.get("desc"), "model_prompt_en": item.get("mp")}
        for item in raw_scenes
        if isinstance(item, dict)
    ]
    fixed_beats = [payload.get(key) for key in ("h", "setup", "escalation", "reversal", "end")]
    beats = (
        _expand_wire_beats(fixed_beats)
        if any(isinstance(item, dict) for item in fixed_beats)
        else _expand_wire_beats(payload.get("beats"))
    )
    return {
        "title": sb.get("t"),
        "story_bible": {
            "title": sb.get("t"), "logline": sb.get("l"), "synopsis": sb.get("s"),
            "themes": sb.get("th"), "continuity_rules": sb.get("cr"),
        },
        "character_bible": characters,
        "visual_bible": {"style_prompt": vb.get("sp"), "global_negative_prompt": vb.get("neg")},
        "scene_bible": scenes,
        "story_beats": beats,
    }


def validate_v3_stage1(payload: dict[str, Any]) -> list[str]:
    """Hard-validate the compact V3 bible before a second paid call is allowed."""
    errors: list[str] = []
    for key in _V3_STAGE1_KEYS:
        if not payload.get(key):
            errors.append(f"{key} missing or empty")
    characters = payload.get("character_bible") if isinstance(payload.get("character_bible"), list) else []
    character_ids: list[str] = []
    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            errors.append(f"character_bible[{index}] must be an object")
            continue
        character_id = _text(character.get("character_id"))
        if not character_id or not _text(character.get("name")):
            errors.append(f"character_bible[{index}] requires character_id and name")
        character_ids.append(character_id)
        for key in ("model_identity_tags_en", "model_wardrobe_tags_en"):
            tags = character.get(key)
            if not isinstance(tags, list) or not tags or any(
                not str(tag).isascii() or not str(tag).isprintable() for tag in tags
            ):
                errors.append(f"character_bible[{index}].{key} requires non-empty English tags")
        wardrobe_text = " | ".join(
            _text(tag).casefold() for tag in character.get("model_wardrobe_tags_en") or []
        )
        if not re.search(r"\b(?:shoes?|sneakers?|boots?|loafers?|heels?|sandals?)\b", wardrobe_text):
            errors.append(f"character_bible[{index}].model_wardrobe_tags_en requires explicit footwear")
        identity_tags = " | ".join(
            _text(tag).casefold() for tag in character.get("model_identity_tags_en") or []
        )
        if not re.search(r"\b(?:black|brown|white|silver|blonde|green) hair\b", identity_tags):
            errors.append(f"character_bible[{index}].model_identity_tags_en requires one explicit hair color")
        if not re.search(r"\b(?:brown|black|blue|green|gray) eyes\b", identity_tags):
            errors.append(f"character_bible[{index}].model_identity_tags_en requires one explicit eye color")
        voice = character.get("voice_profile") if isinstance(character.get("voice_profile"), dict) else {}
        if not all(_text(voice.get(key)) for key in ("language", "age", "timbre", "pace")):
            errors.append(f"character_bible[{index}].voice_profile incomplete")
    if "" in character_ids or len(character_ids) != len(set(character_ids)):
        errors.append("character_bible character_id values must be non-empty and unique")
    visual = payload.get("visual_bible") if isinstance(payload.get("visual_bible"), dict) else {}
    for field in ("style_prompt", "global_negative_prompt"):
        value = _text(visual.get(field))
        if not value or not value.isascii() or not value.isprintable():
            errors.append(f"visual_bible.{field} requires printable ASCII English")
    scenes = payload.get("scene_bible") if isinstance(payload.get("scene_bible"), list) else []
    scene_ids: list[str] = []
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            errors.append(f"scene_bible[{index}] must be an object")
            continue
        scene_id = _text(scene.get("scene_id"))
        scene_ids.append(scene_id)
        model_prompt = _text(scene.get("model_prompt_en"))
        if not scene_id or not model_prompt or not model_prompt.isascii() or not model_prompt.isprintable():
            errors.append(f"scene_bible[{index}] requires scene_id and English model_prompt_en")
    if "" in scene_ids or len(scene_ids) != len(set(scene_ids)):
        errors.append("scene_bible scene_id values must be non-empty and unique")
    beats = payload.get("story_beats") if isinstance(payload.get("story_beats"), list) else []
    roles: set[str] = set()
    beat_ids: list[str] = []
    for index, beat in enumerate(beats):
        if not isinstance(beat, dict):
            errors.append(f"story_beats[{index}] must be an object")
            continue
        beat_ids.append(_text(beat.get("beat_id")))
        roles.add(_text(beat.get("role")).lower())
        if not all(_text(beat.get(key)) for key in ("beat_id", "role", "dramatic_question", "visible_proof", "payoff_or_hook")):
            errors.append(f"story_beats[{index}] incomplete")
    missing_roles = {"hook", "setup", "escalation", "reversal"} - roles
    if missing_roles:
        errors.append(f"story_beats missing roles: {', '.join(sorted(missing_roles))}")
    if not roles.intersection({"cliffhanger", "close"}):
        errors.append("story_beats require cliffhanger or close")
    if "" in beat_ids or len(beat_ids) != len(set(beat_ids)):
        errors.append("story_beats beat_id values must be non-empty and unique")
    return errors


def _compact_dialogue(items: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, list) and len(item) >= 6:
            output.append({
                "speaker_id": item[0], "text": item[1], "start_s": item[2], "end_s": item[3],
                "delivery_style": item[4], "max_chars": item[5],
            })
        elif isinstance(item, dict):
            output.append(copy.deepcopy(item))
    return output


def _compact_audio(items: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, list) and len(item) >= 4:
            output.append({"type": item[0], "prompt": item[1], "start_s": item[2], "end_s": item[3]})
        elif isinstance(item, dict):
            output.append(copy.deepcopy(item))
    return output


def _visible_ids_with_action_actor(shot: dict[str, Any]) -> list[str]:
    """Normalize the only cross-field relation JSON Schema cannot express.

    A physical action's actor is visible by definition. MiniMax forced-tool
    schemas can enum-check both fields independently but cannot require
    ``act.sub`` to be a member of ``c``. Append that same locked ID when the
    model omits it; later Stage-1 fact validation still rejects unknown IDs and
    group-shot/duration rules remain unchanged.
    """
    raw_visible = shot.get("c")
    if not isinstance(raw_visible, list):
        raw_visible = shot.get("visible_character_ids")
    if not isinstance(raw_visible, list):
        raw_visible = shot.get("character_ids")
    visible = [
        _text(value) for value in raw_visible or []
        if _text(value)
    ] if isinstance(raw_visible, list) else []
    action = shot.get("act") if isinstance(shot.get("act"), dict) else {}
    # Tool schemas guide, but do not guarantee, that every provider keeps the
    # compact aliases.  The canonical spelling has exactly the same meaning.
    for actor_key in ("sub", "actor_id"):
        actor = _text(action.get(actor_key))
        if actor and actor not in visible:
            visible.append(actor)
    return visible


def _panel_with_visible_action_actor(panel: dict[str, Any]) -> dict[str, Any]:
    """Normalize a full-panel wire without inventing or remapping any ID."""
    normalized = copy.deepcopy(panel)
    raw_visible = normalized.get("character_ids")
    if not isinstance(raw_visible, list):
        raw_visible = normalized.get("visible_character_ids")
    visible = [
        _text(value) for value in raw_visible or [] if _text(value)
    ] if isinstance(raw_visible, list) else []
    for source_key in ("action_spec", "action_components", "act"):
        source = normalized.get(source_key)
        if not isinstance(source, dict):
            continue
        for actor_key in ("actor_id", "sub"):
            actor = _text(source.get(actor_key))
            if actor and actor not in visible:
                visible.append(actor)
    normalized["character_ids"] = visible
    return normalized


def _stage2_slots(payload: dict[str, Any], shot_count: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Return exact fixed slots and raw-wire errors without filling a missing shot."""
    expected = [f"p{index:02d}" for index in range(1, int(shot_count) + 1)]
    errors: list[str] = []
    unexpected = sorted(set(payload) - set(expected))
    if unexpected:
        errors.append(f"unexpected stage2 slots: {unexpected}")
    shots: list[dict[str, Any]] = []
    for slot in expected:
        shot = payload.get(slot)
        if isinstance(shot, str):
            # Some Anthropic-compatible providers serialize each forced-tool
            # property a second time.  Unwrap only when it is still exactly
            # one JSON object; all ordinary contract validation remains in
            # force below.
            try:
                decoded_shot = _parse_json_lenient(shot)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded_shot = None
            if isinstance(decoded_shot, dict):
                shot = decoded_shot
        if not isinstance(shot, dict):
            errors.append(f"{slot} missing or not an object")
            continue
        shot = copy.deepcopy(shot)

        def wire_object(compact_key: str, canonical_key: str) -> dict[str, Any]:
            value = shot.get(compact_key)
            if isinstance(value, str):
                try:
                    value = _parse_json_lenient(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    value = None
            if not isinstance(value, dict):
                value = shot.get(canonical_key)
                if isinstance(value, str):
                    try:
                        value = _parse_json_lenient(value)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        value = None
            return copy.deepcopy(value) if isinstance(value, dict) else {}

        camera = wire_object("cam", "camera_plan")
        shot["cam"] = {
            "size": _text(camera.get("size") or camera.get("shot_size")),
            "angle": _text(camera.get("angle")),
            "move": _text(camera.get("move") or camera.get("movement")),
            "comp": _text(camera.get("comp") or camera.get("composition")),
        }
        transition = wire_object("tr", "transition")
        shot["tr"] = {
            "type": _text(transition.get("type")),
            "motivation": _text(transition.get("motivation")),
        }
        edit_hint = wire_object("edit", "edit_hint")
        shot["edit"] = {
            "moment": _text(edit_hint.get("moment") or edit_hint.get("preferred_moment")),
            "in": _text(edit_hint.get("in") or edit_hint.get("edit_in_hint")),
            "out": _text(edit_hint.get("out") or edit_hint.get("edit_out_hint")),
        }
        action = copy.deepcopy(shot.get("act")) if isinstance(shot.get("act"), dict) else {}
        for state_key in ("f", "l"):
            nested_state = _text(action.get(state_key))
            top_state = _text(shot.get(state_key))
            if nested_state and top_state and nested_state != top_state:
                errors.append(f"{slot}.{state_key} conflicts with {slot}.act.{state_key}")
            elif nested_state and not top_state:
                shot[state_key] = nested_state
            action.pop(state_key, None)
        shot["act"] = action
        shots.append(shot)
        unexpected_action = sorted(set(action) - {"sub", "code", "obj"})
        if unexpected_action:
            errors.append(f"{slot}.act contains unexpected fields: {unexpected_action}")
        visible_ids = _visible_ids_with_action_actor(shot)
        if not visible_ids:
            errors.append(f"{slot}.c must contain at least one visible character ID")
        elif len(visible_ids) != len(set(_text(value) for value in visible_ids)):
            errors.append(f"{slot}.c must not contain duplicate character IDs")
        try:
            compile_action_spec(
                action,
                visible_character_ids=visible_ids,
                start_state=shot.get("f"),
                end_state=shot.get("l"),
            )
        except ActionContractError as exc:
            errors.append(f"{slot}.act invalid: {exc}")
        if _text(action.get("code")) == "DROP_OBJECT" and not re.search(
            r"\b(?:into|onto|to|in|on)\b", _text(action.get("obj")), re.I
        ):
            errors.append(
                f"{slot}.act DROP_OBJECT obj must name moving object and destination "
                "(example: 'coins into charity box')"
            )
        for object_name, keys in (
            ("cam", ("size", "angle", "move", "comp")),
            ("tr", ("type", "motivation")),
            ("edit", ("moment", "in", "out")),
        ):
            value = shot.get(object_name) if isinstance(shot.get(object_name), dict) else {}
            for key in keys:
                if not _text(value.get(key)):
                    errors.append(f"{slot}.{object_name}.{key} missing or empty")
    return shots, errors


def expand_v3_stage2(payload: dict[str, Any], shot_count: Optional[int] = None) -> list[dict[str, Any]]:
    """Expand the token-efficient stage-2 wire schema into regular V3 panels."""
    if isinstance(payload.get("panels"), list):
        return [
            _panel_with_visible_action_actor(panel)
            for panel in payload["panels"] if isinstance(panel, dict)
        ]
    if shot_count is not None and any(re.fullmatch(r"p\d{2,3}", str(key)) for key in payload):
        compact, _errors = _stage2_slots(payload, shot_count)
    else:
        compact = payload.get("shots")
        if not isinstance(compact, list):
            return []
    panels: list[dict[str, Any]] = []
    previous_id: str | None = None
    previous_state: dict[str, Any] = {}
    for index, shot in enumerate(compact, 1):
        if not isinstance(shot, dict):
            continue
        panel_id = _text(shot.get("id")) or f"ep01_panel{index:02d}"
        raw_camera = shot.get("cam")
        raw_transition = shot.get("tr")
        raw_edit = shot.get("edit")
        camera = raw_camera if isinstance(raw_camera, list) else []
        transition = raw_transition if isinstance(raw_transition, list) else []
        edit = raw_edit if isinstance(raw_edit, list) else []
        suggested_state_in = copy.deepcopy(
            shot.get("si") if isinstance(shot.get("si"), dict) else {}
        )
        state_in = copy.deepcopy(previous_state if previous_id is not None else suggested_state_in)
        state_out = copy.deepcopy(shot.get("so") if isinstance(shot.get("so"), dict) else {})
        if not state_out:
            state_out = {"visible_state": _text(shot.get("l"))}
        first_state = _text(shot.get("f"))
        final_state = _text(shot.get("l"))
        visible_ids = _visible_ids_with_action_actor(shot)
        action_spec = compile_action_spec(
            shot.get("act") if isinstance(shot.get("act"), dict) else {},
            visible_character_ids=visible_ids,
            start_state=first_state,
            end_state=final_state,
        )
        action_components = derived_action_components(action_spec)
        panel = {
            "panel_id": panel_id,
            "name": panel_id,
            "scene_id": _text(shot.get("s")),
            "character_ids": visible_ids,
            "continuity_group": "main",
            "previous_panel_id": previous_id,
            "continuity_state_in": state_in,
            "continuity_state_out": state_out,
            "source_generation_duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
            "edit_duration_seconds": shot.get("d"),
            "shot_role": _text(shot.get("r")),
            "story_beat_id": _text(shot.get("b")),
            "visible_action": action_spec["h3_action_en"],
            "action_code": action_spec["action_code"],
            "action_spec": action_spec,
            "action_components": action_components,
            "first_state": first_state,
            "final_state": final_state,
            "cause": _text(shot.get("why")),
            "next_hook": _text(shot.get("next")),
            "first_frame": first_state,
            "last_frame": final_state,
            "camera_plan": {
                "shot_size": _text(raw_camera.get("size")) if isinstance(raw_camera, dict) else (_text(camera[0]) if len(camera) > 0 else ""),
                "angle": _text(raw_camera.get("angle")) if isinstance(raw_camera, dict) else (_text(camera[1]) if len(camera) > 1 else ""),
                "movement": _text(raw_camera.get("move")) if isinstance(raw_camera, dict) else (_text(camera[2]) if len(camera) > 2 else ""),
                "composition": _text(raw_camera.get("comp")) if isinstance(raw_camera, dict) else (_text(camera[3]) if len(camera) > 3 else ""),
            },
            "transition": {
                "type": _text(raw_transition.get("type")) if isinstance(raw_transition, dict) else (_text(transition[0]) if transition else ""),
                "motivation": _text(raw_transition.get("motivation")) if isinstance(raw_transition, dict) else (_text(transition[1]) if len(transition) > 1 else ""),
            },
            "edit_hint": {
                "preferred_moment": _text(raw_edit.get("moment")) if isinstance(raw_edit, dict) else (_text(edit[0]) if edit else ""),
                "edit_in_hint": _text(raw_edit.get("in")) if isinstance(raw_edit, dict) else (_text(edit[1]) if len(edit) > 1 else ""),
                "edit_out_hint": _text(raw_edit.get("out")) if isinstance(raw_edit, dict) else (_text(edit[2]) if len(edit) > 2 else ""),
            },
            "priority": _text(shot.get("pri")),
            "group_shot_reason": _text(shot.get("g")),
            "spoken_dialogue": _compact_dialogue(shot.get("dlg")),
            "subtitle_timeline": [],
            "on_screen_text": [],
            "audio_cues": _compact_audio(shot.get("aud")),
            "background_music": _text(shot.get("bgm")) or "auto_contextual",
            "ambience": _text(shot.get("amb")) or "auto_contextual",
            "sfx": [],
            "cuts": [{
                "time_range": f"0-{SOURCE_GENERATION_DURATION_SECONDS:g}s",
                "name": _text(shot.get("r")) or f"shot_{index}",
                "intensity": "SMOOTH",
                "shot_description": (
                    f"{first_state}; the visible action is {action_spec['h3_action_en']}; the camera records the causal change "
                    f"until {final_state}, preserving character identity, wardrobe, scene geography and lighting continuity."
                ),
            }],
            "transitions": [],
        }
        if previous_id is not None and suggested_state_in != state_in:
            panel["llm_suggested_continuity_state_in"] = suggested_state_in
        panels.append(panel)
        previous_id = panel_id
        previous_state = state_out
    return panels


def validate_v3_stage2(
    payload: dict[str, Any], stage1: dict[str, Any], shot_count: int, target_seconds: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    raw_slot_errors: list[str] = []
    if any(re.fullmatch(r"p\d{2,3}", str(key)) for key in payload):
        _raw_shots, raw_slot_errors = _stage2_slots(payload, shot_count)
        if raw_slot_errors:
            return [], raw_slot_errors
    panels = expand_v3_stage2(payload, shot_count)
    errors: list[str] = list(raw_slot_errors)
    if len(panels) != int(shot_count):
        errors.append(f"shot_count {len(panels)} must equal requested exact {int(shot_count)}")
    else:
        allocated_durations = allocate_edit_durations(target_seconds, len(panels))
        for panel, allocated in zip(panels, allocated_durations):
            suggested = panel.get("edit_duration_seconds")
            if suggested is not None:
                panel["llm_suggested_edit_duration_seconds"] = suggested
            panel["edit_duration_seconds"] = allocated
            timed_lanes = (
                "spoken_dialogue", "subtitle_timeline",
                "postproduction_on_screen_text", "audio_cues",
            )
            source_clock = 0.0
            try:
                source_clock = max(source_clock, float(suggested or 0))
            except (TypeError, ValueError):
                pass
            for lane_name in timed_lanes:
                for cue in panel.get(lane_name) or []:
                    if not isinstance(cue, dict):
                        continue
                    try:
                        source_clock = max(source_clock, float(cue.get("end_s") or 0))
                    except (TypeError, ValueError):
                        continue
            if source_clock > allocated + 1e-9:
                scale = allocated / source_clock
                panel["timeline_scale_factor"] = scale
                for lane_name in timed_lanes:
                    for cue in panel.get(lane_name) or []:
                        if not isinstance(cue, dict):
                            continue
                        try:
                            cue["start_s"] = round(float(cue.get("start_s")) * scale, 6)
                            cue["end_s"] = round(float(cue.get("end_s")) * scale, 6)
                        except (TypeError, ValueError):
                            continue
    errors.extend(validate_platform_shot_plan(panels, target_seconds))
    character_ids = {_text(item.get("character_id")) for item in stage1.get("character_bible") or [] if isinstance(item, dict)}
    scene_ids = {_text(item.get("scene_id")) for item in stage1.get("scene_bible") or [] if isinstance(item, dict)}
    beat_ids = {_text(item.get("beat_id")) for item in stage1.get("story_beats") or [] if isinstance(item, dict)}
    for index, panel in enumerate(panels):
        if _text(panel.get("scene_id")) not in scene_ids:
            errors.append(f"panel[{index}].scene_id not present in validated stage 1")
        visible_ids = [_text(value) for value in panel.get("character_ids") or []]
        if len(visible_ids) != len(set(visible_ids)):
            errors.append(f"panel[{index}].character_ids must not contain duplicate IDs")
        unknown = set(visible_ids) - character_ids
        if unknown:
            errors.append(f"panel[{index}].character_ids unknown: {sorted(unknown)}")
        if _text(panel.get("story_beat_id")) not in beat_ids:
            errors.append(f"panel[{index}].story_beat_id not present in validated stage 1")
    return panels, errors


# ── M3 / OpenAI-style chat completions call ──────────────────────────────
def _call_m3(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    protocol: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: float = 0.3,
    timeout_seconds: Optional[float] = None,
    tool_name: Optional[str] = None,
    tool_schema: Optional[dict[str, Any]] = None,
) -> str:
    """Call MiniMax using Anthropic Messages by default, or explicit OpenAI compatibility.

    Only ``MiniMax_API_KEY`` is accepted implicitly.  An OpenAI credential is
    never reused for MiniMax. Structured tool calls never fall back to text.
    """
    resolved_key = (api_key or os.environ.get("MiniMax_API_KEY") or M3_API_KEY).strip()
    if not resolved_key:
        raise MissingMiniMaxAPIKey(
            "未配置 MiniMax_API_KEY。实时故事拆分已停止；如需示例，请显式选择 DEMO 模式。"
        )

    import urllib.request
    import urllib.error

    resolved_timeout = minimax_request_timeout_seconds(timeout_seconds)
    config = minimax_configuration_status(base_url=base_url, model=model, protocol=protocol)
    if config["protocol"] == "anthropic":
        requested_budget = DEFAULT_ANTHROPIC_MAX_TOKENS if max_tokens is None else int(max_tokens)
        completion_budget = min(max(1, requested_budget), MAX_ANTHROPIC_MAX_TOKENS)
        req_body: dict[str, Any] = {
            "model": config["model"],
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            "temperature": min(1.0, max(0.01, float(temperature))),
            "max_tokens": completion_budget,
        }
        if tool_name:
            if not isinstance(tool_schema, dict):
                raise ValueError("tool_schema is required for a forced Anthropic tool call")
            req_body["tools"] = [{
                "name": tool_name,
                "description": "Submit the validated structured production contract for this stage.",
                "input_schema": tool_schema,
            }]
            req_body["tool_choice"] = {"type": "tool", "name": tool_name}
        headers = {
            "X-Api-Key": resolved_key,
            "Anthropic-Version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        requested_budget = MAX_M3_COMPLETION_TOKENS if max_tokens is None else int(max_tokens)
        completion_budget = min(max(1, requested_budget), MAX_M3_COMPLETION_TOKENS)
        req_body = {
            "model": config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + (
                        "\nOUTPUT BUDGET: Return minified JSON only. Use terse concrete strings, stable IDs and "
                        "compact arrays; do not repeat bible prose inside panels. Never omit required contract fields."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": completion_budget,
            "reasoning_split": True,
        }
        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }
    req = urllib.request.Request(
        config["endpoint"],
        data=json.dumps(req_body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=resolved_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        if config["protocol"] == "anthropic":
            stop_reason = str(data.get("stop_reason") or "").strip().lower()
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            if stop_reason == "max_tokens":
                output_tokens = usage.get("output_tokens")
                token_note = f", output_tokens={output_tokens}" if output_tokens is not None else ""
                raise MiniMaxOutputTruncated(
                    "MiniMax Anthropic 输出达到 max_tokens 上限"
                    f"（stop_reason=max_tokens{token_note}）；截断结果不会进入合同解析。"
                )
            blocks = data.get("content") if isinstance(data.get("content"), list) else []
            if tool_name:
                matches = [
                    block for block in blocks
                    if isinstance(block, dict)
                    and block.get("type") == "tool_use"
                    and block.get("name") == tool_name
                ]
                if len(matches) != 1:
                    block_types = sorted({
                        str(block.get("type")) for block in blocks if isinstance(block, dict)
                    })
                    raise RuntimeError(
                        f"MiniMax Anthropic 未返回唯一目标 tool_use（expected={tool_name}; "
                        f"matching_count={len(matches)}; block_types={block_types}）；文本不会作为合同回退。"
                    )
                tool_input = matches[0].get("input")
                if not isinstance(tool_input, dict):
                    raise RuntimeError("MiniMax Anthropic tool_use.input 必须是 JSON object")
                return json.dumps(tool_input, ensure_ascii=False, separators=(",", ":"))
            texts = [
                str(block.get("text") or "") for block in blocks
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            content = "".join(texts).strip()
            if not content:
                raise RuntimeError("MiniMax Anthropic response has no text content")
            return content

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"API response missing 'choices': {sorted(data.keys())}")
        choice = data["choices"][0]
        finish_reason = str(choice.get("finish_reason") or "").strip().lower()
        if finish_reason in {"length", "max_tokens", "max_completion_tokens"}:
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            completion_tokens = usage.get("completion_tokens")
            token_note = f"，completion_tokens={completion_tokens}" if completion_tokens is not None else ""
            raise MiniMaxOutputTruncated(
                "MiniMax OpenAI 兼容输出达到 max_completion_tokens 上限"
                f"（finish_reason={finish_reason}{token_note}）；截断正文不会进入 JSON 解析。"
            )
        message = choice["message"]
        content = re.sub(
            r"^\s*<think>[\s\S]*?</think>\s*", "", str(message.get("content") or ""), count=1,
        )
        if not content.strip():
            raise RuntimeError("MiniMax OpenAI response content is empty after reasoning removal")
        return content
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"MiniMax API HTTP {e.code}; response body withheld") from e
    except (TimeoutError, socket.timeout) as e:
        raise MiniMaxRequestTimeout(
            f"MiniMax 请求超过 {resolved_timeout:g} 秒，已停止等待。"
            "本次结果未写入项目、未保存合同；请检查网络或适当调高 "
            "AI_MANGA_MINIMAX_TIMEOUT_SECONDS 后安全重试。系统不会自动再次发起付费请求。"
        ) from e
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
        if isinstance(getattr(e, "reason", None), (TimeoutError, socket.timeout)):
            raise MiniMaxRequestTimeout(
                f"MiniMax 请求超过 {resolved_timeout:g} 秒，已停止等待。"
                "本次结果未写入项目、未保存合同；可以安全重试，系统不会自动重复付费请求。"
            ) from e
        raise RuntimeError(f"M3 API call failed: {e}") from e


# ── Main entry point ──────────────────────────────────────────────────────
def split_story(
    story_text: str,
    *,
    topic: str = "",
    synopsis: str = "",
    target_audience: str = "general audience",
    total_duration_seconds: Optional[float] = None,
    shot_count: Optional[int] = None,
    platform: str = "custom",
    style: str = DEFAULT_STYLE,
    min_panels: int = DEFAULT_MIN_PANELS,
    max_panels: int = DEFAULT_MAX_PANELS,
    use_lora: bool = True,
    lora_strength: float = 1.0,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    language: str = "cn",
    prompt_mode: Optional[str] = None,
    visual_style: str = "premium comic-book animation",
    style_enforcement: str = "",
    background_music: str = "epic_brass",
    ambience: str = "office_quiet",
    sage_mode: str = "auto",
    ref_image_size: str = "match",
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    character_brief: str = "",
    demo_mode: bool = False,
    # --- Compatibility kwargs (absorbed for web_app.py forward-compat) ---
    title: Optional[str] = None,            # accepted from web_app.py, stored as result["title"]
    story: Optional[str] = None,            # accepted as alias for story_text
    ep_id: Optional[str] = None,            # accepted for web_app.py, no-op (orchestrator owns ep_id)
    episodes_path: Optional[object] = None, # accepted for web_app.py, no-op (orchestrator owns episodes.json)
    draft_dir: Optional[object] = None,
    stage1_checkpoint_hash: Optional[str] = None,
    stage1_only: bool = False,
    voice_language: Optional[str] = None,   # accepted as alias for `language`
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    progress_cb: Optional[Callable] = None,
    **kwargs: Any,                            # final absorb-all for any unknown kwarg
) -> dict[str, Any]:
    """Split a story into a versioned, production-ready episode contract.

    Live generation requires a MiniMax key.  Bundled data is available only
    when ``demo_mode=True`` and is marked ``source_mode=DEMO`` throughout.
    """
    if story_text is None and story is not None:
        story_text = story
    if synopsis.strip():
        story_text = synopsis.strip()
    topic = (topic or title or "").strip() or (story_text or "").strip()[:80]
    target_audience = target_audience.strip() or "general audience"
    platform = platform.strip() or "custom"
    if total_duration_seconds is not None:
        total_duration_seconds = float(total_duration_seconds)
        if total_duration_seconds <= 0:
            raise ValueError("total_duration_seconds must be positive")
    resolved_total_duration = float(total_duration_seconds or (float(duration_seconds) * max_panels))
    bounds = shot_count_bounds(resolved_total_duration)
    if shot_count is None:
        shot_count = auto_episode_shot_count(resolved_total_duration)
    shot_count = int(shot_count)
    if not demo_mode and (shot_count < bounds["minimum"] or shot_count > bounds["maximum"]):
        raise ValueError(
            f"shot_count must be {bounds['minimum']}-{bounds['maximum']} for "
            f"{resolved_total_duration:g}s final duration (1.5-4.0s edits)"
        )
    if shot_count > 400:
        raise ValueError("shot_count must not exceed 400")
    min_panels = max_panels = shot_count
    if voice_language is not None:
        lowered_voice = voice_language.lower()
        if lowered_voice.startswith("chi") or "chinese" in lowered_voice:
            language = "cn"
        elif lowered_voice.startswith("ja") or "japanese" in lowered_voice:
            language = "jp"
        else:
            language = "en"
    del episodes_path, kwargs

    if min_panels < 1 or max_panels < min_panels:
        raise ValueError("panel range must satisfy 1 <= min_panels <= max_panels")
    if aspect_ratio not in {"16:9", "9:16", "1:1"}:
        raise ValueError("aspect_ratio must be '16:9', '9:16' or '1:1'")
    if background_music not in MUSIC_PRESETS:
        raise ValueError(f"unknown background_music: {background_music}")
    if ambience not in AMBIENCE_PRESETS:
        raise ValueError(f"unknown ambience: {ambience}")
    if ref_image_size not in {"match", "max"}:
        raise ValueError("ref_image_size must be 'match' or 'max'")
    resolved_prompt_mode = (prompt_mode or style or "comic").strip().lower()
    if resolved_prompt_mode not in {"comic", "cinematic"}:
        raise ValueError("prompt_mode must be 'comic' or 'cinematic'")
    resolved_voice_language = voice_language or ("Chinese" if language == "cn" else "English")
    creative_brief = {
        "topic": topic,
        "synopsis": (story_text or "").strip(),
        "visual_style": style_enforcement or visual_style,
        "target_audience": target_audience,
        "total_duration_seconds": resolved_total_duration,
        "shot_count": int(shot_count or max_panels),
        "shot_plan_version": SHOT_PLAN_VERSION,
        "language": resolved_voice_language,
        "platform": platform,
        "aspect_ratio": aspect_ratio,
    }
    settings = {
        "prompt_mode": resolved_prompt_mode,
        "visual_style": visual_style,
        "style_enforcement": style_enforcement or visual_style,
        "aspect_ratio": aspect_ratio,
        "duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
        "source_generation_duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
        "target_edit_duration_seconds": resolved_total_duration,
        "shot_plan_version": SHOT_PLAN_VERSION,
        "use_lora": bool(use_lora),
        "lora_strength": float(lora_strength),
        "sage_mode": sage_mode,
        "ref_image_size": ref_image_size,
        "background_music": background_music,
        "ambience": ambience,
        "voice_language": resolved_voice_language,
        "platform": platform,
        "total_duration_seconds": resolved_total_duration,
        "shot_count": int(shot_count or max_panels),
        "creative_brief": creative_brief,
    }

    if demo_mode:
        demo_story = "Bundled DEMO: a freckled child superhero confronts a giant mech-kaiju on a rooftop."
        parsed = copy.deepcopy(COMIC_EXAMPLE_HERO_KAIJU)
        parsed["title"] = parsed.get("title") or "DEMO — Rooftop Hero"
        parsed["demo_original_request_ignored"] = bool((story_text or "").strip())
        demo_settings = copy.deepcopy(settings)
        # The bundled one-shot legacy sample is never represented as the
        # user's requested platform edit.  It remains an explicitly labelled
        # DEMO until a dedicated modern demo contract is supplied.
        demo_settings.pop("shot_plan_version", None)
        demo_settings.pop("target_edit_duration_seconds", None)
        demo_settings["creative_brief"] = {
            **creative_brief,
            "topic": "Bundled rooftop hero demo",
            "synopsis": demo_story,
        }
        result = enrich_episode_contract(
            parsed,
            story_text=demo_story,
            source_mode="DEMO",
            settings=demo_settings,
        )
        result["demo_notice"] = (
            "DEMO DATA — 固定内置样例，未根据输入故事生成；不得作为用户故事拆分结果。"
        )
        return result

    if not story_text or not story_text.strip():
        raise ValueError("story_text (or legacy 'story') must be non-empty")

    explicit_character_count = explicit_requested_character_count(
        "\n".join(filter(None, [topic, story_text, character_brief]))
    )
    identity_hints = explicit_identity_equivalence_hints(
        "\n".join(filter(None, [story_text, character_brief]))
    )
    single_scene = explicit_single_scene("\n".join(filter(None, [topic, story_text, character_brief])))
    system_prompt = _stage1_system_prompt(
        language,
        requested_character_count=explicit_character_count,
        identity_equivalence_hints=identity_hints,
        single_scene=single_scene,
    )
    request_payload = {
        "requested_title": title or "",
        "creative_brief": creative_brief,
        "panel_range": {"min": min_panels, "max": max_panels, "exact": shot_count},
        "render_settings": settings,
        "character_brief_from_user": character_brief,
        "requested_character_count": explicit_character_count,
        "identity_equivalence_hints": identity_hints,
        "single_scene": single_scene,
    }
    checkpoint_settings = {
        key: copy.deepcopy(value)
        for key, value in request_payload.items()
        if key != "creative_brief"
    }
    resolved_provider = minimax_configuration_status(model=model)
    checkpoint: Optional[dict[str, Any]] = None
    if stage1_checkpoint_hash:
        if not ep_id:
            raise ValueError("ep_id is required to resume a stage1 checkpoint")
        checkpoint = load_stage1_checkpoint(
            ep_id, stage1_checkpoint_hash,
            creative_brief=creative_brief,
            settings=checkpoint_settings,
            protocol=resolved_provider["protocol"],
            model=resolved_provider["model"],
            draft_dir=draft_dir,
        )
        resumed_stage1 = copy.deepcopy(checkpoint["validated_stage1"])

        def _call_stage1(*_args: Any, **_kwargs: Any) -> str:
            return json.dumps(resumed_stage1, ensure_ascii=False)
    else:
        _call_stage1 = _call_m3
    user_prompt = (
        "STAGE 1/2. Return only compact bibles and story beats from this exact brief; "
        "do not emit panels:\n"
        + json.dumps(request_payload, ensure_ascii=False, separators=(",", ":"))
    )
    if progress_cb:
        progress_cb("stage1", "阶段 1/2：生成并校验故事、人物、场景与节拍圣经（第 1 次计划调用）")
    try:
        raw = _call_stage1(
            system_prompt, user_prompt, api_key=api_key, model=model,
            tool_name="submit_v3_stage1",
            tool_schema=_v3_stage1_tool_schema(explicit_character_count, single_scene),
        )
    except MissingMiniMaxAPIKey:
        raise
    except Exception as exc:
        _stage_error(1, 1, f"MiniMax 调用失败：{exc}", exc)
    stage1_wire = _parse_stage_response(raw, 1)
    stage1_shape = _safe_stage1_wire_shape(stage1_wire)
    stage1 = expand_v3_stage1(stage1_wire)
    stage1_errors = validate_v3_stage1(stage1)
    if stage1_errors:
        _stage_error(
            1, 1,
            "圣经合同硬校验失败：" + " | ".join(stage1_errors)
            + f"；安全结构诊断：{stage1_shape}",
        )
    if checkpoint is None and ep_id:
        checkpoint = save_stage1_checkpoint(
            ep_id, stage1,
            creative_brief=creative_brief,
            settings=checkpoint_settings,
            protocol=resolved_provider["protocol"],
            model=resolved_provider["model"],
            draft_dir=draft_dir,
        )
    if stage1_only:
        if checkpoint is None:
            raise ValueError("ep_id is required when stage1_only=True")
        return checkpoint
    if progress_cb:
        progress_cb("stage2", f"阶段 2/2：基于已锁定圣经生成 exact {shot_count} 镜短镜合同（第 2 次计划调用）")
    stage2_request = {
        "exact_shot_count": shot_count,
        "exact_total_edit_seconds": resolved_total_duration,
        "immutable_stage1": {key: stage1.get(key) for key in _V3_STAGE1_KEYS},
        "render_settings": {
            "aspect_ratio": aspect_ratio,
            "language": resolved_voice_language,
            "platform": platform,
        },
    }
    if checkpoint is not None:
        checkpoint = record_stage2_status(
            ep_id, checkpoint["checkpoint_sha256"],
            status="running", draft_dir=draft_dir,
        )
    try:
        stage2_raw = _call_m3(
            _stage2_system_prompt(language, shot_count, resolved_total_duration),
            json.dumps(stage2_request, ensure_ascii=False, separators=(",", ":")),
            api_key=api_key,
            model=model,
            temperature=0.05,
            tool_name="submit_v3_stage2",
            tool_schema=_v3_stage2_tool_schema(
                shot_count,
                character_ids=[
                    _text(item.get("character_id"))
                    for item in stage1.get("character_bible") or [] if isinstance(item, dict)
                ],
                scene_ids=[
                    _text(item.get("scene_id"))
                    for item in stage1.get("scene_bible") or [] if isinstance(item, dict)
                ],
                beat_ids=[
                    _text(item.get("beat_id"))
                    for item in stage1.get("story_beats") or [] if isinstance(item, dict)
                ],
            ),
        )
    except MissingMiniMaxAPIKey:
        if checkpoint is not None:
            record_stage2_status(
                ep_id, checkpoint["checkpoint_sha256"], status="failed",
                error_code="stage2_key_missing", draft_dir=draft_dir,
            )
        raise
    except Exception as exc:
        if checkpoint is not None:
            record_stage2_status(
                ep_id, checkpoint["checkpoint_sha256"], status="failed",
                error_code="stage2_call_failed", draft_dir=draft_dir,
            )
        _stage_error(2, 2, f"MiniMax 调用失败：{exc}", exc)
    try:
        stage2 = _parse_stage_response(stage2_raw, 2)
    except MiniMaxGenerationStageError:
        if checkpoint is not None:
            record_stage2_status(
                ep_id, checkpoint["checkpoint_sha256"], status="failed",
                error_code="stage2_parse_failed", draft_dir=draft_dir,
            )
        raise
    try:
        stage2_panels, stage2_errors = validate_v3_stage2(
            stage2, stage1, shot_count, resolved_total_duration,
        )
    except Exception as exc:
        if checkpoint is not None:
            record_stage2_status(
                ep_id, checkpoint["checkpoint_sha256"], status="failed",
                error_code="stage2_contract_compile_failed", draft_dir=draft_dir,
            )
        _stage_error(2, 2, f"镜头合同编译失败：{type(exc).__name__}: {exc}", exc)
    if stage2_errors:
        if checkpoint is not None:
            record_stage2_status(
                ep_id, checkpoint["checkpoint_sha256"], status="failed",
                error_code="stage2_contract_invalid", draft_dir=draft_dir,
            )
        _stage_error(2, 2, "镜头合同硬校验失败：" + " | ".join(stage2_errors))
    parsed = copy.deepcopy(stage1)
    parsed["panels"] = stage2_panels

    def fail_final_contract(error_code: str) -> None:
        if checkpoint is not None:
            record_stage2_status(
                ep_id, checkpoint["checkpoint_sha256"], status="failed",
                error_code=error_code, draft_dir=draft_dir,
            )

    missing_sections = [
        key for key in ("story_bible", "character_bible", "scene_bible", "panels")
        if not parsed.get(key)
    ]
    if missing_sections:
        fail_final_contract("stage2_final_sections_missing")
        _stage_error(2, 2, f"完整合同缺少 V3 sections: {', '.join(missing_sections)}")
    panels = parsed.get("panels", [])
    if not isinstance(panels, list) or not panels:
        fail_final_contract("stage2_final_panels_missing")
        _stage_error(2, 2, "完整合同没有非空 panels array")
    n = len(panels)
    if n < min_panels or n > max_panels:
        fail_final_contract("stage2_final_shot_count_invalid")
        _stage_error(2, 2, f"镜头数 {n} 不等于要求 [{min_panels},{max_panels}]")
    raw_plan_errors = validate_platform_shot_plan(panels, resolved_total_duration)
    if raw_plan_errors:
        fail_final_contract("stage2_final_shot_plan_invalid")
        _stage_error(2, 2, "平台短剧镜头计划失败：" + " | ".join(raw_plan_errors))

    try:
        result = enrich_episode_contract(
            parsed,
            story_text=story_text,
            source_mode="LIVE",
            settings=settings,
        )
        errors = validate_episode_contract(result)
    except Exception as exc:
        fail_final_contract("stage2_final_contract_compile_failed")
        _stage_error(2, 2, f"完整 V3 合同编译异常：{type(exc).__name__}: {exc}", exc)
    if errors:
        fail_final_contract("stage2_final_contract_invalid")
        _stage_error(2, 2, "完整 V3 合同编译后校验失败：" + " | ".join(errors))
    result["quality_warnings"] = []
    result["schema_version"] = PROMPT_SCHEMA_VERSION
    result["generation_plan"] = {
        "kind": "v3_two_stage",
        "planned_calls": 2,
        "completed_calls": 2,
        "stages": ["bibles_and_beats", "exact_shots_and_prompt_contracts"],
    }
    if checkpoint is not None:
        checkpoint = record_stage2_status(
            ep_id, checkpoint["checkpoint_sha256"],
            status="completed", draft_dir=draft_dir,
        )
        result["generation_plan"]["stage1_checkpoint"] = {
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_path": checkpoint["checkpoint_path"],
            "registration_status": "unregistered",
            "approval_status": "not_approved",
        }
    if progress_cb:
        progress_cb("complete", f"两阶段 V3 合同完成：{len(result['character_bible'])} 人 / {n} 镜")
    return result


def split_story_stage1(
    story_text: str,
    *,
    ep_id: str,
    draft_dir: Optional[object] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run and persist only validated V3 Stage 1; register no episode."""
    if kwargs.get("stage1_checkpoint_hash"):
        raise ValueError("split_story_stage1 cannot resume an existing checkpoint")
    return split_story(
        story_text, ep_id=ep_id, draft_dir=draft_dir,
        stage1_only=True, **kwargs,
    )


def resume_story_stage2(
    story_text: str,
    *,
    ep_id: str,
    checkpoint_hash: str,
    draft_dir: Optional[object] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Resume only Stage 2 from an exact input-bound Stage-1 checkpoint."""
    return split_story(
        story_text,
        ep_id=ep_id,
        draft_dir=draft_dir,
        stage1_checkpoint_hash=checkpoint_hash,
        **kwargs,
    )


def split_series(
    story_text: Optional[str] = None,
    *,
    topic: str = "",
    synopsis: str = "",
    episode_count: int,
    seconds_per_episode: float,
    shots_per_episode: Optional[int] = None,
    target_audience: str = "general audience",
    platform: str = "custom",
    visual_style: str = "premium serialized animation",
    style_enforcement: str = "",
    aspect_ratio: str = "16:9",
    language: str = "cn",
    voice_language: Optional[str] = None,
    prompt_mode: str = "cinematic",
    use_lora: bool = True,
    lora_strength: float = 1.0,
    sage_mode: str = "auto",
    ref_image_size: str = "match",
    background_music: str = "epic_brass",
    ambience: str = "office_quiet",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    progress_cb: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    """Generate one strict V4 season bible/outline; no render jobs are created."""
    synopsis = (synopsis or story_text or "").strip()
    topic = topic.strip()
    episode_count = int(episode_count)
    seconds_per_episode = float(seconds_per_episode)
    shots_per_episode = int(shots_per_episode) if shots_per_episode not in (None, 0) else None
    if not topic or not synopsis:
        raise ValueError("topic and synopsis are required for series generation")
    if not 2 <= episode_count <= 100:
        raise ValueError("episode_count must be between 2 and 100")
    if not 4 <= seconds_per_episode <= 900:
        raise ValueError("seconds_per_episode must be between 4 and 900")
    if aspect_ratio not in {"16:9", "9:16", "1:1"}:
        raise ValueError("aspect_ratio must be '16:9', '9:16' or '1:1'")
    density = shot_count_bounds(seconds_per_episode)
    if shots_per_episode is not None and not density["minimum"] <= shots_per_episode <= density["maximum"]:
        raise ValueError(
            "shots_per_episode cannot satisfy 1.5-4.0 second edits; "
            f"use {density['minimum']}-{density['maximum']}"
        )
    if background_music not in MUSIC_PRESETS or ambience not in AMBIENCE_PRESETS:
        raise ValueError("unknown music or ambience preset")
    resolved_voice = voice_language or ("Chinese" if language == "cn" else "Japanese" if language in {"jp", "ja"} else "English")
    brief = {
        "topic": topic,
        "synopsis": synopsis,
        "target_audience": target_audience,
        "episode_count": episode_count,
        "seconds_per_episode": seconds_per_episode,
        "shots_per_episode": shots_per_episode,
        "shot_plan_version": SHOT_PLAN_VERSION,
        "language": resolved_voice,
        "platform": platform,
        "aspect_ratio": aspect_ratio,
        "visual_style": style_enforcement or visual_style,
    }
    settings = {
        "episode_count": episode_count,
        "seconds_per_episode": seconds_per_episode,
        "shots_per_episode": shots_per_episode,
        "shot_plan_version": SHOT_PLAN_VERSION,
        "source_generation_duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
        "creative_brief": brief,
        "target_audience": target_audience,
        "visual_style": visual_style,
        "style_enforcement": style_enforcement or visual_style,
        "aspect_ratio": aspect_ratio,
        "voice_language": resolved_voice,
        "language": language,
        "platform": platform,
        "prompt_mode": prompt_mode,
        "use_lora": bool(use_lora),
        "lora_strength": float(lora_strength),
        "sage_mode": sage_mode,
        "ref_image_size": ref_image_size,
        "background_music": background_music,
        "ambience": ambience,
    }
    if progress_cb:
        progress_cb("series", "正在建立全季故事、人物、世界、场景圣经与连续大纲")
    system_prompt = build_series_system_prompt(
        language, episode_count, seconds_per_episode, shots_per_episode
    )
    user_prompt = json.dumps({"creative_brief": brief}, ensure_ascii=False, indent=2)
    parsed = _parse_json_lenient(
        _call_m3(system_prompt, user_prompt, api_key=api_key, model=model)
    )
    for key in ("series_bible", "shared_character_bible", "world_bible", "shared_scene_bible", "season_outline"):
        if not parsed.get(key):
            raise ValueError(f"LLM series response missing required V4 section: {key}")
    result = normalize_series_contract(parsed, settings=settings)
    errors = validate_series_contract(result)
    if errors:
        raise ValueError("V4 series contract validation failed: " + " | ".join(errors))
    result["quality_warnings"] = []
    if progress_cb:
        progress_cb("series", f"全季 V4 合同完成：{episode_count} 集")
    return result


def _effective_series_characters(series: dict[str, Any], episode_id: str) -> list[dict[str, Any]]:
    characters = copy.deepcopy(series.get("shared_character_bible") or [])
    by_id = {item.get("character_id"): item for item in characters}
    for outline in series.get("season_outline") or []:
        # A change inside the target episode is panel-scoped below; it only
        # becomes the baseline for later episodes after this episode ends.
        if outline.get("episode_id") == episode_id:
            break
        for event in outline.get("wardrobe_change_events") or []:
            character = by_id.get(event.get("character_id"))
            if character is None:
                continue
            character["editorial_wardrobe_description"] = _text(event.get("to"))
            character["wardrobe_prompt"] = _text(event.get("to"))
            character["model_wardrobe_tags_en"] = list(event.get("model_wardrobe_tags_en") or [])
    return characters


def generate_series_episode(
    series: dict[str, Any],
    episode_id: str,
    *,
    instruction: str = "",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Generate/regenerate exactly one V3 episode while locking all V4 facts."""
    series_errors = validate_series_contract(series)
    # Existing V3 episode errors do not prevent regenerating a selected entry;
    # shared/outline errors do.
    structural_errors = [error for error in series_errors if not error.startswith("episode_contracts.")]
    if structural_errors:
        raise ValueError("cannot derive episode from invalid V4 contract: " + " | ".join(structural_errors))
    context = series_episode_context(series, episode_id)
    outline = context["episode_outline"]
    shot_count = int(outline["shot_count"])
    seconds = float(outline["duration_seconds"])
    language = _text((series.get("render_settings") or {}).get("language")) or "cn"
    system_prompt = build_storyboard_system_prompt(language, shot_count, shot_count) + f"""

SERIES OVERRIDE — you are also the season's senior episode writer and continuity supervisor.
Generate only episode {episode_id}. Shared characters, English model tags, voice profiles, world rules,
shared scenes and visual style supplied by the application are immutable. The first panel state_in is the
episode continuity_state_in; the final panel state_out is continuity_state_out. Use exactly {shot_count}
panels whose edit_duration_seconds total exactly {seconds:g} seconds; every source clip is 10.125 seconds.
Execute hook/setup/escalation/reversal/cliffhanger-or-close with one visible action and state change per shot.
Do not add characters, locations, wardrobe changes, time jumps,
knowledge, injuries or props not authorized by the episode outline. Set series_beat_index on every panel.
Leave model_wardrobe_overrides_en empty; the application deterministically applies approved change events.
Return a V3 JSON object only."""
    request = {
        "revision_instruction": instruction.strip() or "Create the strongest production-ready version of this episode.",
        "immutable_series_context": context,
    }
    parsed = _parse_json_lenient(
        _call_m3(system_prompt, json.dumps(request, ensure_ascii=False, indent=2), api_key=api_key, model=model)
    )
    panels = parsed.get("panels") if isinstance(parsed.get("panels"), list) else []
    if len(panels) != shot_count:
        raise ValueError(f"episode {episode_id} must contain exactly {shot_count} panels")
    raw_plan_errors = validate_platform_shot_plan(panels, seconds)
    if raw_plan_errors:
        raise ValueError(
            f"episode {episode_id} platform shot plan failed: " + " | ".join(raw_plan_errors)
        )
    valid_char_ids = {item.get("character_id") for item in series.get("shared_character_bible") or []}
    valid_scene_ids = {item.get("scene_id") for item in series.get("shared_scene_bible") or []}
    for index, panel in enumerate(panels):
        unknown_chars = set(panel.get("character_ids") or []) - valid_char_ids
        if unknown_chars or panel.get("scene_id") not in valid_scene_ids:
            raise ValueError(
                f"episode {episode_id} panel[{index}] references facts outside the shared series bible"
            )
    wardrobe_events = outline.get("wardrobe_change_events") or []
    valid_beat_indexes = {int(beat.get("beat_index") or 0) for beat in outline.get("beats") or []}
    for index, panel in enumerate(panels):
        raw_beat_index = panel.get("series_beat_index")
        if wardrobe_events and raw_beat_index is None:
            raise ValueError(
                f"episode {episode_id} panel[{index}] requires series_beat_index for wardrobe continuity"
            )
        beat_index = int(raw_beat_index or min(index + 1, max(valid_beat_indexes or {1})))
        if valid_beat_indexes and beat_index not in valid_beat_indexes:
            raise ValueError(f"episode {episode_id} panel[{index}] references unknown series beat {beat_index}")
        overrides: dict[str, list[str]] = {}
        for event in wardrobe_events:
            if beat_index >= int(event.get("effective_beat") or 1):
                overrides[str(event.get("character_id"))] = list(
                    event.get("model_wardrobe_tags_en") or []
                )
        panel["series_beat_index"] = beat_index
        panel["model_wardrobe_overrides_en"] = overrides
    panels[0]["continuity_state_in"] = copy.deepcopy(outline["continuity_state_in"])
    panels[0]["previous_panel_id"] = None
    panels[-1]["continuity_state_out"] = copy.deepcopy(outline["continuity_state_out"])
    parsed["character_bible"] = _effective_series_characters(series, episode_id)
    parsed["scene_bible"] = copy.deepcopy(series.get("shared_scene_bible") or [])
    parsed["visual_bible"] = copy.deepcopy(series.get("visual_bible") or {})
    parsed["story_beats"] = [{
        "beat_id": f"beat_{_text(beat.get('purpose')).lower()}",
        "role": _text(beat.get("purpose")).lower(),
        "dramatic_question": _text(beat.get("summary")),
        "visible_proof": _text(beat.get("visible_proof")),
        "payoff_or_hook": _text(beat.get("summary")),
    } for beat in outline.get("beats") or []]
    render_settings = copy.deepcopy(series.get("render_settings") or {})
    render_settings.update({
        "duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
        "source_generation_duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
        "target_edit_duration_seconds": seconds,
        "shot_plan_version": SHOT_PLAN_VERSION,
        "total_duration_seconds": seconds,
        "shot_count": shot_count,
        "creative_brief": {
            **copy.deepcopy(series.get("creative_brief") or {}),
            "topic": outline.get("title"),
            "synopsis": outline.get("logline"),
            "total_duration_seconds": seconds,
            "shot_count": shot_count,
        },
    })
    episode = enrich_episode_contract(
        parsed,
        story_text=_text((parsed.get("story_bible") or {}).get("synopsis")) or outline["logline"],
        source_mode="LIVE",
        settings=render_settings,
    )
    episode.update({
        "series_id": (series.get("series_bible") or {}).get("series_id"),
        "series_episode_id": episode_id,
        "series_episode_index": outline["episode_index"],
        "continuity_state_in": copy.deepcopy(outline["continuity_state_in"]),
        "continuity_state_out": copy.deepcopy(outline["continuity_state_out"]),
        "wardrobe_change_events": copy.deepcopy(outline.get("wardrobe_change_events") or []),
        "series_sha256": series.get("series_sha256"),
    })
    validation_errors = validate_episode_contract(episode)
    if validation_errors:
        raise ValueError("derived V3 episode validation failed: " + " | ".join(validation_errors))
    updated = copy.deepcopy(series)
    updated.setdefault("episode_contracts", {})[episode_id] = episode
    updated.setdefault("episode_approvals", {})[episode_id] = False
    return updated


def update_series_outline_episode(
    series: dict[str, Any],
    episode_id: str,
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Edit one outline card while preserving its season boundary and shared facts."""
    if not isinstance(replacement, dict):
        raise TypeError("replacement must be an object")
    updated = copy.deepcopy(series)
    for index, current in enumerate(updated.get("season_outline") or []):
        if current.get("episode_id") != episode_id:
            continue
        candidate = copy.deepcopy(replacement)
        for locked in (
            "episode_id", "episode_index", "duration_seconds", "shot_count",
            "continuity_state_in", "continuity_state_out", "wardrobe_change_events", "time_jump_event",
        ):
            candidate[locked] = copy.deepcopy(current.get(locked))
        updated["season_outline"][index] = candidate
        updated.setdefault("episode_contracts", {}).pop(episode_id, None)
        updated.setdefault("episode_approvals", {})[episode_id] = False
        errors = validate_series_contract(updated)
        if errors:
            raise ValueError("outline edit violates V4 contract: " + " | ".join(errors))
        return updated
    raise KeyError(f"unknown series episode: {episode_id}")


def update_contract_item(
    episode: dict[str, Any],
    item_type: str,
    item_id: str,
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Replace one creative item, rebuild prompts, and invalidate affected approvals."""
    if not isinstance(replacement, dict):
        raise TypeError("replacement must be an object")
    updated = copy.deepcopy(episode)
    item_type = item_type.strip().lower()
    approval = copy.deepcopy(updated.get("approval_state") or {})
    creative = approval.setdefault("creative", {"story": False, "characters": False, "storyboard": False})
    assets = approval.setdefault("assets", {"character_ids": [], "scene_ids": []})

    if item_type == "story":
        updated["story_bible"] = copy.deepcopy(replacement)
        creative.update({"story": False, "characters": False, "storyboard": False})
        assets.update({"character_ids": [], "scene_ids": []})
    elif item_type == "visual":
        updated["visual_bible"] = copy.deepcopy(replacement)
        creative.update({"characters": False, "storyboard": False})
        assets.update({"character_ids": [], "scene_ids": []})
    else:
        collection_key, id_key = {
            "character": ("character_bible", "character_id"),
            "scene": ("scene_bible", "scene_id"),
            "panel": ("panels", "panel_id"),
        }.get(item_type, (None, None))
        if not collection_key:
            raise ValueError("item_type must be story, visual, character, scene or panel")
        items = updated.get(collection_key) or []
        found = False
        for index, item in enumerate(items):
            if _text(item.get(id_key)) == item_id:
                candidate = copy.deepcopy(replacement)
                candidate[id_key] = item_id
                # JSON editors own creative facts only. Durable runtime
                # evidence stays backend-owned and cannot be overwritten (or
                # resurrected from a stale browser widget).
                for runtime_field in (
                    "reference_images", "asset_status", "asset_hash", "content_hash",
                    "asset_manifest", "asset_manifest_path", "asset_approval",
                    "asset_review_status", "asset_rejection_history", "asset_error",
                    "approved", "approved_at", "prompt_id", "error", "jobs", "pipeline",
                    "deliveries", "output_path", "preview_path", "qa", "review",
                ):
                    if runtime_field in item:
                        candidate[runtime_field] = copy.deepcopy(item[runtime_field])
                if item_type == "panel" and "subtitle_timeline" in candidate:
                    candidate["_subtitle_user_edited"] = True
                items[index] = candidate
                found = True
                break
        if not found:
            raise KeyError(f"unknown {item_type} item: {item_id}")
        if item_type == "character":
            creative["characters"] = False
            creative["storyboard"] = False
            assets["character_ids"] = [value for value in assets.get("character_ids", []) if value != item_id]
        elif item_type == "scene":
            creative["storyboard"] = False
            assets["scene_ids"] = [value for value in assets.get("scene_ids", []) if value != item_id]
        else:
            creative["storyboard"] = False

    updated["approval_state"] = approval
    brief = updated.get("creative_brief") if isinstance(updated.get("creative_brief"), dict) else {}
    story_text = _text((updated.get("story_bible") or {}).get("synopsis")) or _text(brief.get("synopsis"))
    return enrich_episode_contract(
        updated,
        story_text=story_text,
        source_mode=_text(updated.get("source_mode")) or "LIVE",
        settings=copy.deepcopy(updated.get("render_settings") or {}),
    )


def regenerate_contract_item(
    episode: dict[str, Any],
    item_type: str,
    item_id: str,
    instruction: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Use MiniMax to regenerate exactly one item while preserving stable IDs."""
    item_type = item_type.strip().lower()
    if item_type == "story":
        current = episode.get("story_bible") or {}
    else:
        collection_key, id_key = {
            "character": ("character_bible", "character_id"),
            "scene": ("scene_bible", "scene_id"),
            "panel": ("panels", "panel_id"),
        }.get(item_type, (None, None))
        if not collection_key:
            raise ValueError("item_type must be story, character, scene or panel")
        current = next(
            (item for item in episode.get(collection_key, []) if _text(item.get(id_key)) == item_id),
            None,
        )
        if current is None:
            raise KeyError(f"unknown {item_type} item: {item_id}")
    runtime_only_fields = {
        "reference_images", "asset_status", "asset_hash", "content_hash", "asset_manifest",
        "asset_manifest_path", "asset_approval", "asset_review_status", "asset_rejection_history",
        "approved", "approved_at", "prompt_id", "error", "jobs", "pipeline", "deliveries",
    }
    creative_current = {
        key: copy.deepcopy(value) for key, value in current.items()
        if key not in runtime_only_fields
    }
    system_prompt = f"""You are a top-tier screenwriter, model-specific image prompt engineer and
MiniMax H3 prompt master revising exactly one
{item_type} object in an approved AI-animation V3 contract. Return strict JSON in the form
{{"item": {{...}}}} only. Preserve the existing stable ID {item_id or 'story'} and every fact not targeted
by the revision. Character revisions require a complete voice_profile. Panel revisions require valid
character/scene references, first/last frames, continuity links/states, spoken_dialogue and audio_cues.
Character revisions also require audience-language editorial descriptions plus complete English
model_identity_tags_en/model_wardrobe_tags_en; scene revisions require English model_prompt_en.
Set panel subtitle_timeline and on_screen_text to []; the application derives subtitles from approved
spoken dialogue and H3 must render no visible text.

For character items, rewrite model_identity_tags_en and model_wardrobe_tags_en as concise English tags for the
configured still-image checkpoint. Encode one canonical subject, one outfit, one full-body studio composition;
do not add a character sheet, multiple views, a second person or background garment display. Use the supplied
failure instruction as diagnosis, not as prose to copy into the image prompt.
You MUST materially rewrite the creative fields targeted by the instruction. Returning the current item unchanged
is an error. Never return generated-asset paths, approval state, rejection history, worker state or other runtime data.

For scene items, produce an environment-only English model_prompt_en with one coherent camera view and no people,
signage, collage, split view or repeated product display.

For panel items:
{H3_PROMPT_MASTER_RULES}"""
    context = {
        "instruction": instruction.strip() or "Improve clarity and production readiness without changing intent.",
        "current_item": creative_current,
        "creative_brief": episode.get("creative_brief") or {},
        "story_bible": episode.get("story_bible") or {},
        "valid_character_ids": [item.get("character_id") for item in episode.get("character_bible", [])],
        "valid_scene_ids": [item.get("scene_id") for item in episode.get("scene_bible", [])],
    }
    raw = _call_m3(system_prompt, json.dumps(context, ensure_ascii=False, indent=2), api_key=api_key, model=model)
    parsed = _parse_json_lenient(raw)
    replacement = parsed.get("item")
    if not isinstance(replacement, dict):
        raise ValueError("MiniMax item regeneration did not return an item object")
    replacement = {
        key: value for key, value in replacement.items()
        if key not in runtime_only_fields
    }
    if replacement == creative_current:
        raise ValueError("MiniMax item regeneration returned the item unchanged")
    targeted_fields = {
        "character": ("model_identity_tags_en", "model_wardrobe_tags_en", "negative_prompt"),
        "scene": ("model_prompt_en", "positive_prompt", "negative_prompt"),
        "panel": ("positive_prompt", "negative_prompt", "cuts", "camera_movement"),
    }.get(item_type, ())
    if targeted_fields and not any(
        replacement.get(field) != creative_current.get(field) for field in targeted_fields
    ):
        raise ValueError(
            f"MiniMax {item_type} regeneration did not change any model-facing prompt field"
        )
    return update_contract_item(episode, item_type, item_id, replacement)


# ── CLI for manual testing ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python story_splitter.py <story.txt> [--en]")
        print("   or: python story_splitter.py --example  (output the bundled example as JSON)")
        sys.exit(0)
    if sys.argv[1] == "--example":
        print(json.dumps(COMIC_EXAMPLE_HERO_KAIJU, ensure_ascii=False, indent=2))
        sys.exit(0)
    story = open(sys.argv[1], encoding="utf-8").read()
    lang = "en" if "--en" in sys.argv else "cn"
    result = split_story(story, language=lang)
    print(json.dumps(result, ensure_ascii=False, indent=2))
