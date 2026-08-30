# -*- coding: utf-8 -*-
"""Versioned prompt contracts for the AI manga production pipeline.

This module is intentionally pure: it does not call an LLM, ComfyUI, or the
filesystem.  It turns an LLM storyboard response into an auditable production
contract consumed by the UI and task/render services.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from typing import Any, Iterable

from action_catalog import (
    ActionContractError,
    compile_panel_action,
    derived_action_components,
)


PROMPT_SCHEMA_VERSION = "ai-manga.prompt-package/v3"
SERIES_SCHEMA_VERSION = "ai-manga.series-package/v4"

# Platform-short-drama contract.  H3 produces a fixed source clip; editorial
# selection, subtitles and delivery turn that source into the much shorter
# publishable shot.  Keeping both clocks explicit prevents the old 6x10s
# storyboard from being mistaken for a 60-second edited episode.
SHOT_PLAN_VERSION = "platform-short-drama/v1"
SOURCE_GENERATION_DURATION_SECONDS = 10.125
MIN_EDIT_DURATION_SECONDS = 1.5
MAX_EDIT_DURATION_SECONDS = 4.0
SHOT_ROLES = ("hook", "setup", "escalation", "reversal", "cliffhanger", "close")
GROUP_SHOT_ROLES = {"setup", "close"}


def shot_count_bounds(target_edit_duration_seconds: float) -> dict[str, int]:
    """Return the feasible and recommended shot density for a final edit."""
    target = float(target_edit_duration_seconds)
    if target < len({"hook", "setup", "escalation", "reversal", "close"}) * MIN_EDIT_DURATION_SECONDS:
        raise ValueError("target_edit_duration_seconds must be >= 7.5 for the five-beat platform arc")
    minimum = max(5, math.ceil((target / MAX_EDIT_DURATION_SECONDS) - 1e-9))
    maximum = math.floor((target / MIN_EDIT_DURATION_SECONDS) + 1e-9)
    preferred = min(max(round(target / 3.0), minimum), maximum)
    if 59.5 <= target <= 60.5:
        # Editorial baseline for a 60s platform episode.  The duration bounds
        # make 15 the effective minimum even though the creative range is 14-24.
        preferred = min(max(preferred, max(minimum, 14)), min(maximum, 24))
    return {"minimum": minimum, "maximum": maximum, "preferred": preferred}


def auto_episode_shot_count(target_edit_duration_seconds: float) -> int:
    """Choose a deterministic platform-short-drama shot count (60s -> 20)."""
    return shot_count_bounds(target_edit_duration_seconds)["preferred"]


def allocate_edit_durations(
    target_edit_duration_seconds: float, shot_count: int,
) -> list[float]:
    """Allocate millisecond-exact 1.5-4.0s edit durations across shots."""
    target_ms = int(round(float(target_edit_duration_seconds) * 1000))
    count = int(shot_count)
    if count < 1:
        raise ValueError("shot_count must be positive")
    if not count * 1500 <= target_ms <= count * 4000:
        bounds = shot_count_bounds(target_ms / 1000)
        raise ValueError(
            "shot_count cannot satisfy 1.5-4.0 second edit durations; "
            f"use {bounds['minimum']}-{bounds['maximum']} shots"
        )
    base, remainder = divmod(target_ms, count)
    durations = [base + (1 if index < remainder else 0) for index in range(count)]
    return [value / 1000 for value in durations]


def shot_plan_cost_summary(target_edit_duration_seconds: float, shot_count: int) -> dict[str, Any]:
    """Pure UI/reporting helper for final-vs-generated duration and workload."""
    count = int(shot_count)
    source_total = round(count * SOURCE_GENERATION_DURATION_SECONDS, 3)
    target = float(target_edit_duration_seconds)
    return {
        "shot_count": count,
        "target_edit_duration_seconds": target,
        "source_generation_duration_seconds_per_shot": SOURCE_GENERATION_DURATION_SECONDS,
        "total_source_generation_duration_seconds": source_total,
        "source_to_edit_ratio": round(source_total / target, 3) if target else 0.0,
        "gpu_generation_jobs": count,
    }


_CONCRETE_ACTION_RE = re.compile(
    r"\b(?:grab|reach|turn|open|close|drop|run|step|point|raise|slam|pull|push|"
    r"throw|catch|tear|duck|stand|sit|enter|exit|lift|place|press|knock|walk|"
    r"strike|fall|reveal|hide|hand|pass|look)\w*\b|"
    r"拿|抓|握|伸手|转身|打开|关上|掉落|跑|走|迈|指|举|砸|拉|推|扔|接|撕|"
    r"躲|站起|坐下|进入|离开|抬起|放下|按下|敲|冲|停下|递给|摔|掀开|藏起",
    re.IGNORECASE,
)
_EMPTY_SLOGAN_RE = re.compile(
    r"^(?:友谊|团结|梦想|希望|坚持|相信自己|我们最棒|一起加油|胜利|友情万岁|"
    r"friendship|teamwork|hope|dream|believe|never give up|we can do it|victory)"
    r"[!！。\s]*$",
    re.IGNORECASE,
)

# Language-neutral filmability checks. Keep these UTF-8 literals separate from
# the legacy action regex above so Chinese actions are not dependent on a
# historically mojibake source fragment.
_VISIBLE_PHYSICAL_ACTION_RE = re.compile(
    r"\b(?:grab|reach|turn|open|close|drop|run|step|point|raise|slam|pull|push|"
    r"throw|catch|tear|duck|stand|sit|enter|exit|lift|place|press|knock|walk|"
    r"strike|fall|reveal|hide|hand|pass|look|slide|pick|set|hold|write|draw|"
    r"unlock|lock|remove|attach|cut|pour|wipe|fold|unfold)\w*\b|"
    r"抓住|抬起|举起|伸手|转身|打开|关上|掉落|跑向|走向|迈步|指向|砸向|拉开|推开|推到|"
    r"扔下|接住|撕开|蹲下|站起|坐下|进入|离开|放下|按下|敲击|冲向|停下|递给|摔下|"
    r"揭开|藏起|滑过|拿起|拾起|摆放|握住|写下|画出|解锁|锁上|取下|装上|切开|倒入|擦去|折起|展开|"
    r"摊开|按住|递出|放入|走出|望向|注视|收回|移开",
    re.IGNORECASE,
)
_ABSTRACT_ACTION_RE = re.compile(
    r"\b(?:think|feel|believe|realize|remember|decide|hope|fear|understand|want|"
    r"intend|consider|wonder|know|love|hate)\w*\b|"
    r"思考|觉得|认为|意识到|回忆|决定|希望|害怕|理解|想要|打算|考虑|怀疑|知道|爱上|讨厌",
    re.IGNORECASE,
)
_ACTION_CONNECTOR_RE = re.compile(
    r"\b(?:and then|and|then|before|after|while)\b|然后|接着|随后|继而|一边.+一边|同时",
    re.IGNORECASE,
)
_VISIBLE_RESULT_RE = re.compile(
    r"\b(?:open|closed|raised|lowered|inside|outside|onto|into|across|off|on|"
    r"revealed|broken|lit|dark|empty|full|upright|fallen|visible|hidden)\b|"
    r"打开|关闭|开启|落下|抬高|举到|放到|移到|推至|拉至|递到|进入|离开|露出|显露|"
    r"亮起|熄灭|碎裂|断开|清空|装满|站稳|倒地|藏入|贴上|取下|停在|停住|留在|抵达",
    re.IGNORECASE,
)


def visible_action_evidence(value: Any) -> dict[str, Any]:
    """Classify a filmable action without rewriting or exposing its text."""
    action = _text(value)
    compact_length = len(re.sub(r"\s+", "", action))
    verbs = _VISIBLE_PHYSICAL_ACTION_RE.findall(action)
    has_physical_verb = bool(verbs)
    has_abstract_term = bool(_ABSTRACT_ACTION_RE.search(action))
    has_result = bool(_VISIBLE_RESULT_RE.search(action))
    has_sequence_connector = bool(_ACTION_CONNECTOR_RE.search(action))
    if not action or compact_length < 8:
        category = "too_short"
    elif _EMPTY_SLOGAN_RE.fullmatch(action):
        category = "slogan"
    elif has_abstract_term:
        category = "abstract_or_mental"
    elif not has_physical_verb:
        category = "no_physical_verb"
    elif has_sequence_connector:
        category = "multiple_actions"
    elif not has_result:
        category = "no_visible_result"
    else:
        category = "valid_single_visible_action"
    return {
        "characters": compact_length,
        "category": category,
        "has_physical_verb": has_physical_verb,
        "has_visible_result": has_result,
    }


def validate_platform_shot_plan(
    panels: Iterable[dict[str, Any]], target_edit_duration_seconds: float,
) -> list[str]:
    """Fail closed on non-filmable or editorially invalid short-drama shots."""
    items = [panel for panel in panels if isinstance(panel, dict)]
    errors: list[str] = []
    bounds = shot_count_bounds(target_edit_duration_seconds)
    if not bounds["minimum"] <= len(items) <= bounds["maximum"]:
        errors.append(
            f"shot_count {len(items)} cannot cover {float(target_edit_duration_seconds):g}s "
            f"with 1.5-4.0s edits; expected {bounds['minimum']}-{bounds['maximum']}"
        )
    total = 0.0
    previous_camera_signature = ""
    roles: set[str] = set()
    for index, panel in enumerate(items):
        prefix = f"panel[{index}]"
        try:
            source_duration = float(panel.get("source_generation_duration_seconds"))
        except (TypeError, ValueError):
            source_duration = 0.0
        if abs(source_duration - SOURCE_GENERATION_DURATION_SECONDS) > 0.0001:
            errors.append(f"{prefix}.source_generation_duration_seconds must equal 10.125")
        try:
            edit_duration = float(panel.get("edit_duration_seconds"))
        except (TypeError, ValueError):
            edit_duration = 0.0
        total += edit_duration
        if not MIN_EDIT_DURATION_SECONDS <= edit_duration <= MAX_EDIT_DURATION_SECONDS:
            errors.append(f"{prefix}.edit_duration_seconds must be 1.5-4.0")
        role = _text(panel.get("shot_role")).lower()
        roles.add(role)
        if role not in SHOT_ROLES:
            errors.append(f"{prefix}.shot_role must be one of {', '.join(SHOT_ROLES)}")
        if not _text(panel.get("story_beat_id")):
            errors.append(f"{prefix}.story_beat_id missing")
        has_action_contract = bool(
            isinstance(panel.get("action_spec"), dict)
            or isinstance(panel.get("action_components"), dict)
            or _text(panel.get("action_code"))
        )
        if has_action_contract:
            try:
                compile_panel_action(panel, allow_legacy=True)
            except ActionContractError as exc:
                errors.append(f"{prefix}.action_contract invalid: {exc}")
        else:
            # Pre-action-catalog projects remain readable, but no action code is
            # inferred from prose.  New V3 contracts always use action_spec.
            action = _text(panel.get("visible_action"))
            action_evidence = visible_action_evidence(action)
            if action_evidence["category"] != "valid_single_visible_action":
                errors.append(
                    f"{prefix}.visible_action must describe one concrete visible action "
                    f"(characters={action_evidence['characters']}; category={action_evidence['category']}; "
                    f"has_physical_verb={action_evidence['has_physical_verb']}; "
                    f"has_visible_result={action_evidence['has_visible_result']})"
                )
        first_state = _text(panel.get("first_state"))
        final_state = _text(panel.get("final_state"))
        if not first_state or not final_state or first_state.casefold() == final_state.casefold():
            errors.append(f"{prefix}.first_state/final_state must show a visible state change")
        if not _text(panel.get("cause")) or not _text(panel.get("next_hook")):
            errors.append(f"{prefix}.cause and next_hook are required")
        camera = panel.get("camera_plan") if isinstance(panel.get("camera_plan"), dict) else {}
        if not all(_text(camera.get(key)) for key in ("shot_size", "angle", "movement", "composition")):
            errors.append(f"{prefix}.camera_plan requires shot_size/angle/movement/composition")
        signature = "|".join(_text(camera.get(key)).casefold() for key in ("shot_size", "angle", "composition"))
        if signature and signature == previous_camera_signature:
            errors.append(f"{prefix}.camera_plan repeats the previous composition")
        previous_camera_signature = signature
        transition = panel.get("transition") if isinstance(panel.get("transition"), dict) else {}
        if not all(_text(transition.get(key)) for key in ("type", "motivation")):
            errors.append(f"{prefix}.transition requires type and motivation")
        edit_hint = panel.get("edit_hint") if isinstance(panel.get("edit_hint"), dict) else {}
        if not all(_text(edit_hint.get(key)) for key in ("preferred_moment", "edit_in_hint", "edit_out_hint")):
            errors.append(f"{prefix}.edit_hint requires preferred_moment/edit_in_hint/edit_out_hint")
        if _text(panel.get("priority")).lower() not in {"must_have", "important", "optional"}:
            errors.append(f"{prefix}.priority must be must_have/important/optional")
        visible_count = len(panel.get("character_ids") or [])
        if visible_count > 2:
            if role not in GROUP_SHOT_ROLES:
                errors.append(f"{prefix} dynamic/group shot may show at most 2 visible characters")
            if not _text(panel.get("group_shot_reason")):
                errors.append(f"{prefix}.group_shot_reason required for a group shot")
            if edit_duration > 2.5:
                errors.append(f"{prefix} group establishing/result shot must be <=2.5s")
    if abs(total - float(target_edit_duration_seconds)) > 0.001:
        errors.append(
            f"sum(edit_duration_seconds)={total:g} must equal target {float(target_edit_duration_seconds):g}"
        )
    required = {"hook", "setup", "escalation", "reversal"}
    missing = sorted(required - roles)
    if missing:
        errors.append(f"shot roles missing required structure: {', '.join(missing)}")
    if not roles.intersection({"cliffhanger", "close"}):
        errors.append("shot roles require cliffhanger or close")
    return errors

MODERN_URBAN_STYLE_PROMPT = (
    "anime screencap, hand-drawn 2D cel animation, clean consistent lineart, flat cel shading, "
    "single contemporary TV anime art direction, premium modern Chinese urban animation, "
    "contemporary real-world Chinese city, "
    "grounded everyday production design, natural skin and natural human eye colors, "
    "anatomically coherent mature human proportions, consistent character design, "
    "restrained practical cinematic lighting, realistic modern architecture"
)
MODERN_URBAN_NEGATIVE = (
    "cyberpunk, science-fiction city, futuristic technology, holograms, neon overload, "
    "glowing red eyes, red irises, demonic eyes, doll, toy, figurine, chibi, "
    "super-deformed, plastic skin, porcelain skin, mascot proportions, photorealistic, "
    "realistic photograph, live action, 3d render, CGI, clay model, mixed art styles, "
    "inflated clothing, balloon body, body hidden by oversized garment"
)

DEFAULT_GLOBAL_NEGATIVE = (
    "identity drift, face mismatch, different person, wardrobe change, costume color change, "
    "extra people, missing character, duplicate character, fused bodies, malformed hands, "
    "extra fingers, missing fingers, bad anatomy, inconsistent scale, inconsistent lighting, "
    "background drift, wrong era, wrong location, illegible text, misspelled text, watermark, "
    "logo, signature, low resolution, blurry, compression artifacts"
)

VIEW_PROMPTS = {
    "anchor": (
        "front-facing canonical full body character, standing, head to toe, feet visible, "
        "entire locked outfit and carried accessories visible, eye-level camera, both eyes visible"
    ),
    "正面": "wide shot, standing, front view, full body, head to toe, feet and shoes visible, neutral A-pose, facing camera, symmetrical silhouette",
    "侧面": "wide shot, standing, profile, looking left, full body, head to toe, feet and shoes visible, strict left profile view, neutral pose",
    "背面": "strict back view, full body, same hair shape and exact same wardrobe",
    "全身": "wide shot, long shot, standing, full body, head to toe, neutral A-pose, feet and shoes visible, entire figure centered in frame",
}

HAIR_COLORS = ("black", "brown", "white", "silver", "blonde", "red", "blue", "pink", "green", "purple")
WARDROBE_COLORS = ("black", "blue", "green", "pink", "red", "brown", "gray", "white", "yellow", "purple")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_MODERN_URBAN_RE = re.compile(
    r"现代都市|现代中国城市|modern\s+(?:chinese\s+)?urban|"
    r"contemporary\s+(?:real-world\s+)?(?:chinese\s+)?city",
    flags=re.IGNORECASE,
)
_MODERN_URBAN_CONFLICT_RE = re.compile(
    r"cyberpunk|sci[- ]?fi|science[- ]fiction|futur(?:e|istic)|hologram|"
    r"\bneon\b|glowing\s+red\s+eyes?|red\s+irises?|demonic\s+eyes?|"
    r"\bdolls?\b|\btoys?\b|figurines?|\bchibi\b|super[- ]?deformed|"
    r"plastic\s+skin|porcelain\s+skin|mascot\s+proportions?",
    flags=re.IGNORECASE,
)

_CHINESE_WARDROBE_TAGS = (
    (r"深蓝(?:色)?(?:防水|防雨)?(?:短款|短)?外套", "dark blue rain jacket"),
    (r"藏蓝(?:色)?(?:防水|防雨)?(?:短款|短)?外套", "navy rain jacket"),
    (r"灰(?:色)?连帽衫", "gray hoodie"),
    (r"黑(?:色)?(?:长裤|裤子)", "black pants"),
    (r"黄(?:色)?(?:快递包|邮差包)", "yellow courier bag"),
    (r"黄(?:色)?(?:斜挎包|单肩包)", "yellow crossbody bag"),
    (r"黑(?:色)?(?:靴子|短靴)", "black boots"),
    (r"白(?:色)?衬衫", "white shirt"),
    (r"蓝(?:色)?牛仔裤", "blue jeans"),
)

_CHINESE_SCENE_TAGS = (
    (r"废弃", "abandoned"),
    (r"车站", "train station"),
    (r"站台", "platform"),
    (r"候车室", "waiting room"),
    (r"仓库", "warehouse"),
    (r"屋顶", "rooftop"),
    (r"雨夜|夜雨", "rainy night"),
    (r"冷蓝(?:色)?(?:路灯|灯光|光)", "cool blue lighting"),
    (r"暖金(?:色)?(?:桌灯|灯光|光)", "warm golden lamp light"),
    (r"积水", "rain puddles"),
    (r"倒影", "reflections"),
    (r"斑驳雨棚", "weathered platform canopy"),
    (r"封闭铁门", "locked iron gate"),
    (r"木椅", "wooden benches"),
    (r"停摆时钟", "stopped wall clock"),
    (r"桌灯", "table lamp"),
    (r"窗外雨幕", "rain beyond the windows"),
)


_SCENE_DEVICE_NOUN_RE = re.compile(
    r"\b(?:mobile\s+phones?|smartphones?|phones?|tablets?|devices?)\b|"
    r"(?:手机|平板电脑|平板|设备)",
    re.IGNORECASE,
)
_SCENE_SOCIAL_ROOM_RE = re.compile(
    r"\b(?:living\s+room|gaming\s+room|game\s+room|small\s+room|compact\s+(?:room|interior)|"
    r"room|interior|table|desk)\b|(?:客厅|游戏房|电竞房|房间|室内|桌)",
    re.IGNORECASE,
)
_SCENE_RETAIL_OR_DISPLAY_RE = re.compile(
    r"\b(?:retail|store|shop|showroom|merchandise|inventory|product\s+display|"
    r"display\s+(?:cabinet|case|stand)|glass\s+display\s+case|cabinet|drawer|shelf|"
    r"shelves|shelving|storage\s+rack|product\s+tray|checkout\s+counter|store\s+aisle|"
    r"catalog\s+photography|product\s+photography|flat\s+lay|top[- ]down|overhead|"
    r"bird['’]?s[- ]eye|tabletop\s+close[- ]?up)\b",
    re.IGNORECASE,
)
_SCENE_SOCIAL_DEVICE_NEGATIVE = (
    "retail store, electronics store, phone shop, mobile phone shop, showroom, product showroom, "
    "product display, merchandise display, retail display, display cabinet, glass display case, "
    "cabinet, drawer, open drawer, shelf, shelves, shelving, storage rack, product tray, "
    "phone display stand, checkout counter, store aisle, merchandise, inventory, repeated products, "
    "catalog photography, product photography, flat lay, top-down view, overhead view, "
    "bird's-eye view, tabletop close-up, close-up product shot"
)


def _text(value: Any) -> str:
    """Flatten a scalar/list/dict into stable prompt text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {_text(item)}" for key, item in value.items() if _text(item)
        )
    if isinstance(value, (list, tuple, set)):
        return ", ".join(part for part in (_text(item) for item in value) if part)
    return str(value).strip()


def _tag_list(value: Any) -> list[str]:
    """Normalize model-facing tags without treating editorial prose as a blob."""
    if isinstance(value, (list, tuple, set)):
        parts = [_text(item) for item in value]
    else:
        parts = re.split(r"[,;\n]+", _text(value))
    return _dedupe(part for part in parts if part)


def _modern_urban_style(*values: Any) -> bool:
    return bool(_MODERN_URBAN_RE.search("; ".join(_text(value) for value in values)))


def _modern_urban_safe_tags(value: Any) -> list[str]:
    """Remove model-facing concepts that contradict a grounded modern-city lock."""
    return [tag for tag in _tag_list(value) if not _MODERN_URBAN_CONFLICT_RE.search(tag)]


def _character_safe_negative_tags(value: Any) -> list[str]:
    """Drop environment-only exclusions that would erase the requested person."""
    forbidden = {
        "no people", "no person", "no persons", "no human", "no humans",
        "people", "person", "persons", "human", "humans", "crowd",
        "no character", "no characters", "characters",
    }
    return [tag for tag in _tag_list(value) if tag.casefold() not in forbidden]


def _concrete_modern_wardrobe_tags(
    identity_tags: Iterable[str], wardrobe_tags: Iterable[str]
) -> list[str]:
    """Expand vague LLM clothing labels into one renderable contemporary outfit."""
    identity = ", ".join(identity_tags).casefold()
    source = ", ".join(wardrobe_tags).casefold()
    color = next(
        (value for value in ("black", "blue", "green", "pink", "red", "brown", "gray", "white")
         if re.search(rf"\b{value}\b", source)),
        "navy",
    )
    female = bool(re.search(r"\b(?:1girl|female|woman)\b", identity))
    concrete: list[str] = []
    vague = re.compile(r"\b(?:casual clothes|sportswear|silk clothes|cotton clothes)\b", re.IGNORECASE)
    if "sportswear" in source:
        concrete = [
            f"{color} fitted zip-up track jacket",
            f"{color} straight-leg track pants",
            f"{color} simple sneakers",
        ]
    elif "silk clothes" in source:
        concrete = [
            f"{color} collared long-sleeve blouse",
            "black straight-leg trousers",
            "simple closed-toe low heels" if female else "simple dark sneakers",
        ]
    elif "cotton clothes" in source:
        concrete = [
            f"{color} crew-neck sweatshirt",
            "black straight-leg trousers",
            "simple canvas shoes",
        ]
    elif "casual clothes" in source:
        concrete = [
            f"{color} zip-up casual jacket",
            "white crew-neck T-shirt",
            "black straight-leg trousers",
            "simple sneakers",
        ]
    retained = [tag for tag in wardrobe_tags if not vague.search(tag)]
    return _dedupe([*concrete, *retained]) if concrete else list(wardrobe_tags)


def _wardrobe_color_constraints(wardrobe_tags: Iterable[str]) -> tuple[list[str], list[str]]:
    """Front-load the approved garment color and block conflicting colors."""
    tags = list(wardrobe_tags)
    expected = next((
        color for tag in tags for color in WARDROBE_COLORS
        if re.search(rf"\b{color}\b", tag, re.I)
    ), None)
    if not expected:
        return [], []
    garment_tags = [tag for tag in tags if re.search(r"\b(?:shirt|blouse|sweatshirt|jacket|clothes|sportswear|outfit|dress)\b", tag, re.I)]
    primary = garment_tags[0] if garment_tags else f"{expected} top"
    positive = [
        f"(person wearing {primary}:1.4)",
        *(f"(worn {tag}:1.25)" for tag in garment_tags[:2]),
    ]
    negative = [
        f"{color} clothing, {color} shirt, {color} sweatshirt, {color} jacket"
        for color in WARDROBE_COLORS if color != expected
    ]
    return positive, negative


def _editorial_gender(value: str) -> str | None:
    lowered = value.lower()
    female = bool(
        re.search(r"\b(woman|women|girl|female|lady)\b", lowered)
        or re.search(r"女性|女孩|少女|女人|女士|女青年", value)
    )
    male = bool(
        re.search(r"\b(man|men|boy|male|gentleman)\b", lowered)
        or re.search(r"男性|男孩|少年|男人|男士|男青年|青年男子|成年男子", value)
    )
    if female and male:
        raise ValueError("character editorial identity contains conflicting male/female gender markers")
    return "female" if female else "male" if male else None


def _model_gender(tags: Iterable[str]) -> str | None:
    joined = ", ".join(tags).lower()
    female = bool(re.search(r"(?:^|\W)(?:1girl|female|woman|girl)(?:$|\W)", joined))
    male = bool(re.search(r"(?:^|\W)(?:1boy|male|man|boy)(?:$|\W)", joined))
    if female and male:
        raise ValueError("model_identity_tags_en contains conflicting male/female tags")
    return "female" if female else "male" if male else None


def _identity_fallback_tags(identity: str, signature: str = "") -> tuple[list[str], list[str]]:
    """Conservatively compile common Chinese/English identity facts to SD tags."""
    source = f"{identity}; {signature}".strip("; ")
    lowered = source.lower()
    warnings: list[str] = []
    gender = _editorial_gender(source)
    if gender == "male":
        tags = ["1boy", "male"]
    elif gender == "female":
        tags = ["1girl", "female"]
    else:
        tags = ["1person"]
        warnings.append("gender is not explicit; review model_identity_tags_en before asset approval")

    age_match = re.search(r"(?<!\d)(\d{1,2})\s*(?:岁|[- ]?years?[- ]old)", lowered)
    if age_match:
        age = int(age_match.group(1))
        tags.append(f"{age} years old")
        if age >= 18 and gender:
            tags.append(f"adult {gender}")
    elif re.search(r"青年|年轻|young adult", lowered):
        tags.append("young adult")

    if re.search(r"中国|华人|chinese", source, flags=re.IGNORECASE):
        tags.extend(["Chinese", "East Asian"])

    color_patterns = {
        "black": r"黑发|黑色头发|(?:black(?:\s+[-\w]+){0,2}\s+(?:hair|bob|undercut|braids?|ponytail|bun)|(?:short|long)\s+black\s+(?:hair|bob))\b",
        "brown": r"棕发|棕色头发|(?:brown(?:\s+[-\w]+){0,2}\s+(?:hair|bob|undercut|braids?|ponytail|bun)|(?:short|long)\s+brown\s+(?:hair|bob))\b",
        "white": r"白发|白色头发|(?:white(?:\s+[-\w]+){0,2}\s+(?:hair|bob|undercut|braids?|ponytail|bun)|(?:short|long)\s+white\s+(?:hair|bob))\b",
        "silver": r"银发|银色头发|(?:silver(?:\s+[-\w]+){0,2}\s+(?:hair|bob|undercut|braids?|ponytail|bun)|(?:short|long)\s+silver\s+(?:hair|bob))\b",
        "blonde": r"金发|金色头发|(?:blond(?:e)?(?:\s+[-\w]+){0,2}\s+(?:hair|bob|undercut|braids?|ponytail|bun)|(?:short|long)\s+blond(?:e)?\s+(?:hair|bob))\b",
        "green": r"绿发|绿色头发|(?:green(?:\s+[-\w]+){0,2}\s+(?:hair|bob|undercut|braids?|ponytail|bun)|(?:short|long)\s+green\s+(?:hair|bob))\b",
    }
    hair_colors = [color for color, pattern in color_patterns.items() if re.search(pattern, source, re.I)]
    if len(hair_colors) > 1:
        raise ValueError(f"character editorial identity contains conflicting hair colors: {hair_colors}")
    if hair_colors:
        tags.append(f"{hair_colors[0]} hair")
    else:
        warnings.append("hair color was not mapped; review model_identity_tags_en before asset approval")
    if re.search(r"短(?:黑|棕|白|银|金|绿)?发|短发|short (?:[-\w]+ )?(?:hair|bob)", source, re.I):
        tags.append("short hair")
    elif re.search(r"长(?:黑|棕|白|银|金|绿)?发|长发|long (?:[-\w]+ )?hair", source, re.I):
        tags.append("long hair")

    eye_patterns = {
        "brown eyes": r"棕色?眼(?:睛)?|brown eyes?",
        "black eyes": r"黑色?眼(?:睛)?|black eyes?",
        "blue eyes": r"蓝色?眼(?:睛)?|blue eyes?",
        "green eyes": r"绿色?眼(?:睛)?|green eyes?",
        "gray eyes": r"灰色?眼(?:睛)?|gr[ae]y eyes?",
    }
    tags.extend(tag for tag, pattern in eye_patterns.items() if re.search(pattern, source, re.I))
    if re.search(r"清瘦脸|瘦削脸|slender face|narrow face", source, re.I):
        tags.append("slender face")
    if re.search(r"方脸|square face", source, re.I):
        tags.append("square face")
    if re.search(r"左眉(?:尾|外侧).{0,4}(?:疤|伤痕)|scar.{0,20}left eyebrow", source, re.I):
        tags.append("small scar at outer left eyebrow")
    if re.search(r"左(?:侧)?[^,;]{0,6}银(?:色)?发夹|silver hairpin.{0,12}(?:on (?:the )?)?left", source, re.I):
        tags.append("silver hairpin on left")
    elif re.search(r"银(?:色)?发夹|silver hairpin", source, re.I):
        tags.append("silver hairpin")
    if re.search(r"右眼下.{0,3}(?:痣|小痣)|mole below (?:the )?right eye", source, re.I):
        tags.append("small mole below right eye")
    return _dedupe(tags), warnings


def _wardrobe_fallback_tags(wardrobe: str) -> tuple[list[str], list[str]]:
    if not wardrobe:
        return [], ["wardrobe is empty; define model_wardrobe_tags_en before asset approval"]
    if not _CJK_RE.search(wardrobe):
        return _tag_list(wardrobe), []
    tags = [tag for pattern, tag in _CHINESE_WARDROBE_TAGS if re.search(pattern, wardrobe)]
    warnings = [] if tags else [
        "Chinese wardrobe could not be compiled safely; define model_wardrobe_tags_en before asset approval"
    ]
    return _dedupe(tags), warnings


def _scene_fallback_tags(value: str) -> tuple[list[str], list[str]]:
    if not value:
        return ["empty environment concept art"], [
            "scene description is empty; define model_prompt_en before asset approval"
        ]
    if not _CJK_RE.search(value):
        return _tag_list(value), []
    tags = [tag for pattern, tag in _CHINESE_SCENE_TAGS if re.search(pattern, value)]
    if not tags:
        return ["environment concept art"], [
            "Chinese scene could not be compiled safely; define model_prompt_en before asset approval"
        ]
    return _dedupe(tags), []


def _scene_device_count(value: str) -> int | None:
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    normalized = value
    for word, number in words.items():
        normalized = re.sub(
            rf"\b{word}(?=\s*(?:mobile\s+phones?|smartphones?|phones?|tablets?|devices?)\b)",
            str(number),
            normalized,
            flags=re.IGNORECASE,
        )
    matches = re.finditer(
        r"\b(\d+)(?:\s+[-\w]+){0,4}\s+(?:mobile\s+phones?|smartphones?|phones?|tablets?|devices?)\b|"
        r"([一二两三四五六七八九十])\s*(?:台|部)?\s*(?:手机|平板电脑|平板|设备)",
        normalized,
        re.IGNORECASE,
    )
    chinese_counts = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    counts: list[int] = []
    for match in matches:
        token = str(match.group(1) or match.group(2) or "")
        count = int(token) if token.isdigit() else chinese_counts.get(token)
        if count is not None:
            counts.append(count)
    return max(counts) if counts else None


def _social_device_room_tags(
    tags: list[str], source: str
) -> tuple[list[str], int | None]:
    """Compile multi-phone room prose into a social layout, not a retail array."""
    count = _scene_device_count(source)
    if not (
        count
        and count > 1
        and _SCENE_DEVICE_NOUN_RE.search(source)
        and _SCENE_SOCIAL_ROOM_RE.search(source)
    ):
        return tags, None
    filtered = [
        tag for tag in tags
        if not _SCENE_DEVICE_NOUN_RE.search(tag)
        and not _SCENE_RETAIL_OR_DISPLAY_RE.search(tag)
        and not re.search(r"\b(?:desk|table|chairs?|seats?)\b", tag, re.IGNORECASE)
    ]
    contract = [
        "ordinary contemporary private living room arranged as a casual gaming room",
        "one single ordinary rectangular shared gaming table centered in the room",
        f"exactly {count} separate empty ordinary seats spaced around the table",
        f"exactly {count} separate black-screen smartphones lying flat on the shared tabletop",
        "one phone at each seating place",
        "phones are small secondary personal gaming props within the room",
        "human eye-level wide shot from the room entrance",
        "camera 1.5 meters above the floor",
        "tabletop seen obliquely in natural room perspective",
        "stable horizontal horizon line",
    ]
    return _dedupe([*contract, *filtered]), count


def _danbooru_identity_tags(identity: str) -> tuple[list[str], list[str]]:
    """Return high-priority subject tags and identity-specific negatives."""
    lowered = identity.lower()
    if re.search(r"\b(woman|women|girl|female|lady)\b", lowered) or re.search(r"女性|女孩|少女|女人", identity):
        subject = "1girl"
        count_negatives = ["2girls", "multiple girls", "1boy", "male", "man"]
    elif re.search(r"\b(man|men|boy|male|gentleman)\b", lowered) or re.search(r"男性|男孩|少年|男人", identity):
        subject = "1boy"
        count_negatives = ["2boys", "multiple boys", "1girl", "female", "woman"]
    else:
        subject = "1person"
        count_negatives = ["2people", "multiple people"]

    hair_tags = []
    for color in HAIR_COLORS:
        pattern = (
            rf"\b{re.escape(color)}\s+(?:[-\w]+\s+){{0,3}}"
            r"(?:hair|bob|undercut|pixie|braid|braids|ponytail|bun)\b"
        )
        if re.search(pattern, lowered):
            hair_tags.append(f"{color} hair")
    if "black hair" in hair_tags:
        # Anything V5 otherwise drifted to white/blonde/brown/green under
        # PLUS FACE conditioning; exclude every conflicting common color.
        count_negatives.extend(f"{color} hair" for color in HAIR_COLORS if color != "black")
    return [subject, "solo", "single subject", "same character", *hair_tags], count_negatives


def _danbooru_wardrobe_tags(wardrobe: str) -> list[str]:
    """Compile prose wardrobe locks into SD1.5/Anything-friendly tags.

    The original prose is still retained later in the prompt for auditability;
    these concise tags are placed early because Anything V5 gives earlier
    comma-delimited concepts materially more weight.
    """
    tags: list[str] = []
    for raw_part in wardrobe.split(","):
        part = raw_part.strip().lower()
        if not part or "no wardrobe variation" in part:
            continue
        part = re.sub(r"\b(exact|waterproof|straight|matte|narrow)\b", "", part)
        part = re.sub(r"\btrousers\b", "pants", part)
        part = re.sub(r"\bcrossbody satchel\b", "crossbody bag", part)
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            tags.append(part)
    return tags


def _slug(value: str, prefix: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", (value or "").lower()).strip("_")
    if not cleaned:
        digest = hashlib.sha1((value or prefix).encode("utf-8")).hexdigest()[:8]
        cleaned = digest
    return f"{prefix}_{cleaned}"


def _parse_time_range(value: str) -> tuple[float | None, float | None]:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)", value or "")
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _dedupe(parts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        normalized = (part or "").strip(" ,;\n")
        key = normalized.casefold()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
    return result


_VISIBLE_TEXT_RE = re.compile(
    r"\b(?:"
    r"text|letters?|words?|captions?|subtitles?|speech[- ]?bubbles?|word[- ]?balloons?|"
    r"title[- ]?cards?|logos?|watermarks?|signs?|signage|billboards?|posters?|banners?|"
    r"labels?|typography|written|writing|reads?|says?|display(?:s|ed|ing)?|"
    r"show(?:s|ed|ing)?\s+the\s+words?"
    r")\b",
    flags=re.IGNORECASE,
)


def h3_safe_visual_description(value: Any) -> str:
    """Remove sentence-level visible-text instructions before H3 sees them."""
    source = _text(value)
    if not source:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", source)
    safe = [sentence for sentence in sentences if not _VISIBLE_TEXT_RE.search(sentence)]
    return " ".join(safe).strip()


def normalize_character_bible(raw: Any, fallback_description: str = "") -> list[dict[str, Any]]:
    """Normalize characters and give each one a stable, render-safe ID."""
    characters = raw if isinstance(raw, list) else []
    if not characters and fallback_description.strip():
        characters = [{
            "name": "episode cast",
            "role": "cast",
            "identity_prompt": fallback_description.strip(),
        }]

    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(characters, 1):
        if not isinstance(item, dict):
            item = {"name": f"character {index}", "identity_prompt": _text(item)}
        name = _text(item.get("name") or item.get("display_name") or f"character {index}")
        candidate = _text(item.get("character_id"))
        char_id = candidate if re.fullmatch(r"char_[a-z0-9_]+", candidate) else _slug(name, "char")
        if char_id in used_ids:
            char_id = f"{char_id}_{index:02d}"
        used_ids.add(char_id)

        identity = _text(
            item.get("identity_prompt")
            or item.get("identity_lock")
            or item.get("appearance")
            or item.get("description")
        )
        # A character card can pass through this normalizer more than once
        # (story contract -> task store -> character worker). Preserve the
        # canonical prompt fields instead of silently dropping wardrobe text
        # on the second pass.
        wardrobe = _text(
            item.get("wardrobe_prompt")
            or item.get("wardrobe_lock")
            or item.get("wardrobe")
            or item.get("costume")
        )
        signature = _text(item.get("signature_features") or item.get("continuity_features"))
        editorial_identity = _text(item.get("editorial_identity_description") or identity)
        editorial_wardrobe = _text(item.get("editorial_wardrobe_description") or wardrobe)
        fallback_identity, identity_warnings = _identity_fallback_tags(editorial_identity, signature)
        explicit_identity = _tag_list(item.get("model_identity_tags_en"))
        identity_source = "fallback"
        if explicit_identity and not any(_CJK_RE.search(tag) for tag in explicit_identity):
            editorial_gender = _editorial_gender(editorial_identity)
            explicit_gender = _model_gender(explicit_identity)
            if editorial_gender and explicit_gender and editorial_gender != explicit_gender:
                raise ValueError(
                    f"{char_id}: model_identity_tags_en gender conflicts with editorial identity"
                )
            if editorial_gender and not explicit_gender:
                explicit_identity = [
                    "1boy" if editorial_gender == "male" else "1girl",
                    editorial_gender,
                    *explicit_identity,
                ]
            model_identity_tags = _dedupe([*explicit_identity, *fallback_identity])
            identity_source = "explicit"
            if _model_gender(model_identity_tags):
                identity_warnings = [warning for warning in identity_warnings if "gender" not in warning]
            if any("hair" in tag.lower() for tag in model_identity_tags):
                identity_warnings = [warning for warning in identity_warnings if "hair color" not in warning]
        else:
            model_identity_tags = fallback_identity
            if explicit_identity:
                identity_warnings.append(
                    "model_identity_tags_en contained non-English text and was replaced by conservative fallback tags"
                )

        fallback_wardrobe, wardrobe_warnings = _wardrobe_fallback_tags(editorial_wardrobe)
        explicit_wardrobe = _tag_list(item.get("model_wardrobe_tags_en"))
        wardrobe_source = "fallback"
        if explicit_wardrobe and not any(_CJK_RE.search(tag) for tag in explicit_wardrobe):
            model_wardrobe_tags = _dedupe([*explicit_wardrobe, *fallback_wardrobe])
            wardrobe_warnings = []
            wardrobe_source = "explicit"
        else:
            model_wardrobe_tags = fallback_wardrobe
            if explicit_wardrobe:
                wardrobe_warnings.append(
                    "model_wardrobe_tags_en contained non-English text and was replaced by conservative fallback tags"
                )
        prior_warnings = item.get("model_prompt_warnings")
        if not isinstance(prior_warnings, list):
            prior_warnings = []
        model_prompt_warnings = _dedupe([*prior_warnings, *identity_warnings, *wardrobe_warnings])
        _, automatic_identity_negatives = _danbooru_identity_tags(", ".join(model_identity_tags))
        voice_source = item.get("voice_profile") if isinstance(item.get("voice_profile"), dict) else {}
        if not identity:
            identity = f"Canonical identity for {name}; appearance must be defined before rendering"

        raw_aliases = item.get("aliases") or item.get("character_aliases") or []
        if not isinstance(raw_aliases, (list, tuple, set)):
            raw_aliases = [raw_aliases]
        aliases = _dedupe([
            *[_text(value) for value in raw_aliases],
            _text(item.get("id")),
            candidate if candidate != char_id else "",
            _text(item.get("display_name")),
            name,
        ])

        normalized.append({
            "character_id": char_id,
            "aliases": aliases,
            "name": name,
            "role": _text(item.get("role") or "supporting"),
            "story_function": _text(item.get("story_function")),
            "identity_lock": copy.deepcopy(item.get("identity_lock") or item.get("appearance") or {}),
            "wardrobe_lock": copy.deepcopy(
                item.get("wardrobe_lock")
                or item.get("wardrobe_prompt")
                or item.get("wardrobe")
                or {}
            ),
            "signature_features": signature,
            "identity_prompt": identity,
            "wardrobe_prompt": wardrobe,
            "editorial_identity_description": editorial_identity,
            "editorial_wardrobe_description": editorial_wardrobe,
            "model_identity_tags_en": model_identity_tags,
            "model_wardrobe_tags_en": model_wardrobe_tags,
            "model_tags_source": {"identity": identity_source, "wardrobe": wardrobe_source},
            "model_prompt_warnings": model_prompt_warnings,
        "voice_profile": {
            "language": _text(voice_source.get("language")) or "project language",
            "accent": _text(voice_source.get("accent")) or "neutral",
            "age": _text(voice_source.get("age")) or "match character age",
            "timbre": _text(voice_source.get("timbre")) or "natural, identifiable",
            "pace": _text(voice_source.get("pace")) or "medium",
            "emotion_range": _text(voice_source.get("emotion_range")) or "story appropriate",
            "pronunciation_notes": _text(voice_source.get("pronunciation_notes")),
        },
            "performance_notes": _text(item.get("performance_notes") or item.get("mannerisms")),
            "negative_prompt": ", ".join(_dedupe([
                *automatic_identity_negatives,
                _text(item.get("negative_prompt")) or (
                    f"different identity for {name}, different face, different hair, different age, "
                    "different body type, wardrobe variation, color variation, accessories changed"
                ),
            ])),
            "reference_images": list(item.get("reference_images") or []),
        })
    return normalized


def normalize_visual_bible(raw: Any, *, aspect_ratio: str, visual_style: str) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    style_prompt = _text(source.get("style_prompt") or source.get("style_header") or visual_style)
    style_name = _text(source.get("style_name") or visual_style or "comic")
    modern_urban = _modern_urban_style(visual_style, style_name, style_prompt)
    if modern_urban:
        style_prompt = ", ".join(_dedupe([
            *_tag_list(MODERN_URBAN_STYLE_PROMPT),
            *_modern_urban_safe_tags(style_prompt),
        ]))
    global_negative = _text(source.get("global_negative_prompt")) or DEFAULT_GLOBAL_NEGATIVE
    if modern_urban:
        global_negative = ", ".join(_dedupe([
            *_tag_list(global_negative),
            *_tag_list(MODERN_URBAN_NEGATIVE),
        ]))
    return {
        "style_id": _text(source.get("style_id")) or _slug(visual_style or "comic", "style"),
        "style_name": style_name,
        "style_profile": "modern_urban" if modern_urban else "custom",
        "style_prompt": style_prompt,
        "global_negative_prompt": global_negative,
        "palette": copy.deepcopy(source.get("palette") or []),
        "lighting_rules": _text(source.get("lighting_rules")),
        "lens_language": _text(source.get("lens_language")),
        "composition_rules": _text(source.get("composition_rules")),
        "text_policy": (
            "H3 must render no letters, subtitles, captions, speech bubbles, logos or random visible text. "
            "Subtitles and approved on-screen text are deterministic post-production timelines only."
        ),
        "aspect_ratio": aspect_ratio,
    }


def normalize_scene_bible(raw: Any, panels: list[dict[str, Any]], visual: dict[str, Any]) -> list[dict[str, Any]]:
    scenes = raw if isinstance(raw, list) else []
    if not scenes:
        descriptions: dict[str, str] = {}
        for index, panel in enumerate(panels, 1):
            scene_id = _text(panel.get("scene_id")) or f"scene_{index:02d}"
            descriptions.setdefault(scene_id, _text(panel.get("scene_description")))
        scenes = [
            {"scene_id": scene_id, "name": scene_id, "description": description}
            for scene_id, description in descriptions.items()
        ]

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(scenes, 1):
        if not isinstance(item, dict):
            item = {"description": _text(item)}
        candidate = _text(item.get("scene_id"))
        scene_id = candidate if re.fullmatch(r"scene_[a-z0-9_]+", candidate) else f"scene_{index:02d}"
        description = _text(item.get("description") or item.get("scene_prompt"))
        positive = _text(item.get("positive_prompt")) or description
        editorial_asset = _text(item.get("editorial_asset_description") or item.get("asset_prompt") or positive)
        explicit_model_prompt = _tag_list(item.get("model_prompt_en"))
        fallback_model_prompt, model_warnings = _scene_fallback_tags(
            _text(item.get("description") or item.get("positive_prompt") or item.get("asset_prompt"))
        )
        model_source = "fallback"
        if explicit_model_prompt and not any(_CJK_RE.search(tag) for tag in explicit_model_prompt):
            model_prompt_tags = explicit_model_prompt
            model_warnings = []
            model_source = "explicit"
        else:
            model_prompt_tags = fallback_model_prompt
            if explicit_model_prompt:
                model_warnings.append(
                    "model_prompt_en contained non-English text and was replaced by conservative fallback tags"
                )
        prior_warnings = item.get("model_prompt_warnings")
        if isinstance(prior_warnings, list):
            model_warnings = _dedupe([*prior_warnings, *model_warnings])
        if visual.get("style_profile") == "modern_urban":
            model_prompt_tags = _dedupe([
                *_modern_urban_safe_tags(model_prompt_tags),
                "contemporary Chinese urban environment",
                "grounded real-world architecture",
                "restrained practical lighting",
                "consistent 2D animation background design",
            ])
        scene_model_source = "; ".join(filter(None, [
            description,
            positive,
            editorial_asset,
            ", ".join(model_prompt_tags),
            _text(item.get("continuity_lock") or item.get("continuity")),
        ]))
        model_prompt_tags, social_device_count = _social_device_room_tags(
            model_prompt_tags, scene_model_source
        )
        model_prompt_en = ", ".join(model_prompt_tags)
        negative = _text(item.get("negative_prompt")) or (
            "different location, changed architecture, changed weather, changed time of day, "
            "background drift, prop drift, people, person, character, text, letters, logo, watermark"
        )
        if visual.get("style_profile") == "modern_urban":
            negative = ", ".join(_dedupe([
                *_tag_list(negative),
                *_tag_list(MODERN_URBAN_NEGATIVE),
            ]))
        continuity_lock = copy.deepcopy(item.get("continuity_lock") or item.get("continuity") or {})
        if social_device_count:
            negative = ", ".join(_dedupe([
                *_tag_list(negative),
                *_tag_list(_SCENE_SOCIAL_DEVICE_NEGATIVE),
            ]))
            if not isinstance(continuity_lock, dict):
                continuity_lock = {"legacy_lock": _text(continuity_lock)}
            continuity_lock.update({
                "environment_profile": "social_mobile_gaming_room",
                "layout": "one shared ordinary rectangular gaming table centered in the room",
                "seat_lock": f"exactly {social_device_count} separate empty seats around the table",
                "hero_props": (
                    f"exactly {social_device_count} separate black-screen smartphones lying flat on the tabletop, "
                    "one phone at each seating place"
                ),
                "camera_lock": (
                    "human eye-level wide shot from room entrance, camera 1.5 meters high, "
                    "oblique tabletop perspective, horizontal horizon"
                ),
                "occupancy_lock": "empty environment plate; no people or characters",
            })
        normalized.append({
            "scene_id": scene_id,
            "name": _text(item.get("name")) or scene_id,
            "description": description,
            "positive_prompt": positive,
            "model_prompt_en": model_prompt_en,
            "model_prompt_source": model_source,
            "model_prompt_warnings": model_warnings,
            "environment_profile": (
                "social_mobile_gaming_room" if social_device_count else "generic_environment"
            ),
            "negative_prompt": negative,
            "continuity_lock": continuity_lock,
            "palette": copy.deepcopy(item.get("palette") or visual.get("palette") or []),
            "panel_ids": list(item.get("panel_ids") or []),
            "reference_images": list(item.get("reference_images") or []),
            "editorial_asset_description": editorial_asset,
            # Public asset workers consume this legacy key, so deliberately
            # bind it to the English model lane rather than editorial Chinese.
            "asset_prompt": model_prompt_en,
        })
    return normalized


def build_character_reference_prompt(
    character: dict[str, Any],
    visual_bible: dict[str, Any],
    view: str = "anchor",
) -> dict[str, str]:
    """Build an identity-locked character prompt from a character bible card."""
    char_id = _text(character.get("character_id"))
    editorial_identity = _text(
        character.get("editorial_identity_description") or character.get("identity_prompt")
    )
    editorial_wardrobe = _text(
        character.get("editorial_wardrobe_description")
        or character.get("wardrobe_prompt")
        or character.get("wardrobe_lock")
    )
    signature = _text(character.get("signature_features"))
    if not char_id or not editorial_identity:
        raise ValueError("character_id and editorial identity are required for a reference prompt")
    model_identity_tags = _tag_list(character.get("model_identity_tags_en"))
    if not model_identity_tags or any(_CJK_RE.search(tag) for tag in model_identity_tags):
        model_identity_tags, _ = _identity_fallback_tags(editorial_identity, signature)
    model_wardrobe_tags = _tag_list(character.get("model_wardrobe_tags_en"))
    if not model_wardrobe_tags or any(_CJK_RE.search(tag) for tag in model_wardrobe_tags):
        model_wardrobe_tags, _ = _wardrobe_fallback_tags(editorial_wardrobe)
    model_identity = ", ".join(model_identity_tags)
    model_wardrobe = ", ".join(model_wardrobe_tags)
    style = _text(visual_bible.get("style_prompt") or visual_bible.get("style_name"))
    view_prompt = VIEW_PROMPTS.get(view, view)
    identity_tags, identity_negatives = _danbooru_identity_tags(model_identity)
    wardrobe_tags = _danbooru_wardrobe_tags(model_wardrobe)
    if visual_bible.get("style_profile") == "modern_urban":
        wardrobe_tags = _concrete_modern_wardrobe_tags(model_identity_tags, wardrobe_tags)
    wardrobe_color_positive, wardrobe_color_negative = _wardrobe_color_constraints(wardrobe_tags)
    early_style_tags = (
        [
            "anime screencap",
            "hand-drawn 2D cel animation",
            "clean consistent lineart",
            "flat cel shading",
            "same TV anime art direction",
        ]
        if visual_bible.get("style_profile") == "modern_urban"
        else []
    )
    early_style_negatives = (
        _tag_list(MODERN_URBAN_NEGATIVE)
        if visual_bible.get("style_profile") == "modern_urban"
        else []
    )
    early_reference_tags = (
        [
            "plain light gray studio background",
            "no scenery",
            "fully visible detailed face",
            "defined symmetrical eyes with visible pupils",
            "visible nose and visible mouth",
        ]
        if visual_bible.get("style_profile") == "modern_urban"
        else []
    )
    framing_negatives = [] if view == "anchor" else [
        "close-up", "upper body", "bust portrait", "cropped feet", "feet out of frame"
    ]
    framing_priority = (
        [
            "(solo:1.6)",
            "(single character only:1.55)",
            "(one full-body person centered in frame:1.5)",
            "(head-to-toe standing pose with feet visible:1.35)",
        ]
        if view == "anchor" else []
    )
    positive = ", ".join(_dedupe([
        "masterpiece",
        "best quality",
        *identity_tags,
        *framing_priority,
        *early_style_tags,
        *early_reference_tags,
        *model_identity_tags,
        *wardrobe_color_positive,
        *wardrobe_tags,
        *model_wardrobe_tags,
        view_prompt,
        f"canonical model identity: {model_identity}",
        f"exact model wardrobe lock: {model_wardrobe}" if model_wardrobe else "exact wardrobe lock: preserve described outfit",
        f"[CHARACTER_ID={char_id}] single canonical character reference",
        f"visual style lock: {style}" if style else "clean character reference portrait",
        "single isolated subject centered on one plain neutral studio background, even soft lighting, accurate anatomy, detailed face",
        "same person, same face geometry, same hair silhouette, same age, same body proportions, same outfit and colors",
    ]))
    negative = ", ".join(_dedupe([
        *identity_negatives,
        *framing_negatives,
        *early_style_negatives,
        *wardrobe_color_negative,
        "duo",
        "group",
        "cropped",
        "out of frame",
        _text(character.get("negative_prompt")),
        *_character_safe_negative_tags(visual_bible.get("global_negative_prompt")),
        "faceless, blank face, featureless face, missing eyes, missing pupils, missing nose, missing mouth, face in shadow, silhouette, dramatic backlight",
        "detailed scenery, city street, corridor, neon background, colored light panels",
        "garment display, product display, clothing catalog, empty shirt, floating clothes, giant shirt in background, oversized garment behind person, duplicate clothing, second outfit",
        "multiple people, extra person, duplicate person, twins, two characters, two bodies, side-by-side people, symmetry duplication, split panel, contact sheet, character sheet, reference sheet, multiple views, inset portrait, collage, alternate outfit, alternate hairstyle, expression sheet, text, logo",
    ]))
    return {"positive_prompt": positive, "negative_prompt": negative, "view": view}


def _timeline_item(item: dict[str, Any], kind: str) -> dict[str, Any]:
    time_range = _text(item.get("time_range"))
    start_s, end_s = _parse_time_range(time_range)
    if start_s is None and item.get("start_s") is not None:
        start_s = float(item["start_s"])
    if end_s is None and item.get("end_s") is not None:
        end_s = float(item["end_s"])
    if not time_range and start_s is not None and end_s is not None:
        time_range = f"{start_s:g}-{end_s:g}s"
    result = copy.deepcopy(item)
    result.update({"kind": kind, "time_range": time_range, "start_s": start_s, "end_s": end_s})
    if kind == "spoken_dialogue" and start_s is not None and end_s is not None:
        result.setdefault("max_chars", max(1, round((end_s - start_s) * 6)))
    return result


def derive_subtitle_timeline(
    spoken_dialogue: Iterable[dict[str, Any]],
    *,
    language: str = "",
) -> list[dict[str, Any]]:
    """Derive subtitles from approved spoken lines; never author a second script."""
    subtitles: list[dict[str, Any]] = []
    for cue in spoken_dialogue:
        if not isinstance(cue, dict) or not _text(cue.get("text")):
            continue
        normalized = _timeline_item(cue, "subtitle")
        subtitles.append({
            "kind": "subtitle",
            "time_range": normalized.get("time_range", ""),
            "start_s": normalized.get("start_s"),
            "end_s": normalized.get("end_s"),
            "speaker_id": _text(cue.get("speaker_id")),
            "text": _text(cue.get("text")),
            "language": _text(cue.get("language")) or language,
            "position": "bottom-safe",
            "style": "platform subtitle; deterministic post-production overlay",
        })
    return subtitles


def subtitle_mismatch_warnings(
    spoken_dialogue: Iterable[dict[str, Any]],
    subtitle_timeline: Iterable[dict[str, Any]],
) -> list[str]:
    """Warn when a user-edited subtitle no longer mirrors approved dialogue."""
    spoken = list(spoken_dialogue)
    subtitles = list(subtitle_timeline)
    warnings: list[str] = []
    if len(spoken) != len(subtitles):
        warnings.append(f"subtitle count {len(subtitles)} does not match spoken_dialogue count {len(spoken)}")
    for index, (line, subtitle) in enumerate(zip(spoken, subtitles)):
        for key in ("speaker_id", "text"):
            if _text(line.get(key)) != _text(subtitle.get(key)):
                warnings.append(f"subtitle[{index}].{key} does not match approved spoken_dialogue")
        for key in ("start_s", "end_s"):
            try:
                mismatch = abs(float(line.get(key)) - float(subtitle.get(key))) > 0.001
            except (TypeError, ValueError):
                mismatch = line.get(key) != subtitle.get(key)
            if mismatch:
                warnings.append(f"subtitle[{index}].{key} does not match approved spoken_dialogue")
    return warnings


def continuity_chain_warnings(panels: Iterable[dict[str, Any]]) -> list[str]:
    """Validate previous-panel links and state hand-off within each continuity group."""
    warnings: list[str] = []
    last_by_group: dict[str, dict[str, Any]] = {}
    for index, panel in enumerate(panels):
        panel_id = _text(panel.get("panel_id") or panel.get("name")) or f"panel[{index}]"
        group = _text(panel.get("continuity_group")) or "main"
        previous = last_by_group.get(group)
        expected_previous = _text(previous.get("panel_id") or previous.get("name")) if previous else ""
        actual_previous = _text(panel.get("previous_panel_id"))
        if actual_previous != expected_previous:
            warnings.append(
                f"{panel_id}: previous_panel_id={actual_previous or 'null'}; expected {expected_previous or 'null'} in group {group}"
            )
        state_in = panel.get("continuity_state_in") if isinstance(panel.get("continuity_state_in"), dict) else {}
        previous_out = previous.get("continuity_state_out") if previous and isinstance(previous.get("continuity_state_out"), dict) else {}
        if previous and previous_out != state_in:
            warnings.append(f"{panel_id}: continuity_state_in does not match {expected_previous}.continuity_state_out")
        last_by_group[group] = panel
    return warnings


def _default_shot_role(index: int, count: int) -> str:
    if index == 0:
        return "hook"
    if index == count - 1:
        return "cliffhanger"
    if index == 1:
        return "setup"
    if index == count - 2:
        return "reversal"
    return "escalation"


def enrich_episode_contract(
    parsed: dict[str, Any],
    *,
    story_text: str,
    source_mode: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Return a production-ready v2 episode without mutating the LLM response."""
    episode = copy.deepcopy(parsed)
    panels = [item for item in episode.get("panels", []) if isinstance(item, dict)]
    visual = normalize_visual_bible(
        episode.get("visual_bible"),
        aspect_ratio=_text(settings.get("aspect_ratio")) or "16:9",
        visual_style=(
            _text(settings.get("style_enforcement"))
            or _text(settings.get("visual_style"))
            or _text(episode.get("style"))
            or "comic"
        ),
    )
    # User-selected production settings are authoritative, not suggestions to
    # the LLM.  This prevents a returned visual_bible from silently replacing
    # the chosen preset/custom style.
    if _text(settings.get("style_enforcement")):
        visual = normalize_visual_bible(
            {
                **visual,
                "style_prompt": _text(settings["style_enforcement"]),
                "style_name": _text(settings.get("visual_style")) or visual["style_name"],
            },
            aspect_ratio=_text(settings.get("aspect_ratio")) or "16:9",
            visual_style=(
                _text(settings.get("visual_style"))
                or _text(settings.get("style_enforcement"))
                or visual["style_name"]
            ),
        )
    raw_character_bible = (
        episode.get("character_bible")
        if isinstance(episode.get("character_bible"), list)
        else []
    )
    characters = normalize_character_bible(
        raw_character_bible,
        _text(episode.get("character_anchor_description")),
    )
    scenes = normalize_scene_bible(episode.get("scene_bible"), panels, visual)
    char_by_id = {item["character_id"]: item for item in characters}
    scene_by_id = {item["scene_id"]: item for item in scenes}
    shot_plan_enabled = bool(
        settings.get("shot_plan_version") == SHOT_PLAN_VERSION
        or any(panel.get("shot_plan_version") == SHOT_PLAN_VERSION for panel in panels)
        or (episode.get("shot_plan") or {}).get("version") == SHOT_PLAN_VERSION
    )
    target_edit_duration = float(
        settings.get("target_edit_duration_seconds")
        or settings.get("total_duration_seconds")
        or sum(float(panel.get("edit_duration_seconds") or 0) for panel in panels)
        or 0
    )
    allocated_edit_durations: list[float] = []
    if shot_plan_enabled and panels and target_edit_duration:
        allocated_edit_durations = allocate_edit_durations(target_edit_duration, len(panels))

    # The LLM may use compact ids such as char01 even though normalization
    # replaces them with render-safe stable ids. Keep a deterministic alias
    # table so panel visibility, speakers and wardrobe events follow the same
    # canonical character instead of being silently dropped.
    character_aliases: dict[str, str] = {}
    ambiguous_character_aliases: set[str] = set()

    def add_character_alias(alias: Any, canonical_id: str) -> None:
        values = alias if isinstance(alias, (list, tuple, set)) else [alias]
        for raw_value in values:
            value = _text(raw_value)
            if not value:
                continue
            for key in (value, value.lower()):
                existing = character_aliases.get(key)
                if existing and existing != canonical_id:
                    ambiguous_character_aliases.add(key)
                    character_aliases.pop(key, None)
                elif key not in ambiguous_character_aliases:
                    character_aliases[key] = canonical_id

    for index, character in enumerate(characters, 1):
        raw_character = (
            raw_character_bible[index - 1]
            if index - 1 < len(raw_character_bible)
            and isinstance(raw_character_bible[index - 1], dict)
            else {}
        )
        canonical_id = character["character_id"]
        aliases = [
            canonical_id,
            raw_character.get("character_id"),
            raw_character.get("id"),
            raw_character.get("name"),
            raw_character.get("display_name"),
            raw_character.get("aliases"),
            raw_character.get("character_aliases"),
            character.get("name"),
            character.get("aliases"),
            f"char{index:02d}",
            f"char_{index:02d}",
        ]
        for alias in aliases:
            add_character_alias(alias, canonical_id)

    def resolve_character_id(value: Any) -> str:
        candidate = _text(value)
        return character_aliases.get(candidate, character_aliases.get(candidate.lower(), ""))

    fallback_char_ids = list(char_by_id)
    enriched_panels: list[dict[str, Any]] = []
    last_panel_by_group: dict[str, str] = {}
    last_state_by_group: dict[str, dict[str, Any]] = {}
    for index, panel in enumerate(panels, 1):
        panel_id = _text(panel.get("panel_id") or panel.get("name")) or f"panel_{index:02d}"
        panel["panel_id"] = panel_id
        panel["name"] = _slug(panel_id.replace("panel_", ""), "panel") if not re.fullmatch(r"[a-z0-9_]+", panel_id) else panel_id

        package_source = panel.get("prompt_package") if isinstance(panel.get("prompt_package"), dict) else {}
        visual_cast_text = " ".join([
            _text(panel.get("first_frame")),
            _text(panel.get("last_frame")),
            _text(panel.get("camera_movement")),
            *[
                _text(item.get("shot_description"))
                for item in panel.get("cuts") or [] if isinstance(item, dict)
            ],
        ])
        ensemble_required = bool(
            len(fallback_char_ids) > 1
            and re.search(
                r"\b(?:characters|friends|team members|everyone|entire cast|group)\b|"
                r"大家|众人|全员|五位|四位|三位|他们|她们",
                visual_cast_text,
                flags=re.IGNORECASE,
            )
        )
        if ensemble_required:
            requested_ids = fallback_char_ids
        elif panel.get("character_ids"):
            requested_ids = panel.get("character_ids")
        elif panel.get("characters"):
            requested_ids = panel.get("characters")
        elif package_source.get("character_ids"):
            requested_ids = package_source.get("character_ids")
        elif panel.get("spoken_dialogue"):
            requested_ids = [item.get("speaker_id") for item in panel.get("spoken_dialogue") or []]
        else:
            requested_ids = fallback_char_ids
        if isinstance(requested_ids, str):
            requested_ids = [requested_ids]
        char_ids = _dedupe([
            resolved
            for requested in requested_ids
            if (resolved := resolve_character_id(requested)) in char_by_id
        ])
        panel["character_ids"] = char_ids
        # Keep the canonical action actor on the same alias map as the panel
        # cast.  Otherwise a valid provider alias can be normalized out of
        # character_ids while remaining stale inside action_spec, which makes
        # the later deterministic action compiler reject its own panel.
        for action_field in ("action_spec", "action_components", "act"):
            action_value = panel.get(action_field)
            if not isinstance(action_value, dict):
                continue
            for actor_field in ("actor_id", "sub"):
                raw_actor_id = _text(action_value.get(actor_field))
                actor_id = resolve_character_id(raw_actor_id)
                if actor_id:
                    action_value[actor_field] = actor_id
                    if action_field == "action_spec" and actor_id != raw_actor_id:
                        action_value.pop("spec_sha256", None)
                        action_value.pop("h3_action_en", None)

        scene_id = _text(panel.get("scene_id"))
        if scene_id not in scene_by_id:
            scene_id = scenes[min(index - 1, len(scenes) - 1)]["scene_id"] if scenes else "scene_01"
        panel["scene_id"] = scene_id
        scene = scene_by_id.get(scene_id, {})

        cuts = [_timeline_item(item, "cut") for item in panel.get("cuts", []) if isinstance(item, dict)]
        transitions = [
            _timeline_item(item, "transition") for item in panel.get("transitions", []) if isinstance(item, dict)
        ]
        for cut in cuts:
            original = _text(cut.get("editorial_shot_description") or cut.get("shot_description"))
            cut["editorial_shot_description"] = original
            cut["shot_description"] = h3_safe_visual_description(original)
        for transition in transitions:
            original = _text(
                transition.get("editorial_transition_description")
                or transition.get("transition_description")
            )
            transition["editorial_transition_description"] = original
            transition["transition_description"] = h3_safe_visual_description(original)
        spoken_dialogue = [
            _timeline_item(item, "spoken_dialogue")
            for item in panel.get("spoken_dialogue", []) if isinstance(item, dict)
        ]
        for cue in spoken_dialogue:
            speaker_id = resolve_character_id(cue.get("speaker_id"))
            if speaker_id:
                cue["speaker_id"] = speaker_id
                if speaker_id not in char_ids:
                    char_ids.append(speaker_id)
        panel["character_ids"] = char_ids
        user_edited_subtitles = bool(panel.get("_subtitle_user_edited"))
        if user_edited_subtitles:
            subtitle_timeline = [
                _timeline_item(item, "subtitle")
                for item in panel.get("subtitle_timeline", []) if isinstance(item, dict)
            ]
            subtitle_source = "user_edited"
        else:
            subtitle_timeline = derive_subtitle_timeline(
                spoken_dialogue,
                language=_text(settings.get("voice_language")),
            )
            subtitle_source = "spoken_dialogue_derived"
        subtitle_warnings = subtitle_mismatch_warnings(spoken_dialogue, subtitle_timeline)
        # Visible text generation is not entrusted to H3. If an upstream model
        # returned this lane, retain it only as an editorial/post-production
        # proposal and keep the renderer-facing fields empty.
        postproduction_text = [
            _timeline_item(item, "on_screen_text")
            for item in (
                panel.get("postproduction_on_screen_text")
                or panel.get("on_screen_text")
                or panel.get("dialogue_bubbles")
                or []
            )
            if isinstance(item, dict)
        ]
        audio_cues = [
            _timeline_item(item, "audio_cue")
            for item in (panel.get("audio_cues") or panel.get("sfx") or [])
            if isinstance(item, dict)
        ]
        raw_wardrobe_overrides = (
            panel.get("model_wardrobe_overrides_en")
            if isinstance(panel.get("model_wardrobe_overrides_en"), dict) else {}
        )
        wardrobe_overrides: dict[str, list[str]] = {}
        for raw_char_id, value in raw_wardrobe_overrides.items():
            char_id = resolve_character_id(raw_char_id)
            tags = _tag_list(value)
            if char_id in char_by_id and tags and not any(_CJK_RE.search(tag) for tag in tags):
                wardrobe_overrides[char_id] = tags
        character_prompts = {
            char_id: ", ".join(_dedupe([
                *_tag_list(char_by_id[char_id].get("model_identity_tags_en")),
                *(wardrobe_overrides.get(char_id) or _tag_list(
                    char_by_id[char_id].get("model_wardrobe_tags_en")
                )),
            ]))
            for char_id in char_ids
        }
        editorial_first_frame = _text(panel.get("editorial_first_frame") or panel.get("first_frame"))
        editorial_last_frame = _text(panel.get("editorial_last_frame") or panel.get("last_frame"))
        first_frame = h3_safe_visual_description(editorial_first_frame)
        last_frame = h3_safe_visual_description(editorial_last_frame)
        # Every creative lane is considered untrusted editorial prose.  Build
        # H3's positive package only from sentence-level text-safe variants;
        # originals remain in their bible/editorial fields for later review
        # and deterministic post-production.
        safe_style_prompt = h3_safe_visual_description(visual.get("style_prompt"))
        safe_scene_prompt = h3_safe_visual_description(
            scene.get("model_prompt_en") or scene.get("positive_prompt")
        )
        safe_character_prompts = {
            char_id: h3_safe_visual_description(value)
            for char_id, value in character_prompts.items()
        }
        shot_role = _text(panel.get("shot_role")).lower() or _default_shot_role(index - 1, len(panels))
        first_state = _text(panel.get("first_state")) or editorial_first_frame
        final_state = _text(panel.get("final_state")) or editorial_last_frame
        has_action_contract = bool(
            isinstance(panel.get("action_spec"), dict)
            or isinstance(panel.get("action_components"), dict)
            or _text(panel.get("action_code"))
        )
        compiled_action = None
        if has_action_contract:
            candidate = copy.deepcopy(panel)
            candidate["first_state"] = first_state
            candidate["final_state"] = final_state
            compiled_action = compile_panel_action(candidate, allow_legacy=True)
            visible_action = compiled_action["h3_action_en"]
        else:
            visible_action = _text(panel.get("visible_action")) or _text(
                cuts[0].get("editorial_shot_description") if cuts else ""
            )
        cause = _text(panel.get("cause")) or f"The visible action causes: {final_state}"
        next_hook = _text(panel.get("next_hook")) or (
            "Resolve the episode cliffhanger" if index == len(panels) else "The changed state motivates the next shot"
        )
        raw_camera_plan = panel.get("camera_plan") if isinstance(panel.get("camera_plan"), dict) else {}
        camera_plan = {
            "shot_size": _text(raw_camera_plan.get("shot_size")) or "medium shot",
            "angle": _text(raw_camera_plan.get("angle")) or "eye level",
            "movement": _text(raw_camera_plan.get("movement") or panel.get("camera_movement")) or "controlled push",
            "composition": _text(raw_camera_plan.get("composition")) or editorial_first_frame,
        }
        safe_camera_movement = h3_safe_visual_description(camera_plan.get("movement"))
        safe_visible_action = h3_safe_visual_description(visible_action)
        positive = "\n".join(
            f"[{label}] {value}" for label, value in [
                ("VISUAL_BIBLE", safe_style_prompt),
                ("SCENE_ID", scene_id),
                ("SCENE_LOCK", safe_scene_prompt),
                ("CHARACTER_LOCKS", " | ".join(
                    f"{key}: {value}" for key, value in safe_character_prompts.items() if value
                )),
                ("FIRST_FRAME", first_frame),
                ("VISIBLE_ACTION", safe_visible_action),
                ("LAST_FRAME", last_frame),
                ("SHOT_TIMELINE", " | ".join(_text(item.get("shot_description")) for item in cuts)),
                ("CAMERA", safe_camera_movement),
            ] if _text(value)
        )
        character_negatives = []
        for char_id in char_ids:
            value = _text(char_by_id[char_id].get("negative_prompt"))
            if char_id in wardrobe_overrides:
                value = value.replace("wardrobe variation", "unapproved wardrobe variation")
            character_negatives.append(value)
        negative = ", ".join(_dedupe([
            _text(visual.get("global_negative_prompt")),
            _text(scene.get("negative_prompt")),
            *character_negatives,
            _text(panel.get("negative_prompt")),
        ]))
        if shot_plan_enabled:
            duration = SOURCE_GENERATION_DURATION_SECONDS
            edit_duration = float(
                panel.get("edit_duration_seconds")
                if panel.get("edit_duration_seconds") is not None
                else allocated_edit_durations[index - 1]
            )
        else:
            duration = float(settings.get("duration_seconds") or panel.get("duration_seconds") or 10)
            edit_duration = duration
        continuity_group = _text(panel.get("continuity_group")) or "main"
        expected_previous = last_panel_by_group.get(continuity_group, "")
        previous_panel_id = (
            _text(panel.get("previous_panel_id"))
            if "previous_panel_id" in panel
            else expected_previous
        )
        continuity_state_in = copy.deepcopy(
            panel.get("continuity_state_in")
            if isinstance(panel.get("continuity_state_in"), dict)
            else last_state_by_group.get(continuity_group, {})
        )
        continuity_state_out = copy.deepcopy(panel.get("continuity_state_out") or continuity_state_in)
        requested_music = settings.get("background_music", panel.get("background_music", "auto_contextual"))
        requested_ambience = settings.get("ambience", panel.get("ambience", "auto_contextual"))
        panel_music = (
            panel.get("background_music") or "auto_contextual"
            if requested_music == "auto_contextual" else requested_music
        )
        panel_ambience = (
            panel.get("ambience") or "auto_contextual"
            if requested_ambience == "auto_contextual" else requested_ambience
        )
        sound_timeline = [
            {"kind": "music", "start_s": 0.0, "end_s": duration, "preset": panel_music},
            {"kind": "ambience", "start_s": 0.0, "end_s": duration, "preset": panel_ambience},
            *audio_cues,
        ]

        panel.update({
            "prompt_mode": settings.get("prompt_mode", panel.get("prompt_mode", "comic")),
            "aspect_ratio": settings.get("aspect_ratio", panel.get("aspect_ratio", "16:9")),
            "duration_seconds": duration,
            "source_generation_duration_seconds": (
                SOURCE_GENERATION_DURATION_SECONDS if shot_plan_enabled else duration
            ),
            "edit_duration_seconds": edit_duration,
            "shot_plan_version": SHOT_PLAN_VERSION if shot_plan_enabled else "legacy",
            "shot_role": shot_role,
            "story_beat_id": _text(panel.get("story_beat_id")) or f"beat_{shot_role}",
            "visible_action": visible_action,
            "first_state": first_state,
            "final_state": final_state,
            "cause": cause,
            "next_hook": next_hook,
            "camera_plan": camera_plan,
            "transition": copy.deepcopy(panel.get("transition") or {
                "type": "hard_cut" if index > 1 else "cold_open",
                "motivation": "advance to the next causal visual beat",
            }),
            "edit_hint": copy.deepcopy(panel.get("edit_hint") or {
                "preferred_moment": visible_action,
                "edit_in_hint": first_state,
                "edit_out_hint": final_state,
            }),
            "priority": _text(panel.get("priority")) or "must_have",
            "group_shot_reason": _text(panel.get("group_shot_reason")),
            "use_lora": bool(settings.get("use_lora", panel.get("use_lora", True))),
            "lora_strength": float(settings.get("lora_strength", panel.get("lora_strength", 1.0))),
            "sage_mode": settings.get("sage_mode", panel.get("sage_mode", "auto")),
            "background_music": panel_music,
            "ambience": panel_ambience,
            "voice_language": settings.get("voice_language", panel.get("voice_language", "Chinese")),
            "style_header": visual.get("style_prompt"),
            "scene_description": scene.get("description") or _text(panel.get("scene_description")),
            "character_anchor_description": " | ".join(character_prompts.values()),
            "editorial_first_frame": editorial_first_frame,
            "editorial_last_frame": editorial_last_frame,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "continuity_group": continuity_group,
            "previous_panel_id": previous_panel_id or None,
            "continuity_state_in": continuity_state_in,
            "continuity_state_out": continuity_state_out,
            "series_beat_index": panel.get("series_beat_index"),
            "model_wardrobe_overrides_en": wardrobe_overrides,
            "cuts": cuts,
            "transitions": transitions,
            "spoken_dialogue": spoken_dialogue,
            "subtitle_timeline": subtitle_timeline,
            "subtitle_source": subtitle_source,
            "subtitle_warnings": subtitle_warnings,
            "postproduction_on_screen_text": postproduction_text,
            "on_screen_text": [],
            "audio_cues": audio_cues,
            # Legacy renderer fallback must also stay empty. Approved text is
            # composited deterministically after H3, never generated in-frame.
            "dialogue_bubbles": [],
            "sfx": audio_cues,
            "positive_prompt": positive,
            "negative_prompt": negative,
            "prompt_package": {
                "schema_version": PROMPT_SCHEMA_VERSION,
                "panel_id": panel_id,
                "scene_id": scene_id,
                "character_ids": char_ids,
                "character_prompts": character_prompts,
                "model_wardrobe_overrides_en": wardrobe_overrides,
                "positive_prompt": positive,
                "negative_prompt": negative,
                "first_frame_prompt": first_frame,
                "last_frame_prompt": last_frame,
                "camera_timeline": cuts + transitions,
                "spoken_dialogue_timeline": spoken_dialogue,
                "subtitle_timeline": subtitle_timeline,
                "subtitle_source": subtitle_source,
                "subtitle_warnings": subtitle_warnings,
                "postproduction_on_screen_text_timeline": postproduction_text,
                "on_screen_text_timeline": [],
                "h3_visible_text_policy": "forbidden",
                "sound_timeline": sound_timeline,
                "render_settings": copy.deepcopy(settings),
                "shot_plan": {
                    "version": SHOT_PLAN_VERSION if shot_plan_enabled else "legacy",
                    "source_generation_duration_seconds": duration,
                    "edit_duration_seconds": edit_duration,
                    "shot_role": shot_role,
                    "story_beat_id": _text(panel.get("story_beat_id")) or f"beat_{shot_role}",
                    "visible_action": visible_action,
                    "action_spec": copy.deepcopy(compiled_action) if compiled_action else None,
                    "first_state": first_state,
                    "final_state": final_state,
                    "cause": cause,
                    "next_hook": next_hook,
                    "camera_plan": camera_plan,
                    "transition": copy.deepcopy(panel.get("transition") or {}),
                    "edit_hint": copy.deepcopy(panel.get("edit_hint") or {}),
                    "priority": _text(panel.get("priority")) or "must_have",
                },
            },
        })
        if compiled_action:
            panel["action_spec"] = copy.deepcopy(compiled_action)
            panel["action_code"] = compiled_action["action_code"]
            panel["action_components"] = derived_action_components(compiled_action)
        enriched_panels.append(panel)
        last_panel_by_group[continuity_group] = panel_id
        last_state_by_group[continuity_group] = copy.deepcopy(continuity_state_out)

    story_bible = episode.get("story_bible") if isinstance(episode.get("story_bible"), dict) else {}
    story_bible = {
        "title": _text(story_bible.get("title") or episode.get("title")),
        "logline": _text(story_bible.get("logline") or episode.get("subtitle")),
        "synopsis": _text(story_bible.get("synopsis")) or story_text.strip(),
        "genre": _text(story_bible.get("genre")),
        "target_audience": (
            _text(story_bible.get("target_audience"))
            or _text((settings.get("creative_brief") or {}).get("target_audience"))
        ),
        "themes": copy.deepcopy(story_bible.get("themes") or []),
        "world_rules": copy.deepcopy(story_bible.get("world_rules") or []),
        "continuity_rules": copy.deepcopy(story_bible.get("continuity_rules") or []),
        "source_sha256": hashlib.sha256(story_text.strip().encode("utf-8")).hexdigest(),
    }

    episode.update({
        "schema_version": PROMPT_SCHEMA_VERSION,
        "source_mode": source_mode,
        "is_demo": source_mode == "DEMO",
        "demo_notice": (
            "DEMO DATA — bundled sample; it was not generated from the user's story."
            if source_mode == "DEMO" else ""
        ),
        "story_bible": story_bible,
        "character_bible": characters,
        "visual_bible": visual,
        "scene_bible": scenes,
        "render_settings": copy.deepcopy(settings),
        "panels": enriched_panels,
        "panel_count": len(enriched_panels),
        "shot_plan": (
            {
                "version": SHOT_PLAN_VERSION,
                **shot_plan_cost_summary(target_edit_duration, len(enriched_panels)),
                "structure": [panel.get("shot_role") for panel in enriched_panels],
                "story_beats": copy.deepcopy(episode.get("story_beats") or []),
            }
            if shot_plan_enabled and enriched_panels else {"version": "legacy"}
        ),
        "character_anchor_description": " | ".join(
            ", ".join(_dedupe([
                *_tag_list(item.get("model_identity_tags_en")),
                *_tag_list(item.get("model_wardrobe_tags_en")),
            ]))
            for item in characters
        ),
        "creative_brief": copy.deepcopy(settings.get("creative_brief") or episode.get("creative_brief") or {}),
        "approval_state": copy.deepcopy(episode.get("approval_state") or {
            "creative": {"story": False, "characters": False, "storyboard": False},
            "assets": {"character_ids": [], "scene_ids": []},
        }),
    })
    episode["continuity_warnings"] = continuity_chain_warnings(enriched_panels)
    episode["subtitle_warnings"] = [
        f"{panel['panel_id']}: {warning}"
        for panel in enriched_panels
        for warning in panel.get("subtitle_warnings", [])
    ]
    return episode


def repair_episode_character_references(episode: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a persisted V3 contract with canonical character references.

    This is a deterministic migration for contracts whose LLM response used
    aliases such as ``char01`` while their normalized Character Bible uses
    stable ids. It preserves generated reference paths but revokes storyboard
    and asset approvals because the render inputs have changed.
    """
    source = copy.deepcopy(episode)
    settings = copy.deepcopy(source.get("render_settings") or {})
    if not settings:
        first_panel = next(iter(source.get("panels") or []), {})
        settings = copy.deepcopy((first_panel.get("prompt_package") or {}).get("render_settings") or {})
    story = source.get("story_bible") if isinstance(source.get("story_bible"), dict) else {}
    repaired = enrich_episode_contract(
        source,
        story_text=_text(story.get("synopsis") or source.get("title")),
        source_mode=_text(source.get("source_mode")) or "LIVE",
        settings=settings,
    )
    state = copy.deepcopy(repaired.get("approval_state") or {})
    creative = state.get("creative") if isinstance(state.get("creative"), dict) else {}
    creative["storyboard"] = False
    state["creative"] = creative
    state["assets"] = {"character_ids": [], "scene_ids": []}
    repaired["approval_state"] = state
    return repaired


def _auto_episode_shot_count(seconds_per_episode: float) -> int:
    """Backward-compatible alias for the platform edit-density policy."""
    return auto_episode_shot_count(seconds_per_episode)


def normalize_series_contract(
    parsed: dict[str, Any],
    *,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Normalize a season outline without mutating the LLM response.

    Full renderable episodes remain V3 contracts in ``episode_contracts``;
    shared season facts live only in this V4 envelope.
    """
    source = copy.deepcopy(parsed if isinstance(parsed, dict) else {})
    expected_count = int(settings.get("episode_count") or len(source.get("season_outline") or []))
    if expected_count < 1:
        raise ValueError("episode_count must be positive")
    seconds = float(settings.get("seconds_per_episode") or 0)
    if seconds < 4:
        raise ValueError("seconds_per_episode must be >= 4")
    requested_shots = settings.get("shots_per_episode")
    requested_shots = int(requested_shots) if requested_shots not in (None, "", 0) else None
    density = shot_count_bounds(seconds)
    if requested_shots is not None and not density["minimum"] <= requested_shots <= density["maximum"]:
        raise ValueError(
            "shots_per_episode cannot satisfy 1.5-4.0 second edits; "
            f"use {density['minimum']}-{density['maximum']}"
        )
    default_shots = requested_shots or _auto_episode_shot_count(seconds)

    raw_series = source.get("series_bible") if isinstance(source.get("series_bible"), dict) else {}
    brief = settings.get("creative_brief") if isinstance(settings.get("creative_brief"), dict) else {}
    title = _text(raw_series.get("title") or source.get("title") or brief.get("topic"))
    series_id = _text(raw_series.get("series_id"))
    if not re.fullmatch(r"series_[a-z0-9_]+", series_id):
        series_id = _slug(title or _text(brief.get("topic")) or "series", "series")
    series_bible = {
        "series_id": series_id,
        "title": title,
        "premise": _text(raw_series.get("premise") or brief.get("synopsis")),
        "genre": _text(raw_series.get("genre")),
        "target_audience": _text(raw_series.get("target_audience") or brief.get("target_audience")),
        "themes": copy.deepcopy(raw_series.get("themes") or []),
        "story_engine": _text(raw_series.get("story_engine")),
        "season_arc": _text(raw_series.get("season_arc")),
        "immutable_facts": copy.deepcopy(raw_series.get("immutable_facts") or []),
        "style_lock": _text(raw_series.get("style_lock") or settings.get("style_enforcement")),
    }
    visual = normalize_visual_bible(
        source.get("visual_bible"),
        aspect_ratio=_text(settings.get("aspect_ratio")) or "16:9",
        visual_style=_text(settings.get("style_enforcement") or settings.get("visual_style")) or "comic",
    )
    if _text(settings.get("style_enforcement")):
        visual = normalize_visual_bible(
            {
                **visual,
                "style_prompt": _text(settings.get("style_enforcement")),
                "style_name": _text(settings.get("visual_style")) or visual["style_name"],
            },
            aspect_ratio=_text(settings.get("aspect_ratio")) or "16:9",
            visual_style=(
                _text(settings.get("visual_style"))
                or _text(settings.get("style_enforcement"))
                or visual["style_name"]
            ),
        )
    characters = normalize_character_bible(
        source.get("shared_character_bible") or source.get("character_bible"),
        "",
    )
    scenes = normalize_scene_bible(
        source.get("shared_scene_bible") or source.get("scene_bible"),
        [],
        visual,
    )
    raw_world = source.get("world_bible") if isinstance(source.get("world_bible"), dict) else {}
    world_bible = {
        "setting": _text(raw_world.get("setting")),
        "time_period": _text(raw_world.get("time_period")),
        "world_rules": copy.deepcopy(raw_world.get("world_rules") or []),
        "geography": copy.deepcopy(raw_world.get("geography") or {}),
        "timeline_rules": copy.deepcopy(raw_world.get("timeline_rules") or []),
        "forbidden_retcons": copy.deepcopy(raw_world.get("forbidden_retcons") or []),
    }

    raw_outline = [item for item in source.get("season_outline", []) if isinstance(item, dict)]
    outline: list[dict[str, Any]] = []
    for position, item in enumerate(raw_outline, 1):
        episode_index = int(item.get("episode_index") or position)
        episode_id = f"ep_{episode_index:03d}"
        beats = []
        for beat_index, beat in enumerate(item.get("beats") or [], 1):
            if not isinstance(beat, dict):
                continue
            beats.append({
                "beat_index": int(beat.get("beat_index") or beat_index),
                "purpose": _text(beat.get("purpose")),
                "summary": _text(beat.get("summary")),
                "visible_proof": _text(beat.get("visible_proof")),
                "character_ids": list(beat.get("character_ids") or []),
                "scene_ids": list(beat.get("scene_ids") or []),
            })
        wardrobe_events = []
        for event in item.get("wardrobe_change_events") or []:
            if not isinstance(event, dict):
                continue
            wardrobe_events.append({
                "character_id": _text(event.get("character_id")),
                "from": _text(event.get("from")),
                "to": _text(event.get("to")),
                "reason": _text(event.get("reason")),
                "effective_beat": int(event.get("effective_beat") or 1),
                "model_wardrobe_tags_en": _tag_list(event.get("model_wardrobe_tags_en")),
            })
        outline.append({
            "episode_id": episode_id,
            "editorial_episode_id": _text(item.get("episode_id")) or episode_id,
            "episode_index": episode_index,
            "title": _text(item.get("title")),
            "logline": _text(item.get("logline")),
            "duration_seconds": seconds,
            "shot_count": requested_shots or int(item.get("shot_count") or default_shots),
            "shot_plan_version": SHOT_PLAN_VERSION,
            "source_generation_duration_seconds_per_shot": SOURCE_GENERATION_DURATION_SECONDS,
            "beats": beats,
            "continuity_state_in": copy.deepcopy(item.get("continuity_state_in") or {}),
            "continuity_state_out": copy.deepcopy(item.get("continuity_state_out") or {}),
            "wardrobe_change_events": wardrobe_events,
            "time_jump_event": copy.deepcopy(item.get("time_jump_event")),
            "cliffhanger_or_payoff": _text(item.get("cliffhanger_or_payoff")),
        })

    raw_contracts = source.get("episode_contracts") or {}
    if isinstance(raw_contracts, list):
        raw_contracts = {
            _text(item.get("series_episode_id") or item.get("episode_id")): item
            for item in raw_contracts if isinstance(item, dict)
        }
    known_ids = {item["episode_id"] for item in outline}
    episode_contracts = {
        episode_id: copy.deepcopy(contract)
        for episode_id, contract in raw_contracts.items()
        if episode_id in known_ids and isinstance(contract, dict)
    } if isinstance(raw_contracts, dict) else {}
    raw_approvals = source.get("episode_approvals") if isinstance(source.get("episode_approvals"), dict) else {}
    approvals = {episode_id: bool(raw_approvals.get(episode_id)) for episode_id in known_ids}

    normalized = {
        "schema_version": SERIES_SCHEMA_VERSION,
        "series_bible": series_bible,
        "shared_character_bible": characters,
        "world_bible": world_bible,
        "visual_bible": visual,
        "shared_scene_bible": scenes,
        "season_outline": outline,
        "episode_contracts": episode_contracts,
        "episode_approvals": approvals,
        "season_approved": bool(source.get("season_approved")),
        "creative_brief": copy.deepcopy(brief),
        "render_settings": copy.deepcopy(settings),
        "episode_count": expected_count,
        "seconds_per_episode": seconds,
        "shots_per_episode": requested_shots,
        "shot_plan_version": SHOT_PLAN_VERSION,
        "source_mode": _text(source.get("source_mode")) or "LIVE",
        "is_demo": bool(source.get("is_demo")),
    }
    normalized["series_sha256"] = hashlib.sha256(
        repr((series_bible, characters, world_bible, scenes, outline)).encode("utf-8")
    ).hexdigest()
    return normalized


def series_episode_context(series: dict[str, Any], episode_id: str) -> dict[str, Any]:
    """Return immutable shared facts and one episode boundary for V3 derivation."""
    outline = series.get("season_outline") or []
    index = next((i for i, item in enumerate(outline) if item.get("episode_id") == episode_id), None)
    if index is None:
        raise KeyError(f"unknown series episode: {episode_id}")
    return {
        "schema_version": SERIES_SCHEMA_VERSION,
        "series_bible": copy.deepcopy(series.get("series_bible") or {}),
        "shared_character_bible": copy.deepcopy(series.get("shared_character_bible") or []),
        "world_bible": copy.deepcopy(series.get("world_bible") or {}),
        "visual_bible": copy.deepcopy(series.get("visual_bible") or {}),
        "shared_scene_bible": copy.deepcopy(series.get("shared_scene_bible") or []),
        "episode_outline": copy.deepcopy(outline[index]),
        "previous_episode": copy.deepcopy(outline[index - 1]) if index else None,
        "next_episode": copy.deepcopy(outline[index + 1]) if index + 1 < len(outline) else None,
    }


def validate_series_contract(series: dict[str, Any]) -> list[str]:
    """Return strict V4 shape, count, shared-fact and cross-episode errors."""
    errors: list[str] = []
    if series.get("schema_version") != SERIES_SCHEMA_VERSION:
        errors.append("series.schema_version must be ai-manga.series-package/v4")
    expected_count = int(series.get("episode_count") or 0)
    outline = series.get("season_outline") if isinstance(series.get("season_outline"), list) else []
    if len(outline) != expected_count:
        errors.append(f"season_outline must contain exactly {expected_count} episodes")
    characters = series.get("shared_character_bible") or []
    scenes = series.get("shared_scene_bible") or []
    if not characters:
        errors.append("shared_character_bible missing or empty")
    if not scenes:
        errors.append("shared_scene_bible missing or empty")
    char_ids = {item.get("character_id") for item in characters}
    scene_ids = {item.get("scene_id") for item in scenes}
    for index, character in enumerate(characters):
        identity_tags = character.get("model_identity_tags_en") or []
        wardrobe_tags = character.get("model_wardrobe_tags_en") or []
        if not identity_tags or not wardrobe_tags:
            errors.append(f"shared_character_bible[{index}] English model tags incomplete")
        if any(_CJK_RE.search(str(tag)) for tag in [*identity_tags, *wardrobe_tags]):
            errors.append(f"shared_character_bible[{index}] model tags must be English")
        if character.get("model_prompt_warnings"):
            errors.extend(
                f"shared_character_bible[{index}] {warning}"
                for warning in character.get("model_prompt_warnings") or []
            )
    for index, scene in enumerate(scenes):
        prompt = _text(scene.get("model_prompt_en"))
        if not prompt or _CJK_RE.search(prompt):
            errors.append(f"shared_scene_bible[{index}].model_prompt_en must be English")
        errors.extend(
            f"shared_scene_bible[{index}] {warning}"
            for warning in scene.get("model_prompt_warnings") or []
        )

    seconds = float(series.get("seconds_per_episode") or 0)
    previous: dict[str, Any] | None = None
    for position, item in enumerate(outline, 1):
        episode_id = item.get("episode_id")
        if episode_id != f"ep_{position:03d}" or item.get("episode_index") != position:
            errors.append(f"season_outline[{position - 1}] ID/index must be ep_{position:03d}/{position}")
        if float(item.get("duration_seconds") or 0) != seconds:
            errors.append(f"{episode_id}.duration_seconds must equal {seconds:g}")
        shot_count = int(item.get("shot_count") or 0)
        bounds = shot_count_bounds(seconds)
        if shot_count < bounds["minimum"] or shot_count > bounds["maximum"]:
            errors.append(
                f"{episode_id}.shot_count must be {bounds['minimum']}-{bounds['maximum']} "
                "for 1.5-4.0 second final edits"
            )
        if item.get("shot_plan_version") != SHOT_PLAN_VERSION:
            errors.append(f"{episode_id}.shot_plan_version must be {SHOT_PLAN_VERSION}")
        if abs(float(item.get("source_generation_duration_seconds_per_shot") or 0) - SOURCE_GENERATION_DURATION_SECONDS) > 0.0001:
            errors.append(f"{episode_id}.source_generation_duration_seconds_per_shot must equal 10.125")
        if not item.get("title") or not item.get("logline") or not item.get("beats"):
            errors.append(f"{episode_id} title/logline/beats incomplete")
        if not isinstance(item.get("continuity_state_in"), dict) or not isinstance(item.get("continuity_state_out"), dict):
            errors.append(f"{episode_id} continuity states must be objects")
        if previous is not None and item.get("continuity_state_in") != previous.get("continuity_state_out"):
            errors.append(f"{episode_id}.continuity_state_in must exactly equal {previous.get('episode_id')}.continuity_state_out")
        for beat_index, beat in enumerate(item.get("beats") or []):
            unknown_chars = set(beat.get("character_ids") or []) - char_ids
            unknown_scenes = set(beat.get("scene_ids") or []) - scene_ids
            if unknown_chars or unknown_scenes:
                errors.append(
                    f"{episode_id}.beats[{beat_index}] unknown references: characters={sorted(unknown_chars)}, scenes={sorted(unknown_scenes)}"
                )
            if _text(beat.get("purpose")).lower() not in SHOT_ROLES:
                errors.append(f"{episode_id}.beats[{beat_index}].purpose is not a platform story role")
            if not _text(beat.get("visible_proof")):
                errors.append(f"{episode_id}.beats[{beat_index}].visible_proof missing")
        beat_roles = {_text(beat.get("purpose")).lower() for beat in item.get("beats") or []}
        if not {"hook", "setup", "escalation", "reversal"}.issubset(beat_roles):
            errors.append(f"{episode_id}.beats must cover hook/setup/escalation/reversal")
        if not beat_roles.intersection({"cliffhanger", "close"}):
            errors.append(f"{episode_id}.beats requires cliffhanger or close")
        for event_index, event in enumerate(item.get("wardrobe_change_events") or []):
            if event.get("character_id") not in char_ids:
                errors.append(f"{episode_id}.wardrobe_change_events[{event_index}] unknown character")
            if not all(event.get(key) for key in ("from", "to", "reason", "model_wardrobe_tags_en")):
                errors.append(f"{episode_id}.wardrobe_change_events[{event_index}] incomplete")
            if any(_CJK_RE.search(str(tag)) for tag in event.get("model_wardrobe_tags_en") or []):
                errors.append(f"{episode_id}.wardrobe_change_events[{event_index}] model tags must be English")
            if int(event.get("effective_beat") or 0) not in {
                int(beat.get("beat_index") or 0) for beat in item.get("beats") or []
            }:
                errors.append(f"{episode_id}.wardrobe_change_events[{event_index}] effective_beat is unknown")
        time_jump = item.get("time_jump_event")
        if time_jump is not None and (
            not isinstance(time_jump, dict)
            or not all(_text(time_jump.get(key)) for key in ("from", "to", "reason"))
        ):
            errors.append(f"{episode_id}.time_jump_event must contain from/to/reason")
        previous = item

    shared_char_by_id = {item.get("character_id"): item for item in characters}
    shared_scene_by_id = {item.get("scene_id"): item for item in scenes}
    wardrobe_state = {
        char_id: list(item.get("model_wardrobe_tags_en") or [])
        for char_id, item in shared_char_by_id.items()
    }
    wardrobe_before_episode: dict[str, dict[str, list[str]]] = {}
    for outline_item in outline:
        current_id = str(outline_item.get("episode_id"))
        wardrobe_before_episode[current_id] = copy.deepcopy(wardrobe_state)
        for event in outline_item.get("wardrobe_change_events") or []:
            wardrobe_state[str(event.get("character_id"))] = list(
                event.get("model_wardrobe_tags_en") or []
            )

    contracts = series.get("episode_contracts") or {}
    for approved_id, approved in (series.get("episode_approvals") or {}).items():
        if approved and approved_id not in contracts:
            errors.append(f"episode_approvals.{approved_id} cannot be true without a V3 contract")
    for episode_id, contract in contracts.items():
        outline_item = next((item for item in outline if item.get("episode_id") == episode_id), None)
        if outline_item is None:
            errors.append(f"episode_contracts contains unknown episode {episode_id}")
            continue
        if contract.get("schema_version") != PROMPT_SCHEMA_VERSION:
            errors.append(f"episode_contracts.{episode_id} must remain a V3 contract")
        panels = contract.get("panels") or []
        if len(panels) != int(outline_item.get("shot_count") or 0):
            errors.append(f"episode_contracts.{episode_id} panel count does not match outline shot_count")
        if abs(sum(float(panel.get("edit_duration_seconds") or 0) for panel in panels) - seconds) > 0.001:
            errors.append(f"episode_contracts.{episode_id} edit duration does not equal seconds_per_episode")
        for panel_index, panel in enumerate(panels):
            if abs(float(panel.get("source_generation_duration_seconds") or 0) - SOURCE_GENERATION_DURATION_SECONDS) > 0.0001:
                errors.append(
                    f"episode_contracts.{episode_id}.panels[{panel_index}] source duration must equal 10.125"
                )
        if contract.get("series_episode_id") != episode_id:
            errors.append(f"episode_contracts.{episode_id}.series_episode_id mismatch")
        if contract.get("continuity_state_in") != outline_item.get("continuity_state_in"):
            errors.append(f"episode_contracts.{episode_id}.continuity_state_in drift")
        if contract.get("continuity_state_out") != outline_item.get("continuity_state_out"):
            errors.append(f"episode_contracts.{episode_id}.continuity_state_out drift")
        if contract.get("series_sha256") != series.get("series_sha256"):
            errors.append(f"episode_contracts.{episode_id}.series_sha256 drift")
        actual_characters = {
            item.get("character_id"): item for item in contract.get("character_bible") or []
        }
        if set(actual_characters) != set(shared_char_by_id):
            errors.append(f"episode_contracts.{episode_id} character IDs drift from shared bible")
        for char_id, shared in shared_char_by_id.items():
            actual = actual_characters.get(char_id) or {}
            if actual.get("model_identity_tags_en") != shared.get("model_identity_tags_en"):
                errors.append(f"episode_contracts.{episode_id}.{char_id} identity tags drift")
            if actual.get("voice_profile") != shared.get("voice_profile"):
                errors.append(f"episode_contracts.{episode_id}.{char_id} voice profile drift")
            if actual.get("model_wardrobe_tags_en") != wardrobe_before_episode.get(episode_id, {}).get(char_id):
                errors.append(f"episode_contracts.{episode_id}.{char_id} baseline wardrobe drift")
        actual_scenes = {item.get("scene_id"): item for item in contract.get("scene_bible") or []}
        if set(actual_scenes) != set(shared_scene_by_id):
            errors.append(f"episode_contracts.{episode_id} scene IDs drift from shared bible")
        for scene_id, shared in shared_scene_by_id.items():
            if (actual_scenes.get(scene_id) or {}).get("model_prompt_en") != shared.get("model_prompt_en"):
                errors.append(f"episode_contracts.{episode_id}.{scene_id} model prompt drift")
        if (contract.get("visual_bible") or {}).get("style_prompt") != (
            series.get("visual_bible") or {}
        ).get("style_prompt"):
            errors.append(f"episode_contracts.{episode_id} visual style drift")
        current_events = outline_item.get("wardrobe_change_events") or []
        for panel_index, panel in enumerate(panels):
            beat_index = int(panel.get("series_beat_index") or 0)
            expected_overrides = {
                str(event.get("character_id")): list(event.get("model_wardrobe_tags_en") or [])
                for event in current_events
                if beat_index >= int(event.get("effective_beat") or 1)
            }
            if (panel.get("model_wardrobe_overrides_en") or {}) != expected_overrides:
                errors.append(
                    f"episode_contracts.{episode_id}.panels[{panel_index}] wardrobe override drift"
                )
    return errors
