"""Auditable scene-reference generation for scene_bible entries.

The workflow deliberately creates an empty environment plate: characters and
all text are excluded so H3 can combine this spatial reference with separately
approved character assets.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from PIL import Image, ImageDraw

from runtime_config import comfyui_root, comfyui_server


DEFAULT_CHECKPOINT = "animagine-xl-3.1.safetensors"
STRUCTURAL_SCENE_CHECKPOINT = "RealVisXL_V5.0_fp16.safetensors"
SOCIAL_LAYOUT_CHECKPOINT = "anything-v5-PrtRE.safetensors"
SOCIAL_LAYOUT_CONTROLNET = "control_v11p_sd15_mlsd_fp16.safetensors"
SOCIAL_LAYOUT_STRENGTH = 0.55
SOCIAL_LAYOUT_END_PERCENT = 0.68
CONVENIENCE_LAYOUT_STRENGTH = 0.38
CONVENIENCE_LAYOUT_END_PERCENT = 0.55


_UNSAFE_SCENE_FRAGMENT = re.compile(
    r"\b(?:1girl|1boy|girl|boy|woman|man|female|male|person|people|human|character|"
    r"hero|heroine|protagonist|figure|silhouette|crowd|portrait|caption|subtitle|"
    r"speech bubble|text|letters?|words?|logo|watermark|sign|signage|poster|billboard)\b|"
    r"(?:人物|角色|女性|男性|少女|少男|女孩|男孩|人群|路人|主角|字幕|文字|招牌|海报)",
    re.IGNORECASE,
)

_COMPACT_INTERIOR_RE = re.compile(
    r"small\s+room|compact\s+(?:room|interior)|cramped|tiny\s+(?:room|interior)|"
    r"\b\d+(?:\.\d+)?\s*(?:square\s*meters?|m2|m²)\b|狭窄|狭小|小房间|平方米",
    re.IGNORECASE,
)
_NON_VISUAL_SCENE_FRAGMENT = re.compile(
    r"\b(?:noise|sound|audio|ambience|quiet atmosphere)\b|喧嚣声|声音|音效|安静氛围",
    re.IGNORECASE,
)
_STYLE_SCENE_FRAGMENT = re.compile(
    r"\b(?:anime|animation|2d|3d|line ?art|painterly|illustration|cinematic|photorealistic|"
    r"masterpiece|best quality|aesthetic|style|screencap)\b",
    re.IGNORECASE,
)

_DEVICE_NOUN_RE = re.compile(
    r"\b(?:mobile\s+phones?|smartphones?|phones?|tablets?|devices?)\b|"
    r"(?:手机|平板电脑|平板|设备)",
    re.IGNORECASE,
)
_SOCIAL_ROOM_RE = re.compile(
    r"\b(?:living\s+room|gaming\s+room|game\s+room|small\s+room|compact\s+(?:room|interior)|"
    r"room|interior|table|desk)\b|(?:客厅|游戏房|电竞房|房间|室内|桌)",
    re.IGNORECASE,
)
_RETAIL_OR_DISPLAY_RE = re.compile(
    r"\b(?:retail|store|shop|showroom|merchandise|inventory|product\s+display|"
    r"display\s+(?:cabinet|case|stand)|glass\s+display\s+case|cabinet|drawer|shelf|"
    r"shelves|shelving|storage\s+rack|product\s+tray|checkout\s+counter|store\s+aisle|"
    r"catalog\s+photography|product\s+photography|flat\s+lay|top[- ]down|overhead|"
    r"bird['’]?s[- ]eye|tabletop\s+close[- ]?up)\b",
    re.IGNORECASE,
)

_SOCIAL_DEVICE_ROOM_NEGATIVE = (
    "retail store, electronics store, phone shop, mobile phone shop, showroom, product showroom, "
    "product display, merchandise display, retail display, display cabinet, glass display case, "
    "cabinet, drawer, open drawer, shelf, shelves, shelving, storage rack, product tray, "
    "phone display stand, checkout counter, store aisle, merchandise, inventory, repeated products, "
    "catalog photography, product photography, flat lay, tabletop close-up, close-up product shot"
)

_ANIMAGINE_NEGATIVE = (
    "nsfw, lowres, bad, text, error, fewer, extra, missing, worst quality, "
    "jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, "
    "chromatic aberration, signature, extra digits, artistic error, username, scan, abstract"
)
_CONVENIENCE_STORE_RE = re.compile(
    r"\b(?:convenience\s+store|konbini|checkout\s+counter)\b|便利店",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_CHINESE_NUMBERS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _environment_only_text(*values: Any) -> str:
    safe: list[str] = []
    for value in values:
        for fragment in re.split(r"[,;，；。\n]+", str(value or "")):
            cleaned = re.sub(r"\s+", " ", fragment).strip(" .")
            if (
                cleaned
                and not _UNSAFE_SCENE_FRAGMENT.search(cleaned)
                and not _NON_VISUAL_SCENE_FRAGMENT.search(cleaned)
            ):
                safe.append(cleaned)
    return ", ".join(dict.fromkeys(safe))


def _ascii_model_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" ,")
    return text if text.isascii() and text.isprintable() else ""


def _device_count(value: str) -> int | None:
    """Extract an explicit device count without guessing from unrelated numbers."""
    normalized = value
    for word, number in _NUMBER_WORDS.items():
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
        flags=re.IGNORECASE,
    )
    counts: list[int] = []
    for match in matches:
        token = str(match.group(1) or match.group(2) or "").strip()
        count = int(token) if token.isdigit() else _CHINESE_NUMBERS.get(token)
        if count is not None:
            counts.append(count)
    # A legacy prompt can contain "one phone at each seat" after the actual
    # contract "five ... smartphones".  Choosing the first match incorrectly
    # reduced the entire asset to one device.  The largest explicit count is
    # the room-level inventory; per-seat wording remains a relationship lock.
    return max(counts) if counts else None


def _social_device_room(value: str, count: int | None) -> bool:
    return bool(count and count > 1 and _DEVICE_NOUN_RE.search(value) and _SOCIAL_ROOM_RE.search(value))


def _without_display_layout(value: str) -> str:
    """Remove legacy device-array/camera tags before applying the approved room layout."""
    safe: list[str] = []
    for fragment in re.split(r"[,;\n]+", value):
        cleaned = re.sub(r"\s+", " ", fragment).strip(" .")
        if not cleaned:
            continue
        if _DEVICE_NOUN_RE.search(cleaned) or _RETAIL_OR_DISPLAY_RE.search(cleaned):
            continue
        if re.search(r"\b(?:desk|table|chairs?|seats?)\b", cleaned, re.IGNORECASE):
            continue
        safe.append(cleaned)
    return ", ".join(dict.fromkeys(safe))


def _without_style_fragments(value: str) -> str:
    """Keep physical scene facts out of the structural model's style prompt."""
    return ", ".join(
        dict.fromkeys(
            part.strip() for part in str(value or "").split(",")
            if part.strip() and not _STYLE_SCENE_FRAGMENT.search(part)
        )
    )


def _composition_for_aspect(aspect: str) -> tuple[int, int, str]:
    if aspect == "9:16":
        return 544, 960, (
            "single vertical environment view, deep perspective, centered architectural leading lines, "
            "one continuous full-bleed frame, stable horizon"
        )
    if aspect == "1:1":
        return 768, 768, (
            "single square environment view, balanced architectural composition, one continuous full-bleed frame, stable horizon"
        )
    return 960, 544, (
        "single wide environment view, cinematic establishing composition, one-point perspective, "
        "one continuous full-bleed frame, stable horizon"
    )


def _draw_social_room_layout(width: int, height: int, count: int) -> Image.Image:
    """Create an architectural MLSD guide without repeating prop silhouettes.

    Exact chair/phone rectangles made lineart ControlNet copy a diagram instead
    of drawing a room.  The guide now owns only perspective, enclosure and the
    shared table.  The approved prompt remains responsible for prop semantics.
    """
    del count
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    stroke = max(3, min(width, height) // 96)
    # Enclosed room, back wall and floor perspective.
    draw.rectangle((stroke, stroke, width - stroke, height - stroke), outline="black", width=stroke)
    back_left, back_right = int(width * 0.22), int(width * 0.78)
    back_top, back_floor = int(height * 0.13), int(height * 0.44)
    draw.rectangle((back_left, back_top, back_right, back_floor), outline="black", width=stroke)
    draw.line((stroke, height - stroke, back_left, back_floor), fill="black", width=stroke)
    draw.line((width - stroke, height - stroke, back_right, back_floor), fill="black", width=stroke)
    # A window and door give the diffusion model recognizable room grammar.
    draw.rectangle(
        (int(width * 0.30), int(height * 0.20), int(width * 0.48), int(height * 0.36)),
        outline="black", width=stroke,
    )
    draw.rectangle(
        (int(width * 0.60), int(height * 0.19), int(width * 0.72), back_floor),
        outline="black", width=stroke,
    )
    # One slightly trapezoidal shared table, leaving clear circulation space.
    table = [
        (int(width * 0.34), int(height * 0.47)),
        (int(width * 0.66), int(height * 0.47)),
        (int(width * 0.82), int(height * 0.73)),
        (int(width * 0.18), int(height * 0.73)),
    ]
    draw.polygon(table, outline="black", fill="white", width=stroke)
    for x1, y1, x2, y2 in (
        (0.18, 0.73, 0.16, 0.87), (0.82, 0.73, 0.84, 0.87),
        (0.34, 0.47, 0.33, 0.57), (0.66, 0.47, 0.67, 0.57),
    ):
        draw.line((int(width*x1), int(height*y1), int(width*x2), int(height*y2)), fill="black", width=stroke)
    return image


def _prepare_social_layout_image(
    scene: Mapping[str, Any], visual_bible: Mapping[str, Any], story_hash: str
) -> str | None:
    source = "; ".join([
        str(scene.get("description") or ""),
        str(scene.get("model_prompt_en") or ""),
        str(scene.get("positive_prompt") or ""),
        str(scene.get("continuity_lock") or ""),
    ])
    count = _device_count(source)
    if not _social_device_room(source, count):
        return None
    aspect = str(visual_bible.get("aspect_ratio") or "16:9")
    width, height, _ = _composition_for_aspect(aspect)
    digest = hashlib.sha256(
        f"{story_hash}|{scene.get('scene_id')}|{aspect}|{count}|architecture-layout-v2".encode("utf-8")
    ).hexdigest()[:12]
    relative = Path("scene_layouts") / f"{_safe(str(scene.get('scene_id') or 'scene'))}_{digest}.png"
    target = comfyui_root() / "input" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    _draw_social_room_layout(width, height, int(count or 1)).save(target, format="PNG")
    return relative.as_posix()


def _draw_convenience_store_layout(width: int, height: int) -> Image.Image:
    """Create a clean MLSD guide for entrance/counter/shelf geography."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    stroke = max(3, min(width, height) // 96)
    draw.rectangle((stroke, stroke, width - stroke, height - stroke), outline="black", width=stroke)
    horizon = int(height * 0.49)
    draw.line((stroke, horizon, width - stroke, horizon), fill="black", width=stroke)
    # Street-facing double entrance on the left rear wall.
    door = (int(width * 0.07), int(height * 0.15), int(width * 0.42), horizon)
    draw.rectangle(door, outline="black", width=stroke)
    mid_x = (door[0] + door[2]) // 2
    draw.line((mid_x, door[1], mid_x, door[3]), fill="black", width=stroke)
    handle_y = int(height * 0.34)
    draw.line((mid_x - stroke * 3, handle_y, mid_x - stroke, handle_y), fill="black", width=stroke)
    draw.line((mid_x + stroke, handle_y, mid_x + stroke * 3, handle_y), fill="black", width=stroke)
    # Sparse shelves on the right rear wall.
    shelf = (int(width * 0.57), int(height * 0.16), int(width * 0.93), horizon)
    draw.rectangle(shelf, outline="black", width=stroke)
    for ratio in (0.32, 0.49, 0.66, 0.83):
        y = int(shelf[1] + (shelf[3] - shelf[1]) * ratio)
        draw.line((shelf[0], y, shelf[2], y), fill="black", width=stroke)
    # Checkout counter anchors the foreground without blocking the entrance.
    counter = (
        int(width * 0.27), int(height * 0.58),
        int(width * 0.96), int(height * 0.84),
    )
    draw.rectangle(counter, outline="black", fill="white", width=stroke)
    draw.line(
        (counter[0], int(height * 0.64), counter[2], int(height * 0.64)),
        fill="black", width=stroke,
    )
    # Small separated hero props: clear donation box and lidded cup.
    draw.rectangle(
        (int(width * 0.54), int(height * 0.47), int(width * 0.65), int(height * 0.56)),
        outline="black", width=stroke,
    )
    cup = (int(width * 0.72), int(height * 0.48), int(width * 0.78), int(height * 0.56))
    draw.rectangle(cup, outline="black", width=stroke)
    draw.line((cup[0] - stroke, cup[1], cup[2] + stroke, cup[1]), fill="black", width=stroke)
    return image


def _prepare_convenience_store_layout_image(
    scene: Mapping[str, Any], visual_bible: Mapping[str, Any], story_hash: str
) -> str | None:
    source = "; ".join([
        str(scene.get("description") or ""),
        str(scene.get("model_prompt_en") or ""),
        str(scene.get("positive_prompt") or ""),
    ])
    if not _CONVENIENCE_STORE_RE.search(source):
        return None
    aspect = str(visual_bible.get("aspect_ratio") or "16:9")
    width, height, _ = _composition_for_aspect(aspect)
    digest = hashlib.sha256(
        f"{story_hash}|{scene.get('scene_id')}|{aspect}|convenience-layout-v1".encode("utf-8")
    ).hexdigest()[:12]
    relative = Path("scene_layouts") / f"{_safe(str(scene.get('scene_id') or 'scene'))}_{digest}.png"
    target = comfyui_root() / "input" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    _draw_convenience_store_layout(width, height).save(target, format="PNG")
    return relative.as_posix()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "scene"


def stable_scene_seed(scene: Mapping[str, Any], story_hash: str = "") -> int:
    material = "|".join([
        story_hash,
        str(scene.get("scene_id") or ""),
        str(scene.get("positive_prompt") or scene.get("description") or ""),
        str(scene.get("continuity_lock") or ""),
    ])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def build_scene_reference_workflow(
    scene: Mapping[str, Any],
    visual_bible: Mapping[str, Any],
    *,
    story_hash: str = "",
    checkpoint: str = DEFAULT_CHECKPOINT,
    seed: Optional[int] = None,
    layout_image_name: str | None = None,
    structural_checkpoint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene_id = str(scene.get("scene_id") or "scene")
    actual_seed = stable_scene_seed(scene, story_hash) if seed is None else int(seed)
    aspect = str(visual_bible.get("aspect_ratio") or "16:9")
    width, height, composition = _composition_for_aspect(aspect)
    style = _ascii_model_text(_environment_only_text(visual_bible.get("style_prompt"))) or "modern Japanese anime background art, clean lineart"
    explicit_environment = scene.get("model_prompt_en") or scene.get("model_environment_tags_en")
    if explicit_environment:
        environment = _environment_only_text(explicit_environment)
    else:
        environment = _environment_only_text(
            scene.get("positive_prompt"),
            scene.get("description"),
            scene.get("continuity_lock"),
            scene.get("palette"),
        )
    environment = environment or "approved empty environment"
    scene_source = "; ".join([
        str(scene.get("description") or ""),
        str(scene.get("model_prompt_en") or ""),
        str(scene.get("positive_prompt") or ""),
        str(scene.get("continuity_lock") or ""),
    ])
    compact_interior = bool(_COMPACT_INTERIOR_RE.search(scene_source))
    continuity = scene.get("continuity_lock") if isinstance(scene.get("continuity_lock"), Mapping) else {}
    hero_props = _environment_only_text(continuity.get("hero_props"))
    prop_source = "; ".join([
        environment,
        hero_props,
        str(scene.get("description") or ""),
    ])
    detected_device_count = _device_count(prop_source)
    social_device_room = _social_device_room(scene_source, detected_device_count)
    convenience_store = bool(_CONVENIENCE_STORE_RE.search(scene_source))
    device_count = str(detected_device_count or "")
    required_prop_lock = (
        f"required hero props clearly visible and countable: {hero_props}" if hero_props else ""
    )
    if social_device_room:
        # The scene plate owns room identity and table geography.  Exact
        # repeated prop silhouettes are left to the H3 shot composition; both
        # SDXL and SD1.5 otherwise turn the background into a catalog/grid.
        device_count = str(detected_device_count)
        environment = _without_display_layout(environment)
        required_prop_lock = (
            "(one single large ordinary rectangular shared gaming table dominates the center foreground:1.5), "
            "the complete tabletop is clearly visible and occupies the lower-middle of the frame, "
            f"the table is naturally prepared for a group of {device_count} close friends playing a mobile game, "
            "ordinary home chairs around the shared table, a few small black-screen smartphones resting naturally on the tabletop"
        )
    if social_device_room:
        composition = (
            "single human eye-level wide shot of the entire living room and gaming area, "
            "camera at room entrance about 1.5 meters above the floor, "
            "centered table seen obliquely in perspective, tabletop and all empty seating places visible, "
            "stable horizontal horizon line, one continuous full-bleed frame"
        )
    elif compact_interior:
        composition = (
            "single vertical eye-level wide-angle compact room interior, camera at seated human eye height, "
            "near side walls and back wall visible, ordinary desk and required devices fill the lower half, "
            "one continuous full-bleed frame, stable horizontal eye line"
        )
    scale_lock = (
        (
            "ordinary contemporary private living room arranged as a casual gaming room, "
            "bright even daytime practical lighting, pale painted walls, ordinary home furniture, "
            "near side walls and back wall visibly enclose the room, shared table is the spatial center, "
            "phones are small secondary personal gaming props within the room"
        )
        if social_device_room else (
            "bright evenly lit contemporary gaming room at noon, pale painted walls, light wood furniture, "
            "ordinary eye-level compact interior view, near side walls and back wall visibly enclose the space, "
            "spatial scale unmistakably tiny, one practical desk fills most of the room, "
            "every described key prop fully visible and countable"
        )
        if compact_interior
        else "ordinary eye-level environment view, every described key prop fully visible"
    )
    convenience_lock = (
        "indoors, empty convenience store interior, checkout counter in foreground, cash register, "
        "store shelves with plain geometric product blocks, every package face solid-color blank and unmarked, "
        "no price cards, no shelf labels, no display placards, no printed designs, "
        "glass entrance doors, blue rainy night visible only outside beyond glass, "
        "completely dry interior and dry floor, warm fluorescent ceiling lights, "
        "(small transparent charity donation box with completely blank sides clearly visible on checkout counter:1.45), "
        "(plain unmarked paper hot drink cup beside donation box:1.3), "
        "eye-level wide shot, clear retail interior layout, no corridor, no kiosk"
        if convenience_store else ""
    )
    positive = ", ".join(filter(None, [
        "anime background, scenery",
        required_prop_lock,
        convenience_lock,
        scale_lock,
        composition,
        "empty establishing shot, empty environment, no people, no humans, no characters",
        environment,
        style,
        "stable architecture and prop layout, coherent lighting, physically continuous space",
        "masterpiece, best quality, very aesthetic, absurdres, safe, newest",
    ]))
    negative = ", ".join(filter(None, [
        _ANIMAGINE_NEGATIVE,
        _ascii_model_text(visual_bible.get("global_negative_prompt")),
        _ascii_model_text(scene.get("negative_prompt")),
        "people, person, human, crowd, 1girl, girl, woman, female, 1boy, boy, man, male, character, protagonist, figure, silhouette, face, body, portrait",
        "fantasy girl, fantasy woman, fantasy character, qipao girl, decorative character",
        "split screen, split image, collage, diptych, triptych, multiple views, multiple panels, comic panel layout, contact sheet, image grid, border, picture-in-picture",
        "subtitle, caption, speech bubble, text, signage, sign, poster, billboard, typography, logo, watermark, signature, letters, numbers, random text, illegible text, printed packaging, price tag, shelf label, display card, placard, branded container",
        _SOCIAL_DEVICE_ROOM_NEGATIVE if social_device_room else "",
        (
            "corridor, hallway, tunnel, kiosk, booth, gazebo, shrine, outdoor stall, warehouse, "
            "residential room, empty room without shelves, missing checkout counter, missing store shelves, "
            "missing donation box, missing hot drink cup, rain inside, indoor rainfall, wet interior, wet floor, "
            "ceiling-dominant composition, low angle"
            if convenience_store else ""
        ),
        (
            "vast hall, large room, warehouse, atrium, empty office floor, corridor, monumental architecture, "
            "ceiling-dominant view, low-angle ceiling shot, overhead view, top-down view, bird's-eye view, "
            "empty bare room, classroom, school, lecture hall, conference room, rows of desks, "
            "multiple tables, multiple desks, retail seating rows, chairless room, missing seats, "
            "missing central table, table out of frame, bare tabletop, empty tabletop, phones absent, missing required props, wrong object count, "
            "dark room, nighttime, dim lighting, sepia, "
            "abandoned room, industrial hall, traditional wooden room, long narrow tunnel, dramatic shadows"
            if compact_interior else ""
        ),
    ]))
    if seed is None:
        seed_material = "|".join([
            story_hash,
            scene_id,
            positive,
            negative,
            str(len(scene.get("asset_rejection_history") or [])),
            "scene-prompt-seed-v7-animagine-tags",
        ])
        actual_seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:4], "big"
        )
    convenience_layout = bool(convenience_store and layout_image_name)
    layout_strength = CONVENIENCE_LAYOUT_STRENGTH if convenience_layout else SOCIAL_LAYOUT_STRENGTH
    layout_end_percent = (
        CONVENIENCE_LAYOUT_END_PERCENT if convenience_layout else SOCIAL_LAYOUT_END_PERCENT
    )
    selected_checkpoint = (
        SOCIAL_LAYOUT_CHECKPOINT
        if (social_device_room or convenience_layout) and layout_image_name
        else checkpoint
    )
    two_pass_structure = bool(convenience_store and structural_checkpoint and not layout_image_name)
    if convenience_layout:
        # Anything V5 is an SD1.5 checkpoint with a much shorter effective
        # CLIP prompt budget than SDXL.  A long editorial prompt truncates the
        # entrance/counter facts and turns the line guide into refrigerators.
        # Keep every layout-defining noun in the first compact tag sequence.
        positive = (
            "masterpiece, best quality, anime background, empty convenience store interior at night, "
            "camera inside store looking across checkout counter toward entrance, eye-level wide shot, "
            "solid checkout counter in lower-right foreground, small clear acrylic coin donation box "
            "and plain lidded paper cup separated on counter, transparent glass double entrance at left rear, "
            "dark blue rainy street and soft blue rain bokeh visible outside through doors, "
            "sparse blank product shelves on right, warm amber ceiling lights, completely dry interior, "
            "clean stable architecture, no people"
        )
        negative = (
            "people, person, character, text, letters, numbers, logo, label, signage, poster, paper notice, "
            "screen text, watermark, branded package, refrigerator, freezer, vending machine, display cabinet, "
            "glass merchandise case, shelf wall replacing entrance, indoor shelves behind entrance, "
            "outdoor camera, exterior viewpoint, camera outside store, window wall, city skyline, skyscraper, "
            "street scene dominating frame, floor tile grid, tiled floor dominating frame, "
            "missing checkout counter, missing glass doors, missing donation box, missing paper cup, "
            "rain inside, wet interior, corridor, laboratory, split screen, collage, low quality, blurry"
        )
    if Path(selected_checkpoint).name.casefold().startswith("animagine-xl-3.1"):
        if aspect == "9:16":
            width, height = 768, 1344
        elif aspect == "1:1":
            width, height = 1024, 1024
        else:
            width, height = 1344, 768
        sampler_name, scheduler, steps, cfg = "euler_ancestral", "normal", 28, 6.0
    elif convenience_layout:
        sampler_name, scheduler, steps, cfg = "euler_ancestral", "normal", 28, 7.0
    else:
        sampler_name, scheduler, steps, cfg = "dpmpp_2m", "karras", 32, 7.0
    structure_positive = ""
    structure_negative = ""
    if two_pass_structure:
        structure_environment = _without_style_fragments(environment)
        structure_positive = ", ".join(filter(None, [
            "professional architectural visualization, accurate retail interior layout, single eye-level wide shot",
            "empty contemporary convenience store, checkout counter dominating the foreground",
            "one small clear acrylic coin donation box with a visible coin slot and completely blank sides, "
            "beside one plain paper hot drink cup with a lid, both isolated and clearly visible on the counter",
            "sparsely stocked retail shelves with uniform solid-color unbranded boxes, every box face blank and turned away from camera",
            "street-facing transparent glass double entrance at left rear with visible metal frame, handles and threshold, "
            "dark blue rainy street, wet exterior pavement and rain streaks unmistakably visible beyond the glass",
            "completely dry clean interior floor, warm amber practical lights, blank unmarked wall and ceiling panels",
            structure_environment,
            "physically coherent perspective, practical fluorescent lighting, high detail",
        ]))
        structure_negative = ", ".join(filter(None, [
            "people, person, human, character, text, letters, numbers, words, labels, logo, brand, watermark",
            "menu board, price board, poster, paper notice, billboard, wall sign, illuminated sign, readable packaging",
            "printed package, colorful branded package, package label, receipt, advertisement, promotional card",
            "missing checkout counter, missing donation box, missing hot drink cup, missing glass doors, empty shelves",
            "opaque door, frosted door, rolling shutter, white shutter, display cabinet instead of entrance, "
            "adjacent indoor room beyond entrance, indoor shelves visible through entrance, laboratory, clinic, beaker, test tube",
            "bakery display case, pastry display, scattered coins, loose coins, rain inside, indoor rainfall, wet floor",
            "corridor, hallway, warehouse, kiosk, booth, low angle, ceiling-dominant composition",
        ]))
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": structural_checkpoint}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": structure_positive, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": structure_negative, "clip": ["1", 1]}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "5": {"class_type": "KSampler", "inputs": {
                "seed": actual_seed, "steps": 30, "cfg": 7.0, "sampler_name": "ddim",
                "scheduler": "normal", "denoise": 1.0, "model": ["1", 0],
                "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
            }},
            "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
            "7": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": selected_checkpoint}},
            "8": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["7", 1]}},
            "9": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["7", 1]}},
            "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["7", 2]}},
            "11": {"class_type": "KSampler", "inputs": {
                "seed": (actual_seed + 1) % (2 ** 32), "steps": 22, "cfg": 5.5,
                "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 0.62,
                "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0],
                "latent_image": ["10", 0],
            }},
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["7", 2]}},
            "13": {"class_type": "SaveImage", "inputs": {
                "filename_prefix": f"sceneref/{_safe(scene_id)}_{uuid.uuid4().hex[:8]}", "images": ["12", 0],
            }},
        }
    elif (social_device_room or convenience_layout) and layout_image_name:
        workflow = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": selected_checkpoint}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
            "4": {"class_type": "LoadImage", "inputs": {"image": layout_image_name}},
            "5": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": SOCIAL_LAYOUT_CONTROLNET}},
            "6": {"class_type": "ControlNetApplyAdvanced", "inputs": {
                "positive": ["2", 0], "negative": ["3", 0], "control_net": ["5", 0],
                "image": ["4", 0], "strength": layout_strength,
                "start_percent": 0.0, "end_percent": layout_end_percent,
            }},
            "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "8": {"class_type": "KSampler", "inputs": {
                "seed": actual_seed, "steps": steps, "cfg": cfg, "sampler_name": sampler_name,
                "scheduler": scheduler, "denoise": 1.0, "model": ["1", 0],
                "positive": ["6", 0], "negative": ["6", 1], "latent_image": ["7", 0],
            }},
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
            "10": {"class_type": "SaveImage", "inputs": {
                "filename_prefix": f"sceneref/{_safe(scene_id)}_{uuid.uuid4().hex[:8]}", "images": ["9", 0],
            }},
        }
    else:
        workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": selected_checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": actual_seed, "steps": steps, "cfg": cfg, "sampler_name": sampler_name,
            "scheduler": scheduler, "denoise": 1.0, "model": ["1", 0],
            "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
        }},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": f"sceneref/{_safe(scene_id)}_{uuid.uuid4().hex[:8]}", "images": ["6", 0],
        }},
        }
    return workflow, {
        "schema_version": 1, "asset_type": "scene", "scene_id": scene_id,
        "seed": actual_seed, "checkpoint": selected_checkpoint, "aspect_ratio": aspect,
        "width": width, "height": height, "positive_prompt": positive,
        "negative_prompt": negative, "model_positive_prompt": positive,
        "model_negative_prompt": negative, "sanitized_environment_prompt": environment,
        "composition_policy": composition, "prompt_format": "english_environment_tags_en",
        "compact_interior_lock": compact_interior,
        "environment_profile": (
            "social_mobile_gaming_room" if social_device_room else
            "convenience_store_layout" if convenience_layout else
            "generic_environment"
        ),
        "required_prop_lock": required_prop_lock,
        "required_device_count": int(device_count) if device_count.isdigit() else None,
        "seed_material_version": 6 if layout_image_name else 7,
        "layout_conditioning": "sd15_lineart_controlnet" if layout_image_name else "none",
        "layout_image": layout_image_name,
        "layout_controlnet": SOCIAL_LAYOUT_CONTROLNET if layout_image_name else None,
        "layout_strength": layout_strength if layout_image_name else None,
        "layout_end_percent": layout_end_percent if layout_image_name else None,
        "two_pass_structure": two_pass_structure,
        "structural_checkpoint": structural_checkpoint if two_pass_structure else None,
        "structural_positive_prompt": structure_positive if two_pass_structure else None,
        "structural_negative_prompt": structure_negative if two_pass_structure else None,
        "stylization_denoise": 0.62 if two_pass_structure else None,
        "text_policy": "no_text_model_output",
        "convenience_store_lock": convenience_store,
        "sampler_profile": {
            "sampler_name": sampler_name, "scheduler": scheduler,
            "steps": steps, "cfg": cfg, "width": width, "height": height,
        },
    }


def _api(endpoint: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        comfyui_server() + endpoint, data=data,
        headers={"Content-Type": "application/json"}, method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"ComfyUI API error {exc.code}: {exc.read().decode(errors='replace')[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ComfyUI unavailable: {exc}") from exc


def _history_output_images(
    result: Mapping[str, Any], graph: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return images from the graph's SaveImage nodes.

    Save node ids are an implementation detail and change whenever a
    conditioning branch gains nodes.  Reading a hard-coded id made the
    ControlNet workflow report failure even though ComfyUI completed and
    saved an image successfully.
    """
    outputs = result.get("outputs") if isinstance(result, Mapping) else {}
    if not isinstance(outputs, Mapping):
        return []
    save_ids = [
        str(node_id) for node_id, node in graph.items()
        if isinstance(node, Mapping) and node.get("class_type") == "SaveImage"
    ]
    images: list[dict[str, Any]] = []
    for node_id in save_ids:
        output = outputs.get(node_id) or {}
        if isinstance(output, Mapping):
            images.extend(item for item in output.get("images") or [] if isinstance(item, dict))
    if images:
        return images
    # Compatibility for server-side graph transforms that remap output ids.
    for output in outputs.values():
        if isinstance(output, Mapping):
            images.extend(item for item in output.get("images") or [] if isinstance(item, dict))
    return images


def generate_scene_asset(
    scene: Mapping[str, Any],
    visual_bible: Mapping[str, Any],
    *,
    story_hash: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    structural_checkpoint: str | None = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    api_func: Callable[[str, Optional[dict[str, Any]]], dict[str, Any]] = _api,
    timeout: float = 600.0,
) -> dict[str, Any]:
    source = "; ".join([
        str(scene.get("description") or ""),
        str(scene.get("model_prompt_en") or ""),
        str(scene.get("positive_prompt") or ""),
    ])
    root = comfyui_root()
    layout_dependencies_ready = (
        (root / "models" / "checkpoints" / SOCIAL_LAYOUT_CHECKPOINT).is_file()
        and (root / "models" / "controlnet" / SOCIAL_LAYOUT_CONTROLNET).is_file()
    )
    convenience_layout = bool(_CONVENIENCE_STORE_RE.search(source) and layout_dependencies_ready)
    layout_image_name = None
    if convenience_layout:
        layout_image_name = _prepare_convenience_store_layout_image(
            scene, visual_bible, story_hash
        )
    elif bool(scene.get("use_layout_control")) and layout_dependencies_ready:
        layout_image_name = _prepare_social_layout_image(scene, visual_bible, story_hash)
    graph, manifest = build_scene_reference_workflow(
        scene, visual_bible, story_hash=story_hash, checkpoint=checkpoint,
        layout_image_name=layout_image_name, structural_checkpoint=structural_checkpoint,
    )
    prompt_id = api_func("/prompt", {"prompt": graph})["prompt_id"]
    if progress_cb:
        progress_cb(f"场景 {manifest['scene_id']} 已提交：{prompt_id}")
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        history = api_func(f"/history/{prompt_id}", None)
        if prompt_id in history:
            result = history[prompt_id]
            if result.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"scene generation failed: {result.get('status')}")
            images = _history_output_images(result, graph)
            if not images:
                raise RuntimeError("scene generation completed without image output")
            item = images[0]
            source = comfyui_root() / "output" / item.get("subfolder", "") / item["filename"]
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = comfyui_root() / "input" / f"sceneref_{_safe(manifest['scene_id'])}_{uuid.uuid4().hex[:8]}{source.suffix.lower()}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest.update({
                "prompt_id": prompt_id, "graph": graph, "source_path": str(source),
                "reference_images": [str(destination)], "status": "completed",
            })
            return manifest
        time.sleep(2.0)
    raise TimeoutError(f"scene generation timed out after {timeout:.0f}s; prompt_id={prompt_id}")
