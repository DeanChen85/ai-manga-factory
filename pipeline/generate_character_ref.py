# -*- coding: utf-8 -*-
"""Identity-locked character reference generation through ComfyUI.

Character assets are derived from a versioned ``character_bible`` card.  A
stable story/character seed creates the anchor; all other views start from a
fresh latent while IPAdapter PLUS FACE preserves anchor identity.  The returned
manifest records the actual prompt, seed, anchor and conditioning mode so a task
store can persist and audit the job.
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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from prompt_contracts import build_character_reference_prompt, normalize_character_bible
from runtime_config import comfyui_root, comfyui_server


COMFYUI_SERVER = comfyui_server()
COMFYUI_ROOT = comfyui_root()
COMFYUI_INPUT = COMFYUI_ROOT / "input"
COMFYUI_OUTPUT = COMFYUI_ROOT / "output"

AVAILABLE_CHECKPOINTS = {
    "anything-v5": "anything-v5-PrtRE.safetensors",
    "animagine-xl-3.1": "animagine-xl-3.1.safetensors",
    "realvisxl-v5": "RealVisXL_V5.0_fp16.safetensors",
}
DEFAULT_CHECKPOINT = AVAILABLE_CHECKPOINTS["anything-v5"]
PRODUCTION_CHECKPOINT = AVAILABLE_CHECKPOINTS["animagine-xl-3.1"]

# Non-anchor views start from a fresh latent so the text prompt owns framing and
# pose. PLUS FACE carries only the anchor identity into that new composition.
IPADAPTER_PRESET = "PLUS FACE (portraits)"
VIEW_IPADAPTER_WEIGHT = {
    "正面": 0.90,
    "侧面": 0.95,
    "背面": 0.75,
    # At 0.90 PLUS FACE repeatedly copied the anchor's bust framing. Lowering
    # the full-body weight lets the explicit wide-shot/wardrobe prompt own the
    # composition while retaining the anchor as an identity signal.
    "全身": 0.65,
}
IPADAPTER_START_AT = 0.0
IPADAPTER_END_AT = 1.0


def comfyui_api(endpoint: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
    """Call the local ComfyUI API."""
    url = f"{COMFYUI_SERVER}{endpoint}"
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI API error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"ComfyUI API 连接失败: {exc}") from exc


def check_comfyui_ready() -> bool:
    try:
        comfyui_api("/system_stats")
        return True
    except (RuntimeError, OSError):
        return False


def stable_character_seed(character: dict[str, Any], story_hash: str = "") -> int:
    """Derive a deterministic uint32 seed from story identity and character ID."""
    material = "|".join([
        story_hash.strip(),
        str(character.get("character_id", "")).strip(),
        str(character.get("identity_prompt", "")).strip(),
        str(character.get("wardrobe_prompt", character.get("wardrobe_lock", ""))).strip(),
    ])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:4], "big")


def _character_card(character: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(character, dict):
        cards = normalize_character_bible([character])
    else:
        cards = normalize_character_bible([{
            "character_id": "char_legacy",
            "name": "legacy character",
            "identity_prompt": str(character),
        }])
    if not cards:
        raise ValueError("character bible card is required")
    card = cards[0]
    # These model-facing English tags are intentionally kept outside the
    # generic contract normalizer so older contracts remain compatible.
    if isinstance(character, dict):
        for field in ("model_identity_tags_en", "model_wardrobe_tags_en"):
            if character.get(field):
                card[field] = character[field]
    return card


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return token or "character"


def _tag_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{key}: {_tag_text(item)}" for key, item in value.items() if _tag_text(item))
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_tag_text(item) for item in value if _tag_text(item))
    return str(value or "").strip()


def _dedupe_tags(parts: Iterable[str]) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for part in parts:
        for token in re.split(r"[,;\n]+", str(part or "")):
            cleaned = re.sub(r"\s+", " ", token).strip(" .")
            key = cleaned.casefold()
            if cleaned and key not in seen:
                result.append(cleaned)
                seen.add(key)
    return ", ".join(result)


_HAIR_COLORS = ("black", "brown", "blonde", "white", "silver", "gray", "red", "orange", "blue", "green", "purple", "pink")
_CLOTHING_CONFLICTS = (
    "qipao", "cheongsam", "dress", "gown", "fantasy robe", "ornate fantasy costume",
    "fantasy armor", "school uniform", "maid outfit", "kimono", "hanfu",
)


def _gender_tag(source: str) -> tuple[str, list[str]]:
    lowered = source.casefold()
    female = bool(re.search(r"\b(?:1girl|girl|woman|female|lady|feminine)\b", lowered))
    male = bool(re.search(r"\b(?:1boy|boy|man|male|gentleman|masculine)\b", lowered))
    if male and not female:
        return "1boy, solo male", ["1girl", "woman", "female", "breasts", "feminine appearance"]
    if female and not male:
        return "1girl, solo female", ["1boy", "man", "male", "beard", "masculine appearance"]
    return "solo, single character", ["multiple people", "mixed-gender cast"]


def _dynamic_character_negative(identity: str, wardrobe: str, *, view: str) -> str:
    source = f"{identity}, {wardrobe}".casefold()
    _gender, opposite_gender = _gender_tag(source)
    expected_hair = {
        color for color in _HAIR_COLORS
        if re.search(rf"\b{re.escape(color)}\b(?=[^,;]{{0,32}}\bhair\b)", source)
    }
    wrong_hair = [f"{color} hair" for color in _HAIR_COLORS if color not in expected_hair]
    wrong_clothing = [item for item in _CLOTHING_CONFLICTS if item not in source]
    generic = [
        "wrong gender", "gender swap", "wrong hair color", "wrong hairstyle",
        "wrong outfit", "wardrobe variation", "clothing color change", "fantasy accessories",
        "multiple characters", "two characters", "two bodies", "twins", "duplicate person",
        "side-by-side people", "symmetrical duplication", "extra limbs", "deformed hands",
        "text", "letters", "logo", "watermark",
    ]
    if view in {"anchor", "全身", "full_body", "full body"}:
        generic.extend(["upper body", "bust portrait", "cropped legs", "cropped feet", "close-up"])
    return _dedupe_tags([*opposite_gender, *wrong_hair, *wrong_clothing, *generic])


def _anything_v5_prompts(
    card: Mapping[str, Any], visual: Mapping[str, Any], base_prompt: Mapping[str, str], *, view: str
) -> tuple[str, str, dict[str, Any]]:
    identity = _tag_text(card.get("model_identity_tags_en")) or _tag_text(card.get("identity_prompt"))
    wardrobe = _tag_text(card.get("model_wardrobe_tags_en")) or _tag_text(
        card.get("wardrobe_prompt") or card.get("wardrobe_lock")
    )
    gender, _opposite = _gender_tag(f"{identity}, {wardrobe}")
    view_tags = {
        "anchor": "front view, full body, head to toe, feet visible, neutral standing pose, centered, looking at viewer",
        "正面": "front view, standing, neutral pose, symmetrical character reference",
        "侧面": "side view, profile, standing, neutral pose, character reference",
        "背面": "back view, from behind, standing, neutral pose, character reference",
        "全身": "full body, wide shot, head to toe, feet and shoes visible, standing, centered",
        "front": "front view, standing, neutral pose, symmetrical character reference",
        "side": "side view, profile, standing, neutral pose, character reference",
        "back": "back view, from behind, standing, neutral pose, character reference",
        "full_body": "full body, wide shot, head to toe, feet and shoes visible, standing, centered",
    }.get(view, f"{view}, neutral character reference")
    framing_priority = (
        "(solo:1.6), (single character only:1.55), "
        "(one full-body person centered in frame:1.5), "
        "(head-to-toe standing pose with feet visible:1.35)"
        if view == "anchor" else ""
    )
    positive = _dedupe_tags([
        "masterpiece, best quality, high quality",
        gender,
        identity,
        framing_priority,
        wardrobe,
        _tag_text(card.get("signature_features")),
        view_tags,
        "anime character reference, clean lineart",
        _tag_text(visual.get("style_prompt")),
        "solo, single character, plain neutral background, consistent design, no text",
    ])
    dynamic_negative = _dynamic_character_negative(identity, wardrobe, view=view)
    negative = _dedupe_tags([base_prompt.get("negative_prompt", ""), dynamic_negative])
    return positive, negative, {
        "format": "anything_v5_danbooru_en",
        "identity_tags_en": identity,
        "wardrobe_tags_en": wardrobe,
        "gender_tag": gender,
        "view_tags": view_tags,
        "dynamic_negative": dynamic_negative,
    }


_ANIMAGINE_NEGATIVE = (
    "nsfw, lowres, bad, text, error, fewer, extra, missing, worst quality, "
    "jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, "
    "chromatic aberration, signature, extra digits, artistic error, username, scan, abstract"
)


def _animagine_xl31_prompts(
    card: Mapping[str, Any], visual: Mapping[str, Any], *, view: str
) -> tuple[str, str, dict[str, Any]]:
    """Build the Danbooru-style contract recommended by Animagine XL 3.1."""
    identity = _tag_text(card.get("model_identity_tags_en")) or _tag_text(card.get("identity_prompt"))
    wardrobe = _tag_text(card.get("model_wardrobe_tags_en")) or _tag_text(
        card.get("wardrobe_prompt") or card.get("wardrobe_lock")
    )
    gender, _opposite = _gender_tag(f"{identity}, {wardrobe}")
    view_tags = {
        "anchor": "full body, standing, front view, looking at viewer, head to toe, feet visible",
        "正面": "full body, standing, front view, looking at viewer, head to toe, feet visible",
        "侧面": "full body, standing, side view, profile, head to toe, feet visible",
        "背面": "full body, standing, back view, from behind, head to toe, feet visible",
        "全身": "full body, standing, wide shot, head to toe, feet visible",
        "front": "full body, standing, front view, looking at viewer, head to toe, feet visible",
        "side": "full body, standing, side view, profile, head to toe, feet visible",
        "back": "full body, standing, back view, from behind, head to toe, feet visible",
        "full_body": "full body, standing, wide shot, head to toe, feet visible",
    }.get(view, f"full body, standing, {view}, head to toe, feet visible")
    # Only printable ASCII reaches this tag-trained checkpoint.  Editorial
    # Chinese remains visible in Web but cannot corrupt CLIP conditioning.
    model_style = _tag_text(visual.get("style_prompt"))
    if not model_style.isascii() or not model_style.isprintable():
        model_style = "modern Japanese anime, clean lineart, consistent character design"
    proportion_tag = (
        "adult proportions" if re.search(r"\b(?:adult|man|woman|male|female)\b", identity, re.I)
        else "natural human proportions"
    )
    positive = _dedupe_tags([
        gender,
        identity,
        wardrobe,
        _tag_text(card.get("signature_features")),
        view_tags,
        f"single character, solo, {proportion_tag}, detailed face, symmetrical eyes",
        "plain light gray background, simple background, soft even studio lighting",
        model_style,
        "masterpiece, best quality, very aesthetic, absurdres, safe, newest",
    ])
    dynamic_negative = _dynamic_character_negative(identity, wardrobe, view=view)
    adult_negative = (
        "chibi, super deformed, child, loli, child proportions"
        if re.search(r"\b(?:adult|man|woman|male|female)\b", identity, re.I) else ""
    )
    footwear_negative = (
        "barefoot, bare feet, missing shoes"
        if re.search(r"\b(?:shoes?|sneakers?|boots?|loafers?|heels?)\b", wardrobe, re.I)
        else ""
    )
    negative = _dedupe_tags([
        _ANIMAGINE_NEGATIVE,
        dynamic_negative,
        adult_negative,
        footwear_negative,
        "cropped feet, feet out of frame, busy background, colorful background, dramatic background",
    ])
    return positive, negative, {
        "format": "animagine_xl_31_danbooru_en",
        "source_guidance": "cagliostrolab/animagine-xl-3.1 model card",
        "identity_tags_en": identity,
        "wardrobe_tags_en": wardrobe,
        "view_tags": view_tags,
        "quality_tags": "masterpiece, best quality, very aesthetic, absurdres, safe, newest",
        "dynamic_negative": dynamic_negative,
    }


def build_character_reference_workflow(
    character: str | dict[str, Any],
    visual_bible: Optional[dict[str, Any]] = None,
    *,
    view: str = "anchor",
    checkpoint: str = DEFAULT_CHECKPOINT,
    story_hash: str = "",
    seed: Optional[int] = None,
    anchor_image: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a Comfy graph plus an auditable manifest without submitting it."""
    card = _character_card(character)
    visual = dict(visual_bible or {})
    source_prompt = build_character_reference_prompt(card, visual, view=view)
    checkpoint_name = Path(checkpoint).name.casefold()
    if checkpoint_name.startswith("anything-v5"):
        positive_prompt, negative_prompt, prompt_metadata = _anything_v5_prompts(
            card, visual, source_prompt, view=view
        )
    elif checkpoint_name.startswith("animagine-xl-3.1"):
        positive_prompt, negative_prompt, prompt_metadata = _animagine_xl31_prompts(
            card, visual, view=view
        )
    else:
        identity = _tag_text(card.get("model_identity_tags_en")) or _tag_text(card.get("identity_prompt"))
        wardrobe = _tag_text(card.get("model_wardrobe_tags_en")) or _tag_text(card.get("wardrobe_prompt"))
        positive_prompt = source_prompt["positive_prompt"]
        negative_prompt = _dedupe_tags([
            source_prompt["negative_prompt"],
            _dynamic_character_negative(identity, wardrobe, view=view),
        ])
        prompt_metadata = {"format": "natural_language", "identity_tags_en": identity, "wardrobe_tags_en": wardrobe}
    base_seed = stable_character_seed(card, story_hash) if seed is None else int(seed)
    conditioned = bool(anchor_image)
    actual_seed = base_seed
    if conditioned:
        # Deterministic per-view variation avoids repeating the anchor pose while
        # preserving reproducibility for a story/character/view tuple.
        view_offset = int.from_bytes(hashlib.sha256(view.encode("utf-8")).digest()[:4], "big")
        actual_seed = (base_seed + view_offset) % (2**32)
    denoise = 1.0
    # A 2:3 anchor repeatedly encouraged knee-crops or two side-by-side
    # figures.  The approval bundle is anchored by one canonical head-to-toe
    # subject, so anchor and full-body use a narrow 1:2 frame.
    if checkpoint_name.startswith("animagine-xl-3.1"):
        width, height = (
            (768, 1344) if view in {"anchor", "全身", "full_body"}
            else (896, 1152)
        )
        sampler_name, scheduler, steps, cfg = "euler_ancestral", "normal", 28, 6.0
    else:
        width, height = (512, 1024) if view in {"anchor", "全身", "full_body"} else (640, 960)
        sampler_name, scheduler, steps, cfg = "dpmpp_2m", "karras", (35 if not conditioned else 32), 7.0

    identity_nodes: dict[str, Any] = {}
    sampler_model = ["4", 0]
    if conditioned:
        ipadapter_weight = VIEW_IPADAPTER_WEIGHT.get(view, 0.80)
        identity_nodes = {
            "10": {
                "class_type": "LoadImage",
                "inputs": {"image": str(anchor_image), "upload": "image"},
            },
            "12": {
                "class_type": "IPAdapterUnifiedLoader",
                "inputs": {
                    "model": ["4", 0],
                    "preset": IPADAPTER_PRESET,
                },
            },
            "13": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["12", 0],
                    "ipadapter": ["12", 1],
                    "image": ["10", 0],
                    "weight": ipadapter_weight,
                    "weight_type": "ease out",
                    "combine_embeds": "concat",
                    "start_at": IPADAPTER_START_AT,
                    "end_at": IPADAPTER_END_AT,
                    "embeds_scaling": "V only",
                },
            },
        }
        sampler_model = ["13", 0]

    char_id = _safe_token(card["character_id"])
    view_token = _safe_token(view)
    workflow: dict[str, Any] = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": sampler_model,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": f"charref/{char_id}_{view_token}_{uuid.uuid4().hex[:8]}",
                "images": ["8", 0],
            },
        },
        **identity_nodes,
    }
    manifest = {
        "character_id": card["character_id"],
        "view": view,
        "checkpoint": checkpoint,
        "base_seed": base_seed,
        "seed": actual_seed,
        "anchor_image": str(anchor_image or ""),
        "conditioning_mode": "anchor_ipadapter_plus_face" if conditioned else "text_to_image_anchor",
        "identity_adapter_preset": IPADAPTER_PRESET if conditioned else "",
        "identity_adapter_weight": identity_nodes.get("13", {}).get("inputs", {}).get("weight"),
        "identity_adapter_start_at": IPADAPTER_START_AT if conditioned else None,
        "identity_adapter_end_at": IPADAPTER_END_AT if conditioned else None,
        "latent_source": "empty_latent",
        "conditioning_fallback": "disabled_fail_closed" if conditioned else "not_applicable",
        "denoise": workflow["3"]["inputs"]["denoise"],
        "prompt_format": prompt_metadata["format"],
        "prompt_metadata": prompt_metadata,
        "source_positive_prompt": source_prompt["positive_prompt"],
        "source_negative_prompt": source_prompt["negative_prompt"],
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "sampler_profile": {
            "sampler_name": sampler_name, "scheduler": scheduler,
            "steps": steps, "cfg": cfg, "width": width, "height": height,
        },
    }
    return workflow, manifest


def _wait_for_image(prompt_id: str, max_wait: int = 300) -> Path:
    started = time.time()
    while time.time() - started < max_wait:
        history = comfyui_api(f"/history/{prompt_id}")
        if prompt_id in history:
            result = history[prompt_id]
            if result.get("status", {}).get("status_str") == "error":
                raise RuntimeError(f"生成失败: {result['status'].get('messages', [])}")
            images = result.get("outputs", {}).get("9", {}).get("images", [])
            if images:
                item = images[0]
                return COMFYUI_OUTPUT / item.get("subfolder", "") / item["filename"]
        time.sleep(2)
    raise RuntimeError(f"生成超时（{max_wait}秒）")


def _submit_reference(
    character: str | dict[str, Any],
    visual_bible: Optional[dict[str, Any]],
    *,
    view: str,
    checkpoint: str,
    story_hash: str,
    seed: Optional[int],
    anchor_image: Optional[str],
    progress_cb: Optional[Callable[[str], None]],
) -> dict[str, Any]:
    if not check_comfyui_ready():
        raise RuntimeError("ComfyUI 未启动，请先启动 ComfyUI")
    workflow, manifest = build_character_reference_workflow(
        character,
        visual_bible,
        view=view,
        checkpoint=checkpoint,
        story_hash=story_hash,
        seed=seed,
        anchor_image=anchor_image,
    )
    if progress_cb:
        progress_cb(f"提交 {manifest['character_id']} / {view}，seed={manifest['seed']}")
    prompt_id = comfyui_api("/prompt", {"prompt": workflow})["prompt_id"]
    output_path = _wait_for_image(prompt_id)
    COMFYUI_INPUT.mkdir(parents=True, exist_ok=True)
    dest_name = f"charref_{_safe_token(manifest['character_id'])}_{_safe_token(view)}_{uuid.uuid4().hex[:8]}.png"
    dest_path = COMFYUI_INPUT / dest_name
    shutil.copy2(output_path, dest_path)
    manifest.update({
        "prompt_id": prompt_id,
        "source_path": str(output_path),
        "path": str(dest_path),
        "filename": dest_name,
        "status": "completed",
    })
    if progress_cb:
        progress_cb(f"完成 {view}: {dest_name}")
    return manifest


def generate_character_anchor(
    character_desc: str | dict[str, Any],
    style_prefix: str = "",
    checkpoint: str = DEFAULT_CHECKPOINT,
    progress_cb: Optional[Callable[[str], None]] = None,
    *,
    visual_bible: Optional[dict[str, Any]] = None,
    story_hash: str = "",
    seed: Optional[int] = None,
    return_manifest: bool = False,
) -> str | dict[str, Any]:
    visual = dict(visual_bible or {})
    if style_prefix and not visual.get("style_prompt"):
        visual["style_prompt"] = style_prefix
    manifest = _submit_reference(
        character_desc,
        visual,
        view="anchor",
        checkpoint=checkpoint,
        story_hash=story_hash,
        seed=seed,
        anchor_image=None,
        progress_cb=progress_cb,
    )
    return manifest if return_manifest else manifest["filename"]


def generate_character_sheet(
    character_desc: str | dict[str, Any],
    style_prefix: str = "",
    angles: Optional[List[str]] = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    progress_cb: Optional[Callable[[str], None]] = None,
    *,
    visual_bible: Optional[dict[str, Any]] = None,
    story_hash: str = "",
    seed: Optional[int] = None,
    anchor_image: Optional[str] = None,
    allow_unconditioned: bool = False,
    return_manifest: bool = False,
) -> Dict[str, Any]:
    """Generate views conditioned on a previously generated anchor image."""
    if not anchor_image and not allow_unconditioned:
        raise ValueError("anchor_image is required; independent random multi-view generation is disabled")
    visual = dict(visual_bible or {})
    if style_prefix and not visual.get("style_prompt"):
        visual["style_prompt"] = style_prefix
    results: Dict[str, Any] = {}
    for angle in angles or ["正面", "侧面", "全身"]:
        manifest = _submit_reference(
            character_desc,
            visual,
            view=angle,
            checkpoint=checkpoint,
            story_hash=story_hash,
            seed=seed,
            anchor_image=anchor_image,
            progress_cb=progress_cb,
        )
        results[angle] = manifest if return_manifest else manifest["filename"]
    return results


def derive_canonical_reference_crops(
    anchor_path: str | Path,
    *,
    character_id: str,
) -> dict[str, dict[str, Any]]:
    """Create identity-safe reference crops without asking diffusion to redraw.

    Real target testing showed that PLUS FACE can preserve a face while freely
    changing clothing or adding extra figures.  Production therefore uses one
    approved full-body canonical render plus deterministic portrait/torso crops.
    Generated angle variants remain available through ``generate_character_sheet``
    for experimentation, but are not part of the default approval bundle.
    """
    from PIL import Image

    source = Path(anchor_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"canonical anchor is missing: {source}")
    COMFYUI_INPUT.mkdir(parents=True, exist_ok=True)
    definitions = {
        "portrait_crop": (0.18, 0.02, 0.82, 0.48),
        "torso_crop": (0.10, 0.00, 0.90, 0.72),
    }
    derived: dict[str, dict[str, Any]] = {}
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        for role, (left, top, right, bottom) in definitions.items():
            crop_box = (
                round(width * left), round(height * top),
                round(width * right), round(height * bottom),
            )
            crop = image.crop(crop_box)
            destination = COMFYUI_INPUT / (
                f"charref_{_safe_token(character_id)}_{role}_{uuid.uuid4().hex[:8]}.png"
            )
            crop.save(destination, format="PNG", optimize=True)
            derived[role] = {
                "character_id": character_id,
                "view": role,
                "conditioning_mode": "deterministic_crop_from_canonical_anchor",
                "source_anchor_path": str(source),
                "crop_box": list(crop_box),
                "path": str(destination),
                "filename": destination.name,
                "width": crop.width,
                "height": crop.height,
                "status": "completed",
            }
    return derived


def generate_character_assets(
    character: dict[str, Any],
    visual_bible: dict[str, Any],
    *,
    story_hash: str,
    checkpoint: str = PRODUCTION_CHECKPOINT,
    angles: Optional[List[str]] = None,
    generate_angle_variants: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Generate anchor + conditioned views and return a persistence manifest."""
    seed = stable_character_seed(_character_card(character), story_hash)
    anchor = generate_character_anchor(
        character,
        checkpoint=checkpoint,
        visual_bible=visual_bible,
        story_hash=story_hash,
        seed=seed,
        progress_cb=progress_cb,
        return_manifest=True,
    )
    if generate_angle_variants:
        views = generate_character_sheet(
            character,
            checkpoint=checkpoint,
            visual_bible=visual_bible,
            story_hash=story_hash,
            seed=seed,
            anchor_image=anchor["filename"],
            angles=angles,
            progress_cb=progress_cb,
            return_manifest=True,
        )
        bundle_mode = "generated_ipadapter_angle_variants"
    else:
        views = derive_canonical_reference_crops(
            anchor["path"], character_id=anchor["character_id"]
        )
        bundle_mode = "canonical_anchor_plus_deterministic_crops"
    reference_images = [anchor["path"], *[item["path"] for item in views.values()]]
    return {
        "character_id": anchor["character_id"],
        "story_hash": story_hash,
        "seed": seed,
        "anchor": anchor,
        "views": views,
        "reference_images": reference_images,
        "reference_bundle_mode": bundle_mode,
        "status": "completed",
    }


if __name__ == "__main__":
    raise SystemExit("Import this module through the task worker; direct GPU test execution is disabled.")
