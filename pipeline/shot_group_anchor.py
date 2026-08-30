"""Generate and approve per-shot composition anchors.

The H3 opening frame is a stronger composition authority than detached
character references.  This module builds auditable vertical storyboard
frames from the approved scene plate and approved character references.  A
single-character state-changing shot receives distinct opening/final anchors
so H3 is not told that an identical image is both ends of the action.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from runtime_config import comfyui_root, comfyui_server, projects_dir
from task_store import RenderJobStore, default_store


GROUP_ANCHOR_CONTRACT = "shot-composition-anchor/v12-closed-hand-contact"
DEFAULT_CHECKPOINT = "animagine-xl-3.1.safetensors"
WIDTH = 768
HEIGHT = 1344


def _action_spec(panel: Mapping[str, Any]) -> dict[str, str]:
    raw = panel.get("action_spec") if isinstance(panel.get("action_spec"), Mapping) else {}
    return {
        "action_code": str(raw.get("action_code") or panel.get("action_code") or "").strip().upper(),
        "target": str(raw.get("target") or "").strip(),
        "start_state": str(
            raw.get("start_state") or panel.get("first_frame") or panel.get("first_state") or ""
        ).strip(),
        "end_state": str(
            raw.get("end_state") or panel.get("last_frame") or panel.get("final_state") or ""
        ).strip(),
    }


def requires_paired_state_anchor(
    panel: Mapping[str, Any], character_count: int | None = None,
) -> bool:
    """Return the canonical first/final anchor requirement for a shot.

    Existing single-character gates always authored two visibly different
    states.  A two-character shot also requires them when the canonical action
    contract describes a real state transition (for example HAND_OBJECT).
    """
    count = character_count
    if count is None:
        count = len([value for value in panel.get("character_ids") or [] if value])
    if count == 1:
        return True
    if count != 2:
        return False
    action = _action_spec(panel)
    first = re.sub(r"\s+", " ", action["start_state"].lower()).strip()
    last = re.sub(r"\s+", " ", action["end_state"].lower()).strip()
    return bool(action["action_code"] and first and last and first != last)


def panel_anchor_contract_sha256(panel: Mapping[str, Any]) -> str:
    """Hash every panel field that makes a reviewed group anchor authoritative."""
    payload = {
        "schema": GROUP_ANCHOR_CONTRACT,
        "panel_id": str(panel.get("panel_id") or "").strip(),
        "character_ids": [str(value) for value in panel.get("character_ids") or []],
        "scene_id": str(panel.get("scene_id") or "").strip(),
        "action": _action_spec(panel),
        "camera_plan": panel.get("camera_plan") if isinstance(panel.get("camera_plan"), Mapping) else {},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def requires_approved_group_anchor(
    panel: Mapping[str, Any], metadata: Mapping[str, Any] | None = None,
    character_count: int | None = None,
) -> bool:
    """Fail closed when state or the latest visual rejection needs composition authority."""
    count = character_count
    if count is None:
        count = len([value for value in panel.get("character_ids") or [] if value])
    if count not in {1, 2}:
        return False
    action = _action_spec(panel)
    if bool(panel.get("require_group_anchor")) or action["action_code"] == "HAND_OBJECT":
        return True
    audit = (metadata or {}).get("qa_rejection_audit")
    if not isinstance(audit, list) or not audit or not isinstance(audit[-1], Mapping):
        return False
    category = str(audit[-1].get("category") or "legacy_unclassified").strip()
    return category != "action_timing_or_edit_window"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _snapshot(ep_id: str, store: RenderJobStore) -> dict[str, Any]:
    """Read the minimum project state from the injected durable store."""
    project = (projects_dir() / ep_id).resolve()
    episode_path = project / "episode.json"
    if not episode_path.is_file():
        raise FileNotFoundError(f"episode contract is missing: {episode_path}")
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    if not isinstance(episode, dict):
        raise ValueError("episode contract must be a JSON object")
    return {
        "project_dir": str(project),
        "episode": episode,
        "assets": {"items": store.list_assets(ep_id)},
        "pipeline": store.get_pipeline(ep_id) or {},
    }


def _project_file(path: str | Path, project: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or not resolved.is_relative_to(project):
        raise RuntimeError(f"{label} must be a regular file inside the episode project")
    return resolved


def _api(path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        comfyui_server().rstrip("/") + path,
        data=(json.dumps(payload).encode("utf-8") if payload is not None else None),
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("ComfyUI returned a non-object response")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "shot"


def _copy_to_comfy_input(source: Path, role: str) -> str:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    name = f"group_{_sha256(source)[:12]}_{_safe(role)}{source.suffix.lower()}"
    destination = (comfyui_root() / "input" / name).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file() or _sha256(destination) != _sha256(source):
        shutil.copy2(source, destination)
    return destination.name


def _approved_character_cutout(source: Path) -> Image.Image:
    """Remove the nearly uniform reference-sheet backdrop deterministically."""
    rgb = Image.open(source).convert("RGB")
    pixels = np.asarray(rgb, dtype=np.float32)
    border = np.concatenate((
        pixels[:8].reshape(-1, 3), pixels[-8:].reshape(-1, 3),
        pixels[:, :8].reshape(-1, 3), pixels[:, -8:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    distance = np.sqrt(np.sum((pixels - background) ** 2, axis=2))
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    score = np.maximum(distance, saturation * 1.7)
    mean = pixels.mean(axis=2)
    alpha = np.clip((score - 12.0) * (255.0 / 32.0), 0, 255).astype(np.uint8)
    subject_signal = (alpha > 12) & ((saturation > 20) | (mean < 185))
    support = np.asarray(
        Image.fromarray((subject_signal * 255).astype(np.uint8), mode="L").filter(
            ImageFilter.MaxFilter(31)
        )
    )
    unsupported_neutral = (saturation < 22) & (mean > 180) & (support < 128)
    alpha[unsupported_neutral] = 0
    alpha_image = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(0.8))
    cutout = rgb.convert("RGBA")
    cutout.putalpha(alpha_image)
    ys, xs = np.nonzero(subject_signal)
    bbox = None
    if len(xs):
        padding = 8
        bbox = (
            max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
            min(rgb.width, int(xs.max()) + padding + 1),
            min(rgb.height, int(ys.max()) + padding + 1),
        )
    if not bbox:
        raise RuntimeError(f"approved character reference has no detectable subject: {source}")
    return cutout.crop(bbox)


def _resize_height(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _paste_subject(base: Image.Image, subject: Image.Image, xy: tuple[int, int]) -> None:
    alpha = subject.getchannel("A")
    shadow = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_alpha = Image.new("L", base.size, 0)
    shadow_alpha.paste(alpha, (xy[0] + 10, xy[1] + 14))
    shadow.putalpha(shadow_alpha.filter(ImageFilter.GaussianBlur(12)))
    dark = Image.new("RGBA", base.size, (12, 20, 32, 60))
    dark.putalpha(shadow.getchannel("A").point(lambda value: min(70, value)))
    base.alpha_composite(dark)
    base.alpha_composite(subject, dest=xy)


def _draw_action_target(
    base: Image.Image, panel: Mapping[str, Any], *, state: str,
) -> None:
    """Draw the canonical prop state used as temporal guidance, never text."""
    action = _action_spec(panel)
    if action["action_code"] != "HAND_OBJECT":
        return
    target = action["target"].lower()
    if not target:
        raise ValueError("HAND_OBJECT shot anchor requires action_spec.target")
    if not any(token in target for token in (
        "wallet", "purse", "phone", "cup", "drink", "coin", "change",
        "box", "package", "parcel", "card", "ticket", "document", "key",
    )):
        raise ValueError(f"unsupported deterministic HAND_OBJECT target: {action['target']}")

    # The caller places the approved, unmodified character hands around these
    # coordinates. The opening prop touches the clerk's hand; the final prop
    # spans the small gap between the clerk and rider hands.
    # These coordinates are derived from the deterministic two-person layout
    # below.  The prop is composited *over* the approved source hand pixels so
    # the opening visibly touches the clerk palm and the final state bridges
    # the clerk and rider palms rather than floating at waist/floor height.
    center_x, center_y = ((260, 675) if state == "first" else (322, 680))
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if any(token in target for token in ("wallet", "purse")):
        # The final prop bridges the approved clerk and rider hand pixels.  A
        # shorter rectangle left a visible air gap beside the rider's hand and
        # caused H3 to invent a second wallet on the floor while animating the
        # transfer.  This wider final-state authority remains exactly one prop.
        half_width = 46 if state == "first" else 80
        box = (center_x - half_width, center_y - 29, center_x + half_width, center_y + 29)
        draw.rounded_rectangle(
            box, radius=9, fill=(18, 21, 27, 255), outline=(65, 59, 57, 255), width=5,
        )
        draw.line(
            (box[0] + 9, center_y - 7, box[2] - 9, center_y - 7),
            fill=(112, 96, 77, 255), width=3,
        )
        draw.ellipse(
            (center_x + 21, center_y - 12, center_x + 29, center_y - 4),
            fill=(150, 126, 88, 255),
        )
    elif "phone" in target:
        draw.rounded_rectangle(
            (center_x - 26, center_y - 47, center_x + 26, center_y + 47),
            radius=8, fill=(18, 23, 31, 255), outline=(200, 211, 222, 255), width=4,
        )
    elif any(token in target for token in ("cup", "drink")):
        draw.rounded_rectangle(
            (center_x - 34, center_y - 43, center_x + 30, center_y + 40),
            radius=8, fill=(194, 120, 67, 255), outline=(244, 225, 194, 255), width=4,
        )
        draw.ellipse((center_x - 34, center_y - 49, center_x + 30, center_y - 34), fill=(238, 223, 197, 255))
    elif any(token in target for token in ("coin", "change", "key")):
        draw.ellipse(
            (center_x - 28, center_y - 28, center_x + 28, center_y + 28),
            fill=(211, 166, 57, 255), outline=(255, 233, 151, 255), width=4,
        )
        if "key" in target:
            draw.rounded_rectangle(
                (center_x + 22, center_y - 7, center_x + 76, center_y + 7),
                radius=5, fill=(211, 166, 57, 255),
            )
    else:
        draw.rounded_rectangle(
            (center_x - 48, center_y - 34, center_x + 48, center_y + 34),
            radius=8, fill=(156, 107, 58, 255), outline=(241, 211, 154, 255), width=4,
        )
    shadow = overlay.getchannel("A").filter(ImageFilter.GaussianBlur(9))
    shifted_shadow = Image.new("L", base.size, 0)
    shifted_shadow.paste(shadow, (7, 10))
    shadow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    shadow_layer.putalpha(shifted_shadow.point(lambda value: min(85, value)))
    base.alpha_composite(shadow_layer)
    base.alpha_composite(overlay)


def _compose_approved_group_anchor(
    scene_path: Path, character_paths: list[Path], destination: Path,
    panel: Mapping[str, Any], *, state: str = "first",
) -> None:
    """Place the exact approved cast into the exact approved scene.

    The second character is placed behind the checkout counter and the first
    character remains a large foreground subject facing them.  This avoids the
    identity swaps and missing cast observed when diffusion authored the anchor.
    """
    if len(character_paths) not in {1, 2}:
        raise ValueError("deterministic shot anchor requires one or two characters")
    if state not in {"first", "last"}:
        raise ValueError("shot anchor state must be first or last")
    scene = Image.open(scene_path).convert("RGB")
    if len(character_paths) == 1:
        crop_top = min(max(0, round(scene.height * 0.14)), scene.height - 16)
        background = scene.crop((0, crop_top, scene.width, scene.height)).resize(
            (WIDTH, HEIGHT), Image.Resampling.LANCZOS,
        ).convert("RGBA")
        door = Image.new("RGBA", (280, 772), (18, 52, 92, 215))
        draw = ImageDraw.Draw(door)
        if state == "first":
            for offset in range(-500, 620, 34):
                draw.line(
                    (offset + 80, 12, offset + 15, 760),
                    fill=(178, 215, 245, 115), width=3,
                )
        draw.line((3, 0, 3, 772), fill=(210, 224, 236, 220), width=7)
        draw.line((276, 0, 276, 772), fill=(210, 224, 236, 220), width=7)
        draw.line((140, 0, 140, 772), fill=(135, 164, 190, 170), width=4)
        door_x = 252 if state == "first" else 470
        background.alpha_composite(door, dest=(door_x, 238))
        rider = _resize_height(_approved_character_cutout(character_paths[0]), 790)
        rider_x = (WIDTH - rider.width) // 2 if state == "first" else 42
        _paste_subject(background, rider, (rider_x, 330))
        destination.parent.mkdir(parents=True, exist_ok=True)
        background.convert("RGB").save(destination, format="PNG", optimize=True)
        return
    crop_top = min(max(0, round(scene.height * 0.25)), scene.height - 16)
    background = scene.crop((0, crop_top, scene.width, scene.height)).resize(
        (WIDTH, HEIGHT), Image.Resampling.LANCZOS,
    ).convert("RGBA")
    counter_occlusion = background.crop((0, 535, 390, 1135))
    clerk = _resize_height(_approved_character_cutout(character_paths[1]), 820)
    rider = _resize_height(_approved_character_cutout(character_paths[0]), 900)
    clerk_x, clerk_y = 90, 170
    rider_x, rider_y = ((430, 270) if state == "first" else (240, 180))
    _paste_subject(background, clerk, (clerk_x, clerk_y))
    # Reapply the approved counter foreground so the clerk is visibly behind it.
    background.alpha_composite(counter_occlusion, dest=(0, 535))
    # Bring only the clerk's approved right forearm/hand back over the counter.
    # This uses source pixels, avoiding an invented or duplicated limb.
    clerk_arm = clerk.crop((135, 225, clerk.width, 585))
    background.alpha_composite(clerk_arm, dest=(clerk_x + 135, clerk_y + 225))
    _paste_subject(background, rider, (rider_x, rider_y))
    _draw_action_target(background, panel, state=state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    background.convert("RGB").save(destination, format="PNG", optimize=True)


def build_group_anchor_workflow(
    *,
    scene_image: str,
    character_images: list[str],
    positive_prompt: str,
    negative_prompt: str,
    filename_prefix: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    seed: int = 0,
) -> dict[str, Any]:
    """Return an auditable Comfy graph for exactly two regional identities."""
    if len(character_images) != 2:
        raise ValueError("group anchor requires exactly two character images")
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage", "inputs": {"image": scene_image, "upload": "image"}},
        "3": {"class_type": "ImageScale", "inputs": {
            "image": ["2", 0], "upscale_method": "lanczos",
            "width": WIDTH, "height": HEIGHT, "crop": "center",
        }},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["1", 2]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": ["1", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 1]}},
        "7": {"class_type": "IPAdapterUnifiedLoader", "inputs": {
            "model": ["1", 0], "preset": "PLUS FACE (portraits)",
        }},
        "8": {"class_type": "LoadImage", "inputs": {"image": character_images[0], "upload": "image"}},
        "9": {"class_type": "MaskRectAreaAdvanced", "inputs": {
            "x": 0, "y": 110, "width": 384, "height": 1200,
            "image_width": WIDTH, "image_height": HEIGHT, "blur_radius": 48,
        }},
        "10": {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["7", 0], "ipadapter": ["7", 1], "image": ["8", 0],
            "weight": 1.0, "weight_type": "ease out", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 0.9, "embeds_scaling": "V only",
            "attn_mask": ["9", 0],
        }},
        "11": {"class_type": "LoadImage", "inputs": {"image": character_images[1], "upload": "image"}},
        "12": {"class_type": "MaskRectAreaAdvanced", "inputs": {
            "x": 384, "y": 110, "width": 384, "height": 1200,
            "image_width": WIDTH, "image_height": HEIGHT, "blur_radius": 48,
        }},
        "13": {"class_type": "IPAdapterAdvanced", "inputs": {
            "model": ["10", 0], "ipadapter": ["7", 1], "image": ["11", 0],
            "weight": 1.0, "weight_type": "ease out", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": 0.9, "embeds_scaling": "V only",
            "attn_mask": ["12", 0],
        }},
        "14": {"class_type": "KSampler", "inputs": {
            "seed": int(seed) % (2**32), "steps": 30, "cfg": 6.5,
            "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 0.95,
            "model": ["13", 0], "positive": ["5", 0], "negative": ["6", 0],
            "latent_image": ["4", 0],
        }},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["1", 2]}},
        "16": {"class_type": "SaveImage", "inputs": {
            "filename_prefix": filename_prefix, "images": ["15", 0],
        }},
    }


def _identity_text(card: Mapping[str, Any]) -> str:
    fields = (
        card.get("model_identity_tags_en"), card.get("identity_prompt"),
        card.get("model_wardrobe_tags_en"), card.get("wardrobe_prompt"),
    )
    return ", ".join(str(value).strip() for value in fields if str(value or "").strip())


def compile_group_anchor_prompts(
    panel: Mapping[str, Any], characters: list[Mapping[str, Any]],
    scene: Mapping[str, Any], visual: Mapping[str, Any],
    *, review_feedback: str = "",
) -> tuple[str, str]:
    if len(characters) not in {1, 2}:
        raise ValueError("shot composition anchor prompt requires one or two character cards")
    camera = panel.get("camera_plan") if isinstance(panel.get("camera_plan"), Mapping) else {}
    if len(characters) == 1:
        cast_prompt = (
            f"exactly one adult person visible, face and complete body clear, "
            f"the person occupies most of the frame: {_identity_text(characters[0])}, "
            "the person stands at the open glass entrance with visible blue rain outside"
        )
        cast_negative = "missing person, extra person, crowd, duplicate person"
    else:
        cast_prompt = ", ".join([
            "exactly two distinct people visible, both adults, both faces large and clear, two separate complete upper bodies, each person occupies about forty percent of the frame",
            f"left foreground character: {_identity_text(characters[0])}",
            f"right character behind checkout counter: {_identity_text(characters[1])}",
            "the left character faces and reaches toward the right character; the right character visibly faces the left character",
        ])
        cast_negative = "one person, solo, missing person, empty clerk position, extra person, crowd"
    action = _action_spec(panel)
    action_prompt = ""
    if action["action_code"]:
        action_prompt = (
            f"canonical action {action['action_code']}; only one {action['target']} visible; "
            f"opening state {action['start_state']}; final state {action['end_state']}"
        )
    correction = re.sub(r"\s+", " ", str(review_feedback or "")).strip()[:800]
    positive = ", ".join(filter(None, [
        "masterpiece, best quality, premium Japanese 2D animation, clean line art",
        "single vertical cinematic frame, medium eye-level two-shot, camera at chest height, no ceiling",
        cast_prompt,
        action_prompt,
        str(panel.get("first_frame") or panel.get("first_state") or ""),
        str(panel.get("final_state") or panel.get("visible_action") or ""),
        str(scene.get("model_prompt_en") or scene.get("positive_prompt") or scene.get("description") or ""),
        str(visual.get("style_prompt") or ""),
        f"camera composition {camera.get('composition') or 'two shot over counter'}",
        "plain blank geometric surfaces, unlettered props, dry indoor counter and floor",
        f"reviewer correction that must be satisfied: {correction}" if correction else "",
    ]))
    negative = ", ".join([
        cast_negative,
        "duplicate person, merged bodies, fused face, identity swap, same face",
        "cropped head, cropped face, hidden face, back-only clerk, clerk outside frame",
        "text, letters, numbers, logo, sign, watermark, label, badge, pseudo text",
        "split screen, collage, contact sheet, multiple panels",
        "ceiling-dominant framing, overhead view, wide empty upper space",
        "rain indoors, wet counter, wet floor, blurry, low quality, bad anatomy",
    ])
    return positive, negative


def _matches_rejected_candidate(
    first_sha256: str, last_sha256: str | None,
    rejection_audit: list[Mapping[str, Any]],
) -> bool:
    for rejected in rejection_audit:
        rejected_first = str(rejected.get("sha256") or "")
        rejected_last = str(rejected.get("last_sha256") or "") or None
        if first_sha256 != rejected_first:
            continue
        if rejected_last is None or rejected_last == last_sha256:
            return True
    return False


def generate_group_anchor(
    ep_id: str, job_id: str, *, store: RenderJobStore | None = None,
    api_func: Callable[[str, Mapping[str, Any] | None], dict[str, Any]] | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Generate one candidate and persist its status in the render job."""
    store = store or default_store()
    api_func = api_func or _api
    snapshot = _snapshot(ep_id, store)
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    episode = snapshot.get("episode") or {}
    pipeline = snapshot.get("pipeline") or {}
    if pipeline.get("contract_status") != "approved" or pipeline.get("assets_status") != "approved":
        raise RuntimeError("group anchor requires approved episode contract and assets")
    panels = episode.get("panels") or []
    index = int(job.get("panel_index") or 0)
    if index < 1 or index > len(panels):
        raise RuntimeError("job panel index is outside episode contract")
    panel = panels[index - 1]
    character_ids = [str(value) for value in panel.get("character_ids") or []]
    if len(character_ids) not in {1, 2}:
        raise RuntimeError("shot composition anchor currently requires one or two panel characters")
    cards_by_id = {
        str(card.get("character_id") or card.get("id") or ""): card
        for card in episode.get("character_bible") or []
    }
    characters = [cards_by_id.get(character_id) for character_id in character_ids]
    if any(not isinstance(card, Mapping) for card in characters):
        raise RuntimeError("group anchor character card is missing")
    scene_id = str(panel.get("scene_id") or "")
    scene = next((item for item in episode.get("scene_bible") or [] if str(item.get("scene_id") or "") == scene_id), None)
    if not isinstance(scene, Mapping):
        raise RuntimeError("group anchor scene card is missing")

    approved_assets = {
        (str(item.get("asset_type") or ""), str(item.get("source_id") or "")): item
        for item in snapshot.get("assets", {}).get("items", [])
        if item.get("approved") and item.get("status") == "succeeded"
    }
    project_dir = Path(str(snapshot["project_dir"])).resolve()
    char_paths: list[Path] = []
    asset_contracts: list[dict[str, Any]] = []
    for character_id in character_ids:
        asset = approved_assets.get(("character", character_id)) or {}
        refs = asset.get("reference_images") or []
        if not refs:
            raise RuntimeError(f"approved character reference missing: {character_id}")
        char_paths.append(_project_file(refs[0], project_dir, label=f"character reference {character_id}"))
        asset_contracts.append({
            "asset_id": str(asset.get("asset_id") or ""),
            "asset_type": "character", "source_id": character_id,
            "prompt_hash": str(asset.get("prompt_hash") or ""),
            "content_hash": str(asset.get("content_hash") or ""),
        })
    scene_asset = approved_assets.get(("scene", scene_id)) or {}
    scene_refs = scene_asset.get("reference_images") or []
    if not scene_refs:
        raise RuntimeError(f"approved scene reference missing: {scene_id}")
    scene_path = _project_file(scene_refs[0], project_dir, label=f"scene reference {scene_id}")
    asset_contracts.append({
        "asset_id": str(scene_asset.get("asset_id") or ""),
        "asset_type": "scene", "source_id": scene_id,
        "prompt_hash": str(scene_asset.get("prompt_hash") or ""),
        "content_hash": str(scene_asset.get("content_hash") or ""),
    })
    output_dir = project_dir / "group_anchors"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
    rejection_audit = [
        value for value in metadata.get("group_anchor_rejection_audit") or []
        if isinstance(value, Mapping)
    ]
    latest_feedback = str(
        (rejection_audit[-1] if rejection_audit else {}).get("rejection_reason") or ""
    ).strip()
    paired_state_required = requires_paired_state_anchor(panel, len(character_ids))
    candidate = {
        "schema": GROUP_ANCHOR_CONTRACT, "status": "running",
        "panel_id": str(panel.get("panel_id") or job.get("panel_name") or ""),
        "character_ids": character_ids, "scene_id": scene_id,
        "panel_contract_sha256": panel_anchor_contract_sha256(panel),
        "paired_state_required": paired_state_required,
        "asset_contracts": asset_contracts,
        "review_feedback": latest_feedback or None,
        "review_feedback_sha256": (
            hashlib.sha256(latest_feedback.encode("utf-8")).hexdigest()
            if latest_feedback else None
        ),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata["group_anchor_candidate"] = candidate
    store.update_job(job_id, metadata=metadata)

    positive, negative = compile_group_anchor_prompts(
        panel, characters, scene, episode.get("visual_bible") or {},
        review_feedback=latest_feedback,
    )
    manifest_path: Path | None = None
    try:
        # api_func/timeout remain accepted for facade compatibility. This
        # fail-closed path deliberately uses no diffusion before human review.
        _ = api_func, timeout
        prompt_id = f"deterministic-{uuid.uuid4().hex[:12]}"
        destination = output_dir / f"{_safe(str(job.get('panel_name') or 'shot'))}_{prompt_id[-8:]}.png"
        _compose_approved_group_anchor(scene_path, char_paths, destination, panel, state="first")
        last_destination: Path | None = None
        if paired_state_required:
            last_destination = destination.with_name(f"{destination.stem}_last.png")
            _compose_approved_group_anchor(
                scene_path, char_paths, last_destination, panel, state="last",
            )
        manifest_path = destination.with_suffix(".group_anchor.json")
        first_sha256 = _sha256(destination)
        last_sha256 = _sha256(last_destination) if last_destination else None
        candidate.update({
            "prompt_id": prompt_id,
            "path": str(destination), "sha256": first_sha256,
            "last_path": str(last_destination) if last_destination else None,
            "last_sha256": last_sha256,
            "scene_reference": {"path": str(scene_path), "sha256": _sha256(scene_path)},
            "character_references": [
                {"source_id": source_id, "path": str(path), "sha256": _sha256(path)}
                for source_id, path in zip(character_ids, char_paths)
            ],
            "positive_prompt": positive, "negative_prompt": negative,
            "renderer_contract": GROUP_ANCHOR_CONTRACT,
            "manifest_path": str(manifest_path),
        })
        if _matches_rejected_candidate(first_sha256, last_sha256, rejection_audit):
            raise RuntimeError(
                "duplicate_rejected_candidate: regenerated anchor bytes match a previously rejected hash"
            )
        candidate.update({
            "status": "succeeded",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        _write_json_atomic(manifest_path, candidate)
        metadata = json.loads(json.dumps((store.get_job(job_id) or {}).get("metadata") or {}, ensure_ascii=False))
        metadata["group_anchor_candidate"] = candidate
        metadata.pop("approved_group_anchor", None)
        store.update_job(job_id, metadata=metadata)
        return candidate
    except Exception as exc:
        metadata = json.loads(json.dumps((store.get_job(job_id) or {}).get("metadata") or {}, ensure_ascii=False))
        failed = dict(metadata.get("group_anchor_candidate") or candidate)
        failed.update({
            **candidate,
            "status": "failed", "error": str(exc),
            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        if manifest_path is not None:
            failed["manifest_path"] = str(manifest_path)
            _write_json_atomic(manifest_path, failed)
        metadata["group_anchor_candidate"] = failed
        store.update_job(job_id, metadata=metadata)
        raise


def approve_group_anchor(
    ep_id: str, job_id: str, *, expected_sha256: str,
    reason: str, approved_by: str = "reviewer", store: RenderJobStore | None = None,
) -> dict[str, Any]:
    """Bind the reviewed candidate hash as H3 Picture 1 authority."""
    if not reason.strip() or not approved_by.strip():
        raise ValueError("group anchor approval reason and approved_by are required")
    store = store or default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
    candidate = metadata.get("group_anchor_candidate") or {}
    path = Path(str(candidate.get("path") or "")).resolve()
    snapshot = _snapshot(ep_id, store)
    project = Path(str(snapshot["project_dir"])).resolve()
    if candidate.get("status") != "succeeded" or not path.is_file() or not path.is_relative_to(project):
        raise RuntimeError("group anchor candidate is not a valid project image")
    actual_sha = _sha256(path)
    if actual_sha != str(expected_sha256 or "") or actual_sha != str(candidate.get("sha256") or ""):
        raise RuntimeError("group anchor hash changed before approval")
    episode = snapshot.get("episode") or {}
    index = int(job.get("panel_index") or 0)
    panels = episode.get("panels") or []
    if index < 1 or index > len(panels):
        raise RuntimeError("group anchor panel no longer exists")
    panel = panels[index - 1]
    current_character_ids = [str(value) for value in panel.get("character_ids") or []]
    current_scene_id = str(panel.get("scene_id") or "")
    if (
        current_character_ids != list(candidate.get("character_ids") or [])
        or current_scene_id != str(candidate.get("scene_id") or "")
        or str(panel.get("panel_id") or job.get("panel_name") or "")
        != str(candidate.get("panel_id") or "")
    ):
        raise RuntimeError("group anchor candidate is stale for the current panel contract")
    if str(candidate.get("panel_contract_sha256") or "") != panel_anchor_contract_sha256(panel):
        raise RuntimeError("group anchor candidate action/camera contract is stale")
    paired_state_required = requires_paired_state_anchor(panel, len(current_character_ids))
    if bool(candidate.get("paired_state_required")) != paired_state_required:
        raise RuntimeError("group anchor paired-state contract is stale for the current panel")
    last_path_value = str(candidate.get("last_path") or "").strip()
    if paired_state_required and not last_path_value:
        raise RuntimeError("paired state-changing anchor is missing its final-state image")
    if last_path_value:
        last_path = _project_file(last_path_value, project, label="group anchor final state")
        if _sha256(last_path) != str(candidate.get("last_sha256") or ""):
            raise RuntimeError("group anchor final-state hash changed before approval")
    current_assets = {
        str(item.get("asset_id") or ""): item for item in store.list_assets(ep_id)
    }
    for contract in candidate.get("asset_contracts") or []:
        current = current_assets.get(str(contract.get("asset_id") or "")) or {}
        if (
            current.get("status") != "succeeded"
            or not current.get("approved")
            or str(current.get("prompt_hash") or "") != str(contract.get("prompt_hash") or "")
            or str(current.get("content_hash") or "") != str(contract.get("content_hash") or "")
        ):
            raise RuntimeError("group anchor source asset changed before approval")
    for reference in [candidate.get("scene_reference") or {}, *(candidate.get("character_references") or [])]:
        source_path = _project_file(reference.get("path") or "", project, label="group anchor source")
        if _sha256(source_path) != str(reference.get("sha256") or ""):
            raise RuntimeError("group anchor source file changed before approval")
    rejection_audit = [
        value for value in metadata.get("group_anchor_rejection_audit") or []
        if isinstance(value, Mapping)
    ]
    if _matches_rejected_candidate(
        actual_sha, str(candidate.get("last_sha256") or "") or None, rejection_audit,
    ):
        raise RuntimeError("group anchor hash was previously rejected and cannot be approved")
    approved = {
        **candidate, "status": "approved", "approved_by": approved_by,
        "approval_reason": reason.strip(),
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata["approved_group_anchor"] = approved
    return store.update_job(job_id, metadata=metadata)


def reject_group_anchor(
    ep_id: str, job_id: str, *, reason: str,
    rejected_by: str = "reviewer", store: RenderJobStore | None = None,
) -> dict[str, Any]:
    """Reject the visible candidate without deleting its immutable audit file."""
    if not reason.strip() or not rejected_by.strip():
        raise ValueError("group anchor rejection reason and rejected_by are required")
    store = store or default_store()
    job = store.get_job(job_id, ep_id=ep_id)
    if not job:
        raise KeyError(f"unknown job {ep_id}/{job_id}")
    metadata = json.loads(json.dumps(job.get("metadata") or {}, ensure_ascii=False))
    candidate = dict(metadata.get("group_anchor_candidate") or {})
    if candidate.get("status") != "succeeded":
        raise RuntimeError("only a succeeded group anchor candidate can be rejected")
    path = Path(str(candidate.get("path") or "")).resolve()
    project = Path(str(_snapshot(ep_id, store)["project_dir"])).resolve()
    if not path.is_file() or not path.is_relative_to(project) or _sha256(path) != candidate.get("sha256"):
        raise RuntimeError("group anchor candidate changed before rejection")
    last_path_value = str(candidate.get("last_path") or "").strip()
    if last_path_value:
        last_path = _project_file(last_path_value, project, label="group anchor final state")
        if _sha256(last_path) != str(candidate.get("last_sha256") or ""):
            raise RuntimeError("group anchor final-state candidate changed before rejection")
    audit = list(metadata.get("group_anchor_rejection_audit") or [])
    audit.append({
        **candidate, "status": "rejected", "rejected_by": rejected_by,
        "rejection_reason": reason.strip(),
        "rejected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    candidate.update({
        "status": "rejected", "rejected_by": rejected_by,
        "rejection_reason": reason.strip(),
        "rejected_at": audit[-1]["rejected_at"],
    })
    manifest_path_value = str(candidate.get("manifest_path") or "").strip()
    if manifest_path_value:
        manifest_path = Path(manifest_path_value).resolve()
        if not manifest_path.is_relative_to(project):
            raise RuntimeError("group anchor manifest must remain inside the episode project")
        _write_json_atomic(manifest_path, candidate)
    metadata["group_anchor_candidate"] = candidate
    metadata["group_anchor_rejection_audit"] = audit[-20:]
    metadata.pop("approved_group_anchor", None)
    return store.update_job(job_id, metadata=metadata)


__all__ = [
    "GROUP_ANCHOR_CONTRACT", "build_group_anchor_workflow",
    "compile_group_anchor_prompts", "generate_group_anchor", "approve_group_anchor",
    "reject_group_anchor", "requires_paired_state_anchor", "requires_approved_group_anchor",
    "panel_anchor_contract_sha256",
]
