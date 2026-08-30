"""Read-only clone/deployment preflight for the live ComfyUI H3 runtime."""
from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Mapping

from runtime_config import comfyui_root, comfyui_server, ffmpeg_executable, ffprobe_executable
from render_video_h3 import (
    H3_AUDIO_VAE,
    H3_CLIP,
    H3_LORA_CANDIDATES,
    H3_LORA_RECOMMENDED,
    H3_UNET,
    H3_VIDEO_VAE,
)


REQUIRED_NODES = (
    "UNETLoader", "VAELoader", "CLIPLoader", "LoadImage", "SaveVideo",
    "MiniMaxH3ReferenceToVideo", "MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler",
    "PathchSageAttentionKJ",
)
RECOMMENDED_NODES = ("MiniMaxH3AddGuide",)
REQUIRED_MODELS = (H3_UNET, H3_CLIP, H3_VIDEO_VAE, H3_AUDIO_VAE)


def _http_json(url: str, timeout: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object response from {url}")
    return value


def _installed_model_names(root: Path) -> dict[str, str]:
    model_root = root / "models"
    result: dict[str, str] = {}
    if not model_root.is_dir():
        return result
    for path in model_root.rglob("*"):
        if path.is_file():
            result.setdefault(path.name.casefold(), path.relative_to(root).as_posix())
    return result


def _executable_status(resolver) -> dict[str, Any]:
    try:
        value = str(resolver())
    except (OSError, RuntimeError) as exc:
        return {"available": False, "error": str(exc)}
    resolved = shutil.which(value) or (value if Path(value).is_file() else None)
    return {"available": bool(resolved), "path": str(resolved or value)}


def run_preflight(
    *,
    root: str | Path | None = None,
    server: str | None = None,
    object_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    comfy_root_path = Path(root or comfyui_root()).resolve()
    endpoint = str(server or comfyui_server()).rstrip("/")
    failures: list[str] = []
    warnings: list[str] = []
    try:
        live_nodes = dict(object_info) if object_info is not None else _http_json(f"{endpoint}/object_info")
    except Exception as exc:
        live_nodes = {}
        failures.append(f"ComfyUI object_info unavailable: {exc}")
    missing_nodes = sorted(set(REQUIRED_NODES) - set(live_nodes))
    missing_recommended = sorted(set(RECOMMENDED_NODES) - set(live_nodes))
    if missing_nodes:
        failures.append("missing required nodes: " + ", ".join(missing_nodes))
    if missing_recommended:
        warnings.append("missing recommended nodes: " + ", ".join(missing_recommended))

    installed = _installed_model_names(comfy_root_path)
    missing_models = [name for name in REQUIRED_MODELS if name.casefold() not in installed]
    installed_lora = next(
        (installed.get(candidate.casefold()) for candidate in H3_LORA_CANDIDATES if installed.get(candidate.casefold())),
        None,
    )
    if missing_models:
        failures.append("missing required models: " + ", ".join(missing_models))
    if not installed_lora:
        failures.append("missing compatible H3 Turbo LoRA: " + ", ".join(H3_LORA_CANDIDATES))
    elif Path(installed_lora).name != H3_LORA_RECOMMENDED:
        warnings.append(f"legacy Turbo LoRA selected: {installed_lora}")

    ffmpeg = _executable_status(ffmpeg_executable)
    ffprobe = _executable_status(ffprobe_executable)
    if not ffmpeg["available"]:
        failures.append("ffmpeg unavailable")
    if not ffprobe["available"]:
        failures.append("ffprobe unavailable")
    return {
        "schema": "ai-manga-comfy-preflight/v1",
        "passed": not failures,
        "comfyui_root": str(comfy_root_path),
        "comfyui_server": endpoint,
        "nodes": {"missing_required": missing_nodes, "missing_recommended": missing_recommended},
        "models": {
            "missing_required": missing_models,
            "turbo_lora": installed_lora,
            "preferred_turbo_lora": H3_LORA_RECOMMENDED,
        },
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "failures": failures,
        "warnings": warnings,
    }


def main() -> int:
    result = run_preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

