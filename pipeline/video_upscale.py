"""
video_upscale.py — Optional video super-resolution post-pass via FlashVSR (no code copy).

FlashVSR v1.1 (JunhaoZhuang/FlashVSR-v1.1) is a 2x/4x video super-resolution model.
T8 wraps it as `MiniMaxH3FlashVSRRestoreT8Advanced`.

This module does NOT bundle T8 or FlashVSR. It provides a thin wrapper
that, when both are installed, builds a ComfyUI workflow calling T8's
FlashVSR node. If either is missing, the wrapper emits a SKIPPED log
and the input mp4 is returned unchanged.

OFF by default. Only triggered by the human review process or by
hash-bound QA detecting low resolution.

License: Apache-2.0 (this file). FlashVSR: see upstream; T8: GPL-3.0.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


T8_FLASHVSR_NODES = {
    "execution_plan": "MiniMaxH3FlashVSRExecutionPlanT8Advanced",
    "model": "MiniMaxH3FlashVSRModelT8Advanced",
    "restore": "MiniMaxH3FlashVSRRestoreT8Advanced",
}

# FlashVSR profile presets (mirrors T8's "Quality Locked" / "Memory Safe")
FLASHVSR_PROFILES = {
    "quality_locked": {"scale": 2.0, "lcsa": "3.0", "num_steps": 11},
    "balanced_dynamic": {"scale": 2.0, "lcsa": "3.0", "num_steps": 11, "low_motion_budget": True},
    "memory_safe": {"scale": 2.0, "lcsa": "3.0", "num_steps": 11, "tiled": True},
}


@dataclass
class UpscaleResult:
    ok: bool
    output_path: Path | None
    skipped: bool
    skip_reason: str | None
    scale: float
    profile: str


def build_flashvsr_workflow(
    input_mp4: Path,
    first_frame: Path,
    output_prefix: str,
    profile: str = "quality_locked",
    scale: float = 2.0,
) -> dict:
    """Build a ComfyUI workflow that calls T8's FlashVSR wrapper.

    FlashVSR is implemented as: load frames -> execution plan -> restore ->
    combine. T8 also bundles a 24-channel latent upscaler that we may
    route through MiniMaxH3LearnedLatentUpscaleT8Advanced.
    """
    if profile not in FLASHVSR_PROFILES:
        raise ValueError(f"profile must be one of {list(FLASHVSR_PROFILES)}, got {profile}")
    p = FLASHVSR_PROFILES[profile]
    if scale not in (2.0, 4.0):
        raise ValueError(f"scale must be 2.0 or 4.0, got {scale}")

    plan = T8_FLASHVSR_NODES["execution_plan"]
    model = T8_FLASHVSR_NODES["model"]
    restore = T8_FLASHVSR_NODES["restore"]

    wf = {
        "1": {"class_type": "VHS_LoadVideo",
              "inputs": {"video": str(input_mp4), "force_rate": 24}},
        "2": {"class_type": "LoadImage",
              "inputs": {"image": str(first_frame)}},
        "10": {"class_type": model,
               "inputs": {"first_frame": ["2", 0]}},
        "11": {"class_type": plan,
               "inputs": {"frames": ["1", 0], "scale": scale,
                          "lcsa": p["lcsa"], "num_steps": p["num_steps"],
                          "tiled": p.get("tiled", False),
                          "low_motion_budget": p.get("low_motion_budget", False)}},
        "12": {"class_type": restore,
               "inputs": {"video": ["1", 0], "plan": ["11", 0],
                          "preserve_audio": True}},
        "20": {"class_type": "VHS_VideoCombine",
               "inputs": {"frames": ["12", 0], "frame_rate": 24,
                          "filename_prefix": output_prefix,
                          "format": "video/h264-mp4"}},
    }
    return wf


def t8_node_registered(node_name: str, comfy_url: str = "http://127.0.0.1:8188") -> bool:
    try:
        with urllib.request.urlopen(f"{comfy_url.rstrip('/')}/object_info", timeout=5) as resp:
            data = json.loads(resp.read())
        return node_name in data
    except Exception:
        return False


def run_upscale(
    input_mp4: Path,
    output_dir: Path,
    profile: str = "quality_locked",
    scale: float = 2.0,
    comfy_url: str = "http://127.0.0.1:8188",
    timeout: int = 1800,
) -> UpscaleResult:
    """Submit FlashVSR upscale to ComfyUI, wait, return result.

    Skips cleanly if T8 / FlashVSR not installed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not t8_node_registered(T8_FLASHVSR_NODES["restore"], comfy_url):
        return UpscaleResult(
            ok=False, output_path=None, skipped=True,
            skip_reason=f"T8 node '{T8_FLASHVSR_NODES['restore']}' not registered",
            scale=scale, profile=profile,
        )

    first_frame = output_dir / "first_frame.png"
    if not _extract_first_frame(input_mp4, first_frame):
        return UpscaleResult(
            ok=False, output_path=None, skipped=True,
            skip_reason="ffmpeg first-frame extraction failed",
            scale=scale, profile=profile,
        )

    wf = build_flashvsr_workflow(
        input_mp4, first_frame,
        output_prefix=f"flashvsr_{profile}_{int(time.time())}",
        profile=profile, scale=scale,
    )

    req = urllib.request.Request(
        f"{comfy_url.rstrip('/')}/prompt",
        data=json.dumps({"prompt": wf}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as exc:
        return UpscaleResult(
            ok=False, output_path=None, skipped=False,
            skip_reason=f"submit failed: {exc}",
            scale=scale, profile=profile,
        )
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        return UpscaleResult(
            ok=False, output_path=None, skipped=False,
            skip_reason=f"no prompt_id: {resp}",
            scale=scale, profile=profile,
        )

    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(10)
        try:
            hist = json.loads(
                urllib.request.urlopen(f"{comfy_url}/history/{prompt_id}", timeout=10).read()
            )
        except Exception:
            continue
        entry = hist.get(prompt_id)
        if entry and entry.get("outputs"):
            for node_out in entry["outputs"].values():
                videos = node_out.get("gifs") or node_out.get("videos") or []
                if videos:
                    path_str = videos[0].get("fullpath") or videos[0].get("filename")
                    if path_str and Path(path_str).exists():
                        return UpscaleResult(
                            ok=True, output_path=Path(path_str), skipped=False,
                            skip_reason=None, scale=scale, profile=profile,
                        )
        if entry and entry.get("status", {}).get("status_str") == "error":
            return UpscaleResult(
                ok=False, output_path=None, skipped=False,
                skip_reason="ComfyUI reported error",
                scale=scale, profile=profile,
            )
    return UpscaleResult(
        ok=False, output_path=None, skipped=False,
        skip_reason=f"timeout after {timeout}s",
        scale=scale, profile=profile,
    )


def _extract_first_frame(mp4: Path, out_png: Path, ffmpeg_bin: str = "ffmpeg") -> bool:
    import subprocess
    out_png.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(mp4), "-frames:v", "1", str(out_png)],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode == 0 and out_png.exists() and out_png.stat().st_size > 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python -m pipeline.video_upscale <input_mp4> <output_dir> [quality_locked|balanced_dynamic|memory_safe] [2|4]")
        sys.exit(0)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2])
    prof = sys.argv[3] if len(sys.argv) > 3 else "quality_locked"
    sc = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    r = run_upscale(inp, out, profile=prof, scale=sc)
    print(f"ok={r.ok} skipped={r.skipped} scale={r.scale} profile={r.profile} out={r.output_path} reason={r.skip_reason}")