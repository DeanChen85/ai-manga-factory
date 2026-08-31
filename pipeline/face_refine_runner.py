"""
face_refine_runner.py — Optional face/skin refine post-pass via T8 nodes (no code copy).

Strategy: this module does NOT bundle T8 code. It provides a thin wrapper that,
when T8 nodes ARE available in the user's ComfyUI, builds a workflow JSON
that calls T8's FaceRefine / SkinFinish nodes via the public /prompt API.

If T8 is not installed, the wrapper emits a SKIPPED log + the input mp4
is returned unchanged. This is OFF by default and only triggered
explicitly by the human review process or by hash-bound QA detecting
identity drift.

This is a wrapper, not a copy. We only reference T8 node names that are
documented in T8's public API.

License: Apache-2.0 (this file). T8 nodes: GPL-3.0 (user-installed).
"""
from __future__ import annotations

import copy
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


# T8 public node names we may invoke. T8 may not be installed; that's fine.
T8_FACE_REFINE_NODES = {
    "face_refine": "MiniMaxH3FaceRefineSamplerT8Advanced",
    "face_refine_plan": "MiniMaxH3FaceRefinePlanT8Advanced",
    "skin_finish": "MiniMaxH3SkinFinishT8",
    "skin_finish_advanced": "MiniMaxH3SkinFinishAdvancedT8",
    "face_refine_conditioning": "MiniMaxH3FaceRefineConditioningT8Advanced",
}


@dataclass
class FaceRefineResult:
    ok: bool
    output_path: Path | None
    skipped: bool
    skip_reason: str | None
    node_used: str | None


def t8_node_registered(node_name: str, comfy_url: str = "http://127.0.0.1:8188") -> bool:
    """Return True iff the given T8 node is registered in the running ComfyUI.

    Falls back to False on any error (network, missing endpoint, etc).
    """
    try:
        with urllib.request.urlopen(f"{comfy_url.rstrip('/')}/object_info", timeout=5) as resp:
            data = json.loads(resp.read())
        return node_name in data
    except Exception:
        return False


def build_face_refine_workflow(
    input_mp4: Path,
    first_frame: Path,
    face_ref: Path,
    output_prefix: str,
    profile: str = "safe",  # "safe" | "aggressive"
) -> dict:
    """Build a ComfyUI workflow JSON calling T8's FaceRefine + SkinFinish.

    The structure mirrors T8's documentation: load mp4 as frames, pass
    through FaceRefinePlan + FaceRefineSampler, optionally SkinFinish,
    save result. If T8 isn't installed, this workflow will fail to load
    in ComfyUI — callers MUST check `t8_node_registered` first.
    """
    if profile not in ("safe", "aggressive"):
        raise ValueError(f"profile must be safe|aggressive, got {profile}")
    plan_node = T8_FACE_REFINE_NODES["face_refine_plan"]
    sampler_node = T8_FACE_REFINE_NODES["face_refine"]
    skin_node = T8_FACE_REFINE_NODES["skin_finish"] if profile == "aggressive" else None

    wf = {
        # 1. Load input mp4 (T8 expects first frame as image)
        "1": {"class_type": "VHS_LoadVideo",
              "inputs": {"video": str(input_mp4), "force_rate": 24, "force_size": "Disabled"}},
        "2": {"class_type": "VHS_GetFrameCount",
              "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "LoadImage",
              "inputs": {"image": str(first_frame)}},
        "4": {"class_type": "LoadImage",
              "inputs": {"image": str(face_ref)}},

        # 2. Face refine plan (T8-specific)
        "10": {"class_type": plan_node,
               "inputs": {
                   "frames": ["1", 0],
                   "reference_face": ["4", 0],
                   "profile": profile,
               }},

        # 3. Face refine sampler
        "11": {"class_type": sampler_node,
               "inputs": {
                   "video_frames": ["1", 0],
                   "plan": ["10", 0],
                   "denoise": 0.25 if profile == "safe" else 0.45,
                   "seed": 0,
               }},

        # 4. Save video
        "20": {"class_type": "VHS_VideoCombine",
               "inputs": {"frames": ["11", 0],
                          "frame_rate": 24,
                          "filename_prefix": output_prefix,
                          "format": "video/h264-mp4"}},
    }
    if skin_node:
        # 5. Optional skin finish pass (only for aggressive)
        wf["12"] = {"class_type": skin_node,
                    "inputs": {"video": ["11", 0], "intensity": 0.3}}
        wf["20"]["inputs"]["frames"] = ["12", 0]
    return wf


def run_face_refine(
    input_mp4: Path,
    face_reference: Path,
    output_dir: Path,
    comfy_url: str = "http://127.0.0.1:8188",
    profile: str = "safe",
    timeout: int = 1200,
) -> FaceRefineResult:
    """Submit face refine to ComfyUI, wait, return result.

    Returns:
      - ok=True, output_path set: refine succeeded
      - ok=False, skipped=True, skip_reason set: T8 not installed (or another prerequisite missing)
      - ok=False, skipped=False: refine failed mid-execution
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check T8 is available
    if not t8_node_registered(T8_FACE_REFINE_NODES["face_refine"], comfy_url):
        return FaceRefineResult(
            ok=False, output_path=None, skipped=True,
            skip_reason=f"T8 node '{T8_FACE_REFINE_NODES['face_refine']}' not registered in ComfyUI",
            node_used=None,
        )

    # First-frame extraction (for T8's reference input)
    first_frame = output_dir / "first_frame.png"
    if not _extract_first_frame(input_mp4, first_frame):
        return FaceRefineResult(
            ok=False, output_path=None, skipped=True,
            skip_reason="ffmpeg first-frame extraction failed",
            node_used=None,
        )

    wf = build_face_refine_workflow(
        input_mp4, first_frame, face_reference,
        output_prefix=f"face_refine_{profile}_{int(time.time())}",
        profile=profile,
    )

    # Submit
    req = urllib.request.Request(
        f"{comfy_url.rstrip('/')}/prompt",
        data=json.dumps({"prompt": wf}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as exc:
        return FaceRefineResult(
            ok=False, output_path=None, skipped=False,
            skip_reason=f"submit failed: {exc}",
            node_used=T8_FACE_REFINE_NODES["face_refine"],
        )
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        return FaceRefineResult(
            ok=False, output_path=None, skipped=False,
            skip_reason=f"no prompt_id: {resp}",
            node_used=T8_FACE_REFINE_NODES["face_refine"],
        )

    # Wait
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(8)
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
                        return FaceRefineResult(
                            ok=True, output_path=Path(path_str), skipped=False,
                            skip_reason=None,
                            node_used=T8_FACE_REFINE_NODES["face_refine"],
                        )
        if entry and entry.get("status", {}).get("status_str") == "error":
            return FaceRefineResult(
                ok=False, output_path=None, skipped=False,
                skip_reason="ComfyUI reported error",
                node_used=T8_FACE_REFINE_NODES["face_refine"],
            )
    return FaceRefineResult(
        ok=False, output_path=None, skipped=False,
        skip_reason=f"timeout after {timeout}s",
        node_used=T8_FACE_REFINE_NODES["face_refine"],
    )


def _extract_first_frame(mp4: Path, out_png: Path, ffmpeg_bin: str = "ffmpeg") -> bool:
    import subprocess
    out_png.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(mp4), "-frames:v", "1", str(out_png)],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode == 0 and out_png.exists() and out_png.stat().st_size > 0


# === CLI ===

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python -m pipeline.face_refine_runner <input_mp4> <face_reference.png> <output_dir> [safe|aggressive]")
        sys.exit(0)
    inp = Path(sys.argv[1])
    face = Path(sys.argv[2])
    out = Path(sys.argv[3])
    prof = sys.argv[4] if len(sys.argv) > 4 else "safe"
    result = run_face_refine(inp, face, out, profile=prof)
    print(f"ok={result.ok} skipped={result.skipped} path={result.output_path} reason={result.skip_reason}")