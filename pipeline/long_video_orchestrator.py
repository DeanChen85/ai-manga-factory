"""
long_video_orchestrator.py — 单镜头多段 H3 长视频编排（原创实现）。

为什么需要这个：
H3 单段生成有性能红线（10s 段解码 40+ 分钟不可用）。本模块把"长镜头"切成
n 个 2-15s 短段，段间用 ffmpeg 抽末帧作为下一段 first_frame，对接 ComfyUI
执行 + 自动 retry + 失败恢复 + 收尾 concat 成完整镜头。

与 T8 / RH 关系：
- 不复制 T8 节点代码（GPL-3.0 隔离）；独立实现
- 与 `task_store` 集成：每个段是一个 sub_job，hash 绑定
- 与 `h3_profiles` 集成：每段用 profile 控制时长/像素/步数
- 与 `render_video_h3` 集成：每段复用 H3 graph 提交逻辑

License: Apache-2.0（与本仓库一致）
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# === 配置常量 ===

# H3 推荐下界：低于 2s 的段在 shot_group_anchor / 连续性管理上易出问题
MIN_SEGMENT_SECONDS = 2
# H3 推荐上界：超过 15s 解码成本爆炸
MAX_SEGMENT_SECONDS = 15
# 17n+5 帧格在 24fps 下，n 范围对应 [2, 15]s
FRAME_LATTICE = [17 * n + 5 for n in range(1, 7)]  # n=1..6 → 22, 39, 56, 73, 90, 107
# 每个 lattice 帧数对应的秒数（24 fps）
LATTICE_SECONDS = {f: round(f / 24, 4) for f in FRAME_LATTICE}

# 默认段时长：5s（与 H3 proof profile 一致）
DEFAULT_SEGMENT_LENGTH = 124  # 5.167s
# 末帧时间偏移（避免末帧黑色缓冲）
LAST_FRAME_OFFSET = 0.05

# === 数据结构 ===


@dataclass
class SegmentSpec:
    """一段视频的描述。"""
    index: int
    target_seconds: float
    first_frame_path: str | None  # 上一段末帧；第一段为 None
    prompt: str
    seed: int
    extra_inputs: dict = field(default_factory=dict)


@dataclass
class SegmentResult:
    """一段执行的产物。"""
    index: int
    output_path: Path
    last_frame_path: Path
    duration_seconds: float
    retry_count: int
    elapsed_seconds: float


# === 帧格吸附：把秒数 snap 到 17n+5 / 24fps 的最近合法值 ===


def snap_seconds_to_lattice(target: float) -> int:
    """把任意秒数吸附到 17n+5 帧格（24fps 下的合法 H3 时长）。

    Returns: 帧数（不是秒数）。H3 节点需要帧数。
    """
    if target < MIN_SEGMENT_SECONDS:
        target = MIN_SEGMENT_SECONDS
    if target > MAX_SEGMENT_SECONDS:
        target = MAX_SEGMENT_SECONDS
    target_frames = round(target * 24)
    # 找最近的 lattice
    return min(FRAME_LATTICE, key=lambda f: abs(f - target_frames))


def segment_count_for_total(total_seconds: float, segment_seconds: float = 5.0) -> int:
    """根据总时长计算段数（每段不超过 15s）。"""
    if total_seconds <= 0:
        return 0
    segment = max(MIN_SEGMENT_SECONDS, min(segment_seconds, MAX_SEGMENT_SECONDS))
    return max(1, -(-int(total_seconds * 1000) // int(segment * 1000)))  # ceil


# === 末帧抽取与拼接 ===


def extract_last_frame(
    mp4_path: Path, out_png: Path, ffmpeg_bin: str = "ffmpeg"
) -> bool:
    """从生成的 mp4 抽最后一帧到 png。

    Returns: True if extraction succeeded.
    """
    if not mp4_path.exists():
        return False
    out_png.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [ffmpeg_bin, "-y", "-sseof", f"-{LAST_FRAME_OFFSET}", "-i", str(mp4_path),
         "-frames:v", "1", str(out_png)],
        capture_output=True, text=True, timeout=120,
    )
    return r.returncode == 0 and out_png.exists() and out_png.stat().st_size > 0


def concat_segments(
    segment_paths: list[Path], out_path: Path, ffmpeg_bin: str = "ffmpeg"
) -> bool:
    """把多段 mp4 拼成完整镜头。

    先尝试 `-c copy` 流复制；失败时回退到重编码。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.with_suffix(".concat.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")

    # 第一次尝试：流复制
    r = subprocess.run(
        [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(out_path)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode == 0:
        list_file.unlink(missing_ok=True)
        return True

    # 回退：重编码
    r2 = subprocess.run(
        [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out_path)],
        capture_output=True, text=True, timeout=600,
    )
    list_file.unlink(missing_ok=True)
    return r2.returncode == 0


# === 段规划：把一个长镜头合同切分成 SegmentSpec 列表 ===


def plan_segments(
    total_seconds: float,
    base_prompt: str,
    base_seed: int,
    segment_seconds: float = 5.0,
) -> list[SegmentSpec]:
    """根据总时长规划段。

    段数 = ceil(total / segment_seconds)
    每段 frame_count = snap(segment_seconds)
    第 1 段 first_frame = None；后续段 = 上一段末帧
    """
    if total_seconds <= 0:
        return []

    n_segments = segment_count_for_total(total_seconds, segment_seconds)
    frame_count = snap_seconds_to_lattice(segment_seconds)
    actual_segment_seconds = frame_count / 24

    plan = []
    for i in range(n_segments):
        plan.append(SegmentSpec(
            index=i,
            target_seconds=actual_segment_seconds,
            first_frame_path=None,  # 由 orchestrator 在运行时填
            prompt=f"[Shot {i+1}/{n_segments}] {base_prompt}",
            seed=base_seed + i * 31,
        ))
    return plan


# === 编排：实际提交到 ComfyUI + 监控 + 续接 ===


def submit_segment_to_comfyui(
    segment: SegmentSpec,
    comfy_url: str,
    workflow_payload: dict,
    output_dir: Path,
    timeout: int = 900,
) -> Path | None:
    """提交一段到 ComfyUI /prompt 端点，等待完成，返回 mp4 路径。

    workflow_payload: 已含本段的 prompt/seed/first_frame
    """
    import urllib.request
    out_dir = output_dir / f"seg{segment.index + 1:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        f"{comfy_url.rstrip('/')}/prompt",
        data=json.dumps({"prompt": workflow_payload}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as exc:
        print(f"  seg{segment.index+1}: submit failed: {exc}")
        return None
    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        print(f"  seg{segment.index+1}: no prompt_id in response: {resp}")
        return None

    # 等完成
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
        if entry:
            if entry.get("outputs"):
                # 找 SaveVideo 节点输出
                for node_out in entry["outputs"].values():
                    videos = node_out.get("videos") or node_out.get("gifs") or []
                    if videos:
                        # 真实路径在 comfyui/output；客户端要 url 下载
                        # 这里只返回期望路径，由 caller 实际去拉
                        return Path(videos[0]["fullpath"]) if "fullpath" in videos[0] else None
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                return None
    return None


# === 顶层编排入口 ===


def orchestrate_long_shot(
    total_seconds: float,
    base_prompt: str,
    base_seed: int,
    comfy_url: str,
    workflow_template: dict,
    output_dir: Path,
    ffmpeg_bin: str = "ffmpeg",
    segment_seconds: float = 5.0,
    max_retries: int = 2,
) -> dict:
    """编排一个长镜头的完整生成。

    Returns: {"ok": bool, "mp4": Path|None, "segments": [SegmentResult]}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_segments(total_seconds, base_prompt, base_seed, segment_seconds)
    if not plan:
        return {"ok": False, "mp4": None, "segments": []}

    last_frame_path: Path | None = None
    results: list[SegmentResult] = []
    segment_mp4s: list[Path] = []

    for i, seg in enumerate(plan):
        if last_frame_path is not None:
            seg.first_frame_path = str(last_frame_path)
        # 注入到 workflow
        wf = inject_into_workflow(workflow_template, seg, output_dir)
        t0 = time.time()
        ok = False
        for attempt in range(max_retries + 1):
            mp4 = submit_segment_to_comfyui(seg, comfy_url, wf, output_dir)
            if mp4 and mp4.exists():
                seg_mp4 = mp4
                # 抽末帧
                lf = output_dir / f"seg{i+1:03d}_last.png"
                if extract_last_frame(seg_mp4, lf, ffmpeg_bin):
                    last_frame_path = lf
                else:
                    last_frame_path = None  # 下一段不再做帧衔接
                results.append(SegmentResult(
                    index=seg.index,
                    output_path=seg_mp4,
                    last_frame_path=lf if lf.exists() else Path(""),
                    duration_seconds=seg.target_seconds,
                    retry_count=attempt,
                    elapsed_seconds=time.time() - t0,
                ))
                segment_mp4s.append(seg_mp4)
                ok = True
                break
        if not ok:
            return {"ok": False, "mp4": None, "segments": results, "failed_at": i}

    # concat
    final_mp4 = output_dir / "final.mp4"
    if not concat_segments(segment_mp4s, final_mp4, ffmpeg_bin):
        return {"ok": False, "mp4": None, "segments": results}
    return {"ok": True, "mp4": final_mp4, "segments": results}


def inject_into_workflow(template: dict, seg: SegmentSpec, output_dir: Path) -> dict:
    """把段参数注入到 workflow 模板。

    这是一个简化版：替换 prompt 文本和 seed。如果用户的工作流
    有专门的 first_frame 节点，可以在此扩展。
    """
    import copy
    wf = copy.deepcopy(template)
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})
        # 替换提示词（CLIPTextEncode）
        if "text" in inputs and isinstance(inputs["text"], str) and not inputs["text"].startswith("##"):
            inputs["text"] = seg.prompt
        # 替换 seed
        if "seed" in inputs and isinstance(inputs["seed"], int):
            inputs["seed"] = seg.seed
        # 设置 first_frame
        if seg.first_frame_path and ("first_frame" in inputs or "image" in inputs):
            target_key = "first_frame" if "first_frame" in inputs else "image"
            inputs[target_key] = seg.first_frame_path
    return wf


# === CLI 入口（仅做 plan 和 concat，便于测试） ===

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plan":
        total = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
        seg = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
        plan = plan_segments(total, "(base prompt)", 7000, seg)
        for s in plan:
            print(f"  seg{s.index+1}: {s.target_seconds:.2f}s @ {snap_seconds_to_lattice(s.target_seconds)} frames, seed={s.seed}")
    else:
        print("Usage:")
        print("  python -m pipeline.long_video_orchestrator plan <total_seconds> [segment_seconds]")
        sys.exit(0)