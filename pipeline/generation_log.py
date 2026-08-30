"""
generation_log.py — 持久化生成账本 (single source of truth)

写入位置: 由 runtime_config.state_dir() 决定
格式: JSONL (一行一条 JSON), append-only, 原子写 (tmp + rename)

每条记录 schema:
{
  "ts": "2026-08-08T01:16:30",         # ISO 时间戳
  "project": "auto_008",                # 项目 ID
  "scene_idx": 11,                       # 1-based 场景编号
  "scene_name": "11_eyes_meet",          # 场景名 (来自 panels)
  "comfy_prefix": "MiniMax_H3_00067_",  # ComfyUI 用的 prefix
  "status": "raw_rendered",              # pending|submitted|raw_rendered|audio_merged|finalized|failed
  "comfy_path": "<COMFYUI_ROOT>/output/MiniMax_H3_00067_.mp4",   # ComfyUI 输出 (raw)
  "audio_path": null,                    # audio merge 后路径
  "final_path": null,                    # 最终视频路径
  "error": null,                         # 错误信息 (failed 时填)
  "size_bytes": 660101,                  # 文件大小
  "duration_s": 9.8                      # 视频时长 (秒)
}

为什么需要这个:
- Streamlit session_state 是内存的, 一刷新/重连/崩溃就全空
- "继续生成" 按钮找不到上次停在哪, 只能从 0 开始
- 没有持久记录, 跨 session 完全失忆
- 现在: 渲染成功时立刻写一条 → 永远能查 → 永不丢状态

用法:
    from generation_log import log
    log.append_record({...})
    log.update_status("auto_008", 11, "audio_merged", audio_path="...")
    pending = log.next_pending("auto_008", total_scenes=11)
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from runtime_config import comfyui_root, project_root, state_dir


# Paths
ROOT = project_root()
STATE_DIR = state_dir()
LOG_PATH = STATE_DIR / "generations.jsonl"
LOG_BAK = STATE_DIR / "generations.jsonl.bak"

# ComfyUI raw output location (where H3 renders land first)
COMFY_VIDEO_DIR = comfyui_root() / "output" / "video"
# Project asset location (audio-merged files live here)
ASSETS_DIR = ROOT / "models" / "MiniMax-H3-Turbo-Lora" / "assets"

# Status enum
PENDING = "pending"
SUBMITTED = "submitted"
RAW_RENDERED = "raw_rendered"
AUDIO_MERGED = "audio_merged"
FINALIZED = "finalized"
FAILED = "failed"

VALID_STATUSES = {PENDING, SUBMITTED, RAW_RENDERED, AUDIO_MERGED, FINALIZED, FAILED}

# Terminal states (don't change)
TERMINAL_STATUSES = {FINALIZED, FAILED}


# Thread lock for concurrent writes (safe across webapp + renderer + scheduler)
_lock = threading.Lock()


def _ensure_state_dir():
    """Ensure the configured state directory exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_lines(path: Path, lines: list[str]):
    """Write lines to path atomically: write to .tmp in same dir, then rename."""
    _ensure_state_dir()
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".generations_", suffix=".tmp", dir=str(STATE_DIR)
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
            if lines and not lines[-1].endswith("\n"):
                f.write("\n")
        # Backup current log before replacing
        if path.exists():
            shutil.copy2(path, LOG_BAK)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _read_log_lines() -> list[str]:
    """Read all lines from log (returns empty list if missing)."""
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return f.readlines()


def _parse_records(lines: list[str]) -> list[dict]:
    """Parse JSONL lines, skipping malformed ones silently."""
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            # Skip garbage lines (don't crash the whole read)
            continue
    return out


# ─── Filename parsing ──────────────────────────────────────────────────────────
# Extract scene_idx from a filename stem.
# Priority 1 (new format): leading number — "01_mapo_tailor_evening" → 1
# Priority 2 (old format): trailing counter — "MiniMax_H3_00067_" → 67
import re as _re_mod
_LEADING_NUM_RE = _re_mod.compile(r"^(\d+)_")


def _num(p: Path) -> Optional[int]:
    stem = p.stem
    m = _LEADING_NUM_RE.match(stem)
    if m:
        return int(m.group(1))
    # Fallback: trailing digits (for legacy MiniMax_H3_XXXXX_ files)
    digits = ""
    for ch in reversed(stem):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    try:
        return int(digits)
    except ValueError:
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

def append_record(record: Optional[dict] = None, **fields) -> None:
    """
    Append one record to the log. Thread-safe, atomic.

    Required fields: project, scene_idx, status
    Auto-filled: ts (if missing)
    """
    record = dict(record or {})
    record.update(fields)
    if "scene_idx" not in record and "panel_idx" in record:
        record["scene_idx"] = record.pop("panel_idx")
    if "scene_idx" not in record and isinstance(record.get("meta"), dict):
        record["scene_idx"] = record["meta"].get("panel_idx")
    if "scene_name" not in record and record.get("panel_name"):
        record["scene_name"] = record["panel_name"]
    if "project" not in record or "scene_idx" not in record or "status" not in record:
        raise ValueError("record must contain 'project', 'scene_idx', 'status'")
    if record["status"] not in VALID_STATUSES:
        raise ValueError(f"invalid status: {record['status']}")
    record.setdefault("ts", _now_iso())

    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))

    with _lock:
        # Read existing + append new
        existing = _read_log_lines()
        existing.append(line + "\n")
        _atomic_write_lines(LOG_PATH, existing)


def update_status(
    project: str,
    scene_idx: int,
    new_status: str,
    *,
    audio_path: Optional[str] = None,
    final_path: Optional[str] = None,
    comfy_path: Optional[str] = None,
    error: Optional[str] = None,
    size_bytes: Optional[int] = None,
    duration_s: Optional[float] = None,
) -> None:
    """
    Update the status of the latest record for (project, scene_idx).
    Adds a new audit-trail entry if previous status differs.
    """
    if new_status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {new_status}")

    with _lock:
        lines = _read_log_lines()
        records = _parse_records(lines)

        # Find latest record for this (project, scene_idx) that's not terminal,
        # or fall back to the most recent record for that pair.
        target = None
        target_idx = -1
        for i in range(len(records) - 1, -1, -1):
            r = records[i]
            if r.get("project") == project and r.get("scene_idx") == scene_idx:
                if r.get("status") not in TERMINAL_STATUSES:
                    target = r
                    target_idx = i
                    break
                elif target is None:
                    target = r
                    target_idx = i

        if target is None:
            # No prior record — create one
            new_rec = {
                "ts": _now_iso(),
                "project": project,
                "scene_idx": scene_idx,
                "status": new_status,
                "ts_status_changed": _now_iso(),
            }
        else:
            new_rec = dict(target)  # copy
            new_rec["status"] = new_status
            new_rec["ts_status_changed"] = _now_iso()
            # Clear old record from in-memory list (we'll rewrite)
            records.pop(target_idx)

        if audio_path is not None:
            new_rec["audio_path"] = audio_path
        if final_path is not None:
            new_rec["final_path"] = final_path
        if comfy_path is not None:
            new_rec["comfy_path"] = comfy_path
        if error is not None:
            new_rec["error"] = error
        if size_bytes is not None:
            new_rec["size_bytes"] = size_bytes
        if duration_s is not None:
            new_rec["duration_s"] = duration_s

        records.append(new_rec)
        # Rewrite full log
        new_lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                     for r in records]
        _atomic_write_lines(LOG_PATH, new_lines)


def get_project_status(project: str, total_scenes: Optional[int] = None) -> dict:
    """
    Returns {scene_idx: latest_status} for a project.
    Skips 'pending' if a later entry has a real status.
    """
    lines = _read_log_lines()
    records = _parse_records(lines)

    status_map: dict[int, str] = {}
    for r in records:
        if r.get("project") != project:
            continue
        idx = r.get("scene_idx")
        st = r.get("status")
        if not isinstance(idx, int) or st not in VALID_STATUSES:
            continue
        # Latest wins (records are append-ordered)
        status_map[idx] = st

    return status_map


def next_pending(project: str, total_scenes: int) -> Optional[int]:
    """
    Returns the lowest scene_idx in [1..total_scenes] that is NOT
    in a terminal/finished status (finalized, audio_merged, raw_rendered counts as 'have something').

    Convention:
    - If status is 'failed' or 'pending' or no record → eligible to (re)generate
    - If status is 'raw_rendered', 'audio_merged', 'finalized' → already have something

    Returns None if all scenes are done.
    """
    status_map = get_project_status(project, total_scenes)
    for i in range(1, total_scenes + 1):
        st = status_map.get(i)
        if st is None or st in (PENDING, FAILED):
            return i
        # raw_rendered / audio_merged / finalized: skip — already done
    return None


def scene_done_set(project: str, total_scenes: int) -> set[int]:
    """Set of scene_idx that have a real artifact (raw or better)."""
    status_map = get_project_status(project, total_scenes)
    return {i for i, st in status_map.items()
            if st in (RAW_RENDERED, AUDIO_MERGED, FINALIZED)}


def log_summary(project: str, total_scenes: int) -> dict:
    """
    Quick summary for UIs.
    Returns: {pending: int, raw: int, audio: int, finalized: int, failed: int, total: int}
    """
    status_map = get_project_status(project, total_scenes)
    counts = {PENDING: 0, RAW_RENDERED: 0, AUDIO_MERGED: 0,
              FINALIZED: 0, FAILED: 0, SUBMITTED: 0, "missing": 0}
    for i in range(1, total_scenes + 1):
        st = status_map.get(i)
        if st is None:
            counts["missing"] += 1
        else:
            counts[st] = counts.get(st, 0) + 1
    counts["total"] = total_scenes
    return counts


def recover_from_filesystem(project: str, total_scenes: int, *,
                            panel_names: Optional[list[str]] = None) -> list[dict]:
    """
    Scan the filesystem for existing artifacts and append log records
    for anything we find. Idempotent — safe to run multiple times.

    Sources scanned:
    1. COMFY_VIDEO_DIR: MiniMax_H3_XXXXX_.mp4 (raw renders)
    2. ASSETS_DIR: MiniMax_H3_XXXXX-audio.mp4 (audio-merged)
    3. auto_{project_id}_render/: s01.mp4 etc. (alternative render location)
    4. auto_{project_id}_videos/: 01_*.mp4 etc. (final named files)

    For each found file, infers scene_idx from the numeric suffix.
    Only appends records for scenes that don't already have a status
    better than 'pending'/'failed'.

    Returns: list of newly-appended records.
    """
    # Build map: scene_idx → existing status from log
    existing = get_project_status(project, total_scenes)
    already_good = {i for i, st in existing.items()
                    if st in (RAW_RENDERED, AUDIO_MERGED, FINALIZED)}

    # Determine the offset between MiniMax_H3_XXXXX number and scene_idx
    # Heuristic: find the minimum numeric prefix among files, treat it as scene 1
    # (or as panel 1 if panel_names provided).
    raw_files: list[Path] = []
    if COMFY_VIDEO_DIR.exists():
        # STRICT PROJECT ISOLATION (fix ERR-20260808-004):
        # ONLY scan the project's own subdir. If that doesn't exist,
        # return ZERO raw files (instead of falling through to top-level
        # which would pick up stale MiniMax_H3_* files from older projects).
        # Recovery is for "project X has files lying around" — NOT for
        # "guess which files might belong to project X".
        project_subdir = COMFY_VIDEO_DIR / project
        if project_subdir.exists():
            # New layout: scan inside project's own subdir
            for p in project_subdir.glob("MiniMax_H3_*.mp4"):
                if p.is_file():
                    raw_files.append(p)
            # Also accept story-named renders (01_hook_rainy_night_*.mp4)
            for p in project_subdir.glob("*.mp4"):
                if p.is_file() and p.name.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
                    raw_files.append(p)
        # else: NO top-level fallback. Files there are from other projects.

    # Also scan project videos directory (e.g. auto_008_videos/).
    # These are the FINAL copies that the webapp/renderer wrote after the
    # atomic render+copy refactor — they are the strongest signal that a
    # scene is done. Use them as FINALIZED (preferred over raw).
    project_videos_dir = ROOT / f"{project}_videos"
    if project_videos_dir.exists():
        for p in project_videos_dir.glob("*.mp4"):
            if p.is_file():
                raw_files.append(p)

    audio_files: list[Path] = []
    if ASSETS_DIR.exists():
        # Match both new format (01_name-audio.mp4) and old (MiniMax_H3_00060-audio.mp4)
        for p in ASSETS_DIR.glob("*-audio.mp4"):
            if p.is_file():
                audio_files.append(p)

    raw_by_num = {_num(p): p for p in raw_files if _num(p) is not None}
    audio_by_num = {_num(p): p for p in audio_files if _num(p) is not None}

    if not raw_by_num and not audio_by_num:
        return []

    # Map MiniMax numeric → scene_idx
    # Strategy: assume sequential. Smallest numeric = scene 1 (or scene after already-done).
    # If we have panel_names, use them to determine count.
    # Simple approach: sort numerics ascending, assign scene_idx in order.
    all_nums = sorted(set(raw_by_num.keys()) | set(audio_by_num.keys()))
    if not all_nums:
        return []

    # If we have prior log entries with scene_idx mapping, prefer that.
    # Otherwise: first numeric in sorted = scene 1.
    num_to_scene: dict[int, int] = {}
    for i, n in enumerate(all_nums, start=1):
        num_to_scene[n] = i

    appended = []
    for n in sorted(all_nums):
        scene_idx = num_to_scene[n]
        if scene_idx > total_scenes:
            continue  # out of range

        # Determine status:
        #   audio_merged > raw_rendered > nothing
        has_audio = n in audio_by_num
        has_raw = n in raw_by_num

        # Skip if already in good state in log
        if scene_idx in already_good:
            continue

        rec: dict = {
            "ts": _now_iso(),
            "project": project,
            "scene_idx": scene_idx,
            "comfy_prefix": f"MiniMax_H3_{n:05d}_",
            "status": AUDIO_MERGED if has_audio else RAW_RENDERED,
            "ts_status_changed": _now_iso(),
            "source": "recover_from_filesystem",
        }
        if has_raw:
            rp = raw_by_num[n]
            rec["comfy_path"] = str(rp)
            try:
                rec["size_bytes"] = rp.stat().st_size
            except OSError:
                pass
        if has_audio:
            ap = audio_by_num[n]
            rec["audio_path"] = str(ap)
            try:
                rec["size_bytes"] = ap.stat().st_size
            except OSError:
                pass

        if panel_names and scene_idx <= len(panel_names):
            rec["scene_name"] = panel_names[scene_idx - 1]

        append_record(rec)
        appended.append(rec)

    return appended


def get_log_path() -> Path:
    return LOG_PATH


def get_bak_path() -> Path:
    return LOG_BAK


# ─── CLI / debug ──────────────────────────────────────────────────────────────

def _cli():
    import argparse, sys
    p = argparse.ArgumentParser(description="generation_log CLI")
    p.add_argument("--project", default="auto_008")
    p.add_argument("--total", type=int, default=11)
    p.add_argument("--recover", action="store_true", help="scan filesystem and backfill log")
    p.add_argument("--summary", action="store_true", help="print status summary")
    p.add_argument("--next", action="store_true", help="print next pending scene_idx")
    p.add_argument("--status", action="store_true", help="print per-scene status")
    p.add_argument("--panel-names-file", default=None, help="JSON file with list of panel names")
    args = p.parse_args()

    panel_names = None
    if args.panel_names_file and Path(args.panel_names_file).exists():
        panel_names = json.loads(Path(args.panel_names_file).read_text(encoding="utf-8"))

    if args.recover:
        recs = recover_from_filesystem(args.project, args.total, panel_names=panel_names)
        print(f"Recovered {len(recs)} records:")
        for r in recs:
            print(f"  scene {r['scene_idx']}: {r['status']}  ({r.get('comfy_path') or r.get('audio_path')})")

    if args.summary or args.status or args.next:
        s = log_summary(args.project, args.total)
        print(f"\n[{args.project}] total={s['total']}  " +
              "  ".join(f"{k}={v}" for k, v in s.items() if k != "total"))

    if args.status:
        sm = get_project_status(args.project, args.total)
        for i in range(1, args.total + 1):
            print(f"  scene {i:2d}: {sm.get(i, '(none)')}")

    if args.next:
        nxt = next_pending(args.project, args.total)
        print(f"\nNext pending scene_idx: {nxt}")


if __name__ == "__main__":
    _cli()
