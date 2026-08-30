"""Single-episode background worker for character assets and H3 render jobs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Iterable

from continuity_safe import DEFAULT_SAPI_VOICE, run_continuity_safe_chain
from shot_group_anchor import generate_group_anchor
from orchestrator import (
    prepare_all_assets as generate_all_assets_stage,
    prepare_character_assets as generate_character_assets_stage,
    run_episode_jobs,
)
from runtime_config import project_root
from task_store import default_store, project_snapshot


def run_worker(
    ep_id: str,
    *,
    statuses: Iterable[str] = ("pending", "failed"),
    ensure_character_assets: bool = True,
    timeout: float = 2400.0,
    character_assets_only: bool = False,
    assets_only: bool = False,
    continuity_safe_from: str | None = None,
    continuity_voice: str = DEFAULT_SAPI_VOICE,
    continuity_motion: str = "slow_push",
    continuity_burn_subtitles: bool | None = None,
    group_anchor_job_id: str | None = None,
    max_jobs: int | None = None,
) -> dict:
    """Run one episode under a durable per-episode worker lease."""
    store = default_store()
    launch_token = os.environ.pop("AI_MANGA_WORKER_LAUNCH_TOKEN", None)
    if not store.acquire_worker(ep_id, launch_token=launch_token):
        return {"ep_id": ep_id, "started": False, "reason": "worker_already_running", "snapshot": project_snapshot(ep_id)}
    heartbeat_stop = threading.Event()

    def keep_lease_alive() -> None:
        while not heartbeat_stop.wait(30.0):
            store.heartbeat_worker(ep_id)

    heartbeat = threading.Thread(target=keep_lease_alive, name=f"worker-heartbeat-{ep_id}", daemon=True)
    heartbeat.start()
    try:
        if group_anchor_job_id:
            result = generate_group_anchor(
                ep_id, group_anchor_job_id, store=store, timeout=timeout,
            )
            return {
                "ep_id": ep_id, "started": True, "mode": "group_anchor",
                "job_id": group_anchor_job_id, "candidate": result,
            }
        if continuity_safe_from:
            result = run_continuity_safe_chain(
                ep_id, continuity_safe_from,
                preferred_voice=continuity_voice,
                motion=continuity_motion,
                # Compatibility-only input.  Panel clips are always clean;
                # delivery export owns the optional one-time subtitle burn.
                burn_subtitles=continuity_burn_subtitles,
                timeout=timeout,
            )
            return {"ep_id": ep_id, "started": True, "mode": "continuity_safe", **result}
        if character_assets_only or assets_only:
            stage = generate_all_assets_stage if assets_only else generate_character_assets_stage
            snapshot = stage(
                ep_id,
                progress_cb=lambda phase, message: print(f"[{phase}] {message}", flush=True),
            )
            mode = "assets_only" if assets_only else "character_assets_only"
            return {"ep_id": ep_id, "started": True, "mode": mode, "snapshot": snapshot}
        return {
            "started": True,
            **run_episode_jobs(
                ep_id,
                statuses=tuple(statuses),
                ensure_character_assets=ensure_character_assets,
                timeout=timeout,
                max_jobs=max_jobs,
                progress_cb=lambda phase, message: print(f"[{phase}] {message}", flush=True),
            ),
        }
    finally:
        heartbeat_stop.set()
        heartbeat.join(timeout=2.0)
        store.release_worker(ep_id)


def start_worker(
    ep_id: str,
    *,
    statuses: Iterable[str] = ("pending", "failed"),
    ensure_character_assets: bool = True,
    timeout: float = 2400.0,
    max_jobs: int | None = None,
) -> dict:
    """Start a hidden non-blocking process; Web should call this entry point."""
    return _start_worker_process(
        ep_id,
        statuses=statuses,
        ensure_character_assets=ensure_character_assets,
        timeout=timeout,
        character_assets_only=False,
        assets_only=False,
        continuity_safe_from=None,
        group_anchor_job_id=None,
        max_jobs=max_jobs,
    )


def start_character_assets(ep_id: str, *, timeout: float = 1800.0) -> dict:
    """Start only the missing-character-assets stage in a hidden worker."""
    return _start_worker_process(
        ep_id,
        statuses=(),
        ensure_character_assets=True,
        timeout=timeout,
        character_assets_only=True,
        assets_only=False,
        continuity_safe_from=None,
        group_anchor_job_id=None,
    )


def start_assets(ep_id: str, *, timeout: float = 1800.0) -> dict:
    """Start the character+scene asset dependency stage without H3 panels."""
    return _start_worker_process(
        ep_id,
        statuses=(),
        ensure_character_assets=True,
        timeout=timeout,
        character_assets_only=False,
        assets_only=True,
        continuity_safe_from=None,
        group_anchor_job_id=None,
    )


def start_continuity_safe(
    ep_id: str,
    job_id: str,
    *,
    preferred_voice: str = DEFAULT_SAPI_VOICE,
    motion: str = "slow_push",
    burn_subtitles: bool | None = None,
    timeout: float = 900.0,
) -> dict:
    """Start the explicit CPU/FFmpeg continuity-safe chain in a hidden worker.

    ``burn_subtitles`` is accepted for compatibility but ignored by the render
    stage.  It may be removed by callers at their convenience.
    """
    if not str(job_id).strip():
        raise ValueError("job_id is required")
    return _start_worker_process(
        ep_id,
        statuses=(),
        ensure_character_assets=False,
        timeout=timeout,
        character_assets_only=False,
        assets_only=False,
        continuity_safe_from=str(job_id),
        continuity_voice=str(preferred_voice or DEFAULT_SAPI_VOICE),
        continuity_motion=motion,
        continuity_burn_subtitles=burn_subtitles,
        group_anchor_job_id=None,
    )


def start_group_anchor(ep_id: str, job_id: str, *, timeout: float = 900.0) -> dict:
    """Generate one reviewable group-composition anchor off the Web thread."""
    if not str(job_id).strip():
        raise ValueError("job_id is required")
    return _start_worker_process(
        ep_id,
        statuses=(),
        ensure_character_assets=False,
        timeout=timeout,
        character_assets_only=False,
        assets_only=False,
        continuity_safe_from=None,
        group_anchor_job_id=str(job_id),
    )


def _start_worker_process(
    ep_id: str,
    *,
    statuses: Iterable[str],
    ensure_character_assets: bool,
    timeout: float,
    character_assets_only: bool,
    assets_only: bool,
    continuity_safe_from: str | None = None,
    continuity_voice: str = DEFAULT_SAPI_VOICE,
    continuity_motion: str = "slow_push",
    continuity_burn_subtitles: bool | None = None,
    group_anchor_job_id: str | None = None,
    max_jobs: int | None = None,
) -> dict:
    store = default_store()
    active = store.worker_info(ep_id)
    if active and active["active"]:
        return {"ep_id": ep_id, "started": False, "reason": "worker_already_running", "pid": active["pid"]}
    launch_token = uuid.uuid4().hex
    if not store.reserve_worker_launch(ep_id, launch_token):
        active = store.worker_info(ep_id) or {}
        return {
            "ep_id": ep_id, "started": False, "reason": "worker_already_running",
            "pid": active.get("pid"),
        }
    logs = project_root() / "logs" / "workers"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{ep_id}.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--ep-id", ep_id,
        "--statuses", ",".join(statuses),
        "--timeout", str(timeout),
    ]
    if max_jobs is not None:
        if int(max_jobs) <= 0:
            store.release_worker_launch(ep_id, launch_token)
            raise ValueError("max_jobs must be positive when provided")
        command.extend(["--max-jobs", str(int(max_jobs))])
    if not ensure_character_assets:
        command.append("--no-character-assets")
    if character_assets_only:
        command.append("--character-assets-only")
    if assets_only:
        command.append("--assets-only")
    if continuity_safe_from:
        command.extend([
            "--continuity-safe-from", continuity_safe_from,
            "--continuity-voice", continuity_voice,
            "--continuity-motion", continuity_motion,
        ])
    if group_anchor_job_id:
        command.extend(["--group-anchor-job-id", group_anchor_job_id])
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    child_env = os.environ.copy()
    child_env["PYTHONUTF8"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["AI_MANGA_WORKER_LAUNCH_TOKEN"] = launch_token
    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(project_root()),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                close_fds=True,
                env=child_env,
            )
        store.set_worker_launch_pid(ep_id, launch_token, process.pid)
    except Exception:
        store.release_worker_launch(ep_id, launch_token)
        raise
    return {"ep_id": ep_id, "started": True, "pid": process.pid, "log_path": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="AI manga episode worker")
    parser.add_argument("--ep-id", required=True)
    parser.add_argument("--statuses", default="pending,failed")
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--no-character-assets", action="store_true")
    parser.add_argument("--character-assets-only", action="store_true")
    parser.add_argument("--assets-only", action="store_true")
    parser.add_argument("--continuity-safe-from")
    parser.add_argument("--group-anchor-job-id")
    parser.add_argument("--continuity-voice", default=DEFAULT_SAPI_VOICE)
    parser.add_argument("--continuity-motion", choices=("slow_push", "locked"), default="slow_push")
    parser.add_argument(
        "--no-continuity-burn-subtitles", action="store_true",
        help="deprecated no-op; panel clips never burn subtitles",
    )
    args = parser.parse_args()
    statuses = tuple(value.strip() for value in args.statuses.split(",") if value.strip())
    result = run_worker(
        args.ep_id,
        statuses=statuses,
        ensure_character_assets=not args.no_character_assets,
        timeout=args.timeout,
        character_assets_only=args.character_assets_only,
        assets_only=args.assets_only,
        continuity_safe_from=args.continuity_safe_from,
        continuity_voice=args.continuity_voice,
        continuity_motion=args.continuity_motion,
        continuity_burn_subtitles=None,
        group_anchor_job_id=args.group_anchor_job_id,
        max_jobs=args.max_jobs,
    )
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
    if not result.get("started"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
