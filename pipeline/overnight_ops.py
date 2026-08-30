"""Fail-closed, auditable controller for unattended overnight production.

The controller owns policy and scheduling only.  It never imports a renderer
or submits a Comfy graph.  Actual work is started through the public service
callbacks supplied by :mod:`render_service`.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from runtime_config import comfyui_root, comfyui_server, project_root, projects_dir, state_dir
from task_store import default_store, project_snapshot


JsonDict = dict[str, Any]
TERMINAL = {"succeeded", "failed", "cancelled", "rejected"}
ACTIVE = {"queued", "submitted", "running"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _http_json(url: str, timeout: float = 10.0) -> JsonDict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object response from {url}")
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OvernightPolicy:
    max_shots: int = 12
    max_failures: int = 2
    max_total_retries: int = 2
    max_retries_per_job: int = 2
    max_runtime_hours: float = 10.0
    minimum_start_window_minutes: float = 20.0
    poll_seconds: float = 15.0
    worker_timeout_minutes: float = 90.0
    minimum_free_disk_gb: float = 50.0
    maximum_gpu_temperature_c: float = 78.0
    minimum_free_vram_mb: int = 4096
    require_gpu_probe: bool = True
    required_nodes: tuple[str, ...] = (
        "UNETLoader", "VAELoader", "CLIPLoader", "LoadImage", "SaveVideo",
        "MiniMaxH3ReferenceToVideo", "MiniMaxH3TurboLoRA", "MiniMaxH3TurboSampler",
        "PathchSageAttentionKJ",
    )
    recommended_nodes: tuple[str, ...] = ("MiniMaxH3AddGuide",)

    def validate(self) -> None:
        positive = {
            "max_shots": self.max_shots,
            "max_failures": self.max_failures,
            "max_total_retries": self.max_total_retries,
            "max_retries_per_job": self.max_retries_per_job,
            "max_runtime_hours": self.max_runtime_hours,
            "worker_timeout_minutes": self.worker_timeout_minutes,
            "minimum_free_disk_gb": self.minimum_free_disk_gb,
            "maximum_gpu_temperature_c": self.maximum_gpu_temperature_c,
            "minimum_free_vram_mb": self.minimum_free_vram_mb,
        }
        invalid = [name for name, value in positive.items() if float(value) <= 0]
        if invalid:
            raise ValueError("overnight policy values must be positive: " + ",".join(invalid))
        if self.minimum_start_window_minutes < 0 or self.poll_seconds < 0:
            raise ValueError("start window and poll interval cannot be negative")


@dataclass(frozen=True)
class GpuSnapshot:
    available: bool
    name: str | None = None
    temperature_c: float | None = None
    memory_total_mb: int | None = None
    memory_used_mb: int | None = None
    memory_free_mb: int | None = None
    driver_version: str | None = None
    error: str | None = None


def probe_nvidia_gpu(*, runner: Callable[..., Any] = subprocess.run) -> GpuSnapshot:
    command = [
        "nvidia-smi", "--query-gpu=name,temperature.gpu,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits", "--id=0",
    ]
    try:
        result = runner(command, check=True, capture_output=True, text=True, timeout=10)
        row = next((line.strip() for line in str(result.stdout).splitlines() if line.strip()), "")
        values = [part.strip() for part in row.split(",")]
        if len(values) != 6:
            raise RuntimeError(f"unexpected nvidia-smi output: {row!r}")
        return GpuSnapshot(
            available=True, name=values[0], temperature_c=float(values[1]),
            memory_total_mb=int(values[2]), memory_used_mb=int(values[3]),
            memory_free_mb=int(values[4]), driver_version=values[5],
        )
    except Exception as exc:
        return GpuSnapshot(available=False, error=f"nvidia-smi unavailable or invalid: {exc}")


class SingleGpuLease:
    """Atomic cross-process lease with stale-owner recovery."""

    def __init__(self, path: Path, run_id: str, *, now_fn: Callable[[], datetime] = _utc_now, stale_seconds: float = 180.0):
        self.path = path.resolve()
        self.run_id = run_id
        self.now_fn = now_fn
        self.stale_seconds = stale_seconds
        self.token = uuid.uuid4().hex
        self.acquired = False

    def _record(self) -> JsonDict:
        return {"run_id": self.run_id, "token": self.token, "pid": os.getpid(), "heartbeat_at": _iso(self.now_fn())}

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(2):
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(self._record(), handle, ensure_ascii=False)
                self.acquired = True
                return True
            except FileExistsError:
                try:
                    record = json.loads(self.path.read_text(encoding="utf-8"))
                    heartbeat = datetime.fromisoformat(str(record["heartbeat_at"]).replace("Z", "+00:00"))
                    stale = (self.now_fn() - heartbeat).total_seconds() > self.stale_seconds
                except Exception:
                    stale = False
                if not stale:
                    return False
                stale_path = self.path.with_name(f"{self.path.name}.stale.{uuid.uuid4().hex}")
                try:
                    os.replace(self.path, stale_path)
                except OSError:
                    return False
        return False

    def heartbeat(self) -> None:
        if self.acquired:
            _atomic_json(self.path, self._record())

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
            if record.get("token") == self.token:
                self.path.unlink(missing_ok=True)
        finally:
            self.acquired = False


def _node_snapshot(object_info: Mapping[str, Any], required_nodes: Sequence[str]) -> JsonDict:
    present = sorted(str(name) for name in object_info)
    missing = sorted(name for name in required_nodes if name not in object_info)
    selected: JsonDict = {}
    model_options: JsonDict = {}
    for name in required_nodes:
        node = object_info.get(name)
        if not isinstance(node, Mapping):
            continue
        selected[name] = {
            key: node.get(key) for key in ("display_name", "category", "description", "python_module") if node.get(key) is not None
        }
        required = ((node.get("input") or {}).get("required") or {}) if isinstance(node.get("input"), Mapping) else {}
        for input_name, spec in required.items():
            if isinstance(spec, list) and spec and isinstance(spec[0], list):
                values = [str(value) for value in spec[0]]
                if any(token in str(input_name).lower() for token in ("model", "unet", "vae", "clip", "lora")):
                    model_options[f"{name}.{input_name}"] = values
    return {
        "object_info_sha256": _sha256_json(object_info), "node_count": len(present),
        "required": selected, "missing_required": missing,
        "model_options": model_options,
    }


def _git_revision(path: Path) -> str | None:
    try:
        head = (path / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            reference = head.split(":", 1)[1].strip()
            return (path / ".git" / reference).read_text(encoding="utf-8").strip() or None
        return head or None
    except OSError:
        return None


def _installation_snapshot(root: Path) -> JsonDict:
    """Record lightweight local revisions and H3-related filenames/sizes."""
    custom_root = root / "custom_nodes"
    custom_nodes: list[JsonDict] = []
    if custom_root.is_dir():
        for child in sorted(custom_root.iterdir(), key=lambda path: path.name.lower()):
            if child.is_dir() and any(token in child.name.lower() for token in ("minimax", "h3", "sage")):
                custom_nodes.append({"name": child.name, "revision": _git_revision(child)})
    model_files: list[JsonDict] = []
    model_root = root / "models"
    if model_root.is_dir():
        for path in sorted(model_root.rglob("*")):
            if path.is_file() and any(token in path.name.lower() for token in ("minimax", "h3", "turbo")):
                try:
                    model_files.append({
                        "path": str(path.relative_to(root)).replace("\\", "/"),
                        "size": path.stat().st_size,
                    })
                except OSError:
                    continue
    return {
        "comfyui_revision": _git_revision(root), "custom_nodes": custom_nodes,
        "h3_model_files": model_files, "snapshot_sha256": _sha256_json({
            "custom_nodes": custom_nodes, "h3_model_files": model_files,
        }),
    }


def _remaining_jobs(snapshot: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(job) for job in snapshot.get("jobs") or [] if str(job.get("status")) != "succeeded"]


def _job_content_blocker(job: Mapping[str, Any]) -> JsonDict | None:
    """Return a fail-closed blocker for a technically succeeded artifact.

    A container/probe success is intentionally insufficient.  The stored QA
    must explicitly pass, identify decoded visual bytes, and bind its source
    path to the current output.  This mirrors the public UI/release gate while
    keeping the overnight controller independent of Web internals.
    """
    if str(job.get("status") or "").lower() != "succeeded":
        return None
    metadata = job.get("metadata") if isinstance(job.get("metadata"), Mapping) else {}
    qa = metadata.get("content_qa") if isinstance(metadata.get("content_qa"), Mapping) else {}
    analysis = qa.get("analysis") if isinstance(qa.get("analysis"), Mapping) else {}
    qa_status = str(qa.get("status") or qa.get("result") or "").lower()
    passed = qa.get("passed") is True or qa_status in {"passed", "pass"}
    visual_hash = str(analysis.get("decoded_visual_sha256") or qa.get("decoded_visual_sha256") or "").strip()
    source = str(analysis.get("source_path") or qa.get("source_path") or "").strip()
    output = str(job.get("output_path") or "").strip()
    reasons = [str(reason) for reason in qa.get("reasons") or [] if str(reason).strip()]
    if not passed:
        reason = "content_qa_failed" if qa else "content_qa_missing"
    elif not visual_hash:
        reason = "content_qa_visual_hash_missing"
    elif not source or not output:
        reason = "content_qa_source_binding_missing"
    else:
        try:
            reason = None if Path(source).resolve() == Path(output).resolve() else "content_qa_source_stale"
        except OSError:
            reason = "content_qa_source_invalid"
    if reason is None:
        return None
    return {
        "job_id": str(job.get("job_id") or ""), "panel_index": job.get("panel_index"),
        "technical_status": str(job.get("status") or ""), "reason": reason,
        "qa_status": qa_status or ("passed" if passed else "missing"),
        "qa_reasons": reasons, "error": job.get("error"),
    }


def _release_blockers(snapshot: Mapping[str, Any]) -> list[JsonDict]:
    """Describe why generated content is not eligible for manual export."""
    blockers: list[JsonDict] = []
    for job in snapshot.get("jobs") or []:
        if str(job.get("status") or "").lower() != "succeeded":
            continue
        metadata = job.get("metadata") if isinstance(job.get("metadata"), Mapping) else {}
        artifact = str(metadata.get("artifact_sha256") or "")
        qa = metadata.get("content_qa") if isinstance(metadata.get("content_qa"), Mapping) else {}
        analysis = qa.get("analysis") if isinstance(qa.get("analysis"), Mapping) else {}
        visual = str(analysis.get("decoded_visual_sha256") or "")
        selection = metadata.get("edit_selection") if isinstance(metadata.get("edit_selection"), Mapping) else {}
        selection_hash = str(selection.get("selection_sha256") or "")
        review = metadata.get("editorial_review") if isinstance(metadata.get("editorial_review"), Mapping) else {}
        release = metadata.get("release") if isinstance(metadata.get("release"), Mapping) else {}
        reasons: list[str] = []
        if not selection_hash or str(selection.get("source_artifact_sha256") or "") != artifact:
            reasons.append("edit_selection_missing_or_stale")
        if (
            str(review.get("status") or "").lower() != "approved"
            or str(review.get("artifact_sha256") or "") != artifact
            or str(review.get("decoded_visual_sha256") or "") != visual
            or str(review.get("edit_selection_sha256") or "") != selection_hash
        ):
            reasons.append("editorial_review_missing_or_stale")
        if (
            str(release.get("status") or "").lower() not in {"approved", "released"}
            or str(release.get("artifact_sha256") or "") != artifact
            or str(release.get("decoded_visual_sha256") or "") != visual
            or str(release.get("edit_selection_sha256") or "") != selection_hash
        ):
            reasons.append("release_missing_or_stale")
        if reasons:
            blockers.append({
                "job_id": str(job.get("job_id") or ""), "panel_index": job.get("panel_index"),
                "reasons": reasons,
            })
    pipeline = snapshot.get("pipeline") if isinstance(snapshot.get("pipeline"), Mapping) else {}
    if str(pipeline.get("release_status") or "").lower() not in {"approved", "released"}:
        blockers.append({"scope": "episode", "reasons": ["episode_release_not_approved"]})
    return blockers


def _frozen_job(job: Mapping[str, Any]) -> JsonDict:
    metadata = dict(job.get("metadata") or {})
    inputs = dict(metadata.get("inputs") or {})
    return {
        "job_id": job.get("job_id"), "panel_index": job.get("panel_index"),
        "panel_name": job.get("panel_name"), "status": job.get("status"),
        "prompt_id": job.get("prompt_id"),
        "input_hash": job.get("input_hash"), "retry_count": int(job.get("retry_count") or 0),
        "max_retries": int(job.get("max_retries") or 0),
        "settings": metadata.get("settings") or {},
        "asset_dependencies": inputs.get("asset_dependencies") or [],
        "reference_inputs": inputs.get("reference_inputs") or [],
        "continuity": inputs.get("continuity") or metadata.get("continuity") or {},
        "content_qa": metadata.get("content_qa") or {},
        "edit_selection": metadata.get("edit_selection") or {},
        "editorial_review": metadata.get("editorial_review") or {},
        "release": metadata.get("release") or {},
    }


def _report_markdown(report: Mapping[str, Any]) -> str:
    preflight = report.get("preflight") or {}
    lines = [
        f"# Night run {report.get('run_id')}", "",
        f"- Status: **{report.get('status')}**",
        f"- Started: {report.get('started_at')}",
        f"- Finished: {report.get('finished_at')}",
        f"- Frozen manifest: `{report.get('production_manifest_path') or 'not created'}`",
        f"- Preflight: {'PASS' if preflight.get('passed') else 'FAIL'}",
        f"- Shots started: {report.get('shots_started', 0)}",
        f"- Retry operations: {report.get('retry_operations', 0)}",
        f"- Dead letters: {len(report.get('dead_letters') or [])}", "",
        f"- Automatic export attempted: **{bool(report.get('automatic_export_attempted'))}**", "",
    ]
    if preflight.get("failures"):
        lines.extend(["## Preflight failures", "", *[f"- {item}" for item in preflight["failures"]], ""])
    if preflight.get("warnings"):
        lines.extend(["## Warnings", "", *[f"- {item}" for item in preflight["warnings"]], ""])
    lines.extend(["## Episodes", ""])
    for episode in report.get("episodes") or []:
        lines.append(
            f"- `{episode.get('ep_id')}`: {episode.get('status')} "
            f"(started={episode.get('shots_started', 0)}, retries={episode.get('retry_operations', 0)})"
        )
        for event in episode.get("reconciliation_events") or []:
            lines.append(
                f"  - Reconcile `{event.get('job_id')}` / `{event.get('prompt_id')}`: "
                f"{event.get('disposition')} ({event.get('reason')})"
            )
        for blocker in episode.get("content_qa_blockers") or []:
            lines.append(
                f"  - Content QA blocker `{blocker.get('job_id')}`: {blocker.get('reason')}"
                + (f" — {', '.join(blocker.get('qa_reasons') or [])}" if blocker.get("qa_reasons") else "")
            )
        for blocker in episode.get("release_blockers") or []:
            identity = blocker.get("job_id") or blocker.get("scope") or "unknown"
            lines.append(f"  - Release blocker `{identity}`: {', '.join(blocker.get('reasons') or [])}")
    return "\n".join(lines) + "\n"


def preflight(
    ep_ids: Sequence[str], policy: OvernightPolicy, *, stop_at: datetime,
    status_fn: Callable[[str], JsonDict] = project_snapshot,
    http_json_fn: Callable[[str, float], JsonDict] = _http_json,
    gpu_probe_fn: Callable[[], GpuSnapshot] = probe_nvidia_gpu,
    worker_info_fn: Callable[[str], Mapping[str, Any] | None] | None = None,
    active_workers_fn: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    now_fn: Callable[[], datetime] = _utc_now,
) -> JsonDict:
    failures: list[str] = []
    warnings: list[str] = []
    now = now_fn()
    if stop_at.tzinfo is None:
        stop_at = stop_at.replace(tzinfo=timezone.utc)
    if stop_at <= now + timedelta(minutes=policy.minimum_start_window_minutes):
        failures.append("stop window leaves insufficient time to start a new job")
    try:
        system_stats = http_json_fn(f"{comfyui_server()}/system_stats", 10.0)
        object_info = http_json_fn(f"{comfyui_server()}/object_info", 20.0)
        queue = http_json_fn(f"{comfyui_server()}/queue", 10.0)
    except Exception as exc:
        system_stats, object_info, queue = {}, {}, {}
        failures.append(f"ComfyUI health check failed: {exc}")
    node_snapshot = _node_snapshot(object_info, policy.required_nodes)
    recommended_snapshot = _node_snapshot(object_info, policy.recommended_nodes)
    installation_snapshot = _installation_snapshot(comfyui_root())
    if node_snapshot["missing_required"]:
        failures.append("ComfyUI missing required nodes: " + ",".join(node_snapshot["missing_required"]))
    if recommended_snapshot["missing_required"]:
        warnings.append(
            "ComfyUI is missing recommended continuity capabilities: "
            + ",".join(recommended_snapshot["missing_required"])
            + "; Ref2VA identity references remain available, but arbitrary-frame hard guides are unavailable"
        )
    if queue.get("queue_running") or queue.get("queue_pending"):
        failures.append("ComfyUI already has running or pending work")
    gpu = gpu_probe_fn()
    if not gpu.available:
        message = gpu.error or "GPU telemetry is unavailable"
        (failures if policy.require_gpu_probe else warnings).append(message)
    else:
        if gpu.temperature_c is None or gpu.temperature_c >= policy.maximum_gpu_temperature_c:
            failures.append(f"GPU temperature is unsafe: {gpu.temperature_c}C")
        if gpu.memory_free_mb is None or gpu.memory_free_mb < policy.minimum_free_vram_mb:
            failures.append(f"GPU free VRAM below threshold: {gpu.memory_free_mb}MB")
    disk = shutil.disk_usage(projects_dir())
    free_disk_gb = disk.free / (1024 ** 3)
    if free_disk_gb < policy.minimum_free_disk_gb:
        failures.append(f"project disk free space below threshold: {free_disk_gb:.2f}GB")
    episodes: list[JsonDict] = []
    shot_count = 0
    # Dependency-injected worker probes must form one isolated view of worker
    # state. Falling back to the process-global SQLite store for only the
    # active-worker half can leak an unrelated live production worker into an
    # offline audit/test using a synthetic worker_info_fn.
    worker_probe_injected = worker_info_fn is not None
    worker_info_fn = worker_info_fn or (lambda ep_id: default_store().worker_info(ep_id))
    if active_workers_fn is None:
        active_workers_fn = (lambda: []) if worker_probe_injected else default_store().active_workers
    active_workers = [dict(worker) for worker in active_workers_fn()]
    if active_workers:
        failures.append("one or more production workers/launch reservations are already active")
    for ep_id in ep_ids:
        snapshot = status_fn(ep_id)
        remaining = _remaining_jobs(snapshot)
        shot_count += len(remaining)
        worker = worker_info_fn(ep_id)
        if worker and worker.get("active"):
            failures.append(f"episode {ep_id} already has an active worker")
        if not snapshot.get("jobs"):
            failures.append(f"episode {ep_id} has no registered jobs")
        episodes.append({
            "ep_id": ep_id, "project_dir": snapshot.get("project_dir"),
            "remaining_shots": len(remaining), "jobs": [_frozen_job(job) for job in snapshot.get("jobs") or []],
        })
    if shot_count > policy.max_shots:
        warnings.append(
            f"planned remaining shots {shot_count} exceed nightly maximum {policy.max_shots}; "
            "the controller will stop before the next shot at the budget"
        )
    return {
        "passed": not failures, "checked_at": _iso(now), "failures": failures, "warnings": warnings,
        "disk": {"path": str(projects_dir()), "free_gb": round(free_disk_gb, 3)},
        "gpu": asdict(gpu), "comfy": {"server": comfyui_server(), "system_stats": system_stats,
        "queue": queue, "nodes": node_snapshot, "recommended_nodes": recommended_snapshot,
        "installation": installation_snapshot},
        "active_workers": active_workers, "episodes": episodes,
    }


def _wait_for_worker(
    ep_id: str, job_id: str, *, status_fn: Callable[[str], JsonDict], deadline: datetime,
    timeout_at: datetime, policy: OvernightPolicy, lease: SingleGpuLease,
    sleep_fn: Callable[[float], None], now_fn: Callable[[], datetime],
) -> JsonDict:
    """Wait for the one-shot worker without applying start-only gates.

    A running H3 process naturally consumes VRAM and can temporarily cross a
    temperature threshold.  Releasing the lease while that child still owns
    the GPU would be worse than waiting.  Deadline and resource gates are
    therefore checked before the *next* launch, never inside this wait.
    ``timeout_at`` is retained for report/API compatibility; the worker owns
    the actual render timeout and must reach a durable terminal state.
    """
    del deadline, timeout_at, now_fn
    latest = status_fn(ep_id)
    def target_is_active(snapshot: Mapping[str, Any]) -> bool:
        target = next((job for job in snapshot.get("jobs") or [] if str(job.get("job_id")) == job_id), None)
        return bool(target and str(target.get("status")) in ACTIVE)

    while target_is_active(latest):
        lease.heartbeat()
        sleep_fn(policy.poll_seconds)
        latest = status_fn(ep_id)
    return latest


def run_overnight_production(
    ep_ids: Iterable[str], *, policy: OvernightPolicy | None = None,
    stop_at: datetime | None = None,
    status_fn: Callable[[str], JsonDict] = project_snapshot,
    start_fn: Callable[..., JsonDict], resume_fn: Callable[..., JsonDict],
    retry_fn: Callable[[str, str], JsonDict],
    reconcile_fn: Callable[[str, str], JsonDict] | None = None,
    http_json_fn: Callable[[str, float], JsonDict] = _http_json,
    gpu_probe_fn: Callable[[], GpuSnapshot] = probe_nvidia_gpu,
    worker_info_fn: Callable[[str], Mapping[str, Any] | None] | None = None,
    active_workers_fn: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = _utc_now,
    reports_dir: str | Path | None = None,
) -> JsonDict:
    """Run approved episodes serially under one GPU lease and fixed budgets."""
    policy = policy or OvernightPolicy()
    policy.validate()
    ep_ids = tuple(dict.fromkeys(str(value).strip() for value in ep_ids if str(value).strip()))
    if not ep_ids:
        raise ValueError("at least one ep_id is required")
    started = now_fn()
    requested_stop = stop_at or (started + timedelta(hours=policy.max_runtime_hours))
    if requested_stop.tzinfo is None:
        requested_stop = requested_stop.replace(tzinfo=timezone.utc)
    deadline = min(requested_stop, started + timedelta(hours=policy.max_runtime_hours))
    run_id = f"night-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    reports_root = Path(reports_dir or (project_root() / "logs" / "night_runs")).resolve()
    report_json = reports_root / f"{run_id}.report.json"
    report_md = reports_root / f"{run_id}.report.md"
    lease = SingleGpuLease(state_dir() / "overnight-gpu.lock", run_id, now_fn=now_fn)
    report: JsonDict = {
        "schema": "ai-manga-night-run-report/v1", "run_id": run_id, "status": "preflight",
        "started_at": _iso(started), "stop_at": _iso(deadline), "policy": asdict(policy),
        "episodes": [], "shots_started": 0, "retry_operations": 0, "dead_letters": [],
        "production_manifest_path": None, "automatic_export_attempted": False,
    }
    try:
        checks = preflight(
            ep_ids, policy, stop_at=deadline, status_fn=status_fn, http_json_fn=http_json_fn,
            gpu_probe_fn=gpu_probe_fn, worker_info_fn=worker_info_fn, now_fn=now_fn,
            active_workers_fn=active_workers_fn,
        )
        report["preflight"] = checks
        if not checks["passed"]:
            report["status"] = "preflight_failed"
            return report
        if not lease.acquire():
            checks["passed"] = False
            checks["failures"].append("single GPU overnight lease is already held")
            report["status"] = "preflight_failed"
            return report
        manifest = {
            "schema": "ai-manga-production-run/v1", "run_id": run_id,
            "frozen_at": _iso(now_fn()), "stop_at": _iso(deadline), "policy": asdict(policy),
            "environment": {
                "project_root": str(project_root()), "projects_dir": str(projects_dir()),
                "comfyui_root": str(comfyui_root()), "comfyui_server": comfyui_server(),
            },
            "hardware": checks["gpu"], "comfy": checks["comfy"], "episodes": checks["episodes"],
        }
        manifest["content_sha256"] = _sha256_json(manifest)
        manifest_path = reports_root / f"{run_id}.production-run.json"
        reports_root.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, default=str)
        report["production_manifest_path"] = str(manifest_path)
        report["production_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        report["status"] = "running"

        def resource_guard() -> tuple[bool, str]:
            if now_fn() >= deadline:
                return False, "stop_window_reached"
            disk_free = shutil.disk_usage(projects_dir()).free / (1024 ** 3)
            if disk_free < policy.minimum_free_disk_gb:
                return False, "disk_threshold_reached"
            gpu = gpu_probe_fn()
            if not gpu.available:
                return (not policy.require_gpu_probe, "gpu_probe_unavailable")
            if gpu.temperature_c is None or gpu.temperature_c >= policy.maximum_gpu_temperature_c:
                return False, "gpu_temperature_threshold_reached"
            if gpu.memory_free_mb is None or gpu.memory_free_mb < policy.minimum_free_vram_mb:
                return False, "gpu_vram_threshold_reached"
            return True, "ok"

        for ep_id in ep_ids:
            episode_result: JsonDict = {
                "ep_id": ep_id, "status": "pending", "shots_started": 0,
                "retry_operations": 0, "content_qa_blockers": [], "release_blockers": [],
                "reconciliation_events": [], "export_eligible": False,
            }
            report["episodes"].append(episode_result)
            failures_seen: set[str] = set()
            while True:
                latest = status_fn(ep_id)
                content_blockers = [
                    blocker for job in latest.get("jobs") or []
                    if (blocker := _job_content_blocker(job)) is not None
                ]
                if content_blockers:
                    episode_result["content_qa_blockers"] = content_blockers
                    episode_result["release_blockers"] = _release_blockers(latest)
                    for blocker in content_blockers:
                        report["dead_letters"].append({
                            "ep_id": ep_id, **blocker,
                            "reason": f"content_gate:{blocker['reason']}",
                            "recorded_at": _iso(now_fn()),
                        })
                    episode_result.update(
                        status="dead_letter", reason="technical_success_blocked_by_content_qa",
                    )
                    break
                failed = [job for job in latest.get("jobs") or [] if str(job.get("status")) == "failed"]
                failures_seen.update(str(job.get("job_id")) for job in failed)
                unfinished = _remaining_jobs(latest)
                if not unfinished:
                    release_blockers = _release_blockers(latest)
                    episode_result["release_blockers"] = release_blockers
                    episode_result["export_eligible"] = not release_blockers
                    episode_result["status"] = "content_ready_release_pending" if release_blockers else "release_approved"
                    break
                non_runnable = [
                    job for job in unfinished
                    if str(job.get("status")) not in {"queued", "pending", "failed", "submitted", "running"}
                ]
                if non_runnable:
                    target = sorted(non_runnable, key=lambda job: int(job.get("panel_index") or 0))[0]
                    dead = {
                        "ep_id": ep_id, "job_id": target.get("job_id"), "panel_index": target.get("panel_index"),
                        "retry_count": int(target.get("retry_count") or 0), "error": target.get("error"),
                        "reason": f"non_runnable_status:{target.get('status')}", "recorded_at": _iso(now_fn()),
                    }
                    report["dead_letters"].append(dead)
                    episode_result.update(status="dead_letter", reason=dead["reason"])
                    break
                resume_remote = False
                if failed:
                    target = sorted(failed, key=lambda job: int(job.get("panel_index") or 0))[0]
                    prompt_id = str(target.get("prompt_id") or "").strip()
                    if prompt_id:
                        if reconcile_fn is None:
                            reconciliation = {
                                "disposition": "submission_unknown",
                                "reason": "public_reconciliation_facade_unavailable",
                                "prompt_id": prompt_id,
                            }
                        else:
                            try:
                                reconciliation = dict(reconcile_fn(ep_id, str(target["job_id"])))
                            except Exception as exc:
                                reconciliation = {
                                    "disposition": "submission_unknown",
                                    "reason": f"reconciliation_failed:{type(exc).__name__}:{exc}",
                                    "prompt_id": prompt_id,
                                }
                        event = {
                            "job_id": str(target.get("job_id") or ""),
                            "panel_index": target.get("panel_index"),
                            "prompt_id": prompt_id,
                            "disposition": str(reconciliation.get("disposition") or "submission_unknown"),
                            "reason": str(reconciliation.get("reason") or ""),
                            "recorded_at": _iso(now_fn()),
                        }
                        episode_result["reconciliation_events"].append(event)
                        disposition = event["disposition"]
                        if disposition == "recovered":
                            # Re-read durable state and run the existing content
                            # gate before considering another shot.
                            continue
                        if disposition == "remote_active":
                            # This only launches a waiter/recovery worker for an
                            # already billable prompt. It must not consume retry
                            # or new-shot budget and must not submit a new graph.
                            resume_remote = True
                        elif disposition != "safe_to_retry":
                            dead = {
                                "ep_id": ep_id, "job_id": target.get("job_id"),
                                "panel_index": target.get("panel_index"),
                                "retry_count": int(target.get("retry_count") or 0),
                                "error": target.get("error"),
                                "prompt_id": prompt_id,
                                "reason": f"remote_reconciliation:{disposition}",
                                "reconciliation_reason": event["reason"],
                                "recorded_at": _iso(now_fn()),
                            }
                            report["dead_letters"].append(dead)
                            episode_result.update(status="dead_letter", reason=dead["reason"])
                            break
                    retry_count = int(target.get("retry_count") or 0)
                    retry_limit = min(policy.max_retries_per_job, int(target.get("max_retries") or policy.max_retries_per_job))
                    budget_exhausted = (
                        len(failures_seen) > policy.max_failures
                        or report["retry_operations"] >= policy.max_total_retries
                        or retry_count >= retry_limit
                    )
                    if budget_exhausted and not resume_remote:
                        dead = {
                            "ep_id": ep_id, "job_id": target.get("job_id"), "panel_index": target.get("panel_index"),
                            "retry_count": retry_count, "error": target.get("error"),
                            "reason": "retry_or_failure_budget_exhausted", "recorded_at": _iso(now_fn()),
                        }
                        report["dead_letters"].append(dead)
                        episode_result.update(status="dead_letter", reason=dead["reason"])
                        break
                safe, reason = (True, "already_remote_active") if resume_remote else resource_guard()
                if (
                    not safe
                    or (
                        not resume_remote
                        and now_fn() + timedelta(minutes=policy.minimum_start_window_minutes) >= deadline
                    )
                ):
                    episode_result.update(status="stopped", reason=reason if not safe else "insufficient_start_window")
                    report["status"] = "window_or_resource_stop"
                    break
                if failed and not resume_remote:
                    retry_fn(ep_id, str(target["job_id"]))
                    report["retry_operations"] += 1
                    episode_result["retry_operations"] += 1
                    # Public resume records the durable queue transition; the
                    # public start facade remains the only launch operation.
                    resume_fn(ep_id, statuses=("pending",))
                    launch_job_id = str(target["job_id"])
                elif resume_remote:
                    launch_job_id = str(target["job_id"])
                else:
                    if report["shots_started"] >= policy.max_shots:
                        episode_result.update(status="stopped", reason="nightly_shot_budget")
                        report["status"] = "budget_stop"
                        break
                    report["shots_started"] += 1
                    episode_result["shots_started"] += 1
                    launch_job_id = str(sorted(
                        (job for job in unfinished if str(job.get("status")) in {"queued", "pending"}),
                        key=lambda job: int(job.get("panel_index") or 0),
                    )[0]["job_id"])
                launch = start_fn(
                    ep_id, statuses=("pending",), ensure_character_assets=False, max_jobs=1,
                    timeout=policy.worker_timeout_minutes * 60.0,
                )
                if not launch.get("started"):
                    episode_result.update(status="stopped", reason=launch.get("reason") or "retry_worker_start_failed")
                    break
                _wait_for_worker(
                    ep_id, launch_job_id, status_fn=status_fn, deadline=deadline,
                    timeout_at=now_fn() + timedelta(minutes=policy.worker_timeout_minutes),
                    policy=policy, lease=lease, sleep_fn=sleep_fn, now_fn=now_fn,
                )
            if episode_result["status"] not in {"content_ready_release_pending", "release_approved"}:
                if report["status"] == "running":
                    report["status"] = "completed_with_failures"
                # Fail closed: never start a later episode after one episode
                # reaches dead-letter or is stopped.
                break
        if report["status"] == "running":
            release_pending = any(
                episode.get("status") == "content_ready_release_pending"
                for episode in report["episodes"]
            )
            report["status"] = "completed_release_pending" if release_pending else "succeeded"
        return report
    except Exception as exc:
        report["status"] = "controller_failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report
    finally:
        report["finished_at"] = _iso(now_fn())
        lease.release()
        report["report_json_path"] = str(report_json)
        report["report_markdown_path"] = str(report_md)
        _atomic_json(report_json, report)
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(_report_markdown(report), encoding="utf-8")


__all__ = [
    "GpuSnapshot", "OvernightPolicy", "SingleGpuLease", "preflight",
    "probe_nvidia_gpu", "run_overnight_production",
]
