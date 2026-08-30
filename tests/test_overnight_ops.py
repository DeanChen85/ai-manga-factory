from __future__ import annotations

import json
import sys
import tempfile
import unittest
import time
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import overnight_ops
import task_store


NOW = datetime(2026, 8, 13, 14, 0, tzinfo=timezone.utc)


def good_gpu(**changes):
    values = {
        "available": True, "name": "RTX 3090", "temperature_c": 52.0,
        "memory_total_mb": 24576, "memory_used_mb": 2048,
        "memory_free_mb": 22528, "driver_version": "test-driver", "error": None,
    }
    values.update(changes)
    return overnight_ops.GpuSnapshot(**values)


def http_ok(url, _timeout):
    if url.endswith("/object_info"):
        return {
            name: {"display_name": name, "python_module": "test.nodes"}
            for name in (
                *overnight_ops.OvernightPolicy().required_nodes,
                *overnight_ops.OvernightPolicy().recommended_nodes,
            )
        }
    if url.endswith("/queue"):
        return {"queue_running": [], "queue_pending": []}
    return {"system": {"comfyui_version": "0.test"}}


def job(status="queued", retry_count=0, error=None, *, qa="auto", prompt_id=None):
    output = "X:/projects/ep1/videos/p1.mp4"
    metadata = {
        "settings": {"steps": 8},
        "inputs": {"asset_dependencies": [{"asset_id": "char-a", "content_hash": "asset-sha"}]},
    }
    if status == "succeeded" and qa == "auto":
        metadata.update({
            "artifact_sha256": "artifact-sha",
            "content_qa": {"passed": True, "status": "passed", "analysis": {
                "decoded_visual_sha256": "visual-sha", "source_path": output,
            }},
        })
    elif qa == "failed":
        metadata["content_qa"] = {
            "passed": False, "status": "failed", "reasons": ["static_visual", "near_duplicate"],
            "analysis": {"decoded_visual_sha256": "bad-visual", "source_path": output},
        }
    return {
        "job_id": "ep1:0001:p1", "panel_index": 1, "panel_name": "p1",
        "status": status, "input_hash": "input-sha", "retry_count": retry_count,
        "max_retries": 2, "error": error, "output_path": output, "metadata": metadata,
        "prompt_id": prompt_id,
    }


def snapshot(current_job):
    return {
        "ep_id": "ep1", "project_dir": "X:/projects/ep1",
        "jobs": [dict(current_job)], "counts": {"total": 1},
    }


class OvernightOpsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path_patches = [
            mock.patch.object(overnight_ops, "project_root", return_value=self.root),
            mock.patch.object(overnight_ops, "projects_dir", return_value=self.root),
            mock.patch.object(overnight_ops, "state_dir", return_value=self.root / "state"),
            mock.patch.object(overnight_ops, "comfyui_root", return_value=self.root / "ComfyUI"),
            mock.patch.object(overnight_ops, "comfyui_server", return_value="http://comfy.test"),
        ]
        for patcher in self.path_patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.path_patches):
            patcher.stop()
        self.temp.cleanup()

    def policy(self, **changes):
        values = overnight_ops.OvernightPolicy().__dict__.copy()
        values.update({
            "minimum_free_disk_gb": 0.001, "poll_seconds": 0,
            "minimum_start_window_minutes": 0,
        })
        values.update(changes)
        return overnight_ops.OvernightPolicy(**values)

    def test_preflight_fails_closed_without_gpu_probe_and_reports_reason(self):
        result = overnight_ops.preflight(
            ["ep1"], self.policy(require_gpu_probe=True), stop_at=NOW + timedelta(hours=1),
            status_fn=lambda _ep: snapshot(job()), http_json_fn=http_ok,
            gpu_probe_fn=lambda: overnight_ops.GpuSnapshot(False, error="nvidia-smi missing"),
            worker_info_fn=lambda _ep: None, now_fn=lambda: NOW,
        )
        self.assertFalse(result["passed"])
        self.assertIn("nvidia-smi missing", result["failures"])
        self.assertEqual(result["comfy"]["nodes"]["missing_required"], [])

    def test_preflight_warns_when_latest_hard_guide_capability_is_missing(self):
        def without_recommended(url, timeout):
            value = http_ok(url, timeout)
            if url.endswith("/object_info"):
                value.pop("MiniMaxH3AddGuide", None)
            return value

        result = overnight_ops.preflight(
            ["ep1"], self.policy(), stop_at=NOW + timedelta(hours=1),
            status_fn=lambda _ep: snapshot(job()), http_json_fn=without_recommended,
            gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            active_workers_fn=lambda: [], now_fn=lambda: NOW,
        )
        self.assertTrue(result["passed"])
        self.assertIn("hard guides", " | ".join(result["warnings"]))

    def test_preflight_blocks_worker_owned_by_another_project(self):
        result = overnight_ops.preflight(
            ["ep1"], self.policy(), stop_at=NOW + timedelta(hours=1),
            status_fn=lambda _ep: snapshot(job()), http_json_fn=http_ok,
            gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            active_workers_fn=lambda: [{
                "ep_id": "another_episode", "owner": "other-host:9000",
                "pid": 9000, "heartbeat": 123.0, "active": True,
            }],
            now_fn=lambda: NOW,
        )
        self.assertFalse(result["passed"])
        self.assertIn("production workers", " | ".join(result["failures"]))
        self.assertEqual(result["active_workers"][0]["ep_id"], "another_episode")

    def test_preflight_blocks_queue_worker_disk_temperature_vram_and_budget(self):
        def busy_http(url, timeout):
            value = http_ok(url, timeout)
            if url.endswith("/queue"):
                return {"queue_running": [[1]], "queue_pending": []}
            return value

        with mock.patch.object(overnight_ops.shutil, "disk_usage") as disk:
            DiskUsage = namedtuple("DiskUsage", "total used free")
            disk.return_value = DiskUsage(100, 99, 1)
            result = overnight_ops.preflight(
                ["ep1"], self.policy(max_shots=1, minimum_free_disk_gb=1),
                stop_at=NOW + timedelta(hours=1),
                status_fn=lambda _ep: {**snapshot(job()), "jobs": [job(), {**job(), "job_id": "ep1:0002:p2"}]},
                http_json_fn=busy_http, gpu_probe_fn=lambda: good_gpu(temperature_c=90, memory_free_mb=100),
                worker_info_fn=lambda _ep: {"active": True}, now_fn=lambda: NOW,
            )
        joined = " | ".join(result["failures"])
        for expected in ("already has", "already has an active worker", "disk free", "temperature", "VRAM"):
            self.assertIn(expected, joined)
        self.assertIn("exceed nightly", " | ".join(result["warnings"]))

    def test_success_freezes_manifest_before_public_worker_start_and_writes_reports(self):
        state = {"job": job(), "manifest_seen": False}
        calls = []

        def status_fn(_ep):
            return snapshot(state["job"])

        reports = self.root / "reports"

        def start_fn(ep_id, **kwargs):
            manifests = list(reports.glob("*.production-run.json"))
            self.assertEqual(len(manifests), 1, "production manifest must exist before start")
            state["manifest_seen"] = True
            state["job"] = job("succeeded")
            calls.append(("start", ep_id, kwargs))
            return {"started": True, "pid": 123}

        result = overnight_ops.run_overnight_production(
            ["ep1"], policy=self.policy(), stop_at=NOW + timedelta(hours=1),
            status_fn=status_fn, start_fn=start_fn,
            resume_fn=lambda *args, **kwargs: calls.append(("resume", args, kwargs)) or {},
            retry_fn=lambda *args: calls.append(("retry", args)) or {},
            http_json_fn=http_ok, gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            sleep_fn=lambda _seconds: None, now_fn=lambda: NOW, reports_dir=reports,
        )
        self.assertEqual(result["status"], "completed_release_pending")
        self.assertTrue(state["manifest_seen"])
        self.assertEqual(result["shots_started"], 1)
        manifest = json.loads(Path(result["production_manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["episodes"][0]["jobs"][0]["input_hash"], "input-sha")
        self.assertEqual(manifest["episodes"][0]["jobs"][0]["asset_dependencies"][0]["content_hash"], "asset-sha")
        written = json.loads(Path(result["report_json_path"]).read_text(encoding="utf-8"))
        self.assertEqual(written["status"], "completed_release_pending")
        self.assertFalse(written["automatic_export_attempted"])
        self.assertIn("episode_release_not_approved", json.dumps(written["episodes"][0]["release_blockers"]))
        self.assertIn("Preflight: PASS", Path(result["report_markdown_path"]).read_text(encoding="utf-8"))
        self.assertFalse((self.root / "state" / "overnight-gpu.lock").exists())

    def test_failed_job_retries_twice_only_then_dead_letters_and_stops_later_episode(self):
        state = {"job": job(), "starts": 0}
        retried = []

        def status_fn(ep_id):
            current = dict(state["job"])
            current["job_id"] = f"{ep_id}:0001:p1"
            return {**snapshot(current), "ep_id": ep_id, "jobs": [current]}

        def start_fn(ep_id, **_kwargs):
            state["starts"] += 1
            # Initial failure, first retry failure, second retry failure.
            state["job"] = job("failed", min(state["starts"] - 1, 2), "render failed")
            return {"started": True}

        def retry_fn(ep_id, job_id):
            retried.append((ep_id, job_id))
            state["job"] = job("queued", len(retried))
            return state["job"]

        result = overnight_ops.run_overnight_production(
            ["ep1", "ep2"], policy=self.policy(max_shots=2, max_total_retries=2, max_retries_per_job=2),
            stop_at=NOW + timedelta(hours=1), status_fn=status_fn, start_fn=start_fn,
            resume_fn=lambda *_args, **_kwargs: {}, retry_fn=retry_fn,
            http_json_fn=http_ok, gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            sleep_fn=lambda _seconds: None, now_fn=lambda: NOW, reports_dir=self.root / "reports",
        )
        self.assertEqual(len(retried), 2)
        self.assertEqual(state["starts"], 3)
        self.assertEqual(len(result["dead_letters"]), 1)
        self.assertEqual(result["episodes"][0]["status"], "dead_letter")
        self.assertEqual(len(result["episodes"]), 1, "later episode must not start after dead letter")

    def test_failed_remote_prompt_recovers_before_retry_and_never_submits_again(self):
        state = {"job": job("failed", error="timeout", prompt_id="prompt-existing")}
        retries = []
        starts = []

        def reconcile(_ep_id, _job_id):
            state["job"] = job("succeeded", prompt_id="prompt-existing")
            return {"disposition": "recovered", "reason": "comfy_history_success"}

        result = overnight_ops.run_overnight_production(
            ["ep1"], policy=self.policy(), stop_at=NOW + timedelta(hours=1),
            status_fn=lambda _ep: {**snapshot(state["job"]), "pipeline": {"release_status": "pending"}},
            start_fn=lambda *args, **kwargs: starts.append((args, kwargs)) or {"started": True},
            resume_fn=lambda *_a, **_k: {},
            retry_fn=lambda *args: retries.append(args) or {}, reconcile_fn=reconcile,
            http_json_fn=http_ok, gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            sleep_fn=lambda _seconds: None, now_fn=lambda: NOW,
            reports_dir=self.root / "recovered-report",
        )
        self.assertEqual(result["status"], "completed_release_pending")
        self.assertEqual(retries, [])
        self.assertEqual(starts, [])
        self.assertEqual(result["episodes"][0]["reconciliation_events"][0]["disposition"], "recovered")

    def test_remote_active_launches_waiter_without_retry_budget_or_new_shot(self):
        state = {"job": job("failed", error="timeout", prompt_id="prompt-existing")}
        retries = []
        starts = []

        def reconcile(_ep_id, _job_id):
            state["job"] = job("submitted", prompt_id="prompt-existing")
            return {"disposition": "remote_active", "reason": "prompt_present_in_comfy_queue"}

        def start(_ep_id, **kwargs):
            starts.append(kwargs)
            state["job"] = job("succeeded", prompt_id="prompt-existing")
            return {"started": True}

        result = overnight_ops.run_overnight_production(
            ["ep1"], policy=self.policy(), stop_at=NOW + timedelta(hours=1),
            status_fn=lambda _ep: {**snapshot(state["job"]), "pipeline": {"release_status": "pending"}},
            start_fn=start, resume_fn=lambda *_a, **_k: {},
            retry_fn=lambda *args: retries.append(args) or {}, reconcile_fn=reconcile,
            http_json_fn=http_ok, gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            sleep_fn=lambda _seconds: None, now_fn=lambda: NOW,
            reports_dir=self.root / "active-report",
        )
        self.assertEqual(result["status"], "completed_release_pending")
        self.assertEqual(retries, [])
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["max_jobs"], 1)
        self.assertEqual(result["shots_started"], 0)
        self.assertEqual(result["retry_operations"], 0)

    def test_submission_unknown_deadletters_without_retry_or_worker_start(self):
        failed = job("failed", error="timeout", prompt_id="prompt-existing")
        retries = []
        starts = []
        result = overnight_ops.run_overnight_production(
            ["ep1", "ep2"], policy=self.policy(), stop_at=NOW + timedelta(hours=1),
            status_fn=lambda _ep: snapshot(failed),
            start_fn=lambda *args, **kwargs: starts.append((args, kwargs)) or {"started": True},
            resume_fn=lambda *_a, **_k: {},
            retry_fn=lambda *args: retries.append(args) or {},
            reconcile_fn=lambda *_a: {
                "disposition": "submission_unknown", "reason": "history:missing;queue:prompt_absent",
            },
            http_json_fn=http_ok, gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            sleep_fn=lambda _seconds: None, now_fn=lambda: NOW,
            reports_dir=self.root / "unknown-report",
        )
        self.assertEqual(result["episodes"][0]["status"], "dead_letter")
        self.assertEqual(len(result["episodes"]), 1)
        self.assertEqual(retries, [])
        self.assertEqual(starts, [])
        self.assertIn("submission_unknown", result["dead_letters"][0]["reason"])
        self.assertFalse(result["automatic_export_attempted"])

    def test_explicit_safe_to_retry_is_the_only_remote_path_that_calls_retry(self):
        state = {"job": job("failed", error="Comfy error", prompt_id="prompt-existing")}
        retries = []

        def retry(_ep_id, _job_id):
            retries.append(_job_id)
            state["job"] = job("queued", retry_count=1)
            return state["job"]

        def start(_ep_id, **_kwargs):
            state["job"] = job("succeeded", retry_count=1)
            return {"started": True}

        result = overnight_ops.run_overnight_production(
            ["ep1"], policy=self.policy(), stop_at=NOW + timedelta(hours=1),
            status_fn=lambda _ep: {**snapshot(state["job"]), "pipeline": {"release_status": "pending"}},
            start_fn=start, resume_fn=lambda *_a, **_k: {}, retry_fn=retry,
            reconcile_fn=lambda *_a: {
                "disposition": "safe_to_retry", "reason": "explicit_comfy_history_error",
            },
            http_json_fn=http_ok, gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            sleep_fn=lambda _seconds: None, now_fn=lambda: NOW,
            reports_dir=self.root / "safe-retry-report",
        )
        self.assertEqual(len(retries), 1)
        self.assertEqual(result["retry_operations"], 1)
        self.assertEqual(result["status"], "completed_release_pending")

    def test_single_gpu_lease_is_atomic_and_owner_safe(self):
        path = self.root / "gpu.lock"
        first = overnight_ops.SingleGpuLease(path, "run-a", now_fn=lambda: NOW)
        second = overnight_ops.SingleGpuLease(path, "run-b", now_fn=lambda: NOW)
        self.assertTrue(first.acquire())
        self.assertFalse(second.acquire())
        second.release()
        self.assertTrue(path.exists(), "non-owner must not release the lease")
        first.release()
        self.assertFalse(path.exists())

    def test_running_worker_is_not_interrupted_by_low_vram_and_next_start_is_blocked(self):
        state = {"reads": 0, "starts": 0}
        low_probe = {"enabled": False}

        def status_fn(_ep):
            state["reads"] += 1
            if state["starts"] == 0:
                return {**snapshot(job("queued")), "jobs": [
                    job("queued"), {**job("queued"), "job_id": "ep1:0002:p2", "panel_index": 2},
                ]}
            if state["reads"] < 4:
                low_probe["enabled"] = True
                return {**snapshot(job("running")), "jobs": [
                    job("running"), {**job("queued"), "job_id": "ep1:0002:p2", "panel_index": 2},
                ]}
            return {**snapshot(job()), "jobs": [job("succeeded"), {**job(), "job_id": "ep1:0002:p2", "panel_index": 2}]}

        def start_fn(_ep, **kwargs):
            self.assertEqual(kwargs["max_jobs"], 1)
            state["starts"] += 1
            return {"started": True}

        def gpu():
            return good_gpu(memory_free_mb=100 if low_probe["enabled"] else 22000)

        result = overnight_ops.run_overnight_production(
            ["ep1"], policy=self.policy(max_shots=2), stop_at=NOW + timedelta(hours=1),
            status_fn=status_fn, start_fn=start_fn, resume_fn=lambda *_a, **_k: {}, retry_fn=lambda *_a: {},
            http_json_fn=http_ok, gpu_probe_fn=gpu, worker_info_fn=lambda _ep: None,
            sleep_fn=lambda _seconds: None, now_fn=lambda: NOW, reports_dir=self.root / "reports",
        )
        self.assertEqual(state["starts"], 1, "low VRAM during render must not release/interrupt or start the next shot")
        self.assertEqual(result["episodes"][0]["reason"], "gpu_vram_threshold_reached")

    def test_technical_success_with_content_qa_failure_deadletters_and_never_starts_next_shot(self):
        for failed_position in (1, 2):
            with self.subTest(failed_position=failed_position):
                first = job("succeeded", qa="failed" if failed_position == 1 else "auto")
                second = {**job("succeeded" if failed_position == 2 else "queued", qa="failed" if failed_position == 2 else "auto"),
                          "job_id": "ep1:0002:p2", "panel_index": 2, "panel_name": "p2",
                          "output_path": "X:/projects/ep1/videos/p2.mp4"}
                if failed_position == 2:
                    second["metadata"] = {**second["metadata"], "content_qa": {
                        **second["metadata"]["content_qa"],
                        "analysis": {**second["metadata"]["content_qa"]["analysis"], "source_path": second["output_path"]},
                    }}
                starts = []
                reports = self.root / f"reports-{failed_position}"
                result = overnight_ops.run_overnight_production(
                    ["ep1"], policy=self.policy(max_shots=2), stop_at=NOW + timedelta(hours=1),
                    status_fn=lambda _ep, jobs=(first, second): {
                        "ep_id": "ep1", "project_dir": "X:/projects/ep1", "jobs": [dict(item) for item in jobs],
                        "pipeline": {"release_status": "pending"},
                    },
                    start_fn=lambda *args, **kwargs: starts.append((args, kwargs)) or {"started": True},
                    resume_fn=lambda *_a, **_k: {}, retry_fn=lambda *_a: {},
                    http_json_fn=http_ok, gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
                    sleep_fn=lambda _seconds: None, now_fn=lambda: NOW, reports_dir=reports,
                )
                self.assertEqual(result["episodes"][0]["status"], "dead_letter")
                self.assertEqual(result["episodes"][0]["reason"], "technical_success_blocked_by_content_qa")
                self.assertEqual(starts, [], "content failure must stop before any next-shot launch")
                self.assertFalse(result["automatic_export_attempted"])
                report_text = Path(result["report_markdown_path"]).read_text(encoding="utf-8")
                self.assertIn("Content QA blocker", report_text)
                self.assertIn("static_visual", report_text)
                self.assertIn("Release blocker", report_text)

    def test_content_ready_report_lists_human_and_release_blockers_without_export(self):
        completed = job("succeeded")
        snap = {
            **snapshot(completed), "pipeline": {"release_status": "pending"},
        }
        result = overnight_ops.run_overnight_production(
            ["ep1"], policy=self.policy(), stop_at=NOW + timedelta(hours=1),
            status_fn=lambda _ep: snap,
            start_fn=lambda *_a, **_k: self.fail("completed episode must not launch another worker"),
            resume_fn=lambda *_a, **_k: {}, retry_fn=lambda *_a: {},
            http_json_fn=http_ok, gpu_probe_fn=good_gpu, worker_info_fn=lambda _ep: None,
            sleep_fn=lambda _seconds: None, now_fn=lambda: NOW, reports_dir=self.root / "release-report",
        )
        self.assertEqual(result["status"], "completed_release_pending")
        self.assertFalse(result["episodes"][0]["export_eligible"])
        reasons = json.dumps(result["episodes"][0]["release_blockers"])
        self.assertIn("edit_selection_missing_or_stale", reasons)
        self.assertIn("editorial_review_missing_or_stale", reasons)
        self.assertIn("release_missing_or_stale", reasons)
        self.assertFalse(result["automatic_export_attempted"])

    def test_render_service_exports_public_overnight_facade(self):
        import render_service

        self.assertTrue(callable(render_service.run_overnight))
        self.assertTrue(callable(render_service.reconcile_job))
        self.assertIn("run_overnight", render_service.__all__)
        self.assertIn("reconcile_job", render_service.__all__)

    def test_active_workers_reports_only_live_rows_without_mutation(self):
        database = self.root / "workers.sqlite3"
        store = task_store.RenderJobStore(database)
        with store._connection() as connection:
            connection.execute(
                "INSERT INTO workers(ep_id,owner,pid,heartbeat) VALUES(?,?,?,?)",
                ("live", "host:1", 1, time.time()),
            )
            connection.execute(
                "INSERT INTO workers(ep_id,owner,pid,heartbeat) VALUES(?,?,?,?)",
                ("stale", "host:2", 2, time.time() - 500),
            )
        self.assertEqual([item["ep_id"] for item in store.active_workers(stale_after=120)], ["live"])
        with store._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM workers").fetchone()[0]
        self.assertEqual(count, 2, "read-only global worker probe must not delete stale rows")


if __name__ == "__main__":
    unittest.main()
