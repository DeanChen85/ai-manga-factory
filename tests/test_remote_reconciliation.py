from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import render_service
import render_video_h3
import task_store


class RemoteReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = task_store.RenderJobStore(self.root / "jobs.sqlite3")
        self.ep_id = "ep_reconcile"
        self.job_id = f"{self.ep_id}:0001:p1"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def register_failed(self, *, prompt_id: str = "prompt-existing", error: str = "render timed out") -> dict:
        return self.store.register_jobs(self.ep_id, [{
            "job_id": self.job_id,
            "panel_index": 1,
            "panel_name": "p1",
            "status": "failed",
            "prompt_id": prompt_id,
            "output_path": str(self.root / "p1.mp4"),
            "preview_path": str(self.root / "p1.mp4"),
            "input_hash": "input-sha",
            "retry_count": 0,
            "max_retries": 2,
            "error": error,
            "metadata": {"inputs": {"reference_inputs": []}},
        }])[0]

    def test_history_success_recovers_without_queue_or_new_submission(self) -> None:
        self.register_failed()
        calls: list[str] = []

        def api(path, _payload):
            calls.append(path)
            return {
                "prompt-existing": {
                    "status": {"status_str": "success", "completed": True},
                    "outputs": {"video": {}},
                }
            }

        def complete(job, _entry, store, _probe, **_runtime):
            output = Path(job["output_path"])
            output.write_bytes(b"accepted-existing-output")
            store.update_job(
                job["job_id"], status="succeeded", output_path=str(output),
                preview_path=str(output), error=None,
            )
            return output

        with mock.patch.object(render_video_h3, "_complete_job_from_history", side_effect=complete):
            result = render_video_h3.reconcile_render_job(
                self.job_id, store=self.store, api_func=api,
            )

        self.assertEqual(result["disposition"], "recovered")
        self.assertEqual(calls, ["/history/prompt-existing"])
        self.assertEqual(self.store.get_job(self.job_id)["status"], "succeeded")
        self.assertEqual(self.store.get_job(self.job_id)["prompt_id"], "prompt-existing")

    def test_active_queue_preserves_prompt_and_does_not_authorize_retry(self) -> None:
        self.register_failed()

        def api(path, _payload):
            if path.startswith("/history/"):
                return {}
            return {"queue_running": [[7, "prompt-existing", {}, {}]], "queue_pending": []}

        result = render_video_h3.reconcile_render_job(
            self.job_id, store=self.store, api_func=api,
        )
        current = self.store.get_job(self.job_id)
        self.assertEqual(result["disposition"], "remote_active")
        self.assertEqual(current["status"], "submitted")
        self.assertEqual(current["prompt_id"], "prompt-existing")
        self.assertNotIn("remote_retry_authorization", current["metadata"])

    def test_explicit_history_error_is_only_path_that_authorizes_retry(self) -> None:
        self.register_failed()

        result = render_video_h3.reconcile_render_job(
            self.job_id,
            store=self.store,
            api_func=lambda path, _payload: {
                "prompt-existing": {
                    "status": {"status_str": "error", "messages": ["model execution failed"]}
                }
            },
        )
        failed = self.store.get_job(self.job_id)
        self.assertEqual(result["disposition"], "safe_to_retry")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(
            failed["metadata"]["remote_retry_authorization"]["prompt_id"],
            "prompt-existing",
        )

        with mock.patch.object(task_store, "default_store", return_value=self.store):
            retried = task_store.retry_job(self.ep_id, self.job_id)
        self.assertEqual(retried["status"], "queued")
        self.assertIsNone(retried["prompt_id"])
        self.assertEqual(retried["retry_count"], 1)
        self.assertNotIn("remote_retry_authorization", retried["metadata"])
        self.assertEqual(
            retried["metadata"]["remote_reconciliation_history"][-1]["prompt_id"],
            "prompt-existing",
        )

    def test_network_failure_or_missing_history_is_submission_unknown(self) -> None:
        for mode in ("network", "absent"):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    store = task_store.RenderJobStore(root / "jobs.sqlite3")
                    store.register_jobs(self.ep_id, [{
                        "job_id": self.job_id, "panel_index": 1, "panel_name": "p1",
                        "status": "failed", "prompt_id": "prompt-existing",
                        "output_path": str(root / "p1.mp4"), "input_hash": "input-sha",
                        "retry_count": 0, "max_retries": 2, "error": "render timed out",
                        "metadata": {},
                    }])

                    def api(path, _payload):
                        if mode == "network":
                            raise OSError("Comfy unavailable")
                        return {} if path.startswith("/history/") else {
                            "queue_running": [], "queue_pending": [],
                        }

                    result = render_video_h3.reconcile_render_job(
                        self.job_id, store=store, api_func=api,
                    )
                    current = store.get_job(self.job_id)
                    self.assertEqual(result["disposition"], "submission_unknown")
                    self.assertEqual(current["status"], "submitted")
                    self.assertEqual(current["prompt_id"], "prompt-existing")
                    self.assertNotIn("remote_retry_authorization", current["metadata"])

    def test_retry_rejects_unreconciled_prompt_while_resume_preserves_it_for_recovery(self) -> None:
        self.register_failed()
        with mock.patch.object(task_store, "default_store", return_value=self.store):
            with self.assertRaisesRegex(RuntimeError, "must be reconciled"):
                task_store.retry_job(self.ep_id, self.job_id)
            summary = task_store.resume_jobs(self.ep_id, statuses=("failed",))
        current = self.store.get_job(self.job_id)
        self.assertEqual(summary["resumed"], 1)
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["prompt_id"], "prompt-existing")
        self.assertEqual(current["retry_count"], 1)

    def test_confirmed_comfy_restart_authorizes_one_bounded_retry_after_history_loss(self) -> None:
        self.register_failed()
        self.store.update_job(self.job_id, retry_count=2, max_retries=2)

        def api(path, _payload):
            if path.startswith("/history/"):
                return {}
            return {"queue_running": [], "queue_pending": []}

        with self.assertRaisesRegex(RuntimeError, "confirmation is required"):
            render_video_h3.authorize_retry_after_comfy_restart(
                self.job_id, confirmed=False, store=self.store, api_func=api,
            )
        authorized = render_video_h3.authorize_retry_after_comfy_restart(
            self.job_id, confirmed=True, store=self.store, api_func=api,
        )
        self.assertEqual(authorized["disposition"], "safe_to_retry")
        current = self.store.get_job(self.job_id)
        self.assertEqual(current["retry_count"], 1)
        self.assertEqual(
            current["metadata"]["remote_retry_authorization"]["source"],
            "operator_restart_attestation",
        )
        with mock.patch.object(task_store, "default_store", return_value=self.store):
            retried = task_store.retry_job(self.ep_id, self.job_id)
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(retried["retry_count"], 2)
        self.assertIsNone(retried["prompt_id"])

    def test_comfy_restart_recovery_refuses_nonempty_queue(self) -> None:
        self.register_failed()

        def api(path, _payload):
            if path.startswith("/history/"):
                return {}
            return {"queue_running": [[1, "another-prompt", {}, {}]], "queue_pending": []}

        with self.assertRaisesRegex(RuntimeError, "queue must be empty"):
            render_video_h3.authorize_retry_after_comfy_restart(
                self.job_id, confirmed=True, store=self.store, api_func=api,
            )

    def test_legacy_strict_predecessor_failure_restores_retry_budget_without_gpu_attempt(self) -> None:
        self.register_failed(prompt_id="", error=(
            "strict continuity predecessor is not succeeded: ep_reconcile:0000:p0"
        ))
        self.store.update_job(self.job_id, retry_count=2, max_retries=2)
        with mock.patch.object(task_store, "default_store", return_value=self.store):
            summary = task_store.resume_jobs(self.ep_id, statuses=("failed",))
        current = self.store.get_job(self.job_id)
        self.assertEqual(summary["resumed"], 1)
        self.assertEqual(current["status"], "queued")
        self.assertEqual(current["retry_count"], 0)
        self.assertIsNone(current["error"])

    def test_render_service_repairs_already_queued_legacy_dependency_blocks(self) -> None:
        self.register_failed(prompt_id="", error=(
            "strict continuity predecessor is not succeeded: ep_reconcile:0000:p0"
        ))
        self.store.update_job(self.job_id, status="queued", retry_count=2, max_retries=2)
        with mock.patch.object(render_service, "default_store", return_value=self.store):
            repaired = render_service._repair_legacy_strict_predecessor_blocks(self.ep_id)
        current = self.store.get_job(self.job_id)
        self.assertEqual(repaired, [self.job_id])
        self.assertEqual(current["retry_count"], 0)
        self.assertIsNone(current["error"])
        self.assertEqual(
            current["metadata"]["dependency_retry_budget_repair_audit"][-1]["previous_retry_count"],
            2,
        )

    def test_concurrent_reconciliation_has_one_remote_query_owner(self) -> None:
        self.register_failed()
        entered = threading.Event()
        release = threading.Event()
        call_count = 0
        call_lock = threading.Lock()

        def api(path, _payload):
            nonlocal call_count
            self.assertTrue(path.startswith("/history/"))
            with call_lock:
                call_count += 1
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return {
                "prompt-existing": {
                    "status": {"status_str": "error", "messages": ["definite failure"]}
                }
            }

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                render_video_h3.reconcile_render_job,
                self.job_id, store=self.store, api_func=api,
            )
            self.assertTrue(entered.wait(timeout=5))
            second = pool.submit(
                render_video_h3.reconcile_render_job,
                self.job_id, store=self.store, api_func=api,
            )
            second_result = second.result(timeout=5)
            release.set()
            first_result = first.result(timeout=5)

        self.assertEqual(call_count, 1)
        self.assertEqual(first_result["disposition"], "safe_to_retry")
        self.assertEqual(second_result["disposition"], "remote_active")
        self.assertEqual(self.store.get_job(self.job_id)["status"], "failed")

    def test_web_resume_reconciles_explicit_remote_error_before_bulk_queue(self) -> None:
        failed = {
            "job_id": self.job_id, "status": "failed",
            "prompt_id": "prompt-existing",
        }
        with (
            mock.patch.object(render_service, "list_jobs", return_value=[failed]),
            mock.patch.object(render_service, "default_store", return_value=self.store),
            mock.patch.object(render_service, "reconcile_render_job", return_value={
                "disposition": "safe_to_retry", "reason": "explicit_comfy_history_error",
            }) as reconcile,
            mock.patch.object(render_service, "_retry_job", return_value={"status": "queued"}) as retry,
            mock.patch.object(render_service, "_resume_jobs", return_value={
                "ep_id": self.ep_id, "resumed": 1, "job_ids": [self.job_id], "skipped": [],
            }) as bulk,
        ):
            result = render_service.resume(self.ep_id)
        reconcile.assert_called_once_with(self.job_id, store=self.store)
        retry.assert_called_once_with(self.ep_id, self.job_id)
        bulk.assert_called_once()
        self.assertEqual(result["remote_retries"], [self.job_id])

    def test_web_resume_never_retries_submission_unknown(self) -> None:
        failed = {
            "job_id": self.job_id, "status": "failed",
            "prompt_id": "prompt-existing",
        }
        with (
            mock.patch.object(render_service, "list_jobs", return_value=[failed]),
            mock.patch.object(render_service, "default_store", return_value=self.store),
            mock.patch.object(render_service, "reconcile_render_job", return_value={
                "disposition": "submission_unknown", "reason": "history_missing",
            }),
            mock.patch.object(render_service, "_retry_job") as retry,
            mock.patch.object(render_service, "_resume_jobs", return_value={
                "ep_id": self.ep_id, "resumed": 0, "job_ids": [], "skipped": [self.job_id],
            }),
        ):
            result = render_service.resume(self.ep_id)
        retry.assert_not_called()
        self.assertEqual(result["remote_retries"], [])

    def test_new_prompt_contract_renews_bounded_retry_budget_with_audit(self) -> None:
        self.register_failed(prompt_id="")
        self.store.update_job(
            self.job_id,
            retry_count=2,
            metadata={"settings": {"runtime_prompt_contract": "h3-runtime/v3-official-shape"}},
        )
        with mock.patch.object(render_service, "default_store", return_value=self.store):
            renewed = render_service._renew_retry_budget_for_prompt_contract(
                self.ep_id, self.job_id,
            )
            repeated = render_service._renew_retry_budget_for_prompt_contract(
                self.ep_id, self.job_id,
            )
        current = self.store.get_job(self.job_id, ep_id=self.ep_id)
        self.assertTrue(renewed)
        self.assertFalse(repeated)
        self.assertEqual(current["retry_count"], 0)
        audit = current["metadata"]["prompt_contract_revision_audit"][-1]
        self.assertEqual(audit["from"], "h3-runtime/v3-official-shape")
        self.assertEqual(audit["to"], render_video_h3.H3_RUNTIME_PROMPT_CONTRACT)

    def test_render_service_exposes_public_reconciliation_facade(self) -> None:
        self.assertIn("reconcile_job", render_service.__all__)
        self.assertTrue(callable(render_service.reconcile_job))
        self.assertIn("authorize_retry_after_comfy_restart", render_service.__all__)
        self.assertTrue(callable(render_service.authorize_retry_after_comfy_restart))


if __name__ == "__main__":
    unittest.main()
