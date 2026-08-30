from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(TESTS))

import orchestrator
import series_service
import series_store
import task_store
from story_splitter import generate_series_episode, split_series
from test_series_contract_v4 import raw_series, raw_v3_episode


class SeriesV4BackendIntegrationTests(unittest.TestCase):
    """Offline creative-contract -> durable season production integration."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.old_env = dict(os.environ)
        os.environ["AI_MANGA_PROJECTS_DIR"] = str(self.base / "projects")
        os.environ["AI_MANGA_JOB_DB"] = str(self.base / "state" / "jobs.sqlite3")
        os.environ["AI_MANGA_SERIES_DB"] = str(self.base / "state" / "series.sqlite3")
        os.environ["AI_FACTORY_ROOT"] = str(self.base)
        task_store._default_store = None
        series_store._default = None
        orchestrator.PROJECTS_DIR = self.base / "projects"

    def tearDown(self):
        task_store._default_store = None
        series_store._default = None
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tempdir.cleanup()

    def _file(self, name: str, content: bytes = b"asset") -> str:
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    def _fake_minimax_v4(self) -> dict:
        calls: list[str] = []

        def fake_head_writer(system: str, user: str, **_kwargs):
            calls.append(system)
            return json.dumps(raw_series(), ensure_ascii=False)

        with mock.patch("story_splitter._call_m3", side_effect=fake_head_writer):
            series = split_series(
                topic="future letter", synopsis="A courier receives a warning from the future.",
                episode_count=3, seconds_per_episode=20, shots_per_episode=5,
                target_audience="young adults", visual_style="serialized noir comic",
                style_enforcement="serialized noir comic, stable ink and amber-blue palette",
                aspect_ratio="9:16", language="cn", api_key="offline-mock",
                background_music="soft_piano", ambience="rain_night_city",
            )
        self.assertTrue(calls)
        for item in list(series["season_outline"]):
            with mock.patch(
                "story_splitter._call_m3", return_value=json.dumps(raw_v3_episode(), ensure_ascii=False),
            ):
                series = generate_series_episode(series, item["episode_id"], api_key="offline-mock")
            series["episode_approvals"][item["episode_id"]] = True
        series["season_approved"] = True
        return series

    def _prepare_assets(self, series_id: str) -> None:
        def character_generator(source, visual, **_kwargs):
            return {"reference_images": [self._file("generated/hero.png", b"hero-v4")]}

        def scene_generator(source, visual, **_kwargs):
            return {"reference_images": [self._file("generated/station.png", b"station-v4")]}

        series_service.run_prepare_shared_assets(
            series_id, character_generator=character_generator, scene_generator=scene_generator,
        )
        series_service.approve_shared_assets(series_id)

    def _complete_jobs(self, ep_id: str) -> None:
        store = task_store.default_store()
        for job in task_store.list_jobs(ep_id):
            output = Path(job["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"clip:{ep_id}:{job['job_id']}".encode())
            store.update_job(
                job["job_id"], status="succeeded", output_path=str(output), preview_path=str(output),
                probe={"duration_seconds": 10.0, "video": {"width": 1080, "height": 1920, "fps": 24.0}},
                metadata={
                    **(job.get("metadata") or {}),
                    "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                },
            )

    def test_v4_contract_facade_recovers_resumes_and_exports_exact_season(self):
        v4 = self._fake_minimax_v4()
        series_id = v4["series_bible"]["series_id"]

        draft = series_service.prepare_series_contract(series_id, v4)
        self.assertEqual(draft["series_contract_v4"], v4)
        self.assertEqual(draft["series"]["spec"]["runtime"]["v4_contract"], v4)
        series_service.approve_series(series_id, expected_hash=draft["series"]["contract_hash"])
        self._prepare_assets(series_id)
        shared_hash = series_service.status_series(series_id)["series"]["shared_assets_hash"]

        registered = series_service.register_series_contract_episodes(series_id, v4)
        self.assertEqual(registered["counts"]["registered"], 3)
        self.assertEqual(registered["series"]["shared_assets_hash"], shared_hash)
        self.assertEqual(
            [item["ep_id"] for item in registered["episodes"]],
            [f"{series_id}_ep_001", f"{series_id}_ep_002", f"{series_id}_ep_003"],
        )
        self.assertEqual(
            registered["episodes"][1]["continuity_state_in"]["contract_state"],
            v4["season_outline"][0]["continuity_state_out"],
        )
        dependency_hash_sets = []
        for record in registered["episodes"]:
            jobs = task_store.list_jobs(record["ep_id"])
            self.assertTrue(all(
                float(job["metadata"]["inputs"]["settings"]["duration_seconds"]) == 124 / 24
                for job in jobs
            ))
            self.assertTrue(all(
                job["metadata"]["inputs"]["settings"]["render_profile"] == "proof"
                and job["metadata"]["inputs"]["settings"]["delivery_eligible"] is False
                for job in jobs
            ))
            self.assertEqual(sum(
                float(job["metadata"]["inputs"]["settings"]["edit_duration_seconds"])
                for job in jobs
            ), 20.0)
            dependency_hash_sets.append({
                dep["content_hash"]
                for job in jobs
                for dep in job["metadata"]["inputs"]["asset_dependencies"]
            })
        self.assertTrue(dependency_hash_sets[0])
        self.assertEqual(dependency_hash_sets[0], dependency_hash_sets[1])
        self.assertEqual(dependency_hash_sets[1], dependency_hash_sets[2])
        with self.assertRaisesRegex(RuntimeError, "previous_episode_not_succeeded"):
            series_service.start_episode(series_id, 2)

        calls: list[str] = []
        failed_once = {"value": False}

        def flaky_runner(ep_id: str, **_kwargs):
            calls.append(ep_id)
            if ep_id.endswith("ep_002") and not failed_once["value"]:
                failed_once["value"] = True
                first = task_store.list_jobs(ep_id)[0]
                task_store.default_store().update_job(first["job_id"], status="failed", error="offline injected fault")
            else:
                self._complete_jobs(ep_id)
            return {"started": True}

        tail = Path(self._file("continuity/mock_tail.png", b"tail-frame"))
        with mock.patch.object(series_service, "_tail_frame", return_value=tail):
            first_run = series_service.run_series_production(series_id, episode_runner=flaky_runner)
        self.assertEqual([item["status"] for item in first_run["snapshot"]["episodes"]], ["succeeded", "failed", "registered"])
        self.assertEqual(calls, [f"{series_id}_ep_001", f"{series_id}_ep_002"])

        series_service.retry_episode(series_id, 2)
        with mock.patch.object(series_service, "_tail_frame", return_value=tail):
            second_run = series_service.run_series_production(series_id, episode_runner=flaky_runner)
        self.assertEqual([item["status"] for item in second_run["snapshot"]["episodes"]], ["succeeded"] * 3)
        self.assertEqual(calls.count(f"{series_id}_ep_001"), 1)
        third_contract = task_store.project_snapshot(f"{series_id}_ep_003")["episode"]
        self.assertEqual(third_contract["panels"][0]["first_frame_path"], str(tail))
        self.assertEqual(
            third_contract["panels"][0]["series_continuity_state_in"]["contract_state"],
            v4["season_outline"][2]["continuity_state_in"],
        )

        # Reopen both SQLite stores to prove that the full V4 and completed
        # episode state survive process/browser restart.
        task_store._default_store = None
        series_store._default = None
        recovered = series_service.status_series(series_id)
        self.assertEqual(recovered["series_contract_v4"], v4)
        self.assertEqual(recovered["counts"]["complete"], 3)

        def fake_exporter(actual_series_id: str, number: int, preset: str):
            output = self._file(f"deliveries/{number}/final.mp4", f"episode-{number}".encode())
            manifest = self._file(f"deliveries/{number}/manifest.json", b"{}")
            return {"output_path": output, "manifest_path": manifest, "preset": preset}

        package = series_service.export_season(
            series_id, "vertical_9_16", episode_exporter=fake_exporter,
        )
        with zipfile.ZipFile(package["package_path"]) as bundle:
            names = set(bundle.namelist())
            self.assertIn("season_manifest.json", names)
            self.assertIn("series.json", names)
            self.assertTrue(all(f"episodes/{number:03d}/final.mp4" in names for number in range(1, 4)))
            manifest = json.loads(bundle.read("season_manifest.json"))
        self.assertEqual(manifest["series_sha256"], v4["series_sha256"])
        self.assertEqual(manifest["episode_count"], 3)

    def test_v4_registration_rejects_inexact_approval_duration_hash_and_state_chain(self):
        v4 = self._fake_minimax_v4()
        series_id = v4["series_bible"]["series_id"]
        draft = series_service.prepare_series_contract(series_id, v4)
        series_service.approve_series(series_id, expected_hash=draft["series"]["contract_hash"])

        unapproved = copy.deepcopy(v4)
        unapproved["episode_approvals"]["ep_003"] = False
        with self.assertRaisesRegex(ValueError, "all V4 episode contracts must be approved"):
            series_service.register_series_contract_episodes(series_id, unapproved)

        missing = copy.deepcopy(v4)
        missing["episode_contracts"].pop("ep_003")
        with self.assertRaisesRegex(ValueError, "exactly N"):
            series_service.register_series_contract_episodes(series_id, missing)

        bad_duration = copy.deepcopy(v4)
        bad_duration["season_outline"][0]["duration_seconds"] = 19
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            series_service.prepare_series_contract(series_id, bad_duration)

        bad_chain = copy.deepcopy(v4)
        bad_chain["season_outline"][1]["continuity_state_in"] = {"illegal": "reset"}
        with self.assertRaisesRegex(ValueError, "must exactly equal"):
            series_service.prepare_series_contract(series_id, bad_chain)

        stale_hash = copy.deepcopy(v4)
        stale_hash["shared_character_bible"][0]["model_identity_tags_en"].append("blue eyes")
        with self.assertRaisesRegex(ValueError, "series_sha256"):
            series_service.prepare_series_contract(series_id, stale_hash)


if __name__ == "__main__":
    unittest.main()
