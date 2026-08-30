from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

import orchestrator
import series_service
import series_store
import task_store


class SeriesPipelineTests(unittest.TestCase):
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

    def _spec(self, with_assets: bool = True) -> dict:
        char_refs = [self._file("shared/hero.png", b"hero")] if with_assets else []
        scene_refs = [self._file("shared/station.png", b"station")] if with_assets else []
        return {
            "title": "Night Line",
            "theme": "memory and responsibility",
            "synopsis": "An investigator follows a signal across two connected nights.",
            "episode_count": 2,
            "episode_seconds": 4.0,
            "visual_bible": {"style_prompt": "cinematic ink animation", "aspect_ratio": "16:9"},
            "world_bible": {"rules": "the signal appears only after midnight"},
            "character_bible": [{
                "character_id": "char_hero", "name": "Hero",
                "identity_prompt": "young man, short black hair, amber eyes",
                "reference_images": char_refs,
            }],
            "scene_bible": [{
                "scene_id": "scene_station", "description": "empty station at night",
                "positive_prompt": "rain-lit empty station", "panel_ids": [],
                "reference_images": scene_refs,
            }],
        }

    def _episodes(self) -> list[dict]:
        rows = []
        for number in (1, 2):
            rows.append({
                "episode_number": number,
                "story_bible": {"logline": f"Connected episode {number}"},
                "panels": [{
                    "panel_id": f"panel_{number:02d}", "scene_id": "scene_station",
                    "character_ids": ["char_hero"], "scene_description": "empty station at night",
                    "motion": f"episode {number} action",
                    "spoken_dialogue": [{"start_s": 0.5, "end_s": 1.5, "speaker_id": "char_hero", "text": f"line {number}"}],
                    "prompt_package": {"scene_id": "scene_station", "positive_prompt": f"episode {number} beat"},
                }],
            })
        return rows

    def _ready_series(self) -> dict:
        draft = series_service.prepare_series("series_night", self._spec())
        series_service.approve_series("series_night", expected_hash=draft["series"]["contract_hash"])
        series_service.approve_shared_assets("series_night")
        return series_service.register_episodes("series_night", self._episodes())

    def _complete_jobs(self, ep_id: str) -> None:
        store = task_store.default_store()
        for job in task_store.list_jobs(ep_id):
            output = Path(job["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"clip-{ep_id}".encode())
            store.update_job(
                job["job_id"], status="succeeded", output_path=str(output), preview_path=str(output),
                probe={"duration_seconds": 4.0, "video": {"width": 1920, "height": 1080, "fps": 24.0}},
                metadata={**job["metadata"], "artifact_sha256": hashlib_sha(ep_id)},
            )

    def test_exact_episode_count_and_duration_are_hard_gates(self):
        draft = series_service.prepare_series("series_night", self._spec())
        series_service.approve_series("series_night", expected_hash=draft["series"]["contract_hash"])
        with self.assertRaisesRegex(ValueError, "exactly 2 episodes"):
            series_service.register_episodes("series_night", self._episodes()[:1])
        invalid = self._episodes()
        invalid[0]["panels"][0]["duration_seconds"] = 3.0
        with self.assertRaisesRegex(ValueError, "does not equal required"):
            series_service.register_episodes("series_night", invalid)
        self.assertEqual(series_store.default_series_store().list_episodes("series_night"), [])
        self.assertEqual(task_store.list_jobs("series_night_ep_001"), [])

    def test_shared_assets_generate_once_and_every_episode_uses_same_hashes(self):
        draft = series_service.prepare_series("series_night", self._spec(with_assets=False))
        series_service.approve_series("series_night", expected_hash=draft["series"]["contract_hash"])
        calls: list[str] = []

        def character_generator(source, visual, **kwargs):
            calls.append(source["character_id"])
            return {"reference_images": [self._file("generated/hero.png", b"generated-hero")]}

        def scene_generator(source, visual, **kwargs):
            calls.append(source["scene_id"])
            return {"reference_images": [self._file("generated/station.png", b"generated-station")]}

        series_service.run_prepare_shared_assets(
            "series_night", character_generator=character_generator, scene_generator=scene_generator,
        )
        series_service.run_prepare_shared_assets(
            "series_night", character_generator=character_generator, scene_generator=scene_generator,
        )
        self.assertEqual(calls, ["char_hero", "scene_station"])
        approved = series_service.approve_shared_assets("series_night")
        self.assertEqual(approved["series"]["shared_assets_status"], "approved")
        registered = series_service.register_episodes("series_night", self._episodes())
        dependency_hashes = []
        for record in registered["episodes"]:
            job = task_store.list_jobs(record["ep_id"])[0]
            dependency_hashes.append({item["content_hash"] for item in job["metadata"]["inputs"]["asset_dependencies"]})
        self.assertEqual(dependency_hashes[0], dependency_hashes[1])
        self.assertEqual(len(dependency_hashes[0]), 2)

    def test_unapproved_shared_assets_and_unfinished_predecessor_block_start(self):
        draft = series_service.prepare_series("series_night", self._spec())
        series_service.approve_series("series_night", expected_hash=draft["series"]["contract_hash"])
        series_service.register_episodes("series_night", self._episodes())
        with self.assertRaisesRegex(RuntimeError, "shared_assets_not_approved"):
            series_service.start_series("series_night")
        series_service.approve_shared_assets("series_night")
        with self.assertRaisesRegex(RuntimeError, "previous_episode_not_succeeded"):
            series_service.start_episode("series_night", 2)

    def test_cross_episode_chain_break_is_a_hard_gate(self):
        self._ready_series()
        store = series_store.default_series_store()
        with store.connection() as conn:
            conn.execute(
                "UPDATE series_episodes SET predecessor_ep_id='wrong_ep' WHERE series_id=? AND episode_number=2",
                ("series_night",),
            )
        with self.assertRaisesRegex(RuntimeError, "cross_episode_continuity_chain_broken"):
            series_service.start_series("series_night")

    def test_series_runner_enforces_continuity_and_skips_success_on_restart(self):
        snapshot = self._ready_series()
        calls: list[str] = []
        tail = Path(self._file("continuity/ep1_tail.png", b"tail"))

        def runner(ep_id: str, **kwargs):
            calls.append(ep_id)
            self._complete_jobs(ep_id)
            return {"started": True}

        with mock.patch.object(series_service, "_tail_frame", return_value=tail):
            result = series_service.run_series_production("series_night", episode_runner=runner)
            rerun = series_service.run_series_production("series_night", episode_runner=runner)
        self.assertTrue(result["started"])
        self.assertEqual(len(calls), 2)
        self.assertEqual([item["status"] for item in rerun["snapshot"]["episodes"]], ["succeeded", "succeeded"])
        episode_two = rerun["snapshot"]["episodes"][1]
        self.assertEqual(episode_two["continuity_state_in"]["last_clip_path"], rerun["snapshot"]["episodes"][0]["last_clip_path"])
        ep_two_contract = task_store.project_snapshot(episode_two["ep_id"])["episode"]
        self.assertEqual(ep_two_contract["panels"][0]["first_frame_path"], str(tail))
        reregistered = series_service.register_episodes("series_night", self._episodes())
        self.assertEqual([item["status"] for item in reregistered["episodes"]], ["succeeded", "succeeded"])
        self.assertTrue(all(job["status"] == "succeeded" for item in reregistered["episodes"] for job in task_store.list_jobs(item["ep_id"])))
        series_store._default = None
        task_store._default_store = None
        recovered = series_service.status_series("series_night")
        self.assertEqual(recovered["counts"]["complete"], 2)

    def test_retry_cancel_and_background_commands_are_series_scoped(self):
        snapshot = self._ready_series()
        first = snapshot["episodes"][0]
        job = task_store.list_jobs(first["ep_id"])[0]
        task_store.default_store().update_job(job["job_id"], status="failed", error="mock")
        retried = series_service.retry_episode("series_night", 1)
        self.assertEqual(task_store.list_jobs(first["ep_id"])[0]["status"], "queued")
        cancelled = series_service.cancel_episode("series_night", 1)
        self.assertEqual(cancelled["episodes"][0]["status"], "cancelled")
        with mock.patch.object(series_service.subprocess, "Popen") as popen:
            popen.return_value.pid = 9001
            assets = series_service.prepare_shared_assets("series_night", timeout=20)
            season = series_service.start_series("series_night", timeout=20)
        self.assertIn("--shared-assets-only", assets["command"])
        self.assertNotIn("--shared-assets-only", season["command"])
        self.assertEqual(popen.call_args_list[0].kwargs["stdin"], subprocess_devnull())

    def test_season_package_requires_and_contains_every_episode(self):
        self._ready_series()
        records = series_store.default_series_store().list_episodes("series_night")
        for record in records:
            self._complete_jobs(record["ep_id"])
        series_service.status_series("series_night")

        def exporter(series_id: str, number: int, preset: str):
            output = Path(self._file(f"deliveries/{number}/final.mp4", f"episode-{number}".encode()))
            manifest = Path(self._file(f"deliveries/{number}/manifest.json", b"{}"))
            return {"output_path": str(output), "manifest_path": str(manifest), "preset": preset}

        package = series_service.export_season(
            "series_night", "vertical_9_16", episode_exporter=exporter,
        )
        self.assertTrue(Path(package["package_path"]).is_file())
        with zipfile.ZipFile(package["package_path"]) as bundle:
            self.assertIn("episodes/001/final.mp4", bundle.namelist())
            self.assertIn("episodes/002/final.mp4", bundle.namelist())


def hashlib_sha(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def subprocess_devnull():
    import subprocess
    return subprocess.DEVNULL


if __name__ == "__main__":
    unittest.main()
