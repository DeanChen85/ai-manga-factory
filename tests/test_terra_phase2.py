from __future__ import annotations

import copy
import json
import hashlib
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from PIL import Image


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

import orchestrator
import atomic_io
import continuity_safe
import render_service
import render_video_h3 as renderer
import scene_asset
import story_splitter
import subtitle_delivery
import task_store
import video_delivery
import video_quality
import worker


def phase2_episode(char_a: str, char_b: str, scene_a: str, scene_b: str) -> dict:
    return {
        "ep_id": "ep_phase2",
        "schema_version": "comic-production-v2",
        "story_bible": {"logline": "Two investigators cross a storm-lit station."},
        "visual_bible": {"style_prompt": "premium cinematic ink animation", "aspect_ratio": "16:9"},
        "render_settings": {"duration_seconds": 4.0, "continuity_mode": "strict"},
        "character_bible": [
            {"character_id": "char_a", "name": "A", "identity_prompt": "short black hair", "reference_images": [char_a]},
            {"character_id": "char_b", "name": "B", "identity_prompt": "silver braid", "reference_images": [char_b]},
        ],
        "scene_bible": [
            {"scene_id": "scene_station", "description": "empty rain-lit station", "positive_prompt": "rain-lit station", "panel_ids": ["panel_01", "panel_02"], "reference_images": [scene_a]},
            {"scene_id": "scene_roof", "description": "empty roof at dawn", "positive_prompt": "roof at dawn", "panel_ids": ["panel_03"], "reference_images": [scene_b]},
        ],
        "panels": [
            {
                "panel_id": "panel_01", "scene_id": "scene_station", "character_ids": ["char_a"],
                "continuity_group": "station_chain", "scene_description": "empty rain-lit station",
                "final_state": "A holds at the left platform mark",
                "motion": "A enters and looks left", "spoken_dialogue": [{"time_range": "0.5-1.5s", "speaker_id": "char_a", "text": "有人吗？"}],
                "on_screen_text": [{"start_s": 2.0, "end_s": 2.5, "text": "第七码头"}],
                "prompt_package": {"scene_id": "scene_station", "positive_prompt": "story beat A enters; continuity locked", "negative_prompt": "identity drift"},
            },
            {
                "panel_id": "panel_02", "scene_id": "scene_station", "character_ids": ["char_a"],
                "continuity_group": "station_chain", "scene_description": "empty rain-lit station",
                "final_state": "A faces the source of the click",
                "motion": "A turns toward a metallic click", "spoken_dialogue": [{"start_s": 0.25, "end_s": 1.25, "speaker_id": "char_a", "text": "谁在那里？"}],
                "prompt_package": {"scene_id": "scene_station", "positive_prompt": "reaction beat; preserve action continuity", "negative_prompt": "identity drift"},
            },
            {
                "panel_id": "panel_03", "scene_id": "scene_roof", "character_ids": ["char_b"],
                "scene_description": "empty roof at dawn", "motion": "B watches the horizon",
                "final_state": "B remains framed against the dawn horizon",
                "spoken_dialogue": [],
                "prompt_package": {"scene_id": "scene_roof", "positive_prompt": "independent roof beat", "negative_prompt": "identity drift"},
            },
        ],
    }


class TerraPhase2Tests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.old_env = dict(os.environ)
        os.environ["AI_MANGA_PROJECTS_DIR"] = str(self.base / "projects")
        os.environ["AI_MANGA_JOB_DB"] = str(self.base / "state" / "jobs.sqlite3")
        os.environ["AI_FACTORY_ROOT"] = str(self.base)
        task_store._default_store = None
        orchestrator.PROJECTS_DIR = self.base / "projects"

    def tearDown(self):
        task_store._default_store = None
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tempdir.cleanup()

    def _file(self, name: str, payload: bytes) -> str:
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)

    def _png(self, name: str, size: tuple[int, int] = (608, 1056)) -> Path:
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (32, 38, 46)).save(path)
        return path

    def _episode(self) -> dict:
        return phase2_episode(
            self._file("inputs/char_a.png", b"char-a"),
            self._file("inputs/char_b.png", b"char-b"),
            self._file("inputs/scene_station.png", b"scene-station"),
            self._file("inputs/scene_roof.png", b"scene-roof"),
        )

    def _approve(self, episode: dict) -> dict:
        snapshot = task_store.prepare_contract("ep_phase2", episode)
        task_store.approve_contract("ep_phase2", expected_hash=snapshot["pipeline"]["contract_hash"])
        return task_store.approve_assets("ep_phase2")

    @staticmethod
    def _with_edit_plan(episode: dict, edit_duration: float = 4.0) -> dict:
        episode.setdefault("render_settings", {})["target_edit_duration_seconds"] = round(
            len(episode["panels"]) * edit_duration, 6
        )
        for index, panel in enumerate(episode["panels"], 1):
            panel.update({
                "source_generation_duration_seconds": 10.125,
                "edit_duration_seconds": edit_duration,
                "shot_role": panel.get("shot_role") or "setup",
                "story_beat_id": panel.get("story_beat_id") or f"beat_{index:02d}",
                "visible_action": panel.get("visible_action") or panel.get("motion") or f"action {index}",
                "first_state": panel.get("first_state") or f"state before beat {index}",
                "final_state": panel.get("final_state") or f"state after beat {index}",
                "camera_plan": panel.get("camera_plan") or {
                    "shot_size": "medium", "angle": "eye-level",
                    "movement": "slow push", "composition": "single coherent shot",
                },
            })
        return episode

    def _mark_all_success(self) -> list[dict]:
        store = task_store.default_store()
        for job in store.list_jobs("ep_phase2"):
            output = Path(job["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"clip-{job['panel_index']}".encode())
            artifact_sha = hashlib.sha256(output.read_bytes()).hexdigest()
            analysis = self._quality_analysis(output)
            requested_edit = ((job["metadata"].get("inputs") or {}).get("shot_plan") or {}).get(
                "edit_duration_seconds"
            )
            edit_selection = None
            if requested_edit is not None:
                edit_selection = {
                    "in_seconds": 0.0, "out_seconds": float(requested_edit),
                    "duration_seconds": float(requested_edit), "reason": "offline approved test window",
                    "metrics": {"window_mean_luma_change": 0.1, "candidate_count": 1,
                                "viable_candidate_count": 1},
                    "selector": {"name": "test-selector", "version": "1"},
                    "source_artifact_sha256": artifact_sha,
                    "source_decoded_visual_sha256": analysis["decoded_visual_sha256"],
                }
                edit_selection["selection_sha256"] = video_quality._selection_hash(edit_selection)
            metadata = {
                **job["metadata"], "artifact_sha256": artifact_sha,
                "content_qa": {"passed": True, "analysis": analysis, "reasons": []},
                "editorial_review": {
                    "status": "approved", "artifact_sha256": artifact_sha,
                    "decoded_visual_sha256": analysis["decoded_visual_sha256"],
                },
                "release": {
                    "status": "approved", "artifact_sha256": artifact_sha,
                    "decoded_visual_sha256": analysis["decoded_visual_sha256"],
                },
            }
            if edit_selection:
                metadata["edit_selection"] = edit_selection
                metadata["editorial_review"]["edit_selection_sha256"] = edit_selection["selection_sha256"]
                metadata["release"]["edit_selection_sha256"] = edit_selection["selection_sha256"]
            store.update_job(
                job["job_id"], status="succeeded", output_path=str(output),
                preview_path=str(output),
                probe={"duration_seconds": 10.125 if edit_selection else 4.0, "video": {"width": 1920, "height": 1080, "fps": 24.0}},
                metadata=metadata,
            )
        return store.list_jobs("ep_phase2")

    @staticmethod
    def _quality_analysis(path, **kwargs):
        token = hashlib.sha256(Path(path).name.encode("utf-8")).hexdigest()
        return {
            "decoded_visual_sha256": token,
            "sample_stream_sha256": token,
            "perceptual_hashes": [token, token],
            "static": False,
            "metrics": {"sample_count": 2, "mean_adjacent_luma_change": 0.1,
                        "first_last_luma_change": 0.1},
            "algorithm": {"name": "test-visual-qa", "version": "1"},
        }

    def test_contract_and_asset_approval_gate_round_trip(self):
        episode = self._episode()
        draft = task_store.prepare_contract("ep_phase2", episode)
        self.assertEqual(draft["pipeline"]["contract_status"], "draft")
        self.assertFalse(task_store.production_gate("ep_phase2")["ready"])
        task_store.approve_contract("ep_phase2")
        self.assertIn("assets_not_approved", task_store.production_gate("ep_phase2")["reasons"])
        approved = task_store.approve_assets("ep_phase2")
        self.assertTrue(task_store.production_gate("ep_phase2")["ready"])
        self.assertEqual(approved["pipeline"]["assets_status"], "approved")
        reloaded = task_store.project_snapshot("ep_phase2")
        self.assertEqual(reloaded["pipeline"]["contract_status"], "approved")
        self.assertTrue(all(asset["approved"] for asset in reloaded["assets"]["items"]))

    def test_episode_json_atomic_writers_use_unique_temps_retry_and_preserve_foreign_temp(self):
        self.assertIs(task_store._write_json_atomic, orchestrator._write_json_atomic)
        episode_path = self.base / "projects" / "ep_atomic" / "episode.json"
        episode_path.parent.mkdir(parents=True)
        foreign_temp = episode_path.parent / f".{episode_path.name}.other-writer.tmp"
        foreign_temp.write_text("foreign writer owns this", encoding="utf-8")

        real_replace = atomic_io.os.replace
        replace_attempts = []

        def transient_permission_error(source, destination):
            replace_attempts.append((Path(source), Path(destination)))
            if len(replace_attempts) < 3:
                raise PermissionError(5, "simulated Windows target lock")
            return real_replace(source, destination)

        with mock.patch.object(atomic_io.os, "replace", side_effect=transient_permission_error):
            task_store._write_json_atomic(
                episode_path, {"writer": "retry"},
                replace_attempts=4, initial_backoff=0, sleep_func=lambda _: None,
            )
        self.assertEqual(len(replace_attempts), 3)
        self.assertEqual(json.loads(episode_path.read_text(encoding="utf-8")), {"writer": "retry"})
        self.assertTrue(foreign_temp.is_file())

        barrier = threading.Barrier(8)
        errors: list[Exception] = []

        def concurrent_writer(index: int) -> None:
            try:
                barrier.wait(timeout=2.0)
                task_store._write_json_atomic(episode_path, {"writer": index, "valid": True})
            except Exception as exc:  # assertion reports every thread failure
                errors.append(exc)

        threads = [threading.Thread(target=concurrent_writer, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5.0)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        final_payload = json.loads(episode_path.read_text(encoding="utf-8"))
        self.assertTrue(final_payload["valid"])
        self.assertIn(final_payload["writer"], range(8))
        self.assertTrue(foreign_temp.is_file())
        own_leftovers = [
            path for path in episode_path.parent.glob(f".{episode_path.name}.*.tmp")
            if path != foreign_temp
        ]
        self.assertEqual(own_leftovers, [])

    def test_contract_validation_rejects_unknown_scene_without_silent_rebind(self):
        episode = self._episode()
        episode["panels"][0]["scene_id"] = "scene_missing"
        task_store.prepare_contract("ep_phase2", episode)
        with self.assertRaisesRegex(ValueError, "unknown scene_id"):
            task_store.approve_contract("ep_phase2")

    def test_scene_and_character_refs_are_role_bound_and_selectively_invalidate(self):
        episode = self._episode()
        self._approve(episode)
        jobs = self._mark_all_success()
        first_inputs = jobs[0]["metadata"]["inputs"]
        self.assertEqual(
            {item["role"] for item in first_inputs["reference_inputs"]},
            {"character_reference", "scene_reference"},
        )
        self.assertEqual(
            {(dep["asset_type"], dep["source_id"]) for dep in first_inputs["asset_dependencies"]},
            {("character", "char_a"), ("scene", "scene_station")},
        )
        Path(episode["character_bible"][0]["reference_images"][0]).write_bytes(b"char-a-updated")
        refreshed = task_store.prepare_contract("ep_phase2", episode)
        statuses = {job["panel_name"]: job["status"] for job in refreshed["jobs"]}
        self.assertEqual(statuses["panel_01"], "queued")
        self.assertEqual(statuses["panel_02"], "queued")
        self.assertEqual(statuses["panel_03"], "succeeded")
        self.assertFalse(task_store.production_gate("ep_phase2")["ready"])

    def test_ensemble_reference_selection_does_not_starve_later_characters(self):
        refs = []
        expected_anchors = []
        for character_id in ("char_a", "char_b", "char_c", "char_d", "char_e"):
            character_paths = [
                Path(self._file(f"ensemble/{character_id}_{index}.png", f"{character_id}-{index}".encode()))
                for index in range(3)
            ]
            expected_anchors.append(character_paths[0])
            refs.extend({
                "role": "character_reference",
                "source_id": character_id,
                "resolved": path,
            } for path in character_paths)
        selected = orchestrator._select_character_reference_paths(refs)
        self.assertEqual(selected, expected_anchors)
        single_actor = [item for item in refs if item["source_id"] == "char_a"]
        self.assertEqual(
            orchestrator._select_character_reference_paths(single_actor),
            [item["resolved"] for item in single_actor],
        )

    def test_creative_asset_prompt_change_queues_regeneration_not_old_file_reuse(self):
        episode = self._episode()
        self._approve(episode)
        self._mark_all_success()
        task_store.prepare_contract("ep_phase2", episode)
        self._mark_all_success()
        changed = json.loads(json.dumps(episode))
        changed["scene_bible"][1]["positive_prompt"] = "storm-torn roof at blue dawn"
        snapshot = task_store.prepare_contract("ep_phase2", changed)
        assets = {(item["asset_type"], item["source_id"]): item for item in snapshot["assets"]["items"]}
        self.assertEqual(assets[("scene", "scene_roof")]["status"], "queued")
        self.assertEqual(assets[("scene", "scene_roof")]["reference_images"], [])
        # Persisting the same in-memory episode during another asset stage can
        # still carry its old scene paths. They must not resurrect this queued
        # dependency before the scene generator actually succeeds.
        repeated = task_store.prepare_contract("ep_phase2", changed)
        repeated_assets = {
            (item["asset_type"], item["source_id"]): item
            for item in repeated["assets"]["items"]
        }
        self.assertEqual(repeated_assets[("scene", "scene_roof")]["status"], "queued")
        self.assertEqual(repeated_assets[("scene", "scene_roof")]["reference_images"], [])
        self.assertEqual(snapshot["jobs"][0]["status"], "succeeded")
        self.assertEqual(snapshot["jobs"][1]["status"], "succeeded")
        self.assertEqual(snapshot["jobs"][2]["status"], "queued")

    def test_registering_same_contract_does_not_erase_failed_asset_diagnostics(self):
        episode = self._episode()
        snapshot = task_store.prepare_contract("ep_phase2", episode)
        failed = next(
            item for item in snapshot["assets"]["items"]
            if item["asset_type"] == "character" and item["source_id"] == "char_a"
        )
        task_store.default_store().update_asset(
            failed["asset_id"],
            status="failed",
            approved=False,
            content_hash=None,
            reference_images=[],
            prompt_id="comfy-prompt-42",
            error="checkpoint prompt compile failed",
            retry_count=1,
        )

        repeated = task_store.prepare_contract("ep_phase2", episode)
        restored = next(
            item for item in repeated["assets"]["items"]
            if item["asset_id"] == failed["asset_id"]
        )
        self.assertEqual(restored["status"], "failed")
        self.assertEqual(restored["error"], "checkpoint prompt compile failed")
        self.assertEqual(restored["prompt_id"], "comfy-prompt-42")
        self.assertEqual(restored["retry_count"], 1)
        self.assertEqual(restored["reference_images"], [])

    def test_reject_asset_queues_only_target_and_keeps_audit_for_regeneration(self):
        episode = self._episode()
        self._approve(episode)
        self._mark_all_success()
        task_store.prepare_contract("ep_phase2", episode)
        self._mark_all_success()
        snapshot = render_service.reject_asset(
            "ep_phase2", asset_type="scene", source_id="scene_roof", reason="contains character and fake text"
        )
        self.assertEqual(snapshot["asset_action"]["status"], "queued")
        assets = {(item["asset_type"], item["source_id"]): item for item in snapshot["assets"]["items"]}
        rejected = assets[("scene", "scene_roof")]
        self.assertEqual(rejected["status"], "queued")
        self.assertFalse(rejected["approved"])
        self.assertIsNone(rejected["content_hash"])
        self.assertEqual(rejected["reference_images"], [])
        self.assertEqual(rejected["metadata"]["rejection_audit"][-1]["reason"], "contains character and fake text")
        self.assertEqual([job["status"] for job in snapshot["jobs"]], ["succeeded", "succeeded", "queued"])
        self.assertEqual(snapshot["pipeline"]["contract_status"], "approved")
        self.assertEqual(snapshot["pipeline"]["assets_status"], "pending")
        rejected_scene = next(item for item in snapshot["episode"]["scene_bible"] if item["scene_id"] == "scene_roof")
        self.assertEqual(rejected_scene["reference_images"], [])
        retried = render_service.retry_asset("ep_phase2", rejected["asset_id"], reason="retry after prompt fix")
        self.assertEqual(retried["asset_action"]["action"], "retry_requested")
        retried_asset = next(item for item in retried["assets"]["items"] if item["asset_id"] == rejected["asset_id"])
        self.assertEqual(len(retried_asset["metadata"]["rejection_audit"]), 2)
        regenerated: list[str] = []

        def scene_generator(scene, visual_bible, **kwargs):
            regenerated.append(scene["scene_id"])
            path = self._file(f"regenerated/{scene['scene_id']}.png", b"clean-empty-scene")
            return {"prompt_id": "scene-retry", "reference_images": [path]}

        regenerated_snapshot = orchestrator.prepare_scene_assets(
            "ep_phase2", generator=scene_generator
        )
        self.assertEqual(regenerated, ["scene_roof"])
        regenerated_asset = next(
            item for item in regenerated_snapshot["assets"]["items"]
            if item["asset_id"] == rejected["asset_id"]
        )
        self.assertEqual(regenerated_asset["status"], "succeeded")
        self.assertFalse(regenerated_asset["approved"])

    def test_repeated_reject_and_regenerate_clicks_launch_only_one_asset_worker(self):
        self._approve(self._episode())
        first_reject = render_service.reject_asset(
            "ep_phase2", asset_type="scene", source_id="scene_roof", reason="first review rejection",
        )
        asset_id = first_reject["asset_action"]["asset_id"]
        render_service.retry_asset("ep_phase2", asset_id, reason="first regenerate request")

        with mock.patch.object(worker.subprocess, "Popen") as popen:
            popen.return_value.pid = 7331
            first_start = render_service.prepare_assets("ep_phase2", timeout=20)
            # Model the second Web click arriving before the first child has
            # imported worker.py and acquired its normal runtime lease.
            render_service.reject_asset(
                "ep_phase2", asset_id, reason="duplicate review click",
            )
            render_service.retry_asset("ep_phase2", asset_id, reason="duplicate regenerate request")
            duplicate_start = render_service.prepare_assets("ep_phase2", timeout=20)

        self.assertTrue(first_start["started"])
        self.assertFalse(duplicate_start["started"])
        self.assertEqual(duplicate_start["reason"], "worker_already_running")
        self.assertEqual(duplicate_start["pid"], 7331)
        popen.assert_called_once()
        self.assertTrue(popen.call_args.kwargs["env"]["AI_MANGA_WORKER_LAUNCH_TOKEN"])

    def test_continuity_dependency_cascades_only_same_chain(self):
        episode = self._episode()
        self._approve(episode)
        jobs = self._mark_all_success()
        task_store.prepare_contract("ep_phase2", episode)
        current = task_store.list_jobs("ep_phase2")
        dependency = current[1]["metadata"]["inputs"]["continuity_dependency"]
        self.assertEqual(dependency["previous_job_id"], current[0]["job_id"])
        self.assertTrue(dependency["strict"])
        changed = json.loads(json.dumps(episode))
        changed["panels"][0]["prompt_package"]["positive_prompt"] += "; changed lens"
        refreshed = task_store.prepare_contract("ep_phase2", changed)["jobs"]
        self.assertEqual([job["status"] for job in refreshed], ["queued", "queued", "succeeded"])

    def test_panel_only_edit_keeps_six_assets_approved_and_invalidates_only_job_chain(self):
        ep_id = "ep_panel_only_edit"
        character_ids = [f"char_{index:02d}" for index in range(1, 6)]
        characters = []
        for index, character_id in enumerate(character_ids, 1):
            characters.append({
                "character_id": character_id,
                "name": f"Player {index}",
                "identity_prompt": f"adult player {index}, black hair, distinct face",
                "wardrobe_prompt": f"solid color casual outfit number {index}",
                "model_identity_tags_en": [
                    "1boy" if index % 2 else "1girl", "adult", "black hair", f"player {index}",
                ],
                "model_wardrobe_tags_en": [
                    f"solid color casual outfit {index}", "plain white sneakers",
                ],
                "negative_prompt": "wrong identity, duplicate person, random text",
                "reference_images": [self._file(
                    f"six_assets/{character_id}.png", f"reference-{character_id}".encode("utf-8")
                )],
            })
        scene_ref = self._file("six_assets/scene_room.png", b"approved-room-reference")
        raw = {
            "title": "Five-player room",
            "story_bible": {
                "title": "Five-player room",
                "logline": "Five friends complete one match together.",
                "synopsis": "A six-shot continuous match unfolds around one table.",
            },
            "visual_bible": {
                "style_name": "modern urban anime",
                "style_prompt": "premium modern urban 2D animation, restrained lighting",
                "global_negative_prompt": "random text, logo, identity drift",
            },
            "character_bible": characters,
            "scene_bible": [{
                "scene_id": "scene_room", "name": "Gaming room",
                "description": "compact modern living room with one centered gaming table",
                "positive_prompt": "eye-level wide shot of one modern living room",
                "negative_prompt": "retail store, showroom, random text",
                "panel_ids": [f"panel_{index:02d}" for index in range(1, 7)],
                "reference_images": [scene_ref],
            }],
            "panels": [{
                "panel_id": f"panel_{index:02d}",
                "scene_id": "scene_room",
                "character_ids": character_ids,
                "continuity_group": "main",
                "first_frame": f"wide shot beat {index}, all five friends around the same table",
                "last_frame": f"all five friends hold the continuous final pose for beat {index}",
                "camera_movement": "one slow push-in",
                "cuts": [{
                    "time_range": "0-4s",
                    "name": "single continuous shot",
                    "intensity": "SMOOTH",
                    "shot_description": f"all five friends perform action beat {index} around one table",
                }],
                "spoken_dialogue": [{
                    "start_s": 0.5, "end_s": 1.5,
                    "speaker_id": character_ids[0], "text": f"Beat {index}",
                }],
                "audio_cues": [],
                "continuity_state_in": {"table": "same room"},
                "continuity_state_out": {"table": "same room"},
            } for index in range(1, 7)],
        }
        settings = {
            "prompt_mode": "cinematic", "visual_style": "modern urban anime",
            "style_enforcement": "premium modern urban 2D animation, restrained lighting",
            "aspect_ratio": "16:9", "duration_seconds": 4.0,
            "background_music": "soft_piano", "ambience": "office_quiet",
            "voice_language": "English", "shot_count": 6, "total_duration_seconds": 24.0,
        }
        episode = story_splitter.enrich_episode_contract(
            raw, story_text=raw["story_bible"]["synopsis"], source_mode="LIVE", settings=settings,
        )
        self._with_edit_plan(episode)
        episode = story_splitter.enrich_episode_contract(
            episode, story_text=episode["story_bible"]["synopsis"],
            source_mode="LIVE", settings=episode["render_settings"],
        )
        self._with_edit_plan(episode)
        snapshot = task_store.prepare_contract(ep_id, episode)
        task_store.approve_contract(ep_id, expected_hash=snapshot["pipeline"]["contract_hash"])
        task_store.approve_assets(ep_id)
        store = task_store.default_store()
        for job in store.list_jobs(ep_id):
            output = Path(job["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"accepted-{job['panel_index']}".encode("utf-8"))
            artifact = hashlib.sha256(output.read_bytes()).hexdigest()
            selection = {
                "in_seconds": 0.0, "out_seconds": 4.0, "duration_seconds": 4.0,
                "reason": "test", "metrics": {"window_mean_luma_change": 0.1},
                "selector": {"name": "test-selector", "version": "1"},
                "source_artifact_sha256": artifact,
                "source_decoded_visual_sha256": hashlib.sha256(str(job["job_id"]).encode()).hexdigest(),
            }
            selection["selection_sha256"] = video_quality._selection_hash(selection)
            store.update_job(
                job["job_id"], status="succeeded", output_path=str(output), preview_path=str(output),
                metadata={
                    **job["metadata"], "artifact_sha256": artifact, "edit_selection": selection,
                    "content_qa": {"passed": True, "analysis": {
                        "decoded_visual_sha256": selection["source_decoded_visual_sha256"],
                    }},
                },
            )
        before_assets = {
            item["asset_id"]: (item["prompt_hash"], item["content_hash"])
            for item in store.list_assets(ep_id)
        }
        self.assertEqual(len(before_assets), 6)

        replacement = json.loads(json.dumps(episode["panels"][1]))
        replacement["editorial_first_frame"] = (
            "panel two revised action: the captain taps the phone once while all five remain seated"
        )
        replacement["first_frame"] = replacement["editorial_first_frame"]
        # This is deliberately a backend-only panel edit.  Re-running the
        # prompt enricher here would rewrite unrelated prompt packages and
        # make every job hash stale, masking the dependency behavior under
        # test.
        edited = json.loads(json.dumps(episode))
        edited["panels"][1] = replacement
        refreshed = task_store.prepare_contract(ep_id, edited)
        after_assets = {
            item["asset_id"]: (item["prompt_hash"], item["content_hash"])
            for item in refreshed["assets"]["items"]
        }
        self.assertEqual(after_assets, before_assets)
        self.assertTrue(all(item["status"] == "succeeded" for item in refreshed["assets"]["items"]))
        self.assertTrue(all(item["approved"] for item in refreshed["assets"]["items"]))
        self.assertEqual(refreshed["pipeline"]["assets_status"], "approved")
        self.assertEqual(
            [job["status"] for job in refreshed["jobs"]],
            ["succeeded", "queued", "queued", "queued", "queued", "queued"],
        )

    def test_select_references_recovers_queued_asset_without_changing_creative_hash(self):
        episode = self._episode()
        approved = self._approve(episode)
        asset = next(
            item for item in approved["assets"]["items"]
            if item["asset_type"] == "character" and item["source_id"] == "char_a"
        )
        replacement = Path(episode["character_bible"][0]["reference_images"][0])
        task_store.default_store().update_asset(
            asset["asset_id"], status="queued", approved=False,
            content_hash=None, reference_images=[], manifest_path=None,
            prompt_id=None, completed_at=None, approved_at=None,
        )
        before_hash = task_store.project_snapshot("ep_phase2")["pipeline"]["contract_hash"]

        selected = render_service.select_asset_references(
            "ep_phase2", asset["asset_id"], [str(replacement)],
            reason="reuse approved real reference after panel-only edit",
        )
        recovered = next(
            item for item in selected["assets"]["items"] if item["asset_id"] == asset["asset_id"]
        )
        self.assertEqual(recovered["status"], "succeeded")
        self.assertFalse(recovered["approved"])
        self.assertEqual(recovered["reference_images"], [str(replacement.resolve())])
        self.assertTrue(recovered["content_hash"])
        self.assertEqual(selected["pipeline"]["contract_hash"], before_hash)
        self.assertEqual(selected["pipeline"]["assets_status"], "ready_for_approval")
        self.assertEqual(
            recovered["metadata"]["selection_audit"][-1]["reason"],
            "reuse approved real reference after panel-only edit",
        )

    def test_recover_job_uses_artifact_prompt_history_without_submission(self):
        snapshot = task_store.prepare_contract("ep_phase2", self._episode())
        job = snapshot["jobs"][0]
        artifact = Path(job["output_path"]).with_suffix(".artifact.json")
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({
            "schema_version": 1,
            "job_id": job["job_id"],
            "prompt_id": "existing-comfy-prompt",
            "output_path": job["output_path"],
            "graph_path": str(artifact.with_suffix(".graph.json")),
            "timing_path": str(artifact.with_suffix(".cues.json")),
        }), encoding="utf-8")

        def fake_history_recovery(job_id, *, store):
            current = store.get_job(job_id)
            output = Path(current["output_path"])
            output.write_bytes(b"existing-accepted-clip")
            return store.update_job(
                job_id, status="succeeded", output_path=str(output), preview_path=str(output),
                metadata={**current["metadata"], "artifact_sha256": "existing-artifact"},
            )

        with mock.patch.object(render_service, "recover_render_job", side_effect=fake_history_recovery) as recover:
            restored = render_service.recover_job("ep_phase2", job["job_id"])

        self.assertEqual(restored["status"], "succeeded")
        self.assertEqual(restored["prompt_id"], "existing-comfy-prompt")
        recover.assert_called_once()

    def test_reject_succeeded_panel_archives_output_and_invalidates_strict_downstream(self):
        ep_id = "ep_job_rejection"
        project = self.base / "projects" / ep_id
        videos = project / "videos"
        videos.mkdir(parents=True)
        rows = []
        job_ids = [f"{ep_id}:{index:04d}:panel_{index:02d}" for index in range(1, 7)]
        for index, job_id in enumerate(job_ids, 1):
            output = videos / f"panel_{index:02d}.mp4"
            status = "succeeded" if index <= 2 else ("running" if index == 3 else "queued")
            if status == "succeeded":
                output.write_bytes(f"accepted-or-review-{index}".encode("utf-8"))
            dependency = {}
            if index > 1:
                dependency = {
                    "strict": True,
                    "previous_job_id": job_ids[index - 2],
                    "previous_input_hash": f"input-{index - 1}",
                    "previous_artifact_hash": f"artifact-{index - 1}",
                    "first_frame_source": "previous_tail",
                }
            rows.append({
                "job_id": job_id,
                "panel_index": index,
                "panel_name": f"panel_{index:02d}",
                "status": status,
                "progress": 1.0 if status == "succeeded" else (0.5 if status == "running" else 0.0),
                "prompt_id": f"prompt-{index}" if status in {"succeeded", "running"} else None,
                "output_path": str(output),
                "preview_path": str(output) if status == "succeeded" else None,
                "input_hash": f"input-{index}",
                "probe": {"duration_seconds": 10.125} if status == "succeeded" else {},
                "completed_at": "2026-08-12T00:00:00+00:00" if status == "succeeded" else None,
                "metadata": {
                    "artifact_sha256": f"artifact-{index}" if status == "succeeded" else None,
                    "inputs": {"continuity_dependency": dependency},
                },
            })
        store = task_store.default_store()
        store.register_jobs(ep_id, rows)
        rejected_output = Path(rows[1]["output_path"])

        cancelled = []

        def fake_cancel(job_id, *, interrupt_running=False):
            cancelled.append((job_id, interrupt_running))
            return {"job_id": job_id, "status": "cancelled"}

        with mock.patch.object(render_service, "cancel_render_job", side_effect=fake_cancel):
            result = render_service.reject_job(
                ep_id, job_ids[1], reason="random burned text and wrong final state",
                rejection_category="continuity_or_state",
            )

        current = task_store.list_jobs(ep_id)
        self.assertEqual(cancelled, [(job_ids[2], True)])
        self.assertEqual(result["cancelled_job_ids"], [job_ids[2]])
        self.assertEqual(result["invalidated_job_ids"], job_ids[2:])
        self.assertEqual([job["status"] for job in current], [
            "succeeded", "failed", "failed", "queued", "queued", "queued",
        ])
        self.assertTrue(Path(current[0]["output_path"]).is_file())
        self.assertEqual(current[0]["prompt_id"], "prompt-1")
        rejected = current[1]
        self.assertIsNone(rejected["output_path"])
        self.assertIsNone(rejected["preview_path"])
        self.assertIsNone(rejected["prompt_id"])
        self.assertEqual(rejected["probe"], {})
        self.assertIsNone(rejected["completed_at"])
        self.assertIn("random burned text", rejected["error"])
        audit = rejected["metadata"]["qa_rejection_audit"][-1]
        self.assertEqual(audit["category"], "continuity_or_state")
        retry_feedback = rejected["metadata"]["qa_retry_feedback"]
        self.assertEqual(retry_feedback["reason"], "random burned text and wrong final state")
        self.assertEqual(retry_feedback["category"], "continuity_or_state")
        self.assertEqual(retry_feedback["source"], "human_qa")
        self.assertEqual(len(retry_feedback["sha256"]), 64)
        archived_output = Path(audit["archived_files"]["output_path"]["path"])
        self.assertFalse(rejected_output.exists())
        self.assertTrue(archived_output.is_file())
        self.assertEqual(archived_output.read_bytes(), b"accepted-or-review-2")
        for downstream in current[2:]:
            dependency = downstream["metadata"]["inputs"]["continuity_dependency"]
            self.assertIsNone(dependency["previous_input_hash"])
            self.assertIsNone(dependency["previous_artifact_hash"])
            self.assertEqual(dependency["first_frame_source"], "previous_tail_pending")
            self.assertIsNone(downstream["prompt_id"])
            self.assertIsNone(downstream["output_path"])

        retried = render_service.retry(ep_id, job_ids[1])
        after_retry = task_store.list_jobs(ep_id)
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(Path(retried["output_path"]), rejected_output)
        self.assertEqual(after_retry[0]["status"], "succeeded")
        self.assertEqual([job["status"] for job in after_retry[1:]], [
            "queued", "failed", "queued", "queued", "queued",
        ])
        self.assertEqual(store.next_runnable(ep_id)["job_id"], job_ids[1])

    def test_reviewer_can_classify_a_legacy_job_rejection_without_rewriting_it(self):
        ep_id = "ep_legacy_rejection"
        job_id = f"{ep_id}:0001:p01"
        store = task_store.default_store()
        store.register_jobs(ep_id, [{
            "job_id": job_id, "panel_index": 1, "panel_name": "p01",
            "status": "failed", "input_hash": "input-1",
            "metadata": {"qa_rejection_audit": [{
                "action": "job_rejected", "reason": "handoff completed too late", "at": "t1",
            }]},
        }])
        result = render_service.classify_job_rejection(
            ep_id, job_id, rejection_category="action_timing_or_edit_window",
        )
        metadata = result["job"]["metadata"]
        self.assertNotIn("category", metadata["qa_rejection_audit"][-1])
        classification = metadata["qa_rejection_classification"]
        self.assertEqual(classification["category"], "action_timing_or_edit_window")
        self.assertEqual(classification["rejection_at"], "t1")
        self.assertEqual(classification["rejection_reason"], "handoff completed too late")

    def test_reviewer_must_explicitly_authorize_each_retry_beyond_the_limit(self):
        ep_id = "ep_retry_authorization"
        job_id = f"{ep_id}:0001:p01"
        store = task_store.default_store()
        store.register_jobs(ep_id, [{
            "job_id": job_id, "panel_index": 1, "panel_name": "p01",
            "status": "failed", "input_hash": "input-1",
            "retry_count": 2, "max_retries": 2,
            "error": "H3 prompt exceeds bounded complexity: 537 English words",
            "metadata": {},
        }])
        with self.assertRaisesRegex(ValueError, "reason is required"):
            render_service.authorize_additional_job_retry(ep_id, job_id, reason="")
        result = render_service.authorize_additional_job_retry(
            ep_id, job_id,
            reason="compressed the micro-timeline below the 512-word hard limit",
        )
        authorized = result["job"]
        self.assertEqual(authorized["retry_count"], 2)
        self.assertEqual(authorized["max_retries"], 3)
        audit = authorized["metadata"]["additional_retry_authorization_audit"][-1]
        self.assertEqual(audit["previous_max_retries"], 2)
        self.assertEqual(audit["new_max_retries"], 3)
        with self.assertRaisesRegex(RuntimeError, "extra authorization is not required"):
            render_service.authorize_additional_job_retry(
                ep_id, job_id, reason="must not grant multiple unused retries",
            )

    def test_reject_job_cannot_resurrect_cancelled_downstream_waiter(self):
        ep_id = "ep_concurrent_rejection"
        project = self.base / "projects" / ep_id
        videos = project / "videos"
        videos.mkdir(parents=True)
        job_ids = [f"{ep_id}:{index:04d}:panel_{index:02d}" for index in range(1, 7)]
        rows = []
        for index, current_job_id in enumerate(job_ids, 1):
            output = videos / f"panel_{index:02d}.mp4"
            status = "succeeded" if index <= 2 else ("running" if index == 3 else "queued")
            if status == "succeeded":
                output.write_bytes(f"clip-{index}".encode())
            dependency = ({
                "strict": True,
                "previous_job_id": job_ids[index - 2],
                "previous_input_hash": f"input-{index - 1}",
                "previous_artifact_hash": f"artifact-{index - 1}",
                "first_frame_source": "previous_tail",
            } if index > 1 else {})
            rows.append({
                "job_id": current_job_id, "panel_index": index,
                "panel_name": f"panel_{index:02d}", "status": status,
                "progress": 0.5 if status == "running" else float(status == "succeeded"),
                "prompt_id": f"prompt-{index}" if status in {"succeeded", "running"} else None,
                "output_path": str(output), "preview_path": str(output) if status == "succeeded" else None,
                "input_hash": f"input-{index}",
                "metadata": {"inputs": {"continuity_dependency": dependency}},
            })
        store = task_store.default_store()
        store.register_jobs(ep_id, rows)

        history_entered = threading.Event()
        release_history = threading.Event()
        wait_errors: list[Exception] = []

        def history_api(path, payload):
            self.assertEqual(path, "/history/prompt-3")
            history_entered.set()
            self.assertTrue(release_history.wait(2.0), "test did not release mocked history call")
            return {}

        def wait_in_worker():
            try:
                renderer.wait_render_job(
                    job_ids[2], store=store, api_func=history_api,
                    poll_interval=0.01, timeout=5.0,
                )
            except Exception as exc:  # expected cancellation/invalidation path
                wait_errors.append(exc)

        waiter = threading.Thread(target=wait_in_worker, daemon=True)
        waiter.start()
        self.assertTrue(history_entered.wait(2.0), "mock worker did not enter history poll")

        cancel_calls = []

        def cancel_active(job):
            return renderer.cancel_render_job(
                str(job["job_id"]), store=store,
                api_func=lambda path, payload: cancel_calls.append((path, payload)) or {},
                interrupt_running=True,
            )

        result = task_store.reject_job(
            ep_id, job_ids[1], reason="QA rejected panel 2",
            cancel_job=cancel_active,
        )
        release_history.set()
        waiter.join(2.0)

        self.assertFalse(waiter.is_alive(), "cancelled worker remained stuck in wait_render_job")
        self.assertEqual(len(wait_errors), 1)
        self.assertIn("invalidated while waiting", str(wait_errors[0]))
        self.assertEqual(result["cancelled_job_ids"], [job_ids[2]])
        self.assertEqual([job["status"] for job in store.list_jobs(ep_id)], [
            "succeeded", "failed", "failed", "queued", "queued", "queued",
        ])
        self.assertIsNone(store.get_job(job_ids[2])["prompt_id"])
        self.assertEqual([item[0] for item in cancel_calls], ["/queue", "/interrupt"])

    def test_worker_stops_after_concurrent_qa_chain_invalidation(self):
        ep_id = "ep_worker_qa_stop"
        project = self.base / "projects" / ep_id
        videos = project / "videos"
        videos.mkdir(parents=True)
        job_ids = [f"{ep_id}:{index:04d}:panel_{index:02d}" for index in range(1, 7)]
        rows = []
        for index, current_job_id in enumerate(job_ids, 1):
            output = videos / f"panel_{index:02d}.mp4"
            status = "succeeded" if index <= 2 else ("running" if index == 3 else "queued")
            if status == "succeeded":
                output.write_bytes(f"clip-{index}".encode())
            rows.append({
                "job_id": current_job_id, "panel_index": index,
                "panel_name": f"panel_{index:02d}", "status": status,
                "progress": 0.5 if status == "running" else float(status == "succeeded"),
                "prompt_id": f"prompt-{index}" if status in {"succeeded", "running"} else None,
                "output_path": str(output), "input_hash": f"input-{index}",
                "metadata": {"inputs": {"continuity_dependency": ({
                    "strict": True,
                    "previous_job_id": job_ids[index - 2],
                    "previous_input_hash": f"input-{index - 1}",
                    "previous_artifact_hash": f"artifact-{index - 1}",
                    "first_frame_source": "previous_tail",
                } if index > 1 else {})}},
            })
        store = task_store.default_store()
        store.register_jobs(ep_id, rows)
        episode = {"ep_id": ep_id, "panels": [
            {"panel_id": f"panel_{index:02d}"} for index in range(1, 7)
        ]}

        def reject_during_wait(current_job_id, **kwargs):
            self.assertEqual(current_job_id, job_ids[2])
            task_store.reject_job(
                ep_id, job_ids[1], reason="reviewer invalidated predecessor",
                cancel_job=lambda job: store.update_job(
                    str(job["job_id"]), status="cancelled", error="cancelled by reviewer",
                ),
            )
            raise RuntimeError("render job was invalidated while waiting")

        with mock.patch.object(orchestrator, "_load_episode", return_value=episode), \
             mock.patch.object(orchestrator, "prepare_episode"), \
             mock.patch.object(orchestrator, "production_gate", return_value={"ready": True, "reasons": []}), \
             mock.patch.object(orchestrator, "resume_jobs", return_value={"resumed": 0}), \
             mock.patch.object(orchestrator, "recover_render_job", side_effect=lambda job_id, **kwargs: store.get_job(job_id)), \
             mock.patch.object(orchestrator, "wait_render_job", side_effect=reject_during_wait) as wait, \
             mock.patch.object(orchestrator, "submit_render_job") as submit, \
             mock.patch.object(orchestrator, "update_status"):
            result = orchestrator.run_episode_jobs(ep_id, poll_interval=0.01)

        wait.assert_called_once()
        submit.assert_not_called()
        self.assertEqual(len(result["failures"]), 1)
        self.assertEqual([job["status"] for job in store.list_jobs(ep_id)], [
            "succeeded", "failed", "failed", "queued", "queued", "queued",
        ])

    def test_asset_worker_generates_scene_and_character_then_requires_approval(self):
        episode = self._episode()
        for item in episode["character_bible"] + episode["scene_bible"]:
            item["reference_images"] = []
        task_store.prepare_contract("ep_phase2", episode)
        task_store.approve_contract("ep_phase2")

        def character_generator(character, visual_bible, **kwargs):
            path = self._file(f"generated/{character['character_id']}.png", character["character_id"].encode())
            return {"prompt_id": f"char-{character['character_id']}", "reference_images": [path]}

        def scene_generator(scene, visual_bible, **kwargs):
            path = self._file(f"generated/{scene['scene_id']}.png", scene["scene_id"].encode())
            return {"prompt_id": f"scene-{scene['scene_id']}", "reference_images": [path]}

        snapshot = orchestrator.prepare_all_assets(
            "ep_phase2", character_generator=character_generator, scene_generator=scene_generator,
        )
        self.assertEqual(snapshot["pipeline"]["assets_status"], "ready_for_approval")
        self.assertTrue(all(asset["status"] == "succeeded" for asset in snapshot["assets"]["items"]))
        self.assertFalse(task_store.production_gate("ep_phase2")["ready"])
        approved = task_store.approve_assets("ep_phase2")
        self.assertTrue(task_store.production_gate("ep_phase2")["ready"])
        for job in approved["jobs"]:
            roles = {item["role"] for item in job["metadata"]["inputs"]["reference_inputs"]}
            self.assertIn("scene_reference", roles)
            self.assertIn("character_reference", roles)

    def test_worker_refuses_h3_before_approval(self):
        task_store.prepare_contract("ep_phase2", self._episode())
        with mock.patch.object(orchestrator, "submit_render_job") as submit:
            with self.assertRaisesRegex(RuntimeError, "production gate blocked"):
                orchestrator.run_episode_jobs("ep_phase2")
        submit.assert_not_called()

    def test_max_jobs_one_consumes_one_then_next_run_resumes_and_default_consumes_rest(self):
        self._approve(self._episode())
        submitted_calls: list[dict] = []

        def fake_submit(_panel, _output_path, **kwargs):
            submitted_calls.append(dict(kwargs))
            kwargs["store"].update_job(kwargs["job_id"], status="submitted", prompt_id=f"prompt-{kwargs['job_id']}")
            return {"job_id": kwargs["job_id"], "prompt_id": f"prompt-{kwargs['job_id']}"}

        def fake_wait(job_id, **kwargs):
            store = kwargs["store"]
            current = store.get_job(job_id)
            output = Path(current["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"clip:{job_id}".encode())
            store.update_job(job_id, status="succeeded", output_path=str(output), preview_path=str(output))
            return output

        common = (
            mock.patch.object(orchestrator, "submit_render_job", side_effect=fake_submit),
            mock.patch.object(orchestrator, "wait_render_job", side_effect=fake_wait),
            mock.patch.object(orchestrator, "_extract_tail_frame", side_effect=lambda _src, dst: Path(self._file(str(Path(dst).relative_to(self.base)), b"tail"))),
            mock.patch.object(orchestrator, "update_status"),
        )
        with common[0], common[1], common[2], common[3]:
            first = orchestrator.run_episode_jobs("ep_phase2", max_jobs=1, poll_interval=0.01)
        self.assertEqual(first["jobs_consumed"], 1)
        self.assertEqual([item["status"] for item in first["snapshot"]["jobs"]], ["succeeded", "queued", "queued"])

        with mock.patch.object(orchestrator, "submit_render_job", side_effect=fake_submit), \
             mock.patch.object(orchestrator, "wait_render_job", side_effect=fake_wait), \
             mock.patch.object(orchestrator, "_extract_tail_frame", side_effect=lambda _src, dst: Path(self._file(str(Path(dst).relative_to(self.base)), b"tail"))), \
             mock.patch.object(orchestrator, "update_status"):
            second = orchestrator.run_episode_jobs("ep_phase2", max_jobs=1, poll_interval=0.01)
        self.assertEqual(second["jobs_consumed"], 1)
        self.assertEqual([item["status"] for item in second["snapshot"]["jobs"]], ["succeeded", "succeeded", "queued"])
        continued = next(call for call in submitted_calls if ":0002:" in call["job_id"])
        self.assertFalse(continued["composition_anchor_first"])
        self.assertTrue(Path(continued["first_frame"]).is_file())
        self.assertTrue(Path(continued["character_anchor"]).is_file())
        self.assertTrue(any("scene" in path.stem for path in continued["char_refs"]))

        with mock.patch.object(orchestrator, "submit_render_job", side_effect=fake_submit), \
             mock.patch.object(orchestrator, "wait_render_job", side_effect=fake_wait), \
             mock.patch.object(orchestrator, "_extract_tail_frame", side_effect=lambda _src, dst: Path(self._file(str(Path(dst).relative_to(self.base)), b"tail"))), \
             mock.patch.object(orchestrator, "update_status"):
            final = orchestrator.run_episode_jobs("ep_phase2", poll_interval=0.01)
        self.assertIsNone(final["max_jobs"])
        self.assertEqual([item["status"] for item in final["snapshot"]["jobs"]], ["succeeded"] * 3)

    def test_failed_continuity_chain_does_not_block_independent_panel_recovery(self):
        episode = self._episode()
        self._approve(episode)
        submitted: list[str] = []

        def fake_submit(panel, output_path, **kwargs):
            job_id = kwargs["job_id"]
            if ":0001:" in job_id:
                raise RuntimeError("mock first-panel failure")
            submitted.append(job_id)
            kwargs["store"].update_job(job_id, status="submitted", prompt_id=f"prompt-{job_id}")
            return {"job_id": job_id, "prompt_id": f"prompt-{job_id}"}

        def fake_wait(job_id, **kwargs):
            store = kwargs["store"]
            job = store.get_job(job_id)
            output = Path(job["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"recovered-independent-panel")
            store.update_job(
                job_id, status="succeeded", output_path=str(output), preview_path=str(output),
                probe={"duration_seconds": 4.0, "video": {"width": 1920, "height": 1080, "fps": 24.0}},
                metadata={**job["metadata"], "artifact_sha256": "independent-artifact"},
            )
            return output

        with mock.patch.object(orchestrator, "submit_render_job", side_effect=fake_submit), \
             mock.patch.object(orchestrator, "wait_render_job", side_effect=fake_wait), \
             mock.patch.object(orchestrator, "update_status"):
            result = orchestrator.run_episode_jobs("ep_phase2", poll_interval=0.01)
        statuses = [job["status"] for job in result["snapshot"]["jobs"]]
        self.assertEqual(statuses, ["failed", "queued", "succeeded"])
        self.assertEqual(len(submitted), 1)
        self.assertIn(":0003:", submitted[0])

    def test_post_render_episode_refresh_failure_preserves_validated_success_and_chain(self):
        episode = self._episode()
        episode["panels"][2]["continuity_group"] = "station_chain"
        self._approve(episode)
        store = task_store.default_store()
        jobs = store.list_jobs("ep_phase2")
        first_output = Path(jobs[0]["output_path"])
        first_output.parent.mkdir(parents=True, exist_ok=True)
        first_output.write_bytes(b"accepted-panel-one")
        first_visual = "1" * 64
        store.update_job(
            jobs[0]["job_id"], status="succeeded", progress=1.0,
            output_path=str(first_output), preview_path=str(first_output),
            probe={
                "duration_seconds": 4.0,
                "video": {"width": 1920, "height": 1080, "fps": 24.0},
            },
            metadata={
                **jobs[0]["metadata"], "artifact_sha256": "artifact-1",
                "content_qa": {"passed": True, "analysis": {
                    "decoded_visual_sha256": first_visual, "static": False,
                    "perceptual_hashes": [first_visual],
                }},
            },
        )
        prepared = task_store.prepare_episode
        prepare_calls = 0

        def fail_only_post_render(ep_id, payload):
            nonlocal prepare_calls
            prepare_calls += 1
            if prepare_calls == 1:
                return prepared(ep_id, payload)
            raise PermissionError(5, "simulated locked episode.json after validated render")

        def fake_submit(panel, output_path, **kwargs):
            job_id = kwargs["job_id"]
            kwargs["store"].update_job(
                job_id, status="submitted", prompt_id=f"prompt-{job_id}", progress=0.1,
            )
            return {"job_id": job_id, "prompt_id": f"prompt-{job_id}"}

        def fake_wait(job_id, **kwargs):
            current = kwargs["store"].get_job(job_id)
            output = Path(current["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(f"validated-{job_id}".encode("utf-8"))
            visual = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
            kwargs["store"].update_job(
                job_id, status="succeeded", progress=1.0,
                output_path=str(output), preview_path=str(output), error=None,
                probe={
                    "duration_seconds": 4.0,
                    "video": {"width": 1920, "height": 1080, "fps": 24.0},
                },
                metadata={
                    **current["metadata"], "artifact_sha256": f"artifact-{job_id}",
                    "content_qa": {"passed": True, "analysis": {
                        "decoded_visual_sha256": visual, "static": False,
                        "perceptual_hashes": [visual],
                    }},
                },
            )
            return output

        def fake_tail(video, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"continuity-tail")
            return destination

        with mock.patch.object(orchestrator, "prepare_episode", side_effect=fail_only_post_render), \
             mock.patch.object(orchestrator, "submit_render_job", side_effect=fake_submit), \
             mock.patch.object(orchestrator, "wait_render_job", side_effect=fake_wait), \
             mock.patch.object(orchestrator, "_extract_tail_frame", side_effect=fake_tail), \
             mock.patch.object(orchestrator, "update_status"):
            result = orchestrator.run_episode_jobs("ep_phase2", poll_interval=0.01)

        self.assertEqual(result["failures"], [])
        self.assertEqual(len(result["warnings"]), 2)
        self.assertEqual([job["status"] for job in result["snapshot"]["jobs"]], [
            "succeeded", "succeeded", "succeeded",
        ])
        for current in result["snapshot"]["jobs"][1:]:
            self.assertIsNone(current["error"])
            warning = current["metadata"]["pipeline_warnings"][-1]
            self.assertEqual(warning["stage"], "post_render_episode_refresh")
            self.assertTrue(warning["artifact_preserved"])
            self.assertIn("locked episode.json", warning["error"])

    def test_continuity_safe_chain_requires_approved_anchor_and_commits_only_after_probe(self):
        episode = self._episode()
        episode["panels"][2]["continuity_group"] = "station_chain"
        self._approve(episode)
        store = task_store.default_store()
        jobs = store.list_jobs("ep_phase2")
        first_output = Path(jobs[0]["output_path"])
        first_output.parent.mkdir(parents=True, exist_ok=True)
        first_output.write_bytes(b"accepted-panel-one")
        store.update_job(
            jobs[0]["job_id"], status="succeeded", output_path=str(first_output),
            preview_path=str(first_output),
            probe={
                "duration_seconds": 4.0,
                "video": {"width": 608, "height": 1056, "fps": 24.0},
            },
            metadata={**jobs[0]["metadata"], "artifact_sha256": "accepted-one"},
        )
        store.update_job(jobs[1]["job_id"], status="failed", error="H3 rejected by reviewer")
        anchor = self._png("projects/ep_phase2/continuity/approved_group.png")
        with self.assertRaisesRegex(RuntimeError, "approved visual-state anchor"):
            continuity_safe.run_continuity_safe_chain(jobs[1]["ep_id"], jobs[1]["job_id"])
        approval = render_service.approve_continuity_anchor(
            "ep_phase2", jobs[1]["job_id"], anchor,
            reason="five-person composition manually accepted", approved_by="QA",
        )
        self.assertEqual(approval["anchor_approval"]["sha256"], continuity_safe._sha256_file(anchor))

        ffmpeg_commands: list[list[str]] = []
        tts_calls: list[dict] = []

        def fake_tts(text, output_path, **kwargs):
            Path(output_path).write_bytes(b"mock-wave")
            audit = {
                "engine": "windows_sapi",
                "requested_voice": kwargs["preferred_voice"],
                "selected_voice": "Microsoft Huihui Desktop",
                "selected_culture": "zh-CN",
                "fallback_used": kwargs["preferred_voice"] != "Microsoft Huihui Desktop",
                "voice_fidelity": "single_system_voice_not_character_voice_cloning",
            }
            tts_calls.append({"text": text, **audit})
            return audit

        def fake_runner(command, **kwargs):
            ffmpeg_commands.append(command)
            Path(command[-1]).write_bytes(b"validated-continuity-safe-mp4")
            return mock.Mock(returncode=0)

        def fake_probe(path, **kwargs):
            return {
                "path": str(path), "size_bytes": 30,
                "duration_seconds": continuity_safe.SAFE_DURATION_SECONDS,
                "video": {
                    "codec": "h264", "width": 608, "height": 1056,
                    "fps": 24.0, "pixel_format": "yuv420p",
                },
                "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2},
            }

        result = render_service.run_continuity_safe(
            "ep_phase2", jobs[1]["job_id"],
            preferred_voice="Unavailable Character Voice",
            # Legacy Web behavior is accepted, but must never burn a panel clip.
            burn_subtitles=True,
            runner=fake_runner, probe_func=fake_probe,
            tts_func=fake_tts,
            audio_probe_func=lambda path, **kwargs: 1.139,
            quality_analyzer=lambda path, **kwargs: {
                "decoded_visual_sha256": ("b" if ".partial." in str(path) else "a") * 64,
                "perceptual_hashes": [(("b" if ".partial." in str(path) else "a") * 64)],
                "static": False, "metrics": {"sample_count": 2},
                "algorithm": {"name": "test", "version": "1"},
            },
            ffmpeg="ffmpeg", ffprobe="ffprobe",
        )
        self.assertFalse(result["stopped"])
        self.assertEqual(result["failure"], None)
        self.assertEqual(result["completed_job_ids"], [jobs[1]["job_id"]])
        self.assertEqual([job["status"] for job in result["snapshot"]["jobs"]], [
            "succeeded", "succeeded", "queued",
        ])
        self.assertEqual(len(ffmpeg_commands), 1)
        first_filter = ffmpeg_commands[0][ffmpeg_commands[0].index("-filter_complex") + 1]
        self.assertIn("zoompan=", first_filter)
        self.assertIn("anoisesrc=color=pink", " ".join(ffmpeg_commands[0]))
        self.assertNotIn("subtitles=", first_filter)
        self.assertEqual(len(tts_calls), 1)
        self.assertTrue(tts_calls[0]["fallback_used"])
        accepted = result["snapshot"]["jobs"][1]
        self.assertEqual(accepted["metadata"]["render_mode"], "continuity_safe")
        safe_metadata = accepted["metadata"]["continuity_safe"]
        self.assertEqual(safe_metadata["source_anchor"], str(anchor.resolve()))
        self.assertEqual(safe_metadata["tts_engine"], "windows_sapi")
        self.assertEqual(safe_metadata["tts_voice"], "Microsoft Huihui Desktop")
        self.assertTrue(safe_metadata["voice_fallback_used"])
        self.assertEqual(accepted["probe"]["duration_seconds"], 10.125)
        manifest = json.loads(Path(safe_metadata["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["voice_fidelity"], "single_system_voice_not_character_voice_cloning")
        self.assertEqual(manifest["ambient_bed"], "deterministic_ffmpeg_pink_noise")
        self.assertFalse(manifest["subtitles"]["burned_in"])
        self.assertEqual(manifest["subtitles"]["burn_policy"], "delivery_only")
        self.assertTrue(manifest["subtitles"]["legacy_burn_request_ignored"])
        for field in ("srt_path", "vtt_path", "ass_path"):
            self.assertTrue(Path(manifest["subtitles"][field]).is_file())

    def test_continuity_safe_chain_stops_on_failure_and_leaves_later_job_queued(self):
        episode = self._episode()
        episode["panels"][2]["continuity_group"] = "station_chain"
        panel_four = json.loads(json.dumps(episode["panels"][2]))
        panel_four["panel_id"] = "panel_04"
        panel_four["spoken_dialogue"] = []
        episode["panels"].append(panel_four)
        episode["scene_bible"][1]["panel_ids"].append("panel_04")
        self._approve(episode)
        store = task_store.default_store()
        jobs = store.list_jobs("ep_phase2")
        first_output = Path(jobs[0]["output_path"])
        first_output.parent.mkdir(parents=True, exist_ok=True)
        first_output.write_bytes(b"accepted-panel-one")
        store.update_job(
            jobs[0]["job_id"], status="succeeded", output_path=str(first_output),
            preview_path=str(first_output),
            probe={
                "duration_seconds": 4.0,
                "video": {"width": 608, "height": 1056, "fps": 24.0},
            },
            metadata={**jobs[0]["metadata"], "artifact_sha256": "accepted-one"},
        )
        store.update_job(jobs[1]["job_id"], status="failed", error="rejected H3 clip")
        anchor = self._png("projects/ep_phase2/continuity/approved_chain.png")
        render_service.approve_continuity_anchor(
            "ep_phase2", jobs[1]["job_id"], anchor,
            reason="approved fallback anchor", approved_by="QA",
        )
        command_count = 0

        def fail_second_command(command, **kwargs):
            nonlocal command_count
            command_count += 1
            if command_count == 2:
                raise RuntimeError("simulated FFmpeg failure on downstream panel")
            Path(command[-1]).write_bytes(b"first-safe-output")
            return mock.Mock(returncode=0)

        def fake_tts(text, output_path, **kwargs):
            Path(output_path).write_bytes(b"wave")
            return {
                "engine": "windows_sapi", "selected_voice": "Microsoft Huihui Desktop",
                "selected_culture": "zh-CN", "requested_voice": kwargs["preferred_voice"],
                "fallback_used": False,
                "voice_fidelity": "single_system_voice_not_character_voice_cloning",
            }

        probe = lambda path, **kwargs: {
            "path": str(path), "duration_seconds": 10.125,
            "video": {
                "codec": "h264", "width": 608, "height": 1056,
                "fps": 24.0, "pixel_format": "yuv420p",
            },
            "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2},
        }
        result = continuity_safe.run_continuity_safe_chain(
            "ep_phase2", jobs[1]["job_id"],
            runner=fail_second_command, probe_func=probe,
            tts_func=fake_tts, audio_probe_func=lambda path, **kwargs: 1.0,
            quality_analyzer=lambda path, **kwargs: {
                "decoded_visual_sha256": ("c" if ".partial." in str(path) else "d") * 64,
                "perceptual_hashes": [(("c" if ".partial." in str(path) else "d") * 64)],
                "static": False, "metrics": {"sample_count": 2},
                "algorithm": {"name": "test", "version": "1"},
            },
            ffmpeg="ffmpeg", ffprobe="ffprobe",
        )
        self.assertFalse(result["stopped"])
        self.assertIsNone(result["failure"])
        self.assertEqual(command_count, 1)
        self.assertEqual([job["status"] for job in result["snapshot"]["jobs"]], [
            "succeeded", "succeeded", "queued", "queued",
        ])
        self.assertEqual(result["chain_job_ids"], [jobs[1]["job_id"]])
        self.assertNotIn(
            "continuity_safe_anchor_approval",
            result["snapshot"]["jobs"][2]["metadata"],
        )

    def test_decoded_visual_qa_blocks_same_video_with_different_audio(self):
        ffmpeg = video_delivery.ffmpeg_executable()
        source = self.base / "quality-source.mp4"
        first = self.base / "quality-audio-440.mp4"
        second = self.base / "quality-audio-880.mp4"
        subprocess.run([
            ffmpeg, "-y", "-f", "lavfi", "-i",
            "testsrc2=size=160x160:rate=12:duration=2", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ], check=True, capture_output=True)
        for frequency, destination in ((440, first), (880, second)):
            subprocess.run([
                ffmpeg, "-y", "-i", str(source), "-f", "lavfi", "-i",
                f"sine=frequency={frequency}:sample_rate=48000:duration=2",
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-shortest", str(destination),
            ], check=True, capture_output=True)
        self.assertNotEqual(hashlib.sha256(first.read_bytes()).hexdigest(),
                            hashlib.sha256(second.read_bytes()).hexdigest())
        left = video_quality.analyze_video(first, ffmpeg=ffmpeg)
        right = video_quality.analyze_video(second, ffmpeg=ffmpeg)
        self.assertEqual(left["decoded_visual_sha256"], right["decoded_visual_sha256"])
        comparison = video_quality.compare_analyses(left, right)
        self.assertTrue(comparison["exact_duplicate"])
        evaluation = video_quality.evaluate_content(right, [("panel_01", left)])
        self.assertFalse(evaluation["passed"])
        self.assertIn("exact_visual_duplicate:panel_01", evaluation["reasons"])

    def test_release_facade_is_hash_bound_and_revocation_keeps_files(self):
        episode = self._with_edit_plan(self._episode())
        episode.setdefault("render_settings", {})["production_strategy"] = "direct_production"
        for panel in episode["panels"]:
            panel.setdefault("prompt_package", {}).setdefault("render_settings", {})[
                "production_strategy"
            ] = "direct_production"
        self._approve(episode)
        jobs = self._mark_all_success()
        expected = {}
        for job in jobs:
            output = Path(job["output_path"])
            artifact = hashlib.sha256(output.read_bytes()).hexdigest()
            expected[job["job_id"]] = artifact
            render_service.approve_job_review(
                "ep_phase2", job["job_id"], expected_artifact_sha256=artifact,
                expected_edit_selection_sha256=job["metadata"]["edit_selection"]["selection_sha256"],
                reviewed_by="editor", reason="shot tells the approved beat",
            )
        approved = render_service.approve_episode_release(
            "ep_phase2", expected_artifact_hashes=expected,
            expected_edit_selection_hashes={
                job["job_id"]: job["metadata"]["edit_selection"]["selection_sha256"]
                for job in task_store.list_jobs("ep_phase2")
            },
            approved_by="publisher", reason="all shots editorially accepted",
        )
        self.assertEqual(approved["pipeline"]["release_status"], "approved")
        before = {job["job_id"]: Path(job["output_path"]) for job in approved["jobs"]}
        revoked = render_service.revoke_release(
            "ep_phase2", reason="cross-shot duplicate discovered", revoked_by="QA",
        )
        self.assertEqual(revoked["pipeline"]["release_status"], "revoked")
        self.assertTrue(all(path.is_file() for path in before.values()))
        self.assertTrue(all(job["status"] == "succeeded" for job in revoked["jobs"]))
        self.assertTrue(all(job["metadata"]["release"]["status"] == "revoked"
                            for job in revoked["jobs"]))

    def test_windows_sapi_audits_actual_fallback_and_cleans_request_files(self):
        output = self.base / "tts" / "line.wav"

        def fake_powershell(command, **kwargs):
            request_path = Path(kwargs["env"]["AI_MANGA_TTS_REQUEST"])
            response_path = Path(kwargs["env"]["AI_MANGA_TTS_RESPONSE"])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(request["rate"], 8)
            self.assertEqual(request["preferred_voice"], "Missing Voice")
            Path(request["output_path"]).write_bytes(b"RIFF-mock-wave")
            response_path.write_text(json.dumps({
                "engine": "windows_sapi",
                "selected_voice": "Microsoft Huihui Desktop",
                "selected_culture": "zh-CN",
                "requested_voice": "Missing Voice",
                "fallback_used": True,
                "available_voices": [{"name": "Microsoft Huihui Desktop", "culture": "zh-CN"}],
                "voice_fidelity": "single_system_voice_not_character_voice_cloning",
            }), encoding="utf-8")
            return mock.Mock(returncode=0)

        audit = continuity_safe.synthesize_windows_sapi(
            "鍑嗗濂戒簡", output, preferred_voice="Missing Voice",
            powershell="powershell.exe", runner=fake_powershell,
        )
        self.assertTrue(output.is_file())
        self.assertEqual(audit["selected_voice"], "Microsoft Huihui Desktop")
        self.assertTrue(audit["fallback_used"])
        self.assertEqual(list(output.parent.glob(".*.request.json")), [])
        self.assertEqual(list(output.parent.glob(".*.response.json")), [])

    def test_scene_reference_graph_is_deterministic_auditable_and_people_free(self):
        scene = {
            **self._episode()["scene_bible"][0],
            "positive_prompt": "rain-lit station, fantasy woman in qipao, amber platform lights, signage reading CENTRAL",
            "model_prompt_en": (
                "single empty abandoned train station platform at rainy night, "
                "cold blue lamps, wet reflective puddles, closed iron gate"
            ),
        }
        graph_a, manifest_a = scene_asset.build_scene_reference_workflow(
            scene, {"style_prompt": "ink animation", "aspect_ratio": "16:9"}, story_hash="story"
        )
        graph_b, manifest_b = scene_asset.build_scene_reference_workflow(
            scene, {"style_prompt": "ink animation", "aspect_ratio": "16:9"}, story_hash="story"
        )
        self.assertEqual(manifest_a["seed"], manifest_b["seed"])
        self.assertEqual(graph_a["5"]["inputs"]["seed"], graph_b["5"]["inputs"]["seed"])
        self.assertIn("no people", manifest_a["positive_prompt"])
        self.assertIn("single wide environment view", manifest_a["positive_prompt"])
        self.assertIn("abandoned train station platform", manifest_a["positive_prompt"])
        self.assertNotIn("rain-lit station", manifest_a["positive_prompt"])
        self.assertNotIn("fantasy woman", manifest_a["positive_prompt"])
        self.assertNotIn("signage reading CENTRAL", manifest_a["positive_prompt"])
        self.assertIn("split screen", manifest_a["negative_prompt"])
        self.assertIn("collage", manifest_a["negative_prompt"])
        self.assertIn("diptych", manifest_a["negative_prompt"])
        self.assertIn("multiple views", manifest_a["negative_prompt"])
        self.assertIn("signage", manifest_a["negative_prompt"])
        self.assertIn("random text", manifest_a["negative_prompt"])
        self.assertEqual(manifest_a["text_policy"], "no_text_model_output")
        self.assertEqual(graph_a["2"]["inputs"]["text"], manifest_a["model_positive_prompt"])
        self.assertEqual(graph_a["3"]["inputs"]["text"], manifest_a["model_negative_prompt"])
        self.assertEqual((manifest_a["width"], manifest_a["height"]), (1344, 768))

    def test_convenience_store_scene_uses_hard_retail_layout_and_official_animagine_profile(self):
        scene = {
            "scene_id": "scene_store",
            "description": "雨夜便利店",
            "model_prompt_en": (
                "grounded Japanese convenience store interior, rainy night through glass doors, "
                "warm interior light, wet reflections, donation box near checkout counter"
            ),
        }
        graph, manifest = scene_asset.build_scene_reference_workflow(
            scene,
            {
                "style_prompt": "modern Japanese anime background art, clean lineart",
                "global_negative_prompt": "identity drift, random text, logo",
                "aspect_ratio": "9:16",
            },
            story_hash="store-story",
        )
        self.assertTrue(manifest["convenience_store_lock"])
        for required in ("checkout counter", "store shelves", "glass entrance doors", "donation box"):
            self.assertIn(required, manifest["positive_prompt"])
        for forbidden in ("corridor", "hallway", "kiosk", "missing checkout counter"):
            self.assertIn(forbidden, manifest["negative_prompt"])
        self.assertEqual((manifest["width"], manifest["height"]), (768, 1344))
        self.assertEqual(graph["5"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(graph["5"]["inputs"]["steps"], 28)
        self.assertEqual(graph["5"]["inputs"]["cfg"], 6.0)

    def test_convenience_store_can_use_realvis_structure_then_animagine_stylization(self):
        scene = {
            "scene_id": "scene_store",
            "description": "雨夜便利店",
            "model_prompt_en": (
                "dry convenience store interior, checkout counter foreground, transparent charity box, "
                "plain hot drink cup, glass doors showing blue rainy night outside"
            ),
            "negative_prompt": "text, logos, rain inside, wet floor",
        }
        graph, manifest = scene_asset.build_scene_reference_workflow(
            scene,
            {
                "style_prompt": "modern Japanese anime background art, clean lineart",
                "global_negative_prompt": "random text, logo",
                "aspect_ratio": "9:16",
            },
            story_hash="store-story",
            structural_checkpoint=scene_asset.STRUCTURAL_SCENE_CHECKPOINT,
        )
        self.assertTrue(manifest["two_pass_structure"])
        self.assertEqual(graph["1"]["inputs"]["ckpt_name"], scene_asset.STRUCTURAL_SCENE_CHECKPOINT)
        self.assertEqual(graph["7"]["inputs"]["ckpt_name"], scene_asset.DEFAULT_CHECKPOINT)
        self.assertEqual(graph["10"]["class_type"], "VAEEncode")
        self.assertEqual(graph["11"]["inputs"]["denoise"], 0.62)
        self.assertEqual(graph["13"]["class_type"], "SaveImage")
        self.assertIn("donation box", manifest["structural_positive_prompt"])
        self.assertIn("menu board", manifest["structural_negative_prompt"])
        self.assertIn("rainy street", manifest["structural_positive_prompt"])
        self.assertNotIn("anime", manifest["structural_positive_prompt"].lower())
        self.assertIn("completely dry interior and dry floor", manifest["positive_prompt"])
        self.assertIn("wet floor", manifest["negative_prompt"])

    def test_convenience_store_layout_control_overrides_two_pass_structure(self):
        scene = {
            "scene_id": "scene_store",
            "description": "雨夜便利店",
            "model_prompt_en": "convenience store, checkout counter, glass doors, rainy street",
            "negative_prompt": "text, logo, rain inside",
        }
        graph, manifest = scene_asset.build_scene_reference_workflow(
            scene,
            {
                "style_prompt": "modern Japanese anime background art, clean lineart",
                "global_negative_prompt": "random text, logo",
                "aspect_ratio": "9:16",
            },
            story_hash="store-layout-story",
            layout_image_name="scene_layouts/store.png",
            structural_checkpoint=scene_asset.STRUCTURAL_SCENE_CHECKPOINT,
        )
        self.assertFalse(manifest["two_pass_structure"])
        self.assertEqual(manifest["environment_profile"], "convenience_store_layout")
        self.assertEqual(graph["1"]["inputs"]["ckpt_name"], scene_asset.SOCIAL_LAYOUT_CHECKPOINT)
        self.assertEqual(graph["5"]["inputs"]["control_net_name"], scene_asset.SOCIAL_LAYOUT_CONTROLNET)
        self.assertEqual(graph["6"]["inputs"]["strength"], scene_asset.CONVENIENCE_LAYOUT_STRENGTH)
        self.assertEqual(graph["8"]["class_type"], "KSampler")
        self.assertEqual(graph["8"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(graph["10"]["class_type"], "SaveImage")
        self.assertIn("transparent glass double entrance", manifest["positive_prompt"])
        self.assertIn("refrigerator", manifest["negative_prompt"])
        self.assertLess(len(manifest["positive_prompt"]), 700)

    def test_compact_game_room_uses_social_layout_and_blocks_phone_shop_display(self):
        scene = {
            "scene_id": "scene_game_room",
            "description": "一个只有3平方米的狭窄房间，桌上整齐放着五台手机和平板电脑",
            "model_prompt_en": (
                "small room, 3 square meters, one ordinary desk, exactly 1 phone at each place, "
                "five mobile phones and tablets arranged in one straight row on the table, "
                "phone shop display cabinet, top-down view, city noise and office quiet atmosphere"
            ),
        }
        graph, manifest = scene_asset.build_scene_reference_workflow(
            scene,
            {
                "style_prompt": "anime screencap, modern urban 2D cel animation",
                "aspect_ratio": "9:16",
            },
            story_hash="compact-story",
            layout_image_name="scene_layouts/scene_game_room_test.png",
        )
        self.assertTrue(manifest["compact_interior_lock"])
        self.assertEqual(manifest["environment_profile"], "social_mobile_gaming_room")
        self.assertEqual(manifest["seed_material_version"], 6)
        self.assertEqual(manifest["required_device_count"], 5)
        self.assertIn("one single large ordinary rectangular shared gaming table dominates the center foreground", manifest["positive_prompt"])
        self.assertIn("complete tabletop is clearly visible", manifest["positive_prompt"])
        self.assertIn("naturally prepared for a group of 5 close friends", manifest["positive_prompt"])
        self.assertIn("ordinary home chairs around the shared table", manifest["positive_prompt"])
        self.assertIn("a few small black-screen smartphones resting naturally", manifest["positive_prompt"])
        self.assertNotIn("exactly 1", manifest["positive_prompt"])
        self.assertIn("ordinary contemporary private living room", manifest["positive_prompt"])
        self.assertIn("single human eye-level wide shot", manifest["positive_prompt"])
        self.assertIn("table seen obliquely in perspective", manifest["positive_prompt"])
        self.assertIn("empty environment, no people, no humans, no characters", manifest["positive_prompt"])
        self.assertNotIn("five people", manifest["positive_prompt"])
        self.assertNotIn("one straight row", manifest["positive_prompt"])
        self.assertNotIn("phone shop display cabinet", manifest["positive_prompt"])
        self.assertNotIn("top-down view", manifest["positive_prompt"])
        self.assertNotIn("city noise", manifest["positive_prompt"])
        self.assertNotIn("quiet atmosphere", manifest["positive_prompt"])
        self.assertIn("near side walls and back wall visibly enclose", manifest["positive_prompt"])
        self.assertIn("camera at room entrance about 1.5 meters above the floor", manifest["positive_prompt"])
        self.assertNotIn("deep perspective", manifest["composition_policy"])
        for forbidden in (
            "vast hall", "warehouse", "empty office floor", "ceiling-dominant view",
            "overhead view", "top-down view", "empty bare room",
            "classroom", "rows of desks", "multiple tables",
            "dark room", "traditional wooden room", "long narrow tunnel",
            "retail store", "showroom", "display cabinet", "drawer", "shelf",
            "phone shop", "product display", "glass display case", "product tray",
        ):
            self.assertIn(forbidden, manifest["negative_prompt"])
        self.assertEqual(graph["2"]["inputs"]["text"], manifest["model_positive_prompt"])
        self.assertEqual(graph["3"]["inputs"]["text"], manifest["model_negative_prompt"])
        self.assertEqual(graph["1"]["inputs"]["ckpt_name"], scene_asset.SOCIAL_LAYOUT_CHECKPOINT)
        self.assertEqual(graph["4"]["class_type"], "LoadImage")
        self.assertEqual(graph["4"]["inputs"]["image"], "scene_layouts/scene_game_room_test.png")
        self.assertEqual(graph["5"]["class_type"], "ControlNetLoader")
        self.assertEqual(graph["6"]["class_type"], "ControlNetApplyAdvanced")
        self.assertEqual(graph["6"]["inputs"]["strength"], scene_asset.SOCIAL_LAYOUT_STRENGTH)
        self.assertEqual(graph["6"]["inputs"]["end_percent"], scene_asset.SOCIAL_LAYOUT_END_PERCENT)
        self.assertEqual(graph["8"]["inputs"]["positive"], ["6", 0])
        self.assertEqual(graph["8"]["inputs"]["negative"], ["6", 1])
        self.assertEqual(manifest["layout_conditioning"], "sd15_lineart_controlnet")

    def test_rejected_scene_gets_a_new_deterministic_seed(self):
        scene = {
            "scene_id": "scene_retry",
            "description": "small gaming room with five phones on a shared table",
            "model_prompt_en": "ordinary small gaming room, five smartphones on one shared table",
        }
        visual = {"style_prompt": "anime background", "aspect_ratio": "9:16"}
        _, first = scene_asset.build_scene_reference_workflow(scene, visual, story_hash="story")
        scene["asset_rejection_history"] = [{"reason": "bad composition"}]
        _, retry = scene_asset.build_scene_reference_workflow(scene, visual, story_hash="story")
        self.assertNotEqual(first["seed"], retry["seed"])

    def test_select_existing_scene_preserves_contract_and_unrelated_assets(self):
        episode = self._episode()
        approved = self._approve(episode)
        contract_hash = approved["pipeline"]["contract_hash"]
        replacement = self._file("selected/better_scene.png", b"better-scene")
        snapshot = task_store.select_asset_references(
            "ep_phase2", "ep_phase2:scene:scene_station", [replacement],
        )
        self.assertEqual(snapshot["pipeline"]["contract_hash"], contract_hash)
        self.assertEqual(snapshot["pipeline"]["contract_status"], "approved")
        assets = {(item["asset_type"], item["source_id"]): item for item in snapshot["assets"]["items"]}
        self.assertEqual(assets[("scene", "scene_station")]["reference_images"], [str(Path(replacement).resolve())])
        self.assertFalse(assets[("scene", "scene_station")]["approved"])
        self.assertTrue(assets[("character", "char_a")]["approved"])
        self.assertTrue(assets[("scene", "scene_roof")]["approved"])
        self.assertTrue(assets[("scene", "scene_station")]["metadata"]["selection_audit"])

    def test_scene_history_reads_controlnet_save_node_without_hardcoded_id(self):
        graph = {
            "7": {"class_type": "VAEDecode", "inputs": {}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0]}},
        }
        result = {
            "outputs": {
                "10": {"images": [{"filename": "scene.png", "subfolder": "sceneref", "type": "output"}]}
            }
        }
        self.assertEqual(scene_asset._history_output_images(result, graph)[0]["filename"], "scene.png")

    def test_subtitles_derive_only_from_spoken_dialogue_and_block_mismatch(self):
        episode = self._episode()
        self._approve(episode)
        jobs = self._mark_all_success()
        timeline = subtitle_delivery.build_episode_cues(episode, jobs)
        self.assertEqual([cue["text"] for cue in timeline["cues"]], ["有人吗？", "谁在那里？"])
        self.assertEqual(timeline["cues"][1]["start_seconds"], 4.25)
        self.assertNotIn("第七码头", [cue["text"] for cue in timeline["cues"]])
        bundle = subtitle_delivery.write_subtitle_bundle(episode, jobs, self.base / "delivery/final")
        self.assertIn("00:00:00,500 --> 00:00:01,500", Path(bundle["srt_path"]).read_text(encoding="utf-8-sig"))
        self.assertIn("WEBVTT", Path(bundle["vtt_path"]).read_text(encoding="utf-8-sig"))
        default_ass = Path(bundle["ass_path"]).read_text(encoding="utf-8-sig")
        self.assertIn("[Events]", default_ass)
        self.assertIn("PlayResX: 1280", default_ass)
        self.assertIn("PlayResY: 720", default_ass)
        self.assertEqual(
            bundle["subtitle_canvas"],
            {"play_res_x": 1280, "play_res_y": 720, "safe_margin_bottom_px": 72},
        )
        episode["panels"][0]["subtitle_timeline"] = [{"start_s": 0.5, "end_s": 1.5, "text": "篡改台词"}]
        with self.assertRaisesRegex(ValueError, "conflicts"):
            subtitle_delivery.build_episode_cues(episode, jobs)

    def test_delivery_gate_rejects_failure_and_optional_burn_uses_ass(self):
        episode = self._with_edit_plan(self._episode())
        self._approve(episode)
        jobs = self._mark_all_success()
        store = task_store.default_store()
        store.update_job(jobs[1]["job_id"], status="failed", error="mock failure")
        with self.assertRaisesRegex(RuntimeError, "incomplete panel jobs"):
            video_delivery.export_episode("ep_phase2", "vertical_9_16", ffmpeg="ffmpeg")
        store.update_job(
            jobs[1]["job_id"], status="succeeded",
            probe={"duration_seconds": 4.0, "video": {"width": 1920, "height": 1080, "fps": 24.0}},
        )
        commands = []

        def runner(command, **kwargs):
            commands.append(command)
            Path(command[-1]).write_bytes(b"final")
            return mock.Mock(returncode=0)

        def probe(path, **kwargs):
            return {
                "path": str(path), "size_bytes": 5, "duration_seconds": 12.0,
                "video": {"codec": "h264", "width": 720, "height": 1280, "fps": 30.0, "pixel_format": "yuv420p"},
                "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2},
            }

        manifest = video_delivery.export_episode(
            # Final delivery is now the default and sole subtitle burn stage.
            "ep_phase2", "vertical_9_16",
            runner=runner, probe_func=probe, ffmpeg="ffmpeg", ffprobe="ffprobe",
            quality_analyzer=self._quality_analysis,
        )
        vf = commands[0][commands[0].index("-filter_complex") + 1]
        self.assertIn("subtitles=", vf)
        self.assertIn("trim=start=", vf)
        self.assertTrue(manifest["subtitles"]["burned_in"])
        self.assertEqual(manifest["subtitles"]["burn_stage"], "final_delivery")
        self.assertEqual(manifest["subtitles"]["legacy_preburned_job_ids"], [])
        vertical_ass = Path(manifest["subtitles"]["ass_path"]).read_text(encoding="utf-8-sig")
        self.assertIn("PlayResX: 720", vertical_ass)
        self.assertIn("PlayResY: 1280", vertical_ass)
        vertical_style = next(line for line in vertical_ass.splitlines() if line.startswith("Style: Default"))
        self.assertEqual(int(vertical_style.split(",")[-2]), 256)
        self.assertEqual(
            manifest["subtitles"]["canvas"],
            {"play_res_x": 720, "play_res_y": 1280, "safe_margin_bottom_px": 256},
        )
        self.assertEqual(manifest["subtitles"]["safe_margin_bottom_px"], 256)
        self.assertEqual(manifest["preset"]["delivery_standard"], "720p-v1")
        self.assertEqual(manifest["preset"]["video_bitrate"], "5M")
        self.assertEqual(manifest["release_status"], "approved")
        self.assertEqual(set(manifest["approved_artifact_hashes"]), {job["job_id"] for job in jobs})
        self.assertEqual(set(manifest["approved_visual_hashes"]), {job["job_id"] for job in jobs})
        self.assertEqual(manifest["selected_duration_seconds"], 12.0)
        self.assertEqual(manifest["target_edit_duration_seconds"], 12.0)
        self.assertTrue(Path(manifest["qa_report_path"]).is_file())
        self.assertEqual(
            hashlib.sha256(Path(manifest["qa_report_path"]).read_bytes()).hexdigest(),
            manifest["qa_report_sha256"],
        )
        refreshed = task_store.project_snapshot("ep_phase2")
        current_report = next(
            report for report in refreshed["delivery_reports"]
            if report["manifest_path"] == manifest["manifest_path"]
        )
        self.assertEqual(current_report["release_status"], "approved")
        self.assertEqual(current_report["qa_report_sha256"], manifest["qa_report_sha256"])
        with zipfile.ZipFile(manifest["package_path"]) as bundle:
            names = set(bundle.namelist())
            self.assertIn("subtitles/ep_phase2_vertical_9_16.srt", names)
            self.assertIn("subtitles/ep_phase2_vertical_9_16.vtt", names)
            self.assertIn("subtitles/ep_phase2_vertical_9_16.ass", names)
            self.assertIn("reports/content-qa.json", names)

        # Legacy continuity-safe clips may already have subtitles in their
        # pixels.  The delivery gate must not apply a second burn; callers can
        # still package/transcode the legacy artifact explicitly with burning
        # disabled until the panel is regenerated as a clean master.
        legacy_job = store.get_job(jobs[1]["job_id"], ep_id="ep_phase2")
        legacy_metadata = dict(legacy_job["metadata"])
        legacy_metadata["continuity_safe"] = {
            **dict(legacy_metadata.get("continuity_safe") or {}),
            "subtitle_paths": {"burned_in": True},
        }
        store.update_job(jobs[1]["job_id"], metadata=legacy_metadata)
        with self.assertRaisesRegex(RuntimeError, "legacy burned-in subtitles"):
            video_delivery.export_episode(
                "ep_phase2", "vertical_9_16",
                runner=runner, probe_func=probe, ffmpeg="ffmpeg", ffprobe="ffprobe",
                quality_analyzer=self._quality_analysis,
            )
        legacy_manifest = video_delivery.export_episode(
            "ep_phase2", "vertical_9_16", burn_subtitles=False,
            output_path=self.base / "projects/ep_phase2/exports/legacy-clean-pass.mp4",
            runner=runner, probe_func=probe, ffmpeg="ffmpeg", ffprobe="ffprobe",
            quality_analyzer=self._quality_analysis,
        )
        legacy_vf = commands[-1][commands[-1].index("-filter_complex") + 1]
        self.assertNotIn("subtitles=", legacy_vf)
        self.assertEqual(legacy_manifest["subtitles"]["burn_stage"], "legacy_source_clip")
        self.assertEqual(
            legacy_manifest["subtitles"]["legacy_preburned_job_ids"],
            [jobs[1]["job_id"]],
        )

    def test_ass_canvas_uses_landscape_safe_margin_and_portrait_override(self):
        cues = [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "safe caption"}]
        landscape = subtitle_delivery.write_ass(cues, self.base / "landscape.ass")
        portrait = subtitle_delivery.write_ass(
            cues, self.base / "portrait.ass", play_res_x=720, play_res_y=1280,
        )
        landscape_text = landscape.read_text(encoding="utf-8-sig")
        portrait_text = portrait.read_text(encoding="utf-8-sig")
        self.assertIn("PlayResX: 1280", landscape_text)
        self.assertIn("PlayResY: 720", landscape_text)
        self.assertEqual(
            int(next(line for line in landscape_text.splitlines() if line.startswith("Style: Default")).split(",")[-2]),
            72,
        )
        self.assertIn("PlayResX: 720", portrait_text)
        self.assertIn("PlayResY: 1280", portrait_text)
        self.assertEqual(
            int(next(line for line in portrait_text.splitlines() if line.startswith("Style: Default")).split(",")[-2]),
            256,
        )

    def test_six_panel_delivery_zip_has_unique_audit_entries_and_morning_report(self):
        episode = self._episode()
        for panel_index in range(4, 7):
            panel = json.loads(json.dumps(episode["panels"][2]))
            panel["panel_id"] = f"panel_{panel_index:02d}"
            panel["motion"] = f"independent audit beat {panel_index}"
            episode["panels"].append(panel)
            episode["scene_bible"][1]["panel_ids"].append(panel["panel_id"])
        self._with_edit_plan(episode)
        self._approve(episode)
        jobs = self._mark_all_success()
        self.assertEqual(len(jobs), 6)
        store = task_store.default_store()
        for job in jobs:
            audit_dir = self.base / "projects/ep_phase2/audit" / job["panel_name"]
            audit_dir.mkdir(parents=True, exist_ok=True)
            # Continuity-safe historically uses this same basename for every
            # panel, which previously produced duplicate graphs/manifest.json.
            graph = audit_dir / "manifest.json"
            graph.write_text(json.dumps({"job_id": job["job_id"]}), encoding="utf-8")
            cues = audit_dir / "cues.json"
            cues.write_text(json.dumps({"panel": job["panel_name"]}), encoding="utf-8")
            metadata = dict(job["metadata"])
            if int(job["panel_index"]) == 2:
                metadata.update({
                    "render_mode": "continuity_safe",
                    "qa_rejection_audit": [{"reason": "identity drift"}],
                })
            store.update_job(
                job["job_id"], graph_path=str(graph), timing_path=str(cues),
                metadata=metadata,
            )

        def runner(command, **kwargs):
            Path(command[-1]).write_bytes(b"six-panel-final")
            return mock.Mock(returncode=0)

        def probe(path, **kwargs):
            return {
                "path": str(path), "size_bytes": 15, "duration_seconds": 24.0,
                "video": {"codec": "h264", "width": 1280, "height": 720, "fps": 30.0, "pixel_format": "yuv420p"},
                "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2},
            }

        manifest = video_delivery.export_episode(
            "ep_phase2", "landscape_16_9", burn_subtitles=False,
            runner=runner, probe_func=probe, ffmpeg="ffmpeg", ffprobe="ffprobe",
            quality_analyzer=self._quality_analysis,
        )
        with zipfile.ZipFile(manifest["package_path"]) as bundle:
            names = bundle.namelist()
            self.assertEqual(len(names), len(set(names)))
            graph_names = sorted(name for name in names if name.startswith("graphs/"))
            self.assertEqual(graph_names, [
                f"graphs/panel_{index:02d}.manifest.json" for index in range(1, 7)
            ])
            self.assertIn("reports/morning-report.json", names)
            self.assertIn("reports/morning-report.md", names)
            report = json.loads(bundle.read("reports/morning-report.json"))
        self.assertEqual(report["summary"]["job_count"], 6)
        self.assertEqual(report["summary"]["status_counts"], {"succeeded": 6})
        self.assertEqual(report["summary"]["qa_rejection_count"], 1)
        self.assertEqual(report["summary"]["continuity_safe_count"], 1)
        self.assertEqual(len(report["panels"]), 6)
        self.assertGreaterEqual(report["disk"]["free_bytes"], 0)
        self.assertEqual(report["delivery"]["output_path"], manifest["output_path"])
        self.assertTrue(Path(manifest["morning_report"]["json_path"]).is_file())
        self.assertTrue(Path(manifest["morning_report"]["markdown_path"]).is_file())

    def test_public_facade_and_assets_worker_are_distinct_nonblocking_commands(self):
        for name in (
            "prepare_contract", "approve_contract", "prepare_assets", "approve_assets",
            "start_production", "status", "retry", "reject_asset", "retry_asset", "resume", "export",
            "approve_continuity_anchor", "start_continuity_safe", "run_continuity_safe",
        ):
            self.assertTrue(callable(getattr(render_service, name)))
        with mock.patch.object(worker.subprocess, "Popen") as popen:
            popen.return_value.pid = 7001
            assets = render_service.prepare_assets("ep_phase2", timeout=20)
            production = render_service.start_production("ep_other", timeout=20)
            safe = render_service.start_continuity_safe(
                "ep_safe", "ep_safe:0002:panel_02",
                preferred_voice="Microsoft Huihui Desktop", motion="slow_push",
                burn_subtitles=True, timeout=30,
            )
        self.assertTrue(assets["started"])
        self.assertTrue(production["started"])
        self.assertTrue(safe["started"])
        self.assertIn("--assets-only", popen.call_args_list[0].args[0])
        safe_command = popen.call_args_list[2].args[0]
        self.assertIn("--continuity-safe-from", safe_command)
        self.assertIn("ep_safe:0002:panel_02", safe_command)
        self.assertIn("--continuity-voice", safe_command)
        self.assertIn("Microsoft Huihui Desktop", safe_command)
        self.assertNotIn("--no-continuity-burn-subtitles", safe_command)
        self.assertNotIn("--assets-only", safe_command)
        self.assertNotIn("--assets-only", popen.call_args_list[1].args[0])

    def test_assets_only_worker_calls_asset_stage_and_never_h3(self):
        with mock.patch.object(worker, "generate_all_assets_stage", return_value={"ok": True}) as assets_stage, \
             mock.patch.object(worker, "run_episode_jobs") as panel_stage:
            result = worker.run_worker("ep_phase2", assets_only=True)
        self.assertEqual(result["mode"], "assets_only")
        assets_stage.assert_called_once()
        panel_stage.assert_not_called()

    def test_spawned_worker_atomically_takes_over_parent_launch_reservation(self):
        token = "offline-launch-token"
        store = task_store.default_store()
        self.assertTrue(store.reserve_worker_launch("ep_phase2", token))
        with mock.patch.dict(os.environ, {"AI_MANGA_WORKER_LAUNCH_TOKEN": token}), \
             mock.patch.object(worker, "generate_all_assets_stage", return_value={"ok": True}) as assets_stage:
            result = worker.run_worker("ep_phase2", assets_only=True)
        self.assertTrue(result["started"])
        assets_stage.assert_called_once()
        self.assertIsNone(store.worker_info("ep_phase2"))

    def test_h3_prompt_consumes_story_action_continuity_but_forbids_model_text(self):
        panel = self._episode()["panels"][0]
        panel["story_context"] = {"logline": "Two investigators cross a storm-lit station."}
        panel["background_music"] = "epic_brass"
        panel["ambience"] = "office_quiet"
        panel["scene_context"] = {
            "description": "empty rain-lit station with a charity donation box",
            "continuity_lock": {
                "weather_boundary": "rain stays outside glass doors; interior completely dry with no falling moisture",
                "geography": "camera inside store with same lamp positions and checkout counter foreground",
                "hero_props": "transparent charity donation box with blank sides",
                "text_surface_lock": "all surfaces blank",
            },
        }
        prompt = renderer.build_panel_prompt(panel, "locked character A")
        self.assertTrue(prompt.startswith("integrated_multimodal_description:"))
        self.assertIn("A enters and looks left", prompt)
        self.assertIn("story beat A enters", prompt)
        self.assertIn("the deep-blue wet exterior stays behind closed glass", prompt)
        self.assertIn("every visible interior surface remain uniformly dry and clear", prompt)
        self.assertIn("every visible interior surface remain uniformly dry and clear", prompt)
        self.assertIn("geography: camera inside store with same lamp positions", prompt)
        self.assertIn("persistent hero props: transparent charity donation box", prompt)
        self.assertIn("Every visible surface is uniformly blank and unlettered", prompt)
        self.assertIn("overall_soundscape:", prompt)
        self.assertIn("non_diegetic_music:", prompt)
        self.assertIn("muffled exterior water taps the awning", prompt)
        self.assertIn("uninterrupted plain deep-blue glass and blank wall fields", prompt)
        self.assertNotIn("exit sign", prompt.casefold())
        self.assertNotIn("green panel", prompt.casefold())
        self.assertNotIn("pictogram", prompt.casefold())
        self.assertIn("sparse soft piano", prompt)
        self.assertNotIn("quiet indoor office", prompt)
        self.assertNotIn("short brass phrases", prompt)
        self.assertNotIn("Two investigators cross a storm-lit station", prompt)
        self.assertNotIn("第七码头", prompt)

        audio = renderer.resolve_panel_audio(panel)
        self.assertEqual(audio["requested"]["background_music"], "epic_brass")
        self.assertEqual(audio["resolved"]["background_music"], "soft_piano")
        self.assertEqual(audio["resolved"]["ambience"], "rain_outside_glass")
        self.assertEqual(len(audio["overrides"]), 2)

        auto_panel = copy.deepcopy(panel)
        auto_panel["background_music"] = "auto_contextual"
        auto_panel["ambience"] = "auto_contextual"
        auto_audio = renderer.resolve_panel_audio(auto_panel)
        self.assertEqual(auto_audio["resolved"]["background_music"], "soft_piano")
        self.assertEqual(auto_audio["resolved"]["ambience"], "rain_outside_glass")

    def test_h3_runtime_camera_consumes_size_angle_movement_and_composition(self):
        panel = self._episode()["panels"][0]
        panel["scene_context"] = {"description": "dry convenience store interior"}
        panel["camera_plan"] = {
            "shot_size": "medium", "angle": "eye_level",
            "movement": "static", "composition": "over_shoulder_rider",
        }
        panel["visible_action"] = "the rider reaches across the checkout counter"
        prompt = renderer.build_panel_prompt(panel)
        self.assertIn("medium shot; eye level angle; over shoulder rider composition; static", prompt)
        self.assertIn("frame spans head level to checkout counter", prompt)
        self.assertIn("only two people, plain glass and checkout counter", prompt)
        self.assertNotIn("door header", prompt.casefold())


if __name__ == "__main__":
    unittest.main()
