from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

import generation_log
import orchestrator
import render_service
import render_video_h3 as renderer
import task_store
import video_delivery
import worker
import video_quality
from action_catalog import compile_action_spec, derived_action_components


def sample_episode(reference_images=None):
    return {
        "ep_id": "ep_test",
        "aspect_ratio": "16:9",
        "render_settings": {"ref_image_size": "max", "sage_mode": "sage3"},
        "visual_bible": {"style_prompt": "premium ink comic"},
        "character_bible": [{
            "character_id": "char_hero",
            "name": "Hero",
            "identity_prompt": "black-haired woman with an oval face and amber eyes",
            "wardrobe_lock": {"outfit": "navy canvas jacket"},
            "reference_images": list(reference_images or []),
        }],
        "scene_bible": [{
            "scene_id": "scene_warehouse",
            "name": "Warehouse",
            "description": "A locked warehouse interior with one hanging lamp.",
            "positive_prompt": "locked warehouse, one hanging lamp",
            "negative_prompt": "people, text",
            "panel_ids": ["panel_01"],
        }],
        "panels": [{
            "panel_id": "panel_01",
            "name": "panel_01",
            "style": "cinematic",
            "prompt_mode": "cinematic",
            "character_ids": ["char_hero"],
            "scene_id": "scene_warehouse",
            "scene_description": "A locked warehouse interior with one hanging lamp.",
            "spoken_dialogue": [{"start_s": 0.5, "end_s": 2.0, "speaker_id": "char_hero", "text": "到了"}],
            "on_screen_text": [{"start_s": 2.2, "end_s": 3.0, "text": "午夜"}],
            "audio_cues": [{"start_s": 3.1, "end_s": 3.5, "cue_type": "sfx", "prompt": "metal click"}],
            "prompt_package": {
                "positive_prompt": "same locked hero in the same warehouse",
                "negative_prompt": "identity drift, extra people",
                "character_ids": ["char_hero"],
                "scene_id": "scene_warehouse",
                "render_settings": {"ref_image_size": "max", "sage_mode": "sage3", "aspect_ratio": "16:9"},
            },
        }],
    }


class TerraPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tempdir.name)
        self.old_env = dict(os.environ)
        os.environ["AI_FACTORY_ROOT"] = str(self.base)
        os.environ["AI_MANGA_PROJECTS_DIR"] = str(self.base / "projects")
        os.environ["AI_MANGA_JOB_DB"] = str(self.base / "state" / "jobs.sqlite3")
        task_store._default_store = None
        renderer.COMFY = self.base / "comfy"
        renderer.COMFY.mkdir(parents=True, exist_ok=True)
        self.lora_dir = renderer.COMFY / "models" / "loras"
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        (self.lora_dir / "minimax_h3_turbo_ema_ckpt500.safetensors").write_bytes(b"legacy-lora")
        orchestrator.PROJECTS_DIR = self.base / "projects"
        generation_log.ROOT = self.base
        generation_log.STATE_DIR = self.base / "state"
        generation_log.LOG_PATH = generation_log.STATE_DIR / "generations.jsonl"
        generation_log.LOG_BAK = generation_log.STATE_DIR / "generations.jsonl.bak"

    def tearDown(self):
        task_store._default_store = None
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tempdir.cleanup()

    def _image(self, name, payload=b"image"):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def test_prepare_episode_registers_complete_inputs_and_snapshot(self):
        ref = self._image("hero.png")
        snapshot = task_store.prepare_episode("ep_test", sample_episode([str(ref)]))
        self.assertEqual(set(("episode", "jobs", "assets", "deliveries")) - set(snapshot), set())
        self.assertEqual(len(snapshot["jobs"]), 1)
        job = snapshot["jobs"][0]
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["reference_images"], [str(ref)])
        self.assertTrue({"reference_images", "prompt_package", "settings", "character_ids"}.issubset(job["metadata"]["inputs"]))
        self.assertEqual(job["metadata"]["settings"]["ref_image_size"], "match")
        self.assertEqual(job["metadata"]["settings"]["sage_mode"], "sage3")
        self.assertEqual(job["metadata"]["settings"]["reference_fidelity"], "fast")
        self.assertEqual(job["metadata"]["settings"]["render_profile"], "proof")
        self.assertFalse(job["metadata"]["settings"]["delivery_eligible"])
        self.assertEqual(job["metadata"]["settings"]["sage_attention"], "sage3")
        self.assertTrue((self.base / "projects" / "ep_test" / "episode.json").exists())

    def test_resume_preserves_prompt_for_history_recovery_and_skips_success(self):
        ref = self._image("hero.png")
        task_store.prepare_episode("ep_test", sample_episode([str(ref)]))
        store = task_store.default_store()
        job = store.list_jobs("ep_test")[0]
        store.update_job(job["job_id"], status="failed", prompt_id="prompt-old", error="timeout")
        summary = task_store.resume_jobs("ep_test")
        resumed = store.get_job(job["job_id"])
        self.assertEqual(summary["resumed"], 1)
        self.assertEqual(resumed["status"], "queued")
        self.assertEqual(resumed["prompt_id"], "prompt-old")
        self.assertEqual(resumed["error"], "timeout")

    def test_worker_maps_prompt_contract_settings(self):
        episode = sample_episode([])
        panel = episode["panels"][0]
        job = {"metadata": {"settings": {"ref_image_size": "max", "sage_mode": "sage3"}}}
        settings = orchestrator._worker_settings(episode, panel, job)
        self.assertEqual(settings["reference_fidelity"], "fast")
        self.assertEqual(settings["render_profile"], "proof")
        self.assertEqual(settings["sage_attention"], "sage3")
        episode["render_settings"]["production_strategy"] = "direct_production"
        panel["prompt_package"]["render_settings"]["production_strategy"] = "direct_production"
        direct = orchestrator._worker_settings(episode, panel, job)
        self.assertEqual(direct["reference_fidelity"], "identity")
        self.assertEqual(direct["render_profile"], "production")

    def test_changed_inputs_invalidate_old_success_artifact(self):
        ref = self._image("hero.png")
        episode = sample_episode([str(ref)])
        first = task_store.prepare_episode("ep_test", episode)
        store = task_store.default_store()
        job = first["jobs"][0]
        output = Path(job["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"old-success")
        store.update_job(
            job["job_id"], status="succeeded", prompt_id="old-prompt",
            comfy_output_path="old-comfy.mp4", probe={"duration_seconds": 10.0},
        )
        unchanged = task_store.prepare_episode("ep_test", episode)["jobs"][0]
        self.assertEqual(unchanged["status"], "succeeded")
        changed = json.loads(json.dumps(episode))
        changed["panels"][0]["prompt_package"]["positive_prompt"] += " changed camera"
        invalidated = task_store.prepare_episode("ep_test", changed)["jobs"][0]
        self.assertEqual(invalidated["status"], "queued")
        self.assertIsNone(invalidated["prompt_id"])
        self.assertIsNone(invalidated["comfy_output_path"])
        self.assertEqual(invalidated["probe"], {})
        self.assertIsNone(invalidated["completed_at"])

    def test_character_assets_are_worker_stage_and_refresh_panel_refs(self):
        task_store.prepare_episode("ep_test", sample_episode([]))
        task_store.approve_contract("ep_test")
        generated = self._image("generator/hero_anchor.png", b"hero")

        def fake_generator(character, visual_bible, **kwargs):
            return {"character_id": character["character_id"], "reference_images": [str(generated)], "status": "completed"}

        snapshot = orchestrator.prepare_character_assets("ep_test", generator=fake_generator)
        self.assertEqual(snapshot["character_assets_updated"], ["char_hero"])
        refs = snapshot["episode"]["character_bible"][0]["reference_images"]
        self.assertTrue(refs and Path(refs[0]).exists())
        self.assertEqual(snapshot["jobs"][0]["reference_images"], refs)

    def test_graph_has_unique_reference_nodes_and_identity_mode(self):
        graph = renderer.build_h3_ref2va_graph(
            "prompt", 7,
            first_frame_filename="first.png",
            last_frame_filename="last.png",
            character_anchor_filename="anchor.png",
            char_ref_filenames=[f"ref{i}.png" for i in range(6)],
            aspect_ratio="9:16",
            reference_fidelity="identity",
            sage_attention="sage3",
        )
        self.assertEqual(graph["144"]["class_type"], "PathchSageAttentionKJ")
        self.assertEqual([graph[node]["class_type"] for node in renderer.EXTRA_CHAR_REF_NODE_IDS], ["LoadImage"] * 6)
        self.assertEqual(len(graph), len(set(graph)))
        self.assertEqual(graph["136"]["inputs"]["ref_image_size"], "max")
        self.assertEqual(graph["115"]["inputs"]["aspect_ratio"], renderer.ASPECT_RATIO_CHOICES["9:16"])
        self.assertEqual(graph["164"]["class_type"], "MiniMaxH3TurboLoRA")
        self.assertEqual(graph["164"]["inputs"]["lora_name"], "minimax_h3_turbo_ema_ckpt500.safetensors")
        self.assertEqual(graph["164"]["inputs"]["strength"], 1.0)
        self.assertFalse(graph["164"]["inputs"]["low_vram"])
        self.assertEqual(graph["124"]["inputs"]["scheduler"], "simple")
        self.assertEqual(graph["124"]["inputs"]["steps"], 8)
        self.assertEqual(graph["125"]["inputs"]["sampler"], ["165", 0])
        self.assertEqual(graph["165"]["class_type"], "MiniMaxH3TurboSampler")
        self.assertNotIn("123", graph)
        prompt = graph["138"]["inputs"]["value"]
        self.assertIn("<Picture 1> = opening frame and scene authority", prompt)
        self.assertIn("<Picture 2> = final frame authority", prompt)
        self.assertIn("<Picture 3> = primary character identity authority", prompt)
        self.assertNotIn("ref_image_0", prompt)

    def test_h3_lora_resolution_prefers_v4_and_audits_legacy_fallback(self):
        fallback = renderer.resolve_h3_turbo_lora(comfy_root=renderer.COMFY)
        self.assertEqual(fallback["lora_name"], "minimax_h3_turbo_ema_ckpt500.safetensors")
        self.assertEqual(fallback["selection"], "legacy_fallback")
        self.assertFalse(fallback["recommended_available"])
        (self.lora_dir / renderer.H3_LORA_RECOMMENDED).write_bytes(b"v4-600")
        recommended = renderer.resolve_h3_turbo_lora(comfy_root=renderer.COMFY)
        self.assertEqual(recommended["lora_name"], renderer.H3_LORA_RECOMMENDED)
        self.assertEqual(recommended["selection"], "recommended")
        self.assertTrue(recommended["recommended_available"])
        with self.assertRaises(FileNotFoundError):
            renderer.resolve_h3_turbo_lora("not-installed.safetensors", comfy_root=renderer.COMFY)

    def test_h3_reference_ordinals_ignore_sparse_internal_socket_numbers(self):
        bindings = renderer.build_h3_reference_bindings(character_anchor_filename="anchor.png")
        self.assertEqual(bindings[0]["slot"], "ref_images.ref_image_2")
        self.assertEqual(bindings[0]["model_label"], "<Picture 1>")
        graph = renderer.build_h3_ref2va_graph(
            "portrait", 1, character_anchor_filename="anchor.png",
        )
        self.assertIn("<Picture 1> = primary character identity authority", graph["138"]["inputs"]["value"])

    def test_h3_base_mode_omits_lora_and_turbo_step_range_is_enforced(self):
        graph = renderer.build_h3_ref2va_graph("base", 3, use_lora=False)
        self.assertNotIn("164", graph)
        self.assertEqual(graph["124"]["inputs"]["steps"], 20)
        self.assertEqual(graph["124"]["inputs"]["model"], ["144", 0])
        self.assertEqual(graph["126"]["inputs"]["model"], ["144", 0])
        self.assertEqual(graph["125"]["inputs"]["sampler"], ["165", 0])
        for invalid_steps in (3, 9):
            with self.assertRaises(ValueError):
                renderer.build_h3_sampling_contract(turbo_steps=invalid_steps, comfy_root=renderer.COMFY)

    def test_cinematic_prompt_uses_three_timing_lanes_and_actual_aspect(self):
        panel = sample_episode([])["panels"][0]
        panel["aspect_ratio"] = "16:9"
        prompt = renderer.build_panel_prompt(panel, "locked hero")
        self.assertIn("integrated_multimodal_description:", prompt)
        self.assertIn("<d>[Chinese]", prompt)
        self.assertIn("delivery subtitles are composited only after generation", prompt)
        self.assertNotIn(panel["on_screen_text"][0]["text"], prompt)
        self.assertIn("metal click", prompt)
        self.assertIn("16:9", prompt)
        self.assertNotIn("Vertical 9:16", prompt)
        timing = renderer.build_timing_contract(panel, 10.0)
        self.assertEqual(timing["fps"], 24)
        self.assertEqual(timing["spoken_dialogue"][0]["start_seconds"], 0.5)
        self.assertEqual(timing["native_audio_schedule"]["sample_rate_hz"], 32000)
        self.assertEqual(timing["native_audio_schedule"]["channels"], 2)
        self.assertEqual(timing["native_audio_schedule"]["video_flow_shift"], 12.0)
        self.assertEqual(timing["native_audio_schedule"]["audio_flow_shift"], 3.0)

        panel.update({
            "visible_action": "the hero slides one sealed case across the table",
            "first_state": "the sealed case rests beside the hero",
            "final_state": "the sealed case stops in front of the witness",
            "blocking": {
                "start": "the hero holds the case at frame left",
                "motion": "the hero slides the case toward frame right",
                "end": "the witness stops the case with one hand",
            },
            "camera_plan": {"movement": "slow lateral track"},
            "screen_direction": "left to right", "axis": "keep the table axis",
            "eyeline": "hero to witness",
        })
        directed = renderer.build_panel_prompt(panel, "locked hero")
        self.assertIn("Opening state in the 16:9 composition: the hero holds the case at frame left", directed)
        self.assertIn("the hero slides one sealed case across the table", directed)
        self.assertIn("Final state at 10.125 seconds: the witness stops the case", directed)
        self.assertIn("The shot uses slow lateral track; one dominant path", directed)
        self.assertIn("screen direction: left to right", directed)
        self.assertIn("axis: keep the table axis", directed)
        self.assertIn("eyeline: hero to witness", directed)
        self.assertEqual(directed.count("The shot uses"), 1)

        panel["edit_duration_seconds"] = 1.5
        timed = renderer.build_panel_prompt(panel, "locked hero")
        self.assertIn("delivery edit cuts at 1.500 seconds", timed)
        self.assertIn("by 1.250 seconds", timed)
        panel["edit_duration_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "narrative duration"):
            renderer.build_panel_prompt(panel, "locked hero")
        panel.pop("edit_duration_seconds")

        comic_panel = json.loads(json.dumps(panel))
        comic_panel["prompt_mode"] = "comic"
        comic_panel["style"] = "comic"
        comic_prompt = renderer.build_panel_prompt(comic_panel, "locked hero")
        self.assertIn("overall_soundscape:", comic_prompt)
        self.assertIn("From 3.083 to 3.500 seconds", comic_prompt)

    def test_canonical_action_is_the_only_renderer_action_authority(self):
        panel = sample_episode([])["panels"][0]
        compiled = compile_action_spec({
            "actor_id": "char_hero",
            "action_code": "SLIDE_OBJECT",
            "target": "the sealed case across the table",
            "start_state": "the sealed case rests beside char_hero",
            "end_state": "the sealed case rests before the witness",
        }, visible_character_ids=["char_hero"])
        panel.update({
            "action_spec": compiled,
            "action_code": compiled["action_code"],
            "action_components": derived_action_components(compiled),
            "first_state": compiled["start_state"],
            "final_state": compiled["end_state"],
            "visible_action": "TAMPERED DISPLAY ACTION MUST NEVER RENDER",
            "blocking": {
                "start": "TAMPERED BLOCKING START",
                "motion": "TAMPERED BLOCKING MOTION",
                "end": "TAMPERED BLOCKING END",
            },
        })

        prompt = renderer.build_panel_prompt(panel)
        self.assertIn(compiled["h3_action_en"], prompt)
        self.assertEqual(prompt.count(compiled["h3_action_en"]), 1)
        self.assertNotIn("TAMPERED", prompt)
        authority = renderer._compiled_action_authority(panel)
        self.assertEqual(authority["action_code"], "SLIDE_OBJECT")
        self.assertEqual(authority["spec_sha256"], compiled["spec_sha256"])

    def test_verbose_canonical_action_compacts_only_the_runtime_motion_lane(self):
        panel = sample_episode([])["panels"][0]
        compiled = compile_action_spec({
            "actor_id": "char_hero",
            "action_code": "HAND_OBJECT",
            "target": "the only black leather wallet across the checkout counter",
            "start_state": (
                "the only black leather wallet is held in char_hero right hand above the "
                "checkout counter with no wallet on the floor or counter"
            ),
            "end_state": (
                "the same only black leather wallet is gripped in the rider right hand while "
                "char_hero still touches its edge with no other wallet anywhere"
            ),
        }, visible_character_ids=["char_hero"])
        panel.update({
            "action_spec": compiled,
            "action_code": compiled["action_code"],
            "action_components": derived_action_components(compiled),
            "first_state": compiled["start_state"],
            "final_state": compiled["end_state"],
        })

        action, _camera = renderer._runtime_action_and_camera(panel, {})
        prompt = renderer.build_panel_prompt(panel)
        self.assertEqual(
            action,
            "char_hero hands the only black leather wallet across the checkout counter",
        )
        self.assertNotIn(compiled["h3_action_en"], prompt)
        self.assertIn(action, prompt)
        self.assertIn("Opening state", prompt)
        self.assertNotIn("above the.", prompt)
        self.assertNotIn("no other;", prompt)
        self.assertLessEqual(renderer.count_h3_english_words(prompt), 512)
        authority = renderer._compiled_action_authority(panel)
        self.assertEqual(authority["h3_action_en"], compiled["h3_action_en"])

    def test_human_qa_feedback_changes_retry_prompt_without_copying_reviewer_text(self):
        panel = sample_episode([])["panels"][0]
        compiled = compile_action_spec({
            "actor_id": "char_hero",
            "action_code": "HAND_OBJECT",
            "target": "one black wallet",
            "start_state": "char_hero holds one black wallet",
            "end_state": "the rider alone holds the same black wallet",
        }, visible_character_ids=["char_hero"])
        panel.update({
            "action_spec": compiled,
            "action_code": compiled["action_code"],
            "action_components": derived_action_components(compiled),
            "edit_duration_seconds": 1.6,
            "qa_retry_feedback": {
                "reason": "钱包复制了，必须只有一个",
                "category": "continuity_or_state",
                "at": "2026-08-29T00:00:00Z",
            },
        })
        retry_prompt = renderer.build_panel_prompt(panel)
        self.assertIn("exactly one black wallet exists in every frame", retry_prompt)
        self.assertIn("Never duplicate, split, teleport, pre-place or replace", retry_prompt)
        self.assertIn("Never render correction notes", retry_prompt)
        self.assertNotIn("钱包复制了", retry_prompt)

        baseline_panel = json.loads(json.dumps(panel, ensure_ascii=False))
        baseline_panel.pop("qa_retry_feedback")
        baseline_prompt = renderer.build_panel_prompt(baseline_panel)
        self.assertNotEqual(
            hashlib.sha256(retry_prompt.encode("utf-8")).hexdigest(),
            hashlib.sha256(baseline_prompt.encode("utf-8")).hexdigest(),
        )

        timing_panel = json.loads(json.dumps(panel, ensure_ascii=False))
        timing_panel["action_spec"]["target"] = "the only black wallet"
        timing_panel["action_spec"].pop("h3_action_en", None)
        timing_panel["action_spec"].pop("spec_sha256", None)
        timing_compiled = compile_action_spec(
            timing_panel["action_spec"], visible_character_ids=["char_hero"],
        )
        timing_panel["action_spec"] = timing_compiled
        timing_panel["action_components"] = derived_action_components(timing_compiled)
        timing_panel["action_code"] = timing_compiled["action_code"]
        timing_panel["qa_retry_feedback"]["category"] = "action_timing_or_edit_window"
        timing_prompt = renderer.build_panel_prompt(timing_panel)
        self.assertIn("start the black wallet action at frame zero without pause", timing_prompt)
        self.assertNotIn("start the the", timing_prompt)

    def test_submit_snapshots_canonical_action_authority(self):
        panel = sample_episode([])["panels"][0]
        compiled = compile_action_spec({
            "actor_id": "char_hero",
            "action_code": "OPEN_OBJECT",
            "target": "the warehouse door",
            "start_state": "the warehouse door is closed",
            "end_state": "the warehouse door is fully open",
        }, visible_character_ids=["char_hero"])
        panel.update({
            "action_spec": compiled,
            "action_code": compiled["action_code"],
            "action_components": derived_action_components(compiled),
            "first_state": compiled["start_state"],
            "final_state": compiled["end_state"],
            "visible_action": "TAMPERED DISPLAY ACTION",
        })
        store = task_store.RenderJobStore(self.base / "action-authority.sqlite3")
        job = renderer.submit_render_job(
            panel,
            self.base / "projects" / "ep_action" / "videos" / "panel_01.mp4",
            ep_id="ep_action", panel_index=1, store=store,
            api_func=lambda path, payload: {"prompt_id": "mock-action-authority"},
        )

        snapshot = json.loads(Path(job["graph_path"]).read_text(encoding="utf-8"))
        expected = {
            "catalog_version": compiled["catalog_version"],
            "action_code": compiled["action_code"],
            "spec_sha256": compiled["spec_sha256"],
            "compiled_h3_sha256": hashlib.sha256(
                compiled["h3_action_en"].encode("utf-8")
            ).hexdigest(),
            "source": "panel.action_spec",
            "h3_action_en": compiled["h3_action_en"],
        }
        self.assertEqual(snapshot["action_contract"], expected)
        self.assertEqual(store.get_job(job["job_id"], ep_id="ep_action")["metadata"]["action_contract"], expected)
        self.assertNotIn("TAMPERED", snapshot["prompt"])

    def test_submit_snapshots_character_source_ids_in_h3_bindings(self):
        panel = sample_episode([])["panels"][0]
        panel["character_ids"] = ["char_hero", "char_clerk"]
        panel["visible_action"] = "char_hero hands the key to char_clerk"
        panel["first_state"] = "char_hero holds the key"
        panel["final_state"] = "char_clerk holds the key"
        hero = self._image("charrefs/hero.png", b"hero")
        clerk = self._image("charrefs/clerk.png", b"clerk")
        store = task_store.RenderJobStore(self.base / "source-id-bindings.sqlite3")
        job = renderer.submit_render_job(
            panel,
            self.base / "projects" / "ep_source_ids" / "videos" / "panel_01.mp4",
            ep_id="ep_source_ids", panel_index=1,
            character_anchor=hero, character_anchor_source_id="char_hero",
            char_refs=[clerk], extra_reference_source_ids=["char_clerk"],
            store=store,
            api_func=lambda path, payload: {"prompt_id": "mock-source-ids"},
        )
        snapshot = json.loads(Path(job["graph_path"]).read_text(encoding="utf-8"))
        bindings = snapshot["settings"]["reference_bindings"]
        self.assertEqual([item.get("source_id") for item in bindings], ["char_hero", "char_clerk"])
        self.assertIn("<Subject 1> is the approved character char_hero", snapshot["prompt"])
        self.assertIn("<Subject 1> hands the key to <Subject 2>", snapshot["prompt"])

    def test_submit_uses_scene_source_id_when_path_has_no_scene_hint(self):
        panel = sample_episode([])["panels"][0]
        panel["scene_id"] = "scene_warehouse"
        approved_scene = self._image("approved/location.png", b"approved-scene")
        store = task_store.RenderJobStore(self.base / "scene-source-id.sqlite3")
        job = renderer.submit_render_job(
            panel,
            self.base / "projects" / "ep_scene_source" / "videos" / "panel_01.mp4",
            ep_id="ep_scene_source", panel_index=1,
            char_refs=[approved_scene], extra_reference_source_ids=["scene_warehouse"],
            store=store,
            api_func=lambda path, payload: {"prompt_id": "mock-scene-source"},
        )
        snapshot = json.loads(Path(job["graph_path"]).read_text(encoding="utf-8"))
        self.assertEqual(snapshot["reference_images"][0]["role"], "scene_reference")
        self.assertEqual(
            snapshot["settings"]["reference_bindings"][0]["role"],
            "scene_reference",
        )
        self.assertIn("defines the approved location", snapshot["prompt"])

    def test_submit_fails_closed_before_character_refs_omit_scene_reference(self):
        panel = sample_episode([])["panels"][0]
        panel["scene_id"] = "scene_warehouse"
        character_refs = [
            self._image(f"charrefs/char_{index}.png", f"character-{index}".encode())
            for index in range(renderer.MAX_CHAR_REFS)
        ]
        scene_ref = self._image("scenerefs/warehouse.png", b"approved-scene")
        submitted = []

        with self.assertRaisesRegex(ValueError, "would omit its approved scene reference"):
            renderer.submit_render_job(
                panel,
                self.base / "projects" / "ep_capacity" / "videos" / "panel_01.mp4",
                ep_id="ep_capacity", panel_index=1,
                char_refs=[*character_refs, scene_ref],
                extra_reference_source_ids=[
                    *[f"char_{index}" for index in range(renderer.MAX_CHAR_REFS)],
                    "scene_warehouse",
                ],
                store=task_store.RenderJobStore(self.base / "capacity.sqlite3"),
                api_func=lambda path, payload: submitted.append((path, payload)),
            )

        self.assertEqual(submitted, [])

    def test_current_six_panels_compile_to_bounded_h3_runtime_prompts(self):
        episode_path = (
            Path(__file__).resolve().parents[1]
            / "output" / "projects" / "ep_1786340037" / "episode.json"
        )
        self.assertTrue(episode_path.is_file(), "current acceptance episode fixture is missing")
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        self.assertEqual(len(episode.get("panels") or []), 6)
        scenes = {item["scene_id"]: item for item in episode.get("scene_bible") or []}

        for panel_index, panel_source in enumerate(episode["panels"], 1):
            panel = dict(panel_source)
            panel["story_context"] = episode.get("story_bible") or {}
            panel["scene_context"] = scenes.get(panel.get("scene_id"), {})
            character_context = orchestrator._character_context(episode, panel)
            prompt = renderer.build_panel_prompt(panel, character_context)
            body_words = renderer.count_h3_english_words(prompt)
            self.assertGreaterEqual(body_words, renderer.H3_PROMPT_BODY_MIN_ENGLISH_WORDS)
            self.assertLessEqual(body_words, renderer.H3_PROMPT_BODY_MAX_ENGLISH_WORDS)
            self.assertLess(len(prompt.encode("utf-8")), 5000)
            self.assertEqual(prompt.count("The single visible action begins immediately"), 1)
            self.assertEqual(prompt.count("The shot uses"), 1)
            self.assertIn("integrated_multimodal_description:", prompt)
            self.assertIn("overall_soundscape:", prompt)
            self.assertIn("non_diegetic_music:", prompt)
            self.assertIn("delivery subtitles are composited only after generation", prompt)
            self.assertNotIn("[PROMPT PACKAGE", prompt)
            self.assertNotIn("[VISUAL_BIBLE]", prompt)
            self.assertNotIn("[CHARACTER_LOCKS]", prompt)
            self.assertNotIn("[APPROVED STORY CONTEXT]", prompt)
            self.assertNotIn(str((episode.get("story_bible") or {}).get("logline") or ""), prompt)

            for cue in panel.get("spoken_dialogue") or []:
                self.assertEqual(prompt.count(str(cue["text"])), 1)
            for cue in panel.get("audio_cues") or []:
                self.assertEqual(prompt.count(str(cue["prompt"])), 1)
            for cue in panel.get("on_screen_text") or []:
                text = str(cue.get("text") or "")
                if text:
                    self.assertNotIn(text, prompt)

            tags = renderer._package_tags(panel)
            approved_final_state = (
                panel.get("final_state")
                or panel.get("end_state")
                or tags["LAST_FRAME"]
            )
            self.assertIn(approved_final_state, prompt)
            bindings = renderer.build_h3_reference_bindings(
                first_frame_filename="scene.png",
                character_anchor_filename="char_1.png",
                extra_reference_filenames=[f"char_{index}.png" for index in range(2, 6)],
                extra_reference_roles=["character_reference"] * 4,
            )
            complete = renderer.build_panel_prompt(panel, character_context, bindings)
            self.assertLessEqual(
                renderer.count_h3_english_words(complete),
                renderer.H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS,
            )
            self.assertTrue(complete.startswith("subject_definitions:"))
            self.assertIn("<Picture 1> defines the approved opening composition", complete)
            self.assertIn("retention_analysis:", complete)
            self.assertIn("detailed_description:", complete)
            self.assertIn("shown in <Picture 6>", complete)
            self.assertIn(
                "exactly 5 distinct visible characters; no person is added, omitted, merged, duplicated, or replaced",
                complete,
            )
            self.assertIn("<Subject 5>", complete)

            continuation_bindings = renderer.build_h3_reference_bindings(
                first_frame_filename="prior_tail.png",
                character_anchor_filename="char_1.png",
                extra_reference_filenames=[
                    *[f"char_{index}.png" for index in range(2, 6)],
                    "scene.png",
                ],
                extra_reference_roles=["character_reference"] * 4 + ["scene_reference"],
            )
            continuation = renderer._prompt_with_reference_map(prompt, continuation_bindings)
            self.assertLessEqual(
                renderer.count_h3_english_words(continuation),
                renderer.H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS,
            )
            self.assertLessEqual(
                renderer.count_h3_english_words(continuation),
                renderer.H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS,
            )
            self.assertIn("<Picture 7> = scene layout and lighting authority", continuation)
            self.assertIn(
                "[CAST] Exactly 5 distinct people throughout every frame",
                continuation,
            )
            if panel_index == 2:
                self.assertIn("one device each", prompt)
                approved_camera = str(panel.get("camera_movement") or "").split(",", 1)[0]
                self.assertTrue(approved_camera)
                self.assertIn(f"The shot uses {approved_camera}", prompt)
                self.assertIn("Final state at", prompt)
                self.assertEqual(prompt.count("准备好了，开始吧！"), 1)
                self.assertEqual(prompt.count("game loading sound"), 1)
                self.assertIn("delivery subtitles are composited only after generation", prompt)
                self.assertLessEqual(body_words, renderer.H3_PROMPT_BODY_MAX_ENGLISH_WORDS)
                self.assertLessEqual(
                    renderer.count_h3_english_words(continuation),
                    renderer.H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS,
                )

                eight_reference_bindings = renderer.build_h3_reference_bindings(
                    first_frame_filename="prior_tail.png",
                    last_frame_filename="approved_final.png",
                    character_anchor_filename="char_1.png",
                    extra_reference_filenames=[
                        *[f"char_{index}.png" for index in range(2, 6)],
                        "scene.png",
                    ],
                    extra_reference_roles=["character_reference"] * 4 + ["scene_reference"],
                )
                eight_reference_prompt = renderer._prompt_with_reference_map(
                    prompt, eight_reference_bindings,
                )
                self.assertLessEqual(
                    renderer.count_h3_english_words(eight_reference_prompt),
                    renderer.H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS,
                )
                self.assertIn("<Picture 1> = opening frame and scene authority", eight_reference_prompt)
                self.assertIn("<Picture 2> = final frame authority", eight_reference_prompt)
                self.assertIn("<Picture 8> = scene layout and lighting authority", eight_reference_prompt)
                self.assertIn(
                    "[CAST] Exactly 5 distinct people throughout every frame",
                    eight_reference_prompt,
                )
                self.assertIn("one per character-reference", eight_reference_prompt)
                self.assertIn(
                    "Never render picture labels, filenames, subtitles, or reference annotations",
                    eight_reference_prompt,
                )

    def test_h3_prompt_hard_limit_rejects_oversized_uncompiled_input(self):
        oversized = "word " * (renderer.H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS + 1)
        with self.assertRaises(ValueError):
            renderer._prompt_with_reference_map(oversized, [])

    def test_current_panel2_composition_anchors_suppress_portraits_and_scene(self):
        project_root = Path(__file__).resolve().parents[1]
        episode_path = project_root / "output" / "projects" / "ep_1786340037" / "episode.json"
        self.assertTrue(episode_path.is_file(), "current acceptance episode fixture is missing")
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        panel = dict(episode["panels"][1])
        self.assertEqual(panel["panel_id"], "ep01_panel02")
        scenes = {item["scene_id"]: item for item in episode.get("scene_bible") or []}
        panel["story_context"] = episode.get("story_bible") or {}
        panel["scene_context"] = scenes[panel["scene_id"]]

        def _reproject(path_str: str) -> Path:
            """Re-root an absolute path from the original build machine to project_root."""
            # Normalize both Windows and POSIX separators.
            normalized = path_str.replace("\\", "/")
            # Find the 'output' anchor and re-root from there.
            idx = normalized.find("output/")
            if idx >= 0:
                return (project_root / normalized[idx:]).resolve()
            # Fall back to the raw path resolved relative to project_root.
            return (project_root / path_str).resolve()

        character_refs = [
            _reproject(card["reference_images"][0])
            for card in episode.get("character_bible") or []
            if card.get("character_id") in panel["character_ids"]
        ]
        scene_ref = _reproject(scenes[panel["scene_id"]]["reference_images"][0])
        group_anchor = _reproject(panel["last_frame_path"])
        self.assertEqual(len(character_refs), 5)
        self.assertTrue(group_anchor.is_file())
        self.assertTrue(scene_ref.is_file())
        self.assertTrue(all(path.is_file() for path in character_refs))

        store = task_store.RenderJobStore(self.base / "composition-jobs.sqlite3")
        job = renderer.submit_render_job(
            panel,
            self.base / "projects" / "ep_composition" / "videos" / "panel_02.mp4",
            ep_id="ep_composition", panel_index=2,
            job_id="ep_composition:0002:panel_02",
            character_desc=orchestrator._character_context(episode, panel),
            first_frame=group_anchor, last_frame=group_anchor,
            character_anchor=character_refs[0],
            char_refs=[*character_refs[1:], scene_ref],
            composition_anchor_first=True,
            store=store,
            api_func=lambda path, payload: {"prompt_id": "mock-composition-anchor"},
        )

        snapshot = json.loads(Path(job["graph_path"]).read_text(encoding="utf-8"))
        references = snapshot["reference_images"]
        self.assertEqual([item["role"] for item in references], ["first_frame", "last_frame"])
        self.assertEqual(len(references), 2)
        self.assertEqual({item["source_path"] for item in references}, {str(group_anchor)})
        suppressed = set(snapshot["settings"]["suppressed_reference_sources"])
        self.assertEqual(suppressed, {str(path) for path in [*character_refs, scene_ref]})
        self.assertEqual(snapshot["settings"]["reference_policy"], "composition_anchor_first")
        self.assertEqual(snapshot["settings"]["composition_anchor_cast_count"], 5)
        self.assertFalse(snapshot["settings"]["synthetic_last_from_first"])
        self.assertEqual(
            [item["role"] for item in snapshot["settings"]["reference_bindings"]],
            ["first_frame", "last_frame"],
        )
        graph_prompt = snapshot["graph"]["138"]["inputs"]["value"]
        self.assertIn("<Picture 1> defines the approved opening composition", graph_prompt)
        self.assertIn("exactly 5 distinct visible characters", graph_prompt)
        self.assertIn("<Picture 2> defines the approved final composition", graph_prompt)
        self.assertIn("no person is added, omitted, merged, duplicated, or replaced", graph_prompt)
        self.assertIn("one device each", graph_prompt)
        self.assertIn("Opening state in the 9:16 composition", graph_prompt)
        self.assertIn("Final state at 10.125 seconds", graph_prompt)
        self.assertIn("new framing/action starts immediately", graph_prompt)
        self.assertIn("never replay, freeze, or linger", graph_prompt)
        self.assertIn("Every visible surface is uniformly blank and unlettered", graph_prompt)
        self.assertNotIn("character identity authority", graph_prompt)
        self.assertNotIn("scene layout and lighting authority", graph_prompt)
        self.assertLessEqual(
            renderer.count_h3_english_words(graph_prompt),
            renderer.H3_PROMPT_TOTAL_MAX_ENGLISH_WORDS,
        )

    def test_composition_anchor_policy_synthesizes_final_anchor_but_chain_root_keeps_assets(self):
        tail = self._image("continuity/previous_tail.png", b"group-tail")
        character = self._image("charrefs/hero.png", b"portrait")
        scene = self._image("scenerefs/scene_room.png", b"scene")

        strict_selection = renderer.select_h3_reference_sources(
            first_frame=tail, last_frame=None,
            character_anchor=character, extra_references=[scene],
            composition_anchor_first=True,
        )
        self.assertEqual(strict_selection["policy"], "composition_anchor_first")
        self.assertEqual(strict_selection["first_frame"], tail.resolve())
        self.assertEqual(strict_selection["last_frame"], tail.resolve())
        self.assertTrue(strict_selection["synthetic_last_from_first"])
        self.assertIsNone(strict_selection["character_anchor"])
        self.assertEqual(strict_selection["extra_references"], [])
        self.assertEqual(strict_selection["suppressed_references"], [character.resolve(), scene.resolve()])

        panel = sample_episode([])["panels"][0]
        panel["character_ids"] = ["char_a", "char_b", "char_c", "char_d", "char_e"]
        store = task_store.RenderJobStore(self.base / "synthetic-last-jobs.sqlite3")
        job = renderer.submit_render_job(
            panel,
            self.base / "projects" / "ep_synthetic_last" / "videos" / "panel_03.mp4",
            ep_id="ep_synthetic_last", panel_index=3,
            job_id="ep_synthetic_last:0003:panel_03",
            first_frame=tail, last_frame=None,
            character_anchor=character, char_refs=[scene],
            composition_anchor_first=True,
            store=store,
            api_func=lambda path, payload: {"prompt_id": "mock-synthetic-last"},
        )
        snapshot = json.loads(Path(job["graph_path"]).read_text(encoding="utf-8"))
        self.assertEqual(
            [item["role"] for item in snapshot["reference_images"]],
            ["first_frame", "last_frame"],
        )
        self.assertEqual(
            [item["source_path"] for item in snapshot["reference_images"]],
            [str(tail.resolve()), str(tail.resolve())],
        )
        self.assertTrue(snapshot["settings"]["synthetic_last_from_first"])
        self.assertEqual(
            [item["role"] for item in snapshot["settings"]["reference_bindings"]],
            ["first_frame", "last_frame"],
        )
        self.assertIn("<Picture 1> defines the approved opening composition", snapshot["graph"]["138"]["inputs"]["value"])
        self.assertIn("<Picture 2> defines the approved final composition", snapshot["graph"]["138"]["inputs"]["value"])
        self.assertIn("new framing/action starts immediately", snapshot["graph"]["138"]["inputs"]["value"])

        chain_root = renderer.select_h3_reference_sources(
            first_frame=None, last_frame=None,
            character_anchor=character, extra_references=[scene],
            composition_anchor_first=False,
        )
        self.assertEqual(chain_root["policy"], "standard")
        self.assertFalse(chain_root["synthetic_last_from_first"])
        self.assertEqual(chain_root["character_anchor"], character.resolve())
        self.assertEqual(chain_root["extra_references"], [scene.resolve()])

        continued_with_bibles = renderer.select_h3_reference_sources(
            first_frame=tail, last_frame=None,
            character_anchor=character, extra_references=[scene],
            composition_anchor_first=False,
        )
        self.assertEqual(continued_with_bibles["policy"], "standard")
        self.assertEqual(continued_with_bibles["first_frame"], tail.resolve())
        self.assertIsNone(continued_with_bibles["last_frame"])
        self.assertEqual(continued_with_bibles["character_anchor"], character.resolve())
        self.assertEqual(continued_with_bibles["extra_references"], [scene.resolve()])
        self.assertEqual(continued_with_bibles["suppressed_references"], [])

    def test_all_ui_ambience_presets_expand_to_descriptions(self):
        keys = ["rain_night_city", "office_quiet", "forest_morning", "subway_crowd", "storm_thunder", "silence"]
        for key in keys:
            self.assertIn(key, renderer.AMBIENCE_PRESETS)
            self.assertNotEqual(renderer.AMBIENCE_PRESETS[key], key)
            self.assertGreater(len(renderer.AMBIENCE_PRESETS[key].split()), 3)

    def test_submit_only_and_wait_copy_are_explicit_and_samefile_safe(self):
        store = task_store.RenderJobStore(self.base / "jobs.sqlite3")
        edit_inputs = {
            "inputs": {
                "settings": {"edit_duration_seconds": 4.0},
                "shot_plan": {"edit_duration_seconds": 4.0},
            }
        }
        store.register_jobs("ep_test", [
            {
                "job_id": "ep_test:submit", "panel_index": 1,
                "panel_name": "panel_submit", "status": "queued",
                "input_hash": "submit-input", "metadata": edit_inputs,
            },
            {
                "job_id": "ep_test:sibling", "panel_index": 2,
                "panel_name": "panel_sibling", "status": "queued",
                "input_hash": "sibling-input", "metadata": edit_inputs,
            },
        ])
        refs = [self._image(f"refs/{index}.png", f"ref{index}".encode()) for index in range(3)]
        refs.append(self._image("scenerefs/scene_warehouse.png", b"scene"))
        calls = []

        def submit_api(path, payload):
            calls.append((path, payload))
            return {"prompt_id": "prompt-submit"}

        output = self.base / "projects" / "ep_test" / "videos" / "panel_submit.mp4"
        job = renderer.render_panel(
            sample_episode([])["panels"][0], output,
            ep_id="ep_test", job_id="ep_test:submit", panel_index=1,
            character_anchor=refs[0], first_frame=refs[1], last_frame=refs[2], char_refs=[refs[3]],
            wait_for_completion=False, store=store, api_func=submit_api,
            aspect_ratio="16:9", reference_fidelity="identity", sage_attention="sage3",
        )
        self.assertEqual(job["prompt_id"], "prompt-submit")
        self.assertEqual(len(store.list_jobs("ep_test")), 2)
        self.assertIsNotNone(store.get_job("ep_test:sibling", "ep_test"))
        self.assertEqual([item[0] for item in calls], ["/prompt"])
        snapshot = json.loads(Path(job["graph_path"]).read_text(encoding="utf-8"))
        self.assertEqual(len(snapshot["reference_images"]), 4)
        self.assertIn("170", snapshot["graph"])
        sampling = snapshot["settings"]["sampling_contract"]
        self.assertEqual(sampling["lora_selection"], "legacy_fallback")
        self.assertEqual(sampling["lora_name"], "minimax_h3_turbo_ema_ckpt500.safetensors")
        self.assertEqual(sampling["steps"], 8)
        self.assertEqual(sampling["scheduler"], "simple")
        self.assertEqual(sampling["video_clock"]["flow_shift"], 12.0)
        self.assertEqual(sampling["audio_clock"]["sample_rate_hz"], 32000)
        bindings = snapshot["settings"]["reference_bindings"]
        self.assertEqual([item["model_label"] for item in bindings], [
            "<Picture 1>", "<Picture 2>", "<Picture 3>", "<Picture 4>",
        ])
        self.assertEqual([item["role"] for item in bindings], [
            "first_frame", "last_frame", "character_anchor", "scene_reference",
        ])
        self.assertEqual(snapshot["reference_images"][-1]["source_kind"], "scene")

        source = renderer.COMFY / "output" / "video" / "ep_test" / "clip.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"mock-video")

        def history_api(path, payload):
            return {"prompt-submit": {"status": {"status_str": "success"}, "outputs": {"92": {"images": [{"filename": "clip.mp4", "subfolder": "video/ep_test"}]}}}}

        probe = lambda path: {"path": str(path), "duration_seconds": 10.125, "video": {"codec": "h264", "width": 1920, "height": 1080, "fps": 24.0, "pixel_format": "yuv420p"}, "audio": {"codec": "aac", "sample_rate": 32000, "channels": 2}}
        def edit_selector(analysis, **kwargs):
            selection = {
                "in_seconds": 0.0, "out_seconds": 4.0,
                "duration_seconds": 4.0, "reason": "offline fixture",
                "metrics": {"window_mean_luma_change": 0.1},
                "selector": {"name": "offline-test", "version": "1"},
                "source_artifact_sha256": kwargs["source_artifact_sha256"],
                "source_decoded_visual_sha256": analysis["decoded_visual_sha256"],
            }
            selection["selection_sha256"] = video_quality._selection_hash(selection)
            return selection

        final = renderer.wait_render_job(
            job["job_id"], store=store, api_func=history_api, probe_func=probe,
            quality_analyzer=lambda *args, **kwargs: {
                "decoded_visual_sha256": "a" * 64,
                "perceptual_hashes": ["1" * 64, "2" * 64], "static": False,
            },
            edit_selector=edit_selector,
            poll_interval=0.01,
        )
        self.assertEqual(final.read_bytes(), b"mock-video")
        self.assertEqual(renderer._copy_video_atomic(final, final), final.resolve())
        self.assertEqual(store.get_job(job["job_id"])["status"], "succeeded")

        duplicate_source = renderer.COMFY / "output" / "video" / "ep_test" / "clip-audio-2.mp4"
        duplicate_source.write_bytes(b"same-pictures-different-audio-container")
        sibling_output = self.base / "projects" / "ep_test" / "videos" / "panel_sibling.mp4"
        store.update_job(
            "ep_test:sibling", status="submitted", prompt_id="prompt-duplicate",
            output_path=str(sibling_output), preview_path=str(sibling_output),
        )

        def duplicate_history(path, payload):
            return {"prompt-duplicate": {
                "status": {"status_str": "success"},
                "outputs": {"92": {"images": [{
                    "filename": "clip-audio-2.mp4", "subfolder": "video/ep_test",
                }]}}
            }}

        with self.assertRaisesRegex(RuntimeError, "exact_visual_duplicate"):
            renderer.wait_render_job(
                "ep_test:sibling", store=store, api_func=duplicate_history,
                probe_func=probe, quality_analyzer=lambda *args, **kwargs: {
                    "decoded_visual_sha256": "a" * 64,
                    "perceptual_hashes": ["1" * 64, "2" * 64], "static": False,
                }, edit_selector=edit_selector, poll_interval=0.01,
            )
        self.assertEqual(store.get_job("ep_test:sibling")["status"], "failed")

    def test_delivery_builds_safe_fit_export_and_validated_manifest(self):
        hero_ref = self._image("refs/delivery_hero.png", b"hero-ref")
        scene_ref = self._image("refs/delivery_scene.png", b"scene-ref")
        episode = sample_episode([str(hero_ref)])
        episode["render_settings"]["production_strategy"] = "direct_production"
        episode["panels"][0]["prompt_package"]["render_settings"]["production_strategy"] = "direct_production"
        episode["scene_bible"][0]["reference_images"] = [str(scene_ref)]
        episode["render_settings"]["target_edit_duration_seconds"] = 4.0
        episode["panels"][0].update({
            "source_generation_duration_seconds": 10.125,
            "edit_duration_seconds": 4.0,
            "shot_role": "setup", "story_beat_id": "beat_01",
            "visible_action": "the hero checks the warehouse door",
            "first_state": "the door is closed", "final_state": "the lock clicks",
            "camera_plan": {"movement": "slow push in"},
            "transition": {
                "type": "close", "motivation": "end this one-shot fixture",
            },
            "risk": "hand continuity", "failure_code": "RISK_HAND_DRIFT",
        })
        snapshot = task_store.prepare_episode("ep_test", episode)
        task_store.approve_contract(
            "ep_test", expected_hash=snapshot["pipeline"]["contract_hash"],
        )
        task_store.approve_assets("ep_test")
        clip = self.base / "clip.mp4"
        clip.write_bytes(b"clip")
        store = task_store.default_store()
        job = store.list_jobs("ep_test")[0]
        artifact_sha = hashlib.sha256(clip.read_bytes()).hexdigest()
        visual_sha = "d" * 64
        selection = {
            "in_seconds": 0.0, "out_seconds": 4.0, "duration_seconds": 4.0,
            "reason": "offline fixture", "metrics": {"window_mean_luma_change": 0.1},
            "selector": {"name": "offline-test", "version": "1"},
            "source_artifact_sha256": artifact_sha,
            "source_decoded_visual_sha256": visual_sha,
        }
        selection["selection_sha256"] = video_quality._selection_hash(selection)
        store.update_job(
            job["job_id"], status="succeeded", output_path=str(clip),
            probe={"duration_seconds": 10.125, "video": {"width": 1920, "height": 1080, "fps": 24.0}},
            metadata={
                **job["metadata"],
                "artifact_sha256": artifact_sha, "edit_selection": selection,
                "content_qa": {"passed": True, "analysis": {
                    "decoded_visual_sha256": visual_sha, "perceptual_hashes": ["1" * 64],
                    "static": False,
                }},
                "editorial_review": {"status": "pending"},
                "release": {"status": "pending"},
            },
        )
        task_store.approve_job_review(
            "ep_test", job["job_id"], expected_artifact_sha256=artifact_sha,
            expected_edit_selection_sha256=selection["selection_sha256"],
        )
        task_store.approve_episode_release(
            "ep_test", expected_artifact_hashes={job["job_id"]: artifact_sha},
            expected_edit_selection_hashes={job["job_id"]: selection["selection_sha256"]},
        )
        commands = []

        def runner(command, **kwargs):
            commands.append(command)
            Path(command[-1]).write_bytes(b"delivery")
            return mock.Mock(returncode=0)

        def probe(path, **kwargs):
            return {"path": str(path), "size_bytes": 8, "duration_seconds": 4.0,
                    "video": {"codec": "h264", "width": 720, "height": 1280, "fps": 30.0, "pixel_format": "yuv420p"},
                    "audio": {"codec": "aac", "sample_rate": 48000, "channels": 2}}

        manifest = video_delivery.export_episode(
            "ep_test", "tiktok", clip_paths=[clip], runner=runner,
            probe_func=probe, ffmpeg="ffmpeg", ffprobe="ffprobe",
            quality_analyzer=lambda *args, **kwargs: {
                "decoded_visual_sha256": visual_sha, "perceptual_hashes": ["1" * 64],
                "static": False, "metrics": {"sample_count": 1},
            },
        )
        command = commands[0]
        self.assertIn("trim=start=0.000000:end=4.000000", command[command.index("-filter_complex") + 1])
        self.assertIn("pad=720:1280", command[command.index("-filter_complex") + 1])
        self.assertEqual(manifest["preset"]["delivery_standard"], "720p-v1")
        self.assertEqual(manifest["preset"]["width"], 720)
        self.assertEqual(manifest["preset"]["height"], 1280)
        self.assertIn("+faststart", command)
        self.assertIn("yuv420p", command)
        self.assertEqual(manifest["resize_mode"], "fit")
        self.assertFalse(manifest["watermark_added"])
        director_plan = manifest["director_delivery_plan"]
        self.assertEqual(director_plan["execution_policy"]["video_transition"], "hard_cut_only")
        self.assertTrue(director_plan["execution_policy"]["dissolve_forbidden"])
        shot = director_plan["shots"][0]
        self.assertEqual(shot["dominant_camera_move"], "slow push in")
        self.assertEqual(shot["transition"]["requested_type"], "close")
        self.assertEqual(shot["sound_bridge"]["execution"], "not_requested")
        self.assertEqual(shot["failure_code"], "RISK_HAND_DRIFT")
        validation = director_plan["transition_validation"]
        self.assertTrue(validation["passed"])
        self.assertEqual(validation["boundaries"][0]["status"], "terminal")
        self.assertNotIn("dissolve", command[command.index("-filter_complex") + 1].lower())
        self.assertTrue(Path(manifest["manifest_path"]).exists())
        self.assertTrue(Path(manifest["package_path"]).exists())
        import zipfile
        with zipfile.ZipFile(manifest["package_path"]) as bundle:
            self.assertIn("final.mp4", bundle.namelist())
            self.assertIn("manifest.json", bundle.namelist())
            self.assertIn("episode.json", bundle.namelist())
        self.assertEqual(manifest["package_validation"]["status"], "passed")
        self.assertIn("reports/content-qa.json", manifest["package_validation"]["required_members"])

    def test_delivery_package_validation_fails_closed_on_missing_member(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final = root / "final.mp4"
            manifest_path = root / "manifest.json"
            package = root / "delivery.zip"
            final.write_bytes(b"verified-video")
            manifest_path.write_text(
                json.dumps({"release_status": "approved"}), encoding="utf-8",
            )
            import zipfile
            with zipfile.ZipFile(package, "w") as bundle:
                bundle.write(final, "final.mp4")
                bundle.write(manifest_path, "manifest.json")
            with self.assertRaisesRegex(RuntimeError, "missing required members"):
                video_delivery._validate_delivery_package(
                    package, manifest_path=manifest_path, final_path=final,
                    required_members={"final.mp4", "manifest.json", "episode.json"},
                )

    def test_delivery_package_validation_binds_manifest_and_final(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            final = root / "final.mp4"
            manifest_path = root / "manifest.json"
            package = root / "delivery.zip"
            final.write_bytes(b"verified-video")
            manifest_path.write_text(
                json.dumps({"release_status": "approved"}), encoding="utf-8",
            )
            import zipfile
            with zipfile.ZipFile(package, "w") as bundle:
                bundle.write(final, "final.mp4")
                bundle.writestr("manifest.json", json.dumps({"release_status": "pending"}))
            with self.assertRaisesRegex(RuntimeError, "manifest does not match"):
                video_delivery._validate_delivery_package(
                    package, manifest_path=manifest_path, final_path=final,
                    required_members={"final.mp4", "manifest.json"},
                )

    def test_h3_resource_release_is_fail_closed_while_queue_is_not_empty(self):
        calls = []

        def api(path, payload):
            calls.append((path, payload))
            if path == "/queue":
                return {"queue_running": [[0, "active"]], "queue_pending": []}
            return {}

        result = renderer.release_comfy_resources(api_func=api)
        self.assertFalse(result["released"])
        self.assertEqual(calls, [("/queue", None)])

    def test_comfy_queue_state_distinguishes_gpu_running_and_pending_position(self):
        queue = {
            "queue_running": [[7, "gpu-active", {}, {}]],
            "queue_pending": [
                [8, "ahead", {}, {}],
                [9, "target", {}, {}],
                [10, "behind", {}, {}],
            ],
        }
        self.assertEqual(renderer.comfy_queue_state("gpu-active", queue), {
            "state": "running", "position": 0, "pending_total": 3, "error": None,
        })
        self.assertEqual(renderer.comfy_queue_state("target", queue), {
            "state": "pending", "position": 2, "pending_total": 3, "error": None,
        })
        self.assertEqual(
            renderer.comfy_queue_state("missing", queue)["state"],
            "absent_or_history_pending",
        )

    def test_h3_resource_release_unloads_models_only_on_empty_queue(self):
        calls = []

        def api(path, payload):
            calls.append((path, payload))
            return {"queue_running": [], "queue_pending": []} if path == "/queue" else {}

        result = renderer.release_comfy_resources(api_func=api)
        self.assertTrue(result["released"])
        self.assertEqual(calls[1], (
            "/free", {"unload_models": True, "free_memory": True},
        ))

    def test_delivery_probe_rejects_extra_streams_and_av_drift(self):
        base = {
            "video": {
                "codec": "h264", "width": 720, "height": 1280, "fps": 30.0,
                "pixel_format": "yuv420p", "duration_seconds": 4.0,
            },
            "audio": {
                "codec": "aac", "sample_rate": 48000, "channels": 2,
                "duration_seconds": 4.0,
            },
            "stream_counts": {"video": 2, "audio": 1},
        }
        with self.assertRaisesRegex(ValueError, "video stream count mismatch"):
            video_delivery.validate_probe(base)
        drifted = json.loads(json.dumps(base))
        drifted["stream_counts"]["video"] = 1
        drifted["audio"]["duration_seconds"] = 4.25
        with self.assertRaisesRegex(ValueError, "audio/video duration mismatch"):
            video_delivery.validate_probe(drifted)

    def test_motivated_cut_evidence_validates_action_and_match_but_blocks_sound_bridge(self):
        def job(number):
            return {
                "job_id": f"ep:panel{number}", "panel_index": number,
                "panel_name": f"panel{number}", "metadata": {"inputs": {"shot_plan": {}}},
            }

        jobs = [job(1), job(2), job(3)]
        episode = {"panels": [
            {"transition": {"type": "cut_on_action", "motivation": "continue the hand motion"}},
            {"transition": {"type": "match_cut", "motivation": "match the round prop"}},
            {"transition": {"type": "close", "motivation": "finish the sequence"}},
        ]}
        same = "a" * 64
        analyses = [
            {"perceptual_hashes": ["1" * 64, "2" * 64], "metrics": {"adjacent_luma_changes": [0.01]}},
            {"perceptual_hashes": ["3" * 64, same], "metrics": {"adjacent_luma_changes": [0.02]}},
            {"perceptual_hashes": [same, "b" * 64], "metrics": {"adjacent_luma_changes": [0.03]}},
        ]
        release_qa = [{
            "job_id": item["job_id"], "reanalysis": analyses[index],
            "edit_selection": {"selection_sha256": str(index + 1) * 64},
        } for index, item in enumerate(jobs)]
        validated = video_delivery._validate_motivated_transition_plan(
            episode, jobs, release_qa,
        )
        self.assertTrue(validated["passed"])
        self.assertEqual(validated["boundaries"][0]["normalized_type"], "cut_on_action")
        self.assertEqual(validated["boundaries"][1]["normalized_type"], "match_cut")
        self.assertGreaterEqual(
            validated["boundaries"][1]["evidence"]["boundary_perceptual_similarity"],
            video_delivery.MATCH_CUT_SIMILARITY_MIN,
        )
        episode["panels"][0]["sound_bridge"] = {"cue": "door slam", "lead_seconds": 0.1}
        blocked = video_delivery._validate_motivated_transition_plan(episode, jobs, release_qa)
        self.assertFalse(blocked["passed"])
        self.assertIn(
            "sound_bridge_requires_approved_audio_overlap_contract",
            blocked["boundaries"][0]["errors"],
        )
        episode["panels"][0].pop("sound_bridge")
        episode["panels"][0]["transition"]["type"] = "dissolve"
        blocked = video_delivery._validate_motivated_transition_plan(episode, jobs, release_qa)
        self.assertIn("dissolve_or_crossfade_forbidden", blocked["boundaries"][0]["errors"])

    def test_final_close_hard_cut_is_terminal_but_boundary_dependent_request_still_fails(self):
        jobs = [{
            "job_id": "ep:close", "panel_index": 1, "panel_name": "panel_close",
            "metadata": {"inputs": {"shot_plan": {"shot_role": "close"}}},
        }]
        release_qa = [{
            "job_id": "ep:close",
            "reanalysis": {"perceptual_hashes": ["a" * 64], "metrics": {"adjacent_luma_changes": [0.01]}},
            "edit_selection": {"selection_sha256": "1" * 64},
        }]
        episode = {"panels": [{
            "shot_role": "close",
            "transition": {"type": "hard_cut", "motivation": "causal beat advance"},
        }]}
        result = video_delivery._validate_motivated_transition_plan(episode, jobs, release_qa)
        self.assertTrue(result["passed"])
        self.assertEqual(result["boundaries"][0]["normalized_type"], "terminal_close")
        self.assertEqual(result["boundaries"][0]["status"], "terminal")
        self.assertEqual(result["boundaries"][0]["execution"], "none_terminal")

        episode["panels"][0]["transition"] = {
            "type": "match_cut", "motivation": "match into a missing next shot",
        }
        result = video_delivery._validate_motivated_transition_plan(episode, jobs, release_qa)
        self.assertFalse(result["passed"])
        self.assertIn(
            "requested_transition_has_no_following_shot",
            result["boundaries"][0]["errors"],
        )

    def test_generation_log_accepts_legacy_kwargs_and_optional_total(self):
        generation_log.append_record(project="ep_test", panel_name="panel_1", panel_idx=1, status="pending")
        self.assertEqual(generation_log.get_project_status("ep_test"), {1: "pending"})

    def test_render_service_exports_facade_and_start_worker_builds_hidden_command(self):
        self.assertIs(render_service.start_worker, worker.start_worker)
        with mock.patch.object(worker.subprocess, "Popen") as popen:
            popen.return_value.pid = 4321
            result = worker.start_worker("ep_test", ensure_character_assets=False, timeout=30)
            character_result = render_service.prepare_character_assets("ep_character", timeout=30)
        self.assertTrue(result["started"])
        self.assertTrue(character_result["started"])
        full_command = popen.call_args_list[0].args[0]
        character_command = popen.call_args_list[1].args[0]
        self.assertIn("--ep-id", full_command)
        self.assertIn("--no-character-assets", full_command)
        self.assertNotIn("--character-assets-only", full_command)
        self.assertIn("--character-assets-only", character_command)
        self.assertEqual(popen.call_args_list[-1].kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(popen.call_args_list[0].kwargs["stdin"], worker.subprocess.DEVNULL)

    def test_night_worker_command_limits_one_job_without_changing_default_command(self):
        with mock.patch.object(worker.subprocess, "Popen") as popen:
            popen.return_value.pid = 9876
            limited = worker.start_worker("ep_night", max_jobs=1)
            normal = worker.start_worker("ep_normal")
        self.assertTrue(limited["started"])
        self.assertTrue(normal["started"])
        limited_command = popen.call_args_list[0].args[0]
        normal_command = popen.call_args_list[1].args[0]
        self.assertEqual(limited_command[limited_command.index("--max-jobs") + 1], "1")
        self.assertNotIn("--max-jobs", normal_command)


if __name__ == "__main__":
    unittest.main()
