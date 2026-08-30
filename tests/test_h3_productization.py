from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

import h3_director
import h3_profiles
import orchestrator
import task_store


class H3ProductizationTests(unittest.TestCase):
    def test_worker_stops_after_each_non_deliverable_proof_for_human_review(self):
        proof = {
            "status": "succeeded",
            "metadata": {
                "render_profile": "proof",
                "delivery_eligible": False,
                "preview_promotion": {"status": "pending"},
            },
        }
        self.assertTrue(orchestrator._requires_proof_human_gate(proof))
        approved = {
            **proof,
            "metadata": {
                **proof["metadata"],
                "preview_promotion": {"status": "approved"},
            },
        }
        self.assertFalse(orchestrator._requires_proof_human_gate(approved))
        production = {
            **proof,
            "metadata": {"render_profile": "production", "delivery_eligible": True},
        }
        self.assertFalse(orchestrator._requires_proof_human_gate(production))

    def test_multiple_views_of_one_character_compile_to_one_subject(self):
        definitions, _, retention, subject_map = h3_director._reference_sections(
            [
                {"model_label": "<Picture 1>", "role": "first_frame"},
                {"model_label": "<Picture 2>", "role": "character_anchor", "source_id": "char_rider"},
                {"model_label": "<Picture 3>", "role": "character_reference", "source_id": "char_rider"},
                {"model_label": "<Picture 4>", "role": "character_reference", "source_id": "char_rider"},
            ],
            cast_count=1,
        )
        self.assertEqual(subject_map, {"char_rider": "<Subject 1>"})
        self.assertEqual(definitions.count("<Subject 1> is the approved character char_rider"), 1)
        self.assertEqual(definitions.count("additional approved view of <Subject 1>"), 2)
        self.assertNotIn("<Subject 2>", definitions + retention)
        self.assertIn("exactly 1 distinct visible character", definitions)
        self.assertIn("must never create another subject", definitions)

    def test_profiles_make_proof_non_deliverable_and_low_cost(self):
        proof = h3_profiles.resolve_render_profile(h3_profiles.PROOF_THEN_PRODUCTION)
        self.assertEqual(proof["stage"], "proof")
        self.assertFalse(proof["delivery_eligible"])
        self.assertEqual(proof["frame_count"], 124)
        self.assertEqual(proof["turbo_steps"], 6)
        self.assertEqual(proof["ref_image_size"], "match")

        promoted = h3_profiles.resolve_render_profile(
            h3_profiles.PROOF_THEN_PRODUCTION,
            metadata={"preview_promotion": {
                "status": "approved",
                "artifact_sha256": "a", "decoded_visual_sha256": "v",
                "prompt_sha256": "p", "reference_bundle_sha256": "r",
            }},
        )
        self.assertEqual(promoted["stage"], "production")
        self.assertTrue(promoted["delivery_eligible"])
        self.assertEqual(promoted["frame_count"], 243)
        self.assertEqual(promoted["megapixels"], 0.9)
        self.assertEqual(promoted["turbo_steps"], 8)
        self.assertEqual(promoted["ref_image_size"], "max")
        self.assertLess(h3_profiles.profile_cost_summary()["proof_relative_compute"], 0.3)

    def test_director_compiler_uses_official_ref2va_order_and_exact_dialogue(self):
        result = h3_director.compile_h3_director_prompt(
            style="2D animated drama with clean linework and motivated practical lighting",
            aspect_ratio="9:16", duration_seconds=124 / 24,
            scene="a locked rain-night convenience store counter with one donation box",
            action=(
                "Beginning with the wallet beside the register, char_clerk hands the wallet "
                "to char_delivery, ending with the wallet held by char_delivery"
            ),
            first_state="the wallet rests beside the clerk's right hand",
            final_state="the courier holds the wallet while the clerk's hand is empty",
            camera="one unbroken eye-level medium shot with a gentle slow push in",
            continuity="Keep the two identities, wardrobe, counter layout, rain direction and eyeline unchanged",
            cast_count=2,
            spoken_dialogue=[{
                "speaker_id": "char_clerk", "text": "这是你的钱包。",
                "start_seconds": 1.0, "end_seconds": 2.2,
                "delivery_style": "calm and clear",
            }],
            audio_cues=[{
                "prompt": "the entrance bell rings once",
                "start_seconds": 0.2, "end_seconds": 0.6,
            }],
            ambience="steady rain taps the window over a low refrigerator hum",
            music="sparse piano notes at a slow tempo with a short final fade",
            bindings=[
                {"model_label": "<Picture 1>", "role": "first_frame"},
                {"model_label": "<Picture 2>", "role": "character_anchor", "source_id": "char_clerk"},
                {"model_label": "<Picture 3>", "role": "character_reference", "source_id": "char_delivery"},
                {"model_label": "<Picture 4>", "role": "scene_reference"},
            ],
        )
        prompt = result["prompt"]
        section_positions = [prompt.index(name + ":") for name in result["sections"]]
        self.assertEqual(section_positions, sorted(section_positions))
        self.assertTrue(prompt.startswith("subject_definitions:"))
        self.assertIn("[reference generation]", prompt)
        self.assertIn("fully_preserved", prompt)
        self.assertIn("[Shot 1]", prompt)
        self.assertEqual(prompt.count("这是你的钱包。"), 1)
        self.assertIn("<Subject 1> is the approved character char_clerk", prompt)
        self.assertIn("<Subject 2> is the approved character char_delivery", prompt)
        self.assertIn("<Subject 1> hands the wallet to <Subject 2>", prompt)
        self.assertIn("<Subject 1> (S1) says", prompt)
        self.assertNotIn("char_clerk hands the wallet", prompt)
        self.assertIn("Every visible surface is uniformly blank and unlettered", prompt)
        self.assertIn("using simple geometric color fields", prompt)
        self.assertEqual(prompt.count("the entrance bell rings once"), 1)
        self.assertEqual(result["prompt_sha256"], hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        self.assertEqual(result["skill_version"], h3_director.H3_DIRECTOR_SKILL_VERSION)

    def test_director_routes_music_cues_out_of_soundscape(self):
        result = h3_director.compile_h3_director_prompt(
            style="cinematic 2D animation with clean linework",
            aspect_ratio="9:16", duration_seconds=5.0,
            scene="a dry convenience store interior during a rainy night",
            action="the rider places one coin into the transparent donation box",
            first_state="the coin rests in the rider's open palm",
            final_state="the coin rests at the bottom of the donation box",
            camera="one stable close shot",
            continuity="rain remains outside and every interior surface stays dry",
            cast_count=1,
            ambience="quiet refrigerator hum",
            music="sparse soft piano",
            audio_cues=[
                {"cue_type": "sfx", "prompt": "one coin clinks", "start_seconds": 1.0, "end_seconds": 1.4},
                {"cue_type": "ambience", "prompt": "rain intensifies outside", "start_seconds": 2.0, "end_seconds": 3.0},
                {"cue_type": "music", "prompt": "one gentle piano lift", "start_seconds": 3.0, "end_seconds": 4.0},
            ],
        )
        soundscape = result["prompt"].split("overall_soundscape:", 1)[1].split("non_diegetic_music:", 1)[0]
        music = result["prompt"].split("non_diegetic_music:", 1)[1]
        self.assertIn("one coin clinks", soundscape)
        self.assertIn("Ambient layer", soundscape)
        self.assertIn("rain intensifies outside", soundscape)
        self.assertNotIn("one gentle piano lift", soundscape)
        self.assertIn("one gentle piano lift", music)

    def test_director_completes_story_action_inside_approved_edit_window(self):
        result = h3_director.compile_h3_director_prompt(
            style="cinematic 2D animation", aspect_ratio="9:16",
            duration_seconds=5.167, narrative_duration_seconds=1.6,
            scene="a dry neighborhood shop interior",
            action="the clerk immediately hands the wallet to the rider",
            first_state="the clerk holds the wallet",
            final_state="the rider grips the wallet and the clerk releases it",
            camera="one stable eye-level medium shot",
            continuity=(
                "preserve both identities, wardrobe, counter layout, practical lighting, "
                "screen direction, prop ownership, and eyelines without substitution"
            ),
            cast_count=2,
        )
        prompt = result["prompt"]
        self.assertIn("delivery edit cuts at 1.600 seconds", prompt)
        self.assertIn("0.000-0.500 seconds", prompt)
        self.assertIn("0.500-1.350 seconds", prompt)
        self.assertIn("by 1.350 seconds", prompt)
        self.assertIn("Hold unchanged through 5.167 seconds", prompt)

    def test_director_rejects_narrative_duration_outside_edit_contract(self):
        base = {
            "style": "cinematic 2D animation with practical motivated lighting",
            "aspect_ratio": "9:16", "duration_seconds": 5.167,
            "scene": "a dry neighborhood shop interior with a locked counter layout",
            "action": "the clerk immediately hands the wallet to the rider",
            "first_state": "the clerk holds the wallet beside the rider",
            "final_state": "the rider grips the wallet and the clerk releases it",
            "camera": "one stable eye-level medium shot with preserved screen direction",
            "continuity": "preserve identities, wardrobe, lighting, props, eyelines, and scene geography",
            "cast_count": 2,
        }
        for invalid in (0, 1.499, 5.168):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                h3_director.compile_h3_director_prompt(
                    **base, narrative_duration_seconds=invalid,
                )
        exact = h3_director.compile_h3_director_prompt(
            **base, narrative_duration_seconds=1.5,
        )["prompt"]
        self.assertIn("delivery edit cuts at 1.500 seconds", exact)
        self.assertIn("by 1.250 seconds", exact)

    def test_preview_promotion_preserves_proof_and_queues_fresh_production(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = task_store.RenderJobStore(root / "state" / "jobs.sqlite3")
            episode = {
                "ep_id": "ep_proof",
                "aspect_ratio": "9:16",
                "render_settings": {
                    "production_strategy": h3_profiles.PROOF_THEN_PRODUCTION,
                },
                "panels": [{
                    "panel_id": "panel_01",
                    "character_ids": [],
                    "prompt_package": {
                        "positive_prompt": "approved prompt contract",
                        "render_settings": {
                            "production_strategy": h3_profiles.PROOF_THEN_PRODUCTION,
                        },
                    },
                    "source_generation_duration_seconds": 10.125,
                    "edit_duration_seconds": 2.0,
                }],
            }
            previous_default = task_store._default_store
            try:
                task_store._default_store = store
                with mock.patch.object(task_store, "projects_dir", return_value=root / "projects"), \
                     mock.patch.object(task_store, "render_job_db", return_value=store.path):
                    snapshot = task_store.prepare_contract("ep_proof", episode)
                    job = snapshot["jobs"][0]
                    settings = job["metadata"]["settings"]
                    self.assertEqual(settings["render_profile"], "proof")
                    self.assertFalse(settings["delivery_eligible"])
                    self.assertIn("previews", Path(job["output_path"]).parts)

                    output = Path(job["output_path"])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"immutable proof artifact")
                    artifact_sha = hashlib.sha256(output.read_bytes()).hexdigest()
                    selection_sha = "selection-proof-sha"
                    metadata = dict(job["metadata"])
                    metadata.update({
                        "prompt_sha256": "prompt-proof-sha",
                        "reference_bundle_sha256": "refs-proof-sha",
                        "director_skill_version": h3_director.H3_DIRECTOR_SKILL_VERSION,
                        "content_qa": {
                            "passed": True,
                            "analysis": {"decoded_visual_sha256": "decoded-proof-sha"},
                        },
                        "edit_selection": {
                            "selection_sha256": selection_sha,
                            "source_artifact_sha256": artifact_sha,
                            "in_seconds": 0.5, "out_seconds": 2.5,
                            "duration_seconds": 2.0,
                        },
                    })
                    metadata["settings"] = {
                        **settings,
                        "prompt_sha256": "prompt-proof-sha",
                        "reference_bundle_sha256": "refs-proof-sha",
                    }
                    store.update_job(
                        job["job_id"], status="succeeded", progress=1.0,
                        output_path=str(output), preview_path=str(output),
                        metadata=metadata,
                    )
                    with self.assertRaisesRegex(RuntimeError, "proof renders cannot"):
                        task_store.approve_job_review(
                            "ep_proof", job["job_id"],
                            expected_artifact_sha256=artifact_sha,
                            expected_edit_selection_sha256=selection_sha,
                        )

                    promoted = task_store.approve_preview_and_promote(
                        "ep_proof", job["job_id"],
                        expected_artifact_sha256=artifact_sha,
                        expected_edit_selection_sha256=selection_sha,
                    )
                    production = promoted["jobs"][0]
                    production_settings = production["metadata"]["settings"]
                    self.assertEqual(production["status"], "queued")
                    self.assertEqual(production_settings["render_profile"], "production")
                    self.assertTrue(production_settings["delivery_eligible"])
                    self.assertIn("videos", Path(production["output_path"]).parts)
                    self.assertTrue(output.is_file(), "approved proof must remain available for audit")
                    self.assertEqual(
                        production["metadata"]["preview_promotion"]["artifact_sha256"],
                        artifact_sha,
                    )
                    self.assertNotIn("content_qa", production["metadata"])
            finally:
                task_store._default_store = previous_default


if __name__ == "__main__":
    unittest.main()
