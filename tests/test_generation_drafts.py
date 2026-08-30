from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

import generation_drafts
import story_splitter


def _valid_two_stage_response(target_seconds: float = 8.0, shot_count: int = 5):
    roles = ["hook", "setup", "escalation", "reversal", "close"]
    stage1 = {
        "title": "Door",
        "story_bible": {
            "title": "Door", "logline": "A courier crosses a room.",
            "synopsis": "A courier opens a door and reaches a table.",
            "themes": ["resolve"], "continuity_rules": ["blue jacket remains blue"],
        },
        "character_bible": [{
            "character_id": "char_a", "name": "A",
            "identity_prompt": "young adult courier with short black hair",
            "model_identity_tags_en": ["1boy", "young adult", "short black hair", "brown eyes"],
            "model_wardrobe_tags_en": ["blue canvas jacket", "black shoes"],
            "wardrobe_lock": ["blue canvas jacket", "black shoes"],
            "voice_profile": {"language": "English", "age": "young adult", "timbre": "warm", "pace": "medium"},
        }],
        "visual_bible": {
            "style_prompt": "premium cinematic comic animation",
            "global_negative_prompt": "text, watermark, collage",
        },
        "scene_bible": [{
            "scene_id": "scene_room", "description": "small modern room at sunset",
            "model_prompt_en": "small modern room at sunset, one wooden table, grounded lighting",
        }],
        "story_beats": [{
            "beat_id": f"beat_{role}", "role": role,
            "dramatic_question": f"question {role}", "visible_proof": f"proof {role}",
            "payoff_or_hook": f"payoff {role}",
        } for role in roles],
    }
    panels = []
    previous_id = None
    previous_state = {}
    duration = target_seconds / shot_count
    for index in range(shot_count):
        role = roles[index]
        panel_id = f"ep01_panel{index + 1:02d}"
        final_state = {"characters": f"courier at marker {index + 1}"}
        panels.append({
            "panel_id": panel_id, "name": panel_id, "scene_id": "scene_room",
            "character_ids": ["char_a"], "continuity_group": "main",
            "previous_panel_id": previous_id, "continuity_state_in": previous_state,
            "continuity_state_out": final_state,
            "source_generation_duration_seconds": 10.125,
            "edit_duration_seconds": duration, "shot_role": role,
            "story_beat_id": f"beat_{role}",
            "visible_action": f"Courier pushes door {index + 1} open until it touches the wall",
            "first_state": f"door {index + 1} closed",
            "final_state": f"door {index + 1} open",
            "cause": "a sound forces movement", "next_hook": "the next obstacle appears",
            "first_frame": f"courier faces door {index + 1}",
            "last_frame": f"courier passes door {index + 1}",
            "camera_plan": {"shot_size": f"medium {index}", "angle": "eye level", "movement": "slow push", "composition": f"door on third {index}"},
            "transition": {"type": "hard_cut", "motivation": "causal action advance"},
            "edit_hint": {"preferred_moment": "door moves", "edit_in_hint": "hand reaches", "edit_out_hint": "foot lands"},
            "priority": "must_have", "group_shot_reason": "",
            "spoken_dialogue": [], "subtitle_timeline": [], "on_screen_text": [],
            "audio_cues": [], "sfx": [], "transitions": [],
            "cuts": [{"time_range": "0-10.125s", "name": role, "intensity": "SMOOTH", "shot_description": "A steady shot records the courier crossing the doorway."}],
        })
        previous_id, previous_state = panel_id, final_state
    return stage1, {"panels": panels}


class GenerationDraftTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.drafts = Path(self.tempdir.name) / "drafts"
        self.stage1 = {
            "story_bible": {"title": "T", "synopsis": "S"},
            "character_bible": [{"character_id": "char_a", "name": "A"}],
            "scene_bible": [{"scene_id": "scene_a", "model_prompt_en": "modern room"}],
            "story_beats": [{"beat_id": "beat_hook", "role": "hook"}],
        }
        self.brief = {"topic": "future letter", "synopsis": "a courier finds a letter"}
        self.settings = {"shot_count": 8, "target_edit_duration_seconds": 30.0}

    def tearDown(self):
        self.tempdir.cleanup()

    def _save(self):
        return generation_drafts.save_stage1_checkpoint(
            "ep_draft", self.stage1, creative_brief=self.brief,
            settings=self.settings, protocol="anthropic", model="MiniMax-M2.7",
            draft_dir=self.drafts,
        )

    def test_atomic_checkpoint_is_hash_bound_unregistered_and_contains_no_provider_data(self):
        saved = self._save()
        path = Path(saved["checkpoint_path"])
        self.assertTrue(path.is_file())
        self.assertEqual(path.name, f"v3_stage1.{saved['checkpoint_sha256']}.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["lifecycle"]["stage1_status"], "validated")
        self.assertEqual(payload["lifecycle"]["registration_status"], "unregistered")
        self.assertEqual(payload["lifecycle"]["approval_status"], "not_approved")
        self.assertEqual(payload["binding"]["model"], "MiniMax-M2.7")
        serialized = json.dumps(payload).casefold()
        for forbidden in ("api_key", "authorization", "raw_response", "reasoning", "thinking"):
            self.assertNotIn(forbidden, serialized)
        self.assertEqual(self._save()["created_at"], saved["created_at"])
        self.assertEqual(list(self.drafts.glob("*.tmp")), [])
        summaries = generation_drafts.list_stage1_checkpoints(
            "ep_draft", draft_dir=self.drafts,
        )
        self.assertEqual(len(summaries), 1)
        self.assertNotIn("validated_stage1", summaries[0])
        self.assertEqual(summaries[0]["stage2_status"], "pending")

    def test_resume_binding_fails_closed_on_brief_settings_protocol_or_model_change(self):
        saved = self._save()
        common = {
            "ep_id": "ep_draft", "checkpoint_hash": saved["checkpoint_sha256"],
            "creative_brief": self.brief, "settings": self.settings,
            "protocol": "anthropic", "model": "MiniMax-M2.7", "draft_dir": self.drafts,
        }
        loaded = generation_drafts.load_stage1_checkpoint(**common)
        self.assertEqual(loaded["validated_stage1"], self.stage1)
        changes = (
            {"creative_brief": {**self.brief, "topic": "changed"}},
            {"settings": {**self.settings, "shot_count": 9}},
            {"protocol": "openai"},
            {"model": "MiniMax-M2.8"},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, "stale"):
                generation_drafts.load_stage1_checkpoint(**{**common, **change})

    def test_match_checkpoint_restores_latest_only_for_exact_persisted_binding(self):
        first = self._save()
        generation_drafts.record_stage2_status(
            "ep_draft", first["checkpoint_sha256"], status="running", draft_dir=self.drafts,
        )
        generation_drafts.record_stage2_status(
            "ep_draft", first["checkpoint_sha256"], status="failed",
            error_code="stage2_contract_invalid", draft_dir=self.drafts,
        )
        matched = generation_drafts.match_stage1_checkpoint(
            "ep_draft", creative_brief=self.brief, settings=self.settings,
            protocol="anthropic", model="MiniMax-M2.7", draft_dir=self.drafts,
        )
        self.assertEqual(matched["checkpoint_sha256"], first["checkpoint_sha256"])
        self.assertEqual(matched["stage2_status"], "failed")
        self.assertEqual(matched["ep_id"], "ep_draft")
        self.assertNotIn("validated_stage1", matched)
        for changed in (
            {"creative_brief": {**self.brief, "topic": "changed"}},
            {"settings": {**self.settings, "shot_count": 9}},
            {"protocol": "openai"},
            {"model": "MiniMax-M2.8"},
        ):
            arguments = {
                "creative_brief": self.brief, "settings": self.settings,
                "protocol": "anthropic", "model": "MiniMax-M2.7",
                "draft_dir": self.drafts, **changed,
            }
            with self.subTest(changed=changed):
                self.assertEqual(
                    generation_drafts.match_stage1_checkpoint("ep_draft", **arguments), {},
                )

        second = generation_drafts.save_stage1_checkpoint(
            "ep_draft", {**self.stage1, "story_bible": {"title": "T2", "synopsis": "S2"}},
            creative_brief=self.brief, settings=self.settings,
            protocol="anthropic", model="MiniMax-M2.7", draft_dir=self.drafts,
        )
        matched_latest = generation_drafts.match_stage1_checkpoint(
            "ep_draft", creative_brief=self.brief, settings=self.settings,
            protocol="anthropic", model="MiniMax-M2.7", draft_dir=self.drafts,
        )
        self.assertEqual(matched_latest["checkpoint_sha256"], second["checkpoint_sha256"])

    def test_sensitive_stage1_fields_and_provider_error_text_are_rejected(self):
        for field in ("api_key", "reasoning", "raw_response", "thinking"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "forbidden"):
                generation_drafts.save_stage1_checkpoint(
                    "ep_draft", {**self.stage1, field: "must-not-persist"},
                    creative_brief=self.brief, settings=self.settings,
                    protocol="anthropic", model="MiniMax-M2.7", draft_dir=self.drafts,
                )
        saved = self._save()
        generation_drafts.record_stage2_status(
            "ep_draft", saved["checkpoint_sha256"], status="running", draft_dir=self.drafts,
        )
        with self.assertRaisesRegex(ValueError, "opaque safe code"):
            generation_drafts.record_stage2_status(
                "ep_draft", saved["checkpoint_sha256"], status="failed",
                error_code="provider said: here is its raw output with spaces",
                draft_dir=self.drafts,
            )
        failed = generation_drafts.record_stage2_status(
            "ep_draft", saved["checkpoint_sha256"], status="failed",
            error_code="stage2_contract_invalid", draft_dir=self.drafts,
        )
        self.assertEqual(failed["lifecycle"]["stage2_attempt_count"], 1)
        self.assertEqual(failed["lifecycle"]["last_stage2_error_code"], "stage2_contract_invalid")

    def test_stage2_failure_resumes_checkpoint_without_repeating_stage1(self):
        stage1, stage2 = _valid_two_stage_response()
        inputs = {
            "story_text": "A courier opens a door and reaches a table.",
            "ep_id": "ep_resume", "draft_dir": self.drafts,
            "api_key": "offline-only", "total_duration_seconds": 8,
            "shot_count": 5, "min_panels": 5, "max_panels": 5,
        }
        with patch.object(
            story_splitter, "_call_m3",
            side_effect=[json.dumps(stage1), "not-json"],
        ) as initial_calls:
            with self.assertRaises(story_splitter.MiniMaxGenerationStageError):
                story_splitter.split_story(**inputs)
        self.assertEqual(initial_calls.call_count, 2)
        checkpoint_file = next(self.drafts.glob("v3_stage1.*.json"))
        failed = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        self.assertEqual(failed["lifecycle"]["stage2_status"], "failed")
        self.assertEqual(failed["lifecycle"]["last_stage2_error_code"], "stage2_parse_failed")

        with patch.object(
            story_splitter, "_call_m3", return_value=json.dumps(stage2),
        ) as resumed_calls:
            episode = story_splitter.resume_story_stage2(
                inputs["story_text"], ep_id=inputs["ep_id"],
                checkpoint_hash=failed["checkpoint_sha256"], draft_dir=self.drafts,
                api_key=inputs["api_key"], total_duration_seconds=8,
                shot_count=5, min_panels=5, max_panels=5,
            )
        self.assertEqual(resumed_calls.call_count, 1)
        self.assertEqual(resumed_calls.call_args.kwargs["tool_name"], "submit_v3_stage2")
        self.assertEqual(len(episode["panels"]), 5)
        completed = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        self.assertEqual(completed["lifecycle"]["stage2_status"], "completed")
        self.assertEqual(completed["lifecycle"]["stage2_attempt_count"], 2)
        self.assertEqual(
            episode["generation_plan"]["stage1_checkpoint"]["approval_status"],
            "not_approved",
        )

        with patch.object(story_splitter, "_call_m3") as no_paid_call:
            with self.assertRaisesRegex(RuntimeError, "stale"):
                story_splitter.resume_story_stage2(
                    inputs["story_text"] + " changed", ep_id=inputs["ep_id"],
                    checkpoint_hash=failed["checkpoint_sha256"], draft_dir=self.drafts,
                    api_key=inputs["api_key"], total_duration_seconds=8,
                    shot_count=5, min_panels=5, max_panels=5,
                )
        no_paid_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
