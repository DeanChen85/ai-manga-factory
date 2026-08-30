import copy
import json
import os
import socket
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import story_splitter


def minimal_llm_response():
    return {
        "title": "One Shot",
        "subtitle": "A hero enters",
        "story_bible": {
            "title": "One Shot",
            "logline": "A hero must enter the room.",
            "synopsis": "A hero enters a room and reaches the table.",
            "target_audience": "young adult",
        },
        "character_bible": [{
            "character_id": "char_hero",
            "name": "Hero",
            "identity_prompt": "young adult with short black hair, brown eyes and a blue canvas jacket",
            "wardrobe_lock": {"outfit": "blue canvas jacket", "footwear": "black shoes"},
            "model_identity_tags_en": [
                "1boy", "male", "young adult", "short hair", "black hair", "brown eyes"
            ],
            "model_wardrobe_tags_en": ["blue canvas jacket", "black shoes"],
            "voice_profile": {"language": "English", "age": "young adult", "timbre": "warm", "pace": "medium"},
        }],
        "visual_bible": {
            "style_prompt": "premium cinematic comic animation",
            "global_negative_prompt": "identity drift, wardrobe drift, random text, logo, watermark",
        },
        "scene_bible": [{
            "scene_id": "scene_room",
            "description": "small room at sunset with one wooden table",
            "model_prompt_en": "small contemporary room at sunset, one wooden table, grounded lighting",
        }],
        "panels": [{
            "panel_id": "ep01_panel01_entry",
            "name": "ep01_panel01_entry",
            "scene_id": "scene_room",
            "character_ids": ["char_hero"],
            "continuity_group": "main",
            "previous_panel_id": None,
            "continuity_state_in": {},
            "continuity_state_out": {"characters": "hero beside table"},
            "first_frame": "char_hero outside door",
            "last_frame": "char_hero beside table",
            "cuts": [{
                "time_range": "0-10s",
                "name": "entry",
                "intensity": "SMOOTH",
                "shot_description": "A steady eye-level shot follows char_hero through the door and to the table",
            }],
            "transitions": [],
            "spoken_dialogue": [],
            "subtitle_timeline": [],
            "on_screen_text": [],
            "audio_cues": [],
            "sfx": [{"time_range": "1-2s", "tag": "STEP"}],
        }],
    }


def platform_llm_response(target_seconds=8.0, shot_count=5):
    response = minimal_llm_response()
    roles = ["hook", "setup", "escalation", "reversal", "close"]
    response["story_beats"] = [
        {
            "beat_id": f"beat_{role}", "role": role,
            "dramatic_question": f"question {role}",
            "visible_proof": f"visible proof {role}",
            "payoff_or_hook": f"payoff {role}",
        }
        for role in roles
    ]
    panels = []
    previous_id = None
    previous_state = {}
    duration = target_seconds / shot_count
    for index in range(shot_count):
        template = dict(response["panels"][0])
        panel_id = f"ep01_panel{index + 1:02d}_beat"
        role = roles[min(index, len(roles) - 1)]
        final_state = {"characters": f"hero completes action {index + 1}"}
        template.update({
            "panel_id": panel_id, "name": panel_id,
            "previous_panel_id": previous_id,
            "continuity_state_in": previous_state,
            "continuity_state_out": final_state,
            "source_generation_duration_seconds": 10.125,
            "edit_duration_seconds": duration,
            "shot_role": role, "story_beat_id": f"beat_{role}",
            "visible_action": f"Hero pushes door {index + 1} open until it stops against the wall",
            "first_state": f"door {index + 1} closed",
            "final_state": f"door {index + 1} open and hero one step closer",
            "cause": "A sound inside forces the hero to move",
            "next_hook": "The next object blocks the hero's path",
            "camera_plan": {
                "shot_size": f"shot size {index}", "angle": f"angle {index}",
                "movement": "controlled push", "composition": f"composition {index}",
            },
            "transition": {"type": "hard_cut", "motivation": "causal action advance"},
            "edit_hint": {
                "preferred_moment": "door starts opening", "edit_in_hint": "hand reaches handle",
                "edit_out_hint": "foot lands inside",
            },
            "priority": "must_have", "group_shot_reason": "",
            "audio_cues": [], "sfx": [],
            "first_frame": f"hero faces closed door {index + 1}",
            "last_frame": f"hero steps through open door {index + 1}",
        })
        template["cuts"] = [{
            "time_range": "0-10.125s", "name": f"beat {index}", "intensity": "SMOOTH",
            "shot_description": f"A steady eye-level shot {index} follows the hero opening the door and stepping toward the table",
        }]
        panels.append(template)
        previous_id = panel_id
        previous_state = final_state
    response["panels"] = panels
    return response


class StorySplitterV2Tests(unittest.TestCase):
    def test_repository_config_uses_official_defaults_without_a_key(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "pipeline" / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["minimax_api_key"], "")
        self.assertEqual(config["minimax_protocol"], "anthropic")
        self.assertEqual(config["minimax_base_url"], "https://api.minimaxi.com/anthropic")
        self.assertEqual(config["minimax_model"], "MiniMax-M2.7")
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://api.minimaxi.com/anthropic/v1/messages", readme)
        self.assertIn("https://api.minimax.io/anthropic", readme)
        self.assertIn("stop_reason=max_tokens", readme)

    def test_official_minimax_defaults_and_endpoint_normalization(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(story_splitter, "M3_BASE_URL", "https://api.minimaxi.com/v1"), \
             patch.object(story_splitter, "M3_MODEL", "MiniMax-M2.7"):
            status = story_splitter.minimax_configuration_status()
        self.assertEqual(status["protocol"], "anthropic")
        self.assertEqual(status["endpoint"], "https://api.minimaxi.com/anthropic/v1/messages")
        self.assertEqual(status["model"], "MiniMax-M2.7")
        self.assertFalse(status["deprecated"])
        self.assertEqual(
            story_splitter.minimax_chat_completions_url(
                "https://api.minimaxi.com/v1/v1/chat/completions/"
            ),
            "https://api.minimaxi.com/v1/chat/completions",
        )
        self.assertEqual(
            story_splitter.minimax_chat_completions_url("https://api.minimax.io/v1"),
            "https://api.minimax.io/v1/chat/completions",
        )
        self.assertEqual(
            story_splitter.minimax_anthropic_messages_url("https://api.minimaxi.com/v1/chat/completions"),
            "https://api.minimaxi.com/anthropic/v1/messages",
        )

    def test_explicit_same_shop_language_is_a_single_scene_constraint(self):
        self.assertTrue(story_splitter.explicit_single_scene("两位人物只在同一家雨夜便利店。"))
        self.assertTrue(story_splitter.explicit_single_scene("全片只发生在同一间办公室"))
        self.assertFalse(story_splitter.explicit_single_scene("人物在同一座城市的多个地点行动"))

    def test_explicit_chinese_core_character_count_uses_fixed_slots(self):
        self.assertEqual(story_splitter.explicit_requested_character_count("两位核心人物"), 2)
        self.assertEqual(story_splitter.explicit_requested_character_count("核心角色十二位"), 12)

    def test_explicit_legacy_config_remains_allowed_but_is_deprecated(self):
        status = story_splitter.minimax_configuration_status(
            "https://api.minimax.chat/v1", "abab6.5s-chat",
        )
        self.assertTrue(status["deprecated"])
        self.assertEqual(len(status["warnings"]), 2)
        self.assertIn("api.minimax.chat", status["warnings"][0])
        self.assertIn("MiniMax-M2.7", status["warnings"][1])

    def test_official_request_uses_completion_budget_and_never_exposes_key(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps({
                    "choices": [{"message": {
                        "reasoning_content": "private reasoning",
                        "content": '<think>hidden</think>{"ok":true}',
                    }}],
                }).encode("utf-8")

        def fake_urlopen(request, **kwargs):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = kwargs["timeout"]
            return Response()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            content = story_splitter._call_m3(
                "system", "user", api_key="sentinel-secret",
                base_url="https://api.minimaxi.com/v1/", model="MiniMax-M2.7",
                protocol="openai", max_tokens=99999, timeout_seconds=10,
            )
        self.assertEqual(content, '{"ok":true}')
        self.assertEqual(captured["url"], "https://api.minimaxi.com/v1/chat/completions")
        self.assertEqual(captured["body"]["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", captured["body"])
        self.assertTrue(captured["body"]["reasoning_split"])
        self.assertIn("Return minified JSON only", captured["body"]["messages"][0]["content"])
        self.assertNotIn("sentinel-secret", json.dumps(captured["body"]))

    def test_anthropic_default_forces_named_tool_and_returns_only_tool_input(self):
        captured = {}
        tool_input = {"sb": {"t": "title"}, "cb": [], "vb": {}, "sc": [], "beats": []}

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps({
                    "content": [
                        {"type": "thinking", "thinking": "private reasoning", "signature": "opaque"},
                        {"type": "tool_use", "id": "tool_1", "name": "submit_v3_stage1", "input": tool_input},
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 100, "output_tokens": 900},
                }).encode("utf-8")

        def fake_urlopen(request, **kwargs):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = kwargs["timeout"]
            return Response()

        schema = {"type": "object", "required": ["sb"], "properties": {"sb": {"type": "object"}}}
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            raw = story_splitter._call_m3(
                "system", "user", api_key="sentinel-secret", timeout_seconds=10,
                base_url="https://api.minimaxi.com/anthropic",
                tool_name="submit_v3_stage1", tool_schema=schema,
            )
        self.assertEqual(json.loads(raw), tool_input)
        self.assertEqual(captured["url"], "https://api.minimaxi.com/anthropic/v1/messages")
        self.assertEqual(captured["body"]["max_tokens"], 8192)
        self.assertNotIn("max_completion_tokens", captured["body"])
        self.assertEqual(captured["body"]["tool_choice"], {"type": "tool", "name": "submit_v3_stage1"})
        self.assertEqual(captured["body"]["tools"][0]["input_schema"], schema)
        self.assertNotIn("sentinel-secret", json.dumps(captured["body"]))
        self.assertNotIn("Authorization", captured["headers"])

    def test_anthropic_forced_tool_never_falls_back_to_text(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps({
                    "content": [
                        {"type": "thinking", "thinking": "private"},
                        {"type": "text", "text": '{"looks":"valid but is forbidden"}'},
                    ],
                    "stop_reason": "end_turn",
                }).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, "未返回唯一目标 tool_use.*文本不会作为合同回退"):
                story_splitter._call_m3(
                    "system", "user", api_key="offline-only",
                    tool_name="submit_v3_stage1", tool_schema={"type": "object"},
                )

    def test_anthropic_max_tokens_stop_reason_fails_before_tool_input(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps({
                    "content": [{"type": "tool_use", "name": "submit_v3_stage1", "input": {"partial": True}}],
                    "stop_reason": "max_tokens", "usage": {"output_tokens": 8192},
                }).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(
                story_splitter.MiniMaxOutputTruncated,
                "stop_reason=max_tokens.*output_tokens=8192.*不会进入合同解析",
            ):
                story_splitter._call_m3(
                    "system", "user", api_key="offline-only",
                    tool_name="submit_v3_stage1", tool_schema={"type": "object"},
                )

    def test_v3_stage_tool_schemas_bind_exact_names_and_shot_count(self):
        response = platform_llm_response(target_seconds=20, shot_count=7)
        stage1 = dict(response)
        stage1.pop("panels")
        outputs = [json.dumps(stage1), json.dumps({"panels": response["panels"]})]
        calls = []

        def fake_call(*args, **kwargs):
            calls.append((args, kwargs))
            return outputs[len(calls) - 1]

        with patch.object(story_splitter, "_call_m3", side_effect=fake_call):
            story_splitter.split_story(
                "story", api_key="offline-only", total_duration_seconds=20,
                shot_count=7, min_panels=7, max_panels=7,
            )
        self.assertEqual(calls[0][1]["tool_name"], "submit_v3_stage1")
        self.assertEqual(calls[1][1]["tool_name"], "submit_v3_stage2")
        stage2_schema = calls[1][1]["tool_schema"]
        expected_slots = [f"p{index:02d}" for index in range(1, 8)]
        self.assertFalse(stage2_schema["additionalProperties"])
        self.assertEqual(stage2_schema["required"], expected_slots)
        self.assertNotIn("shots", stage2_schema["properties"])
        self.assertEqual(set(stage2_schema["properties"]), set(expected_slots))
        for slot in expected_slots:
            shot_schema = stage2_schema["properties"][slot]
            self.assertFalse(shot_schema["additionalProperties"])
            self.assertTrue({"act", "cam", "tr", "edit"}.issubset(shot_schema["required"]))
            visible_schema = shot_schema["properties"]["c"]
            self.assertTrue(visible_schema["uniqueItems"])
            self.assertEqual(visible_schema["minItems"], 1)
            self.assertEqual(
                visible_schema["items"]["enum"],
                [item["character_id"] for item in stage1["character_bible"]],
            )
            self.assertEqual(
                shot_schema["properties"]["s"]["enum"],
                [item["scene_id"] for item in stage1["scene_bible"]],
            )
            self.assertEqual(
                shot_schema["properties"]["b"]["enum"],
                [item["beat_id"] for item in stage1["story_beats"]],
            )
            self.assertEqual(shot_schema["properties"]["act"]["type"], "object")
            action_codes = shot_schema["properties"]["act"]["properties"]["code"]["enum"]
            self.assertIn("SLIDE_OBJECT", action_codes)
            self.assertIn("LOOK_AT", action_codes)
            self.assertNotIn("verb", shot_schema["properties"]["act"]["properties"])
            self.assertEqual(
                shot_schema["properties"]["act"]["required"],
                ["sub", "code", "obj"],
            )
            self.assertEqual(
                shot_schema["properties"]["act"]["properties"]["sub"]["enum"],
                [item["character_id"] for item in stage1["character_bible"]],
            )
            for object_name in ("cam", "tr", "edit"):
                self.assertEqual(shot_schema["properties"][object_name]["type"], "object")
        stage2_prompt = calls[1][0][0]
        self.assertIn("fixed shot slot p01 through p07", stage2_prompt)
        self.assertIn("action code, not natural-language verb wording", stage2_prompt)
        self.assertIn("Each character ID may appear at most once", stage2_prompt)

    def test_stage2_action_code_compiles_to_deterministic_visible_action(self):
        shot = {
            "id": "ep01_panel01", "r": "hook", "b": "beat_hook", "d": 4.0,
            "c": ["char_hero"], "s": "scene_room",
            "act": {
                "sub": "char_hero", "code": "PRESS_CONTROL",
                "obj": "alarm button",
            },
            "f": "alarm button untouched",
            "l": "alarm button remains depressed and indicator lit",
            "why": "danger approaches",
            "next": "the alarm reveals a hidden threat",
            "cam": {"size": "close-up", "angle": "eye-level", "move": "locked", "comp": "hand and button"},
            "tr": {"type": "close", "motivation": "terminal shot"},
            "edit": {"moment": "indicator lights", "in": "finger enters frame", "out": "indicator holds"},
            "pri": "must_have", "g": "", "si": {}, "so": {"alarm": "lit"}, "dlg": [], "aud": [],
        }
        expanded = story_splitter.expand_v3_stage2({"p01": shot}, shot_count=1)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["action_code"], "PRESS_CONTROL")
        self.assertEqual(expanded[0]["action_spec"]["catalog_version"], "ai-manga.action/v1")
        self.assertEqual(len(expanded[0]["action_spec"]["spec_sha256"]), 64)
        self.assertEqual(expanded[0]["action_components"]["verb"], "presses")
        self.assertEqual(
            expanded[0]["visible_action"],
            "Beginning with alarm button untouched, char_hero presses alarm button, "
            "ending with it remains depressed and indicator lit.",
        )
        self.assertNotIn("alarm button alarm button", expanded[0]["visible_action"])

    def test_stage2_action_code_rejects_unknown_code_without_guessing(self):
        payload = {
            "p01": {
                "c": ["char_hero"], "f": "hero is still", "l": "hero is hopeful",
                "act": {"sub": "char_hero", "code": "FEEL_BRAVE", "obj": "danger"},
                "cam": {"size": "CU", "angle": "eye", "move": "locked", "comp": "face"},
                "tr": {"type": "close", "motivation": "end"},
                "edit": {"moment": "end", "in": "start", "out": "finish"},
            }
        }
        _shots, errors = story_splitter._stage2_slots(payload, 1)
        self.assertTrue(any("approved catalog enum" in error for error in errors))

    def test_stage2_drop_object_requires_moving_object_and_destination(self):
        base = {
            "c": ["char_hero"], "f": "coins rest in open palm",
            "l": "coins settle at bottom of charity box",
            "cam": {"size": "CU", "angle": "eye", "move": "locked", "comp": "hand and box"},
            "tr": {"type": "close", "motivation": "end"},
            "edit": {"moment": "coins fall", "in": "open palm", "out": "coins settle"},
        }
        invalid = {
            **base,
            "act": {"sub": "char_hero", "code": "DROP_OBJECT", "obj": "charity box"},
        }
        _shots, errors = story_splitter._stage2_slots({"p01": invalid}, 1)
        self.assertTrue(any("moving object and destination" in error for error in errors))
        valid = {
            **base,
            "act": {
                "sub": "char_hero", "code": "DROP_OBJECT",
                "obj": "coins into charity box",
            },
        }
        _shots, errors = story_splitter._stage2_slots({"p01": valid}, 1)
        self.assertFalse(any("moving object and destination" in error for error in errors))

    def test_stage2_action_actor_is_deterministically_added_to_visible_cast(self):
        base = {
            "f": "button untouched", "l": "button lit",
            "act": {"sub": "char_other", "code": "PRESS_CONTROL", "obj": "button"},
            "cam": {"size": "CU", "angle": "eye", "move": "locked", "comp": "hand"},
            "tr": {"type": "close", "motivation": "end"},
            "edit": {"moment": "end", "in": "start", "out": "finish"},
        }
        for raw_visible in (["char_hero"], []):
            with self.subTest(raw_visible=raw_visible):
                payload = {"p01": {**base, "c": raw_visible}}
                _shots, errors = story_splitter._stage2_slots(payload, 1)
                self.assertFalse(any("actor_id must belong" in error for error in errors))
                self.assertFalse(any("at least one visible" in error for error in errors))
                panel = story_splitter.expand_v3_stage2(payload, shot_count=1)[0]
                self.assertIn("char_other", panel["character_ids"])

    def test_stage2_forced_tool_double_encoded_slot_is_unwrapped_then_validated(self):
        shot = {
            "c": ["char_hero"], "f": "button untouched", "l": "button lit",
            "act": {"sub": "char_hero", "code": "PRESS_CONTROL", "obj": "button"},
            "cam": {"size": "CU", "angle": "eye", "move": "locked", "comp": "hand"},
            "tr": {"type": "close", "motivation": "end"},
            "edit": {"moment": "end", "in": "start", "out": "finish"},
        }
        slots, errors = story_splitter._stage2_slots(
            {"p01": json.dumps(shot, ensure_ascii=False)}, 1,
        )
        self.assertEqual(errors, [])
        self.assertEqual(slots, [shot])

    def test_stage2_nested_action_states_are_promoted_only_when_unambiguous(self):
        shot = {
            "c": ["char_hero"],
            "act": {
                "sub": "char_hero", "code": "PRESS_CONTROL", "obj": "button",
                "f": "button untouched", "l": "button lit",
            },
            "cam": {"size": "CU", "angle": "eye", "move": "locked", "comp": "hand"},
            "tr": {"type": "close", "motivation": "end"},
            "edit": {"moment": "end", "in": "start", "out": "finish"},
        }
        slots, errors = story_splitter._stage2_slots({"p01": shot}, 1)
        self.assertEqual(errors, [])
        self.assertEqual(slots[0]["f"], "button untouched")
        self.assertEqual(slots[0]["l"], "button lit")
        self.assertNotIn("f", slots[0]["act"])

        conflicting = copy.deepcopy(shot)
        conflicting["f"] = "different start"
        _slots, conflict_errors = story_splitter._stage2_slots({"p01": conflicting}, 1)
        self.assertTrue(any("conflicts" in error for error in conflict_errors))

    def test_stage2_camera_transition_and_edit_long_or_encoded_objects_are_normalized(self):
        shot = {
            "c": ["char_hero"], "f": "button untouched", "l": "button lit",
            "act": {"sub": "char_hero", "code": "PRESS_CONTROL", "obj": "button"},
            "cam": json.dumps({
                "shot_size": "CU", "angle": "eye", "movement": "locked",
                "composition": "hand centered",
            }),
            "transition": {"type": "close", "motivation": "state change"},
            "edit_hint": {
                "preferred_moment": "button lights", "edit_in_hint": "hand enters",
                "edit_out_hint": "indicator holds",
            },
        }
        slots, errors = story_splitter._stage2_slots({"p01": shot}, 1)
        self.assertEqual(errors, [])
        self.assertEqual(slots[0]["cam"]["move"], "locked")
        self.assertEqual(slots[0]["tr"]["motivation"], "state change")
        self.assertEqual(slots[0]["edit"]["out"], "indicator holds")

    def test_stage2_canonical_actor_and_visible_field_names_are_normalized(self):
        shot = {
            "f": "button untouched", "l": "button lit",
            "visible_character_ids": ["char_hero"],
            "act": {"actor_id": "char_other", "code": "PRESS_CONTROL", "obj": "button"},
            "cam": {"size": "CU", "angle": "eye", "move": "locked", "comp": "hand"},
            "tr": {"type": "close", "motivation": "end"},
            "edit": {"moment": "end", "in": "start", "out": "finish"},
        }
        panel = story_splitter.expand_v3_stage2({"shots": [shot]}, shot_count=1)[0]
        self.assertEqual(panel["character_ids"], ["char_hero", "char_other"])
        self.assertEqual(panel["action_spec"]["actor_id"], "char_other")

    def test_stage2_full_panel_action_actor_is_added_to_visible_cast(self):
        panel = {
            "character_ids": ["char_hero"],
            "action_spec": {
                "actor_id": "char_other", "action_code": "PRESS_CONTROL",
                "target": "button", "start_state": "button untouched",
                "end_state": "button lit",
            },
        }
        expanded = story_splitter.expand_v3_stage2({"panels": [panel]}, 1)
        self.assertEqual(expanded[0]["character_ids"], ["char_hero", "char_other"])

    def test_stage2_normalized_action_actor_still_rejects_unknown_stage1_id(self):
        stage1 = platform_llm_response(target_seconds=8, shot_count=5)
        stage1.pop("panels")
        shot = {
            "id": "ep01_panel01", "r": "hook", "b": "beat_hook", "d": 1.6,
            "c": ["char_hero"], "s": "scene_room",
            "act": {"sub": "char_not_in_stage1", "code": "PRESS_CONTROL", "obj": "button"},
            "f": "button untouched", "l": "button remains depressed and indicator lit",
            "why": "danger approaches", "next": "the alarm reveals a threat",
            "cam": {"size": "close-up", "angle": "eye-level", "move": "locked", "comp": "hand and button"},
            "tr": {"type": "close", "motivation": "terminal shot"},
            "edit": {"moment": "indicator lights", "in": "finger enters", "out": "indicator holds"},
            "pri": "must_have", "g": "", "si": {}, "so": {"alarm": "lit"}, "dlg": [], "aud": [],
        }
        roles = ["hook", "setup", "escalation", "reversal", "close"]
        payload = {}
        for index, role in enumerate(roles, 1):
            item = copy.deepcopy(shot)
            item.update({
                "id": f"ep01_panel{index:02d}", "r": role,
                "b": f"beat_{role}",
                "f": f"button {index} untouched",
                "l": f"button {index} remains depressed and indicator lit",
            })
            item["cam"]["comp"] = f"hand and button {index}"
            item["act"] = {
                "sub": "char_not_in_stage1" if index == 1 else "char_hero",
                "code": "PRESS_CONTROL", "obj": f"button {index}",
            }
            payload[f"p{index:02d}"] = item
        _expanded, errors = story_splitter.validate_v3_stage2(
            payload, stage1, 5, 8,
        )
        self.assertTrue(any("character_ids unknown" in error for error in errors))

    def test_stage2_validator_rejects_duplicate_or_unknown_fact_ids(self):
        response = platform_llm_response(target_seconds=20, shot_count=7)
        stage1 = copy.deepcopy(response)
        panels = stage1.pop("panels")
        panels[0]["character_ids"] = [
            stage1["character_bible"][0]["character_id"],
            stage1["character_bible"][0]["character_id"],
        ]
        panels[1]["scene_id"] = "scene_not_in_stage1"
        panels[2]["story_beat_id"] = "beat_not_in_stage1"
        _expanded, errors = story_splitter.validate_v3_stage2(
            {"panels": panels}, stage1, 7, 20,
        )
        self.assertTrue(any("duplicate IDs" in error for error in errors))
        self.assertTrue(any("scene_id not present" in error for error in errors))
        self.assertTrue(any("story_beat_id not present" in error for error in errors))

    def test_stage2_backend_allocates_exact_edit_clock_instead_of_trusting_llm(self):
        response = platform_llm_response(target_seconds=8, shot_count=5)
        panels = copy.deepcopy(response["panels"])
        stage1 = copy.deepcopy(response)
        stage1.pop("panels")
        panels[-1]["edit_duration_seconds"] = 0.1
        panels[0]["edit_duration_seconds"] = 2.4
        panels[0]["audio_cues"] = [{
            "cue_type": "sfx", "prompt": "wallet lands",
            "start_s": 1.8, "end_s": 2.4,
        }]
        expanded, errors = story_splitter.validate_v3_stage2(
            {"panels": panels}, stage1, 5, 8,
        )
        self.assertFalse(any("edit_duration_seconds" in error for error in errors))
        self.assertAlmostEqual(sum(item["edit_duration_seconds"] for item in expanded), 8.0)
        self.assertEqual(expanded[-1]["llm_suggested_edit_duration_seconds"], 0.1)
        self.assertAlmostEqual(expanded[0]["audio_cues"][0]["end_s"], 1.6)
        self.assertAlmostEqual(expanded[0]["timeline_scale_factor"], 2 / 3)

    def test_parallel_sfx_and_ambience_are_valid_but_dialogue_overlap_is_not(self):
        panel = platform_llm_response(target_seconds=8, shot_count=5)["panels"][0]
        panel["audio_cues"] = [
            {"cue_type": "ambience", "prompt": "rain", "start_s": 0.0, "end_s": 1.6},
            {"cue_type": "sfx", "prompt": "door bell", "start_s": 0.4, "end_s": 0.8},
        ]
        self.assertFalse(any("audio_cues" in error and "overlaps" in error for error in story_splitter.validate_panel(panel)))
        panel["spoken_dialogue"] = [
            {"speaker_id": "char_hero", "text": "A", "start_s": 0.1, "end_s": 0.8},
            {"speaker_id": "char_hero", "text": "B", "start_s": 0.7, "end_s": 1.1},
        ]
        self.assertTrue(any("spoken_dialogue" in error and "overlaps" in error for error in story_splitter.validate_panel(panel)))

    def test_stage1_schema_requires_ascii_model_tags_scene_prompt_and_exact_explicit_two(self):
        schema = story_splitter._v3_stage1_tool_schema(2)
        self.assertNotIn("cb", schema["properties"])
        self.assertTrue({"c1", "c2"}.issubset(schema["required"]))
        properties = schema["properties"]["c1"]["properties"]
        self.assertEqual(properties["it"]["items"]["pattern"], r"^[\x20-\x7E]+$")
        self.assertEqual(properties["wt"]["items"]["pattern"], r"^[\x20-\x7E]+$")
        self.assertGreaterEqual(properties["it"]["minItems"], 4)
        self.assertIn("1boy", properties["it"]["examples"][0])
        scene_prompt = schema["properties"]["s1"]["properties"]["mp"]
        self.assertEqual(scene_prompt["pattern"], r"^[\x20-\x7E]+$")
        self.assertIn("ASCII English", scene_prompt["description"])
        visual = schema["properties"]["vb"]["properties"]
        self.assertEqual(visual["sp"]["pattern"], r"^[\x20-\x7E]+$")
        self.assertEqual(visual["neg"]["pattern"], r"^[\x20-\x7E]+$")

    def test_stage1_rejects_non_ascii_or_mojibake_visual_model_prompts(self):
        response = platform_llm_response(target_seconds=8, shot_count=5)
        response.pop("panels")
        response["visual_bible"]["global_negative_prompt"] = "血腥暴力"
        errors = story_splitter.validate_v3_stage1(response)
        self.assertIn(
            "visual_bible.global_negative_prompt requires printable ASCII English",
            errors,
        )

    def test_explicit_core_character_count_is_deterministic_not_name_inferred(self):
        self.assertEqual(story_splitter.explicit_requested_character_count("本剧只有2位核心人物。"), 2)
        self.assertEqual(story_splitter.explicit_requested_character_count("2 core characters race home"), 2)
        self.assertIsNone(story_splitter.explicit_requested_character_count("林川和顾远共同出现"))
        self.assertIsNone(story_splitter.explicit_requested_character_count("2位核心人物，但另处写3位核心人物"))

    def test_time_variant_prompt_reuses_identity_without_automatic_name_merge(self):
        hints = story_splitter.explicit_identity_equivalence_hints("十分钟后，换装的林川疑似出现在门口")
        prompt = story_splitter._stage1_system_prompt(
            "cn", requested_character_count=2, identity_equivalence_hints=hints,
        )
        self.assertIn("exactly 2 core characters", prompt)
        self.assertIn("MUST reuse the same character_id", prompt)
        self.assertIn("unless the user explicitly declares", prompt)
        self.assertIn("later-time version", prompt)
        self.assertNotIn("林川", prompt)

    def test_chinese_model_tags_remain_local_hard_failure_without_translation(self):
        response = platform_llm_response(target_seconds=20, shot_count=7)
        response["character_bible"][0]["model_identity_tags_en"] = ["1boy", "短黑发", "棕色眼睛"]
        response["character_bible"][0]["model_wardrobe_tags_en"] = ["深蓝防雨外套", "黄色快递包"]
        response["scene_bible"][0]["model_prompt_en"] = "雨夜中国城市街道"
        response.pop("panels")
        with patch.object(story_splitter, "_call_m3", return_value=json.dumps(response)) as call_m3:
            with self.assertRaisesRegex(
                story_splitter.MiniMaxGenerationStageError,
                "requires non-empty English tags.*English model_prompt_en",
            ):
                story_splitter.split_story(
                    "本剧只有2位核心人物，十分钟后同一人换装出现。",
                    api_key="offline-only", total_duration_seconds=20,
                    shot_count=7, min_panels=7, max_panels=7,
                )
        call_m3.assert_called_once()
        sent_schema = call_m3.call_args.kwargs["tool_schema"]
        self.assertTrue({"c1", "c2"}.issubset(sent_schema["required"]))
        self.assertNotIn("cb", sent_schema["properties"])

    def test_stage1_fixed_slots_require_primary_scene_and_all_five_beats(self):
        schema = story_splitter._v3_stage1_tool_schema(2, single_scene=True)
        self.assertFalse(schema["additionalProperties"])
        self.assertTrue({
            "sb", "vb", "c1", "c2", "s1", "h", "setup", "escalation", "reversal", "end",
        }.issubset(schema["required"]))
        self.assertNotIn("sc", schema["properties"])
        self.assertNotIn("beats", schema["properties"])
        self.assertNotIn("sx", schema["properties"])
        self.assertEqual(schema["properties"]["h"]["properties"]["r"]["enum"], ["hook"])
        self.assertEqual(
            schema["properties"]["end"]["properties"]["r"]["enum"], ["cliffhanger", "close"],
        )

    def test_unknown_cast_keeps_array_but_primary_scene_is_still_required(self):
        schema = story_splitter._v3_stage1_tool_schema(None, single_scene=False)
        self.assertIn("cb", schema["required"])
        self.assertEqual(schema["properties"]["cb"]["minItems"], 1)
        self.assertIn("s1", schema["required"])
        self.assertIn("sx", schema["properties"])
        self.assertNotIn("sx", schema["required"])

    def test_explicit_single_scene_detection_is_not_inferred_from_story_shape(self):
        self.assertTrue(story_splitter.explicit_single_scene("全片只有一个场景：旧仓库"))
        self.assertTrue(story_splitter.explicit_single_scene("single location chamber drama"))
        self.assertFalse(story_splitter.explicit_single_scene("两人在仓库谈判"))

    def test_fixed_stage1_wire_expands_only_supplied_slots(self):
        character = {
            "id": "char_a", "n": "A", "desc": "hero", "it": ["1boy", "male", "black hair", "brown eyes"],
            "wt": ["blue jacket", "black pants"],
            "v": {"lang": "Chinese", "age": "adult", "tone": "warm", "pace": "medium"},
        }
        scene = {"id": "scene_a", "desc": "仓库", "mp": "grounded modern warehouse interior"}
        fixed = {
            "sb": {"t": "T", "l": "L", "s": "S", "th": ["x"], "cr": ["y"]},
            "vb": {"sp": "grounded anime", "neg": "cyberpunk"},
            "c1": character, "c2": {**character, "id": "char_b", "n": "B"}, "s1": scene,
            "h": {"id": "b_h", "r": "hook", "q": "q", "proof": "p", "pay": "x"},
            "setup": {"id": "b_s", "r": "setup", "q": "q", "proof": "p", "pay": "x"},
            "escalation": {"id": "b_e", "r": "escalation", "q": "q", "proof": "p", "pay": "x"},
            "reversal": {"id": "b_r", "r": "reversal", "q": "q", "proof": "p", "pay": "x"},
            "end": {"id": "b_end", "r": "close", "q": "q", "proof": "p", "pay": "x"},
        }
        expanded = story_splitter.expand_v3_stage1(fixed)
        self.assertEqual([item["character_id"] for item in expanded["character_bible"]], ["char_a", "char_b"])
        self.assertEqual([item["scene_id"] for item in expanded["scene_bible"]], ["scene_a"])
        self.assertEqual([item["role"] for item in expanded["story_beats"]], [
            "hook", "setup", "escalation", "reversal", "close",
        ])
        missing_scene = dict(fixed)
        missing_scene.pop("s1")
        self.assertEqual(story_splitter.expand_v3_stage1(missing_scene)["scene_bible"], [])

    def test_openai_mode_is_explicit_legacy_json_and_never_auto_selected(self):
        captured = {}

        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps({
                    "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]
                }).encode("utf-8")

        def fake_urlopen(request, **_kwargs):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return Response()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            raw = story_splitter._call_m3(
                "system", "user", api_key="offline-only", protocol="openai",
                base_url="https://api.minimax.io/v1", tool_name="submit_v3_stage1",
                tool_schema={"type": "object"},
            )
        self.assertEqual(raw, '{"ok":true}')
        self.assertEqual(captured["url"], "https://api.minimax.io/v1/chat/completions")
        self.assertNotIn("tools", captured["body"])
        self.assertNotIn("tool_choice", captured["body"])
        self.assertEqual(story_splitter.minimax_protocol(), "anthropic")

    def test_m27_length_finish_reason_is_reported_as_truncation_before_json_parse(self):
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps({
                    "choices": [{
                        "finish_reason": "length",
                        "message": {
                            "reasoning_details": [{"type": "reasoning.text", "text": "private"}],
                            "content": '{"sb":{"t":"cut off"',
                        },
                    }],
                    "usage": {"completion_tokens": 2048},
                }).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(
                story_splitter.MiniMaxOutputTruncated,
                "finish_reason=length.*completion_tokens=2048.*不会进入 JSON 解析",
            ):
                story_splitter._call_m3(
                    "system", "user", api_key="offline-only", protocol="openai",
                )

    def test_m27_compact_stage1_wire_expands_from_fenced_content_with_reasoning_details(self):
        compact = {
            "sb": {"t": "雨夜快递", "l": "快递员必须送达药品", "s": "雨夜送药", "th": ["责任"], "cr": ["外套不变"]},
            "cb": [{
                "id": "char_courier", "n": "林舟", "desc": "24岁中国男性快递员",
                "it": ["1boy", "adult male", "Chinese", "short black hair", "brown eyes"],
                "wt": ["navy rain jacket", "yellow courier bag", "black waterproof sneakers"],
                "v": {"lang": "Chinese", "age": "young adult", "tone": "warm baritone", "pace": "medium"},
            }],
            "vb": {"sp": "grounded modern Chinese urban anime", "neg": "cyberpunk, visible text"},
            "sc": [{"id": "scene_street", "desc": "雨夜街口", "mp": "rainy modern Chinese street at night"}],
            "beats": [
                {"id": f"beat_{role}", "r": role, "q": f"q {role}", "proof": f"proof {role}", "pay": f"pay {role}"}
                for role in ("hook", "setup", "escalation", "reversal", "close")
            ],
        }
        raw = "```json\n" + json.dumps(compact, ensure_ascii=False) + "\n```"
        parsed = story_splitter.expand_v3_stage1(story_splitter._parse_stage_response(raw, 1))
        self.assertEqual(parsed["character_bible"][0]["model_identity_tags_en"][0], "1boy")
        self.assertEqual(parsed["scene_bible"][0]["model_prompt_en"], "rainy modern Chinese street at night")
        self.assertEqual(story_splitter.validate_v3_stage1(parsed), [])

    def test_json_extraction_ignores_braces_inside_tool_string_values(self):
        payload = {
            "p01": {
                "f": 'counter state {wallet: "returned"}',
                "l": "a printed brace } remains part of the prop description",
                "note": r"escaped quote \" and slash \\",
            }
        }
        raw = "provider preface\n" + json.dumps(payload, ensure_ascii=False) + "\nprovider suffix"
        self.assertEqual(story_splitter._parse_stage_response(raw, 2), payload)

    def test_m27_stage1_accepts_role_keyed_beats_object_without_order_guessing(self):
        compact = {
            "sb": {"t": "title", "l": "logline", "s": "synopsis", "th": ["theme"], "cr": ["lock"]},
            "cb": [{
                "id": "char_hero", "n": "Hero", "desc": "young courier",
                "it": ["1boy", "adult male", "short black hair", "brown eyes"],
                "wt": ["navy rain jacket", "yellow courier bag", "black waterproof sneakers"],
                "v": {"lang": "Chinese", "age": "young adult", "tone": "warm", "pace": "medium"},
            }],
            "vb": {"sp": "grounded urban anime", "neg": "cyberpunk"},
            "sc": [{"id": "scene_road", "desc": "rainy road", "mp": "modern rainy city road"}],
            "beats": {
                role: {
                    "id": f"beat_{role}", "q": f"q {role}",
                    "proof": f"proof {role}", "pay": f"pay {role}",
                }
                for role in ("hook", "setup", "escalation", "reversal", "cliffhanger")
            },
        }
        expanded = story_splitter.expand_v3_stage1(compact)
        self.assertEqual(
            [beat["role"] for beat in expanded["story_beats"]],
            ["hook", "setup", "escalation", "reversal", "cliffhanger"],
        )
        self.assertEqual(story_splitter.validate_v3_stage1(expanded), [])

    def test_m27_stage1_accepts_explicit_long_role_fields_but_never_position_infers(self):
        explicit = {
            "beats": [
                {"beat_id": "b1", "role": "opening hook", "dramatic_question": "q", "visible_proof": "p", "payoff_or_hook": "x"},
                {"beat_id": "b2", "role": "story setup", "dramatic_question": "q", "visible_proof": "p", "payoff_or_hook": "x"},
                {"beat_id": "b3", "role": "rising action", "dramatic_question": "q", "visible_proof": "p", "payoff_or_hook": "x"},
                {"beat_id": "b4", "role": "turning point", "dramatic_question": "q", "visible_proof": "p", "payoff_or_hook": "x"},
                {"beat_id": "b5", "role": "ending cliffhanger", "dramatic_question": "q", "visible_proof": "p", "payoff_or_hook": "x"},
            ]
        }
        expanded = story_splitter._expand_wire_beats(explicit["beats"])
        self.assertEqual([item["role"] for item in expanded], [
            "hook", "setup", "escalation", "reversal", "cliffhanger",
        ])
        missing_roles = story_splitter._expand_wire_beats([
            {"id": "b1", "q": "q", "proof": "p", "pay": "x"} for _ in range(5)
        ])
        self.assertTrue(all(item["role"] == "" for item in missing_roles))

    def test_stage1_failure_reports_only_safe_wire_shape_not_values(self):
        bad = {
            "sb": {}, "cb": [], "vb": {}, "sc": [],
            "beats": {"unexpected_secret_role_value": {"mystery": "SECRET_CONTENT"}},
        }
        with patch.object(story_splitter, "_call_m3", return_value=json.dumps(bad)):
            with self.assertRaises(story_splitter.MiniMaxGenerationStageError) as caught:
                story_splitter.split_story(
                    "story", api_key="offline-only", total_duration_seconds=20,
                    shot_count=7, min_panels=7, max_panels=7,
                )
        message = str(caught.exception)
        self.assertIn("beats_type=dict", message)
        self.assertIn("beat_field_names=['mystery']", message)
        self.assertNotIn("SECRET_CONTENT", message)
        self.assertNotIn("unexpected_secret_role_value", message)

    def test_minimax_timeout_defaults_to_180_and_is_environment_configurable(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(story_splitter.minimax_request_timeout_seconds(), 180.0)
        with patch.dict(os.environ, {"AI_MANGA_MINIMAX_TIMEOUT_SECONDS": "45"}, clear=True):
            self.assertEqual(story_splitter.minimax_request_timeout_seconds(), 45.0)
        with self.assertRaisesRegex(ValueError, "between 10 and 600"):
            story_splitter.minimax_request_timeout_seconds(5)

    def test_minimax_timeout_is_explicit_and_never_auto_retries(self):
        with patch("urllib.request.urlopen", side_effect=socket.timeout("offline timeout")) as urlopen:
            with self.assertRaisesRegex(
                story_splitter.MiniMaxRequestTimeout,
                "未写入项目、未保存合同.*不会自动再次发起付费请求",
            ):
                story_splitter._call_m3(
                    "system", "user", api_key="offline-only", timeout_seconds=10,
                )
        urlopen.assert_called_once()
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 10.0)

    def test_invalid_json_does_not_trigger_second_paid_request(self):
        with patch.object(story_splitter, "_call_m3", return_value="not-json") as call_m3:
            with self.assertRaisesRegex(ValueError, "不会自动再次调用"):
                story_splitter.split_story(
                    "story", api_key="offline-only", total_duration_seconds=10,
                    shot_count=5, min_panels=5, max_panels=5,
                )
        call_m3.assert_called_once()

    def test_openai_key_is_never_used_as_minimax_fallback(self):
        # Windows normalizes environment variable names to uppercase while
        # lookup remains case-insensitive. Remove every spelling so a real
        # project .env cannot leak into this isolation test.
        env = {
            key: value for key, value in os.environ.items()
            if key.casefold() != "MiniMax_API_KEY".casefold()
        }
        env["OPENAI_API_KEY"] = "must-not-leave-openai-boundary"
        with patch.dict(os.environ, env, clear=True), patch.object(story_splitter, "M3_API_KEY", ""):
            with self.assertRaises(story_splitter.MissingMiniMaxAPIKey):
                story_splitter._call_m3("system", "user")

    def test_demo_requires_explicit_mode_and_cannot_masquerade_as_user_story(self):
        result = story_splitter.split_story(
            "UNIQUE_USER_STORY_THAT_MUST_NOT_APPEAR",
            demo_mode=True,
            min_panels=1,
            max_panels=1,
        )
        encoded = json.dumps(result, ensure_ascii=False)
        self.assertTrue(result["is_demo"])
        self.assertEqual(result["source_mode"], "DEMO")
        self.assertTrue(result["demo_original_request_ignored"])
        self.assertNotIn("UNIQUE_USER_STORY_THAT_MUST_NOT_APPEAR", encoded)

    def test_live_settings_are_forced_into_every_panel(self):
        calls = []
        def fake_call(system, user, **kwargs):
            calls.append((system, user, kwargs))
            return json.dumps(platform_llm_response())
        with patch.object(
            story_splitter,
            "_call_m3",
            side_effect=fake_call,
        ):
            result = story_splitter.split_story(
                "Hero enters a room.",
                topic="Courage",
                synopsis="Hero enters a room.",
                target_audience="young adult",
                total_duration_seconds=8,
                shot_count=5,
                platform="TikTok",
                api_key="test-only",
                min_panels=5,
                max_panels=5,
                prompt_mode="cinematic",
                visual_style="noir",
                style_enforcement="black and amber noir",
                aspect_ratio="9:16",
                duration_seconds=8,
                use_lora=False,
                lora_strength=0.35,
                sage_mode="sageattn3",
                ref_image_size="max",
                background_music="soft_piano",
                ambience="silence",
                voice_language="English",
            )
        panel = result["panels"][0]
        self.assertEqual(result["source_mode"], "LIVE")
        self.assertEqual(panel["prompt_mode"], "cinematic")
        self.assertEqual(panel["aspect_ratio"], "9:16")
        self.assertEqual(panel["duration_seconds"], 10.125)
        self.assertEqual(panel["edit_duration_seconds"], 1.6)
        self.assertFalse(panel["use_lora"])
        self.assertEqual(panel["lora_strength"], 0.35)
        self.assertEqual(panel["sage_mode"], "sageattn3")
        self.assertEqual(panel["background_music"], "soft_piano")
        self.assertEqual(panel["ambience"], "silence")
        self.assertEqual(panel["prompt_package"]["render_settings"]["ref_image_size"], "max")
        self.assertEqual(result["visual_bible"]["style_prompt"], "black and amber noir")
        self.assertIn("black and amber noir", panel["positive_prompt"])
        self.assertEqual(result["schema_version"], "ai-manga.prompt-package/v3")
        self.assertEqual(result["creative_brief"]["topic"], "Courage")
        self.assertEqual(result["creative_brief"]["shot_count"], 5)
        self.assertIn("elite series head writer", calls[0][0])
        self.assertIn('"target_audience":"young adult"', calls[0][1])
        self.assertEqual(panel["subtitle_timeline"], [])
        self.assertEqual(panel["on_screen_text"], [])
        self.assertEqual(result["quality_warnings"], [])

    def test_v3_two_stage_plan_calls_each_stage_once_and_validates_exact_7_in_20s(self):
        response = platform_llm_response(target_seconds=20, shot_count=7)
        stage1 = dict(response)
        stage1.pop("panels")
        outputs = [stage1, {"panels": response["panels"]}]
        calls = []

        def fake_call(system_prompt, user_prompt, **_kwargs):
            calls.append((system_prompt, user_prompt))
            return json.dumps(outputs[len(calls) - 1], ensure_ascii=False)

        with patch.object(story_splitter, "_call_m3", side_effect=fake_call):
            result = story_splitter.split_story(
                "A courier races through a storm.", api_key="offline-only",
                total_duration_seconds=20, shot_count=7, min_panels=7, max_panels=7,
            )
        self.assertEqual(len(calls), 2)
        self.assertIn("STAGE 1 OF 2", calls[0][0])
        self.assertIn("STAGE 2 OF 2", calls[1][0])
        self.assertEqual(len(result["panels"]), 7)
        self.assertAlmostEqual(sum(panel["edit_duration_seconds"] for panel in result["panels"]), 20)
        self.assertEqual(result["generation_plan"]["completed_calls"], 2)

    def test_v3_stage1_hard_failure_never_starts_stage2(self):
        with patch.object(story_splitter, "_call_m3", return_value=json.dumps({"story_bible": {"title": "partial"}})) as call_m3:
            with self.assertRaisesRegex(story_splitter.MiniMaxGenerationStageError, "阶段 1/2.*已发起 1 次"):
                story_splitter.split_story(
                    "story", api_key="offline-only", total_duration_seconds=20,
                    shot_count=7, min_panels=7, max_panels=7,
                )
        call_m3.assert_called_once()

    def test_v3_stage2_failure_returns_no_half_contract_and_never_auto_retries(self):
        stage1 = platform_llm_response(target_seconds=20, shot_count=7)
        stage1.pop("panels")
        outputs = [json.dumps(stage1), "not-json"]
        with patch.object(story_splitter, "_call_m3", side_effect=outputs) as call_m3:
            with self.assertRaisesRegex(story_splitter.MiniMaxGenerationStageError, "阶段 2/2.*已发起 2 次.*最终合同未保存"):
                story_splitter.split_story(
                    "story", api_key="offline-only", total_duration_seconds=20,
                    shot_count=7, min_panels=7, max_panels=7,
                )
        self.assertEqual(call_m3.call_count, 2)

    def test_contract_validation_is_language_neutral_and_allows_a_quiet_shot(self):
        response = platform_llm_response(target_seconds=10, shot_count=5)
        response["panels"][0]["cuts"][0]["shot_description"] = (
            "中景跟拍主角从门外进入房间，镜头缓慢向木桌推进，夕阳从右侧窗户照亮蓝色外套与桌面。"
        )
        response["panels"][0]["sfx"] = []
        with patch.object(story_splitter, "_call_m3", return_value=json.dumps(response, ensure_ascii=False)):
            result = story_splitter.split_story(
                "主角进入房间。",
                api_key="test-only",
                total_duration_seconds=10,
                shot_count=5,
                min_panels=5,
                max_panels=5,
            )
        self.assertEqual(result["quality_warnings"], [])

    def test_single_item_regeneration_is_mocked_and_invalidates_approval(self):
        episode = story_splitter.split_story("", demo_mode=True, min_panels=1, max_panels=1)
        episode["approval_state"]["creative"] = {"story": True, "characters": True, "storyboard": True}
        replacement = dict(episode["story_bible"])
        replacement["logline"] = "Sharper approved premise"
        with patch.object(story_splitter, "_call_m3", return_value=json.dumps({"item": replacement})):
            updated = story_splitter.regenerate_contract_item(
                episode, "story", "", "sharpen premise", api_key="test-only"
            )
        self.assertEqual(updated["story_bible"]["logline"], "Sharper approved premise")
        self.assertFalse(any(updated["approval_state"]["creative"].values()))

    def test_regeneration_uses_model_specific_minimax_h3_prompt_master(self):
        episode = story_splitter.split_story("", demo_mode=True, min_panels=1, max_panels=1)
        panel = episode["panels"][0]
        captured = {}

        def fake_call(system_prompt, user_prompt, **_kwargs):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            replacement = dict(panel)
            replacement["positive_prompt"] = "rewritten chronological H3 shot plan"
            return json.dumps({"item": replacement}, ensure_ascii=False)

        with patch.object(story_splitter, "_call_m3", side_effect=fake_call):
            story_splitter.regenerate_contract_item(
                episode, "panel", panel["panel_id"],
                "characters drifted and dialogue timing was stiff", api_key="test-only",
            )
        self.assertIn("MiniMax H3 prompt master", captured["system"])
        self.assertIn("opening composition and subjects", captured["system"])
        self.assertIn("one dominant camera path", captured["system"])
        self.assertIn("Never solve a failure by stacking repetitive synonyms", captured["system"])
        self.assertIn("characters drifted and dialogue timing was stiff", captured["user"])
        self.assertNotIn("asset_rejection_history", captured["user"])
        self.assertNotIn("reference_images", captured["user"])

    def test_regeneration_rejects_unchanged_item_instead_of_fake_success(self):
        episode = story_splitter.split_story("", demo_mode=True, min_panels=1, max_panels=1)
        panel = episode["panels"][0]
        with patch.object(
            story_splitter, "_call_m3", return_value=json.dumps({"item": panel}, ensure_ascii=False)
        ):
            with self.assertRaisesRegex(ValueError, "unchanged"):
                story_splitter.regenerate_contract_item(
                    episode, "panel", panel["panel_id"], "rewrite it", api_key="test-only"
                )


if __name__ == "__main__":
    unittest.main()
