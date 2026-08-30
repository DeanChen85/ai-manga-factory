import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from prompt_contracts import (
    SERIES_SCHEMA_VERSION,
    normalize_series_contract,
    series_episode_context,
    validate_series_contract,
)
from story_splitter import (
    generate_series_episode,
    split_series,
    update_series_outline_episode,
)
from ui_helpers import (
    prepare_series_via_facade,
    register_series_episodes_via_facade,
    series_episode_counts,
    series_from_service_snapshot,
    series_registration_payloads,
    series_service_spec,
    with_series_episode_approval,
)


def series_settings():
    return {
        "episode_count": 3,
        "seconds_per_episode": 20.0,
        "shots_per_episode": 5,
        "language": "cn",
        "voice_language": "Chinese",
        "visual_style": "serialized noir comic",
        "style_enforcement": "serialized noir comic, stable ink and amber-blue palette",
        "aspect_ratio": "9:16",
        "prompt_mode": "cinematic",
        "use_lora": True,
        "lora_strength": 1.0,
        "sage_mode": "auto",
        "ref_image_size": "match",
        "background_music": "soft_piano",
        "ambience": "rain_night_city",
        "creative_brief": {
            "topic": "写给十年后的自己",
            "synopsis": "快递员发现未来来信，并用三集完成一次不可逆的选择。",
            "target_audience": "年轻成人",
            "episode_count": 3,
            "seconds_per_episode": 20,
        },
    }


def raw_series():
    state_1 = {"timeline": "night 1", "characters": "Lin has no letter", "world": "station closed"}
    state_2 = {"timeline": "night 1", "characters": "Lin holds sealed letter", "world": "station closed"}
    state_3 = {"timeline": "morning 2", "characters": "Lin knows warning", "world": "station gate open"}
    state_4 = {"timeline": "morning 2", "characters": "Lin acts on warning", "world": "route changed"}
    series = {
        "series_bible": {
            "series_id": "series_future_letter",
            "title": "写给十年后的信",
            "premise": "林舟收到未来来信后必须连续作出选择。",
            "genre": "温暖悬疑",
            "target_audience": "年轻成人",
            "themes": ["选择"],
            "story_engine": "每集兑现上一集留下的选择后果",
            "season_arc": "发现来信、验证警告、改变路线",
            "immutable_facts": ["林舟是24岁快递员"],
        },
        "shared_character_bible": [{
            "character_id": "char_linzhou",
            "name": "林舟",
            "role": "protagonist",
            "editorial_identity_description": "24岁中国男性，短黑发，棕色眼睛，左眉尾浅疤",
            "editorial_wardrobe_description": "深蓝防雨外套，黑色长裤，黄色快递包",
            "identity_prompt": "24岁中国男性，短黑发，棕色眼睛，左眉尾浅疤",
            "wardrobe_lock": "深蓝防雨外套，黑色长裤，黄色快递包",
            "model_identity_tags_en": [
                "1boy", "male", "24 years old", "Chinese", "short hair", "black hair",
                "brown eyes", "small scar at outer left eyebrow",
            ],
            "model_wardrobe_tags_en": [
                "dark blue rain jacket", "black pants", "yellow courier bag", "black waterproof shoes",
            ],
            "voice_profile": {
                "language": "Chinese", "accent": "standard Mandarin", "age": "young adult",
                "timbre": "restrained baritone", "pace": "medium", "emotion_range": "restrained to urgent",
            },
        }],
        "world_bible": {
            "setting": "废弃车站与城市快递路线",
            "time_period": "contemporary",
            "world_rules": ["来信内容一旦被读出就不能重置"],
            "geography": {"scene_station": "north of delivery route"},
            "timeline_rules": ["时间只向前推进"],
            "forbidden_retcons": ["不能让林舟忘记上一集"],
        },
        "visual_bible": {"style_prompt": "serialized noir comic, stable ink and amber-blue palette"},
        "shared_scene_bible": [{
            "scene_id": "scene_station",
            "name": "废弃车站",
            "description": "雨夜废弃车站站台",
            "model_prompt_en": "abandoned train station platform, rainy night, cool blue lighting, puddle reflections",
        }],
        "season_outline": [
            {
                "episode_id": "ep_001", "episode_index": 1, "title": "来信", "logline": "林舟发现未来来信。",
                "duration_seconds": 20, "shot_count": 5,
                "beats": [{"beat_index": 1, "purpose": "setup", "summary": "林舟发现信封", "character_ids": ["char_linzhou"], "scene_ids": ["scene_station"]}],
                "continuity_state_in": state_1, "continuity_state_out": state_2,
                "wardrobe_change_events": [], "time_jump_event": None, "cliffhanger_or_payoff": "林舟拿起信",
            },
            {
                "episode_id": "ep_002", "episode_index": 2, "title": "警告", "logline": "林舟读懂警告。",
                "duration_seconds": 20, "shot_count": 5,
                "beats": [{"beat_index": 1, "purpose": "turn", "summary": "林舟读信", "character_ids": ["char_linzhou"], "scene_ids": ["scene_station"]}],
                "continuity_state_in": state_2, "continuity_state_out": state_3,
                "wardrobe_change_events": [], "time_jump_event": {"from": "night 1", "to": "morning 2", "reason": "after reading"},
                "cliffhanger_or_payoff": "车站铁门开启",
            },
            {
                "episode_id": "ep_003", "episode_index": 3, "title": "改道", "logline": "林舟改变明天。",
                "duration_seconds": 20, "shot_count": 5,
                "beats": [{"beat_index": 1, "purpose": "payoff", "summary": "林舟选择新路线", "character_ids": ["char_linzhou"], "scene_ids": ["scene_station"]}],
                "continuity_state_in": state_3, "continuity_state_out": state_4,
                "wardrobe_change_events": [], "time_jump_event": None, "cliffhanger_or_payoff": "未来被改变",
            },
        ],
        "episode_contracts": {},
    }
    roles = ["hook", "setup", "escalation", "reversal", "close"]
    for episode in series["season_outline"]:
        episode["beats"] = [{
            "beat_index": index,
            "purpose": role,
            "summary": f"{episode['title']} visible causal beat {role}",
            "visible_proof": f"Lin visibly opens, lifts or moves the story object in {role}",
            "character_ids": ["char_linzhou"],
            "scene_ids": ["scene_station"],
        } for index, role in enumerate(roles, 1)]
    return series


def raw_v3_episode():
    episode = {
        "title": "来信",
        "story_bible": {"title": "来信", "logline": "林舟发现未来来信。", "synopsis": "林舟进入车站并拿起信。"},
        "panels": [
            {
                "panel_id": "panel_01_arrive", "name": "panel_01_arrive",
                "scene_id": "scene_station", "character_ids": ["char_linzhou"],
                "continuity_group": "main", "previous_panel_id": None,
                "series_beat_index": 1,
                "continuity_state_in": {}, "continuity_state_out": {"beat": "letter noticed"},
                "first_frame": "char_linzhou enters the abandoned platform from frame left",
                "last_frame": "char_linzhou stops beside a dry envelope on the bench",
                "cuts": [{"time_range": "0-10s", "name": "arrival", "intensity": "SMOOTH", "shot_description": "A vertical medium-wide tracking shot follows char_linzhou through the rain toward the empty bench while cool blue lamps outline the dark blue jacket and yellow courier bag against stable platform geography."}],
                "spoken_dialogue": [], "subtitle_timeline": [], "on_screen_text": [], "audio_cues": [],
            },
            {
                "panel_id": "panel_02_letter", "name": "panel_02_letter",
                "scene_id": "scene_station", "character_ids": ["char_linzhou"],
                "continuity_group": "main", "previous_panel_id": "panel_01_arrive",
                "series_beat_index": 1,
                "continuity_state_in": {"beat": "letter noticed"}, "continuity_state_out": {},
                "first_frame": "char_linzhou bends toward the same envelope without changing screen direction",
                "last_frame": "char_linzhou holds the sealed envelope while the courier bag remains on shoulder",
                "cuts": [{"time_range": "0-10s", "name": "pickup", "intensity": "TENSE", "shot_description": "The camera pushes from the dry envelope to char_linzhou's restrained reaction as one hand lifts it, preserving black hair, the exact rain jacket, courier bag, bench position, rainy light and platform layout."}],
                "spoken_dialogue": [], "subtitle_timeline": [], "on_screen_text": [], "audio_cues": [],
            },
        ],
    }
    roles = ["hook", "setup", "escalation", "reversal", "close"]
    panels = []
    previous_id = None
    previous_state = {}
    for index, role in enumerate(roles, 1):
        template = copy.deepcopy(episode["panels"][0 if index < 5 else 1])
        panel_id = f"panel_{index:02d}_{role}"
        final_state = {"beat": f"{role} completed"}
        template.update({
            "panel_id": panel_id, "name": panel_id,
            "previous_panel_id": previous_id,
            "continuity_state_in": previous_state,
            "continuity_state_out": final_state,
            "series_beat_index": index,
            "source_generation_duration_seconds": 10.125,
            "edit_duration_seconds": 4.0,
            "shot_role": role,
            "story_beat_id": f"beat_{role}",
            "visible_action": f"Lin slides the envelope across the desk until it stops under the lamp in beat {index}",
            "first_state": f"envelope at position {index}",
            "final_state": f"envelope visibly moved at position {index}",
            "cause": "The future warning forces the next physical choice",
            "next_hook": "A newly revealed mark changes the next action",
            "camera_plan": {
                "shot_size": f"shot-size-{index}", "angle": f"angle-{index}",
                "movement": "controlled push", "composition": f"composition-{index}",
            },
            "transition": {"type": "hard_cut", "motivation": "causal beat advance"},
            "edit_hint": {
                "preferred_moment": "envelope clears hand", "edit_in_hint": "hand reaches",
                "edit_out_hint": "mark revealed",
            },
            "priority": "must_have", "group_shot_reason": "",
            "spoken_dialogue": [], "subtitle_timeline": [], "on_screen_text": [], "audio_cues": [],
        })
        template["cuts"] = [{
            "time_range": "0-10.125s", "name": role, "intensity": "SMOOTH",
            "shot_description": f"A controlled shot {index} follows Lin opening and lifting the sealed envelope while preserving the station geography and exact approved wardrobe.",
        }]
        panels.append(template)
        previous_id = panel_id
        previous_state = final_state
    episode["panels"] = panels
    return episode


class SeriesContractV4Tests(unittest.TestCase):
    def test_normalizer_and_validator_lock_exact_count_duration_and_state_chain(self):
        series = normalize_series_contract(raw_series(), settings=series_settings())
        self.assertEqual(series["schema_version"], SERIES_SCHEMA_VERSION)
        self.assertEqual(len(series["season_outline"]), 3)
        self.assertEqual(series["season_outline"][1]["continuity_state_in"], series["season_outline"][0]["continuity_state_out"])
        self.assertEqual(validate_series_contract(series), [])
        broken = copy.deepcopy(series)
        broken["season_outline"][1]["continuity_state_in"] = {"reset": True}
        self.assertTrue(any("must exactly equal" in error for error in validate_series_contract(broken)))

    def test_split_series_uses_head_writer_prompt_and_exact_brief(self):
        calls = []
        def fake_call(system, user, **_kwargs):
            calls.append((system, user))
            return json.dumps(raw_series(), ensure_ascii=False)
        with patch("story_splitter._call_m3", side_effect=fake_call):
            result = split_series(
                topic="写给十年后的自己", synopsis="快递员收到未来来信。",
                episode_count=3, seconds_per_episode=20, shots_per_episode=5,
                target_audience="年轻成人", visual_style="serialized noir comic",
                style_enforcement="serialized noir comic, stable ink and amber-blue palette",
                aspect_ratio="9:16", language="cn", api_key="mock-only",
                background_music="soft_piano", ambience="rain_night_city",
            )
        self.assertEqual(result["episode_count"], 3)
        self.assertIn("elite television series head writer", calls[0][0])
        self.assertIn("exactly 3", calls[0][0])
        self.assertIn('"seconds_per_episode": 20.0', calls[0][1])

    def test_generate_one_v3_episode_preserves_shared_facts_and_boundaries(self):
        series = normalize_series_contract(raw_series(), settings=series_settings())
        shared_before = copy.deepcopy(series["shared_character_bible"])
        with patch("story_splitter._call_m3", return_value=json.dumps(raw_v3_episode())):
            updated = generate_series_episode(series, "ep_001", api_key="mock-only")
        episode = updated["episode_contracts"]["ep_001"]
        self.assertEqual(series["episode_contracts"], {})
        self.assertEqual(updated["shared_character_bible"], shared_before)
        self.assertEqual(episode["schema_version"], "ai-manga.prompt-package/v3")
        self.assertEqual(episode["series_episode_id"], "ep_001")
        self.assertEqual(episode["continuity_state_in"], updated["season_outline"][0]["continuity_state_in"])
        self.assertEqual(episode["continuity_state_out"], updated["season_outline"][0]["continuity_state_out"])
        self.assertEqual(sum(panel["edit_duration_seconds"] for panel in episode["panels"]), 20)
        self.assertIn("1boy", episode["panels"][0]["prompt_package"]["character_prompts"]["char_linzhou"])
        self.assertEqual(validate_series_contract(updated), [])
        drifted = copy.deepcopy(updated)
        drifted["episode_contracts"]["ep_001"]["character_bible"][0]["voice_profile"]["timbre"] = "different actor"
        self.assertTrue(any("voice profile drift" in error for error in validate_series_contract(drifted)))
        self.assertEqual(series_episode_counts(updated), {"total": 3, "generated": 1, "approved": 0})
        approved = with_series_episode_approval(updated, "ep_001", True)
        self.assertEqual(series_episode_counts(approved)["approved"], 1)

    def test_outline_edit_preserves_boundaries_and_invalidates_generated_episode(self):
        series = normalize_series_contract(raw_series(), settings=series_settings())
        series["episode_contracts"]["ep_002"] = {"placeholder": True}
        series["episode_approvals"]["ep_002"] = True
        current = copy.deepcopy(series["season_outline"][1])
        replacement = {**current, "title": "更强的警告", "continuity_state_in": {"illegal": "reset"}}
        updated = update_series_outline_episode(series, "ep_002", replacement)
        self.assertEqual(updated["season_outline"][1]["title"], "更强的警告")
        self.assertEqual(updated["season_outline"][1]["continuity_state_in"], current["continuity_state_in"])
        self.assertNotIn("ep_002", updated["episode_contracts"])
        self.assertFalse(updated["episode_approvals"]["ep_002"])
        context = series_episode_context(updated, "ep_002")
        self.assertEqual(context["previous_episode"]["episode_id"], "ep_001")
        self.assertEqual(context["next_episode"]["episode_id"], "ep_003")

    def test_approved_mid_episode_wardrobe_change_is_panel_scoped(self):
        raw = raw_series()
        raw["season_outline"][1]["beats"][1].update({
            "beat_index": 2, "purpose": "setup", "summary": "林舟换上红色救援外套",
            "visible_proof": "林舟脱下深蓝外套并穿上红色救援外套",
            "character_ids": ["char_linzhou"], "scene_ids": ["scene_station"],
        })
        raw["season_outline"][1]["wardrobe_change_events"] = [{
            "character_id": "char_linzhou",
            "from": "深蓝防雨外套",
            "to": "红色救援外套",
            "reason": "进入封锁区前穿救援装备",
            "effective_beat": 2,
            "model_wardrobe_tags_en": [
                "red rescue jacket", "black pants", "yellow courier bag", "black waterproof shoes",
            ],
        }]
        series = normalize_series_contract(raw, settings=series_settings())
        episode_payload = raw_v3_episode()
        episode_payload["panels"][1]["series_beat_index"] = 2
        with patch("story_splitter._call_m3", return_value=json.dumps(episode_payload)):
            updated = generate_series_episode(series, "ep_002", api_key="mock-only")
        panels = updated["episode_contracts"]["ep_002"]["panels"]
        first_prompt = panels[0]["prompt_package"]["character_prompts"]["char_linzhou"]
        second_prompt = panels[1]["prompt_package"]["character_prompts"]["char_linzhou"]
        self.assertIn("dark blue rain jacket", first_prompt)
        self.assertNotIn("red rescue jacket", first_prompt)
        self.assertIn("red rescue jacket", second_prompt)
        self.assertEqual(
            panels[1]["model_wardrobe_overrides_en"]["char_linzhou"][0],
            "red rescue jacket",
        )

    def test_series_service_adapter_persists_runtime_without_rehashing_shared_contract(self):
        from series_store import series_contract_hash

        series = normalize_series_contract(raw_series(), settings=series_settings())
        spec_before = series_service_spec(series)
        series["episode_contracts"]["ep_001"] = {"schema_version": "ai-manga.prompt-package/v3"}
        series["episode_approvals"]["ep_001"] = True
        spec_after = series_service_spec(series)
        self.assertEqual(series_contract_hash(spec_before), series_contract_hash(spec_after))
        self.assertEqual(spec_before["episode_count"], 3)
        self.assertEqual(spec_before["episode_seconds"], 20)
        self.assertIn("v4_contract", spec_before)
        self.assertIn("v4_contract", spec_after["runtime"])

        backend_character = copy.deepcopy(spec_after["character_bible"][0])
        backend_character["reference_images"] = ["shared/linzhou.png"]
        snapshot = {
            "series_id": "series_future_letter",
            "series": {
                "series_id": "series_future_letter", "status": "approved",
                "contract_hash": "hash", "shared_assets_status": "ready_for_approval",
                "shared_assets_hash": None,
                "spec": {**spec_after, "character_bible": [backend_character]},
            },
            "counts": {"expected": 3, "registered": 0, "complete": 0},
            "ready": False,
        }
        restored = series_from_service_snapshot(snapshot)
        self.assertTrue(restored["season_approved"])
        self.assertEqual(restored["shared_character_bible"][0]["reference_images"], ["shared/linzhou.png"])
        self.assertIn("ep_001", restored["episode_contracts"])

    def test_registration_payloads_require_exact_n_generated_and_approved(self):
        series = normalize_series_contract(raw_series(), settings=series_settings())
        with self.assertRaisesRegex(ValueError, "must be generated and approved"):
            series_registration_payloads(series)
        for number, item in enumerate(series["season_outline"], 1):
            episode_id = item["episode_id"]
            series["episode_contracts"][episode_id] = {
                "schema_version": "ai-manga.prompt-package/v3",
                "panels": [{"duration_seconds": 10.125, "edit_duration_seconds": 4.0}] * 5,
            }
            series["episode_approvals"][episode_id] = True
        payloads = series_registration_payloads(series)
        self.assertEqual([item["episode_number"] for item in payloads], [1, 2, 3])
        self.assertEqual(payloads[0]["ep_id"], "series_future_letter_ep_001")

    def test_public_facade_persists_and_restores_v4_runtime(self):
        series = normalize_series_contract(raw_series(), settings=series_settings())
        spec = series_service_spec(series)
        snapshot = {
            "series_id": "series_future_letter",
            "series": {
                "series_id": "series_future_letter", "status": "draft",
                "contract_hash": "series-hash", "shared_assets_status": "pending",
                "shared_assets_hash": None, "spec": spec,
            },
            "shared_assets": [],
            "episodes": [],
            "counts": {"expected": 3, "registered": 0, "complete": 0},
            "ready": False,
        }
        facade = Mock()
        facade.prepare_series.return_value = snapshot
        restored, actual = prepare_series_via_facade(facade, series)
        facade.prepare_series.assert_called_once()
        called_id, called_spec = facade.prepare_series.call_args.args
        self.assertEqual(called_id, "series_future_letter")
        self.assertEqual(called_spec["episode_count"], 3)
        self.assertEqual(actual, snapshot)
        self.assertEqual(restored["series_bible"]["series_id"], "series_future_letter")

    def test_public_facade_registers_exact_approved_season(self):
        series = normalize_series_contract(raw_series(), settings=series_settings())
        for item in series["season_outline"]:
            episode_id = item["episode_id"]
            series["episode_contracts"][episode_id] = {
                "schema_version": "ai-manga.prompt-package/v3",
                "panels": [{"duration_seconds": 10.125, "edit_duration_seconds": 4.0}] * 5,
            }
            series["episode_approvals"][episode_id] = True
        facade = Mock()
        facade.register_episodes.return_value = {"series_id": "series_future_letter"}
        snapshot = register_series_episodes_via_facade(facade, series)
        self.assertEqual(snapshot["series_id"], "series_future_letter")
        called_id, payloads = facade.register_episodes.call_args.args
        self.assertEqual(called_id, "series_future_letter")
        self.assertEqual(len(payloads), 3)
        self.assertEqual([item["episode_number"] for item in payloads], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
