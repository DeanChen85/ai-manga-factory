import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from prompt_contracts import (
    MODERN_URBAN_NEGATIVE,
    MODERN_URBAN_STYLE_PROMPT,
    PROMPT_SCHEMA_VERSION,
    SOURCE_GENERATION_DURATION_SECONDS,
    allocate_edit_durations,
    auto_episode_shot_count,
    build_character_reference_prompt,
    continuity_chain_warnings,
    enrich_episode_contract,
    normalize_character_bible,
    normalize_scene_bible,
    repair_episode_character_references,
    subtitle_mismatch_warnings,
    shot_count_bounds,
    shot_plan_cost_summary,
    validate_platform_shot_plan,
    visible_action_evidence,
)
from action_catalog import compile_action_spec, derived_action_components


def sample_episode():
    return {
        "title": "锁定测试",
        "subtitle": "两个人在同一房间完成交易",
        "story_bible": {"logline": "A courier meets a buyer."},
        "character_bible": [
            {
                "character_id": "char_courier",
                "name": "Courier",
                "role": "protagonist",
                "identity_prompt": (
                    "27-year-old woman, oval face, short black bob, one silver hairpin on the left, "
                    "brown eyes, small mole below right eye, athletic build"
                ),
                "signature_features": "silver hairpin on left; small mole below right eye",
                "wardrobe_lock": {"outfit": "navy canvas jacket", "footwear": "black boots"},
            },
            {
                "character_id": "char_buyer",
                "name": "Buyer",
                "role": "supporting",
                "identity_prompt": "45-year-old man, square face, shaved head, grey eyes, stocky build",
                "wardrobe_lock": {"outfit": "charcoal wool coat"},
            },
        ],
        "visual_bible": {
            "style_name": "noir comic",
            "style_prompt": "premium noir comic animation, hard rim light, ink texture",
        },
        "scene_bible": [
            {
                "scene_id": "scene_warehouse",
                "name": "Warehouse",
                "description": "Empty warehouse at midnight, one hanging lamp, wet concrete floor",
                "positive_prompt": "midnight warehouse, wet concrete, one hanging tungsten lamp",
            }
        ],
        "panels": [
            {
                "panel_id": "ep01_panel01_exchange",
                "name": "ep01_panel01_exchange",
                "scene_id": "scene_warehouse",
                "character_ids": ["char_courier", "char_buyer"],
                "first_frame": "char_courier enters frame left while char_buyer waits frame right",
                "last_frame": "both hands hold the same sealed case",
                "cuts": [{
                    "time_range": "0-10s",
                    "name": "exchange",
                    "intensity": "TENSE",
                    "shot_description": "Tense eye-level two shot with both locked characters and the sealed case centered",
                }],
                "spoken_dialogue": [{
                    "start_s": 0.5,
                    "end_s": 2.5,
                    "speaker_id": "char_courier",
                    "text": "带来了",
                    "delivery_style": "quiet",
                    "max_chars": 6,
                }],
                "on_screen_text": [{
                    "start_s": 3.0,
                    "end_s": 4.0,
                    "text": "午夜",
                    "position": "top-left",
                    "style": "white ink",
                }],
                "audio_cues": [{
                    "start_s": 4.0,
                    "end_s": 4.8,
                    "cue_type": "sfx",
                    "prompt": "metal case click",
                }],
                "transitions": [],
            }
        ],
    }


def settings():
    return {
        "prompt_mode": "comic",
        "visual_style": "noir comic",
        "aspect_ratio": "16:9",
        "duration_seconds": 10.0,
        "use_lora": True,
        "lora_strength": 0.8,
        "sage_mode": "auto",
        "ref_image_size": "max",
        "background_music": "suspense_dark",
        "ambience": "office_quiet",
        "voice_language": "Chinese",
    }


class PromptContractTests(unittest.TestCase):
    def test_platform_shot_density_and_exact_duration_allocation(self):
        self.assertEqual(auto_episode_shot_count(60), 20)
        self.assertEqual(shot_count_bounds(60), {
            "minimum": 15, "maximum": 40, "preferred": 20,
        })
        durations = allocate_edit_durations(60, 20)
        self.assertEqual(len(durations), 20)
        self.assertAlmostEqual(sum(durations), 60.0, places=6)
        self.assertTrue(all(1.5 <= value <= 4.0 for value in durations))
        cost = shot_plan_cost_summary(60, 20)
        self.assertEqual(cost["total_source_generation_duration_seconds"], 202.5)
        self.assertEqual(cost["gpu_generation_jobs"], 20)

    def test_platform_shot_validator_rejects_slogan_group_and_repeated_camera(self):
        roles = ["hook", "setup", "escalation", "reversal", "close"]
        panels = []
        for index, role in enumerate(roles):
            panels.append({
                "source_generation_duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
                "edit_duration_seconds": 2.0,
                "shot_role": role,
                "story_beat_id": f"beat_{role}",
                "visible_action": "Hero slides the sealed package across the table until it stops inside the marked zone",
                "first_state": "locker closed", "final_state": "locker open",
                "cause": "The alarm forces a decision", "next_hook": "A light flashes inside",
                "camera_plan": {
                    "shot_size": f"size-{index}", "angle": "eye-level",
                    "movement": "slow push", "composition": f"composition-{index}",
                },
                "transition": {"type": "hard_cut", "motivation": "causal advance"},
                "edit_hint": {
                    "preferred_moment": "package clears locker", "edit_in_hint": "hand reaches",
                    "edit_out_hint": "package raised",
                },
                "priority": "must_have", "character_ids": ["char_a"],
            })
        self.assertEqual(validate_platform_shot_plan(panels, 10), [])
        panels[1]["visible_action"] = "女店员按下报警器按钮，按钮保持压下"
        panels[1]["action_components"] = {
            "sub": "char_a", "verb": "按下", "obj": "报警器按钮", "res": "按钮保持压下",
        }
        self.assertFalse(any(
            "panel[1].visible_action" in error
            for error in validate_platform_shot_plan(panels, 10)
        ))
        panels[2]["visible_action"] = "友谊万岁"
        panels[2]["character_ids"] = ["char_a", "char_b", "char_c"]
        panels[2]["camera_plan"] = dict(panels[1]["camera_plan"])
        errors = validate_platform_shot_plan(panels, 10)
        self.assertTrue(any("concrete visible action" in error for error in errors))
        self.assertTrue(any("at most 2 visible characters" in error for error in errors))
        self.assertTrue(any("repeats the previous composition" in error for error in errors))

    def test_canonical_action_ignores_tampered_display_but_rejects_code_mismatch(self):
        roles = ["hook", "setup", "escalation", "reversal", "close"]
        panels = []
        for index, role in enumerate(roles):
            spec = compile_action_spec({
                "actor_id": "char_a", "action_code": "SLIDE_OBJECT",
                "target": f"sealed package {index}",
                "start_state": f"package {index} beside char_a",
                "end_state": f"package {index} inside marked zone",
            }, visible_character_ids=["char_a"])
            panels.append({
                "source_generation_duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
                "edit_duration_seconds": 2.0, "shot_role": role,
                "story_beat_id": f"beat_{role}", "character_ids": ["char_a"],
                "action_spec": spec, "action_code": spec["action_code"],
                "action_components": derived_action_components(spec),
                "visible_action": "tampered abstract display prose",
                "first_state": spec["start_state"], "final_state": spec["end_state"],
                "cause": "alarm advances the beat", "next_hook": "next package appears",
                "camera_plan": {
                    "shot_size": f"size-{index}", "angle": "eye-level",
                    "movement": "locked", "composition": f"composition-{index}",
                },
                "transition": {"type": "hard_cut", "motivation": "causal advance"},
                "edit_hint": {
                    "preferred_moment": "package moves", "edit_in_hint": "hand touches",
                    "edit_out_hint": "package settles",
                },
                "priority": "must_have", "group_shot_reason": "",
            })
        self.assertEqual(validate_platform_shot_plan(panels, 10), [])
        panels[2]["action_code"] = "OPEN_OBJECT"
        errors = validate_platform_shot_plan(panels, 10)
        self.assertTrue(any("panel.action_code disagrees" in error for error in errors))

    def test_action_contract_rejects_invisible_actor_and_unknown_legacy_verb(self):
        panel = {
            "source_generation_duration_seconds": SOURCE_GENERATION_DURATION_SECONDS,
            "edit_duration_seconds": 2.0, "shot_role": "hook", "story_beat_id": "beat_hook",
            "character_ids": ["char_visible"],
            "action_components": {
                "sub": "char_hidden", "verb": "按下", "obj": "alarm button", "res": "button lit",
            },
            "visible_action": "ignored", "first_state": "button dark", "final_state": "button lit",
            "cause": "alarm", "next_hook": "light",
            "camera_plan": {"shot_size": "CU", "angle": "eye", "movement": "locked", "composition": "button"},
            "transition": {"type": "close", "motivation": "terminal"},
            "edit_hint": {"preferred_moment": "press", "edit_in_hint": "touch", "edit_out_hint": "lit"},
            "priority": "must_have", "group_shot_reason": "",
        }
        errors = validate_platform_shot_plan([panel], 7.5)
        self.assertTrue(any("actor_id must belong" in error for error in errors))
        panel["action_components"] = {
            "sub": "char_visible", "verb": "bravely decides", "obj": "alarm button", "res": "button lit",
        }
        errors = validate_platform_shot_plan([panel], 7.5)
        self.assertTrue(any("exact registered alias" in error for error in errors))

    def test_visible_action_validator_accepts_one_concrete_chinese_action(self):
        evidence = visible_action_evidence("林川把密封药盒推到顾远面前停住")
        self.assertEqual(evidence["category"], "valid_single_visible_action")
        self.assertTrue(evidence["has_physical_verb"])
        self.assertTrue(evidence["has_visible_result"])

    def test_visible_action_validator_rejects_chinese_mental_or_abstract_action(self):
        mental = visible_action_evidence("林川终于意识到顾远隐瞒了危险真相")
        abstract = visible_action_evidence("两个人之间重新燃起希望与坚定勇气")
        self.assertEqual(mental["category"], "abstract_or_mental")
        self.assertEqual(abstract["category"], "abstract_or_mental")

    def test_visible_action_validator_rejects_chained_multiple_actions(self):
        evidence = visible_action_evidence("林川打开铁门然后跑进房间并举起药盒")
        self.assertEqual(evidence["category"], "multiple_actions")

    def test_scene_contract_compiles_five_phone_story_as_empty_social_game_room(self):
        scenes = normalize_scene_bible([{
            "scene_id": "scene_game_room",
            "description": (
                "A tiny contemporary room with five friends playing a mobile game around one table"
            ),
            "model_prompt_en": (
                "small room, 3 square meters, one ordinary desk, five mobile phones and tablets "
                "neatly arranged in a straight row, product display cabinet, overhead view"
            ),
            "negative_prompt": "changed location",
            "continuity_lock": {"time": "noon"},
        }], [], {"style_profile": "modern_urban", "palette": []})
        self.assertEqual(len(scenes), 1)
        scene = scenes[0]
        positive = scene["model_prompt_en"]
        negative = scene["negative_prompt"]
        continuity = scene["continuity_lock"]
        self.assertEqual(scene["environment_profile"], "social_mobile_gaming_room")
        for required in (
            "ordinary contemporary private living room arranged as a casual gaming room",
            "one single ordinary rectangular shared gaming table centered in the room",
            "exactly 5 separate empty ordinary seats spaced around the table",
            "exactly 5 separate black-screen smartphones lying flat on the shared tabletop",
            "one phone at each seating place",
            "human eye-level wide shot from the room entrance",
            "tabletop seen obliquely in natural room perspective",
        ):
            self.assertIn(required, positive)
        for leaked in (
            "neatly arranged in a straight row", "product display cabinet", "overhead view",
        ):
            self.assertNotIn(leaked, positive)
        self.assertNotIn("five people", positive)
        for forbidden in (
            "retail store", "showroom", "display cabinet", "drawer", "shelf",
            "phone shop", "product display", "top-down view",
        ):
            self.assertIn(forbidden, negative)
        self.assertEqual(continuity["time"], "noon")
        self.assertIn("exactly 5 separate empty seats", continuity["seat_lock"])
        self.assertIn("one phone at each seating place", continuity["hero_props"])
        self.assertIn("no people or characters", continuity["occupancy_lock"])

    def test_normalized_character_preserves_wardrobe_on_second_pass(self):
        card = {
            "character_id": "char_messenger",
            "name": "Messenger",
            "identity_prompt": "single woman with black hair",
            "wardrobe_prompt": "exact dark teal waterproof short coat, black straight trousers",
        }
        first = normalize_character_bible([card])[0]
        second = normalize_character_bible([first])[0]
        self.assertEqual(second["wardrobe_prompt"], card["wardrobe_prompt"])
        built = build_character_reference_prompt(second, {})
        self.assertIn(card["wardrobe_prompt"], built["positive_prompt"])
        self.assertIn("dark teal short coat", built["positive_prompt"])
        self.assertIn("black pants", built["positive_prompt"])
        self.assertNotIn("model sheet", built["positive_prompt"])
        self.assertIn("multiple views", built["negative_prompt"])

    def test_character_aliases_survive_normalization_and_repair_panel_speakers(self):
        episode = sample_episode()
        episode["character_bible"][0]["character_id"] = "char01"
        episode["character_bible"][0]["aliases"] = ["hero-courier", "messenger_alpha"]
        episode["panels"][0]["character_ids"] = ["hero-courier", "char_buyer"]
        episode["panels"][0]["spoken_dialogue"][0]["speaker_id"] = "char01"
        episode["panels"][0]["first_state"] = episode["panels"][0]["first_frame"]
        episode["panels"][0]["final_state"] = episode["panels"][0]["last_frame"]
        episode["panels"][0]["action_spec"] = compile_action_spec({
            "actor_id": "hero-courier", "action_code": "SLIDE_OBJECT",
            "target": "sealed case",
            "start_state": episode["panels"][0]["first_frame"],
            "end_state": episode["panels"][0]["last_frame"],
        }, visible_character_ids=["hero-courier", "char_buyer"])
        alias_hash = episode["panels"][0]["action_spec"]["spec_sha256"]
        enriched = enrich_episode_contract(
            episode, story_text="story", source_mode="LIVE", settings=settings()
        )
        courier = enriched["character_bible"][0]
        panel = enriched["panels"][0]
        self.assertEqual(courier["character_id"], "char_courier")
        self.assertIn("hero-courier", courier["aliases"])
        self.assertIn("char01", courier["aliases"])
        self.assertEqual(panel["character_ids"], ["char_courier", "char_buyer"])
        self.assertEqual(panel["spoken_dialogue"][0]["speaker_id"], "char_courier")
        self.assertEqual(panel["action_spec"]["actor_id"], "char_courier")
        self.assertNotEqual(panel["action_spec"]["spec_sha256"], alias_hash)

        persisted = copy.deepcopy(enriched)
        persisted["approval_state"] = {
            "creative": {"story": True, "characters": True, "storyboard": True},
            "assets": {"character_ids": ["char_courier"], "scene_ids": ["scene_warehouse"]},
        }
        persisted["panels"][0]["character_ids"] = ["messenger_alpha", "char_buyer"]
        persisted["panels"][0]["prompt_package"]["character_ids"] = ["messenger_alpha", "char_buyer"]
        persisted["panels"][0]["spoken_dialogue"][0]["speaker_id"] = "hero-courier"
        repaired = repair_episode_character_references(persisted)
        repaired_panel = repaired["panels"][0]
        self.assertEqual(repaired_panel["character_ids"], ["char_courier", "char_buyer"])
        self.assertEqual(repaired_panel["spoken_dialogue"][0]["speaker_id"], "char_courier")
        self.assertFalse(repaired["approval_state"]["creative"]["storyboard"])
        self.assertEqual(repaired["approval_state"]["assets"], {"character_ids": [], "scene_ids": []})

    def test_plural_ensemble_shot_keeps_all_cast_not_only_the_speaker(self):
        episode = sample_episode()
        episode["panels"][0]["character_ids"] = []
        episode["panels"][0]["cuts"][0]["shot_description"] = (
            "All friends and team members gather around one table while the speaker raises a phone."
        )
        episode["panels"][0]["spoken_dialogue"][0]["speaker_id"] = "char_courier"
        enriched = enrich_episode_contract(
            episode, story_text="ensemble story", source_mode="LIVE", settings=settings()
        )
        panel = enriched["panels"][0]
        self.assertEqual(panel["character_ids"], ["char_courier", "char_buyer"])
        self.assertEqual(
            panel["prompt_package"]["character_ids"],
            ["char_courier", "char_buyer"],
        )

    def test_modern_urban_profile_removes_cyber_doll_and_red_eye_drift(self):
        episode = sample_episode()
        episode["visual_bible"] = {
            "style_name": "现代都市",
            "style_prompt": "modern Chinese urban animation, cyberpunk neon city, doll-like people",
        }
        episode["scene_bible"][0]["model_prompt_en"] = (
            "cyberpunk warehouse, futuristic holograms, neon city, glowing red eyes, city interior"
        )
        modern_settings = {
            **settings(),
            "visual_style": "现代都市",
            "style_enforcement": MODERN_URBAN_STYLE_PROMPT,
        }
        result = enrich_episode_contract(
            episode, story_text="grounded city story", source_mode="LIVE", settings=modern_settings
        )
        visual = result["visual_bible"]
        scene = result["scene_bible"][0]
        panel = result["panels"][0]
        self.assertEqual(visual["style_profile"], "modern_urban")
        for forbidden in ("cyberpunk", "futuristic", "hologram", "doll", "glowing red eyes"):
            self.assertNotIn(forbidden, visual["style_prompt"].lower())
            self.assertNotIn(forbidden, scene["model_prompt_en"].lower())
        for required in (
            "cyberpunk", "glowing red eyes", "doll", "figurine", "chibi",
            "photorealistic", "mixed art styles", "inflated clothing",
        ):
            self.assertIn(required, MODERN_URBAN_NEGATIVE)
            self.assertIn(required, visual["global_negative_prompt"])
            self.assertIn(required, scene["negative_prompt"])
            self.assertIn(required, panel["negative_prompt"])
        self.assertIn("grounded real-world architecture", scene["model_prompt_en"])
        self.assertIn("natural human eye colors", panel["positive_prompt"])
        character_prompt = build_character_reference_prompt(result["character_bible"][0], visual)
        character_positive = character_prompt["positive_prompt"]
        self.assertTrue(character_positive.startswith("masterpiece, best quality"))
        self.assertRegex(character_positive, r"\b1(?:girl|boy)\b")
        self.assertIn("(solo:1.6)", character_positive)
        gender_tag = "1girl" if "1girl" in character_positive else "1boy"
        self.assertLess(character_positive.index(gender_tag), character_positive.index("anime screencap"))
        self.assertIn("anime screencap, hand-drawn 2D cel animation", character_prompt["positive_prompt"])
        self.assertIn("natural human eye colors", character_prompt["positive_prompt"])
        self.assertIn("glowing red eyes", character_prompt["negative_prompt"])
        self.assertIn("body hidden by oversized garment", character_prompt["negative_prompt"])

    def test_modern_character_prompt_keeps_person_and_expands_vague_clothes(self):
        visual = {
            "style_profile": "modern_urban",
            "style_prompt": MODERN_URBAN_STYLE_PROMPT,
            "global_negative_prompt": "no text, no people, cyberpunk",
        }
        card = normalize_character_bible([{
            "character_id": "char_friend",
            "name": "Friend",
            "identity_prompt": "23-year-old Chinese woman, long black hair, glasses",
            "model_identity_tags_en": ["1girl", "female", "23 years old", "black hair", "glasses"],
            "model_wardrobe_tags_en": ["pink silk clothes", "high heels"],
        }])[0]
        prompt = build_character_reference_prompt(card, visual)
        self.assertIn("pink collared long-sleeve blouse", prompt["positive_prompt"])
        self.assertIn("(person wearing pink collared long-sleeve blouse:1.4)", prompt["positive_prompt"])
        self.assertIn("(one full-body person centered in frame:1.5)", prompt["positive_prompt"])
        self.assertIn("(single character only:1.55)", prompt["positive_prompt"])
        self.assertIn("green clothing", prompt["negative_prompt"])
        self.assertIn("giant shirt in background", prompt["negative_prompt"])
        self.assertIn("side-by-side people", prompt["negative_prompt"])
        self.assertIn("black straight-leg trousers", prompt["positive_prompt"])
        self.assertIn("fully visible detailed face", prompt["positive_prompt"])
        self.assertIn("visible nose and visible mouth", prompt["positive_prompt"])
        self.assertNotIn("no people", prompt["negative_prompt"])
        self.assertIn("faceless", prompt["negative_prompt"])
        self.assertIn("neon background", prompt["negative_prompt"])

    def test_enriches_versioned_prompt_package_and_separates_timeline_lanes(self):
        result = enrich_episode_contract(
            sample_episode(), story_text="快递员在仓库把箱子交给买家。", source_mode="LIVE", settings=settings()
        )
        panel = result["panels"][0]
        package = panel["prompt_package"]
        self.assertEqual(result["schema_version"], PROMPT_SCHEMA_VERSION)
        self.assertEqual(package["character_ids"], ["char_courier", "char_buyer"])
        self.assertIn("char_courier", package["positive_prompt"])
        self.assertIn("scene_warehouse", package["positive_prompt"])
        self.assertEqual(package["spoken_dialogue_timeline"][0]["speaker_id"], "char_courier")
        self.assertEqual(package["subtitle_timeline"][0]["text"], panel["spoken_dialogue"][0]["text"])
        self.assertEqual(package["subtitle_source"], "spoken_dialogue_derived")
        self.assertEqual(package["on_screen_text_timeline"], [])
        self.assertEqual(panel["postproduction_on_screen_text"][0]["text"], "午夜")
        self.assertEqual(package["sound_timeline"][-1]["kind"], "audio_cue")
        self.assertEqual(panel["dialogue_bubbles"], [])
        self.assertEqual(panel["on_screen_text"], [])
        self.assertEqual(package["h3_visible_text_policy"], "forbidden")

    def test_user_edited_subtitle_warns_and_continuity_break_is_reported(self):
        episode = sample_episode()
        episode["panels"][0]["_subtitle_user_edited"] = True
        episode["panels"][0]["subtitle_timeline"] = [{
            "start_s": 0.5, "end_s": 2.5, "speaker_id": "char_courier", "text": "different line"
        }]
        result = enrich_episode_contract(episode, story_text="story", source_mode="LIVE", settings=settings())
        self.assertTrue(result["subtitle_warnings"])
        self.assertTrue(subtitle_mismatch_warnings(
            result["panels"][0]["spoken_dialogue"], result["panels"][0]["subtitle_timeline"]
        ))
        broken = [result["panels"][0], {**result["panels"][0], "panel_id": "panel_02", "previous_panel_id": None}]
        self.assertTrue(continuity_chain_warnings(broken))

    def test_h3_visual_prompt_strips_visible_text_instructions(self):
        episode = sample_episode()
        episode["panels"][0]["cuts"][0]["shot_description"] = (
            "The courier crosses the warehouse. A large caption HELLO appears above her head. "
            "The camera tracks left through amber light."
        )
        episode["panels"][0]["transitions"] = [{
            "time_range": "2.0-2.5s",
            "transition_description": "A whip pan accelerates. A subtitle NEXT fills the frame.",
        }]
        episode["panels"][0]["first_frame"] = (
            "The courier enters from frame left. A title card START appears behind her."
        )
        episode["panels"][0]["last_frame"] = (
            "The courier grips the case. A neon sign reads END above the door."
        )
        episode["panels"][0]["camera_movement"] = (
            "Slow dolly forward. A poster saying SALE enters frame right."
        )
        episode["scene_bible"][0]["positive_prompt"] = (
            "Amber warehouse aisles remain stable. A wall logo ACME is clearly visible."
        )
        result = enrich_episode_contract(episode, story_text="story", source_mode="LIVE", settings=settings())
        panel = result["panels"][0]
        for forbidden in ("HELLO", "NEXT", "START", "END", "SALE", "ACME"):
            self.assertNotIn(forbidden, panel["cuts"][0]["shot_description"])
            self.assertNotIn(forbidden, panel["positive_prompt"])
        self.assertNotIn("NEXT", panel["transitions"][0]["transition_description"])
        self.assertNotIn("START", panel["first_frame"])
        self.assertNotIn("END", panel["last_frame"])
        self.assertIn("courier crosses", panel["cuts"][0]["shot_description"])
        self.assertIn("camera tracks", panel["cuts"][0]["shot_description"])
        self.assertIn("whip pan", panel["transitions"][0]["transition_description"])
        self.assertIn("courier enters", panel["first_frame"])
        self.assertIn("courier grips", panel["last_frame"])
        self.assertIn("HELLO", panel["cuts"][0]["editorial_shot_description"])
        self.assertIn("NEXT", panel["transitions"][0]["editorial_transition_description"])
        self.assertEqual(panel["on_screen_text"], [])

    def test_character_reference_prompt_uses_id_identity_wardrobe_and_style(self):
        episode = enrich_episode_contract(
            sample_episode(), story_text="story", source_mode="LIVE", settings=settings()
        )
        prompt = build_character_reference_prompt(
            episode["character_bible"][0], episode["visual_bible"], view="侧面"
        )
        self.assertTrue(
            prompt["positive_prompt"].startswith(
                "masterpiece, best quality, 1girl, solo, single subject, same character, "
                "black hair"
            )
        )
        self.assertLess(
            prompt["positive_prompt"].index("navy canvas jacket"),
            prompt["positive_prompt"].index("profile, looking left"),
        )
        self.assertIn("[CHARACTER_ID=char_courier]", prompt["positive_prompt"])
        self.assertIn("navy canvas jacket", prompt["positive_prompt"])
        self.assertIn("silver hairpin on left", prompt["positive_prompt"])
        self.assertIn("small mole below right eye", prompt["positive_prompt"])
        self.assertIn("strict left profile", prompt["positive_prompt"])
        for tag in ("2girls", "multiple girls", "duo", "group", "white hair", "brown hair", "cropped", "out of frame"):
            self.assertIn(tag, prompt["negative_prompt"])
        self.assertIn("wardrobe variation", prompt["negative_prompt"])

    def test_chinese_male_courier_compiles_to_safe_anything_v5_tags(self):
        card = normalize_character_bible([{
            "character_id": "char_linzhou",
            "name": "林舟",
            "identity_prompt": "24岁中国男性，短黑发，棕色眼睛，清瘦脸型，左眉尾浅疤",
            "wardrobe_lock": "深蓝防雨外套，灰色连帽衫，黑色长裤，黄色快递包",
        }])[0]
        identity_tags = card["model_identity_tags_en"]
        wardrobe_tags = card["model_wardrobe_tags_en"]
        self.assertIn("1boy", identity_tags)
        self.assertIn("male", identity_tags)
        self.assertIn("black hair", identity_tags)
        self.assertIn("short hair", identity_tags)
        self.assertIn("dark blue rain jacket", wardrobe_tags)
        self.assertIn("yellow courier bag", wardrobe_tags)
        self.assertFalse(any("\u4e00" <= char <= "\u9fff" for tag in identity_tags + wardrobe_tags for char in tag))

        built = build_character_reference_prompt(card, {}, view="全身")
        for required in ("1boy", "male", "black hair", "blue rain jacket", "yellow courier bag"):
            self.assertIn(required, built["positive_prompt"])
        for forbidden in ("1girl", "white hair", "blonde hair"):
            self.assertIn(forbidden, built["negative_prompt"])
            self.assertNotIn(forbidden, built["positive_prompt"])
        self.assertNotIn("24岁", built["positive_prompt"])
        self.assertNotIn("深蓝", built["positive_prompt"])

    def test_explicit_model_gender_cannot_conflict_with_editorial_gender(self):
        with self.assertRaisesRegex(ValueError, "gender conflicts"):
            normalize_character_bible([{
                "character_id": "char_man",
                "name": "Man",
                "identity_prompt": "24岁中国男性，短黑发",
                "wardrobe_lock": "深蓝防雨外套",
                "model_identity_tags_en": ["1girl", "female", "white hair"],
                "model_wardrobe_tags_en": ["white dress"],
            }])

    def test_h3_character_and_scene_locks_prefer_english_model_fields(self):
        episode = sample_episode()
        episode["character_bible"] = [{
            "character_id": "char_linzhou",
            "name": "林舟",
            "identity_prompt": "24岁中国男性，短黑发，棕色眼睛，清瘦脸型，左眉尾浅疤",
            "wardrobe_lock": "深蓝防雨外套，灰色连帽衫，黑色长裤，黄色快递包",
        }]
        episode["scene_bible"] = [{
            "scene_id": "scene_station",
            "description": "废弃车站站台，雨夜，冷蓝灯，积水倒影",
        }]
        episode["panels"][0]["character_ids"] = ["char_linzhou"]
        episode["panels"][0]["scene_id"] = "scene_station"
        result = enrich_episode_contract(episode, story_text="story", source_mode="LIVE", settings=settings())
        card = result["character_bible"][0]
        scene = result["scene_bible"][0]
        package = result["panels"][0]["prompt_package"]
        self.assertIn("1boy", package["character_prompts"]["char_linzhou"])
        self.assertIn("yellow courier bag", package["character_prompts"]["char_linzhou"])
        self.assertNotIn("24岁", package["character_prompts"]["char_linzhou"])
        self.assertIn("abandoned", scene["model_prompt_en"])
        self.assertIn("train station", scene["asset_prompt"])
        self.assertIn(scene["model_prompt_en"], package["positive_prompt"])
        self.assertEqual(card["editorial_identity_description"], card["identity_prompt"])

    def test_demo_contract_is_visibly_marked(self):
        result = enrich_episode_contract(
            sample_episode(), story_text="bundled demo source", source_mode="DEMO", settings=settings()
        )
        self.assertTrue(result["is_demo"])
        self.assertEqual(result["source_mode"], "DEMO")
        self.assertIn("DEMO DATA", result["demo_notice"])


if __name__ == "__main__":
    unittest.main()
