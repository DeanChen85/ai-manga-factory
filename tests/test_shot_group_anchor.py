from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import shot_group_anchor


class ShotGroupAnchorTests(unittest.TestCase):
    def test_graph_uses_scene_composition_and_two_nonoverlapping_identity_masks(self) -> None:
        graph = shot_group_anchor.build_group_anchor_workflow(
            scene_image="scene.png",
            character_images=["rider.png", "clerk.png"],
            positive_prompt="exactly two people",
            negative_prompt="missing person",
            filename_prefix="group/p02",
            seed=123,
        )
        self.assertEqual(graph["2"]["inputs"]["image"], "scene.png")
        self.assertEqual(graph["3"]["class_type"], "ImageScale")
        self.assertEqual(graph["4"]["class_type"], "VAEEncode")
        self.assertEqual(graph["14"]["inputs"]["latent_image"], ["4", 0])
        self.assertEqual(graph["14"]["inputs"]["denoise"], 0.95)
        self.assertEqual(graph["10"]["inputs"]["attn_mask"], ["9", 0])
        self.assertEqual(graph["13"]["inputs"]["attn_mask"], ["12", 0])
        self.assertEqual(graph["13"]["inputs"]["model"], ["10", 0])
        left = graph["9"]["inputs"]
        right = graph["12"]["inputs"]
        self.assertLessEqual(left["x"] + left["width"], right["x"])

    def test_graph_rejects_any_cast_count_other_than_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two"):
            shot_group_anchor.build_group_anchor_workflow(
                scene_image="scene.png", character_images=["rider.png"],
                positive_prompt="", negative_prompt="", filename_prefix="group/p02",
            )

    def test_prompt_keeps_both_identities_positive_without_gender_negative_conflict(self) -> None:
        panel = {
            "first_frame": "rider checks pockets",
            "final_state": "rider reaches toward clerk",
            "camera_plan": {"composition": "over shoulder rider"},
        }
        characters = [
            {"model_identity_tags_en": "adult male rider", "model_wardrobe_tags_en": "yellow rain jacket"},
            {"model_identity_tags_en": "adult female clerk", "model_wardrobe_tags_en": "dark green vest"},
        ]
        positive, negative = shot_group_anchor.compile_group_anchor_prompts(
            panel, characters, {"model_prompt_en": "small store counter"}, {},
        )
        self.assertIn("adult male rider", positive)
        self.assertIn("adult female clerk", positive)
        self.assertIn("exactly two distinct people", positive)
        self.assertNotIn("female", negative)
        self.assertNotIn("male", negative)
        self.assertIn("missing person", negative)

    def test_single_character_prompt_is_supported_for_opening_composition(self) -> None:
        positive, negative = shot_group_anchor.compile_group_anchor_prompts(
            {"first_frame": "rider outside in rain", "final_state": "rider enters store"},
            [{"model_identity_tags_en": "adult male rider", "model_wardrobe_tags_en": "yellow jacket"}],
            {"model_prompt_en": "glass store entrance"}, {},
        )
        self.assertIn("exactly one adult person visible", positive)
        self.assertIn("blue rain outside", positive)
        self.assertIn("extra person", negative)

    def test_single_character_state_anchors_are_distinct_and_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "scene.png"
            character_path = root / "rider.png"
            first_path = root / "first.png"
            last_path = root / "last.png"
            scene = Image.new("RGB", (768, 1344), (205, 215, 225))
            scene_draw = ImageDraw.Draw(scene)
            scene_draw.rectangle((0, 540, 768, 1344), fill=(170, 155, 130))
            scene.save(scene_path)
            character = Image.new("RGB", (512, 900), "white")
            character_draw = ImageDraw.Draw(character)
            character_draw.ellipse((185, 90, 325, 230), fill=(25, 25, 30))
            character_draw.rectangle((140, 215, 370, 760), fill=(235, 175, 35))
            character_draw.rectangle((185, 760, 245, 880), fill=(20, 20, 25))
            character_draw.rectangle((270, 760, 330, 880), fill=(20, 20, 25))
            character.save(character_path)
            shot_group_anchor._compose_approved_group_anchor(
                scene_path, [character_path], first_path, {}, state="first",
            )
            shot_group_anchor._compose_approved_group_anchor(
                scene_path, [character_path], last_path, {}, state="last",
            )
            self.assertTrue(first_path.is_file())
            self.assertTrue(last_path.is_file())
            self.assertNotEqual(first_path.read_bytes(), last_path.read_bytes())

    def test_two_character_hand_object_gets_distinct_wallet_state_anchors(self) -> None:
        panel = {
            "character_ids": ["rider", "clerk"],
            "action_code": "HAND_OBJECT",
            "action_spec": {
                "action_code": "HAND_OBJECT", "target": "black wallet",
                "start_state": "clerk holds the black wallet",
                "end_state": "rider grips the black wallet",
            },
        }
        self.assertTrue(shot_group_anchor.requires_paired_state_anchor(panel, 2))
        self.assertTrue(shot_group_anchor.requires_approved_group_anchor(panel, {}, 2))
        first_contract = shot_group_anchor.panel_anchor_contract_sha256(panel)
        changed_panel = {**panel, "action_spec": {**panel["action_spec"], "target": "phone"}}
        self.assertNotEqual(
            first_contract, shot_group_anchor.panel_anchor_contract_sha256(changed_panel),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "scene.png"
            rider_path = root / "rider.png"
            clerk_path = root / "clerk.png"
            first_path = root / "first.png"
            last_path = root / "last.png"
            scene = Image.new("RGB", (768, 1344), (205, 215, 225))
            ImageDraw.Draw(scene).rectangle((0, 535, 768, 1135), fill=(180, 160, 130))
            scene.save(scene_path)
            for path, color in ((rider_path, (235, 175, 35)), (clerk_path, (35, 80, 55))):
                person = Image.new("RGB", (512, 900), "white")
                draw = ImageDraw.Draw(person)
                draw.ellipse((185, 90, 325, 230), fill=(35, 25, 25))
                draw.rectangle((140, 215, 370, 760), fill=color)
                draw.rectangle((185, 760, 245, 880), fill=(20, 20, 25))
                draw.rectangle((270, 760, 330, 880), fill=(20, 20, 25))
                person.save(path)
            shot_group_anchor._compose_approved_group_anchor(
                scene_path, [rider_path, clerk_path], first_path, panel, state="first",
            )
            shot_group_anchor._compose_approved_group_anchor(
                scene_path, [rider_path, clerk_path], last_path, panel, state="last",
            )
            self.assertNotEqual(first_path.read_bytes(), last_path.read_bytes())
            self.assertEqual(Image.open(first_path).getpixel((260, 675)), (18, 21, 27))
            self.assertEqual(Image.open(last_path).getpixel((322, 680)), (18, 21, 27))
            self.assertEqual(Image.open(last_path).getpixel((402, 680)), (65, 59, 57))

    def test_prompt_includes_action_target_and_latest_reviewer_correction(self) -> None:
        panel = {
            "character_ids": ["rider", "clerk"],
            "action_spec": {
                "action_code": "HAND_OBJECT", "target": "black wallet",
                "start_state": "clerk holds wallet", "end_state": "rider holds wallet",
            },
        }
        positive, _ = shot_group_anchor.compile_group_anchor_prompts(
            panel,
            [{"identity_prompt": "rider"}, {"identity_prompt": "clerk"}],
            {"description": "store"}, {},
            review_feedback="show exactly one wallet between both people",
        )
        self.assertIn("HAND_OBJECT", positive)
        self.assertIn("black wallet", positive)
        self.assertIn("show exactly one wallet between both people", positive)

    def test_exact_rejected_hash_is_detected_fail_closed(self) -> None:
        self.assertTrue(shot_group_anchor._matches_rejected_candidate(
            "first", "last", [{"sha256": "first", "last_sha256": "last"}],
        ))
        self.assertTrue(shot_group_anchor._matches_rejected_candidate(
            "first", "new-last", [{"sha256": "first", "last_sha256": None}],
        ))
        self.assertFalse(shot_group_anchor._matches_rejected_candidate(
            "new-first", "last", [{"sha256": "first", "last_sha256": "last"}],
        ))


if __name__ == "__main__":
    unittest.main()
