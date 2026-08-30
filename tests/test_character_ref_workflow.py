from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

import generate_character_ref
from generate_character_ref import (
    PRODUCTION_CHECKPOINT,
    build_character_reference_workflow,
    derive_canonical_reference_crops,
    stable_character_seed,
)


CHARACTER = {
    "character_id": "char_hero",
    "name": "Hero",
    "identity_prompt": "19-year-old man, narrow face, black undercut hair, amber eyes, lean build",
    "wardrobe_lock": {"outfit": "red leather jacket", "footwear": "black boots"},
}
MODEL_TAG_CHARACTER = {
    **CHARACTER,
    "model_identity_tags_en": ["1boy", "young adult man", "narrow face", "black undercut hair", "amber eyes", "lean build"],
    "model_wardrobe_tags_en": ["red leather jacket", "black pants", "black boots"],
}
VISUAL = {"style_prompt": "premium graphic novel animation"}


class CharacterReferenceWorkflowTests(unittest.TestCase):
    def test_production_bundle_uses_animagine_and_identity_safe_anchor_crops(self):
        self.assertEqual(PRODUCTION_CHECKPOINT, "animagine-xl-3.1.safetensors")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            anchor = base / "anchor.png"
            Image.new("RGB", (640, 960), "navy").save(anchor)
            with mock.patch.object(generate_character_ref, "COMFYUI_INPUT", base / "input"):
                views = derive_canonical_reference_crops(anchor, character_id="char_hero")
            self.assertEqual(set(views), {"portrait_crop", "torso_crop"})
            self.assertTrue(all(Path(item["path"]).is_file() for item in views.values()))
            self.assertTrue(all(
                item["conditioning_mode"] == "deterministic_crop_from_canonical_anchor"
                for item in views.values()
            ))

    def test_seed_is_stable_for_story_and_character(self):
        first = stable_character_seed(CHARACTER, "story-hash")
        second = stable_character_seed(CHARACTER, "story-hash")
        self.assertEqual(first, second)
        self.assertNotEqual(first, stable_character_seed(CHARACTER, "different-story"))

    def test_anchor_workflow_is_seeded_text_to_image_and_auditable(self):
        graph, manifest = build_character_reference_workflow(
            CHARACTER, VISUAL, story_hash="story-hash", view="anchor"
        )
        self.assertEqual(graph["5"]["class_type"], "EmptyLatentImage")
        self.assertEqual(graph["3"]["inputs"]["seed"], manifest["seed"])
        self.assertEqual(manifest["conditioning_mode"], "text_to_image_anchor")
        self.assertEqual(graph["6"]["inputs"]["text"], manifest["positive_prompt"])
        self.assertEqual(graph["7"]["inputs"]["text"], manifest["negative_prompt"])
        self.assertEqual(manifest["prompt_format"], "anything_v5_danbooru_en")
        self.assertIn("1boy", manifest["positive_prompt"])
        self.assertIn("red leather jacket", manifest["positive_prompt"])

    def test_animagine_anchor_uses_official_tag_and_sampler_profile(self):
        graph, manifest = build_character_reference_workflow(
            MODEL_TAG_CHARACTER, VISUAL, story_hash="story-hash", view="anchor",
            checkpoint=PRODUCTION_CHECKPOINT,
        )
        self.assertEqual(manifest["prompt_format"], "animagine_xl_31_danbooru_en")
        self.assertIn("masterpiece", manifest["positive_prompt"])
        self.assertIn("very aesthetic", manifest["positive_prompt"])
        self.assertIn("absurdres", manifest["positive_prompt"])
        self.assertIn("adult proportions", manifest["positive_prompt"])
        self.assertIn("chibi", manifest["negative_prompt"])
        self.assertIn("barefoot", manifest["negative_prompt"])
        self.assertEqual((graph["5"]["inputs"]["width"], graph["5"]["inputs"]["height"]), (768, 1344))
        self.assertEqual(graph["3"]["inputs"]["sampler_name"], "euler_ancestral")
        self.assertEqual(graph["3"]["inputs"]["scheduler"], "normal")
        self.assertEqual(graph["3"]["inputs"]["steps"], 28)
        self.assertEqual(graph["3"]["inputs"]["cfg"], 6.0)

    def test_anything_v5_consumes_model_english_tags_and_dynamic_conflicts(self):
        graph, manifest = build_character_reference_workflow(
            MODEL_TAG_CHARACTER, VISUAL, story_hash="story-hash", view="anchor"
        )
        positive = manifest["positive_prompt"]
        negative = manifest["negative_prompt"]
        self.assertIn("black undercut hair", positive)
        self.assertIn("red leather jacket", positive)
        self.assertIn("1girl", negative)
        self.assertIn("white hair", negative)
        self.assertIn("blonde hair", negative)
        self.assertIn("green hair", negative)
        self.assertIn("qipao", negative)
        self.assertIn("dress", negative)
        self.assertNotIn("black hair", negative)
        self.assertEqual(graph["6"]["inputs"]["text"], positive)
        self.assertEqual(graph["7"]["inputs"]["text"], negative)

    def test_dynamic_negative_does_not_reject_expected_female_hair_or_dress(self):
        female = {
            "character_id": "char_lead",
            "name": "Lead",
            "identity_prompt": "adult woman with blonde wavy hair and blue eyes",
            "wardrobe_prompt": "navy dress and white boots",
            "model_identity_tags_en": "1girl, adult woman, blonde wavy hair, blue eyes",
            "model_wardrobe_tags_en": "navy dress, white boots",
        }
        _, manifest = build_character_reference_workflow(female, VISUAL, view="anchor")
        self.assertIn("1boy", manifest["negative_prompt"])
        self.assertNotIn("blonde hair", manifest["negative_prompt"])
        self.assertNotIn(", dress,", f", {manifest['negative_prompt']},")

    def test_sheet_uses_ipadapter_identity_with_fresh_latent_for_composition(self):
        graph, manifest = build_character_reference_workflow(
            CHARACTER,
            VISUAL,
            story_hash="story-hash",
            view="侧面",
            anchor_image="charref_char_hero_anchor.png",
        )
        self.assertEqual(graph["10"]["class_type"], "LoadImage")
        self.assertEqual(graph["10"]["inputs"]["image"], "charref_char_hero_anchor.png")
        self.assertEqual(graph["5"]["class_type"], "EmptyLatentImage")
        self.assertEqual(graph["12"]["class_type"], "IPAdapterUnifiedLoader")
        self.assertEqual(graph["12"]["inputs"]["model"], ["4", 0])
        self.assertEqual(graph["12"]["inputs"]["preset"], "PLUS FACE (portraits)")
        self.assertEqual(graph["13"]["class_type"], "IPAdapterAdvanced")
        self.assertEqual(graph["13"]["inputs"]["model"], ["12", 0])
        self.assertEqual(graph["13"]["inputs"]["ipadapter"], ["12", 1])
        self.assertEqual(graph["13"]["inputs"]["image"], ["10", 0])
        self.assertEqual(graph["13"]["inputs"]["weight"], 0.95)
        self.assertEqual(graph["13"]["inputs"]["end_at"], 1.0)
        self.assertEqual(graph["3"]["inputs"]["model"], ["13", 0])
        self.assertEqual(graph["3"]["inputs"]["latent_image"], ["5", 0])
        self.assertEqual(graph["3"]["inputs"]["denoise"], 1.0)
        self.assertNotIn("VAEEncode", {node["class_type"] for node in graph.values()})
        self.assertNotEqual(manifest["seed"], manifest["base_seed"])
        self.assertEqual(manifest["conditioning_mode"], "anchor_ipadapter_plus_face")
        self.assertEqual(manifest["latent_source"], "empty_latent")
        self.assertEqual(manifest["conditioning_fallback"], "disabled_fail_closed")
        self.assertEqual(manifest["anchor_image"], "charref_char_hero_anchor.png")

    def test_full_body_prioritizes_wide_framing_over_anchor_bust(self):
        graph, manifest = build_character_reference_workflow(
            CHARACTER,
            VISUAL,
            story_hash="story-hash",
            view="全身",
            anchor_image="charref_char_hero_anchor.png",
        )
        self.assertEqual(graph["13"]["inputs"]["weight"], 0.65)
        self.assertEqual(graph["5"]["inputs"]["width"], 512)
        self.assertEqual(graph["5"]["inputs"]["height"], 1024)
        self.assertIn("wide shot", manifest["positive_prompt"])
        self.assertIn("feet and shoes visible", manifest["positive_prompt"])
        self.assertIn("upper body", manifest["negative_prompt"])

    def test_anchor_uses_narrow_canvas_and_hard_single_subject_contract(self):
        graph, manifest = build_character_reference_workflow(
            CHARACTER,
            VISUAL,
            story_hash="story-hash",
            view="anchor",
        )
        self.assertEqual(graph["5"]["inputs"]["width"], 512)
        self.assertEqual(graph["5"]["inputs"]["height"], 1024)
        self.assertIn("(solo:1.6)", manifest["positive_prompt"])
        self.assertIn("(single character only:1.55)", manifest["positive_prompt"])
        self.assertIn("side-by-side people", manifest["negative_prompt"])
        self.assertIn("symmetry duplication", manifest["negative_prompt"])


if __name__ == "__main__":
    unittest.main()
