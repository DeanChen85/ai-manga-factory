import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

import comfy_preflight


class ComfyPreflightTests(unittest.TestCase):
    def test_exact_nodes_models_lora_and_media_tools_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for folder, filename in (
                ("diffusion_models", comfy_preflight.H3_UNET),
                ("text_encoders", comfy_preflight.H3_CLIP),
                ("vae", comfy_preflight.H3_VIDEO_VAE),
                ("vae", comfy_preflight.H3_AUDIO_VAE),
                ("loras", comfy_preflight.H3_LORA_RECOMMENDED),
            ):
                path = root / "models" / folder / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"offline-placeholder")
            object_info = {name: {} for name in comfy_preflight.REQUIRED_NODES}
            with mock.patch.object(
                comfy_preflight, "_executable_status",
                return_value={"available": True, "path": "offline-tool"},
            ):
                result = comfy_preflight.run_preflight(root=root, object_info=object_info)
        self.assertTrue(result["passed"])
        self.assertEqual(result["nodes"]["missing_required"], [])
        self.assertEqual(result["models"]["missing_required"], [])
        self.assertEqual(Path(result["models"]["turbo_lora"]).name, comfy_preflight.H3_LORA_RECOMMENDED)

    def test_missing_runtime_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            comfy_preflight, "_executable_status",
            return_value={"available": False, "error": "missing"},
        ):
            result = comfy_preflight.run_preflight(root=temp, object_info={})
        self.assertFalse(result["passed"])
        self.assertTrue(result["nodes"]["missing_required"])
        self.assertTrue(result["models"]["missing_required"])
        self.assertIn("ffmpeg unavailable", result["failures"])

    def test_node_package_conflict_warning(self):
        """T8 + RH dual sampler both installed should warn."""
        fake_nodes = {
            "RHMiniMaxH3DualSigmaSampler": {},
            "MiniMaxH3AudioConditioningT8": {},
        }
        warnings = comfy_preflight._check_node_packages(fake_nodes)
        self.assertTrue(any("conflicts with" in w for w in warnings), f"Expected conflict warning, got: {warnings}")

    def test_no_conflict_when_only_one_installed(self):
        fake_nodes = {"RHMiniMaxH3DualSigmaSampler": {}}
        warnings = comfy_preflight._check_node_packages(fake_nodes)
        self.assertFalse(any("conflicts with" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
