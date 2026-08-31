#!/usr/bin/env python3
"""Tests for video_upscale.py — FlashVSR via T8 wrapper, hermetic."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))

from video_upscale import (
    t8_node_registered, build_flashvsr_workflow, run_upscale,
    T8_FLASHVSR_NODES, FLASHVSR_PROFILES, UpscaleResult,
)


class TestBuildWorkflow(unittest.TestCase):
    @staticmethod
    def _find_node(wf, class_name):
        for node in wf.values():
            if isinstance(node, dict) and node.get("class_type") == class_name:
                return node
        raise KeyError(f"node {class_name} not found in workflow")

    def test_quality_locked_profile(self):
        wf = build_flashvsr_workflow(
            input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
            output_prefix="x", profile="quality_locked", scale=2.0,
        )
        self.assertIn(T8_FLASHVSR_NODES["restore"], str(wf))
        self.assertIn(T8_FLASHVSR_NODES["execution_plan"], str(wf))
        self.assertIn(T8_FLASHVSR_NODES["model"], str(wf))

    def test_memory_safe_uses_tiled(self):
        wf = build_flashvsr_workflow(
            input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
            output_prefix="x", profile="memory_safe", scale=2.0,
        )
        # Find the execution_plan node (whatever its key)
        plan = self._find_node(wf, T8_FLASHVSR_NODES["execution_plan"])
        self.assertTrue(plan["inputs"]["tiled"])

    def test_balanced_dynamic_has_low_motion(self):
        wf = build_flashvsr_workflow(
            input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
            output_prefix="x", profile="balanced_dynamic", scale=2.0,
        )
        plan = self._find_node(wf, T8_FLASHVSR_NODES["execution_plan"])
        self.assertTrue(plan["inputs"]["low_motion_budget"])

    @staticmethod
    def _find_node(wf, class_name):
        for node in wf.values():
            if isinstance(node, dict) and node.get("class_type") == class_name:
                return node
        raise KeyError(f"node {class_name} not found in workflow")

    def test_invalid_profile_raises(self):
        with self.assertRaises(ValueError):
            build_flashvsr_workflow(
                input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
                output_prefix="x", profile="unknown", scale=2.0,
            )

    def test_invalid_scale_raises(self):
        with self.assertRaises(ValueError):
            build_flashvsr_workflow(
                input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
                output_prefix="x", profile="quality_locked", scale=3.0,
            )

    def test_scale_4x_works(self):
        wf = build_flashvsr_workflow(
            input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
            output_prefix="x", profile="quality_locked", scale=4.0,
        )
        plan = self._find_node(wf, T8_FLASHVSR_NODES["execution_plan"])
        self.assertEqual(plan["inputs"]["scale"], 4.0)


class TestRunSkipsOnMissingT8(unittest.TestCase):
    def setUp(self):
        self.tmp = Path("test_video_upscale.tmp")
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skips_when_t8_missing(self):
        result = run_upscale(
            input_mp4=self.tmp / "fake.mp4",
            output_dir=self.tmp,
            profile="quality_locked",
            scale=2.0,
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.skipped)
        self.assertIsNone(result.output_path)
        self.assertIn("not registered", result.skip_reason)


if __name__ == "__main__":
    unittest.main()