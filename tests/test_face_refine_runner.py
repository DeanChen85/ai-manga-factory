#!/usr/bin/env python3
"""Tests for face_refine_runner.py — T8 wrapper, hermetic."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))

from face_refine_runner import (
    t8_node_registered, build_face_refine_workflow, run_face_refine,
    T8_FACE_REFINE_NODES, FaceRefineResult,
)


class TestBuildWorkflow(unittest.TestCase):
    def test_safe_profile_omits_skin_finish(self):
        wf = build_face_refine_workflow(
            input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
            face_ref=Path("fr.png"), output_prefix="test", profile="safe",
        )
        # Should not contain skin_finish node in safe profile
        from face_refine_runner import T8_FACE_REFINE_NODES
        self.assertNotIn(T8_FACE_REFINE_NODES["skin_finish"], str(wf))

    def test_aggressive_includes_skin_finish(self):
        wf = build_face_refine_workflow(
            input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
            face_ref=Path("fr.png"), output_prefix="test", profile="aggressive",
        )
        from face_refine_runner import T8_FACE_REFINE_NODES
        self.assertIn(T8_FACE_REFINE_NODES["skin_finish"], str(wf))

    def test_invalid_profile_raises(self):
        with self.assertRaises(ValueError):
            build_face_refine_workflow(
                input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
                face_ref=Path("fr.png"), output_prefix="test", profile="bogus",
            )

    def test_workflow_has_face_refine_plan(self):
        wf = build_face_refine_workflow(
            input_mp4=Path("in.mp4"), first_frame=Path("ff.png"),
            face_ref=Path("fr.png"), output_prefix="test", profile="safe",
        )
        self.assertIn(T8_FACE_REFINE_NODES["face_refine_plan"], str(wf))
        self.assertIn(T8_FACE_REFINE_NODES["face_refine"], str(wf))


class TestRunWithT8Missing(unittest.TestCase):
    """When T8 is not installed (node not registered), should skip cleanly."""

    def setUp(self):
        self.tmp = Path("test_face_refine_runner.tmp")
        self.tmp.mkdir(exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skips_when_node_not_registered(self):
        # We don't actually need a real mp4 because we'll skip before that
        result = run_face_refine(
            input_mp4=self.tmp / "fake.mp4",
            face_reference=self.tmp / "fr.png",
            output_dir=self.tmp,
            profile="safe",
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.skipped)
        self.assertIsNone(result.output_path)
        self.assertIn("not registered", result.skip_reason)


class TestNodeRegisteredCheck(unittest.TestCase):
    def test_returns_false_on_network_error(self):
        # Invalid URL => exception => False
        self.assertFalse(t8_node_registered("SomeNode", "http://invalid-host:99999"))


if __name__ == "__main__":
    unittest.main()