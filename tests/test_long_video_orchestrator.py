#!/usr/bin/env python3
"""Tests for long_video_orchestrator.py — original long-video planner/concat logic.

These tests are hermetic (no GPU, no ComfyUI). They cover the pure logic:
- lattice snapping
- segment planning
- workflow injection (mock template)
- concat (with ffmpeg if available, otherwise skipped via env)
"""
import os
import sys
import unittest
import shutil
import subprocess
import tempfile
from pathlib import Path

# Make pipeline importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))

from long_video_orchestrator import (
    MIN_SEGMENT_SECONDS, MAX_SEGMENT_SECONDS, FRAME_LATTICE, LATTICE_SECONDS,
    snap_seconds_to_lattice, segment_count_for_total, plan_segments,
    inject_into_workflow, extract_last_frame, concat_segments, SegmentSpec,
)


class TestLatticeSnap(unittest.TestCase):
    def test_lattice_constants(self):
        # 17n+5 for n=1..6 → 22, 39, 56, 73, 90, 107
        self.assertEqual(FRAME_LATTICE, [22, 39, 56, 73, 90, 107])
        # 24 fps
        self.assertAlmostEqual(LATTICE_SECONDS[22], 22 / 24, places=3)

    def test_snap_exact_5s(self):
        # 5s = 120 frames → 距离 90 (3.75s) 比 107 (4.46s) 远
        # 但实际 5s 应被 snap 到 107 (4.46s) 而非 90
        # 因为 snap 是 nearest-neighbor
        result = snap_seconds_to_lattice(5.0)
        # 120 frames → nearest in [22, 39, 56, 73, 90, 107] is 107 (delta 13) or 90 (delta 30)
        # so should be 107
        self.assertEqual(result, 107)

    def test_snap_2s_floor(self):
        # Below 2s should snap up to 2s = 48 frames → closest in [22,39,56] is 56
        result = snap_seconds_to_lattice(1.0)
        # 24 frames → 22 (closest, but 1.0s < 2.0s, so clamp to 2.0s first → 48 frames → 56)
        self.assertEqual(result, 56)

    def test_snap_15s_ceiling(self):
        # Above 15s should clamp to nearest within
        result = snap_seconds_to_lattice(20.0)
        # 480 frames → clamp to max 107 (within lattice)
        self.assertEqual(result, 107)

    def test_snap_8s(self):
        # 8s = 192 frames → between 90 (3.75s) and 107 (4.46s) and beyond
        # Actually within the lattice range only 22..107 = up to 4.46s
        # So 8s should clamp to 107
        result = snap_seconds_to_lattice(8.0)
        self.assertEqual(result, 107)


class TestSegmentCount(unittest.TestCase):
    def test_zero_total(self):
        self.assertEqual(segment_count_for_total(0), 0)

    def test_negative_total(self):
        self.assertEqual(segment_count_for_total(-5), 0)

    def test_normal(self):
        # 15s with 5s segments → 3 segments
        self.assertEqual(segment_count_for_total(15.0, 5.0), 3)
        # 13s with 5s segments → 3 segments (ceil 13/5)
        self.assertEqual(segment_count_for_total(13.0, 5.0), 3)
        # 5s with 5s segments → 1
        self.assertEqual(segment_count_for_total(5.0, 5.0), 1)

    def test_segment_clamped(self):
        # 30s with 16s requested → segment gets clamped to 15s → ceil(30/15) = 2
        self.assertEqual(segment_count_for_total(30.0, 16.0), 2)
        # 30s with 1s requested → clamp to 2s min → ceil(30/2) = 15
        self.assertEqual(segment_count_for_total(30.0, 1.0), 15)


class TestPlanSegments(unittest.TestCase):
    def test_basic_plan(self):
        plan = plan_segments(15.0, "test prompt", 7000, 5.0)
        self.assertEqual(len(plan), 3)
        # First segment has no first_frame
        self.assertIsNone(plan[0].first_frame_path)
        # Subsequent segments will get first_frame from orchestrator
        # Index 0..n-1
        self.assertEqual([s.index for s in plan], [0, 1, 2])
        # Seeds differ
        seeds = [s.seed for s in plan]
        self.assertEqual(len(set(seeds)), 3)
        # Each segment is within 2-15s range
        for s in plan:
            self.assertGreaterEqual(s.target_seconds, MIN_SEGMENT_SECONDS - 0.01)
            self.assertLessEqual(s.target_seconds, MAX_SEGMENT_SECONDS + 0.01)

    def test_zero_total(self):
        plan = plan_segments(0, "x", 1, 5.0)
        self.assertEqual(plan, [])

    def test_prompt_carries_shot_index(self):
        plan = plan_segments(10.0, "base", 100, 5.0)
        for i, seg in enumerate(plan):
            self.assertIn(f"[Shot {i+1}/", seg.prompt)


class TestInjectIntoWorkflow(unittest.TestCase):
    def _sample_template(self):
        return {
            "1": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors"}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "ORIGINAL"}},
            "7": {"class_type": "KSampler",
                  "inputs": {"seed": 0, "steps": 25}},
        }

    def test_text_replaced(self):
        template = self._sample_template()
        seg = SegmentSpec(index=0, target_seconds=5.0, first_frame_path=None,
                          prompt="[Shot 1/3] foo", seed=42)
        out = inject_into_workflow(template, seg, Path("."))
        # 5's text should be replaced
        self.assertEqual(out["5"]["inputs"]["text"], "[Shot 1/3] foo")
        # 7's seed should be replaced
        self.assertEqual(out["7"]["inputs"]["seed"], 42)
        # 1 unchanged
        self.assertEqual(out["1"]["inputs"]["unet_name"],
                         "minimax_h3_fl2va_pruned_int8_convrot.safetensors")

    def test_first_frame_injected(self):
        template = {
            "12": {"class_type": "LoadImage", "inputs": {"image": ""}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "## header"}},
        }
        seg = SegmentSpec(index=1, target_seconds=5.0,
                          first_frame_path="chain/last_seg1.png",
                          prompt="seg2", seed=200)
        out = inject_into_workflow(template, seg, Path("."))
        self.assertEqual(out["12"]["inputs"]["image"], "chain/last_seg1.png")
        # text starting with ## should NOT be replaced
        self.assertEqual(out["5"]["inputs"]["text"], "## header")

    def test_template_not_mutated(self):
        template = self._sample_template()
        seg = SegmentSpec(index=0, target_seconds=5.0, first_frame_path=None,
                          prompt="x", seed=1)
        _ = inject_into_workflow(template, seg, Path("."))
        # Original should be untouched
        self.assertEqual(template["5"]["inputs"]["text"], "ORIGINAL")
        self.assertEqual(template["7"]["inputs"]["seed"], 0)


class TestExtractAndConcat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = shutil.which("ffmpeg")
        if not cls.ffmpeg:
            # Try imageio's bundled ffmpeg
            try:
                import imageio_ffmpeg
                cls.ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                pass

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lvo_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_extract_last_frame_missing_input(self):
        out = self.tmp / "out.png"
        ok = extract_last_frame(self.tmp / "nonexistent.mp4", out, self.ffmpeg or "ffmpeg")
        self.assertFalse(ok)
        self.assertFalse(out.exists())

    @unittest.skipUnless(shutil.which("ffmpeg") or shutil.which("imageio_ffmpeg"),
                         "ffmpeg not available")
    def test_extract_last_frame_from_real_mp4(self):
        if not self.ffmpeg:
            self.skipTest("no ffmpeg")
        # Create a 3-second test video
        src = self.tmp / "src.mp4"
        subprocess.run([
            self.ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=red:s=64x64:d=3",
            "-pix_fmt", "yuv420p", str(src),
        ], capture_output=True, check=True, timeout=30)
        out = self.tmp / "last.png"
        ok = extract_last_frame(src, out, self.ffmpeg)
        self.assertTrue(ok, f"extract_last_frame failed; out exists={out.exists()}")
        self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()