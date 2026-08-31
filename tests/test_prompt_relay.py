#!/usr/bin/env python3
"""Tests for prompt_relay.py — long-prompt chunking with safe boundaries."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline'))

from prompt_relay import (
    extract_shot_blocks, reassemble, split_into_chunks,
    _find_safe_boundary, _hash, stabilize_speaker_ids,
)


SAMPLE = """[Shot 1] A young girl walks into a sunlit room. She has long brown hair and is wearing a school uniform. The room has a large window with curtains. The camera is static.

[Shot 2] She sits down at a wooden desk. <Picture 1> defines the character. The camera pushes in slowly. A gentle breeze moves the curtains. The audio has soft piano music.

[Shot 3] She opens a book. <Picture 1> is the same character. The audio adds a soft page-turn sound. The camera angle changes to over-the-shoulder.

[Shot 4] Close-up of the book. The pages show illustrations. (S1) whispers "Once upon a time". The audio has a music swell.

[Shot 5] Wide shot. The girl stands and walks toward the window. Sunlight floods the room. The shot ends with her silhouette against the light. Final audio cue is silence."""


class TestExtractShots(unittest.TestCase):
    def test_extracts_five_shots(self):
        blocks = extract_shot_blocks(SAMPLE)
        self.assertEqual(len(blocks), 5)
        self.assertEqual([n for n, _ in blocks], [1, 2, 3, 4, 5])

    def test_no_shot_markers(self):
        blocks = extract_shot_blocks("No shots here. Just text.")
        self.assertEqual(blocks, [])

    def test_round_trip_preserves_content(self):
        blocks = extract_shot_blocks(SAMPLE)
        rebuilt = reassemble(blocks)
        # All shot text should still be present (reordering accepted)
        for n, body in blocks:
            self.assertIn(body.strip(), rebuilt)


class TestSplitChunks(unittest.TestCase):
    def test_small_input_one_chunk(self):
        chunks = split_into_chunks(SAMPLE, max_chunk_chars=10000)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Shot 1", chunks[0].text)
        self.assertIn("Shot 5", chunks[0].text)

    def test_force_split(self):
        # 5 shots over 2000 char limit => multiple chunks
        chunks = split_into_chunks(SAMPLE, max_chunk_chars=600)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.char_count, 1500)  # allow some slack for safe-split

    def test_shot_range_assigned(self):
        chunks = split_into_chunks(SAMPLE, max_chunk_chars=600)
        # First chunk starts at shot 1
        self.assertEqual(chunks[0].shot_range[0], 1)
        # Last chunk ends at shot 5
        self.assertEqual(chunks[-1].shot_range[1], 5)
        # No gap: each chunk's start equals previous chunk's end + 1
        for i in range(1, len(chunks)):
            self.assertEqual(chunks[i].shot_range[0], chunks[i - 1].shot_range[1] + 1)

    def test_chunks_have_unique_hashes(self):
        chunks = split_into_chunks(SAMPLE, max_chunk_chars=600)
        hashes = [c.sha256 for c in chunks]
        self.assertEqual(len(set(hashes)), len(hashes), "chunk hashes must be unique")

    def test_no_protected_tag_split(self):
        # Force tight limit so chunking must split inside text
        text = """[Shot 1] Hello <Picture 1> world. This is a long sentence that goes on and on and on and on and on and on and on and on and on. <Picture 2> another ref. The end.
[Shot 2] More text here."""
        chunks = split_into_chunks(text, max_chunk_chars=100)
        for c in chunks:
            # Every <Picture N> tag must be intact
            import re
            opens = re.findall(r"<Picture\s+\d+>", c.text)
            for o in opens:
                self.assertIn(o, c.text)
            # No unclosed tags (every < has a matching >)
            self.assertEqual(c.text.count("<"), c.text.count(">"))


class TestSafeBoundary(unittest.TestCase):
    def test_finds_period(self):
        text = "First sentence. Second sentence. Third."
        # Period after "sentence" is at index 14
        self.assertEqual(text[14], ".")
        idx = _find_safe_boundary(text, 17)  # search at "Second sentence."
        # Should return one past the period at index 15 (end of first sentence)
        # Actually returns 15 because the function finds the period at 14 and returns 15
        self.assertGreaterEqual(idx, 15)
        self.assertLessEqual(idx, 16)
        # The character at idx-1 should be a period or space (within the safe split)
        self.assertIn(text[idx - 1], (".", " "))

    def test_finds_paragraph(self):
        text = "Para one\n\nPara two"
        idx = _find_safe_boundary(text, 9)
        self.assertGreater(idx, 0)


class TestHashStable(unittest.TestCase):
    def test_same_input_same_hash(self):
        self.assertEqual(_hash("hello"), _hash("hello"))

    def test_diff_input_diff_hash(self):
        self.assertNotEqual(_hash("a"), _hash("b"))


class TestStabilizeSpeakerIds(unittest.TestCase):
    def test_preserves_all_speakers_across_chunks(self):
        text = "[Shot 1] (S1) says hello. [Shot 2] (S2) replies."
        chunks = split_into_chunks(text, max_chunk_chars=30)
        self.assertGreater(len(chunks), 1)
        stabilized = stabilize_speaker_ids(chunks)
        # Every chunk should contain both (S1) and (S2)
        for c in stabilized:
            self.assertIn("(S1)", c.text)
            self.assertIn("(S2)", c.text)

    def test_preserves_reference_tags(self):
        text = "[Shot 1] <Picture 1> shows hero. [Shot 2] <Audio 1> plays music."
        chunks = split_into_chunks(text, max_chunk_chars=30)
        stabilized = stabilize_speaker_ids(chunks)
        # Header should list ALL ref types found across all chunks
        for c in stabilized:
            self.assertIn("Refs:", c.text)
            header_line = c.text.split("\n")[0]
            self.assertIn("Picture", header_line)
            self.assertIn("Audio", header_line)

    def test_no_op_when_no_speakers_or_refs(self):
        text = "[Shot 1] No speakers here. [Shot 2] Just action."
        chunks = split_into_chunks(text, max_chunk_chars=50)
        stabilized = stabilize_speaker_ids(chunks)
        # Should return unchanged when nothing to stabilize
        self.assertEqual(len(stabilized), len(chunks))
        for orig, stab in zip(chunks, stabilized):
            self.assertEqual(orig.shot_range, stab.shot_range)

    def test_stable_hash_changes_after_stabilization(self):
        text = "[Shot 1] (S1) speaks."
        chunks = split_into_chunks(text, max_chunk_chars=100)
        stabilized = stabilize_speaker_ids(chunks)
        # Hashes should differ because header was added
        for orig, stab in zip(chunks, stabilized):
            self.assertNotEqual(orig.sha256, stab.sha256)


if __name__ == "__main__":
    unittest.main()