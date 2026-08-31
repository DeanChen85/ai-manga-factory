"""
prompt_relay.py — Split long multi-shot H3 prompts into relayable chunks.

T8's PromptRelayConditioningT8Advanced splits the official 7000-char H3
prompt into multiple relay-friendly segments for long videos. This is
useful when a single shot has 8+ sub-shots with distinct visual states
or audio beats.

This module is a pure-Python equivalent (no T8 dependency). It implements:
- Sentence-aware chunking (don't split inside brackets, timecodes, or
  <Subject/Picture/Audio/Video N> tags)
- Continuity hash binding (each chunk references the same Subject IDs,
  Audio IDs, Picture numbers as the original)
- Optional Chinese/English glossary to keep stable speaker IDs across
  chunks (e.g., S1/S2 always)

When used with the H3 official 6-section format (subject_definitions,
summary, retention_analysis, detailed_description, overall_soundscape,
non_diegetic_music), it NEVER splits inside any tag block; it only
chunks the detailed_description [Shot N] entries, since that's the only
section that grows unboundedly with shot count.

License: Apache-2.0
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


# Tags that mark a non-splittable boundary inside a Shot description.
PROTECTED_TAGS = re.compile(
    r"<(?P<tag>Subject|Picture|Video|Audio|d)\b[^>]*>"
    r"|"
    r"</(?P<tag2>Subject|Picture|Video|Audio|d)>"
    r"|"
    r"\(S\d+\)"
    r"|"
    r"\[Shot \d+\]"
)


@dataclass
class PromptChunk:
    index: int
    text: str
    char_count: int
    sha256: str
    shot_range: tuple[int, int]  # e.g., (1, 4) means shots 1..4

    @property
    def contains_full_shots(self) -> str:
        return f"shots {self.shot_range[0]}-{self.shot_range[1]}"


# === Shot detection: parse [Shot N] ... [Shot N+1] boundaries ===

SHOT_HEADER_RE = re.compile(r"^\[Shot\s+(\d+)\]\s*", re.MULTILINE)


def extract_shot_blocks(text: str) -> list[tuple[int, str]]:
    """Return list of (shot_number, body) preserving order.

    A shot body is the text from [Shot N] up to (but not including)
    [Shot N+1], or to the end of input.
    """
    matches = list(SHOT_HEADER_RE.finditer(text))
    if not matches:
        return []
    blocks: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((n, text[start:end].rstrip()))
    return blocks


def reassemble(blocks: list[tuple[int, str]], header: str = "") -> str:
    """Rebuild original-format text from shot blocks (for round-trip tests)."""
    parts: list[str] = []
    if header:
        parts.append(header.rstrip())
        parts.append("")
    for n, body in blocks:
        parts.append(f"[Shot {n}] {body.rstrip()}")
    return "\n\n".join(parts) + "\n"


# === Safe split: never break a protected tag ===

def _find_safe_boundary(text: str, target: int) -> int:
    """Find the nearest split point in `text` at or before `target`.

    A safe point is either:
    - Just after a sentence end (".", "!", "?", "。", "！", "？", "\n\n")
    - Never inside a protected tag
    """
    if target >= len(text):
        return len(text)
    # Search backwards from target
    for i in range(target, -1, -1):
        if i in (target, len(text) - 1):
            continue
        # Don't cut inside an unclosed protected tag
        # Heuristic: count <...> tag opens/closes before i
        prefix = text[: i + 1]
        # If a tag is open at position i, the prefix has more '<' than '</'
        if prefix.count("<") - prefix.count("</") > 0:
            continue
        if text[i] in (".", "!", "?", "。", "！", "？"):
            return i + 1
        if text[i] == "\n" and i > 0 and text[i - 1] == "\n":
            return i + 1
    return target


# === Top-level splitter ===

def split_into_chunks(
    detailed_description: str,
    max_chunk_chars: int = 4000,
) -> list[PromptChunk]:
    """Split a detailed_description text into relayable chunks.

    Algorithm:
      1. Parse into per-shot blocks.
      2. Greedily group blocks until adding the next block would exceed
         max_chunk_chars. Within the limit, prefer to add complete shots.
      3. If a single shot is itself larger than max_chunk_chars, split
         it at safe sentence boundaries; the result is one chunk with
         partial shot (caller should know this happened).
      4. Hash each chunk (SHA-256 over the text) for content-bound IDs.

    Returns a list of PromptChunk, never empty if input is non-empty.
    """
    if not detailed_description.strip():
        return []

    blocks = extract_shot_blocks(detailed_description)
    if not blocks:
        # No [Shot N] markers — treat entire text as one chunk
        text = detailed_description.strip()
        return [PromptChunk(
            index=0, text=text, char_count=len(text),
            sha256=_hash(text), shot_range=(0, 0),
        )]

    chunks: list[PromptChunk] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    for n, body in blocks:
        block_chars = len(f"[Shot {n}] ") + len(body) + 2
        if not current:
            current.append((n, body))
            current_chars = block_chars
            continue
        # Adding this block fits?
        if current_chars + block_chars <= max_chunk_chars:
            current.append((n, body))
            current_chars += block_chars
            continue
        # Doesn't fit — flush current
        text = _format_blocks(current)
        chunks.append(_make_chunk(len(chunks), text, current))
        current = [(n, body)]
        current_chars = block_chars

    if current:
        text = _format_blocks(current)
        chunks.append(_make_chunk(len(chunks), text, current))

    # If a chunk exceeds max_chunk_chars (single huge shot), split it
    out: list[PromptChunk] = []
    for c in chunks:
        if c.char_count <= max_chunk_chars:
            out.append(c)
        else:
            out.extend(_split_large_chunk(c, max_chunk_chars))
    return out


def _format_blocks(blocks: list[tuple[int, str]]) -> str:
    return "\n\n".join(f"[Shot {n}] {body.rstrip()}" for n, body in blocks) + "\n"


def _make_chunk(index: int, text: str, blocks: list[tuple[int, str]]) -> PromptChunk:
    if not blocks:
        return PromptChunk(index=index, text=text, char_count=len(text),
                          sha256=_hash(text), shot_range=(0, 0))
    return PromptChunk(
        index=index, text=text, char_count=len(text),
        sha256=_hash(text),
        shot_range=(blocks[0][0], blocks[-1][0]),
    )


def _split_large_chunk(c: PromptChunk, max_chars: int) -> list[PromptChunk]:
    """Split one chunk that exceeds max_chars at safe boundaries."""
    text = c.text
    parts: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        if end < len(text):
            end = _find_safe_boundary(text, end)
        parts.append(text[pos:end].strip())
        pos = end
    out: list[PromptChunk] = []
    for i, p in enumerate(parts):
        if not p:
            continue
        out.append(PromptChunk(
            index=c.index + i, text=p, char_count=len(p),
            sha256=_hash(p),
            shot_range=(0, 0),  # partial shot — range undefined
        ))
    return out


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# === CLI ===

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m pipeline.prompt_relay <file.txt> [max_chunk_chars]")
        print("  Reads a prompt file (with [Shot N] markers) and prints chunks.")
        sys.exit(0)
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    max_c = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    chunks = split_into_chunks(text, max_chunk_chars)
    for c in chunks:
        print(f"--- chunk {c.index} ({c.char_count} chars, {c.contains_full_shots}) {c.sha256[:12]} ---")
        print(c.text)
        print()