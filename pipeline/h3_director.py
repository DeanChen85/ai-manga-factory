"""Official-shape MiniMax H3 prompt compiler for narrative single shots.

This module is deterministic and provider-independent.  The LLM chooses a
canonical action code and fills the approved episode contract; this compiler is
the only authority that turns those fields into the final H3 text.  It follows
MiniMax's public base/Ref2VA section order while keeping each generated clip to
one visible action and one dominant camera path.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping, Sequence


H3_DIRECTOR_SKILL_VERSION = "ai-manga.h3-drama-director/v2"
H3_OFFICIAL_PROMPT_SHAPE = "minimax-h3-public-ref2va/2026-08"
H3_PROMPT_MIN_ENGLISH_WORDS = 120
H3_PROMPT_MAX_ENGLISH_WORDS = 512


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;.")


def _sentence(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    return text if text.endswith((".", "?", "!")) else text + "."


def english_word_count(value: Any) -> int:
    return len(re.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", str(value or "")))


def _speaker_map(cues: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for cue in cues:
        speaker = _clean(cue.get("speaker_id") or cue.get("speaker") or "character")
        if speaker not in result:
            result[speaker] = f"S{len(result) + 1}"
    return result


def _character_subject_map(
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Bind canonical episode character IDs to H3's model-visible subjects."""
    result: dict[str, str] = {}
    for binding in bindings:
        if _clean(binding.get("role")) not in {"character_anchor", "character_reference"}:
            continue
        source_id = _clean(binding.get("source_id"))
        if source_id and source_id not in result:
            result[source_id] = f"<Subject {len(result) + 1}>"
    return result


def _ground_character_ids(value: Any, subject_map: Mapping[str, str]) -> str:
    """Replace canonical IDs in approved action text with referenced subjects."""
    text = str(value or "")
    for source_id in sorted(subject_map, key=len, reverse=True):
        text = re.sub(
            rf"(?<![A-Za-z0-9_-]){re.escape(source_id)}(?![A-Za-z0-9_-])",
            subject_map[source_id],
            text,
        )
    return text


def _reference_sections(
    bindings: Sequence[Mapping[str, Any]],
    *,
    cast_count: int,
) -> tuple[str, str, str, dict[str, str]]:
    definitions: list[str] = []
    retention: list[str] = []
    subject_map = _character_subject_map(bindings)
    assigned_subjects: list[str] = []
    primary_picture_by_subject: dict[str, str] = {}
    unidentified_subjects = 0
    for index, binding in enumerate(bindings, 1):
        label = _clean(binding.get("model_label")) or f"<Picture {index}>"
        role = _clean(binding.get("role"))
        if role == "first_frame":
            definitions.append(
                f"{label} defines the approved opening composition; lock placements, wardrobe, props, and environment."
            )
            retention.append(
                f"{label}: fully_preserved opening identities, wardrobe, props, lighting, and layout."
            )
        elif role == "last_frame":
            definitions.append(
                f"{label} defines the approved final composition and completed physical state."
            )
            retention.append(
                f"{label} (final composition): fully_preserved - converge on its completed state and subject placement."
            )
        elif role in {"character_anchor", "character_reference"}:
            source_id = _clean(binding.get("source_id"))
            if source_id:
                subject = subject_map[source_id]
            else:
                unidentified_subjects += 1
                subject = f"<Subject {len(subject_map) + unidentified_subjects}>"
            if subject not in assigned_subjects:
                assigned_subjects.append(subject)
            approved_character = f" approved character {source_id}" if source_id else " character"
            if subject not in primary_picture_by_subject:
                primary_picture_by_subject[subject] = label
                definitions.append(
                    f"{subject} is the{approved_character} shown in {label}; lock face, hair, proportions, and wardrobe."
                )
                retention.append(
                    f"{subject}: fully_preserved from {label}; keep identity, wardrobe, proportions, and features throughout."
                )
            else:
                definitions.append(
                    f"{label} is an additional approved view of {subject}; it shows the same person and must never create another subject."
                )
                retention.append(
                    f"{label} (additional view of {subject}): fully_preserved - use it only to confirm the same face, wardrobe, proportions, and distinguishing features."
                )
        elif role == "scene_reference":
            definitions.append(
                f"{label} defines the approved location, layout, lighting, palette, weather, and props."
            )
            retention.append(
                f"{label}: fully_preserved environment layout, lighting, palette, weather, and props."
            )
        else:
            definitions.append(f"{label} is an approved visual reference for the target shot.")
            retention.append(
                f"{label} (visual reference): fully_preserved - retain the approved characteristics assigned to this reference."
            )
    if cast_count:
        definitions.append(
            f"The target shot contains exactly {cast_count} distinct visible character{'s' if cast_count != 1 else ''}; "
            "no person is added, omitted, merged, duplicated, or replaced."
        )
    summary = (
        "[reference generation] One continuous shot preserves approved identities and location while completing "
        "one observable action from opening to final state."
    )
    return " ".join(definitions), summary, " ".join(retention), subject_map


def compile_h3_director_prompt(
    *,
    style: str,
    aspect_ratio: str,
    duration_seconds: float,
    narrative_duration_seconds: float | None = None,
    scene: str,
    action: str,
    first_state: str,
    final_state: str,
    camera: str,
    continuity: str,
    cast_count: int,
    spoken_dialogue: Sequence[Mapping[str, Any]] = (),
    audio_cues: Sequence[Mapping[str, Any]] = (),
    ambience: str = "",
    music: str = "",
    bindings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compile one exact H3 prompt and return immutable audit metadata."""
    duration = float(duration_seconds)
    if not 4.0 <= duration <= 15.1:
        raise ValueError("H3 director prompt duration must stay within 4-15 seconds")
    required = {
        "style": style, "scene": scene, "action": action,
        "first_state": first_state, "final_state": final_state, "camera": camera,
    }
    missing = [key for key, value in required.items() if not _clean(value)]
    if missing:
        raise ValueError("H3 director prompt fields missing: " + ", ".join(missing))
    narrative_duration = (
        duration if narrative_duration_seconds is None else float(narrative_duration_seconds)
    )
    if not 1.5 <= narrative_duration <= duration + 1e-6:
        raise ValueError("H3 narrative duration must stay between 1.5 seconds and source duration")

    speakers = _speaker_map(spoken_dialogue)
    subject_map = _character_subject_map(bindings)
    dialogue_parts: list[str] = []
    for cue in spoken_dialogue:
        text = str(cue.get("text") or cue.get("line") or "").strip()
        if not text:
            continue
        speaker = _clean(cue.get("speaker_id") or cue.get("speaker") or "character")
        speaker_id = speakers[speaker]
        speaker_display = subject_map.get(speaker, speaker)
        delivery = _clean(cue.get("delivery_style") or "natural measured delivery")
        start = float(cue.get("start_seconds") or 0)
        end = float(cue.get("end_seconds") or duration)
        dialogue_parts.append(
            f"From {start:.3f} to {end:.3f} seconds, {speaker_display} ({speaker_id}) says with {delivery}: "
            f"<d>[Chinese] {text}</d> The speaker closes their lips when the line ends."
        )

    sound_parts = [_sentence(ambience)] if _clean(ambience) else []
    music_parts = [_sentence(music)] if _clean(music) and _clean(music).upper() != "N/A" else []
    seen_sounds: set[tuple[float, float, str]] = set()
    for cue in audio_cues:
        sound = _clean(cue.get("prompt") or cue.get("text") or cue.get("cue_type"))
        if sound:
            start = float(cue.get("start_seconds") or 0)
            end = float(cue.get("end_seconds") or duration)
            cue_type = _clean(cue.get("cue_type") or cue.get("type") or "sfx").casefold()
            identity = (round(start, 3), round(end, 3), f"{cue_type}:{sound.casefold()}")
            if identity in seen_sounds:
                continue
            seen_sounds.add(identity)
            timed = f"From {start:.3f} to {end:.3f} seconds, {sound}."
            if cue_type == "music":
                music_parts.append(timed)
            elif cue_type == "ambience":
                sound_parts.append("Ambient layer: " + timed)
            else:
                sound_parts.append(timed)
    soundscape = " ".join(part for part in sound_parts if part) or "Quiet natural room tone remains continuous."
    music_text = " ".join(part for part in music_parts if part) or "N/A"

    if abs(narrative_duration - duration) <= 1e-6:
        final_timing = (
            f"Final state at {duration:.3f} seconds: "
            f"{_clean(_ground_character_ids(final_state, subject_map))}."
        )
    else:
        completion_deadline = max(0.75, narrative_duration - 0.25)
        launch_end = min(0.5, completion_deadline / 2.0)
        grounded_final = _clean(_ground_character_ids(final_state, subject_map))
        final_timing = (
            f"The delivery edit cuts at {narrative_duration:.3f} seconds. "
            f"Micro-timeline: 0.000-{launch_end:.3f} seconds, begin that single action immediately without preparation; "
            f"{launch_end:.3f}-{completion_deadline:.3f} seconds, finish it into {grounded_final}; "
            f"by {completion_deadline:.3f} seconds, lock that final state. "
            f"Hold unchanged through {duration:.3f} seconds; no critical motion continues after {completion_deadline:.3f} seconds."
        )
    opening = (
        f"[Shot 1] {_sentence(style)} Opening state in the {aspect_ratio} composition: {_clean(_ground_character_ids(first_state, subject_map))}. "
        f"The approved location is {_clean(_ground_character_ids(scene, subject_map))}. The shot uses {_clean(_ground_character_ids(camera, subject_map))}. "
        f"The single visible action begins immediately: {_sentence(_ground_character_ids(action, subject_map))} "
        f"The motion develops continuously; no cut, replay, freeze, or unrelated action. "
        f"{final_timing} "
        f"{_sentence(_ground_character_ids(continuity, subject_map))} "
        "Every visible surface is uniformly blank and unlettered, using simple geometric color fields. "
        "Spoken lines are audio-only; delivery subtitles are composited only after generation."
    )
    detailed = " ".join([opening, *dialogue_parts])

    if bindings:
        definitions, summary, retention, _ = _reference_sections(
            bindings, cast_count=int(cast_count),
        )
        prompt = "\n\n".join([
            f"subject_definitions: {definitions}",
            f"summary: {summary}",
            f"retention_analysis: {retention}",
            f"detailed_description: {detailed}",
            f"overall_soundscape: {soundscape}",
            f"non_diegetic_music: {music_text}",
        ])
        mode = "Ref2VA"
        sections = (
            "subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music",
        )
    else:
        prompt = "\n\n".join([
            f"integrated_multimodal_description: {detailed}",
            f"overall_soundscape: {soundscape}",
            f"non_diegetic_music: {music_text}",
        ])
        mode = "T2VA"
        sections = (
            "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music",
        )

    words = english_word_count(prompt)
    if words < H3_PROMPT_MIN_ENGLISH_WORDS:
        raise ValueError(f"H3 prompt is under-specified: {words} English words")
    if words > H3_PROMPT_MAX_ENGLISH_WORDS:
        raise ValueError(f"H3 prompt exceeds bounded complexity: {words} English words")
    encoded = prompt.encode("utf-8")
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(encoded).hexdigest(),
        "prompt_bytes": len(encoded),
        "english_words": words,
        "mode": mode,
        "sections": list(sections),
        "skill_version": H3_DIRECTOR_SKILL_VERSION,
        "official_prompt_shape": H3_OFFICIAL_PROMPT_SHAPE,
    }
