# -*- coding: utf-8 -*-
"""Canonical, deterministic action contract for storyboard and H3 prompts.

This module is deliberately dependency-free.  It never calls an LLM and never
guesses a semantic action from prose.  Approved action codes are the wire
contract; human-readable H3 text is a derived, hashed representation.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


ACTION_CATALOG_VERSION = "ai-manga.action/v1"

# One authoritative catalog.  ``legacy_aliases`` are exact compatibility
# spellings accepted from old structured ``action_components.verb`` values;
# they are not fuzzy matches or translations performed at runtime.
ACTION_CATALOG: dict[str, dict[str, Any]] = {
    "MOVE_TOWARD": {"h3_verb": "moves toward", "legacy_aliases": ("move", "moves toward", "walk", "run", "step", "走向", "跑向")},
    "ENTER_SPACE": {"h3_verb": "enters", "legacy_aliases": ("enter", "enters", "进入")},
    "EXIT_SPACE": {"h3_verb": "exits", "legacy_aliases": ("exit", "exits", "离开", "走出")},
    "TURN_TOWARD": {"h3_verb": "turns toward", "legacy_aliases": ("turn", "turns toward", "转身")},
    "LOOK_AT": {"h3_verb": "looks at", "legacy_aliases": ("look", "looks at", "望向", "注视")},
    "REACH_FOR": {"h3_verb": "reaches for", "legacy_aliases": ("reach", "reaches for", "伸手")},
    "PICK_UP": {"h3_verb": "picks up", "legacy_aliases": ("pick", "picks up", "lift", "raise", "拿起", "抬起", "举起")},
    "HOLD_OBJECT": {"h3_verb": "holds", "legacy_aliases": ("hold", "holds", "grab", "抓住", "握住", "按住")},
    "PLACE_OBJECT": {"h3_verb": "places", "legacy_aliases": ("place", "places", "set", "放下")},
    "SLIDE_OBJECT": {"h3_verb": "slides", "legacy_aliases": ("slide", "slides", "推到")},
    "PUSH_OBJECT": {"h3_verb": "pushes", "legacy_aliases": ("push", "pushes")},
    "PULL_OBJECT": {"h3_verb": "pulls", "legacy_aliases": ("pull", "pulls", "拉开")},
    "OPEN_OBJECT": {"h3_verb": "opens", "legacy_aliases": ("open", "opens", "打开", "推开", "揭开")},
    "CLOSE_OBJECT": {"h3_verb": "closes", "legacy_aliases": ("close", "closes", "关上")},
    "PRESS_CONTROL": {"h3_verb": "presses", "legacy_aliases": ("press", "presses", "按下")},
    "HAND_OBJECT": {"h3_verb": "hands", "legacy_aliases": ("hand", "hands", "pass", "递给", "递出")},
    "POINT_AT": {"h3_verb": "points at", "legacy_aliases": ("point", "points at", "指向")},
    "DROP_OBJECT": {"h3_verb": "drops", "legacy_aliases": ("drop", "drops", "掉落", "扔下", "摔下")},
    "THROW_OBJECT": {"h3_verb": "throws", "legacy_aliases": ("throw", "throws", "slam", "砸向")},
    "CATCH_OBJECT": {"h3_verb": "catches", "legacy_aliases": ("catch", "catches", "接住")},
    "REVEAL_OBJECT": {"h3_verb": "reveals", "legacy_aliases": ("reveal", "reveals", "露出")},
    "HIDE_OBJECT": {"h3_verb": "hides", "legacy_aliases": ("hide", "hides", "藏起")},
    "STAND_UP": {"h3_verb": "stands up beside", "legacy_aliases": ("stand", "stands up", "站起")},
    "SIT_DOWN": {"h3_verb": "sits down on", "legacy_aliases": ("sit", "sits down", "坐下")},
    "CROUCH_DOWN": {"h3_verb": "crouches beside", "legacy_aliases": ("duck", "crouches", "蹲下")},
    "STRIKE_OBJECT": {"h3_verb": "strikes", "legacy_aliases": ("strike", "strikes", "knock", "敲击")},
    "TEAR_OBJECT": {"h3_verb": "tears", "legacy_aliases": ("tear", "tears", "撕开")},
    "POUR_CONTENT": {"h3_verb": "pours into", "legacy_aliases": ("pour", "pours into", "倒入")},
    "INSERT_OBJECT": {"h3_verb": "inserts into", "legacy_aliases": ("insert", "inserts into", "放入")},
    "REMOVE_OBJECT": {"h3_verb": "removes from", "legacy_aliases": ("remove", "removes from", "收回", "移开")},
    "UNLOCK_OBJECT": {"h3_verb": "unlocks", "legacy_aliases": ("unlock", "unlocks", "解锁")},
    "LOCK_OBJECT": {"h3_verb": "locks", "legacy_aliases": ("lock", "locks", "锁上")},
    "WIPE_SURFACE": {"h3_verb": "wipes", "legacy_aliases": ("wipe", "wipes", "擦去")},
    "WRITE_MARK": {"h3_verb": "writes on", "legacy_aliases": ("write", "writes on", "draw")},
}
ACTION_CODES = tuple(ACTION_CATALOG)


class ActionContractError(ValueError):
    """Raised when an action wire cannot be validated without guessing."""


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip(" \t\r\n,;.")


def _consistent(name: str, *values: Any) -> str:
    present = [_text(value) for value in values if _text(value)]
    if len(set(present)) > 1:
        raise ActionContractError(f"{name} fields disagree")
    return present[0] if present else ""


def _legacy_alias_index() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for code, definition in ACTION_CATALOG.items():
        for value in definition["legacy_aliases"]:
            key = _text(value).casefold()
            previous = aliases.get(key)
            if previous and previous != code:
                raise RuntimeError(f"ambiguous legacy action alias: {value}")
            aliases[key] = code
    return aliases


_LEGACY_ALIAS_TO_CODE = _legacy_alias_index()


def migrate_legacy_action_exact(value: Any) -> str | None:
    """Return an explicitly registered code; unknown prose is never inferred."""
    normalized = _text(value).casefold()
    return _LEGACY_ALIAS_TO_CODE.get(normalized) if normalized else None


def _drop_target_overlap(target: str, end_state: str) -> tuple[str, bool]:
    """Remove only a literal target suffix/prefix overlap, never paraphrase."""
    target_words = target.split()
    end_words = end_state.split()
    for count in range(min(len(target_words), len(end_words)), 0, -1):
        if [word.casefold() for word in target_words[-count:]] == [
            word.casefold() for word in end_words[:count]
        ]:
            return " ".join(end_words[count:]), True
    if end_state.casefold().startswith(target.casefold()):
        return end_state[len(target):].lstrip(" ,;:-"), True
    return end_state, False


def _compile_h3_sentence(
    *, actor_id: str, action_code: str, target: str,
    start_state: str, end_state: str,
) -> str:
    verb = str(ACTION_CATALOG[action_code]["h3_verb"])
    final_text, overlapped = _drop_target_overlap(target, end_state)
    if overlapped:
        if not final_text:
            final_text = "in the approved final state"
        if re.match(r"^(?:is|are|remains?|rests?|stays?|stands?|lies|sits|has|have)\b", final_text, re.I):
            final_text = "it " + final_text
        else:
            final_text = "it is " + final_text
    return (
        f"Beginning with {start_state}, {actor_id} {verb} {target}, "
        f"ending with {final_text}."
    )


def compile_action_spec(
    spec: Mapping[str, Any],
    *,
    visible_character_ids: Sequence[str],
    start_state: Any = None,
    end_state: Any = None,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Validate and compile one canonical action without semantic invention."""
    if not isinstance(spec, Mapping):
        raise ActionContractError("action spec must be an object")
    version = _text(spec.get("catalog_version"))
    if version and version != ACTION_CATALOG_VERSION:
        raise ActionContractError(f"unsupported catalog_version: {version}")

    visible = [_text(value) for value in visible_character_ids]
    if not visible or any(not value for value in visible):
        raise ActionContractError("visible_character_ids must be non-empty strings")
    if len(visible) != len(set(visible)):
        raise ActionContractError("visible_character_ids must not contain duplicates")

    actor_id = _consistent("actor_id", spec.get("actor_id"), spec.get("sub"))
    action_code = _consistent("action_code", spec.get("action_code"), spec.get("code"))
    target = _consistent("target", spec.get("target"), spec.get("obj"))
    resolved_start = _consistent("start_state", spec.get("start_state"), start_state)
    resolved_end = _consistent("end_state", spec.get("end_state"), end_state)

    if not action_code and allow_legacy:
        action_code = migrate_legacy_action_exact(spec.get("verb")) or ""
        if not action_code and _text(spec.get("verb")):
            raise ActionContractError("legacy verb is not an exact registered alias")
    if not actor_id:
        raise ActionContractError("actor_id is required")
    if actor_id not in visible:
        raise ActionContractError("actor_id must belong to visible_character_ids")
    if action_code not in ACTION_CATALOG:
        raise ActionContractError("action_code must use the approved catalog enum")
    if not target:
        raise ActionContractError("target is required")
    if not resolved_start:
        raise ActionContractError("start_state is required")
    if not resolved_end:
        raise ActionContractError("end_state is required")

    canonical = {
        "catalog_version": ACTION_CATALOG_VERSION,
        "actor_id": actor_id,
        "action_code": action_code,
        "target": target,
        "start_state": resolved_start,
        "end_state": resolved_end,
    }
    spec_sha256 = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    compiled_sentence = _compile_h3_sentence(**{
        key: canonical[key]
        for key in ("actor_id", "action_code", "target", "start_state", "end_state")
    })
    provided_hash = _text(spec.get("spec_sha256"))
    if provided_hash and provided_hash != spec_sha256:
        raise ActionContractError("spec_sha256 does not match canonical action fields")
    provided_sentence = _text(spec.get("h3_action_en"))
    if provided_sentence and provided_sentence != _text(compiled_sentence):
        raise ActionContractError("h3_action_en does not match canonical action fields")
    return {
        **canonical,
        "h3_action_en": compiled_sentence,
        "spec_sha256": spec_sha256,
    }


def derived_action_components(compiled: Mapping[str, Any]) -> dict[str, str]:
    """Return the legacy/UI mirror; callers must not treat it as authoritative."""
    code = _text(compiled.get("action_code"))
    if code not in ACTION_CATALOG:
        raise ActionContractError("compiled action has an unknown action_code")
    return {
        "sub": _text(compiled.get("actor_id")),
        "code": code,
        "action_code": code,
        "verb": str(ACTION_CATALOG[code]["h3_verb"]),
        "obj": _text(compiled.get("target")),
        "res": _text(compiled.get("end_state")),
    }


def compile_panel_action(panel: Mapping[str, Any], *, allow_legacy: bool = True) -> dict[str, Any]:
    """Compile a panel action and reject every duplicated-field disagreement."""
    if not isinstance(panel, Mapping):
        raise ActionContractError("panel must be an object")
    visible = panel.get("character_ids")
    if not isinstance(visible, Sequence) or isinstance(visible, (str, bytes)):
        visible = []
    raw_spec = panel.get("action_spec")
    components = panel.get("action_components")
    top_code = _text(panel.get("action_code"))

    if isinstance(raw_spec, Mapping):
        compiled = compile_action_spec(raw_spec, visible_character_ids=visible)
    elif isinstance(components, Mapping):
        compiled = compile_action_spec(
            components,
            visible_character_ids=visible,
            start_state=panel.get("first_state"),
            end_state=panel.get("final_state"),
            allow_legacy=allow_legacy,
        )
    elif top_code:
        raise ActionContractError("action_code requires canonical action_spec")
    else:
        raise ActionContractError("canonical action_spec is missing")

    if top_code and top_code != compiled["action_code"]:
        raise ActionContractError("panel.action_code disagrees with action_spec")
    panel_start = _text(panel.get("first_state"))
    panel_end = _text(panel.get("final_state"))
    if panel_start and panel_start != compiled["start_state"]:
        raise ActionContractError("panel.first_state disagrees with action_spec")
    if panel_end and panel_end != compiled["end_state"]:
        raise ActionContractError("panel.final_state disagrees with action_spec")
    if isinstance(components, Mapping) and isinstance(raw_spec, Mapping):
        component_code = _consistent(
            "action_components.action_code",
            components.get("action_code"), components.get("code"),
        )
        expected = derived_action_components(compiled)
        checks = {
            "sub": compiled["actor_id"],
            "obj": compiled["target"],
            "res": compiled["end_state"],
        }
        if component_code and component_code != compiled["action_code"]:
            raise ActionContractError("action_components.action_code disagrees with action_spec")
        for key, value in checks.items():
            present = _text(components.get(key))
            if present and present != value:
                raise ActionContractError(f"action_components.{key} disagrees with action_spec")
        present_verb = _text(components.get("verb"))
        if present_verb and present_verb != expected["verb"]:
            raise ActionContractError("action_components.verb disagrees with action_spec")
    return compiled
