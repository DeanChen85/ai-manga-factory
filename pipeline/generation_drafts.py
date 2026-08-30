"""Durable, non-production checkpoints for validated generation stages.

Only the validated stage-1 contract is persisted.  Provider responses,
reasoning blocks, prompts and credentials are deliberately outside this wire
format.  A checkpoint is bound to the creative brief, render settings,
protocol and model so stage 2 cannot resume against changed inputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from atomic_io import write_json_atomic
from runtime_config import projects_dir


SCHEMA_VERSION = "ai-manga.v3-stage1-checkpoint/v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_KEYS = {
    "api_key", "apikey", "api-key", "authorization", "x-api-key",
    "secret", "token", "access_token", "refresh_token", "password",
    "raw", "raw_response", "provider_response", "reasoning", "thinking",
    "chain_of_thought", "system_prompt", "user_prompt",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _safe_id(value: str, label: str) -> str:
    candidate = str(value or "").strip()
    if not _SAFE_ID.fullmatch(candidate):
        raise ValueError(f"{label} must match {_SAFE_ID.pattern}")
    return candidate


def _sensitive_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).strip().casefold()
            child = f"{path}.{key}"
            if name in _SENSITIVE_KEYS or name.endswith("_api_key"):
                found.append(child)
            found.extend(_sensitive_paths(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_sensitive_paths(item, f"{path}[{index}]"))
    return found


def _assert_checkpoint_safe(stage1: Mapping[str, Any]) -> None:
    if not isinstance(stage1, Mapping) or not stage1:
        raise ValueError("stage1 must be a non-empty validated mapping")
    unsafe = _sensitive_paths(stage1)
    if unsafe:
        raise ValueError(
            "stage1 contains fields forbidden in generation checkpoints: "
            + ", ".join(unsafe)
        )


def _binding(
    *, stage1_hash: str, creative_brief: Mapping[str, Any],
    settings: Mapping[str, Any], protocol: str, model: str,
) -> dict[str, str]:
    resolved_protocol = str(protocol or "").strip().lower()
    resolved_model = str(model or "").strip()
    if not resolved_protocol or not resolved_model:
        raise ValueError("protocol and model are required for a resumable checkpoint")
    return {
        "stage1_sha256": stage1_hash,
        "creative_brief_sha256": canonical_sha256(dict(creative_brief or {})),
        "settings_sha256": canonical_sha256(dict(settings or {})),
        "protocol": resolved_protocol,
        "model": resolved_model,
    }


def checkpoint_sha256(binding: Mapping[str, Any]) -> str:
    return canonical_sha256({
        "schema_version": SCHEMA_VERSION,
        "binding": dict(binding),
    })


def draft_directory(ep_id: str, draft_dir: str | Path | None = None) -> Path:
    safe_ep_id = _safe_id(ep_id, "ep_id")
    return (
        Path(draft_dir).resolve()
        if draft_dir is not None
        else (projects_dir() / safe_ep_id / "drafts").resolve()
    )


def checkpoint_path(
    ep_id: str, checkpoint_hash: str, draft_dir: str | Path | None = None,
) -> Path:
    digest = str(checkpoint_hash or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("checkpoint_hash must be a lowercase SHA-256 digest")
    return draft_directory(ep_id, draft_dir) / f"v3_stage1.{digest}.json"


def save_stage1_checkpoint(
    ep_id: str,
    stage1: Mapping[str, Any],
    *,
    creative_brief: Mapping[str, Any],
    settings: Mapping[str, Any],
    protocol: str,
    model: str,
    draft_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically save one validated Stage-1 result as an unregistered draft."""
    safe_ep_id = _safe_id(ep_id, "ep_id")
    _assert_checkpoint_safe(stage1)
    clean_stage1 = copy.deepcopy(dict(stage1))
    stage1_hash = canonical_sha256(clean_stage1)
    binding = _binding(
        stage1_hash=stage1_hash, creative_brief=creative_brief,
        settings=settings, protocol=protocol, model=model,
    )
    digest = checkpoint_sha256(binding)
    path = checkpoint_path(safe_ep_id, digest, draft_dir)
    now = _utc_now()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": digest,
        "ep_id": safe_ep_id,
        "lifecycle": {
            "stage1_status": "validated",
            "stage2_status": "pending",
            "registration_status": "unregistered",
            "approval_status": "not_approved",
            "stage2_attempt_count": 0,
        },
        "binding": binding,
        "validated_stage1": clean_stage1,
        "created_at": now,
        "updated_at": now,
    }
    # Same immutable binding is idempotent; never overwrite an existing audit
    # timestamp or a later stage-2 status merely because the UI double-clicked.
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _validate_payload(existing, expected_ep_id=safe_ep_id, expected_hash=digest)
        if existing.get("validated_stage1") != clean_stage1:
            raise RuntimeError("checkpoint hash collision or corrupted existing draft")
        return {**existing, "checkpoint_path": str(path)}
    write_json_atomic(path, payload)
    return {**payload, "checkpoint_path": str(path)}


def _validate_payload(
    payload: Mapping[str, Any], *, expected_ep_id: str, expected_hash: str,
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported generation checkpoint schema")
    if payload.get("ep_id") != expected_ep_id:
        raise RuntimeError("generation checkpoint episode mismatch")
    binding = payload.get("binding")
    stage1 = payload.get("validated_stage1")
    if not isinstance(binding, Mapping) or not isinstance(stage1, Mapping):
        raise RuntimeError("generation checkpoint is incomplete")
    _assert_checkpoint_safe(stage1)
    if canonical_sha256(dict(stage1)) != binding.get("stage1_sha256"):
        raise RuntimeError("generation checkpoint stage1 hash mismatch")
    if checkpoint_sha256(binding) != expected_hash or payload.get("checkpoint_sha256") != expected_hash:
        raise RuntimeError("generation checkpoint binding hash mismatch")
    lifecycle = payload.get("lifecycle") or {}
    if (
        lifecycle.get("registration_status") != "unregistered"
        or lifecycle.get("approval_status") != "not_approved"
    ):
        raise RuntimeError("generation checkpoint must remain unregistered and unapproved")


def load_stage1_checkpoint(
    ep_id: str,
    checkpoint_hash: str,
    *,
    creative_brief: Mapping[str, Any],
    settings: Mapping[str, Any],
    protocol: str,
    model: str,
    draft_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load only when every Stage-2 input still matches the saved binding."""
    safe_ep_id = _safe_id(ep_id, "ep_id")
    digest = str(checkpoint_hash or "").strip().lower()
    path = checkpoint_path(safe_ep_id, digest, draft_dir)
    if not path.is_file():
        raise FileNotFoundError(f"stage1 checkpoint is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("generation checkpoint root must be an object")
    _validate_payload(payload, expected_ep_id=safe_ep_id, expected_hash=digest)
    actual = dict(payload["binding"])
    expected = _binding(
        stage1_hash=str(actual["stage1_sha256"]), creative_brief=creative_brief,
        settings=settings, protocol=protocol, model=model,
    )
    mismatches = [key for key in expected if expected[key] != actual.get(key)]
    if mismatches:
        raise RuntimeError(
            "stage1 checkpoint is stale for current inputs: " + ", ".join(mismatches)
        )
    return {**payload, "checkpoint_path": str(path)}


def list_stage1_checkpoints(
    ep_id: str, *, draft_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List sanitized checkpoint metadata for a Web resume picker."""
    safe_ep_id = _safe_id(ep_id, "ep_id")
    directory = draft_directory(safe_ep_id, draft_dir)
    summaries: list[dict[str, Any]] = []
    for path in sorted(directory.glob("v3_stage1.*.json")) if directory.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            digest = str(payload.get("checkpoint_sha256") or "")
            _validate_payload(payload, expected_ep_id=safe_ep_id, expected_hash=digest)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        lifecycle = dict(payload.get("lifecycle") or {})
        binding = dict(payload.get("binding") or {})
        summaries.append({
            "checkpoint_sha256": digest,
            "checkpoint_path": str(path),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
            "stage1_status": lifecycle.get("stage1_status"),
            "stage2_status": lifecycle.get("stage2_status"),
            "stage2_attempt_count": int(lifecycle.get("stage2_attempt_count") or 0),
            "registration_status": lifecycle.get("registration_status"),
            "approval_status": lifecycle.get("approval_status"),
            "protocol": binding.get("protocol"),
            "model": binding.get("model"),
        })
    return sorted(summaries, key=lambda item: str(item.get("updated_at") or ""), reverse=True)


def match_stage1_checkpoint(
    ep_id: str,
    *,
    creative_brief: Mapping[str, Any],
    settings: Mapping[str, Any],
    protocol: str,
    model: str,
    draft_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return the latest safe resumable summary whose persisted binding exactly matches."""
    safe_ep_id = _safe_id(ep_id, "ep_id")
    expected_brief = canonical_sha256(dict(creative_brief or {}))
    expected_settings = canonical_sha256(dict(settings or {}))
    expected_protocol = str(protocol or "").strip().lower()
    expected_model = str(model or "").strip()
    if not expected_protocol or not expected_model:
        return {}
    for summary in list_stage1_checkpoints(safe_ep_id, draft_dir=draft_dir):
        if summary.get("stage1_status") != "validated":
            continue
        if summary.get("stage2_status") not in {"pending", "failed"}:
            continue
        try:
            path = Path(str(summary["checkpoint_path"]))
            payload = json.loads(path.read_text(encoding="utf-8"))
            digest = str(summary.get("checkpoint_sha256") or "")
            _validate_payload(payload, expected_ep_id=safe_ep_id, expected_hash=digest)
        except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError):
            continue
        binding = dict(payload.get("binding") or {})
        if (
            binding.get("creative_brief_sha256") == expected_brief
            and binding.get("settings_sha256") == expected_settings
            and str(binding.get("protocol") or "").casefold() == expected_protocol.casefold()
            and str(binding.get("model") or "") == expected_model
        ):
            return {**copy.deepcopy(summary), "ep_id": safe_ep_id}
    return {}


def record_stage2_status(
    ep_id: str,
    checkpoint_hash: str,
    *,
    status: str,
    error_code: str | None = None,
    draft_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Record a safe Stage-2 audit status without provider output or errors."""
    allowed = {"pending", "running", "failed", "completed"}
    if status not in allowed:
        raise ValueError(f"status must be one of {sorted(allowed)}")
    safe_ep_id = _safe_id(ep_id, "ep_id")
    digest = str(checkpoint_hash or "").strip().lower()
    path = checkpoint_path(safe_ep_id, digest, draft_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_payload(payload, expected_ep_id=safe_ep_id, expected_hash=digest)
    lifecycle = dict(payload.get("lifecycle") or {})
    previous = str(lifecycle.get("stage2_status") or "pending")
    if previous == "completed" and status != "completed":
        raise RuntimeError("completed stage2 checkpoint cannot be reopened")
    if status == "running" and previous != "running":
        lifecycle["stage2_attempt_count"] = int(lifecycle.get("stage2_attempt_count") or 0) + 1
    lifecycle["stage2_status"] = status
    if status == "failed":
        safe_code = str(error_code or "stage2_failed").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", safe_code):
            raise ValueError("error_code must be an opaque safe code, not provider text")
        lifecycle["last_stage2_error_code"] = safe_code
    elif status == "completed":
        lifecycle.pop("last_stage2_error_code", None)
        lifecycle["stage2_completed_at"] = _utc_now()
    payload["lifecycle"] = lifecycle
    payload["updated_at"] = _utc_now()
    write_json_atomic(path, payload)
    return {**payload, "checkpoint_path": str(path)}


__all__ = [
    "SCHEMA_VERSION", "canonical_sha256", "checkpoint_sha256",
    "draft_directory", "checkpoint_path", "save_stage1_checkpoint",
    "load_stage1_checkpoint", "list_stage1_checkpoints", "match_stage1_checkpoint",
    "record_stage2_status",
]
