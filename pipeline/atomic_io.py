"""Small durable atomic-write primitives shared by pipeline processes.

Temporary files are unique to one write attempt.  This matters on Windows,
where Streamlit and a worker can otherwise contend for the same fixed
``episode.json.tmp`` path and one process may replace or delete the other's
staging file.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping


def _unique_sibling(path: Path) -> Path:
    return path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )


def _unlink_own_temp(
    path: Path, *, attempts: int, initial_backoff: float,
    sleep_func: Callable[[float], None],
) -> None:
    """Delete exactly this writer's temporary file, never sibling temp files."""
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            sleep_func(min(initial_backoff * (2 ** attempt), 0.25))


def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    replace_attempts: int = 8,
    initial_backoff: float = 0.025,
    sleep_func: Callable[[float], None] = time.sleep,
) -> None:
    """Durably write JSON and atomically replace ``path`` with Windows retries.

    Each caller owns one unique same-directory temp file.  Data is flushed and
    fsynced before ``os.replace``.  A transient Windows ``PermissionError`` is
    retried with bounded exponential backoff; other errors are surfaced
    immediately.  Cleanup names only the caller's own temp path.
    """
    destination = Path(path)
    if replace_attempts < 1:
        raise ValueError("replace_attempts must be at least 1")
    if initial_backoff < 0:
        raise ValueError("initial_backoff cannot be negative")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _unique_sibling(destination)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(replace_attempts):
            try:
                os.replace(temporary, destination)
                return
            except PermissionError:
                if attempt + 1 >= replace_attempts:
                    raise
                sleep_func(min(initial_backoff * (2 ** attempt), 0.25))
    finally:
        _unlink_own_temp(
            temporary, attempts=replace_attempts,
            initial_backoff=initial_backoff, sleep_func=sleep_func,
        )


__all__ = ["write_json_atomic"]
