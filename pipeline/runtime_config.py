"""Runtime paths and executable discovery for the AI manga pipeline.

The original project embedded one developer's F:/G:/C: paths in every module.
This module keeps the existing machine working while allowing a clean install to
override every external dependency through environment variables.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path


# Project-owned provider routing must be deterministic.  These values contain
# no credentials and are safe for a checked-in/local project .env to override
# stale Windows User environment values.  Secrets intentionally stay outside
# this set: an existing process/user MiniMax_API_KEY always wins.
PROJECT_ENV_OVERRIDE_NAMES = frozenset({
    "MiniMax_PROTOCOL",
    "MiniMax_BASE_URL",
    "MiniMax_MODEL",
})


def _unquote_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env(path: str | Path | None = None) -> tuple[str, ...]:
    """Load simple KEY=VALUE settings with a narrow, safe precedence rule.

    This intentionally avoids a launcher-shell handoff: a child PowerShell
    process cannot mutate its parent cmd.exe environment.  The project's three
    non-secret MiniMax routing values override stale process/User values;
    credentials and every other variable retain process precedence.  Values
    are never printed, and only conventional variable names are accepted.
    """
    env_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return ()
    loaded: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        value = _unquote_env_value(value)
        if name in PROJECT_ENV_OVERRIDE_NAMES and value:
            if os.environ.get(name) != value:
                os.environ[name] = value
                loaded.append(name)
            continue
        if name in os.environ:
            # Windows launchers may preserve literal wrapping quotes.  This
            # sanitizes the same configured value; it never replaces a user's
            # non-empty setting with the .env value.
            cleaned_existing = _unquote_env_value(os.environ[name])
            if cleaned_existing != os.environ[name]:
                os.environ[name] = cleaned_existing
            continue
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)
    return tuple(loaded)


load_project_env()


def _env(*names: str, default: str | Path | None = None) -> str | None:
    """Return the first non-empty configured value, including legacy aliases."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return str(default) if default is not None else None


def project_root() -> Path:
    return Path(_env("AI_MANGA_ROOT", "AI_FACTORY_ROOT", default=Path(__file__).resolve().parents[1])).resolve()


def comfyui_root() -> Path:
    configured = os.environ.get("COMFYUI_ROOT")
    if configured:
        return Path(configured).resolve()
    # A portable ComfyUI Python normally lives at
    # <bundle>/python/python.exe with the application at <bundle>/ComfyUI.
    # Checking main.py prevents an empty clone-local input directory from
    # being mistaken for an installed ComfyUI instance.
    try:
        portable = Path(sys.executable).resolve().parent.parent / "ComfyUI"
        if (portable / "main.py").is_file():
            return portable.resolve()
    except (OSError, RuntimeError):
        pass
    # Portable default for a clone that keeps ComfyUI out of this repository.
    # Existing installations should set COMFYUI_ROOT explicitly in local .env.
    return (project_root() / "external" / "ComfyUI").resolve()


def comfyui_server() -> str:
    return os.environ.get("COMFYUI_SERVER", "http://127.0.0.1:8188").rstrip("/")


def state_dir() -> Path:
    return Path(_env("AI_MANGA_STATE_DIR", "AI_FACTORY_STATE_DIR", default=project_root() / "state")).resolve()


def projects_dir() -> Path:
    return Path(_env("AI_MANGA_PROJECTS_DIR", "AI_FACTORY_PROJECTS_DIR", default=project_root() / "output" / "projects")).resolve()


def render_job_db() -> Path:
    return Path(_env("AI_MANGA_JOB_DB", "AI_FACTORY_JOB_DB", default=state_dir() / "render_jobs.sqlite3")).resolve()


def ffmpeg_executable() -> str:
    configured = _env("FFMPEG_EXE", "FFMPEG_PATH")
    if configured:
        return configured
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("ffmpeg was not found; set FFMPEG_EXE (or FFMPEG_PATH) or add ffmpeg to PATH")


def ffprobe_executable() -> str:
    configured = _env("FFPROBE_EXE", "FFPROBE_PATH")
    if configured:
        return configured
    found = shutil.which("ffprobe")
    if found:
        return found
    ffmpeg = Path(ffmpeg_executable())
    sibling = ffmpeg.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if sibling.exists():
        return str(sibling)
    raise FileNotFoundError("ffprobe was not found; set FFPROBE_EXE (or FFPROBE_PATH) or add ffprobe to PATH")
