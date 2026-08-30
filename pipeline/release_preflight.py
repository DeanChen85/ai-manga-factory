"""Offline GitHub-release preflight without exposing secret contents."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "README.md", "LICENSE", "NOTICE", "SECURITY.md", "CONTRIBUTING.md",
    "THIRD_PARTY_NOTICES.md", ".gitignore", ".env.example",
    "requirements.txt", "启动.bat", "docs/ARCHITECTURE.md",
    "docs/QUICKSTART.md", "docs/COMFYUI_H3_INSTALL.md", "docs/H3_PROMPT_AND_RENDER_PROFILES.md",
    ".github/workflows/ci.yml", "examples/creative_brief.json", "pipeline/comfy_preflight.py",
    "skills/minimax-h3-drama-director/SKILL.md",
    "skills/minimax-h3-drama-director/sources.lock.json",
)
REQUIRED_IGNORES = (
    ".env", "minimax api.txt", "models/", "output/projects/", "state/",
    "logs/", "custom_nodes/",
)
LOCAL_PATH_PATTERNS = (
    re.compile(r"F:\\new ai factory", re.IGNORECASE),
    re.compile(r"G:\\ComfyUI-aki-v3", re.IGNORECASE),
    re.compile(r"C:\\Users\\Dean", re.IGNORECASE),
)
SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:MiniMax_API_KEY|OPENAI_API_KEY|GITHUB_TOKEN)\s*=\s*([^\s#].+)$"
)


def _release_text_files(root: Path) -> Iterable[Path]:
    fixed = [
        root / "README.md", root / ".env.example", root / "启动.bat",
        root / "SECURITY.md", root / "CONTRIBUTING.md",
        root / "THIRD_PARTY_NOTICES.md",
    ]
    for path in fixed:
        if path.is_file():
            yield path
    for folder in (root / "pipeline", root / "docs", root / "skills"):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".txt"}:
                if ".backup" not in path.name:
                    yield path


def _git_state(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"initialized": False, "tracked_secret_risks": []}
    command = ["git", "-C", str(root), "ls-files", "--", ".env", "minimax api.txt", "*.key", "*.pem"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        tracked = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        tracked = ["<git-query-failed>"]
    return {"initialized": True, "tracked_secret_risks": tracked}


def run_preflight(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    failures: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            failures.append({"code": "missing_release_file", "path": relative})

    ignore_path = root / ".gitignore"
    ignore_text = ignore_path.read_text(encoding="utf-8") if ignore_path.is_file() else ""
    for entry in REQUIRED_IGNORES:
        if entry not in ignore_text:
            failures.append({"code": "missing_ignore_rule", "path": entry})

    scanned = 0
    for path in _release_text_files(root):
        scanned += 1
        relative = path.relative_to(root).as_posix()
        try:
            content = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            failures.append({"code": "unreadable_release_text", "path": relative})
            continue
        if any(pattern.search(content) for pattern in LOCAL_PATH_PATTERNS):
            failures.append({"code": "developer_path", "path": relative})
        if path.name != ".env.example" and SECRET_ASSIGNMENT.search(content):
            failures.append({"code": "credential_assignment", "path": relative})

    lock_path = root / "skills" / "minimax-h3-drama-director" / "sources.lock.json"
    if lock_path.is_file():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            sources = lock.get("sources") if isinstance(lock, dict) else None
            if not isinstance(sources, list) or len(sources) < 4:
                failures.append({"code": "source_lock_incomplete", "path": str(lock_path.relative_to(root))})
            else:
                for source in sources:
                    commit = str(source.get("commit") or "") if isinstance(source, dict) else ""
                    if commit and not re.fullmatch(r"[0-9a-f]{40}", commit):
                        failures.append({"code": "source_lock_unpinned", "path": str(source.get("name") or "source")})
        except (OSError, json.JSONDecodeError):
            failures.append({"code": "source_lock_invalid", "path": str(lock_path.relative_to(root))})

    git = _git_state(root)
    if git["tracked_secret_risks"]:
        failures.extend(
            {"code": "tracked_secret_risk", "path": path}
            for path in git["tracked_secret_risks"]
        )
    if not git["initialized"]:
        failures.append({"code": "git_not_initialized", "path": ".git"})
    for local_name in (".env", "minimax api.txt"):
        if (root / local_name).exists():
            warnings.append({"code": "ignored_local_secret_file_present", "path": local_name})

    return {
        "schema": "ai-manga-release-preflight/v1",
        "passed": not failures,
        "root": str(root),
        "scanned_text_files": scanned,
        "failures": failures,
        "warnings": warnings,
        "git": git,
    }


def main() -> int:
    result = run_preflight()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
