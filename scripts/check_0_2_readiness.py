from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

TARGET_VERSION = "0.2.0"
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
WINDOWS_USER_PATH_RE = re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def _tracked_files(root: Path) -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [root / line for line in output.splitlines() if line.strip()]


def _check_version(root: Path) -> tuple[bool, str]:
    pyproject = _read(root / "pyproject.toml")
    init = _read(root / "src" / "cognitiveos" / "__init__.py")
    readme = _read(root / "README.md")
    pyproject_ok = f'version = "{TARGET_VERSION}"' in pyproject
    init_ok = f'__version__ = "{TARGET_VERSION}"' in init
    readme_ok = f"v{TARGET_VERSION}" in readme
    return pyproject_ok and init_ok and readme_ok, "package, __version__, and README align"


def _check_benchmark_extra(root: Path) -> tuple[bool, str]:
    pyproject = _read(root / "pyproject.toml")
    return "ijson>=3.3" in pyproject, "benchmark dependency ijson declared"


def _check_release_workflows(root: Path) -> tuple[bool, str]:
    pypi = _read(root / ".github" / "workflows" / "publish-pypi.yml")
    gh_release = _read(root / ".github" / "workflows" / "publish-github-release.yml")
    pypi_ok = "skip-existing: true" in pypi and "urllib.request.urlopen" in pypi
    gh_ok = "retry()" in gh_release and "gh release upload" in gh_release
    return pypi_ok and gh_ok, "release workflows are idempotent and retry network calls"


def _check_bootstrap_leaks(root: Path) -> tuple[bool, str]:
    leaks: list[str] = []
    for path in _tracked_files(root):
        rel = path.relative_to(root).as_posix()
        if rel == "MEMORY.MD" or rel.startswith(".cognitiveos/"):
            leaks.append(rel)
            continue
        text = _read(path)
        if not text:
            continue
        if EMAIL_RE.search(text) or WINDOWS_USER_PATH_RE.search(text):
            leaks.append(rel)
    if leaks:
        return False, "tracked bootstrap/runtime leak candidates: " + ", ".join(leaks[:8])
    return True, "no tracked MEMORY.MD, .cognitiveos artifacts, emails, or local user paths"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    checks = {
        "version_alignment": _check_version(root),
        "benchmark_dependencies": _check_benchmark_extra(root),
        "release_workflows": _check_release_workflows(root),
        "bootstrap_leak_prevention": _check_bootstrap_leaks(root),
    }
    payload = {name: {"ok": ok, "detail": detail} for name, (ok, detail) in checks.items()}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(ok for ok, _detail in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
