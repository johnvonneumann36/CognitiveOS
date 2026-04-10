from __future__ import annotations

import json
import os
import string
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
SRC_ROOT = REPO_ROOT / "src"


def ensure_repo_imports() -> None:
    src = str(SRC_ROOT)
    repo = str(REPO_ROOT)
    if src not in sys.path:
        sys.path.insert(0, src)
    if repo not in sys.path:
        sys.path.insert(0, repo)


ensure_repo_imports()


@dataclass(slots=True)
class BenchmarkManifest:
    suite: str
    display_name: str
    upstream_repo: str
    vendor_dir: str
    data_dir: str
    defaults: dict[str, Any]
    downloads: list[dict[str, Any]]
    manual_files: list[dict[str, Any]]
    official_eval: dict[str, Any]
    notes: list[str]


def load_yaml_like(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(root: Path, suite: str) -> BenchmarkManifest:
    path = root / "manifests" / f"{suite}.yaml"
    payload = load_yaml_like(path)
    return BenchmarkManifest(
        suite=payload["suite"],
        display_name=payload["display_name"],
        upstream_repo=payload["upstream_repo"],
        vendor_dir=payload["vendor_dir"],
        data_dir=payload["data_dir"],
        defaults=payload.get("defaults", {}),
        downloads=payload.get("downloads", []),
        manual_files=payload.get("manual_files", []),
        official_eval=payload.get("official_eval", {}),
        notes=payload.get("notes", []),
    )


def iter_suites(selected: str) -> list[str]:
    if selected == "all":
        return ["longmemeval", "locomo", "membench"]
    return [selected]


def benchmark_dirs(root: Path) -> dict[str, Path]:
    return {
        "root": root,
        "data": root / "data",
        "vendor": root / "vendor",
        "results": root / "results",
        "config": root / "config",
        "prompts": root / "prompts",
        "notes": root / "notes",
        "manifests": root / "manifests",
    }


def ensure_benchmark_dirs(root: Path) -> None:
    for path in benchmark_dirs(root).values():
        path.mkdir(parents=True, exist_ok=True)


def utc_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def read_prompt(root: Path, name: str) -> str:
    return (root / "prompts" / name).read_text(encoding="utf-8")


def normalize_text(value: str) -> str:
    lowered = value.strip().lower()
    table = str.maketrans("", "", string.punctuation)
    return " ".join(lowered.translate(table).split())


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_text(prediction) == normalize_text(answer))


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0
    overlap = 0
    remaining = answer_tokens.copy()
    for token in pred_tokens:
        if token in remaining:
            overlap += 1
            remaining.remove(token)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )
