from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.adapters.longmemeval import resolve_dataset_path as longmemeval_path
from benchmark.adapters.locomo import resolve_dataset_path as locomo_path
from benchmark.adapters.membench import resolve_dataset_path as membench_path
from benchmark.common import BENCHMARK_ROOT, ensure_benchmark_dirs, iter_suites, load_manifest, write_json


def validate_suites(*, suite: str, benchmark_root: Path = BENCHMARK_ROOT) -> dict[str, Any]:
    ensure_benchmark_dirs(benchmark_root)
    payload: dict[str, Any] = {"status": "success", "suites": {}}
    for name in iter_suites(suite):
        manifest = load_manifest(benchmark_root, name)
        result = _validate_suite(benchmark_root, manifest)
        payload["suites"][name] = result
        if result["status"] != "ready":
            payload["status"] = "partial"
    return payload


def _validate_suite(benchmark_root: Path, manifest: Any) -> dict[str, Any]:
    vendor_dir = benchmark_root / manifest.vendor_dir
    data_dir = benchmark_root / manifest.data_dir
    messages: list[str] = []
    missing: list[str] = []

    if not vendor_dir.exists():
        missing.append(f"Missing vendor repo: {vendor_dir}")
    if manifest.suite == "longmemeval":
        for split in ("oracle", "s"):
            path = longmemeval_path(data_dir, split)
            if not path.exists():
                missing.append(f"Missing dataset split {split}: {path}")
    elif manifest.suite == "locomo":
        path = locomo_path(data_dir, "qa")
        if not path.exists():
            missing.append(f"Missing QA dataset: {path}")
    elif manifest.suite == "membench":
        required_files = [
            membench_path(data_dir, "factual", perspective="participation"),
            membench_path(data_dir, "factual", perspective="observation"),
            membench_path(data_dir, "reflective", perspective="participation"),
            membench_path(data_dir, "reflective", perspective="observation"),
        ]
        for path in required_files:
            if not path.exists():
                missing.append(f"Missing MemBench source file: {path}")
        messages.extend(
            f"Manual file required: {item['relative_path']} under {data_dir}" for item in manifest.manual_files
        )

    if manifest.suite == "membench" and missing:
        next_step = (
            f"Place the extracted MemBench bundle under {data_dir} so the official HighLevel and LowLevel JSON files exist, "
            f"and rerun `python benchmark/scripts/validate.py --suite membench`."
        )
    else:
        next_step = (
            f"python benchmark/scripts/run.py --suite {manifest.suite} --mode smoke"
            if not missing
            else f"python benchmark/scripts/prepare.py --suite {manifest.suite}"
        )

    return {
        "status": "ready" if not missing else "missing",
        "vendor_dir": str(vendor_dir),
        "data_dir": str(data_dir),
        "missing": missing,
        "messages": messages,
        "next_step": next_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark repos, data, and manifests.")
    parser.add_argument("--suite", choices=["longmemeval", "locomo", "membench", "all"], default="all")
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()
    payload = validate_suites(suite=args.suite)
    if args.output_path is not None:
        write_json(args.output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
