from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.common import BENCHMARK_ROOT, ensure_benchmark_dirs, iter_suites, load_manifest, write_json


def prepare_suites(*, suite: str, benchmark_root: Path = BENCHMARK_ROOT) -> dict[str, Any]:
    ensure_benchmark_dirs(benchmark_root)
    results: dict[str, Any] = {"status": "success", "suites": {}}
    for name in iter_suites(suite):
        manifest = load_manifest(benchmark_root, name)
        suite_result = _prepare_suite(benchmark_root, manifest)
        results["suites"][name] = suite_result
        if suite_result["status"] != "success":
            results["status"] = "partial"
    return results


def _prepare_suite(benchmark_root: Path, manifest: Any) -> dict[str, Any]:
    vendor_dir = benchmark_root / manifest.vendor_dir
    data_dir = benchmark_root / manifest.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    steps: list[dict[str, Any]] = []

    steps.append(_ensure_repo(manifest.upstream_repo, vendor_dir))

    for download in manifest.downloads:
        if not download.get("default", False):
            continue
        target = data_dir / download["relative_path"]
        steps.append(_download_file(download["url"], target))

    if manifest.suite == "locomo":
        source = vendor_dir / "data" / "locomo10.json"
        target = data_dir / "locomo10.json"
        if source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            steps.append({"step": "copy_locomo_dataset", "status": "success", "path": str(target)})
        else:
            steps.append(
                {
                    "step": "copy_locomo_dataset",
                    "status": "missing",
                    "reason": f"Expected vendor dataset at {source}",
                }
            )

    if manifest.manual_files:
        steps.append(
            {
                "step": "manual_files",
                "status": "pending",
                "files": manifest.manual_files,
            }
        )

    status = "success"
    if any(step["status"] in {"failed", "missing", "pending"} for step in steps):
        status = "partial"
    return {"status": status, "steps": steps}


def _ensure_repo(repo_url: str, target_dir: Path) -> dict[str, Any]:
    if target_dir.exists() and (target_dir / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(target_dir), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "step": "update_repo",
            "status": "success" if result.returncode == 0 else "failed",
            "path": str(target_dir),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", repo_url, str(target_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "step": "clone_repo",
        "status": "success" if result.returncode == 0 else "failed",
        "path": str(target_dir),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _download_file(url: str, target: Path) -> dict[str, Any]:
    if target.exists():
        return {"step": "download_file", "status": "success", "path": str(target), "cached": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:  # pragma: no cover
        return {"step": "download_file", "status": "failed", "path": str(target), "reason": str(exc)}
    return {"step": "download_file", "status": "success", "path": str(target), "cached": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare external benchmark repos and datasets.")
    parser.add_argument("--suite", choices=["longmemeval", "locomo", "membench", "all"], default="all")
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()
    payload = prepare_suites(suite=args.suite)
    if args.output_path is not None:
        write_json(args.output_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
