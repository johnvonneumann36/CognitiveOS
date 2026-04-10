from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.adapters import longmemeval, locomo, membench
from benchmark.adapters.canonical import CanonicalSample
from benchmark.adapters.cognitiveos_runtime import answer_sample, build_service, ingest_sample
from benchmark.common import (
    BENCHMARK_ROOT,
    ensure_benchmark_dirs,
    load_manifest,
    load_yaml_like,
    utc_run_id,
    write_json,
    write_jsonl,
)


ADAPTERS = {
    "longmemeval": longmemeval,
    "locomo": locomo,
    "membench": membench,
}


def run_suite(
    *,
    suite: str,
    mode: str,
    dataset_split: str | None = None,
    limit: int | None = None,
    output_dir: Path | None = None,
    reuse_runtime: bool = False,
    benchmark_root: Path = BENCHMARK_ROOT,
    dataset_path: Path | None = None,
) -> dict[str, Any]:
    ensure_benchmark_dirs(benchmark_root)
    defaults = load_yaml_like(benchmark_root / "config" / "run.defaults.yaml")
    manifest = load_manifest(benchmark_root, suite)
    adapter = ADAPTERS[suite]
    split = dataset_split or manifest.defaults["dataset_split"]
    resolved_dataset_path = dataset_path or adapter.resolve_dataset_path(benchmark_root / manifest.data_dir, split)
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset is missing for {suite} ({split}): {resolved_dataset_path}. "
            f"Run `python benchmark/scripts/prepare.py --suite {suite}` first."
        )

    suite_output_root = output_dir or benchmark_root / "results" / suite / utc_run_id(suite)
    suite_output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = suite_output_root / "runtime"
    if runtime_root.exists() and not reuse_runtime:
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    samples: list[CanonicalSample] = adapter.load_samples(
        resolved_dataset_path,
        dataset_split=split,
        limit=limit,
    )
    predictions: list[dict[str, Any]] = []
    runtime_records: list[dict[str, Any]] = []

    shared_runtime = runtime_root / "shared" if reuse_runtime else None
    if shared_runtime is not None:
        shared_runtime.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(samples):
        sample_runtime_dir = shared_runtime or runtime_root / f"sample_{index + 1:04d}"
        if not reuse_runtime and sample_runtime_dir.exists():
            shutil.rmtree(sample_runtime_dir)
        sample_runtime_dir.mkdir(parents=True, exist_ok=True)

        service = build_service(runtime_dir=sample_runtime_dir, mode=mode)
        ingested_node_ids = ingest_sample(service, sample)
        prediction = answer_sample(
            service=service,
            benchmark_root=benchmark_root,
            sample=sample,
            mode=mode,
            top_k=defaults["search"]["top_k"],
            include_neighbors=defaults["search"]["include_neighbors"],
            include_evidence=defaults["search"]["include_evidence"],
        )
        predictions.append(
            {
                "question_id": prediction["question_id"],
                "sample_id": prediction["sample_id"],
                "hypothesis": prediction["hypothesis"],
            }
        )
        runtime_records.append(
            {
                "question_id": prediction["question_id"],
                "sample_id": prediction["sample_id"],
                "runtime_dir": str(sample_runtime_dir),
                "ingested_node_ids": ingested_node_ids,
                "retrieved": prediction["retrieved"],
            }
        )

    write_json(suite_output_root / "run_config.json", {
        "suite": suite,
        "mode": mode,
        "dataset_split": split,
        "dataset_path": str(resolved_dataset_path),
        "limit": limit,
        "reuse_runtime": reuse_runtime,
    })
    write_jsonl(suite_output_root / "predictions.jsonl", predictions)

    metrics = {
        "status": "success",
        "mode": mode,
        "suite": suite,
        "dataset_split": split,
        "sample_count": len(samples),
        "local_metrics": adapter.local_metrics(
            samples,
            [{"hypothesis": item["hypothesis"]} for item in predictions],
        ),
        "official_eval": adapter.run_official_eval(
            vendor_dir=benchmark_root / manifest.vendor_dir,
            dataset_path=resolved_dataset_path,
            predictions_path=suite_output_root / "predictions.jsonl",
            output_dir=suite_output_root,
        ),
    }
    write_json(suite_output_root / "metrics.json", metrics)
    write_json(suite_output_root / "runtime_metadata.json", {"samples": runtime_records})
    return {
        "status": "success",
        "suite": suite,
        "mode": mode,
        "output_dir": str(suite_output_root),
        "metrics_path": str(suite_output_root / "metrics.json"),
        "sample_count": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run external long-term memory benchmarks against CognitiveOS.")
    parser.add_argument("--suite", choices=["longmemeval", "locomo", "membench"], required=True)
    parser.add_argument("--mode", choices=["smoke", "provider"], required=True)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reuse-runtime", action="store_true")
    args = parser.parse_args()
    payload = run_suite(
        suite=args.suite,
        mode=args.mode,
        dataset_split=args.dataset_split,
        limit=args.limit,
        output_dir=args.output_dir,
        reuse_runtime=args.reuse_runtime,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
