from __future__ import annotations

import json
from pathlib import Path

from benchmark.common import BENCHMARK_ROOT
from benchmark.scripts.run import run_suite


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_smoke_longmemeval_generates_outputs(tmp_path: Path) -> None:
    dataset_path = _write_json(
        tmp_path / "longmemeval_oracle.json",
        [
            {
                "question_id": "q1",
                "question_type": "single-session-user",
                "question": "What fruit does the user like?",
                "answer": "Apple",
                "haystack_session_ids": ["s1"],
                "haystack_dates": ["2026-01-01"],
                "haystack_sessions": [[{"role": "user", "content": "I like Apple."}]],
            }
        ],
    )

    payload = run_suite(
        suite="longmemeval",
        mode="smoke",
        dataset_split="oracle",
        limit=1,
        output_dir=tmp_path / "results" / "longmemeval",
        benchmark_root=BENCHMARK_ROOT,
        dataset_path=dataset_path,
    )

    output_dir = Path(payload["output_dir"])
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "runtime_metadata.json").exists()


def test_smoke_locomo_generates_outputs(tmp_path: Path) -> None:
    dataset_path = _write_json(
        tmp_path / "locomo10.json",
        [
            {
                "sample_id": "l1",
                "conversation": {
                    "session_1": [{"speaker": "alice", "text": "The project is CognitiveOS."}],
                    "session_1_date_time": "2026-01-02",
                },
                "qa": [{"question": "What is the project?", "answer": "CognitiveOS", "category": "fact"}],
            }
        ],
    )

    payload = run_suite(
        suite="locomo",
        mode="smoke",
        dataset_split="qa",
        limit=1,
        output_dir=tmp_path / "results" / "locomo",
        benchmark_root=BENCHMARK_ROOT,
        dataset_path=dataset_path,
    )

    output_dir = Path(payload["output_dir"])
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "metrics.json").exists()


def test_smoke_membench_generates_outputs(tmp_path: Path) -> None:
    dataset_path = _write_json(
        tmp_path / "all.json",
        {
            "items": [
                {
                    "sample_id": "m1",
                    "question": "What system is being evaluated?",
                    "answer": "CognitiveOS",
                    "memory_type": "reflective",
                    "perspective": "observation",
                    "history": ["The benchmark evaluates CognitiveOS."],
                }
            ]
        },
    )

    payload = run_suite(
        suite="membench",
        mode="smoke",
        dataset_split="all",
        limit=1,
        output_dir=tmp_path / "results" / "membench",
        benchmark_root=BENCHMARK_ROOT,
        dataset_path=dataset_path,
    )

    output_dir = Path(payload["output_dir"])
    assert (output_dir / "predictions.jsonl").exists()
    assert (output_dir / "metrics.json").exists()
