from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmark.adapters.canonical import CanonicalSample, CanonicalSession, CanonicalTurn
from benchmark.common import exact_match, run_command, token_f1, write_json


def resolve_dataset_path(data_dir: Path, dataset_split: str) -> Path:
    mapping = {
        "oracle": data_dir / "longmemeval_oracle.json",
        "s": data_dir / "longmemeval_s_cleaned.json",
        "m": data_dir / "longmemeval_m_cleaned.json",
    }
    return mapping[dataset_split]


def load_samples(
    dataset_path: Path,
    *,
    dataset_split: str | None = None,
    limit: int | None = None,
) -> list[CanonicalSample]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    items = payload[:limit] if limit is not None else payload
    samples: list[CanonicalSample] = []
    for index, item in enumerate(items):
        sessions: list[CanonicalSession] = []
        session_ids = item.get("haystack_session_ids", [])
        session_dates = item.get("haystack_dates", [])
        session_payloads = item.get("haystack_sessions", [])
        for session_index, turns in enumerate(session_payloads):
            session_id = str(session_ids[session_index]) if session_index < len(session_ids) else f"s{session_index + 1}"
            timestamp = str(session_dates[session_index]) if session_index < len(session_dates) else None
            canonical_turns = [
                CanonicalTurn(
                    role=str(turn.get("role", "unknown")),
                    content=str(turn.get("content", "")),
                )
                for turn in turns
                if isinstance(turn, dict)
            ]
            sessions.append(CanonicalSession(session_id=session_id, timestamp=timestamp, turns=canonical_turns))
        question_id = str(item.get("question_id", f"longmemeval-{index}"))
        question_type = str(item.get("question_type", "unknown"))
        samples.append(
            CanonicalSample(
                suite="longmemeval",
                sample_id=question_id,
                question_id=question_id,
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                category=question_type,
                abstention=question_id.endswith("_abs") or question_type == "abstention",
                sessions=sessions,
                metadata={
                    "question_date": item.get("question_date"),
                    "answer_session_ids": item.get("answer_session_ids", []),
                },
            )
        )
    return samples


def local_metrics(samples: list[CanonicalSample], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, float]]] = {}
    exact_scores: list[float] = []
    f1_scores: list[float] = []
    for sample, prediction in zip(samples, predictions, strict=False):
        exact = exact_match(prediction["hypothesis"], sample.answer)
        f1 = token_f1(prediction["hypothesis"], sample.answer)
        exact_scores.append(exact)
        f1_scores.append(f1)
        by_type.setdefault(sample.category or "unknown", []).append({"exact_match": exact, "token_f1": f1})
    return {
        "metric_family": "local_qa",
        "sample_count": len(predictions),
        "exact_match": round(sum(exact_scores) / len(exact_scores), 4) if exact_scores else 0.0,
        "token_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
        "by_question_type": {
            key: {
                "count": len(values),
                "exact_match": round(sum(item["exact_match"] for item in values) / len(values), 4),
                "token_f1": round(sum(item["token_f1"] for item in values) / len(values), 4),
            }
            for key, values in by_type.items()
        },
    }


def run_official_eval(
    *,
    vendor_dir: Path,
    dataset_path: Path,
    predictions_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    eval_script = vendor_dir / "src" / "evaluation" / "evaluate_qa.py"
    if not eval_script.exists():
        return {"status": "skipped", "reason": "Official evaluation script is not available."}
    if not os.getenv("OPENAI_API_KEY"):
        return {"status": "skipped", "reason": "OPENAI_API_KEY is required for official LongMemEval evaluation."}
    result = run_command(
        [sys.executable, str(eval_script), "gpt-4o", str(predictions_path), str(dataset_path)],
        cwd=eval_script.parent,
    )
    write_json(
        output_dir / "official_eval.log.json",
        {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
    )
    if result.returncode != 0:
        return {"status": "failed", "reason": "Official evaluator exited with a non-zero code."}
    return {"status": "success", "reason": "Official LongMemEval evaluator completed."}
