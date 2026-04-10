from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchmark.adapters.canonical import CanonicalSample, CanonicalSession, CanonicalTurn
from benchmark.common import exact_match, token_f1


def resolve_dataset_path(data_dir: Path, _dataset_split: str) -> Path:
    return data_dir / "locomo10.json"


def load_samples(
    dataset_path: Path,
    *,
    dataset_split: str | None = None,
    limit: int | None = None,
) -> list[CanonicalSample]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples: list[CanonicalSample] = []
    for conversation_index, item in enumerate(payload):
        qa_items = item.get("qa", [])
        conversation = item.get("conversation", {})
        sessions = _conversation_sessions(conversation)
        for qa_index, qa in enumerate(qa_items):
            if limit is not None and len(samples) >= limit:
                return samples
            sample_id = f"{item.get('sample_id', f'locomo-{conversation_index}')}-qa-{qa_index}"
            samples.append(
                CanonicalSample(
                    suite="locomo",
                    sample_id=sample_id,
                    question_id=sample_id,
                    question=str(qa.get("question", "")),
                    answer=str(qa.get("answer", "")),
                    category=str(qa.get("category", "qa")),
                    abstention=False,
                    sessions=sessions,
                    metadata={"evidence": qa.get("evidence", [])},
                )
            )
    return samples


def local_metrics(samples: list[CanonicalSample], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    exact_scores = [exact_match(pred["hypothesis"], sample.answer) for sample, pred in zip(samples, predictions, strict=False)]
    f1_scores = [token_f1(pred["hypothesis"], sample.answer) for sample, pred in zip(samples, predictions, strict=False)]
    return {
        "metric_family": "local_qa",
        "sample_count": len(predictions),
        "exact_match": round(sum(exact_scores) / len(exact_scores), 4) if exact_scores else 0.0,
        "token_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0,
    }


def run_official_eval(**_: Any) -> dict[str, Any]:
    return {"status": "skipped", "reason": "LoCoMo v1 uses local QA metrics only."}


def _conversation_sessions(conversation: Any) -> list[CanonicalSession]:
    sessions: list[CanonicalSession] = []
    if isinstance(conversation, dict):
        session_names = sorted(
            [key for key in conversation if re.fullmatch(r"session_\d+", key)],
            key=lambda value: int(value.split("_")[1]),
        )
        for name in session_names:
            turns = conversation.get(name, [])
            timestamp = conversation.get(f"{name}_date_time")
            sessions.append(
                CanonicalSession(
                    session_id=name,
                    timestamp=str(timestamp) if timestamp is not None else None,
                    turns=[
                        CanonicalTurn(
                            role=str(turn.get("speaker", turn.get("role", "unknown"))),
                            content=str(turn.get("text", turn.get("content", ""))),
                        )
                        for turn in turns
                        if isinstance(turn, dict)
                    ],
                )
            )
    return sessions
