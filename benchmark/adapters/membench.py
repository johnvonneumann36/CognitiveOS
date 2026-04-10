from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import ijson

from benchmark.adapters.canonical import CanonicalSample, CanonicalSession, CanonicalTurn
from benchmark.common import exact_match, token_f1


def resolve_dataset_path(data_dir: Path, dataset_split: str, perspective: str | None = None) -> Path:
    if perspective == "participation":
        if dataset_split == "factual":
            return data_dir / "data" / "FirstAgentDataLowLevel.json"
        if dataset_split == "reflective":
            return data_dir / "data" / "FirstAgentDataHighLevel.json"
    if perspective == "observation":
        if dataset_split == "factual":
            return data_dir / "data" / "ThirdAgentDataLowLevel.json"
        if dataset_split == "reflective":
            return data_dir / "data" / "ThirdAgentDataHighLevel.json"
    return data_dir / "data"


def load_samples(
    dataset_path: Path,
    *,
    dataset_split: str | None = None,
    limit: int | None = None,
) -> list[CanonicalSample]:
    raw_items = list(_iter_raw_items(dataset_path, dataset_split or "all", limit=limit))
    samples: list[CanonicalSample] = []
    for index, item in enumerate(raw_items):
        if limit is not None and len(samples) >= limit:
            break
        sessions = _extract_sessions(item["source"])
        sample_id = str(item.get("sample_id", item.get("id", f"membench-{index}")))
        samples.append(
            CanonicalSample(
                suite="membench",
                sample_id=sample_id,
                question_id=str(item.get("question_id", sample_id)),
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                category=_resolve_category(item),
                abstention=False,
                sessions=sessions,
                metadata={
                    "raw_keys": sorted(item.keys()),
                    "source_file": item["source_file"],
                    "memory_type": item["memory_type"],
                    "perspective": item["perspective"],
                },
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
    return {"status": "skipped", "reason": "MemBench v1 uses local subset metrics and manual data validation."}


def _iter_raw_items(dataset_path: Path, dataset_split: str, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    files = _dataset_files(dataset_path, dataset_split)
    yielded = 0
    for file_path, memory_type, perspective in files:
        for item in _iter_file_items(file_path):
            normalized = _normalize_item(item, file_path=file_path, memory_type=memory_type, perspective=perspective)
            if normalized is None:
                continue
            yield normalized
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def _dataset_files(dataset_path: Path, dataset_split: str) -> list[tuple[Path, str, str]]:
    if dataset_path.is_file():
        known = {
            "FirstAgentDataLowLevel.json": ("factual", "participation"),
            "ThirdAgentDataLowLevel.json": ("factual", "observation"),
            "FirstAgentDataHighLevel.json": ("reflective", "participation"),
            "ThirdAgentDataHighLevel.json": ("reflective", "observation"),
        }
        if dataset_path.name in known:
            memory_type, perspective = known[dataset_path.name]
            return [(dataset_path, memory_type, perspective)]
        return [(dataset_path, dataset_split, "unknown")]

    data_root = dataset_path if dataset_path.is_dir() else dataset_path.parent
    data_dir = data_root if data_root.name == "data" else data_root / "data"
    files: list[tuple[Path, str, str]] = []
    if dataset_split in {"factual", "all"}:
        files.extend(
            [
                (data_dir / "FirstAgentDataLowLevel.json", "factual", "participation"),
                (data_dir / "ThirdAgentDataLowLevel.json", "factual", "observation"),
            ]
        )
    if dataset_split in {"reflective", "all"}:
        files.extend(
            [
                (data_dir / "FirstAgentDataHighLevel.json", "reflective", "participation"),
                (data_dir / "ThirdAgentDataHighLevel.json", "reflective", "observation"),
            ]
        )
    return files


def _iter_file_items(file_path: Path) -> Iterable[dict[str, Any]]:
    if "LowLevel" in file_path.name:
        yield from _iter_low_level_items(file_path)
        return
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    yield from _flatten_nested_items(payload)


def _iter_low_level_items(file_path: Path) -> Iterable[dict[str, Any]]:
    top_level_keys = _top_level_keys(file_path)
    for key in top_level_keys:
        with file_path.open("rb") as handle:
            yield from ijson.items(handle, f"{key}.roles.item")


def _top_level_keys(file_path: Path) -> list[str]:
    keys: list[str] = []
    with file_path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "" and event == "map_key":
                keys.append(str(value))
    return keys


def _flatten_nested_items(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and ("QA" in item or "question" in item):
                yield item
            elif isinstance(item, (list, dict)):
                yield from _flatten_nested_items(item)
        return
    if isinstance(payload, dict):
        if "QA" in payload or "question" in payload:
            yield payload
            return
        for value in payload.values():
            if isinstance(value, (list, dict)):
                yield from _flatten_nested_items(value)


def _normalize_item(
    item: dict[str, Any],
    *,
    file_path: Path,
    memory_type: str,
    perspective: str,
) -> dict[str, Any] | None:
    qa = item.get("QA") if isinstance(item.get("QA"), dict) else item
    question = qa.get("question")
    answer = qa.get("answer")
    if not question or answer is None:
        return None
    resolved_memory_type = str(item.get("memory_type", memory_type))
    resolved_perspective = str(item.get("perspective", perspective))
    sample_suffix = item.get("tid", item.get("gid", qa.get("qid", 0)))
    source_label = file_path.stem
    return {
        "sample_id": f"{source_label}:{sample_suffix}",
        "question_id": f"{source_label}:{qa.get('qid', sample_suffix)}",
        "question": str(question),
        "answer": str(answer),
        "category": f"{resolved_memory_type}:{resolved_perspective}",
        "memory_type": resolved_memory_type,
        "perspective": resolved_perspective,
        "source_file": file_path.name,
        "source": item,
    }


def _extract_sessions(item: dict[str, Any]) -> list[CanonicalSession]:
    raw_sessions = item.get("message_list") or item.get("sessions") or item.get("history") or item.get("conversation") or item.get("dialogue") or []
    sessions: list[CanonicalSession] = []
    if isinstance(raw_sessions, list):
        for index, raw_session in enumerate(raw_sessions):
            timestamp = None
            if isinstance(raw_session, str):
                turns = [CanonicalTurn(role="narrator", content=raw_session)]
            elif isinstance(raw_session, dict):
                turn_items = raw_session.get("turns") or raw_session.get("messages") or raw_session.get("dialog") or []
                turns = [
                    CanonicalTurn(
                        role=str(turn.get("role", turn.get("speaker", "unknown"))),
                        content=str(turn.get("content", turn.get("text", ""))),
                    )
                    for turn in turn_items
                    if isinstance(turn, dict)
                ]
                if not turns and raw_session.get("content"):
                    turns = [CanonicalTurn(role="narrator", content=str(raw_session["content"]))]
                timestamp = raw_session.get("time")
            elif isinstance(raw_session, list):
                turns = []
                for turn in raw_session:
                    if not isinstance(turn, dict):
                        continue
                    timestamp = timestamp or turn.get("time")
                    if "user_message" in turn or "assistant_message" in turn:
                        if turn.get("user_message"):
                            turns.append(CanonicalTurn(role="user", content=str(turn["user_message"])))
                        if turn.get("assistant_message"):
                            turns.append(CanonicalTurn(role="assistant", content=str(turn["assistant_message"])))
                    elif "user" in turn or "assistant" in turn:
                        if turn.get("user"):
                            turns.append(CanonicalTurn(role="user", content=str(turn["user"])))
                        if turn.get("assistant"):
                            turns.append(CanonicalTurn(role="assistant", content=str(turn["assistant"])))
                    elif turn.get("message"):
                        turns.append(CanonicalTurn(role="narrator", content=str(turn["message"])))
                    elif turn.get("content") or turn.get("text"):
                        turns.append(
                            CanonicalTurn(
                                role=str(turn.get("role", turn.get("speaker", "unknown"))),
                                content=str(turn.get("content", turn.get("text", ""))),
                            )
                        )
            else:
                turns = []
            sessions.append(
                CanonicalSession(
                    session_id=f"session_{index + 1}",
                    timestamp=str(timestamp) if timestamp else None,
                    turns=turns,
                )
            )
    return sessions


def _resolve_category(item: dict[str, Any]) -> str:
    memory_type = str(item.get("memory_type", item.get("level", item.get("dimension", "unknown"))))
    perspective = str(item.get("perspective", item.get("scenario", "unknown")))
    return f"{memory_type}:{perspective}"
