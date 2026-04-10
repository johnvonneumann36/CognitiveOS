from __future__ import annotations

import json

from benchmark.adapters.canonical import CanonicalSample, CanonicalSession, CanonicalTurn
from benchmark.adapters.cognitiveos_runtime import _session_content
from benchmark.adapters.longmemeval import load_samples as load_longmemeval_samples
from benchmark.adapters.locomo import load_samples as load_locomo_samples
from benchmark.adapters.membench import load_samples as load_membench_samples


def test_longmemeval_adapter_normalizes_sessions(tmp_path) -> None:
    dataset = [
        {
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "What drink does the user prefer?",
            "answer": "Tea",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2026-01-01"],
            "haystack_sessions": [[{"role": "user", "content": "I prefer tea."}]],
        }
    ]
    path = tmp_path / "longmemeval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    samples = load_longmemeval_samples(path)

    assert len(samples) == 1
    assert samples[0].sessions[0].session_id == "s1"
    assert samples[0].sessions[0].turns[0].content == "I prefer tea."


def test_locomo_adapter_expands_each_qa_item(tmp_path) -> None:
    dataset = [
        {
            "sample_id": "locomo1",
            "conversation": {
                "session_1": [{"speaker": "alice", "text": "I moved to Shanghai."}],
                "session_1_date_time": "2026-01-02",
            },
            "qa": [{"question": "Where did Alice move?", "answer": "Shanghai", "category": "fact"}],
        }
    ]
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    samples = load_locomo_samples(path)

    assert len(samples) == 1
    assert samples[0].question == "Where did Alice move?"
    assert samples[0].sessions[0].turns[0].role == "alice"


def test_membench_adapter_handles_generic_history_shape(tmp_path) -> None:
    dataset = {
        "items": [
            {
                "sample_id": "mb1",
                "question": "What project is under review?",
                "answer": "CognitiveOS",
                "memory_type": "factual",
                "perspective": "participation",
                "history": ["The team is reviewing CognitiveOS this week."],
            }
        ]
    }
    path = tmp_path / "membench.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    samples = load_membench_samples(path)

    assert len(samples) == 1
    assert samples[0].category == "factual:participation"
    assert samples[0].sessions[0].turns[0].content.startswith("The team is reviewing")


def test_membench_adapter_handles_qa_and_message_list_shape(tmp_path) -> None:
    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir(parents=True)
    payload = {
        "Single-hop": {
            "roles": [
                {
                    "tid": 0,
                    "message_list": [
                        [
                            {
                                "sid": 0,
                                "user_message": "My brother works as a pilot.",
                                "assistant_message": "That sounds exciting.",
                                "time": "'2024-10-01 08:00' Tuesday",
                            }
                        ]
                    ],
                    "QA": {
                        "qid": 0,
                        "question": "What does my brother do for a living?",
                        "answer": "Pilot",
                    },
                }
            ]
        }
    }
    path = dataset_dir / "FirstAgentDataLowLevel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    samples = load_membench_samples(path, dataset_split="factual")

    assert len(samples) == 1
    assert samples[0].question == "What does my brother do for a living?"
    assert samples[0].answer == "Pilot"
    assert samples[0].category == "factual:participation"
    assert samples[0].sessions[0].turns[0].content == "My brother works as a pilot."


def test_session_content_is_truncated_to_runtime_limit() -> None:
    sample = CanonicalSample(
        suite="longmemeval",
        sample_id="sample-1",
        question_id="q-1",
        question="What happened?",
        answer="A long discussion happened.",
        category="single-session-user",
        sessions=[
            CanonicalSession(
                session_id="session-1",
                timestamp="2026-01-01",
                turns=[
                    CanonicalTurn(role="user", content="A" * 9000),
                    CanonicalTurn(role="assistant", content="B" * 9000),
                ],
            )
        ],
    )

    content = _session_content(sample, sample.sessions[0], max_chars=12_000)

    assert len(content) <= 12_000
    assert "[Truncated for benchmark ingestion." in content
