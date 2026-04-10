from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.adapters.canonical import CanonicalSample, CanonicalSession
from benchmark.common import read_prompt
from cognitiveos.benchmarks.runner import BenchmarkChatProvider, BenchmarkEmbeddingProvider
from cognitiveos.config import AppSettings
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.models import AddPayloadType
from cognitiveos.service import CognitiveOSService


def build_service(*, runtime_dir: Path, mode: str) -> CognitiveOSService:
    settings = AppSettings.from_env(
        db_path=runtime_dir / "benchmark.db",
        memory_output_path=runtime_dir / "MEMORY.MD",
    )
    settings.search_async_access_logging = False
    settings.search_governance_interval_seconds = 3600
    settings.dream_event_threshold = 10_000
    settings.dream_age_min_event_count = 10_000
    settings.long_document_token_threshold = 2_000
    settings.max_main_document_chars = 12_000

    if mode == "provider":
        service = CognitiveOSService.from_settings(settings)
        service.initialize()
        return service

    service = CognitiveOSService(
        settings=settings,
        repository=SQLiteRepository(settings.db_path),
        embedding_provider=BenchmarkEmbeddingProvider(),
        chat_provider=BenchmarkChatProvider(),
    )
    service.initialize()
    return service


def ingest_sample(service: CognitiveOSService, sample: CanonicalSample) -> list[str]:
    node_ids: list[str] = []
    for session in sample.sessions:
        receipt = service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload=_session_content(sample, session, max_chars=service.settings.max_main_document_chars),
            tags=[
                "benchmark",
                f"suite:{sample.suite}",
                f"sample:{sample.sample_id}",
                f"session:{session.session_id}",
            ],
            name=f"{sample.suite}:{sample.sample_id}:{session.session_id}",
            durability="working",
            force=True,
        )
        if receipt.node_id:
            node_ids.append(receipt.node_id)
    service._wait_for_background_tasks()
    return node_ids


def answer_sample(
    *,
    service: CognitiveOSService,
    benchmark_root: Path,
    sample: CanonicalSample,
    mode: str,
    top_k: int,
    include_neighbors: int,
    include_evidence: bool,
) -> dict[str, Any]:
    results = service.search(
        query=sample.question,
        keyword=sample.question,
        top_k=top_k,
        include_neighbors=include_neighbors,
        include_evidence=include_evidence,
    )
    read_map = service.read_nodes([result.id for result in results], include_content=True)
    retrieved = [
        {
            "id": result.id,
            "name": result.name,
            "score": result.score,
            "content": read_map.get(result.id).content if result.id in read_map else None,
        }
        for result in results
    ]
    hypothesis = (
        _provider_answer(
            service=service,
            benchmark_root=benchmark_root,
            sample=sample,
            retrieved=retrieved,
        )
        if mode == "provider"
        else _smoke_answer(sample=sample, retrieved=retrieved)
    )
    return {
        "question_id": sample.question_id,
        "sample_id": sample.sample_id,
        "category": sample.category,
        "expected_answer": sample.answer,
        "hypothesis": hypothesis,
        "retrieved": retrieved,
    }


def _session_content(sample: CanonicalSample, session: CanonicalSession, *, max_chars: int) -> str:
    turn_lines = [f"[{turn.role}] {turn.content}" for turn in session.turns]
    timestamp = session.timestamp or "unknown"
    header = (
        f"Benchmark suite: {sample.suite}\n"
        f"Sample id: {sample.sample_id}\n"
        f"Question id: {sample.question_id}\n"
        f"Session id: {session.session_id}\n"
        f"Timestamp: {timestamp}\n"
        f"Category: {sample.category or 'unknown'}\n\n"
    )
    transcript = "\n".join(turn_lines)
    content = f"{header}Transcript:\n{transcript}"
    if len(content) <= max_chars:
        return content

    note = (
        "\n\n[Truncated for benchmark ingestion. The session exceeded the CognitiveOS node content limit; "
        "head and tail transcript slices were preserved for retrieval.]"
    )
    budget = max(max_chars - len(header) - len("Transcript:\n") - len(note), 200)
    head_budget = budget // 2
    tail_budget = budget - head_budget
    head = transcript[:head_budget].rstrip()
    tail = transcript[-tail_budget:].lstrip()
    if tail:
        truncated_transcript = f"{head}\n...\n{tail}"
    else:
        truncated_transcript = head
    content = f"{header}Transcript:\n{truncated_transcript}{note}"
    if len(content) <= max_chars:
        return content
    overflow = len(content) - max_chars
    if overflow > 0:
        truncated_transcript = truncated_transcript[:-overflow].rstrip()
        content = f"{header}Transcript:\n{truncated_transcript}{note}"
    return content[:max_chars].rstrip()


def _smoke_answer(*, sample: CanonicalSample, retrieved: list[dict[str, Any]]) -> str:
    if sample.abstention and not retrieved:
        return "NOT_ENOUGH_INFORMATION"
    if not retrieved:
        return "NOT_ENOUGH_INFORMATION"
    content = next((item.get("content") for item in retrieved if item.get("content")), "") or ""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return "NOT_ENOUGH_INFORMATION"
    return lines[-1][:240]


def _provider_answer(
    *,
    service: CognitiveOSService,
    benchmark_root: Path,
    sample: CanonicalSample,
    retrieved: list[dict[str, Any]],
) -> str:
    if service.chat_provider is None:
        raise RuntimeError("Provider mode requires a configured chat provider.")
    prompt_name = "abstention.md" if sample.abstention else "answer_generation.md"
    template = read_prompt(benchmark_root, prompt_name)
    context = "\n\n".join(
        f"Result {idx + 1}:\n{item.get('content') or '[no content]'}"
        for idx, item in enumerate(retrieved)
    ).strip()
    if not context:
        context = "[no retrieved context]"
    prompt = template.format(question=sample.question, retrieved_context=context)
    return service._complete_chat(
        system_prompt="You are a strict benchmark answer generator.",
        user_prompt=prompt,
    )
