from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

from cognitiveos.benchmarks.fixtures import DEFAULT_BENCHMARK_CORPUS, materialize_fixture_files
from cognitiveos.config import AppSettings
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.models import AddPayloadType
from cognitiveos.service import CognitiveOSService


class BenchmarkEmbeddingProvider:
    """Deterministic embedding provider for local benchmark reproducibility."""

    FEATURE_GROUPS = (
        ("graph", "memory", "retrieval", "search", "read"),
        ("host", "codex", "bootstrap", "mount", "mcp"),
        ("extractor", "file", "document", "chunk", "html", "markdown"),
        ("dream", "cluster", "compaction", "summary", "consolidation"),
        ("benchmark", "baseline", "performance", "latency", "quality"),
        ("link", "path", "hop", "bridge", "relation", "graph-aware"),
        ("cli", "command", "tool", "contract", "workflow"),
        ("add", "update", "write", "ingest", "ingestion"),
    )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        lowered = text.lower()
        tokens = re.findall(r"[a-z0-9_.-]+", lowered)
        vector: list[float] = []
        for group in self.FEATURE_GROUPS:
            score = 0.0
            for term in group:
                if " " in term:
                    score += float(lowered.count(term))
                else:
                    score += float(tokens.count(term))
            vector.append(score)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [round(value / norm, 6) for value in vector]


class BenchmarkChatProvider:
    """Deterministic summarizer for stable local benchmark output."""

    def summarize(self, content: str) -> str:
        stripped = " ".join(content.split())
        if len(stripped) <= 180:
            return stripped
        sentence_break = stripped.find(". ")
        if 0 < sentence_break <= 180:
            return stripped[: sentence_break + 1]
        return stripped[:177].rstrip() + "..."

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if '"tags"' in system_prompt:
            lowered = user_prompt.lower()
            tags: list[str] = []
            for candidate in (
                "bootstrap",
                "host",
                "memory",
                "retrieval",
                "graph",
                "benchmark",
                "extractor",
                "workflow",
            ):
                if candidate in lowered and candidate not in tags:
                    tags.append(candidate)
            return json.dumps({"tags": tags[:4] or ["document", "runtime"]})
        return self.summarize(user_prompt)


def run_benchmark_suite(
    *,
    iterations: int = 5,
    output_path: Path | None = None,
    runtime_dir: Path | None = None,
    provider_mode: str = "fake",
) -> dict[str, Any]:
    normalized_provider_mode = provider_mode.strip().lower()
    if normalized_provider_mode not in {"fake", "env", "none"}:
        raise ValueError("Unsupported provider mode. Use fake, env, or none.")
    bounded_iterations = max(1, iterations)

    managed_dir: TemporaryDirectory[str] | None = None
    try:
        if runtime_dir is None:
            managed_dir = TemporaryDirectory(prefix="cognitiveos-benchmark-")
            root_dir = Path(managed_dir.name)
        else:
            root_dir = runtime_dir
            root_dir.mkdir(parents=True, exist_ok=True)

        service = _build_service(root_dir=root_dir, provider_mode=normalized_provider_mode)
        fixture_paths = materialize_fixture_files(root_dir / "fixture_files")
        seed_result = _seed_corpus(service=service, fixture_paths=fixture_paths)
        quality = _run_quality_tasks(service=service, ids_by_key=seed_result["ids_by_key"])
        runtime = _run_runtime_benchmarks(
            service=service,
            ids_by_key=seed_result["ids_by_key"],
            fixture_paths=fixture_paths,
            iterations=bounded_iterations,
        )
        dream = _run_dream_benchmark(service=service)

        payload = {
            "suite_name": DEFAULT_BENCHMARK_CORPUS.name,
            "description": DEFAULT_BENCHMARK_CORPUS.description,
            "provider_mode": normalized_provider_mode,
            "runtime_dir": str(root_dir.resolve()),
            "settings_overrides": {
                "search_async_access_logging": service.settings.search_async_access_logging,
                "search_governance_interval_seconds": (
                    service.settings.search_governance_interval_seconds
                ),
                "dream_event_threshold": service.settings.dream_event_threshold,
                "dream_age_min_event_count": service.settings.dream_age_min_event_count,
                "long_document_token_threshold": service.settings.long_document_token_threshold,
                "max_main_document_chars": service.settings.max_main_document_chars,
            },
            "corpus": {
                "content_nodes": len(DEFAULT_BENCHMARK_CORPUS.content_nodes),
                "file_nodes": len(DEFAULT_BENCHMARK_CORPUS.file_nodes),
                "links": len(DEFAULT_BENCHMARK_CORPUS.links),
                "materialized_files": {
                    key: str(path.resolve()) for key, path in sorted(fixture_paths.items())
                },
            },
            "ingestion": seed_result["ingestion"],
            "quality": quality,
            "runtime": runtime,
            "dream": dream,
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return payload
    finally:
        if managed_dir is not None:
            managed_dir.cleanup()


def _build_service(*, root_dir: Path, provider_mode: str) -> CognitiveOSService:
    settings = AppSettings.from_env(
        db_path=root_dir / "benchmark.db",
        memory_output_path=root_dir / "MEMORY.MD",
    )
    settings.search_async_access_logging = False
    settings.search_governance_interval_seconds = 3600
    settings.dream_event_threshold = 10_000
    settings.dream_age_min_event_count = 10_000
    settings.long_document_token_threshold = 200
    settings.max_main_document_chars = 800

    embedding_provider = None
    chat_provider = None
    if provider_mode == "fake":
        embedding_provider = BenchmarkEmbeddingProvider()
        chat_provider = BenchmarkChatProvider()
    elif provider_mode == "env":
        service = CognitiveOSService.from_settings(settings)
        service.initialize()
        return service

    service = CognitiveOSService(
        settings=settings,
        repository=SQLiteRepository(settings.db_path),
        embedding_provider=embedding_provider,
        chat_provider=chat_provider,
    )
    service.initialize()
    return service


def _seed_corpus(
    *,
    service: CognitiveOSService,
    fixture_paths: dict[str, Path],
) -> dict[str, Any]:
    ids_by_key: dict[str, str] = {}
    for fixture in DEFAULT_BENCHMARK_CORPUS.content_nodes:
        receipt = service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload=fixture.payload,
            tags=list(fixture.tags),
            durability=fixture.durability,
            name=fixture.name,
            force=True,
        )
        ids_by_key[fixture.key] = receipt.node_id or ""

    for fixture in DEFAULT_BENCHMARK_CORPUS.file_nodes:
        receipt = service.add_node(
            payload_type=AddPayloadType.FILE,
            payload=str(fixture_paths[fixture.key]),
            tags=list(fixture.tags),
            durability=fixture.durability,
            name=fixture.name,
            force=True,
        )
        ids_by_key[fixture.key] = receipt.node_id or ""

    service._wait_for_background_tasks()

    for fixture in DEFAULT_BENCHMARK_CORPUS.links:
        service.link_nodes(
            src_id=ids_by_key[fixture.src],
            dst_id=ids_by_key[fixture.dst],
            relation=fixture.relation,
        )

    service._wait_for_background_tasks()
    nodes = service.repository.list_all_nodes()
    node_type_counts: dict[str, int] = {}
    file_keys = {fixture.key for fixture in DEFAULT_BENCHMARK_CORPUS.file_nodes}
    for node in nodes:
        node_type_counts[node.node_type] = node_type_counts.get(node.node_type, 0) + 1

    return {
        "ids_by_key": ids_by_key,
        "ingestion": {
            "total_nodes": len(nodes),
            "node_type_counts": node_type_counts,
            "source_document_ids": {
                key: node_id for key, node_id in ids_by_key.items() if key in file_keys
            },
        },
    }


def _run_quality_tasks(
    *,
    service: CognitiveOSService,
    ids_by_key: dict[str, str],
) -> dict[str, Any]:
    key_by_id = {node_id: key for key, node_id in ids_by_key.items()}
    tasks: list[dict[str, Any]] = []
    skipped = 0
    passed = 0
    for fixture in DEFAULT_BENCHMARK_CORPUS.quality_tasks:
        if fixture.requires_embedding and service.embedding_provider is None:
            skipped += 1
            tasks.append(
                {
                    "task_id": fixture.task_id,
                    "status": "skipped",
                    "reason": "Embedding provider is not configured.",
                }
            )
            continue

        started_at = perf_counter()
        results = service.search(
            keyword=fixture.keyword,
            query=fixture.query,
            top_k=fixture.top_k,
            include_neighbors=fixture.include_neighbors,
            include_evidence=fixture.include_evidence,
        )
        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        top_keys = [key_by_id.get(result.id, result.id) for result in results]
        top_node_types = [result.node_type for result in results]
        first_linked = (
            [key_by_id.get(node.id, node.id) for node in results[0].linked_nodes]
            if results
            else []
        )

        checks = {
            "expected_root": (
                fixture.expected_root is None
                or (bool(results) and key_by_id.get(results[0].id) == fixture.expected_root)
            ),
            "expected_linked": all(item in first_linked for item in fixture.expected_linked),
            "expected_node_types_present": all(
                item in top_node_types for item in fixture.expected_node_types_present
            ),
            "expected_node_types_absent": all(
                item not in top_node_types for item in fixture.expected_node_types_absent
            ),
        }
        task_passed = all(checks.values())
        if task_passed:
            passed += 1
        tasks.append(
            {
                "task_id": fixture.task_id,
                "status": "passed" if task_passed else "failed",
                "duration_ms": duration_ms,
                "top_keys": top_keys,
                "top_node_types": top_node_types,
                "first_result_linked_keys": first_linked,
                "checks": checks,
            }
        )

    executed = len(tasks) - skipped
    return {
        "total_tasks": len(DEFAULT_BENCHMARK_CORPUS.quality_tasks),
        "executed_tasks": executed,
        "passed_tasks": passed,
        "skipped_tasks": skipped,
        "tasks": tasks,
    }


def _run_runtime_benchmarks(
    *,
    service: CognitiveOSService,
    ids_by_key: dict[str, str],
    fixture_paths: dict[str, Path],
    iterations: int,
) -> dict[str, Any]:
    service.search(keyword="CognitiveOS", top_k=3, include_neighbors=1)
    service.read_nodes([ids_by_key["project_overview"]], include_content=False)
    service._wait_for_background_tasks()

    operations: dict[str, dict[str, Any]] = {}
    operations["add_content"] = _measure_operation(
        iterations=iterations,
        fn=lambda index: service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload=(
                "Benchmark add payload for runtime instrumentation. "
                f"Iteration {index} captures deterministic baseline behavior."
            ),
            tags=["benchmark", "runtime"],
            name=f"Benchmark Add {index}",
            force=True,
        ),
        post_action=service._wait_for_background_tasks,
        last_metrics=lambda: {},
    )
    operations["add_file"] = _measure_operation(
        iterations=iterations,
        fn=lambda index: service.add_node(
            payload_type=AddPayloadType.FILE,
            payload=str(fixture_paths["bootstrap_guide"]),
            tags=["benchmark", "file"],
            name=f"Benchmark File {index}",
            force=True,
        ),
        post_action=service._wait_for_background_tasks,
        last_metrics=lambda: {},
    )
    operations["search_keyword"] = _measure_operation(
        iterations=iterations,
        fn=lambda _index: service.search(
            keyword="bootstrap",
            top_k=3,
            include_neighbors=1,
            include_evidence=False,
        ),
        last_metrics=lambda: service.last_runtime_metrics,
    )
    if service.embedding_provider is not None:
        operations["search_hybrid"] = _measure_operation(
            iterations=iterations,
            fn=lambda _index: service.search(
                keyword="bootstrap",
                query="Codex bootstrap mount files",
                top_k=3,
                include_neighbors=1,
                include_evidence=False,
            ),
            last_metrics=lambda: service.last_runtime_metrics,
        )
    operations["read_summary"] = _measure_operation(
        iterations=iterations,
        fn=lambda _index: service.read_nodes(
            [ids_by_key["project_overview"], ids_by_key["bootstrap_artifacts"]],
            include_content=False,
        ),
        last_metrics=lambda: service.last_runtime_metrics,
    )
    operations["read_content"] = _measure_operation(
        iterations=iterations,
        fn=lambda _index: service.read_nodes(
            [ids_by_key["long_memory_walkthrough"]],
            include_content=True,
        ),
        last_metrics=lambda: service.last_runtime_metrics,
    )
    return {
        "iterations": iterations,
        "operations": operations,
    }


def _run_dream_benchmark(service: CognitiveOSService) -> dict[str, Any]:
    service.search(keyword="benchmark", top_k=3, include_neighbors=1)
    service.search(keyword="bootstrap", top_k=3, include_neighbors=2)
    service.read_nodes(
        [
            node.id
            for node in service.repository.list_all_nodes()
            if node.node_type in {"memory", "source_document"}
        ][:6],
        include_content=False,
    )
    service._wait_for_background_tasks()

    started_at = perf_counter()
    result = service.run_dream(
        window_hours=720,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=24,
    )
    duration_ms = round((perf_counter() - started_at) * 1000, 3)
    return {
        "duration_ms": duration_ms,
        "status": result.status,
        "candidate_count": len(result.candidate_node_ids),
        "clusters_created": result.clusters_created,
        "pending_compactions": len(result.pending_compactions),
        "notices": result.notices,
        "runtime_metrics": service.last_runtime_metrics,
    }


def _measure_operation(
    *,
    iterations: int,
    fn: Callable[[int], Any],
    last_metrics: Callable[[], dict[str, Any]],
    post_action: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    durations_ms: list[float] = []
    runtime_totals_ms: list[float] = []
    sample_outputs: list[dict[str, Any]] = []
    for index in range(iterations):
        started_at = perf_counter()
        result = fn(index)
        if post_action is not None:
            post_action()
        durations_ms.append(round((perf_counter() - started_at) * 1000, 3))
        metrics = last_metrics() or {}
        timings = metrics.get("timings_ms", {}) if isinstance(metrics, dict) else {}
        runtime_total = timings.get("total")
        if isinstance(runtime_total, int | float):
            runtime_totals_ms.append(round(float(runtime_total), 3))
        if index == 0:
            sample_outputs.append(_sample_output(result))
    return {
        "iterations": iterations,
        "wall_clock_ms": _summarize_numbers(durations_ms),
        "runtime_total_ms": _summarize_numbers(runtime_totals_ms),
        "sample_outputs": sample_outputs,
    }


def _summarize_numbers(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    if len(values) == 1:
        return {
            "count": 1,
            "min": values[0],
            "max": values[0],
            "mean": values[0],
            "median": values[0],
        }
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "stdev": round(statistics.pstdev(values), 3),
    }


def _sample_output(result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        return {
            "result_type": "list",
            "length": len(result),
        }
    if isinstance(result, dict):
        return {
            "result_type": "dict",
            "keys": sorted(result.keys())[:8],
        }
    if hasattr(result, "model_dump"):
        model_payload = result.model_dump()
        return {
            "result_type": type(result).__name__,
            "keys": sorted(model_payload.keys())[:8],
        }
    return {
        "result_type": type(result).__name__,
        "repr": repr(result)[:160],
    }
