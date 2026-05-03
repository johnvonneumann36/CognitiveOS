from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from cognitiveos.collection_ingestion import (
    collection_hint_text,
    inspect_folder_collection,
    repository_hint,
)
from cognitiveos.config import AppSettings
from cognitiveos.db.connection import open_connection
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.document_ingestion import DocumentIngestionPipeline
from cognitiveos.dream import DreamCompiler
from cognitiveos.exceptions import InvalidPayloadError
from cognitiveos.extractors.defaults import DefaultContentExtractor
from cognitiveos.graph_governance import GraphGovernanceEngine
from cognitiveos.metadata_shapes import (
    metadata_profile_kind,
    metadata_profile_section,
    metadata_source_ref,
)
from cognitiveos.models import (
    AddPayloadType,
    BootstrapBundle,
    ConflictNode,
    DreamCompactionResolution,
    DreamCompactionTask,
    DreamResult,
    DreamRunInfo,
    DreamStatus,
    EdgeRecord,
    HostBootstrapStatus,
    HostOnboardingQuestion,
    NodeRecord,
    ReadNodeResult,
    Receipt,
    SearchResult,
)
from cognitiveos.providers.base import ChatProvider, EmbeddingProvider
from cognitiveos.providers.factory import build_chat_provider, build_embedding_provider
from cognitiveos.semantics import cosine_similarity


logger = logging.getLogger(__name__)


class CognitiveOSService:
    DOCUMENT_DESCRIPTION_MAX_CHARS = 800
    DOCUMENT_TAG_LIMIT = 8
    DOCUMENT_PROFILE_PROMPT_CHARS = 16000
    DELETE_VIA_UPDATE_TAG = "__delete__"
    APP_STATE_BOOTSTRAP_ONBOARDING_COMPLETED = "bootstrap_onboarding_completed"
    APP_STATE_BOOTSTRAP_ONBOARDING_ANSWERS = "bootstrap_onboarding_answers"
    APP_STATE_BOOTSTRAP_ONBOARDING_NODE_IDS = "bootstrap_onboarding_node_ids"
    APP_STATE_BOOTSTRAP_HOST_INSTALLS = "bootstrap_host_installs"
    APP_STATE_BOOTSTRAP_HOST_MEMORY_TARGETS = "bootstrap_host_memory_targets"
    CANONICAL_PROFILE_SECTION_NAMES = {
        "identity": "Bootstrap Identity",
        "communication": "Bootstrap Communication Preferences",
        "workspace": "Bootstrap Workspace Goal",
        "engineering": "Engineering Collaboration Preferences",
    }
    CANONICAL_PROFILE_SECTION_TAGS = {
        "identity": ["profile", "bootstrap"],
        "communication": ["profile", "bootstrap"],
        "workspace": ["profile", "bootstrap"],
        "engineering": ["profile", "preferences", "engineering"],
    }
    SUPPORTED_HOST_KINDS = {
        "generic",
        "codex",
        "claude_code",
        "claude_desktop",
        "gemini_cli",
        "cursor",
    }
    ALLOWED_DURABILITY_VALUES = {"working", "durable", "pinned", "ephemeral"}
    BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cognitiveos-bg")

    def __init__(
        self,
        *,
        settings: AppSettings,
        repository: SQLiteRepository,
        extractor: DefaultContentExtractor | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        chat_provider: ChatProvider | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.extractor = extractor or DefaultContentExtractor()
        self.embedding_provider = embedding_provider
        self.chat_provider = chat_provider
        self.last_notices: list[str] = []
        self.last_runtime_metrics: dict[str, Any] = {}
        self._background_futures: list[Future[Any]] = []
        self._last_search_governance_at: datetime | None = None
        self._initialized = False
        self.governance = GraphGovernanceEngine(
            settings=settings,
            repository=repository,
            default_actor=settings.default_actor,
        )
        self.document_ingestion = DocumentIngestionPipeline(
            settings=settings,
            repository=repository,
            extractor=self.extractor,
            default_actor=settings.default_actor,
            summarize_content=self._summarize,
            build_document_profile=self._build_document_profile,
            embed_content=self._embed_content,
            find_similarity_conflicts=self._find_similarity_conflicts,
            schedule_semantic_neighbor_refresh=self._schedule_semantic_neighbor_refresh,
        )
        self.dream_compiler = DreamCompiler(
            repository,
            governance_engine=self.governance,
            max_node_content_chars=settings.max_node_content_chars,
            semantic_neighbor_k=settings.semantic_neighbor_k,
            embedding_provider=embedding_provider,
            chat_provider=chat_provider,
        )

    @classmethod
    def from_settings(cls, settings: AppSettings) -> CognitiveOSService:
        settings.ensure_runtime_paths()
        repository = SQLiteRepository(settings.db_path)
        return cls(
            settings=settings,
            repository=repository,
            embedding_provider=build_embedding_provider(settings),
            chat_provider=build_chat_provider(settings),
        )

    def initialize(self) -> None:
        if self._initialized:
            return
        self.repository.initialize()
        self._initialized = True

    def _submit_background_task(
        self,
        task_name: str,
        fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        future = self.BACKGROUND_EXECUTOR.submit(fn, *args, **kwargs)
        future.task_name = task_name  # type: ignore[attr-defined]
        self._background_futures.append(future)
        self._background_futures = [item for item in self._background_futures if not item.done()]

    def _wait_for_background_tasks(self) -> None:
        pending = self._background_futures
        self._background_futures = []
        for future in pending:
            future.result()

    def _record_access(self, node_ids: list[str], *, access_type: str) -> None:
        if self.settings.search_async_access_logging:
            self._submit_background_task(
                f"record_access:{access_type}",
                self.repository.record_access,
                node_ids,
                access_type=access_type,
            )
            return
        self.repository.record_access(node_ids, access_type=access_type)

    def _schedule_semantic_neighbor_refresh(self, node_id: str) -> None:
        self._submit_background_task(
            "refresh_semantic_neighbors",
            self.repository.refresh_semantic_neighbors_for_node,
            node_id,
            top_k=self.settings.semantic_neighbor_k,
        )

    def _apply_search_governance_if_due(self) -> None:
        interval_seconds = max(0, self.settings.search_governance_interval_seconds)
        now = datetime.now(UTC)
        if (
            self._last_search_governance_at is None
            or interval_seconds == 0
            or (now - self._last_search_governance_at).total_seconds() >= interval_seconds
        ):
            self.governance.apply_relationship_governance()
            self._last_search_governance_at = now

    def search(
        self,
        *,
        query: str | None = None,
        keyword: str | None = None,
        top_k: int = 5,
        include_neighbors: int = 1,
        include_evidence: bool = False,
    ) -> list[SearchResult]:
        total_started = perf_counter()
        self.initialize()
        self.last_notices = []
        timings_ms: dict[str, float] = {}
        if not (query and query.strip()) and not (keyword and keyword.strip()):
            raise InvalidPayloadError(
                "Search requires at least one non-empty query or keyword. "
                "Use keyword for exact recall, query for semantic recall, or both."
            )

        phase_started = perf_counter()
        self._apply_search_governance_if_due()
        timings_ms["governance"] = self._elapsed_ms(phase_started)
        bounded_neighbors = max(0, min(include_neighbors, 3))
        semantic_matches: list[tuple[str, float]] = []
        keyword_matches: list[tuple[str, float]] = []
        candidate_cap = max(
            top_k,
            min(max(top_k * 2, 10), self.settings.search_candidate_cap),
        )

        if query and self.embedding_provider is not None:
            phase_started = perf_counter()
            self._ensure_embeddings()
            timings_ms["ensure_embeddings"] = self._elapsed_ms(phase_started)
            phase_started = perf_counter()
            semantic_matches = self._semantic_search_matches(
                query=query,
                top_k=candidate_cap,
            )
            timings_ms["semantic_search"] = self._elapsed_ms(phase_started)

        if keyword:
            phase_started = perf_counter()
            keyword_matches = self.repository.search_keyword_matches(
                keyword=keyword,
                top_k=candidate_cap,
            )
            timings_ms["keyword_search"] = self._elapsed_ms(phase_started)

        phase_started = perf_counter()
        ordered_ids, scores = self._hybrid_rank(
            semantic_matches=semantic_matches,
            keyword_matches=keyword_matches,
            top_k=top_k,
        )
        timings_ms["hybrid_rank"] = self._elapsed_ms(phase_started)

        phase_started = perf_counter()
        results = self.repository.build_search_results(
            ordered_ids,
            include_neighbors=bounded_neighbors,
            scores=scores,
        )
        timings_ms["build_results"] = self._elapsed_ms(phase_started)
        phase_started = perf_counter()
        results = self.document_ingestion.filter_search_results(
            results,
            top_k=top_k,
            include_evidence=include_evidence,
        )
        timings_ms["filter_results"] = self._elapsed_ms(phase_started)
        phase_started = perf_counter()
        self._record_access([result.id for result in results], access_type="search_hit")
        timings_ms["record_access"] = self._elapsed_ms(phase_started)
        timings_ms["total"] = self._elapsed_ms(total_started)
        self.last_runtime_metrics = {
            "operation": "search",
            "timings_ms": timings_ms,
            "result_count": len(results),
            "top_k": top_k,
            "candidate_cap": candidate_cap,
            "include_neighbors": bounded_neighbors,
        }
        return self._decorate_search_results(results, self._maybe_handle_dream("search"))

    def read_nodes(
        self,
        ids: list[str],
        *,
        include_content: bool = False,
    ) -> dict[str, ReadNodeResult]:
        total_started = perf_counter()
        self.last_notices = []
        timings_ms: dict[str, float] = {}
        phase_started = perf_counter()
        self.governance.apply_relationship_governance()
        timings_ms["governance"] = self._elapsed_ms(phase_started)
        phase_started = perf_counter()
        results = self.repository.read_nodes(ids, include_content=include_content)
        timings_ms["read_nodes"] = self._elapsed_ms(phase_started)
        if include_content:
            phase_started = perf_counter()
            self._hydrate_remote_snapshot_content(results)
            timings_ms["hydrate_snapshot_content"] = self._elapsed_ms(phase_started)
        phase_started = perf_counter()
        self.governance.reinforce_read_coaccess(ids)
        timings_ms["reinforce_read_coaccess"] = self._elapsed_ms(phase_started)
        phase_started = perf_counter()
        self._record_access(list(results.keys()), access_type="read")
        timings_ms["record_access"] = self._elapsed_ms(phase_started)
        timings_ms["total"] = self._elapsed_ms(total_started)
        self.last_runtime_metrics = {
            "operation": "read",
            "timings_ms": timings_ms,
            "result_count": len(results),
            "include_content": include_content,
        }
        return self._decorate_read_results(results, self._maybe_handle_dream("read"))

    def add_node(
        self,
        *,
        payload_type: AddPayloadType,
        payload: str,
        tags: list[str] | None = None,
        durability: str | None = None,
        force: bool = False,
        name: str | None = None,
    ) -> Receipt:
        self.initialize()
        self.last_notices = []
        metadata: dict[str, Any] = {}
        merged_tags = self._merge_tags(tags or [])

        if payload_type is AddPayloadType.FILE:
            receipt = self.document_ingestion.add_file_node(
                payload=payload,
                tags=merged_tags,
                durability=self._resolve_node_durability(
                    durability=durability,
                    node_type="source_document",
                ),
                force=force,
                name=name,
            )
            receipt.notices = self._maybe_handle_dream("add")
            return receipt
        if payload_type is AddPayloadType.FOLDER:
            receipt = self._add_folder_node(
                payload=payload,
                tags=merged_tags,
                durability=durability,
                force=force,
                name=name,
            )
            receipt.notices = self._maybe_handle_dream("add")
            return receipt

        node_name = name
        content = payload.strip()

        if not content:
            raise InvalidPayloadError("Content payload cannot be empty.")
        self._validate_node_content_limit(content)
        profile_section = self._infer_canonical_profile_section(
            tags=merged_tags,
            name=node_name,
            content=content,
        )
        if profile_section is not None:
            receipt = self._upsert_canonical_profile_memory(
                section=profile_section,
                content=content,
                tags=merged_tags,
                durability=durability,
            )
            receipt.notices = self._maybe_handle_dream("add")
            return receipt
        node_type = self._infer_node_type(fallback="memory")

        embedding = self._embed_content(content)
        if embedding is not None and not force:
            conflicts = self._find_similarity_conflicts(embedding)
            if conflicts:
                receipt = Receipt(
                    status="blocked",
                    action_taken="none",
                    reason="High similarity detected.",
                    conflicting_nodes=[ConflictNode(**item) for item in conflicts],
                    suggestion=(
                        "Use 'update' to modify an existing node, or re-run 'add' with force=true."
                    ),
                )
                receipt.notices = self._maybe_handle_dream("add")
                return receipt

        node = NodeRecord(
            id=f"node_{uuid4().hex}",
            name=node_name,
            description=self._summarize(content),
            content=content,
            embedding=embedding,
            tags=merged_tags,
            metadata=metadata,
            node_type=node_type,
            durability=self._resolve_node_durability(
                durability=durability,
                node_type=node_type,
            ),
        )
        action_type = "force_add" if force else "create"
        self.repository.create_node(
            node,
            actor=self.settings.default_actor,
            action_type=action_type,
        )
        self._schedule_semantic_neighbor_refresh(node.id)
        receipt = Receipt(status="success", action_taken="created", node_id=node.id)
        receipt.notices = self._maybe_handle_dream("add")
        return receipt

    def update_node(
        self,
        *,
        node_id: str,
        content: str,
        tags: list[str] | None = None,
        durability: str | None = None,
    ) -> Receipt:
        self.initialize()
        self.last_notices = []
        existing_node = self.repository.get_node(node_id)
        normalized_tags = self._merge_tags(tags) if tags is not None else None
        if normalized_tags is not None and self.DELETE_VIA_UPDATE_TAG in normalized_tags:
            receipt = self.delete_node(node_id=node_id, trigger="update_tag")
            receipt.notices = self._maybe_handle_dream("update")
            return receipt
        clean_content = content.strip()
        if not clean_content:
            raise InvalidPayloadError("Updated content cannot be empty.")
        if existing_node.node_type == "source_document":
            self.document_ingestion._validate_document_content(clean_content)
        else:
            self._validate_node_content_limit(clean_content)
        next_tags = normalized_tags if normalized_tags is not None else existing_node.tags
        description = self._summarize(clean_content)
        embedding = self._embed_content(clean_content)
        if existing_node.node_type == "source_document":
            user_tags, _generated_tags = self._split_document_tags(existing_node)
            if normalized_tags is not None:
                user_tags = normalized_tags
            update_mode = (
                "manual_override" if clean_content != existing_node.content else "profile_refresh"
            )
            profile = self._build_document_profile(
                content=clean_content,
                tags=user_tags,
                source_metadata=existing_node.metadata,
                name=existing_node.name,
            )
            next_tags = profile["tags"]
            description = profile["description"]
            embedding = self._embed_content(profile["embedding_input"])
            next_node = NodeRecord(
                id=existing_node.id,
                name=existing_node.name,
                description=description,
                content=clean_content,
                embedding=embedding,
                tags=next_tags,
                metadata={
                    **existing_node.metadata,
                    "source": {
                        **existing_node.metadata.get("source", {}),
                        **(
                            {}
                            if update_mode == "manual_override"
                            else {"hash": self._source_hash(clean_content)}
                        ),
                    },
                    "document": {
                        **existing_node.metadata.get("document", {}),
                        "content_length": len(clean_content),
                        "token_estimate": max(1, len(clean_content) // 4),
                    },
                    "document_profile": self.document_ingestion._build_document_profile_metadata(
                        generated_tags=profile.get("generated_tags", []),
                        prior_node=existing_node,
                        update_mode=update_mode,
                    ),
                },
                node_type=existing_node.node_type,
                durability=(
                    self._normalize_durability(durability)
                    if durability is not None
                    else existing_node.durability
                ),
                last_reinforced_at=existing_node.last_reinforced_at,
            )
            audit_log_id = self.repository.overwrite_node(
                next_node,
                actor=self.settings.default_actor,
                action_type="update",
            )
        elif existing_node.node_type == "source_collection":
            description = self._generate_collection_description(
                content=clean_content,
                metadata=existing_node.metadata,
                name=existing_node.name,
            )
            embedding = self._embed_content(
                self._compose_collection_embedding_text(
                    description=description,
                    tags=next_tags,
                    metadata=existing_node.metadata,
                    content=clean_content,
                )
            )
            audit_log_id = self.repository.update_node(
                node_id,
                content=clean_content,
                description=description,
                embedding=embedding,
                tags=next_tags,
                durability=(
                    self._normalize_durability(durability) if durability is not None else None
                ),
                actor=self.settings.default_actor,
            )
        else:
            audit_log_id = self.repository.update_node(
                node_id,
                content=clean_content,
                description=description,
                embedding=embedding,
                tags=next_tags,
                durability=(
                    self._normalize_durability(durability) if durability is not None else None
                ),
                actor=self.settings.default_actor,
            )
        self._schedule_semantic_neighbor_refresh(node_id)
        receipt = Receipt(
            status="success",
            action_taken="updated",
            node_id=node_id,
            audit_log_id=audit_log_id,
        )
        receipt.notices = self._maybe_handle_dream("update")
        return receipt

    def delete_node(self, *, node_id: str, trigger: str = "explicit") -> Receipt:
        self.initialize()
        self.last_notices = []
        existing_node = self.repository.get_node(node_id)
        snapshot_path = self._snapshot_path_for_node(existing_node)
        self._remove_known_app_state_node_references(node_id)
        removed = self.repository.delete_nodes([node_id], actor=self.settings.default_actor)
        action_taken = "deleted" if removed else "none"
        notices: list[str] = []
        if removed and snapshot_path is not None:
            try:
                if snapshot_path.exists():
                    snapshot_path.unlink()
            except OSError:
                logger.exception("snapshot_cleanup_failed path=%s", snapshot_path)
                notices.append(
                    f"Snapshot cleanup failed for deleted node: {snapshot_path}"
                )
        return Receipt(
            status="success" if removed else "not_found",
            action_taken=action_taken,
            node_id=node_id if removed else None,
            reason=(
                f"Node deleted via {trigger}."
                if removed
                else f"Node '{node_id}' was not found."
            ),
            suggestion=(
                f"Triggered by reserved tag '{self.DELETE_VIA_UPDATE_TAG}'."
                if removed
                else None
            ),
            edge={
                "removed_node_id": node_id,
                "node_type": existing_node.node_type,
                "trigger": trigger,
                "removed_count": removed,
            }
            if removed
            else None,
            notices=notices,
        )

    def link_nodes(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation: str,
    ) -> Receipt:
        self.initialize()
        self.last_notices = []
        action_taken, edge_payload = self.governance.create_or_reinforce_manual_link(
            src_id=src_id,
            dst_id=dst_id,
            relation=relation,
        )
        receipt = Receipt(
            status="success",
            action_taken=action_taken,
            edge=edge_payload,
        )
        receipt.notices = self._maybe_handle_dream("link")
        return receipt

    def unlink_nodes(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation: str | None = None,
    ) -> Receipt:
        self.initialize()
        removed = self.repository.delete_edge(src_id, dst_id, relation=relation)
        action_taken = "edge_removed" if removed else "none"
        return Receipt(
            status="success" if removed else "not_found",
            action_taken=action_taken,
            edge={"src": src_id, "dst": dst_id, "relation": relation, "removed_count": removed},
        )

    def list_relationships(
        self,
        *,
        node_id: str,
        relation: str | None = None,
        status: str | None = None,
    ) -> list[EdgeRecord]:
        self.initialize()
        self.governance.apply_relationship_governance(node_id=node_id)
        return self.repository.list_relationships(node_id, relation=relation, status=status)

    def reinforce_relationship(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation: str,
        delta: float = 0.25,
    ) -> Receipt:
        self.initialize()
        edge = self.repository.reinforce_edge(src_id, dst_id, relation, delta=delta)
        return Receipt(
            status="success",
            action_taken="edge_reinforced",
            edge=edge.model_dump(),
        )

    def prune_relationships(
        self,
        *,
        node_id: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        self.initialize()
        return self.governance.prune_relationships(node_id=node_id, dry_run=dry_run)

    def pin_memory(self, *, node_id: str) -> Receipt:
        self.initialize()
        node = self.repository.set_node_pinned(node_id, pinned=True)
        return Receipt(
            status="success",
            action_taken="node_pinned",
            node_id=node.id,
            reason="Durability set to pinned.",
        )

    def unpin_memory(self, *, node_id: str) -> Receipt:
        self.initialize()
        node = self.repository.set_node_pinned(node_id, pinned=False)
        return Receipt(
            status="success",
            action_taken="node_unpinned",
            node_id=node.id,
            reason="Durability downgraded to durable.",
        )

    def refresh_source_document(self, *, node_id: str) -> Receipt:
        self.initialize()
        return self.document_ingestion.refresh_source_document(node_id=node_id)

    def run_dream(
        self,
        *,
        output_path: Path | None = None,
        window_hours: int = 168,
        min_accesses: int = 2,
        min_cluster_size: int = 2,
        max_candidates: int = 100,
        similarity_threshold: float | None = None,
        trigger_reason: str | None = None,
        auto_triggered: bool = False,
        background: bool = False,
    ) -> DreamResult:
        self.initialize()
        self.last_notices = []
        self.governance.apply_relationship_governance()
        self._wait_for_background_tasks()
        target_path = output_path or self.settings.memory_output_path
        self._ensure_embeddings()
        if background:
            if self.chat_provider is None:
                raise InvalidPayloadError(
                    "Background dream auto-execution requires a chat provider or host task flow."
                )
            active_run = self.repository.get_active_dream_run()
            if active_run is not None:
                return DreamResult(
                    status="queued",
                    run_id=active_run["run_id"],
                    trigger_reason=trigger_reason or active_run["trigger_reason"],
                    auto_triggered=auto_triggered,
                    notices=[
                        f"Dream run {active_run['run_id']} is already {active_run['status']}."
                    ],
                    memory_path=active_run["memory_path"],
                )
            return self.queue_background_dream(
                output_path=target_path,
                window_hours=window_hours,
                min_accesses=min_accesses,
                min_cluster_size=min_cluster_size,
                max_candidates=max_candidates,
                similarity_threshold=similarity_threshold or self.settings.similarity_threshold,
                trigger_reason=trigger_reason,
                auto_triggered=auto_triggered,
            )

        run_id = self.repository.start_dream_run(
            trigger_reason=trigger_reason or "manual",
            auto_triggered=auto_triggered,
            requires_chat=self.chat_provider is None,
        )
        return self.execute_dream_run(
            run_id=run_id,
            output_path=target_path,
            window_hours=window_hours,
            min_accesses=min_accesses,
            min_cluster_size=min_cluster_size,
            max_candidates=max_candidates,
            similarity_threshold=similarity_threshold or self.settings.similarity_threshold,
            trigger_reason=trigger_reason or "manual",
            auto_triggered=auto_triggered,
        )

    def execute_dream_run(
        self,
        *,
        run_id: str,
        output_path: Path,
        window_hours: int,
        min_accesses: int,
        min_cluster_size: int,
        max_candidates: int,
        similarity_threshold: float,
        trigger_reason: str,
        auto_triggered: bool,
    ) -> DreamResult:
        self.initialize()
        total_started = perf_counter()
        self._wait_for_background_tasks()
        self.repository.mark_dream_run_running(run_id)
        timings_ms: dict[str, float] = {}
        try:
            phase_started = perf_counter()
            result = self.dream_compiler.run(
                run_id=run_id,
                output_path=output_path,
                window_hours=window_hours,
                min_accesses=min_accesses,
                min_cluster_size=min_cluster_size,
                max_candidates=max_candidates,
                similarity_threshold=similarity_threshold,
            )
            timings_ms["compiler_run"] = self._elapsed_ms(phase_started)
        except Exception as exc:
            self.repository.fail_dream_run(
                run_id,
                error=f"Dream execution failed: {exc}",
            )
            raise

        result.run_id = run_id
        result.trigger_reason = trigger_reason
        result.auto_triggered = auto_triggered
        self.repository.complete_dream_run(
            run_id,
            status=result.status,
            candidate_count=len(result.candidate_node_ids),
            clusters_created=result.clusters_created,
            memory_path=result.memory_path,
            notes=result.notices,
            mark_completed=not result.pending_compactions,
        )
        timings_ms["total"] = self._elapsed_ms(total_started)
        self.last_runtime_metrics = {
            "operation": "dream",
            "timings_ms": timings_ms,
            "candidate_count": len(result.candidate_node_ids),
            "clusters_created": result.clusters_created,
            "pending_compactions": len(result.pending_compactions),
        }
        return result

    def queue_background_dream(
        self,
        *,
        output_path: Path | None = None,
        window_hours: int = 168,
        min_accesses: int = 2,
        min_cluster_size: int = 2,
        max_candidates: int = 100,
        similarity_threshold: float | None = None,
        trigger_reason: str | None = None,
        auto_triggered: bool = False,
    ) -> DreamResult:
        self.initialize()
        self._wait_for_background_tasks()
        active_run = self.repository.get_active_dream_run()
        if active_run is not None:
            return DreamResult(
                status="queued",
                run_id=active_run["run_id"],
                trigger_reason=trigger_reason or active_run["trigger_reason"],
                auto_triggered=auto_triggered,
                notices=[f"Dream run {active_run['run_id']} is already {active_run['status']}."],
                memory_path=active_run["memory_path"],
            )

        run_id = self.repository.start_dream_run(
            trigger_reason=trigger_reason or "manual",
            auto_triggered=auto_triggered,
            requires_chat=self.chat_provider is None,
            status="queued",
        )
        target_path = output_path or self.settings.memory_output_path
        self._launch_background_dream(
            run_id=run_id,
            output_path=target_path,
            window_hours=window_hours,
            min_accesses=min_accesses,
            min_cluster_size=min_cluster_size,
            max_candidates=max_candidates,
            similarity_threshold=similarity_threshold or self.settings.similarity_threshold,
            trigger_reason=trigger_reason or "manual",
            auto_triggered=auto_triggered,
        )
        return DreamResult(
            status="queued",
            run_id=run_id,
            trigger_reason=trigger_reason or "manual",
            auto_triggered=auto_triggered,
            memory_path=str(target_path),
            notices=[f"Dream run {run_id} was queued for background execution."],
        )

    def list_dream_compactions(
        self,
        *,
        run_id: str | None = None,
        status: str | None = "pending",
    ) -> list[DreamCompactionTask]:
        self.initialize()
        rows = self.repository.list_dream_compaction_tasks(run_id=run_id, status=status)
        return [self._dream_task_from_row(row) for row in rows]

    def resolve_dream_compaction(
        self,
        *,
        task_id: str,
        title: str | None = None,
        description: str | None = None,
        content: str | None = None,
        use_heuristic: bool = False,
        background: bool | None = None,
    ) -> DreamCompactionResolution:
        self.initialize()
        task_row = self.repository.get_dream_compaction_task(task_id)
        if task_row is None:
            raise InvalidPayloadError(f"Dream compaction task '{task_id}' was not found.")
        if task_row["status"] != "pending":
            raise InvalidPayloadError(f"Dream compaction task '{task_id}' is already resolved.")

        task_payload = dict(task_row)
        if use_heuristic:
            should_background = True if background is None else background
            if should_background:
                self.repository.mark_dream_compaction_task_running(task_id)
                self._launch_background_heuristic_compaction(task_id=task_id)
                return DreamCompactionResolution(
                    status="queued",
                    task_id=task_id,
                    run_id=task_payload["run_id"],
                    resolution_backend="heuristic",
                    remaining_tasks=self.repository.count_pending_dream_compaction_tasks(
                        task_payload["run_id"]
                    ),
                    dream_completed=False,
                )
            super_node = self.execute_heuristic_compaction(task_id=task_id)
            resolution_backend = "heuristic"
        else:
            clean_title = (title or "").strip()
            clean_description = (description or "").strip()
            clean_content = (content or "").strip()
            if not clean_title or not clean_description or not clean_content:
                raise InvalidPayloadError(
                    "Host dream compaction requires non-empty title, description, and content."
                )
            super_node = self.dream_compiler.resolve_task_with_host_payload(
                task_payload,
                title=clean_title,
                description=clean_description,
                content=clean_content,
            )
            resolution_backend = "host_agent"

        return self._finalize_compaction_resolution(
            task_id=task_id,
            run_id=task_payload["run_id"],
            resolution_backend=resolution_backend,
            node_id=super_node.id,
        )

    def execute_heuristic_compaction(self, *, task_id: str) -> NodeRecord:
        self.initialize()
        task_row = self.repository.get_dream_compaction_task(task_id)
        if task_row is None:
            raise InvalidPayloadError(f"Dream compaction task '{task_id}' was not found.")
        task_payload = dict(task_row)
        try:
            super_node = self.dream_compiler.resolve_task_with_heuristic(task_payload)
        except Exception as exc:
            self.repository.fail_dream_compaction_task(
                task_id,
                error=f"Heuristic compaction failed: {exc}",
            )
            run_row = self.repository.get_dream_run(task_payload["run_id"])
            existing_notes = (
                json.loads(run_row["notes_json"] or "[]") if run_row is not None else []
            )
            self.repository.fail_dream_run(
                task_payload["run_id"],
                error=f"Heuristic compaction task {task_id} failed: {exc}",
                notes=existing_notes,
            )
            raise
        self._finalize_compaction_resolution(
            task_id=task_id,
            run_id=task_payload["run_id"],
            resolution_backend="heuristic",
            node_id=super_node.id,
        )
        return super_node

    def list_dream_runs(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
    ) -> list[DreamRunInfo]:
        self.initialize()
        rows = self.repository.list_dream_runs(status=status, limit=limit)
        return [
            DreamRunInfo(
                run_id=row["run_id"],
                status=row["status"],
                trigger_reason=row["trigger_reason"],
                auto_triggered=bool(row["auto_triggered"]),
                requires_chat=bool(row["requires_chat"]),
                candidate_count=int(row["candidate_count"]),
                clusters_created=int(row["clusters_created"]),
                pending_task_count=int(row["pending_task_count"]),
                memory_path=row["memory_path"],
                notes=json.loads(row["notes_json"] or "[]"),
                started_at=row["started_at"],
                completed_at=row["completed_at"],
            )
            for row in rows
        ]

    def compile_memory_snapshot(self, output_path: Path | None = None) -> Path:
        self.initialize()
        target_path = (output_path or self.settings.memory_output_path).resolve()
        compiled_path = self.governance.compile_memory_snapshot(target_path)
        for mirror_path in self._registered_memory_output_paths(primary_path=compiled_path):
            if mirror_path == compiled_path:
                continue
            self.governance.compile_memory_snapshot(mirror_path)
        return compiled_path

    def get_host_bootstrap_status(
        self,
        *,
        host_kind: str = "generic",
        output_dir: Path | None = None,
    ) -> HostBootstrapStatus:
        self.initialize()
        normalized_host_kind = self._normalize_host_kind(host_kind)
        memory_path = self._host_memory_output_path(host_kind=normalized_host_kind)
        bootstrap_dir = output_dir or self.settings.bootstrap_dir
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = self._bootstrap_artifact_paths(
            bootstrap_dir=bootstrap_dir,
            host_kind=normalized_host_kind,
        )
        host_instruction_path, host_project_config_path = self._host_mount_targets(
            normalized_host_kind
        )

        profile_nodes = [
            node
            for node in self.repository.list_all_nodes()
            if metadata_profile_kind(node.metadata) == "system"
        ]
        onboarding_completed = self._bootstrap_onboarding_completed(
            has_profile_nodes=bool(profile_nodes)
        )
        install_record = self._get_host_install_record(normalized_host_kind)
        installed = self._host_installation_exists(
            normalized_host_kind,
            install_record=install_record,
            host_instruction_path=host_instruction_path,
            host_project_config_path=host_project_config_path,
        )
        questions = self._bootstrap_onboarding_questions()
        first_startup = not onboarding_completed and not profile_nodes
        notices: list[str] = []
        if first_startup:
            notices.append(
                "First startup detected. Ask the user the onboarding questions "
                "before relying on memory."
            )
        if normalized_host_kind != "codex":
            notices.append(
                "Automatic project installation is only implemented for the managed host target. "
                "Use the generated prompt and MCP config manually for this host kind."
            )
        elif not installed:
            notices.append(
                "The managed host mount is not installed yet. Run bootstrap with "
                "install=true to write AGENTS.md and .codex/config.toml."
            )

        return HostBootstrapStatus(
            host_kind=normalized_host_kind,
            first_startup=first_startup,
            onboarding_completed=onboarding_completed,
            needs_onboarding=not onboarding_completed,
            installed=installed,
            needs_mount=not installed,
            installed_at=install_record.get("installed_at") if install_record else None,
            memory_path=str(memory_path),
            bootstrap_prompt_path=str(artifact_paths["bootstrap_prompt_path"]),
            system_prompt_path=str(artifact_paths["system_prompt_path"]),
            mount_manifest_path=str(artifact_paths["mount_manifest_path"]),
            mcp_config_path=str(artifact_paths["mcp_config_path"]),
            onboarding_path=str(artifact_paths["onboarding_path"]),
            host_instruction_path=str(host_instruction_path) if host_instruction_path else None,
            host_project_config_path=(
                str(host_project_config_path) if host_project_config_path else None
            ),
            system_prompt_block=self._host_system_prompt(
                memory_path,
                host_kind=normalized_host_kind,
            ),
            onboarding_questions=questions,
            notices=notices,
        )

    def submit_host_onboarding(
        self,
        *,
        answers: dict[str, str],
        host_kind: str = "generic",
        output_dir: Path | None = None,
    ) -> HostBootstrapStatus:
        self.initialize()
        normalized_host_kind = self._normalize_host_kind(host_kind)
        cleaned_answers = {
            key: value.strip()
            for key, value in answers.items()
            if isinstance(value, str) and value.strip()
        }
        missing = [
            question.id
            for question in self._bootstrap_onboarding_questions()
            if question.required and not cleaned_answers.get(question.id)
        ]
        if missing:
            raise InvalidPayloadError(
                "Missing onboarding answers for: " + ", ".join(sorted(missing))
            )

        existing_node_ids = self.repository.get_app_state_json(
            self.APP_STATE_BOOTSTRAP_ONBOARDING_NODE_IDS
        ) or {}
        next_node_ids: dict[str, str] = {}
        for section in self._bootstrap_onboarding_sections(cleaned_answers):
            existing_node_id = existing_node_ids.get(section["section_id"])
            node_id = self._upsert_bootstrap_profile_node(
                node_id=existing_node_id,
                name=section["name"],
                content=section["content"],
                metadata=section["metadata"],
            )
            next_node_ids[section["section_id"]] = node_id

        self.repository.set_app_state_json(
            self.APP_STATE_BOOTSTRAP_ONBOARDING_ANSWERS,
            cleaned_answers,
        )
        self.repository.set_app_state_json(
            self.APP_STATE_BOOTSTRAP_ONBOARDING_NODE_IDS,
            next_node_ids,
        )
        self.repository.set_app_state_value(
            self.APP_STATE_BOOTSTRAP_ONBOARDING_COMPLETED,
            "1",
        )
        self.compile_memory_snapshot()
        self.build_host_bootstrap(
            output_dir=output_dir,
            host_kind=normalized_host_kind,
            install=False,
        )
        return self.get_host_bootstrap_status(
            host_kind=normalized_host_kind,
            output_dir=output_dir,
        )

    def build_host_bootstrap(
        self,
        output_dir: Path | None = None,
        *,
        host_kind: str = "generic",
        install: bool = False,
    ) -> BootstrapBundle:
        self.initialize()
        normalized_host_kind = self._normalize_host_kind(host_kind)
        memory_path = self.compile_memory_snapshot(
            output_path=self._host_memory_output_path(host_kind=normalized_host_kind)
        )
        self._register_host_memory_target(
            host_kind=normalized_host_kind,
            memory_output_path=memory_path,
        )
        bootstrap_dir = output_dir or self.settings.bootstrap_dir
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths = self._bootstrap_artifact_paths(
            bootstrap_dir=bootstrap_dir,
            host_kind=normalized_host_kind,
        )
        status = self.get_host_bootstrap_status(
            host_kind=normalized_host_kind,
            output_dir=bootstrap_dir,
        )

        prompt = self._bootstrap_prompt(memory_path, host_kind=normalized_host_kind)
        artifact_paths["bootstrap_prompt_path"].write_text(prompt, encoding="utf-8")
        system_prompt_block = self._host_system_prompt(
            memory_path,
            host_kind=normalized_host_kind,
        )
        artifact_paths["system_prompt_path"].write_text(
            system_prompt_block,
            encoding="utf-8",
        )
        artifact_paths["onboarding_path"].write_text(
            json.dumps(
                [question.model_dump() for question in self._bootstrap_onboarding_questions()],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        host_instruction_path, host_project_config_path = self._host_mount_targets(
            normalized_host_kind
        )
        mount_manifest = {
            "host_kind": normalized_host_kind,
            "memory_path": str(memory_path),
            "mcp_command": "cognitiveos-mcp",
            "mcp_args": self._host_mcp_args(host_kind=normalized_host_kind),
            "bootstrap_status": {
                "first_startup": status.first_startup,
                "needs_onboarding": status.needs_onboarding,
                "installed": status.installed,
                "needs_mount": status.needs_mount,
            },
            "onboarding_questions_path": str(artifact_paths["onboarding_path"]),
            "system_prompt_path": str(artifact_paths["system_prompt_path"]),
            "host_instruction_target": (
                str(host_instruction_path) if host_instruction_path else None
            ),
            "host_project_config_target": (
                str(host_project_config_path) if host_project_config_path else None
            ),
            "dream_policy": {
                "event_threshold": self.settings.dream_event_threshold,
                "max_age_hours": self.settings.dream_max_age_hours,
                "age_min_event_count": self.settings.dream_age_min_event_count,
                "chat_required_for_auto_trigger": True,
                "compaction_priority": [
                    "chat_provider",
                    "host_agent",
                    "heuristic",
                ],
                "subagent_hint": (
                    "If the host supports subagents/background agents, "
                    "delegate dream as a background task and resolve host compaction "
                    "tasks when a dream run returns pending compactions."
                ),
            },
        }
        artifact_paths["mount_manifest_path"].write_text(
            json.dumps(mount_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        mcp_config = {
            "mcpServers": {
                "cognitiveos": {
                    "command": "cognitiveos-mcp",
                    "args": self._host_mcp_args(),
                }
            }
        }
        artifact_paths["mcp_config_path"].write_text(
            json.dumps(mcp_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        installed = False
        if install:
            installed = self._install_host_mount(
                host_kind=normalized_host_kind,
                system_prompt_block=system_prompt_block,
            )

        final_status = self.get_host_bootstrap_status(
            host_kind=normalized_host_kind,
            output_dir=bootstrap_dir,
        )

        return BootstrapBundle(
            host_kind=normalized_host_kind,
            memory_path=str(memory_path),
            bootstrap_prompt_path=str(artifact_paths["bootstrap_prompt_path"]),
            system_prompt_path=str(artifact_paths["system_prompt_path"]),
            mount_manifest_path=str(artifact_paths["mount_manifest_path"]),
            mcp_config_path=str(artifact_paths["mcp_config_path"]),
            onboarding_path=str(artifact_paths["onboarding_path"]),
            host_instruction_path=(
                str(host_instruction_path) if host_instruction_path else None
            ),
            host_project_config_path=(
                str(host_project_config_path) if host_project_config_path else None
            ),
            installed=installed or final_status.installed,
            status=final_status,
        )

    def reindex_embeddings(self) -> dict[str, Any]:
        self.initialize()
        if self.embedding_provider is None:
            return {
                "status": "skipped",
                "reason": "No embedding provider configured.",
            }

        all_nodes = self.repository.list_all_nodes()
        if not all_nodes:
            return {
                "status": "success",
                "reindexed_nodes": 0,
            }

        batch_size = 16
        reindexed = 0
        for index in range(0, len(all_nodes), batch_size):
            batch = all_nodes[index : index + batch_size]
            embeddings = self.embedding_provider.embed(
                [self._embedding_input_for_node(node) for node in batch]
            )
            for node, embedding in zip(batch, embeddings, strict=True):
                self.repository.update_node_embedding(node.id, embedding)
                reindexed += 1
        return {
            "status": "success",
            "reindexed_nodes": reindexed,
            "vector_count": self.repository.get_vector_count(),
            "embedding_dimension": self.repository.get_embedding_dimension(),
        }

    def doctor(self, *, check_providers: bool = False) -> dict[str, Any]:
        self.initialize()
        governance = self.governance.apply_relationship_governance()
        with open_connection(self.settings.db_path) as connection:
            sqlite_version = connection.execute("select sqlite_version()").fetchone()[0]
            vec_version = connection.execute("select vec_version()").fetchone()[0]

        provider_checks = None
        if check_providers:
            provider_checks = self.test_providers()

        return {
            "status": "success",
            "db_path": str(self.settings.db_path),
            "memory_output_path": str(self.settings.memory_output_path),
            "bootstrap_dir": str(self.settings.bootstrap_dir),
            "sqlite_version": sqlite_version,
            "sqlite_vec_version": vec_version,
            "node_count": self.repository.get_node_count(),
            "vector_count": self.repository.get_vector_count(),
            "embedding_dimension": self.repository.get_embedding_dimension(),
            "relationship_governance": governance,
            "dream_status": self.get_dream_status().model_dump(),
            "providers": {
                "embedding": {
                    "configured": self.embedding_provider is not None,
                    "type": self.settings.embedding_provider_type,
                    "model": self.settings.embedding_model_name,
                },
                "chat": {
                    "configured": self.chat_provider is not None,
                    "type": self.settings.chat_provider_type,
                    "model": self.settings.chat_model_name,
                },
            },
            "provider_checks": provider_checks,
        }

    def test_providers(self) -> dict[str, Any]:
        sample = "CognitiveOS provider smoke test."
        results: dict[str, Any] = {}

        if self.embedding_provider is None:
            results["embedding"] = {"status": "skipped", "reason": "Not configured."}
        else:
            try:
                embedding = self.embedding_provider.embed([sample])[0]
                results["embedding"] = {
                    "status": "success",
                    "dimensions": len(embedding),
                }
            except Exception as exc:
                results["embedding"] = {
                    "status": "error",
                    "error": str(exc),
                }

        if self.chat_provider is None:
            results["chat"] = {"status": "skipped", "reason": "Not configured."}
        else:
            try:
                summary = self.chat_provider.summarize(sample)
                results["chat"] = {
                    "status": "success",
                    "preview": summary[:120],
                }
            except Exception as exc:
                results["chat"] = {
                    "status": "error",
                    "error": str(exc),
                }

        return results

    def get_dream_status(self) -> DreamStatus:
        self.initialize()
        last_dream = self.repository.get_last_completed_dream_run()
        last_completed_at = last_dream["completed_at"] if last_dream is not None else None
        first_event_at = self.repository.get_first_memory_event_time()
        event_count = self.repository.count_memory_events_since(last_completed_at)
        reasons: list[str] = []

        due_events = event_count >= self.settings.dream_event_threshold
        if due_events:
            reasons.append(
                f"{event_count} new events accumulated since the last dream "
                f"(threshold {self.settings.dream_event_threshold})."
            )

        reference_time = last_completed_at or first_event_at
        hours_since_reference: float | None = None
        due_age = False
        reminder = None
        if reference_time is not None:
            reference_dt = datetime.fromisoformat(
                reference_time.replace(" ", "T")
            ).replace(tzinfo=UTC)
            hours_since_reference = (datetime.now(UTC) - reference_dt).total_seconds() / 3600
            age_window_reached = (
                last_completed_at is not None
                and hours_since_reference >= self.settings.dream_max_age_hours
            )
            if (
                age_window_reached
                and event_count >= self.settings.dream_age_min_event_count
            ):
                due_age = True
                reasons.append(
                    f"{hours_since_reference:.1f} hours have passed since the last dream "
                    f"(threshold {self.settings.dream_max_age_hours}h) with "
                    f"{event_count} new events "
                    f"(threshold {self.settings.dream_age_min_event_count})."
                )
            elif age_window_reached and 0 < event_count < self.settings.dream_age_min_event_count:
                reminder = (
                    f"{hours_since_reference:.1f} hours have passed since the last dream, "
                    f"but only {event_count} new events accumulated. "
                    f"Auto-dream waits for at least "
                    f"{self.settings.dream_age_min_event_count} new events."
                )

        due = due_events or due_age
        if due and self.chat_provider is None:
            reminder = (
                "Dream is due but no chat model is configured. Run `dream` to prepare "
                "host compaction tasks, then let the host agent submit compressed clusters. "
                "If the host supports background subagents, delegate dream as a background task."
            )
        return DreamStatus(
            due=due,
            event_count_since_last_dream=event_count,
            last_dream_completed_at=last_completed_at,
            hours_since_last_dream_or_first_event=(
                round(hours_since_reference, 2) if hours_since_reference is not None else None
            ),
            reasons=reasons,
            reminder=reminder,
        )

    @staticmethod
    def _infer_node_type(
        fallback: str,
    ) -> str:
        return fallback

    @staticmethod
    def _default_node_durability(
        node_type: str,
    ) -> str:
        if node_type in {"source_document", "source_collection"}:
            return "durable"
        return "working"

    @staticmethod
    def _merge_tags(*tag_groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in tag_groups:
            for tag in group:
                cleaned = tag.strip()
                if not cleaned:
                    continue
                lowered = cleaned.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                merged.append(cleaned)
        return merged

    def _infer_canonical_profile_section(
        self,
        *,
        tags: list[str],
        name: str | None,
        content: str,
    ) -> str | None:
        lowered_tags = {tag.strip().lower() for tag in tags if tag.strip()}
        lowered_name = (name or "").strip().lower()
        lowered_content = content.lower()

        if (
            "identity" in lowered_tags
            or "identity" in lowered_name
            or any(
                marker in lowered_content
                for marker in (
                    "display_name=",
                    "preferred name:",
                    "full name:",
                    "english_name=",
                    "work_email=",
                    "personal_email=",
                    "timezone=",
                )
            )
        ):
            return "identity"

        if (
            "workspace" in lowered_tags
            or "workspace" in lowered_name
            or "work context" in lowered_name
            or any(
                marker in lowered_content
                for marker in (
                    "workspace goal:",
                    "workspace_goal=",
                    "primary repo path:",
                    "primary_repo_path=",
                    "common work types:",
                    "common_work_types=",
                    "filesystem boundary preference:",
                    "filesystem_boundary_preference=",
                )
            )
        ):
            return "workspace"

        if (
            "engineering" in lowered_tags
            or "engineering" in lowered_name
            or any(
                marker in lowered_content
                for marker in (
                    "primary_stack=",
                    "common_platforms=",
                    "coding_preferences=",
                    "collaboration_preferences=",
                    "avoidances=",
                )
            )
        ):
            return "engineering"

        if (
            "communication" in lowered_tags
            or "preferences" in lowered_tags
            or "communication" in lowered_name
            or any(
                marker in lowered_content
                for marker in (
                    "default language:",
                    "preferred_language=",
                    "response style:",
                    "response_style=",
                    "preferred_response_style=",
                    "output preference:",
                    "output_preferences=",
                    "command preference:",
                )
            )
        ):
            return "communication"

        return None

    def _upsert_canonical_profile_memory(
        self,
        *,
        section: str,
        content: str,
        tags: list[str],
        durability: str | None,
    ) -> Receipt:
        existing_node = self._find_canonical_profile_node(section)
        merged_tags = self._merge_tags(
            self.CANONICAL_PROFILE_SECTION_TAGS.get(section, ["profile"]),
            tags,
        )
        resolved_durability = (
            self._normalize_durability(durability) if durability is not None else "pinned"
        )
        if existing_node is not None:
            merged_content = self._merge_canonical_profile_content(
                section=section,
                existing_content=existing_node.content,
                incoming_content=content,
            )
            next_metadata = self._canonical_profile_metadata(
                section=section,
                existing_metadata=existing_node.metadata,
            )
            audit_log_id = self.repository.overwrite_node(
                NodeRecord(
                    id=existing_node.id,
                    name=self.CANONICAL_PROFILE_SECTION_NAMES.get(section, existing_node.name),
                    description=self._summarize(merged_content),
                    content=merged_content,
                    embedding=self._embed_content(merged_content),
                    tags=merged_tags,
                    metadata=next_metadata,
                    node_type="memory",
                    durability=resolved_durability,
                    last_reinforced_at=existing_node.last_reinforced_at,
                ),
                actor=self.settings.default_actor,
                action_type="profile_upsert",
            )
            self._schedule_semantic_neighbor_refresh(existing_node.id)
            return Receipt(
                status="success",
                action_taken="updated",
                node_id=existing_node.id,
                audit_log_id=audit_log_id,
            )

        merged_content = self._merge_canonical_profile_content(
            section=section,
            existing_content="",
            incoming_content=content,
        )
        node = NodeRecord(
            id=f"node_{uuid4().hex}",
            name=self.CANONICAL_PROFILE_SECTION_NAMES.get(section),
            description=self._summarize(merged_content),
            content=merged_content,
            embedding=self._embed_content(merged_content),
            tags=merged_tags,
            metadata=self._canonical_profile_metadata(section=section),
            node_type="memory",
            durability=resolved_durability,
        )
        audit_log_id = self.repository.create_node(
            node,
            actor=self.settings.default_actor,
            action_type="profile_create",
        )
        self._schedule_semantic_neighbor_refresh(node.id)
        return Receipt(
            status="success",
            action_taken="created",
            node_id=node.id,
            audit_log_id=audit_log_id,
        )

    def _find_canonical_profile_node(self, section: str) -> NodeRecord | None:
        name_fallback = self.CANONICAL_PROFILE_SECTION_NAMES.get(section, "")
        for node in self.repository.list_all_nodes():
            if node.node_type != "memory":
                continue
            if metadata_profile_section(node.metadata) == section:
                return node
            if node.name == name_fallback:
                return node
        return None

    def _canonical_profile_metadata(
        self,
        *,
        section: str,
        existing_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = dict(existing_metadata or {})
        profile_payload = (
            dict(metadata.get("profile"))
            if isinstance(metadata.get("profile"), dict)
            else {}
        )
        bootstrap_section = section if section in {"identity", "communication", "workspace"} else None
        metadata["profile"] = {
            **profile_payload,
            "kind": "system" if bootstrap_section else "user",
            "bootstrap": bool(bootstrap_section),
            "canonical": True,
            "section": section,
        }
        if bootstrap_section:
            metadata["bootstrap_section"] = bootstrap_section
        return metadata

    def _merge_canonical_profile_content(
        self,
        *,
        section: str,
        existing_content: str,
        incoming_content: str,
    ) -> str:
        merged_lines: dict[str, str] = {}
        extra_lines: list[str] = []
        for source in (existing_content, incoming_content):
            for raw_line in source.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                key, value = self._split_profile_line(line)
                if key is None:
                    if line not in extra_lines:
                        extra_lines.append(line)
                    continue
                canonical_key = self._canonical_profile_field_key(section=section, key=key)
                merged_lines[canonical_key] = value

        ordered_fields = self._canonical_profile_field_order(section)
        rendered: list[str] = []
        for field_key in ordered_fields:
            value = merged_lines.pop(field_key, None)
            if value is None:
                continue
            rendered.append(f"{self._canonical_profile_field_label(section=section, key=field_key)}: {value}")
        for field_key, value in merged_lines.items():
            rendered.append(
                f"{self._canonical_profile_field_label(section=section, key=field_key)}: {value}"
            )
        rendered.extend(extra_lines)
        return "\n".join(rendered)

    @staticmethod
    def _split_profile_line(line: str) -> tuple[str | None, str | None]:
        delimiter = ":" if ":" in line else "=" if "=" in line else None
        if delimiter is None:
            return None, None
        key, value = line.split(delimiter, 1)
        normalized_key = " ".join(key.strip().lower().replace("_", " ").split())
        clean_value = value.strip()
        if not normalized_key or not clean_value:
            return None, None
        return normalized_key, clean_value

    @staticmethod
    def _canonical_profile_field_order(section: str) -> list[str]:
        orders = {
            "identity": [
                "preferred name",
                "full name",
                "english name",
                "role",
                "team",
                "employer",
                "location",
                "timezone",
                "locale encoding",
                "work email",
                "personal email",
            ],
            "communication": [
                "default language",
                "response style",
                "output preference",
                "command preference",
            ],
            "workspace": [
                "workspace goal",
                "primary repo path",
                "common work types",
                "common frameworks",
                "filesystem boundary preference",
            ],
            "engineering": [
                "primary stack",
                "common platforms",
                "coding preferences",
                "collaboration preferences",
                "avoidances",
            ],
        }
        return orders.get(section, [])

    @staticmethod
    def _canonical_profile_field_key(*, section: str, key: str) -> str:
        aliases = {
            "identity": {
                "display name": "full name",
                "preferred name": "preferred name",
                "full name": "full name",
                "english name": "english name",
                "role": "role",
                "role or team": "team",
                "team": "team",
                "employer": "employer",
                "location": "location",
                "timezone": "timezone",
                "locale encoding": "locale encoding",
                "work email": "work email",
                "personal email": "personal email",
            },
            "communication": {
                "default language": "default language",
                "preferred language": "default language",
                "response style": "response style",
                "preferred response style": "response style",
                "output preference": "output preference",
                "output preferences": "output preference",
                "command preference": "command preference",
            },
            "workspace": {
                "workspace goal": "workspace goal",
                "workspace context": "workspace goal",
                "primary repo path": "primary repo path",
                "common work types": "common work types",
                "common frameworks": "common frameworks",
                "filesystem boundary preference": "filesystem boundary preference",
            },
            "engineering": {
                "primary stack": "primary stack",
                "common platforms": "common platforms",
                "coding preferences": "coding preferences",
                "collaboration preferences": "collaboration preferences",
                "avoidances": "avoidances",
            },
        }
        return aliases.get(section, {}).get(key, key)

    @staticmethod
    def _canonical_profile_field_label(*, section: str, key: str) -> str:
        labels = {
            "preferred name": "Preferred name",
            "full name": "Full name",
            "english name": "English name",
            "role": "Role",
            "team": "Team",
            "employer": "Employer",
            "location": "Location",
            "timezone": "Timezone",
            "locale encoding": "Locale encoding",
            "work email": "Work email",
            "personal email": "Personal email",
            "default language": "Default language",
            "response style": "Response style",
            "output preference": "Output preference",
            "command preference": "Command preference",
            "workspace goal": "Workspace goal",
            "primary repo path": "Primary repo path",
            "common work types": "Common work types",
            "common frameworks": "Common frameworks",
            "filesystem boundary preference": "Filesystem boundary preference",
            "primary stack": "Primary stack",
            "common platforms": "Common platforms",
            "coding preferences": "Coding preferences",
            "collaboration preferences": "Collaboration preferences",
            "avoidances": "Avoidances",
        }
        return labels.get(key, key.title())

    def _resolve_node_durability(self, *, durability: str | None, node_type: str) -> str:
        if durability is not None:
            return self._normalize_durability(durability)
        return self._default_node_durability(node_type=node_type)

    def _normalize_durability(self, durability: str) -> str:
        normalized = durability.strip().lower()
        if normalized not in self.ALLOWED_DURABILITY_VALUES:
            raise InvalidPayloadError(
                "Invalid durability value. Supported values: "
                f"{sorted(self.ALLOWED_DURABILITY_VALUES)}"
            )
        return normalized

    def _summarize(self, content: str) -> str:
        if len(content) <= 500:
            return content
        if self.chat_provider is not None:
            summary = self.chat_provider.summarize(content).strip()
            if summary:
                return summary[:500]
        return content[:497].rstrip() + "..."

    def _add_folder_node(
        self,
        *,
        payload: str,
        tags: list[str],
        durability: str | None,
        force: bool,
        name: str | None,
    ) -> Receipt:
        profile = inspect_folder_collection(payload, name=name)
        self._validate_node_content_limit(profile.content)

        if not force:
            conflicts = self._find_collection_conflicts(profile.normalized_path)
            if conflicts:
                return Receipt(
                    status="blocked",
                    action_taken="none",
                    reason="A source collection with matching source metadata already exists.",
                    conflicting_nodes=[
                        ConflictNode(id=node.id, name=node.name, similarity=1.0)
                        for node in conflicts[:3]
                    ],
                    suggestion="Re-run 'add' with force=true to store another root anchor.",
                )

        description = self._generate_collection_description(
            content=profile.content,
            metadata=profile.metadata,
            name=profile.display_name,
        )
        embedding = self._embed_content(
            self._compose_collection_embedding_text(
                description=description,
                tags=tags,
                metadata=profile.metadata,
                content=profile.content,
            )
        )
        node = NodeRecord(
            id=f"node_{uuid4().hex}",
            name=profile.display_name,
            description=description,
            content=profile.content,
            embedding=embedding,
            tags=tags,
            metadata=profile.metadata,
            node_type="source_collection",
            durability=self._resolve_node_durability(
                durability=durability,
                node_type="source_collection",
            ),
        )
        action_type = "force_add" if force else "create"
        self.repository.create_node(
            node,
            actor=self.settings.default_actor,
            action_type=action_type,
        )
        self._schedule_semantic_neighbor_refresh(node.id)
        return Receipt(status="success", action_taken="created", node_id=node.id)

    def _find_collection_conflicts(self, normalized_path: str) -> list[NodeRecord]:
        conflicts: list[NodeRecord] = []
        for node in self.repository.list_nodes_by_type(node_type="source_collection"):
            if metadata_source_ref(node.metadata) == normalized_path:
                conflicts.append(node)
        return conflicts

    def _build_document_profile(
        self,
        *,
        content: str,
        tags: list[str],
        source_metadata: dict[str, Any],
        name: str | None,
    ) -> dict[str, Any]:
        generated_description = self._generate_document_description(content=content, name=name)
        generated_tags = self._generate_document_tags(
            content=content,
            description=generated_description,
            source_metadata=source_metadata,
            name=name,
        )
        merged_tags = self._merge_tags(tags, generated_tags)
        embedding_input = self._compose_document_embedding_text(
            description=generated_description,
            tags=merged_tags,
        )
        return {
            "description": generated_description,
            "tags": merged_tags,
            "generated_tags": generated_tags,
            "embedding_input": embedding_input,
        }

    def _generate_document_description(self, *, content: str, name: str | None) -> str:
        system_prompt = (
            "You create retrieval-oriented document descriptions. "
            "Return a compact factual description in 800 characters or fewer. "
            "Do not use markdown. Do not add bullets. "
            "Focus on the document's purpose, major topics, and likely retrieval intent."
        )
        user_prompt = self._build_document_profile_prompt(
            content=content,
            name=name,
            instruction="Return only the description text.",
        )
        try:
            completion = self._complete_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            cleaned = " ".join(completion.split()).strip()
            if cleaned:
                return cleaned[: self.DOCUMENT_DESCRIPTION_MAX_CHARS]
        except (NotImplementedError, RuntimeError, ValueError):
            pass

        if self.chat_provider is not None:
            try:
                summary = self.chat_provider.summarize(content).strip()
                if summary:
                    return summary[: self.DOCUMENT_DESCRIPTION_MAX_CHARS]
            except Exception:
                pass
        fallback = " ".join(content.split()).strip()
        if len(fallback) <= self.DOCUMENT_DESCRIPTION_MAX_CHARS:
            return fallback
        return fallback[: self.DOCUMENT_DESCRIPTION_MAX_CHARS - 3].rstrip() + "..."

    def _generate_document_tags(
        self,
        *,
        content: str,
        description: str,
        source_metadata: dict[str, Any],
        name: str | None,
    ) -> list[str]:
        system_prompt = (
            "You create retrieval tags for ingested documents. "
            "Return strict JSON with a single key: "
            '{"tags":["tag one","tag two"]}. '
            "Return 3 to 8 short tags, each 1 to 3 words, lowercase when natural, "
            "deduplicated, and retrieval-friendly. "
            "Do not include ids, punctuation-heavy labels, or generic tags like document."
        )
        prompt_parts = [
            self._build_document_profile_prompt(
                content=content,
                name=name,
                instruction="Generate retrieval tags for this document.",
            ),
            "",
            f"Current description:\n{description}",
        ]
        suffix = source_metadata.get("source", {}).get("suffix")
        if isinstance(suffix, str) and suffix:
            prompt_parts.extend(["", f"File suffix: {suffix}"])
        try:
            completion = self._complete_chat(
                system_prompt=system_prompt,
                user_prompt="\n".join(prompt_parts).strip(),
            )
        except (NotImplementedError, RuntimeError, ValueError):
            return []
        return self._parse_generated_tags(completion)

    def _complete_chat(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.chat_provider is None:
            raise NotImplementedError("Chat provider is not configured.")
        complete = getattr(self.chat_provider, "complete", None)
        if not callable(complete):
            raise NotImplementedError("Chat provider does not support general completion.")
        result = complete(system_prompt=system_prompt, user_prompt=user_prompt)
        if not isinstance(result, str):
            raise ValueError("Chat completion did not return text.")
        stripped = result.strip()
        if not stripped:
            raise ValueError("Chat completion returned empty text.")
        return stripped

    def _build_document_profile_prompt(
        self,
        *,
        content: str,
        name: str | None,
        instruction: str,
    ) -> str:
        compact = content.strip()
        if len(compact) > self.DOCUMENT_PROFILE_PROMPT_CHARS:
            head = compact[:12000].rstrip()
            tail = compact[-4000:].lstrip()
            compact = (
                f"{head}\n\n[content omitted for length]\n\n{tail}"
            )
        parts = [instruction]
        if name:
            parts.extend(["", f"Document name: {name}"])
        parts.extend(["", "Document content:", compact])
        return "\n".join(parts)

    @classmethod
    def _parse_generated_tags(cls, payload: str) -> list[str]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return []
        raw_tags = data.get("tags")
        if not isinstance(raw_tags, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in raw_tags:
            if not isinstance(item, str):
                continue
            cleaned = " ".join(item.split()).strip(" ,;:.")
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            normalized.append(cleaned)
            if len(normalized) >= cls.DOCUMENT_TAG_LIMIT:
                break
        return normalized

    @staticmethod
    def _compose_document_embedding_text(*, description: str, tags: list[str]) -> str:
        if tags:
            return f"{description}\n\nTags: {', '.join(tags)}"
        return description

    def _generate_collection_description(
        self,
        *,
        content: str,
        metadata: dict[str, Any],
        name: str | None,
    ) -> str:
        system_prompt = (
            "You create retrieval-oriented descriptions for folder source anchors. "
            "Return one compact factual sentence in 800 characters or fewer. "
            "Do not use markdown. Focus on what the folder contains and when it should be revisited."
        )
        hint_text = collection_hint_text(metadata, fallback_content=content)
        prompt_parts = [f"Folder name: {name or 'Unnamed folder'}", hint_text]
        source_ref = metadata.get("source", {}).get("ref") if isinstance(metadata, dict) else None
        if isinstance(source_ref, str) and source_ref:
            prompt_parts.append(f"Folder path: {source_ref}")
        try:
            completion = self._complete_chat(
                system_prompt=system_prompt,
                user_prompt="\n".join(prompt_parts),
            )
            cleaned = " ".join(completion.split()).strip()
            if cleaned:
                return cleaned[: self.DOCUMENT_DESCRIPTION_MAX_CHARS]
        except (NotImplementedError, RuntimeError, ValueError):
            pass
        return self._fallback_collection_description(name=name, metadata=metadata)

    def _fallback_collection_description(
        self,
        *,
        name: str | None,
        metadata: dict[str, Any],
    ) -> str:
        folder_name = name or "Unnamed folder"
        collection = metadata.get("collection", {}) if isinstance(metadata, dict) else {}
        collection_class = collection.get("class") or "workspace_bundle"
        sample_entries = [
            entry
            for entry in collection.get("sample_entries", [])
            if isinstance(entry, str) and entry.strip()
        ]
        important_markers = [
            marker
            for marker in collection.get("important_markers", [])
            if isinstance(marker, str) and marker.strip()
        ]
        if collection_class == "repository":
            marker_text = ", ".join(important_markers[:4]) if important_markers else "no major markers"
            repo_hint = repository_hint(important_markers) or "software project"
            return (
                f"Repository rooted at {folder_name}; top-level markers include {marker_text}; "
                f"likely {repo_hint.lower()}."
            )[: self.DOCUMENT_DESCRIPTION_MAX_CHARS]
        if collection_class == "media_collection":
            media_summary = self._format_collection_count_summary(collection.get("file_type_counts", {}))
            return (
                f"Media collection rooted at {folder_name}; contains mostly {media_summary}; "
                "use this source anchor when revisiting photos, videos, or media files."
            )[: self.DOCUMENT_DESCRIPTION_MAX_CHARS]
        if collection_class == "document_collection":
            doc_summary = self._format_collection_count_summary(collection.get("file_type_counts", {}))
            return (
                f"Document collection rooted at {folder_name}; contains mostly {doc_summary}; "
                "use this source anchor when revisiting documents or notes."
            )[: self.DOCUMENT_DESCRIPTION_MAX_CHARS]
        sample_text = ", ".join(sample_entries[:4]) if sample_entries else "mixed top-level items"
        return (
            f"Workspace bundle rooted at {folder_name}; mixed top-level entries include {sample_text}; "
            "use this source anchor for future re-entry."
        )[: self.DOCUMENT_DESCRIPTION_MAX_CHARS]

    @staticmethod
    def _format_collection_count_summary(file_type_counts: Any) -> str:
        if not isinstance(file_type_counts, dict) or not file_type_counts:
            return "mixed files"
        ordered = sorted(
            (
                (str(file_type), int(count))
                for file_type, count in file_type_counts.items()
                if isinstance(file_type, str) and isinstance(count, int | float)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not ordered:
            return "mixed files"
        return ", ".join(f"{file_type}={count}" for file_type, count in ordered[:3])

    def _compose_collection_embedding_text(
        self,
        *,
        description: str,
        tags: list[str],
        metadata: dict[str, Any],
        content: str,
    ) -> str:
        parts = [description]
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        hint_text = collection_hint_text(metadata, fallback_content=content)
        if hint_text:
            parts.append(hint_text)
        return "\n\n".join(parts)

    def _embedding_input_for_node(self, node: NodeRecord) -> str:
        if node.node_type == "source_document":
            return self._compose_document_embedding_text(
                description=node.description,
                tags=node.tags,
            )
        if node.node_type == "source_collection":
            return self._compose_collection_embedding_text(
                description=node.description,
                tags=node.tags,
                metadata=node.metadata,
                content=node.content,
            )
        return node.content

    @staticmethod
    def _split_document_tags(node: NodeRecord) -> tuple[list[str], list[str]]:
        generated_tags = []
        if isinstance(node.metadata, dict):
            generated_tags = list(
                node.metadata.get("document_profile", {}).get("generated_tags", [])
            )
        generated_lc = {tag.strip().lower() for tag in generated_tags if isinstance(tag, str)}
        user_tags = [
            tag for tag in node.tags if tag.strip().lower() not in generated_lc
        ]
        return user_tags, generated_tags

    @staticmethod
    def _source_hash(content: str) -> str:
        import hashlib

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _embed_content(self, content: str) -> list[float] | None:
        if self.embedding_provider is None:
            return None
        return self.embedding_provider.embed([content])[0]

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((perf_counter() - started_at) * 1000, 3)

    def _validate_node_content_limit(self, content: str) -> None:
        if len(content) > self.settings.max_node_content_chars:
            raise InvalidPayloadError(
                "Content exceeds the effective node content limit. "
                f"Current limit: {self.settings.max_node_content_chars} characters."
            )

    def _ensure_embeddings(self) -> None:
        if self.embedding_provider is None:
            return
        missing_nodes = self.repository.list_nodes_missing_embeddings()
        if not missing_nodes:
            return

        batch_size = 16
        for index in range(0, len(missing_nodes), batch_size):
            batch = missing_nodes[index : index + batch_size]
            embeddings = self.embedding_provider.embed(
                [self._embedding_input_for_node(node) for node in batch]
            )
            for node, embedding in zip(batch, embeddings, strict=True):
                self.repository.update_node_embedding(node.id, embedding)

    def _semantic_search_matches(self, *, query: str, top_k: int) -> list[tuple[str, float]]:
        query_embedding = self._embed_content(query)
        if query_embedding is None:
            return []
        return self.repository.search_semantic_matches(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def _hybrid_rank(
        self,
        *,
        semantic_matches: list[tuple[str, float]],
        keyword_matches: list[tuple[str, float]],
        top_k: int,
    ) -> tuple[list[str], dict[str, dict[str, float]]]:
        scores: dict[str, dict[str, float]] = {}

        semantic_rrf = self._rank_to_rrf(semantic_matches)
        keyword_rrf = self._rank_to_rrf(keyword_matches)

        all_ids = list(
            dict.fromkeys(
                [node_id for node_id, _ in semantic_matches]
                + [node_id for node_id, _ in keyword_matches]
            )
        )

        if not all_ids:
            return [], {}

        semantic_weight = self.settings.hybrid_semantic_weight if semantic_matches else 0.0
        keyword_weight = self.settings.hybrid_keyword_weight if keyword_matches else 0.0
        total_weight = semantic_weight + keyword_weight
        if total_weight == 0:
            semantic_weight = 0.5
            keyword_weight = 0.5
            total_weight = 1.0

        for node_id in all_ids:
            semantic_score = semantic_rrf.get(node_id, 0.0)
            keyword_score = keyword_rrf.get(node_id, 0.0)
            total_score = (
                (semantic_weight / total_weight) * semantic_score
                + (keyword_weight / total_weight) * keyword_score
            )
            scores[node_id] = {
                "semantic_score": round(semantic_score, 6),
                "keyword_score": round(keyword_score, 6),
                "score": round(total_score, 6),
            }

        ordered_ids = [
            node_id
            for node_id, _score in sorted(
                scores.items(),
                key=lambda item: (
                    item[1]["score"],
                    item[1]["semantic_score"],
                    item[1]["keyword_score"],
                ),
                reverse=True,
            )
        ]
        return ordered_ids[:top_k], scores

    @staticmethod
    def _rank_to_rrf(matches: list[tuple[str, float]], *, k: int = 60) -> dict[str, float]:
        return {
            node_id: 1.0 / (k + rank)
            for rank, (node_id, _raw_score) in enumerate(matches, start=1)
        }

    def _find_similarity_conflicts(self, embedding: list[float]) -> list[dict[str, Any]]:
        self._ensure_embeddings()
        conflicts: list[dict[str, Any]] = []
        for node_id, _distance in self.repository.search_semantic_matches(
            query_embedding=embedding,
            top_k=10,
        ):
            node = self.repository.get_node(node_id)
            if node.embedding is None:
                continue
            similarity = cosine_similarity(embedding, node.embedding)
            if similarity >= self.settings.similarity_threshold:
                conflicts.append(
                    {
                        "id": node.id,
                        "name": node.name,
                        "similarity": round(similarity, 4),
                    }
                )
        conflicts.sort(key=lambda item: item["similarity"], reverse=True)
        return conflicts[:3]

    def _maybe_handle_dream(self, operation_name: str) -> list[str]:
        status = self.get_dream_status()
        notices: list[str] = []
        if not status.due:
            if status.reminder:
                notices.append(f"Dream deferred during {operation_name}: {status.reminder}")
            self.last_notices = notices
            return notices

        active_run = self.repository.get_active_dream_run()
        if active_run is not None:
            notices.append(
                f"Dream already {active_run['status']} during {operation_name}: "
                f"{active_run['run_id']}."
            )
            self.last_notices = notices
            return notices

        reason = "; ".join(status.reasons) or f"Dream is due during {operation_name}."
        if self.chat_provider is not None:
            result = self.queue_background_dream(
                trigger_reason=f"auto:{operation_name}",
                auto_triggered=True,
            )
            notices.append(
                f"Dream queued in background during {operation_name}: {reason} "
                f"Run id {result.run_id}."
            )
        else:
            notices.append(
                f"Dream reminder during {operation_name}: {reason} "
                "No chat model is configured, so auto-dream is deferred. "
                "Run `dream` to prepare host compaction tasks. "
                "If the host supports subagents, delegate dream in the background."
            )
        self.last_notices = notices
        return notices

    @staticmethod
    def _dream_task_from_row(row: Any) -> DreamCompactionTask:
        source_nodes = json.loads(row["source_nodes_json"] or "[]")
        return DreamCompactionTask(
            task_id=row["task_id"],
            run_id=row["run_id"],
            status=row["status"],
            requested_backend=row["requested_backend"],
            fallback_backend=row["fallback_backend"],
            reason=row["reason"],
            suggested_title=row["suggested_title"],
            suggested_description=row["suggested_description"],
            prepared_content=row["prepared_content"],
            prompt=row["prompt"],
            source_nodes=source_nodes,
        )

    def _finalize_compaction_resolution(
        self,
        *,
        task_id: str,
        run_id: str,
        resolution_backend: str,
        node_id: str,
    ) -> DreamCompactionResolution:
        self.repository.complete_dream_compaction_task(
            task_id,
            resolved_node_id=node_id,
            resolution_backend=resolution_backend,
        )

        run_row = self.repository.get_dream_run(run_id)
        if run_row is None:
            raise InvalidPayloadError(f"Dream run '{run_id}' was not found.")

        remaining_tasks = self.repository.count_pending_dream_compaction_tasks(run_id)
        notes = json.loads(run_row["notes_json"] or "[]")
        note = (
            f"Dream compaction task {task_id} resolved via "
            f"{resolution_backend} into {node_id}."
        )
        notes.append(note)
        memory_path = str(self.compile_memory_snapshot())
        self.repository.complete_dream_run(
            run_id,
            status="success" if remaining_tasks == 0 else "awaiting_host_compaction",
            candidate_count=int(run_row["candidate_count"]),
            clusters_created=int(run_row["clusters_created"]) + 1,
            memory_path=memory_path,
            notes=notes,
            mark_completed=remaining_tasks == 0,
        )

        return DreamCompactionResolution(
            status="success",
            task_id=task_id,
            run_id=run_id,
            resolution_backend=resolution_backend,
            node_id=node_id,
            remaining_tasks=remaining_tasks,
            dream_completed=remaining_tasks == 0,
            memory_path=memory_path,
        )

    def _launch_background_dream(
        self,
        *,
        run_id: str,
        output_path: Path,
        window_hours: int,
        min_accesses: int,
        min_cluster_size: int,
        max_candidates: int,
        similarity_threshold: float,
        trigger_reason: str,
        auto_triggered: bool,
    ) -> None:
        command = [
            sys.executable,
            "-m",
            "cognitiveos.background_jobs",
            "dream-run",
            "--db-path",
            str(self.settings.db_path),
            "--memory-output-path",
            str(output_path),
            "--run-id",
            run_id,
            "--window-hours",
            str(window_hours),
            "--min-accesses",
            str(min_accesses),
            "--min-cluster-size",
            str(min_cluster_size),
            "--max-candidates",
            str(max_candidates),
            "--similarity-threshold",
            str(similarity_threshold),
            "--trigger-reason",
            trigger_reason,
            "--auto-triggered",
            "1" if auto_triggered else "0",
        ]
        self._spawn_background_process(command)

    def _launch_background_heuristic_compaction(self, *, task_id: str) -> None:
        command = [
            sys.executable,
            "-m",
            "cognitiveos.background_jobs",
            "heuristic-compaction",
            "--db-path",
            str(self.settings.db_path),
            "--memory-output-path",
            str(self.settings.memory_output_path),
            "--task-id",
            task_id,
        ]
        self._spawn_background_process(command)

    def _spawn_background_process(self, command: list[str]) -> None:
        log_path = self._next_background_log_path(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = 0
        if sys.platform == "win32":
            creationflags = (
                getattr(subprocess, "DETACHED_PROCESS", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        with log_path.open("ab") as log_handle:
            subprocess.Popen(
                command,
                cwd=str(self._project_root()),
                stdout=log_handle,
                stderr=log_handle,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                close_fds=True,
            )

    def _next_background_log_path(self, command: list[str]) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        job_name = self._background_job_name(command)
        return self.settings.background_log_dir / f"{timestamp}-{job_name}-{uuid4().hex[:8]}.log"

    @staticmethod
    def _background_job_name(command: list[str]) -> str:
        if len(command) >= 4 and command[1:3] == ["-m", "cognitiveos.background_jobs"]:
            candidate = command[3]
        elif command:
            candidate = Path(command[0]).stem
        else:
            candidate = "background-job"
        safe = [
            char.lower() if char.isalnum() else "-"
            for char in candidate.strip()
        ]
        normalized = "".join(safe).strip("-")
        return normalized or "background-job"

    @staticmethod
    def _decorate_search_results(
        results: list[SearchResult],
        notices: list[str],
    ) -> list[SearchResult]:
        if results and notices:
            results[0].notices = notices
        return results

    def _remove_known_app_state_node_references(self, node_id: str) -> None:
        onboarding_node_ids = (
            self.repository.get_app_state_json(self.APP_STATE_BOOTSTRAP_ONBOARDING_NODE_IDS) or {}
        )
        if not isinstance(onboarding_node_ids, dict):
            return
        filtered = {
            key: value
            for key, value in onboarding_node_ids.items()
            if value != node_id
        }
        if filtered != onboarding_node_ids:
            self.repository.set_app_state_json(
                self.APP_STATE_BOOTSTRAP_ONBOARDING_NODE_IDS,
                filtered,
            )

    @staticmethod
    def _decorate_read_results(
        results: dict[str, ReadNodeResult],
        notices: list[str],
    ) -> dict[str, ReadNodeResult]:
        if notices:
            for result in results.values():
                result.notices = list(dict.fromkeys([*result.notices, *notices]))
        return results

    def _hydrate_remote_snapshot_content(self, results: dict[str, ReadNodeResult]) -> None:
        for result in results.values():
            source = result.metadata.get("source", {}) if isinstance(result.metadata, dict) else {}
            if not self._is_remote_source_kind(source.get("kind")):
                continue
            snapshot = (
                result.metadata.get("snapshot", {}) if isinstance(result.metadata, dict) else {}
            )
            snapshot_path = snapshot.get("path")
            if not snapshot_path:
                continue
            path = Path(snapshot_path)
            if path.exists():
                if snapshot.get("format") == "markdown":
                    snapshot_text = path.read_text(encoding="utf-8")
                    snapshot_body = self._extract_snapshot_markdown_body(snapshot_text)
                    if not self._snapshot_content_hash_matches(snapshot_body, snapshot):
                        result.notices.append(
                            f"Preserved snapshot content hash mismatch: {snapshot_path}. Showing stored note instead."
                        )
                        continue
                    result.content = snapshot_text
                    continue
                snapshot_bytes = path.read_bytes()
                if not self._snapshot_content_hash_matches(snapshot_bytes, snapshot):
                    result.notices.append(
                        f"Preserved snapshot content hash mismatch: {snapshot_path}. Showing stored note instead."
                    )
                    continue
                result.notices.append(
                    f"Binary snapshot preserved at {snapshot_path}; showing stored note instead of raw bytes."
                )
                continue
            result.notices.append(
                f"Preserved snapshot file is unavailable: {snapshot_path}. Showing stored note instead."
            )

    @staticmethod
    def _snapshot_path_for_node(node: NodeRecord) -> Path | None:
        snapshot = node.metadata.get("snapshot") if isinstance(node.metadata, dict) else None
        if not isinstance(snapshot, dict):
            return None
        path = snapshot.get("path")
        return Path(path) if path else None

    @staticmethod
    def _extract_snapshot_markdown_body(snapshot_text: str) -> str:
        if not snapshot_text.startswith("---\n"):
            return snapshot_text.strip()
        delimiter = "\n---\n"
        end_index = snapshot_text.find(delimiter, 4)
        if end_index == -1:
            return snapshot_text.strip()
        return snapshot_text[end_index + len(delimiter):].strip()

    @staticmethod
    def _snapshot_content_hash_matches(content: str | bytes, snapshot: dict[str, Any]) -> bool:
        expected = snapshot.get("content_hash")
        if not expected:
            return True
        payload = content.encode("utf-8") if isinstance(content, str) else content
        return hashlib.sha256(payload).hexdigest() == expected

    @staticmethod
    def _is_remote_source_kind(source_kind: str | None) -> bool:
        return bool(source_kind and source_kind.startswith("remote_"))

    def _normalize_host_kind(self, host_kind: str) -> str:
        normalized = host_kind.strip().lower().replace("-", "_")
        if normalized not in self.SUPPORTED_HOST_KINDS:
            raise InvalidPayloadError(
                f"Unsupported host kind '{host_kind}'. Supported values: "
                f"{sorted(self.SUPPORTED_HOST_KINDS)}"
            )
        return normalized

    def _project_root(self) -> Path:
        return self.settings.project_root.resolve()

    def _bootstrap_artifact_paths(
        self,
        *,
        bootstrap_dir: Path,
        host_kind: str,
    ) -> dict[str, Path]:
        return {
            "bootstrap_prompt_path": bootstrap_dir / "host-bootstrap.md",
            "system_prompt_path": bootstrap_dir / f"{host_kind}-system-prompt.md",
            "mount_manifest_path": bootstrap_dir / "mount-manifest.json",
            "mcp_config_path": bootstrap_dir / "mcp-server.json",
            "onboarding_path": bootstrap_dir / "onboarding-questions.json",
        }

    def _host_mount_targets(self, host_kind: str) -> tuple[Path | None, Path | None]:
        project_root = self._project_root()
        if host_kind == "codex":
            return project_root / "AGENTS.md", project_root / ".codex" / "config.toml"
        return None, None

    def _bootstrap_onboarding_questions(self) -> list[HostOnboardingQuestion]:
        return [
            HostOnboardingQuestion(
                id="display_name",
                prompt="How should the host address you?",
                guidance="A preferred name or form of address for future sessions.",
                example="Bruce",
            ),
            HostOnboardingQuestion(
                id="role_team",
                prompt="What is your role or team?",
                guidance="Use the most durable title or team name, not a temporary assignment.",
                example="Sr. Data Engineer, Data Solution China",
            ),
            HostOnboardingQuestion(
                id="preferred_language",
                prompt="Which language should the host use by default?",
                guidance="State one default language for normal responses.",
                example="Chinese",
            ),
            HostOnboardingQuestion(
                id="response_style",
                prompt="What response style should the host default to?",
                guidance=(
                    "Describe stable preferences such as concise, direct, "
                    "pragmatic, or detailed."
                ),
                example="Concise, direct, pragmatic",
            ),
            HostOnboardingQuestion(
                id="workspace_goal",
                prompt="What is the main purpose of this workspace or project?",
                guidance="Capture the durable goal, not a one-off task.",
                example="Build a local-first cognitive memory runtime for agent hosts",
            ),
        ]

    def _bootstrap_onboarding_completed(self, *, has_profile_nodes: bool) -> bool:
        completed = self.repository.get_app_state_value(
            self.APP_STATE_BOOTSTRAP_ONBOARDING_COMPLETED
        )
        if completed == "1":
            return True
        if has_profile_nodes:
            self.repository.set_app_state_value(
                self.APP_STATE_BOOTSTRAP_ONBOARDING_COMPLETED,
                "1",
            )
            return True
        return False

    def _bootstrap_onboarding_sections(
        self,
        answers: dict[str, str],
    ) -> list[dict[str, Any]]:
        identity_lines = [
            f"Preferred name: {answers['display_name']}",
            f"Role or team: {answers['role_team']}",
        ]
        communication_lines = [
            f"Default language: {answers['preferred_language']}",
            f"Response style: {answers['response_style']}",
        ]
        goal_lines = [
            f"Workspace goal: {answers['workspace_goal']}",
        ]
        return [
            {
                "section_id": "identity",
                "name": "Bootstrap Identity",
                "content": "\n".join(identity_lines),
                "metadata": {"bootstrap_section": "identity"},
            },
            {
                "section_id": "communication",
                "name": "Bootstrap Communication Preferences",
                "content": "\n".join(communication_lines),
                "metadata": {"bootstrap_section": "communication"},
            },
            {
                "section_id": "workspace_goal",
                "name": "Bootstrap Workspace Goal",
                "content": "\n".join(goal_lines),
                "metadata": {"bootstrap_section": "workspace"},
            },
        ]

    def _upsert_bootstrap_profile_node(
        self,
        *,
        node_id: str | None,
        name: str,
        content: str,
        metadata: dict[str, Any],
    ) -> str:
        combined_metadata = {
            **metadata,
            "profile": {
                "kind": "system",
                "bootstrap": True,
                "canonical": True,
                "section": str(metadata.get("bootstrap_section") or ""),
            },
        }
        if node_id:
            self._validate_node_content_limit(content)
            self.repository.overwrite_node(
                NodeRecord(
                    id=node_id,
                    name=name,
                    description=self._summarize(content),
                    content=content,
                    embedding=self._embed_content(content),
                    tags=self.CANONICAL_PROFILE_SECTION_TAGS.get(
                        str(metadata.get("bootstrap_section") or ""),
                        ["profile", "bootstrap"],
                    ),
                    metadata=combined_metadata,
                    node_type="memory",
                    durability="pinned",
                ),
                actor=self.settings.default_actor,
                action_type="update",
            )
            return node_id

        self._validate_node_content_limit(content)
        node = NodeRecord(
            id=f"node_{uuid4().hex}",
            name=name,
            description=self._summarize(content),
            content=content,
            embedding=self._embed_content(content),
            tags=self.CANONICAL_PROFILE_SECTION_TAGS.get(
                str(metadata.get("bootstrap_section") or ""),
                ["profile", "bootstrap"],
            ),
            metadata=combined_metadata,
            node_type="memory",
            durability="pinned",
        )
        self.repository.create_node(
            node,
            actor=self.settings.default_actor,
            action_type="bootstrap_create",
        )
        return node.id

    def _get_host_install_record(self, host_kind: str) -> dict[str, Any] | None:
        installs = self.repository.get_app_state_json(self.APP_STATE_BOOTSTRAP_HOST_INSTALLS) or {}
        record = installs.get(host_kind)
        if isinstance(record, dict):
            return record
        return None

    def _registered_memory_output_paths(self, *, primary_path: Path) -> list[Path]:
        ordered_paths: list[Path] = [primary_path.resolve()]
        if self.settings.memory_output_path.resolve() not in ordered_paths:
            ordered_paths.append(self.settings.memory_output_path.resolve())
        targets = self.repository.get_app_state_json(self.APP_STATE_BOOTSTRAP_HOST_MEMORY_TARGETS) or {}
        for host_kind, record in targets.items():
            if not isinstance(record, dict):
                continue
            value = record.get("memory_output_path")
            if isinstance(value, str) and value.strip():
                candidate = Path(value).expanduser().resolve()
            elif isinstance(host_kind, str) and host_kind.strip():
                candidate = self._host_memory_output_path(host_kind=host_kind)
            else:
                continue
            if candidate not in ordered_paths:
                ordered_paths.append(candidate)
        return ordered_paths

    def _register_host_memory_target(self, *, host_kind: str, memory_output_path: Path) -> None:
        targets = (
            self.repository.get_app_state_json(self.APP_STATE_BOOTSTRAP_HOST_MEMORY_TARGETS) or {}
        )
        targets[host_kind] = {
            "registered_at": datetime.now(UTC).isoformat(),
            "project_root": str(self._project_root()),
            "memory_output_path": str(memory_output_path.resolve()),
        }
        self.repository.set_app_state_json(self.APP_STATE_BOOTSTRAP_HOST_MEMORY_TARGETS, targets)

    def _host_installation_exists(
        self,
        host_kind: str,
        *,
        install_record: dict[str, Any] | None,
        host_instruction_path: Path | None,
        host_project_config_path: Path | None,
    ) -> bool:
        if install_record is None:
            return False
        if host_kind == "codex":
            if host_instruction_path is None or host_project_config_path is None:
                return False
            return self._file_contains(
                host_instruction_path,
                "COGNITIVEOS HOST BOOTSTRAP START",
            ) and self._file_contains(
                host_project_config_path,
                "COGNITIVEOS HOST BOOTSTRAP START",
            )
        return False

    def _install_host_mount(
        self,
        *,
        host_kind: str,
        system_prompt_block: str,
    ) -> bool:
        if host_kind != "codex":
            return False

        host_instruction_path, host_project_config_path = self._host_mount_targets(host_kind)
        if host_instruction_path is None or host_project_config_path is None:
            return False

        host_instruction_block = self._render_host_instruction_block(system_prompt_block)
        self._write_managed_block(
            host_instruction_path,
            marker_name="COGNITIVEOS HOST BOOTSTRAP",
            body=host_instruction_block,
        )
        host_project_config_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_managed_block(
            host_project_config_path,
            marker_name="COGNITIVEOS HOST BOOTSTRAP",
            body=self._codex_project_config_block(),
        )

        installs = self.repository.get_app_state_json(self.APP_STATE_BOOTSTRAP_HOST_INSTALLS) or {}
        installs[host_kind] = {
            "installed_at": datetime.now(UTC).isoformat(),
            "host_instruction_path": str(host_instruction_path),
            "host_project_config_path": str(host_project_config_path),
            "memory_output_path": str(self._host_memory_output_path(host_kind=host_kind)),
        }
        self.repository.set_app_state_json(self.APP_STATE_BOOTSTRAP_HOST_INSTALLS, installs)
        return True

    @staticmethod
    def _file_contains(path: Path, marker: str) -> bool:
        if not path.exists():
            return False
        return marker in path.read_text(encoding="utf-8")

    @staticmethod
    def _write_managed_block(path: Path, *, marker_name: str, body: str) -> None:
        start_marker = f"# {marker_name} START"
        end_marker = f"# {marker_name} END"
        managed_block = f"{start_marker}\n{body.rstrip()}\n{end_marker}\n"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""

        if start_marker in existing and end_marker in existing:
            prefix, remainder = existing.split(start_marker, 1)
            _, suffix = remainder.split(end_marker, 1)
            next_content = prefix.rstrip()
            if next_content:
                next_content += "\n\n"
            next_content += managed_block
            suffix = suffix.lstrip("\n")
            if suffix:
                next_content += "\n" + suffix
        else:
            next_content = existing.rstrip()
            if next_content:
                next_content += "\n\n"
            next_content += managed_block
        path.write_text(next_content.rstrip() + "\n", encoding="utf-8")

    def _render_host_instruction_block(self, system_prompt_block: str) -> str:
        return "\n".join(
            [
                "Use CognitiveOS as the workspace memory runtime.",
                "",
                "Cold-start contract:",
                system_prompt_block,
            ]
        )

    def _codex_project_config_block(self) -> str:
        memory_output_path = self._host_memory_output_path(host_kind="codex")
        return "\n".join(
            [
                "[mcp_servers.cognitiveos]",
                'command = "cognitiveos-mcp"',
                "args = [",
                '  "--transport",',
                '  "stdio",',
                '  "--profile",',
                f'  "{self._host_mcp_profile(host_kind="codex")}",',
                '  "--project-root",',
                f'  "{self._project_root()}",',
                '  "--db-path",',
                f'  "{self.settings.db_path}",',
                '  "--memory-output-path",',
                f'  "{memory_output_path}",',
                "]",
                "startup_timeout_ms = 20000",
            ]
        )

    def _host_system_prompt(self, memory_path: Path, *, host_kind: str) -> str:
        questions = self._bootstrap_onboarding_questions()
        limited_codex_surface = host_kind.strip().lower().replace("-", "_") == "codex"
        lines = [
            "Cold-start mount procedure:",
            (
                f"1. Read `{memory_path}` as the baseline memory before any "
                "memory retrieval call."
            ),
            (
                "2. Call `get_host_bootstrap_status` with the current host kind "
                "at the start of the session."
            ),
            (
                "3. If `needs_onboarding` is true, ask the user these questions "
                "and submit them with `submit_host_onboarding`:"
            ),
        ]
        for question in questions:
            lines.append(f"   - `{question.id}`: {question.prompt}")
        lines.extend(
            [
                (
                    "4. After onboarding, rely on MCP memory tools instead of "
                    "editing MEMORY.MD or SQLite directly."
                ),
                (
                    "5. Prefer `search` then `read` for recall, and "
                    "`add`/`update`/`link` for writes."
                    if limited_codex_surface
                    else "5. Prefer `search` then `read` for recall, and "
                    "`add`/`update`/`link` for writes."
                ),
                (
                    "6. For profile writes, update canonical nodes instead of creating "
                    "parallel ones: `Bootstrap Identity`, `Bootstrap Communication Preferences`, "
                    "`Bootstrap Workspace Goal`, and `Engineering Collaboration Preferences`."
                ),
                (
                    "7. If a response includes a dream reminder and the host "
                    "supports background agents, delegate `dream` in the background."
                ),
                (
                    "8. If a dream run returns pending compactions, resolve them "
                    "through `dream` itself using the returned task payload rather "
                    "than manual file edits."
                ),
                f"9. Host kind for this install target: `{host_kind}`.",
                "",
                "Parameter recipes:",
                (
                    "- This managed host install intentionally mounts the reduced `compact-core` profile "
                    "with `search`, `read`, `add`, `update`, `link`, and `dream`, while "
                    "still omitting `unlink` to reduce tool-schema and parameter errors."
                    if limited_codex_surface
                    else None
                ),
                (
                    "- `search`: pass at least one of `query` or `keyword`; start with "
                    "`include_neighbors=0` or `1`."
                ),
                (
                    "- `read`: pass concrete ids from `search`; set `include_content=true` "
                    "only when summaries are insufficient."
                ),
                (
                    "- `add`: use `type=content` for raw text, `type=file` for a file path "
                    "or URL, and `type=folder` for a local folder root."
                ),
                (
                    "- `update`: reuse the existing node id and send the replacement "
                    "`content`."
                ),
                (
                    "- `link`: pass `src_id`, `dst_id`, and a concrete directed `relation` "
                    "such as `supports` or `depends_on`."
                ),
                (
                    "- `dream`: use `inspect=status|runs|tasks` to inspect; use `task_id` "
                    "plus `title`, `description`, and `content` to resolve a host-authored "
                    "compaction, or set `use_heuristic=true`."
                ),
            ]
        )
        return "\n".join(line for line in lines if line is not None) + "\n"

    def _bootstrap_prompt(self, memory_path: Path, *, host_kind: str) -> str:
        limited_codex_surface = host_kind.strip().lower().replace("-", "_") == "codex"
        return "\n".join(
            [
                "# CognitiveOS Host Bootstrap",
                "",
                "Cold-start mount procedure:",
                (
                    f"1. Load `{memory_path}` as read-only baseline memory "
                    "before the first memory tool call."
                ),
                (
                    f"2. Install the `{host_kind}` host mount if available, then start the "
                    "`cognitiveos-mcp` server and mount it as an MCP server."
                ),
                (
                    "3. At first startup, ask the onboarding questions from "
                    "`onboarding-questions.json` and submit them through "
                    "`submit_host_onboarding` before depending on memory."
                ),
                (
                    "4. Prefer MCP tools for search/read/add/update/link/dream over ad hoc "
                    "filesystem parsing."
                    if limited_codex_surface
                    else "4. Prefer MCP tools for search/read/add/update/link/dream "
                    "over ad hoc filesystem parsing."
                ),
                (
                    "5. For profile writes, update canonical nodes instead of creating parallel "
                    "ones for identity, communication, workspace, or engineering preferences."
                ),
                (
                    "6. If any response includes a dream reminder and the host supports background "
                    "subagents, delegate the `dream` task to a background subagent."
                ),
                (
                    "7. If a dream run returns pending compactions, have the host agent compress "
                    "those clusters and submit the resolution back through `dream`."
                ),
                "",
                "Recommended MCP command:",
                (f"`{self._host_mcp_command(host_kind=host_kind)}`"),
            ]
        ) + "\n"

    @staticmethod
    def _host_mcp_profile(*, host_kind: str = "generic") -> str:
        normalized = host_kind.strip().lower().replace("-", "_")
        if normalized == "codex":
            return "compact-core"
        return "host-core"

    def _host_memory_output_path(self, *, host_kind: str = "generic") -> Path:
        normalized = host_kind.strip().lower().replace("-", "_")
        if normalized == "codex":
            return Path.home() / ".codex" / "MEMORY.MD"
        if normalized != "generic":
            return self._project_root() / "MEMORY.MD"
        return self.settings.memory_output_path

    def _host_mcp_args(self, *, host_kind: str = "generic") -> list[str]:
        memory_output_path = self._host_memory_output_path(host_kind=host_kind)
        return [
            "--transport",
            "stdio",
            "--profile",
            self._host_mcp_profile(host_kind=host_kind),
            "--project-root",
            str(self._project_root()),
            "--db-path",
            str(self.settings.db_path),
            "--memory-output-path",
            str(memory_output_path),
        ]

    def _host_mcp_command(self, *, host_kind: str = "generic") -> str:
        return " ".join(["cognitiveos-mcp", *self._host_mcp_args(host_kind=host_kind)])
