from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.graph_governance import GraphGovernanceEngine
from cognitiveos.metadata_shapes import metadata_profile_kind
from cognitiveos.models import (
    DreamCompactionTask,
    DreamResult,
    DreamSuperNode,
    EdgeRecord,
    NodeRecord,
)
from cognitiveos.providers.base import ChatProvider, EmbeddingProvider


class DreamCompiler:
    def __init__(
        self,
        repository: SQLiteRepository,
        *,
        governance_engine: GraphGovernanceEngine,
        max_node_content_chars: int,
        semantic_neighbor_k: int,
        embedding_provider: EmbeddingProvider | None = None,
        chat_provider: ChatProvider | None = None,
    ) -> None:
        self.repository = repository
        self.governance = governance_engine
        self.max_node_content_chars = max_node_content_chars
        self.semantic_neighbor_k = semantic_neighbor_k
        self.embedding_provider = embedding_provider
        self.chat_provider = chat_provider

    def run(
        self,
        *,
        run_id: str,
        output_path: Path,
        window_hours: int,
        min_accesses: int,
        min_cluster_size: int,
        max_candidates: int,
        similarity_threshold: float,
    ) -> DreamResult:
        candidates = [
            node
            for node in self.repository.list_recent_or_frequent_nodes(
                window_hours=window_hours,
                min_accesses=min_accesses,
                limit=max_candidates,
            )
            if metadata_profile_kind(node.metadata) != "system"
            and node.node_type != "super_node"
            and node.durability != "pinned"
        ]
        clusters = self._build_clusters(
            candidates,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
        )
        relationship_cleanup_plans = self.governance.build_relationship_cleanup_plans(
            candidates,
            clusters,
        )

        super_nodes: list[DreamSuperNode] = []
        durability_suggestions = []
        pending_compactions: list[DreamCompactionTask] = []
        notices: list[str] = []

        for cluster in clusters:
            self.governance.reinforce_cluster_relationships([node.id for node in cluster])
            if self.chat_provider is None:
                pending_compactions.append(
                    self._create_host_compaction_task(
                        run_id=run_id,
                        cluster=cluster,
                        reason="Chat provider is not configured.",
                    )
                )
                continue

            try:
                super_node = self.create_super_node_from_cluster(
                    cluster,
                    title=self._make_super_node_title(cluster),
                    description=self._make_chat_super_node_description(cluster),
                    content=self._make_super_node_content(cluster),
                    backend="chat_provider",
                    run_id=run_id,
                )
            except Exception as exc:
                pending_compactions.append(
                    self._create_host_compaction_task(
                        run_id=run_id,
                        cluster=cluster,
                        reason=f"Chat provider failed during dream compaction: {exc}",
                    )
                )
                notices.append(
                    "Dream delegated a cluster to the host agent because the chat provider "
                    f"failed: {exc}"
                )
                continue

            self._redirect_edges(cluster, super_node)
            super_nodes.append(
                DreamSuperNode(
                    node_id=super_node.id,
                    source_node_ids=[node.id for node in cluster],
                )
            )
            durability_suggestions.extend(
                self.governance.build_cluster_durability_suggestions(
                    cluster,
                    super_node=super_node,
                )
            )

        rendered_path = self.governance.compile_memory_snapshot(output_path)
        status = "awaiting_host_compaction" if pending_compactions else "success"
        if pending_compactions:
            notices.append(
                "Host compaction is required for one or more dream clusters. "
                "Use the pending compaction payload and submit the compressed result."
            )
        if relationship_cleanup_plans:
            notices.append(
                f"Dream identified {len(relationship_cleanup_plans)} relationship cleanup plans."
            )
        if durability_suggestions:
            notices.append(
                f"Dream produced {len(durability_suggestions)} durability suggestions."
            )
        return DreamResult(
            status=status,
            candidate_node_ids=[node.id for node in candidates],
            clusters_created=len(super_nodes),
            super_nodes=super_nodes,
            relationship_cleanup_plans=relationship_cleanup_plans,
            durability_suggestions=durability_suggestions,
            pending_compactions=pending_compactions,
            memory_path=str(rendered_path),
            notices=notices,
        )

    def load_cluster_from_task(self, task_row: dict) -> list[NodeRecord]:
        source_node_ids = json.loads(task_row["source_node_ids_json"])
        return [self.repository.get_node(node_id) for node_id in source_node_ids]

    def resolve_task_with_host_payload(
        self,
        task_row: dict,
        *,
        title: str,
        description: str,
        content: str,
    ) -> NodeRecord:
        cluster = self.load_cluster_from_task(task_row)
        super_node = self.create_super_node_from_cluster(
            cluster,
            title=title,
            description=description,
            content=content,
            backend="host_agent",
            run_id=task_row["run_id"],
            task_id=task_row["task_id"],
        )
        self._redirect_edges(cluster, super_node)
        return super_node

    def resolve_task_with_heuristic(self, task_row: dict) -> NodeRecord:
        cluster = self.load_cluster_from_task(task_row)
        content = task_row["prepared_content"]
        super_node = self.create_super_node_from_cluster(
            cluster,
            title=task_row["suggested_title"] or self._make_super_node_title(cluster),
            description=task_row["suggested_description"]
            or self._make_heuristic_super_node_description(cluster),
            content=content,
            backend="heuristic",
            run_id=task_row["run_id"],
            task_id=task_row["task_id"],
        )
        self._redirect_edges(cluster, super_node)
        return super_node

    def create_super_node_from_cluster(
        self,
        cluster: list[NodeRecord],
        *,
        title: str,
        description: str,
        content: str,
        backend: str,
        run_id: str,
        task_id: str | None = None,
    ) -> NodeRecord:
        metadata = {
            "dream_source_node_ids": [node.id for node in cluster],
            "dream_cluster_size": len(cluster),
            "dream_run_id": run_id,
            "dream_compaction_backend": backend,
        }
        if task_id is not None:
            metadata["dream_task_id"] = task_id

        embedding = None
        if self.embedding_provider is not None:
            embedding = self.embedding_provider.embed([content])[0]

        super_node = NodeRecord(
            id=f"node_{uuid4().hex}",
            name=title,
            description=description,
            content=content[: self.max_node_content_chars],
            embedding=embedding,
            tags=sorted({tag for node in cluster for tag in node.tags}),
            metadata=metadata,
            node_type="super_node",
            durability="working",
        )
        self.repository.create_node(super_node, actor="dream", action_type="create")
        return super_node

    def _build_clusters(
        self,
        candidates: list[NodeRecord],
        *,
        min_cluster_size: int,
        similarity_threshold: float,
    ) -> list[list[NodeRecord]]:
        if not candidates:
            return []

        candidate_ids = {node.id for node in candidates}
        rows_by_id = {node.id: node for node in candidates}
        parent = {node.id: node.id for node in candidates}
        rank = {node.id: 0 for node in candidates}

        def find(node_id: str) -> str:
            root = node_id
            while parent[root] != root:
                root = parent[root]
            while parent[node_id] != node_id:
                next_node = parent[node_id]
                parent[node_id] = root
                node_id = next_node
            return root

        def union(left_id: str, right_id: str) -> None:
            left_root = find(left_id)
            right_root = find(right_id)
            if left_root == right_root:
                return
            if rank[left_root] < rank[right_root]:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            if rank[left_root] == rank[right_root]:
                rank[left_root] += 1

        for edge in self.repository.list_edges_for_nodes(list(candidate_ids)):
            if edge.src_id in candidate_ids and edge.dst_id in candidate_ids:
                union(edge.src_id, edge.dst_id)

        available_ids = [node.id for node in candidates if node.embedding is not None]
        cached_neighbor_rows = self.repository.list_semantic_neighbors(
            available_ids,
            min_similarity=similarity_threshold,
        )
        cached_node_ids = {node_id for node_id, _neighbor_id, _similarity in cached_neighbor_rows}
        missing_cache_ids = [
            node_id
            for node_id in available_ids
            if node_id not in cached_node_ids
        ]
        for node_id in missing_cache_ids:
            self.repository.refresh_semantic_neighbors_for_node(
                node_id,
                top_k=self.semantic_neighbor_k,
            )
        if missing_cache_ids:
            cached_neighbor_rows = self.repository.list_semantic_neighbors(
                available_ids,
                min_similarity=similarity_threshold,
            )

        for node_id, neighbor_id, similarity in cached_neighbor_rows:
            if similarity < similarity_threshold or neighbor_id not in candidate_ids:
                continue
            union(node_id, neighbor_id)

        grouped: dict[str, list[NodeRecord]] = defaultdict(list)
        for node in candidates:
            grouped[find(node.id)].append(rows_by_id[node.id])
        return [
            component
            for component in grouped.values()
            if len(component) >= min_cluster_size
        ]

    def _create_host_compaction_task(
        self,
        *,
        run_id: str,
        cluster: list[NodeRecord],
        reason: str,
    ) -> DreamCompactionTask:
        source_nodes = [self._project_task_source_node(node) for node in cluster]
        prepared_content = self._make_super_node_content(cluster)
        suggested_title = self._make_super_node_title(cluster)
        suggested_description = self._make_heuristic_super_node_description(cluster)
        prompt = self._make_host_compaction_prompt(
            cluster=cluster,
            suggested_title=suggested_title,
            prepared_content=prepared_content,
        )
        task_id = self.repository.create_dream_compaction_task(
            run_id=run_id,
            requested_backend="host_agent",
            fallback_backend="heuristic",
            reason=reason,
            suggested_title=suggested_title,
            suggested_description=suggested_description,
            prepared_content=prepared_content,
            prompt=prompt,
            source_nodes=source_nodes,
            source_node_ids=[node["id"] for node in source_nodes],
        )
        return DreamCompactionTask(
            task_id=task_id,
            run_id=run_id,
            status="pending",
            requested_backend="host_agent",
            fallback_backend="heuristic",
            reason=reason,
            suggested_title=suggested_title,
            suggested_description=suggested_description,
            prepared_content=prepared_content,
            prompt=prompt,
            source_nodes=source_nodes,
        )

    def _redirect_edges(self, cluster: list[NodeRecord], super_node: NodeRecord) -> None:
        cluster_ids = [node.id for node in cluster]
        cluster_id_set = set(cluster_ids)
        external_edges = [
            edge
            for edge in self.repository.list_edges_for_nodes(cluster_ids)
            if not (edge.src_id in cluster_id_set and edge.dst_id in cluster_id_set)
        ]

        self.repository.delete_external_edges_for_nodes(cluster_ids)
        redirected_edges: list[EdgeRecord] = []

        for edge in external_edges:
            if edge.src_id in cluster_id_set:
                redirected_edges.append(
                    EdgeRecord(
                    src_id=super_node.id,
                    dst_id=edge.dst_id,
                    relation=edge.relation,
                    strength_score=edge.strength_score,
                    durability=edge.durability,
                    status=edge.status,
                    metadata={
                        **edge.metadata,
                        "provenance": {
                            **(edge.metadata.get("provenance", {})),
                            "creation_mode": "dream_generated",
                        },
                        "redirect": {
                            **(edge.metadata.get("redirect", {})),
                            "from": edge.src_id,
                        },
                    },
                )
                )
            else:
                redirected_edges.append(
                    EdgeRecord(
                    src_id=edge.src_id,
                    dst_id=super_node.id,
                    relation=edge.relation,
                    strength_score=edge.strength_score,
                    durability=edge.durability,
                    status=edge.status,
                    metadata={
                        **edge.metadata,
                        "provenance": {
                            **(edge.metadata.get("provenance", {})),
                            "creation_mode": "dream_generated",
                        },
                        "redirect": {
                            **(edge.metadata.get("redirect", {})),
                            "to": edge.dst_id,
                        },
                    },
                )
                )

        redirected_edges.extend(
            [
                EdgeRecord(
                    src_id=super_node.id,
                    dst_id=node.id,
                    relation="contains",
                    strength_score=1.0,
                    durability="durable",
                    metadata={"provenance": {"creation_mode": "dream_generated"}},
                )
                for node in cluster
            ]
        )
        self.repository.create_edges(redirected_edges)

    @staticmethod
    def _make_super_node_title(cluster: list[NodeRecord]) -> str:
        first = cluster[0].name or cluster[0].id
        return f"Dream Cluster: {first}"

    def _make_chat_super_node_description(self, cluster: list[NodeRecord]) -> str:
        if self.chat_provider is None:
            raise ValueError("Chat provider is not configured.")
        content = self._make_super_node_content(cluster)
        summary = self.chat_provider.summarize(content).strip()
        if not summary:
            raise ValueError("Chat provider returned an empty dream summary.")
        return summary[:500]

    @staticmethod
    def _make_heuristic_super_node_description(cluster: list[NodeRecord]) -> str:
        joined = "; ".join(node.description for node in cluster if node.description)
        if len(joined) <= 500:
            return joined
        return joined[:497].rstrip() + "..."

    def _make_super_node_content(self, cluster: list[NodeRecord]) -> str:
        lines = [
            "Dream consolidation result.",
            "",
            "Source nodes:",
        ]
        for node in cluster:
            title = node.name or node.id
            lines.append(f"- {title} ({node.id})")
            lines.append(f"  Description: {node.description}")
            lines.append(f"  Content: {node.content}")
        return "\n".join(lines)[: self.max_node_content_chars]

    @staticmethod
    def _make_host_compaction_prompt(
        *,
        cluster: list[NodeRecord],
        suggested_title: str,
        prepared_content: str,
    ) -> str:
        source_ids = ", ".join(node.id for node in cluster)
        return "\n".join(
            [
                "# Dream Compaction Task",
                "",
                "You are the host agent responsible for compressing one CognitiveOS dream cluster.",
                "Produce one consolidated super-node payload.",
                "",
                "Return one valid JSON object with exactly these keys:",
                '- "title": concise cluster title',
                '- "description": 500 characters or fewer',
                (
                    '- "content": compressed long-form synthesis that stays within '
                    "the configured node content limit"
                ),
                "",
                f"Suggested title: {suggested_title}",
                f"Source node ids: {source_ids}",
                "",
                "Compression rules:",
                "- Keep durable facts and reusable operating knowledge.",
                "- Remove repetition, filler, and low-value detail.",
                "- Preserve contradictions, constraints, and important links.",
                "- Do not invent facts that are not supported by the source nodes.",
                "- Do not include markdown fences or explanation outside the JSON object.",
                "",
                "The runtime will create edges and source references after submission.",
                "",
                "Prepared source digest:",
                prepared_content,
            ]
        )

    @staticmethod
    def _project_task_source_node(node: NodeRecord) -> dict[str, object]:
        return {
            "id": node.id,
            "name": node.name,
            "description": node.description,
            "content": node.content,
            "tags": list(node.tags),
        }
