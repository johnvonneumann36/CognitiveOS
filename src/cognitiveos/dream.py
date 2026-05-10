from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.graph_governance import GraphGovernanceEngine
from cognitiveos.metadata_shapes import (
    metadata_entities,
    metadata_profile_kind,
    normalize_entities,
)
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
        cascade_passes: int = 3,
        cascade_threshold_step: float = 0.12,
        cascade_max_threshold: float = 0.95,
        leiden_resolution_start: float = 1.75,
        leiden_resolution_step: float = 0.4,
        leiden_resolution_min: float = 0.85,
        bridge_edge_weight_multiplier: float = 0.65,
    ) -> DreamResult:
        cascade_passes = max(1, cascade_passes)
        layer_names = ["small", "medium", "large"]

        def leiden_resolution_for_pass(pass_index: int) -> float:
            return max(
                leiden_resolution_min,
                leiden_resolution_start - (leiden_resolution_step * pass_index),
            )

        def layer_name_for_pass(pass_index: int) -> str:
            if pass_index < len(layer_names):
                return layer_names[pass_index]
            return f"layer_{pass_index}"

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
        effective_config = {
            "semantic_threshold": similarity_threshold,
            "entityless_threshold": (
                self.governance.settings.entityless_union_similarity_threshold
            ),
            "max_candidates": max_candidates,
            "min_cluster_size": min_cluster_size,
            "entity_extraction_backend": "metadata_entities",
            "cluster_backend": "leiden_knn_graph",
            "cascade_passes": cascade_passes,
            "cascade_threshold_step": cascade_threshold_step,
            "cascade_max_threshold": cascade_max_threshold,
            "leiden_resolution_start": leiden_resolution_start,
            "leiden_resolution_step": leiden_resolution_step,
            "leiden_resolution_min": leiden_resolution_min,
            "bridge_edge_weight_multiplier": bridge_edge_weight_multiplier,
            "layer_names": layer_names[:cascade_passes],
        }
        first_pass_resolution = leiden_resolution_for_pass(0)
        candidate_explanations = [
            {
                "node_id": node.id,
                "decision": "included",
                "reason": "recent_or_frequent_non_system_non_pinned_memory",
                "pass_index": 0,
                "node_type": node.node_type,
                "durability": node.durability,
                "entities": metadata_entities(node.metadata),
            }
            for node in candidates
        ]
        clusters, skipped_unions, entity_gate_decisions = self._build_clusters(
            candidates,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
            leiden_resolution=first_pass_resolution,
            pass_index=0,
            bridge_edge_weight_multiplier=bridge_edge_weight_multiplier,
        )
        cluster_explanations = [
            {
                "cluster_index": index,
                "node_ids": [node.id for node in cluster],
                "size": len(cluster),
                "entities": self._cluster_entities(cluster),
                "decision": "eligible_for_compaction",
                "reason": f"cluster_size >= {min_cluster_size}",
                "pass_index": 0,
                "dream_layer": layer_name_for_pass(0),
                "leiden_resolution": first_pass_resolution,
            }
            for index, cluster in enumerate(clusters)
        ]
        relationship_cleanup_plans = self.governance.build_relationship_cleanup_plans(
            candidates,
            clusters,
        )

        super_nodes: list[DreamSuperNode] = []
        durability_suggestions = []
        pending_compactions: list[DreamCompactionTask] = []
        notices: list[str] = []

        def materialize_clusters(
            pass_clusters: list[list[NodeRecord]],
            *,
            pass_index: int,
        ) -> list[NodeRecord]:
            materialized: list[NodeRecord] = []
            for cluster in pass_clusters:
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
                        dream_layer=layer_name_for_pass(pass_index),
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
                materialized.append(super_node)
                durability_suggestions.extend(
                    self.governance.build_cluster_durability_suggestions(
                        cluster,
                        super_node=super_node,
                    )
                )
            if materialized and pass_index > 0:
                notices.append(
                    f"Dream cascade pass {pass_index} created {len(materialized)} super nodes."
                )
            return materialized

        cascade_candidates = materialize_clusters(clusters, pass_index=0)
        all_candidate_ids = [node.id for node in candidates]

        for pass_index in range(1, cascade_passes):
            if len(cascade_candidates) < min_cluster_size:
                break
            pass_threshold = min(
                cascade_max_threshold,
                similarity_threshold + (cascade_threshold_step * pass_index),
            )
            pass_leiden_resolution = leiden_resolution_for_pass(pass_index)
            all_candidate_ids.extend(node.id for node in cascade_candidates)
            candidate_explanations.extend(
                {
                    "node_id": node.id,
                    "decision": "included",
                    "reason": "new_super_node_from_prior_dream_pass",
                    "pass_index": pass_index,
                    "dream_layer": layer_name_for_pass(pass_index),
                    "node_type": node.node_type,
                    "durability": node.durability,
                    "entities": metadata_entities(node.metadata),
                }
                for node in cascade_candidates
            )
            pass_clusters, pass_skipped, pass_entity_decisions = self._build_clusters(
                cascade_candidates,
                min_cluster_size=min_cluster_size,
                similarity_threshold=pass_threshold,
                leiden_resolution=pass_leiden_resolution,
                pass_index=pass_index,
                bridge_edge_weight_multiplier=bridge_edge_weight_multiplier,
            )
            skipped_unions.extend(
                {**decision, "pass_index": pass_index} for decision in pass_skipped
            )
            entity_gate_decisions.extend(
                {**decision, "pass_index": pass_index} for decision in pass_entity_decisions
            )
            cluster_explanations.extend(
                {
                    "cluster_index": len(cluster_explanations) + index,
                    "node_ids": [node.id for node in cluster],
                    "size": len(cluster),
                    "entities": self._cluster_entities(cluster),
                    "decision": "eligible_for_cascade_compaction",
                    "reason": (
                        f"cascade pass {pass_index} cluster_size >= {min_cluster_size} "
                        f"at threshold {pass_threshold}"
                    ),
                    "pass_index": pass_index,
                    "dream_layer": layer_name_for_pass(pass_index),
                    "leiden_resolution": pass_leiden_resolution,
                }
                for index, cluster in enumerate(pass_clusters)
            )
            relationship_cleanup_plans.extend(
                self.governance.build_relationship_cleanup_plans(
                    cascade_candidates,
                    pass_clusters,
                )
            )
            next_candidates = materialize_clusters(pass_clusters, pass_index=pass_index)
            if not next_candidates:
                break
            cascade_candidates = next_candidates

        rendered_path = self.governance.compile_memory_snapshot(output_path)
        projected_memory = self.governance.describe_memory_projection()
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
            notices.append(f"Dream produced {len(durability_suggestions)} durability suggestions.")
        return DreamResult(
            status=status,
            candidate_node_ids=list(dict.fromkeys(all_candidate_ids)),
            clusters_created=len(super_nodes),
            super_nodes=super_nodes,
            relationship_cleanup_plans=relationship_cleanup_plans,
            durability_suggestions=durability_suggestions,
            pending_compactions=pending_compactions,
            memory_path=str(rendered_path),
            effective_config=effective_config,
            candidate_explanations=candidate_explanations,
            cluster_explanations=cluster_explanations,
            skipped_unions=skipped_unions,
            entity_gate_decisions=entity_gate_decisions,
            projected_memory=projected_memory,
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
        dream_layer: str | None = None,
    ) -> NodeRecord:
        cluster_entities = self._cluster_entities(cluster)
        metadata = {
            "dream_source_node_ids": [node.id for node in cluster],
            "dream_cluster_size": len(cluster),
            "dream_run_id": run_id,
            "dream_compaction_backend": backend,
            "compression_policy_version": "entity-assisted-v1",
            "projection_policy_version": self.governance.projection_policy_version,
        }
        if dream_layer is not None:
            metadata["dream_layer"] = dream_layer
        if cluster_entities:
            metadata["entities"] = cluster_entities
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
        leiden_resolution: float = 1.75,
        pass_index: int = 0,
        bridge_edge_weight_multiplier: float = 0.65,
    ) -> tuple[list[list[NodeRecord]], list[dict[str, Any]], list[dict[str, Any]]]:
        return self._build_clusters_leiden(
            candidates,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
            leiden_resolution=leiden_resolution,
            pass_index=pass_index,
            bridge_edge_weight_multiplier=bridge_edge_weight_multiplier,
        )

    def _build_clusters_leiden(
        self,
        candidates: list[NodeRecord],
        *,
        min_cluster_size: int,
        similarity_threshold: float,
        leiden_resolution: float,
        pass_index: int,
        bridge_edge_weight_multiplier: float,
    ) -> tuple[list[list[NodeRecord]], list[dict[str, Any]], list[dict[str, Any]]]:
        if not candidates:
            return [], [], []

        candidate_ids = {node.id for node in candidates}
        rows_by_id = {node.id: node for node in candidates}
        id_to_idx = {node.id: idx for idx, node in enumerate(candidates)}
        skipped_unions: list[dict[str, Any]] = []
        entity_gate_decisions: list[dict[str, Any]] = []
        edge_weights: dict[tuple[int, int], float] = {}

        def add_edge(left_id: str, right_id: str, weight: float) -> None:
            if left_id == right_id:
                return
            left_idx = id_to_idx[left_id]
            right_idx = id_to_idx[right_id]
            if left_idx > right_idx:
                left_idx, right_idx = right_idx, left_idx
            edge_key = (left_idx, right_idx)
            edge_weights[edge_key] = max(edge_weights.get(edge_key, 0.0), weight)

        for edge in self.repository.list_edges_for_nodes(list(candidate_ids)):
            if edge.src_id in candidate_ids and edge.dst_id in candidate_ids:
                weight = max(float(edge.strength_score) * 3.0, 2.0)
                add_edge(edge.src_id, edge.dst_id, weight)
                entity_gate_decisions.append(
                    {
                        "left_node_id": edge.src_id,
                        "right_node_id": edge.dst_id,
                        "similarity": None,
                        "decision": "allowed",
                        "reason": "explicit_edge_fused",
                        "relation": edge.relation,
                        "weight": weight,
                    }
                )

        available_ids = [node.id for node in candidates if node.embedding is not None]
        cached_neighbor_rows = self.repository.list_semantic_neighbors(
            available_ids,
            min_similarity=similarity_threshold,
        )
        cached_node_ids = {node_id for node_id, _neighbor_id, _similarity in cached_neighbor_rows}
        missing_cache_ids = [node_id for node_id in available_ids if node_id not in cached_node_ids]
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
            if similarity < similarity_threshold:
                skipped_unions.append(
                    {
                        "left_node_id": node_id,
                        "right_node_id": neighbor_id,
                        "similarity": similarity,
                        "reason": "below_semantic_threshold",
                    }
                )
                continue
            if neighbor_id not in candidate_ids:
                skipped_unions.append(
                    {
                        "left_node_id": node_id,
                        "right_node_id": neighbor_id,
                        "similarity": similarity,
                        "reason": "neighbor_not_in_candidate_set",
                    }
                )
                continue
            if not self._can_union_semantic_neighbors(
                rows_by_id[node_id],
                rows_by_id[neighbor_id],
                similarity=similarity,
                entityless_similarity_threshold=(
                    self.governance.settings.entityless_union_similarity_threshold
                ),
            ):
                decision = {
                    "left_node_id": node_id,
                    "right_node_id": neighbor_id,
                    "similarity": similarity,
                    "decision": "blocked",
                    "reason": "entity_gate_rejected_union",
                    "left_entities": metadata_entities(rows_by_id[node_id].metadata),
                    "right_entities": metadata_entities(rows_by_id[neighbor_id].metadata),
                }
                skipped_unions.append(decision)
                entity_gate_decisions.append(decision)
                continue
            entity_gate_decisions.append(
                {
                    "left_node_id": node_id,
                    "right_node_id": neighbor_id,
                    "similarity": similarity,
                    "decision": "allowed",
                    "reason": "entity_gate_allowed_union",
                    "left_entities": metadata_entities(rows_by_id[node_id].metadata),
                    "right_entities": metadata_entities(rows_by_id[neighbor_id].metadata),
                    **self._semantic_bridge_decision(
                        rows_by_id[node_id],
                        rows_by_id[neighbor_id],
                        pass_index=pass_index,
                        bridge_edge_weight_multiplier=bridge_edge_weight_multiplier,
                    ),
                }
            )
            add_edge(
                node_id,
                neighbor_id,
                float(similarity)
                * self._semantic_bridge_weight_multiplier(
                    rows_by_id[node_id],
                    rows_by_id[neighbor_id],
                    bridge_edge_weight_multiplier=bridge_edge_weight_multiplier,
                ),
            )

        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            return self._build_clusters_union_find_fallback(
                candidates,
                min_cluster_size=min_cluster_size,
                edge_pairs=[
                    (candidates[left_idx].id, candidates[right_idx].id)
                    for left_idx, right_idx in edge_weights
                ],
                skipped_unions=skipped_unions,
                entity_gate_decisions=entity_gate_decisions,
            )

        edges_list = list(edge_weights.keys())
        weights_list = [edge_weights[edge] for edge in edges_list]
        graph = ig.Graph(n=len(candidates), edges=edges_list, directed=False)
        graph.es["weight"] = weights_list
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=leiden_resolution,
            seed=0,
        )

        clusters = []
        for component_indices in partition:
            component_nodes = [candidates[idx] for idx in component_indices]
            if len(component_nodes) >= min_cluster_size:
                clusters.append(component_nodes)
        return clusters, skipped_unions, entity_gate_decisions

    @classmethod
    def _semantic_bridge_decision(
        cls,
        left: NodeRecord,
        right: NodeRecord,
        *,
        pass_index: int,
        bridge_edge_weight_multiplier: float,
    ) -> dict[str, Any]:
        left_entities = cls._node_gate_entities(left)
        right_entities = cls._node_gate_entities(right)
        shared_entities = sorted(left_entities & right_entities)
        bridge_risk = bool(
            shared_entities
            and (left_entities - right_entities)
            and (right_entities - left_entities)
        )
        weight_multiplier = (
            bridge_edge_weight_multiplier if bridge_risk else 1.0
        )
        return {
            "pass_index": pass_index,
            "bridge_risk": bridge_risk,
            "shared_entities": shared_entities,
            "left_unique_entities": sorted(left_entities - right_entities),
            "right_unique_entities": sorted(right_entities - left_entities),
            "bridge_weight_multiplier": weight_multiplier,
        }

    @classmethod
    def _semantic_bridge_weight_multiplier(
        cls,
        left: NodeRecord,
        right: NodeRecord,
        *,
        bridge_edge_weight_multiplier: float,
    ) -> float:
        decision = cls._semantic_bridge_decision(
            left,
            right,
            pass_index=0,
            bridge_edge_weight_multiplier=bridge_edge_weight_multiplier,
        )
        return float(decision["bridge_weight_multiplier"])

    @staticmethod
    def _build_clusters_union_find_fallback(
        candidates: list[NodeRecord],
        *,
        min_cluster_size: int,
        edge_pairs: list[tuple[str, str]],
        skipped_unions: list[dict[str, Any]],
        entity_gate_decisions: list[dict[str, Any]],
    ) -> tuple[list[list[NodeRecord]], list[dict[str, Any]], list[dict[str, Any]]]:
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

        for left_id, right_id in edge_pairs:
            union(left_id, right_id)

        grouped: dict[str, list[NodeRecord]] = defaultdict(list)
        for node in candidates:
            grouped[find(node.id)].append(rows_by_id[node.id])
        clusters = [
            component for component in grouped.values() if len(component) >= min_cluster_size
        ]
        return clusters, skipped_unions, entity_gate_decisions

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
        entities = self._cluster_entities(cluster)
        if entities and not self._summary_mentions_entity(summary, entities):
            summary = f"{', '.join(entities)}: {summary}"
        return summary[:500]

    def _make_heuristic_super_node_description(self, cluster: list[NodeRecord]) -> str:
        joined = "; ".join(node.description for node in cluster if node.description)
        entities = self._cluster_entities(cluster)
        if entities and not self._summary_mentions_entity(joined, entities):
            joined = f"{', '.join(entities)}: {joined}"
        if len(joined) <= 500:
            return joined
        return joined[:497].rstrip() + "..."

    def _make_super_node_content(self, cluster: list[NodeRecord]) -> str:
        entities = self._cluster_entities(cluster)
        lines = [
            "Dream consolidation result.",
            "",
            f"Cluster entities: {', '.join(entities) if entities else 'none'}",
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
    def _node_entities(node: NodeRecord) -> list[str]:
        return metadata_entities(node.metadata)

    @classmethod
    def _cluster_entities(cls, cluster: list[NodeRecord]) -> list[str]:
        entities: list[str] = []
        for node in cluster:
            entities.extend(cls._node_entities(node))
        return normalize_entities(entities)

    @classmethod
    def _can_union_semantic_neighbors(
        cls,
        left: NodeRecord,
        right: NodeRecord,
        *,
        similarity: float,
        entityless_similarity_threshold: float,
    ) -> bool:
        left_entities = cls._node_gate_entities(left)
        right_entities = cls._node_gate_entities(right)
        if left_entities or right_entities:
            return bool(left_entities & right_entities)
        return similarity >= entityless_similarity_threshold

    @staticmethod
    def _summary_mentions_entity(summary: str, entities: list[str]) -> bool:
        lowered = summary.lower()
        return any(entity.lower() in lowered for entity in entities)

    @classmethod
    def _node_gate_entities(cls, node: NodeRecord) -> set[str]:
        return {entity.lower() for entity in cls._node_entities(node)}

    def _make_host_compaction_prompt(
        self,
        *,
        cluster: list[NodeRecord],
        suggested_title: str,
        prepared_content: str,
    ) -> str:
        source_ids = ", ".join(node.id for node in cluster)
        entities = self._cluster_entities(cluster)
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
                (
                    "- Mention every cluster entity in the description: "
                    f"{', '.join(entities) if entities else 'none'}."
                ),
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
