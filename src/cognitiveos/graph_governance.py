from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cognitiveos.config import AppSettings
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.metadata_shapes import metadata_profile_kind
from cognitiveos.models import (
    DreamDurabilitySuggestion,
    DreamRelationshipCleanupPlan,
    EdgeRecord,
    NodeRecord,
)


class GraphGovernanceEngine:
    def __init__(
        self,
        *,
        settings: AppSettings,
        repository: SQLiteRepository,
        default_actor: str,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.default_actor = default_actor

    def apply_relationship_governance(
        self,
        *,
        node_id: str | None = None,
    ) -> dict[str, Any]:
        return self.repository.transition_relationship_states(
            weak_after_hours=self.settings.relationship_weak_after_hours,
            stale_after_hours=self.settings.relationship_stale_after_hours,
            weak_strength_threshold=self.settings.relationship_weak_strength_threshold,
            stale_strength_threshold=self.settings.relationship_stale_strength_threshold,
            weak_decay_delta=self.settings.relationship_weak_decay_delta,
            stale_decay_delta=self.settings.relationship_stale_decay_delta,
            node_id=node_id,
        )

    def reinforce_read_coaccess(self, node_ids: list[str]) -> None:
        self.repository.reinforce_edges_between_nodes(
            node_ids,
            delta=self.settings.relationship_recall_reinforcement_delta,
            actor=self.default_actor,
            reason="read_coaccess",
        )

    def reinforce_cluster_relationships(self, node_ids: list[str]) -> None:
        self.repository.reinforce_edges_between_nodes(
            node_ids,
            delta=self.settings.relationship_dream_reinforcement_delta,
            actor="dream",
            reason="dream_cluster",
        )

    def create_or_reinforce_manual_link(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation: str,
    ) -> tuple[str, dict[str, Any]]:
        existing = self.repository.get_edge(src_id, dst_id, relation)
        if existing is None:
            edge = EdgeRecord(
                src_id=src_id,
                dst_id=dst_id,
                relation=relation,
                strength_score=1.0,
                durability="durable",
                status="active",
                metadata={
                    "provenance": {
                        "created_by": self.default_actor,
                        "creation_mode": "manual",
                    }
                },
                last_reinforced_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            )
            self.repository.create_edge(edge)
            return (
                "edge_created",
                {"src": src_id, "dst": dst_id, "relation": relation},
            )

        edge = EdgeRecord(
            src_id=src_id,
            dst_id=dst_id,
            relation=relation,
            strength_score=(
                existing.strength_score + self.settings.relationship_manual_reinforcement_delta
            ),
            durability=existing.durability,
            status="active",
            metadata={
                **existing.metadata,
                "provenance": {
                    **(existing.metadata.get("provenance", {})),
                    "created_by": self.default_actor,
                    "creation_mode": "manual",
                },
                "reinforced": True,
            },
            last_reinforced_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.repository.create_edge(edge)
        return ("edge_reinforced", edge.model_dump())

    def prune_relationships(
        self,
        *,
        node_id: str | None = None,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        governance = self.apply_relationship_governance(node_id=node_id)
        report = self.repository.prune_relationships(node_id=node_id, dry_run=dry_run)
        report["governance"] = governance
        return report

    def build_relationship_cleanup_plans(
        self,
        candidates: list[NodeRecord],
        clusters: list[list[NodeRecord]],
    ) -> list[DreamRelationshipCleanupPlan]:
        if not candidates:
            return []
        candidate_ids = {node.id for node in candidates}
        clustered_ids = {node.id for cluster in clusters for node in cluster}
        plans: list[DreamRelationshipCleanupPlan] = []
        seen: set[tuple[str, str, str, str]] = set()

        for edge in self.repository.list_edges_for_nodes(list(candidate_ids)):
            if edge.status not in {"weak", "stale"}:
                continue
            if edge.durability == "pinned":
                continue
            touches_cluster = edge.src_id in clustered_ids or edge.dst_id in clustered_ids
            recommended_action = "prune" if edge.status == "stale" else "review"
            reason = (
                "Weak relationship inside dream candidate scope should be reviewed "
                "before future consolidation."
                if edge.status == "weak"
                else "Stale relationship inside dream candidate scope is eligible "
                "for pruning."
            )
            if touches_cluster:
                if edge.src_id in clustered_ids and edge.dst_id in clustered_ids:
                    recommended_action = "collapse_into_super_node"
                    reason = (
                        "Relationship is stale or weak inside a dream cluster and "
                        "may be superseded by the cluster super-node."
                    )
                else:
                    recommended_action = "redirect_or_prune"
                    reason = (
                        "Relationship touches a dream cluster and should be "
                        "redirected to the super-node or pruned."
                    )
            dedupe_key = (edge.src_id, edge.dst_id, edge.relation, recommended_action)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            plans.append(
                DreamRelationshipCleanupPlan(
                    src_id=edge.src_id,
                    dst_id=edge.dst_id,
                    relation=edge.relation,
                    current_edge_status=edge.status,
                    recommended_action=recommended_action,
                    reason=reason,
                    strength_score=edge.strength_score,
                )
            )
        return plans

    @staticmethod
    def build_cluster_durability_suggestions(
        cluster: list[NodeRecord],
        *,
        super_node: NodeRecord,
    ) -> list[DreamDurabilitySuggestion]:
        suggestions: list[DreamDurabilitySuggestion] = [
            DreamDurabilitySuggestion(
                node_id=super_node.id,
                current_durability=super_node.durability,
                recommended_durability="durable",
                reason=(
                    "This super-node compresses a dream cluster. Promote it after "
                    "repeated recall or explicit host confirmation."
                ),
                confidence=0.55,
            )
        ]
        for node in cluster:
            if node.durability in {"durable", "pinned"}:
                continue
            if node.node_type == "source_document":
                continue
            confidence = 0.6 if len(cluster) >= 3 else 0.45
            suggestions.append(
                DreamDurabilitySuggestion(
                    node_id=node.id,
                    current_durability=node.durability,
                    recommended_durability="durable",
                    reason=(
                        "This working-memory node repeatedly participated in a "
                        "dream cluster and may represent stable reusable memory."
                    ),
                    confidence=confidence,
                )
            )
        return suggestions

    def compile_memory_snapshot(self, output_path: Path) -> Path:
        projected_nodes = self.repository.list_nodes_for_memory_projection()
        pinned_nodes = [node for node in projected_nodes if node.durability == "pinned"]
        durable_profile_nodes = [
            node
            for node in projected_nodes
            if node.durability == "durable" and metadata_profile_kind(node.metadata) == "system"
        ]
        synthesized_nodes = [
            node
            for node in projected_nodes
            if node.durability == "durable"
            and node.node_type in {"super_node", "memory"}
            and metadata_profile_kind(node.metadata) != "system"
        ]
        source_nodes = [
            node
            for node in projected_nodes
            if node.node_type in {"source_document", "source_collection"}
            and node.durability == "durable"
        ]
        lines = [
            "# CognitiveOS Memory",
            "",
            "Generated from pinned memory and selected durable synthesized memory.",
            "",
        ]
        if not projected_nodes:
            lines.append("- No durable or pinned memory nodes are available yet.")
            output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return output_path

        self._append_memory_section(lines, "Pinned Memory", pinned_nodes)
        self._append_memory_section(lines, "Durable Profile Memory", durable_profile_nodes)
        self._append_memory_section(lines, "Durable Synthesized Memory", synthesized_nodes)
        self._append_memory_section(lines, "Durable Source Memory", source_nodes)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    @staticmethod
    def _append_memory_section(lines: list[str], title: str, nodes: list[NodeRecord]) -> None:
        if not nodes:
            return
        lines.append(f"## {title}")
        lines.append("")
        for node in nodes:
            node_title = node.name or node.id
            lines.append(f"- {node_title}: {node.description}")
        lines.append("")
