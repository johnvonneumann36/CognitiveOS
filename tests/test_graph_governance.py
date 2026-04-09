from pathlib import Path

from cognitiveos.config import AppSettings
from cognitiveos.db.connection import open_connection
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.metadata_shapes import (
    metadata_source_ref,
)
from cognitiveos.models import AddPayloadType
from cognitiveos.service import CognitiveOSService


class FakeEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            embeddings.append(
                [
                    1.0 if "graph" in lowered else 0.0,
                    1.0 if "memory" in lowered else 0.0,
                    1.0 if "governance" in lowered else 0.0,
                    1.0 if "evidence" in lowered else 0.0,
                ]
            )
        return embeddings


class FakeChatProvider:
    def summarize(self, _content: str) -> str:
        return "synthetic summary"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if '"tags"' in system_prompt:
            return '{"tags":["graph governance","memory runtime","evidence flow"]}'
        return (
            "Adaptive governance guide for graph memory retrieval, evidence handling, "
            "and host runtime operation."
        )


def build_service(
    tmp_path: Path,
    *,
    embedding_provider: FakeEmbeddingProvider | None = None,
    chat_provider: FakeChatProvider | None = None,
    relationship_weak_after_hours: int = 72,
    relationship_stale_after_hours: int = 168,
) -> CognitiveOSService:
    settings = AppSettings.from_env(
        db_path=tmp_path / "cognitiveos.db",
        memory_output_path=tmp_path / "MEMORY.MD",
    )
    settings.relationship_weak_after_hours = relationship_weak_after_hours
    settings.relationship_stale_after_hours = relationship_stale_after_hours
    service = CognitiveOSService(
        settings=settings,
        repository=SQLiteRepository(settings.db_path),
        embedding_provider=embedding_provider,
        chat_provider=chat_provider,
    )
    service.initialize()
    return service


def write_long_markdown(path: Path, *, sections: int = 6, suffix: str = "") -> Path:
    blocks: list[str] = []
    for index in range(1, sections + 1):
        body = (
            "Adaptive graph governance keeps durable memory separate from evidence chunks. "
            "Relationship scoring, chunk lineage, and dream safety all matter for retrieval. "
            "This section exists to create a long markdown file with repeated governance detail. "
        ) * 20
        tail = f"\n\nUnique marker section-{index}{suffix}\n"
        blocks.append(f"# Section {index}\n\n{body}{tail}")
    path.write_text("\n\n".join(blocks), encoding="utf-8")
    return path


def test_file_ingestion_keeps_original_content_and_generates_tags(tmp_path: Path) -> None:
    service = build_service(tmp_path, chat_provider=FakeChatProvider())
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md")

    receipt = service.add_node(
        payload_type=AddPayloadType.FILE,
        payload=str(document_path),
        name="Adaptive Governance Plan",
    )

    source_node = service.repository.get_node(receipt.node_id)
    assert source_node.node_type == "source_document"
    assert source_node.durability == "durable"
    assert metadata_source_ref(source_node.metadata) == str(document_path.resolve())
    assert source_node.content == document_path.read_text(encoding="utf-8").strip()
    assert source_node.description.startswith("Adaptive governance guide")
    assert source_node.tags == ["graph governance", "memory runtime", "evidence flow"]
    assert "chunk_count" not in source_node.metadata.get("document", {})
    assert source_node.metadata["document_profile"]["sync_status"] == "source_synced"
    assert source_node.metadata["document_profile"]["source_state"] == "attached"
    assert source_node.metadata["document_profile"]["last_update_mode"] == "source_add"

    relationships = service.list_relationships(node_id=source_node.id)
    contains_edges = [
        edge
        for edge in relationships
        if edge.src_id == source_node.id and edge.relation == "contains"
    ]
    assert not contains_edges


def test_file_ingestion_blocks_duplicate_source_metadata(tmp_path: Path) -> None:
    service = build_service(tmp_path, chat_provider=FakeChatProvider())
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md", sections=2)

    first = service.add_node(payload_type=AddPayloadType.FILE, payload=str(document_path))
    second = service.add_node(payload_type=AddPayloadType.FILE, payload=str(document_path))

    assert first.status == "success"
    assert second.status == "blocked"
    assert "matching source metadata" in (second.reason or "").lower()


def test_file_ingestion_blocks_same_filename_and_modified_time(tmp_path: Path) -> None:
    service = build_service(tmp_path, chat_provider=FakeChatProvider())
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = write_long_markdown(first_dir / "shared.md", sections=2, suffix="-shared")
    second_path = write_long_markdown(second_dir / "shared.md", sections=2, suffix="-shared")

    fixed_mtime = 1_700_000_000
    first_path.touch()
    second_path.touch()
    import os

    os.utime(first_path, (fixed_mtime, fixed_mtime))
    os.utime(second_path, (fixed_mtime, fixed_mtime))

    first = service.add_node(payload_type=AddPayloadType.FILE, payload=str(first_path))
    second = service.add_node(payload_type=AddPayloadType.FILE, payload=str(second_path))

    assert first.status == "success"
    assert second.status == "blocked"


def test_file_ingestion_blocks_same_source_hash(tmp_path: Path) -> None:
    service = build_service(tmp_path, chat_provider=FakeChatProvider())
    first_path = write_long_markdown(tmp_path / "first.md", sections=2, suffix="-hash")
    second_path = tmp_path / "second.md"
    second_path.write_text(first_path.read_text(encoding="utf-8"), encoding="utf-8")

    first = service.add_node(payload_type=AddPayloadType.FILE, payload=str(first_path))
    second = service.add_node(payload_type=AddPayloadType.FILE, payload=str(second_path))

    assert first.status == "success"
    assert second.status == "blocked"


def test_search_returns_source_documents_without_evidence_chunks(tmp_path: Path) -> None:
    service = build_service(tmp_path, chat_provider=FakeChatProvider())
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md")
    service.add_node(payload_type=AddPayloadType.FILE, payload=str(document_path))

    default_results = service.search(
        keyword="governance",
        top_k=20,
        include_neighbors=0,
    )
    assert default_results
    assert all(result.node_type == "source_document" for result in default_results)

    evidence_results = service.search(
        keyword="governance",
        top_k=20,
        include_neighbors=0,
        include_evidence=True,
    )
    assert evidence_results
    assert all(result.node_type == "source_document" for result in evidence_results)


def test_keyword_search_does_not_index_raw_source_document_content(tmp_path: Path) -> None:
    service = build_service(tmp_path, chat_provider=FakeChatProvider())
    document_path = tmp_path / "raw-only.md"
    raw_only_token = "ZXQ491UNIQUERAWCONTENTTOKEN"
    document_path.write_text(
        f"# Hidden Payload\n\n{raw_only_token} only exists in body text.",
        encoding="utf-8",
    )
    service.add_node(payload_type=AddPayloadType.FILE, payload=str(document_path))

    results = service.search(
        keyword=raw_only_token,
        top_k=10,
        include_neighbors=0,
    )
    assert results == []


def test_relationship_governance_and_memory_projection(tmp_path: Path) -> None:
    service = build_service(tmp_path)

    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Adaptive governance preserves durable memory.",
        force=True,
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Evidence chunks should not dominate ordinary recall.",
        force=True,
        name="Beta",
    )
    working = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Transient working memory.",
        force=True,
        name="Working Node",
    )
    assert alpha.node_id and beta.node_id and working.node_id

    service.link_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="supports")
    reinforce = service.reinforce_relationship(
        src_id=alpha.node_id,
        dst_id=beta.node_id,
        relation="supports",
        delta=0.5,
    )
    assert reinforce.action_taken == "edge_reinforced"

    relationships = service.list_relationships(node_id=alpha.node_id, relation="supports")
    assert relationships[0].strength_score >= 1.5

    unlink = service.unlink_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="supports")
    assert unlink.action_taken == "edge_removed"
    assert service.list_relationships(node_id=alpha.node_id, relation="supports") == []

    service.update_node(
        node_id=alpha.node_id,
        content="Adaptive governance preserves durable memory.",
        durability="durable",
    )
    service.pin_memory(node_id=beta.node_id)
    memory_path = service.compile_memory_snapshot()
    memory_text = memory_path.read_text(encoding="utf-8")
    assert "Alpha" in memory_text
    assert "Beta" in memory_text
    assert "Working Node" not in memory_text


def test_dream_treats_source_documents_as_candidates(tmp_path: Path) -> None:
    service = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md")
    source_receipt = service.add_node(
        payload_type=AddPayloadType.FILE,
        payload=str(document_path),
        name="Adaptive Governance Plan",
    )
    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph governance memory runtime",
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph governance evidence retrieval",
        force=True,
        name="Node B",
    )
    assert source_receipt.node_id and a.node_id and b.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="related_to")
    service.search(keyword="governance", top_k=10, include_neighbors=0, include_evidence=True)

    dream_result = service.run_dream(
        output_path=tmp_path / "MEMORY.MD",
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=50,
        similarity_threshold=0.5,
    )
    assert source_receipt.node_id in dream_result.candidate_node_ids
    assert dream_result.durability_suggestions
    assert any(
        suggestion.recommended_durability == "durable"
        for suggestion in dream_result.durability_suggestions
    )


def test_refresh_source_document_reextracts_source_document(tmp_path: Path) -> None:
    service = build_service(tmp_path, chat_provider=FakeChatProvider())
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md", sections=6, suffix="-v1")
    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=str(document_path))
    source_node = service.repository.get_node(receipt.node_id)
    write_long_markdown(document_path, sections=3, suffix="-v2")
    refreshed_receipt = service.refresh_source_document(node_id=source_node.id)
    assert refreshed_receipt.action_taken == "source_document_refreshed"

    refreshed = service.repository.get_node(source_node.id)
    assert refreshed.metadata["source"]["hash"] != source_node.metadata["source"]["hash"]
    assert refreshed.metadata["document_profile"]["sync_status"] == "source_synced"
    assert refreshed.metadata["document_profile"]["source_state"] == "attached"
    assert refreshed.metadata["document_profile"]["last_update_mode"] == "source_refresh"
    assert "chunk_count" not in refreshed.metadata.get("document", {})
    assert refreshed.content == document_path.read_text(encoding="utf-8").strip()


def test_memory_projection_includes_long_source_document_without_summary_node(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path, chat_provider=FakeChatProvider())
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md")
    receipt = service.add_node(
        payload_type=AddPayloadType.FILE,
        payload=str(document_path),
        name="Adaptive Governance Plan",
    )
    source_node = service.repository.get_node(receipt.node_id)

    memory_path = service.compile_memory_snapshot()
    memory_text = memory_path.read_text(encoding="utf-8")
    assert "## Durable Source Memory" in memory_text
    assert f"- {source_node.name}:" in memory_text


def test_document_embedding_uses_description_and_generated_tags(tmp_path: Path) -> None:
    service = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md", sections=2)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=str(document_path))
    source_node = service.repository.get_node(receipt.node_id)

    expected_embedding = service.embedding_provider.embed(
        [f"{source_node.description}\n\nTags: {', '.join(source_node.tags)}"]
    )[0]
    assert source_node.embedding == expected_embedding


def test_source_document_update_marks_manual_override(tmp_path: Path) -> None:
    service = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md", sections=2)
    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=str(document_path))
    source_node = service.repository.get_node(receipt.node_id)

    update = service.update_node(
        node_id=source_node.id,
        content=source_node.content + "\n\nManual addendum.",
    )
    assert update.action_taken == "updated"

    refreshed = service.repository.get_node(source_node.id)
    assert refreshed.metadata["source"]["ref"] == source_node.metadata["source"]["ref"]
    assert refreshed.metadata["source"]["hash"] == source_node.metadata["source"]["hash"]
    assert refreshed.metadata["document_profile"]["sync_status"] == "detached"
    assert refreshed.metadata["document_profile"]["source_state"] == "detached"
    assert refreshed.metadata["document_profile"]["last_update_mode"] == "manual_override"


def test_source_document_update_preserves_user_tags_and_regenerates_generated_tags(
    tmp_path: Path,
) -> None:
    service = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )
    document_path = write_long_markdown(tmp_path / "adaptive-plan.md", sections=2)
    receipt = service.add_node(
        payload_type=AddPayloadType.FILE,
        payload=str(document_path),
        tags=["user-tag"],
    )
    source_node = service.repository.get_node(receipt.node_id)

    service.update_node(
        node_id=source_node.id,
        content=source_node.content,
        tags=["user-tag", "manual-tag"],
    )
    refreshed = service.repository.get_node(source_node.id)

    assert "user-tag" in refreshed.tags
    assert "manual-tag" in refreshed.tags
    assert "graph governance" in refreshed.tags
    assert refreshed.metadata["document_profile"]["sync_status"] == "source_synced"
    assert refreshed.metadata["document_profile"]["source_state"] == "attached"
    assert refreshed.metadata["document_profile"]["last_update_mode"] == "profile_refresh"


def test_dream_emits_relationship_cleanup_plans(tmp_path: Path) -> None:
    service = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
        relationship_weak_after_hours=1,
        relationship_stale_after_hours=2,
    )
    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph governance memory runtime",
        force=True,
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph governance evidence retrieval",
        force=True,
        name="Beta",
    )
    gamma = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="External stale link target",
        force=True,
        name="Gamma",
    )
    assert alpha.node_id and beta.node_id and gamma.node_id
    service.link_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="related_to")
    service.link_nodes(src_id=alpha.node_id, dst_id=gamma.node_id, relation="depends_on")

    with open_connection(service.settings.db_path) as connection:
        connection.execute(
            """
            UPDATE edges
            SET status = 'stale',
                strength_score = 0.1,
                created_at = DATETIME('now', '-3 hours'),
                last_reinforced_at = DATETIME('now', '-3 hours')
            WHERE src_id = ? AND dst_id = ? AND relation = 'depends_on'
            """,
            (alpha.node_id, gamma.node_id),
        )

    service.search(keyword="graph", top_k=10, include_neighbors=0)
    dream_result = service.run_dream(
        output_path=tmp_path / "MEMORY.MD",
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.5,
    )
    assert dream_result.relationship_cleanup_plans
    assert any(
        plan.recommended_action in {"redirect_or_prune", "prune", "collapse_into_super_node"}
        for plan in dream_result.relationship_cleanup_plans
    )


def test_relationship_decay_transitions_and_prune_report(tmp_path: Path) -> None:
    service = build_service(
        tmp_path,
        relationship_weak_after_hours=1,
        relationship_stale_after_hours=2,
    )
    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Alpha relationship source.",
        force=True,
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Beta relationship target.",
        force=True,
        name="Beta",
    )
    assert alpha.node_id and beta.node_id
    service.link_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="supports")

    with open_connection(service.settings.db_path) as connection:
        connection.execute(
            """
            UPDATE edges
            SET created_at = DATETIME('now', '-3 hours'),
                last_reinforced_at = DATETIME('now', '-3 hours')
            """
        )

    relationships = service.list_relationships(node_id=alpha.node_id, relation="supports")
    assert relationships[0].status == "stale"
    assert relationships[0].strength_score < 1.0

    preview = service.prune_relationships(dry_run=True)
    assert preview["governance"]["status_counts"]["stale"] >= 1
    assert preview["candidate_count"] >= 1
    assert preview["candidates"][0]["prune_reason"] == "status=stale"
    assert preview["removed_count"] == 0

    executed = service.prune_relationships(dry_run=False)
    assert executed["removed_count"] >= 1
    assert service.list_relationships(node_id=alpha.node_id, relation="supports") == []


def test_relationship_becomes_weak_before_stale(tmp_path: Path) -> None:
    service = build_service(
        tmp_path,
        relationship_weak_after_hours=1,
        relationship_stale_after_hours=4,
    )
    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Alpha relationship source.",
        force=True,
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Beta relationship target.",
        force=True,
        name="Beta",
    )
    assert alpha.node_id and beta.node_id
    service.link_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="supports")

    with open_connection(service.settings.db_path) as connection:
        connection.execute(
            """
            UPDATE edges
            SET created_at = DATETIME('now', '-2 hours'),
                last_reinforced_at = DATETIME('now', '-2 hours')
            """
        )

    relationship = service.list_relationships(node_id=alpha.node_id, relation="supports")[0]
    assert relationship.status == "weak"


def test_repeated_link_and_read_reinforce_relationships(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Alpha relationship source.",
        force=True,
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Beta relationship target.",
        force=True,
        name="Beta",
    )
    assert alpha.node_id and beta.node_id

    first = service.link_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="supports")
    second = service.link_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="supports")
    assert first.action_taken == "edge_created"
    assert second.action_taken == "edge_reinforced"

    before_read = service.list_relationships(node_id=alpha.node_id, relation="supports")[0]
    service.read_nodes([alpha.node_id, beta.node_id], include_content=False)
    after_read = service.list_relationships(node_id=alpha.node_id, relation="supports")[0]
    assert after_read.strength_score > before_read.strength_score
