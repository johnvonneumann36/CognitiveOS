from pathlib import Path

from cognitiveos.config import AppSettings
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.dream import DreamCompiler
from cognitiveos.metadata_shapes import extract_node_entities
from cognitiveos.models import AddPayloadType, EdgeRecord, NodeRecord
from cognitiveos.service import CognitiveOSService


class SameEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _text in texts]


class CascadeEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            lowered = text.lower()
            if "dream consolidation result" in lowered:
                embeddings.append([1.0, 1.0, 0.0])
            elif "restaurant" in lowered:
                embeddings.append([1.0, 0.0, 0.0])
            elif "visa" in lowered:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return embeddings


class EmptySummaryChatProvider:
    def summarize(self, _content: str) -> str:
        return "compressed summary"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if '"entities"' in system_prompt:
            if "cognitiveos" in user_prompt.lower():
                return '{"entities":["CognitiveOS"]}'
            if "feedray" in user_prompt.lower():
                return '{"entities":["FeedRay"]}'
            return '{"entities":[]}'
        return "synthetic document description"


def build_service(tmp_path: Path) -> tuple[CognitiveOSService, AppSettings]:
    settings = AppSettings.from_env(
        db_path=tmp_path / "cognitiveos.db",
        memory_output_path=tmp_path / "MEMORY.MD",
        project_root=tmp_path,
    )
    settings.similarity_threshold = 0.6
    settings.dream_event_threshold = 50
    service = CognitiveOSService(
        settings=settings,
        repository=SQLiteRepository(settings.db_path),
        embedding_provider=SameEmbeddingProvider(),
        chat_provider=EmptySummaryChatProvider(),
    )
    service.initialize()
    return service, settings


def test_extract_node_entities_uses_shape_only_fallback() -> None:
    assert extract_node_entities(
        name=None,
        description="CognitiveOS release work with LM Studio and CLIProxyAPI",
        content="",
        tags=[],
    ) == ["CognitiveOS", "LM Studio", "CLIProxyAPI"]
    assert (
        extract_node_entities(
            name=None,
            description="feedray packaging",
            content="",
            tags=["feedray"],
        )
        == []
    )
    assert (
        extract_node_entities(
            name=None,
            description="ordinary runtime memory",
            content="",
            tags=[],
        )
        == []
    )


def test_service_entity_extraction_uses_chat_before_heuristic(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)

    receipt = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="ordinary runtime memory for cognitiveos",
        tags=[],
        force=True,
    )
    node = service.repository.get_node(receipt.node_id)

    assert node.metadata["entities"] == ["CognitiveOS"]


def test_repository_does_not_infer_entities_without_service(tmp_path: Path) -> None:
    repository = SQLiteRepository(tmp_path / "cognitiveos.db")
    repository.initialize()
    node = NodeRecord(
        id="node_direct",
        description="CognitiveOS runtime work",
        content="CognitiveOS runtime work",
        tags=["runtime"],
        metadata={},
    )

    repository.create_node(node, actor="test", action_type="create")
    stored = repository.get_node("node_direct")

    assert "entities" not in stored.metadata


def test_dream_keeps_same_embedding_clusters_separated_by_entity(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path)
    for payload, tags in [
        ("CognitiveOS release workflow and memory runtime", ["project", "cognitiveos"]),
        ("CognitiveOS host bootstrap and MCP memory tooling", ["project", "cognitiveos"]),
        ("FeedRay release workflow and package runtime", ["project", "feedray"]),
        ("Feedray event pipeline and package tooling", ["project", "feedray"]),
    ]:
        service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload=payload,
            tags=tags,
            force=True,
        )

    result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.6,
    )

    assert result.clusters_created == 2
    entity_sets = {
        tuple(service.repository.get_node(item.node_id).metadata["entities"])
        for item in result.super_nodes
    }
    assert entity_sets == {("CognitiveOS",), ("FeedRay",)}


def test_dream_fuses_explicit_edges_as_structural_weight(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path)
    left = NodeRecord(
        id="manual_left",
        description="FeedRay release note",
        content="FeedRay release note",
        metadata={"entities": ["FeedRay"]},
    )
    right = NodeRecord(
        id="manual_right",
        description="CognitiveOS release note",
        content="CognitiveOS release note",
        metadata={"entities": ["CognitiveOS"]},
    )
    service.repository.create_node(left, actor="test", action_type="create")
    service.repository.create_node(right, actor="test", action_type="create")
    service.repository.create_edge(
        EdgeRecord(
            src_id=left.id,
            dst_id=right.id,
            relation="manual_link",
            strength_score=1.0,
        )
    )

    result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=0,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.99,
    )

    assert result.clusters_created == 1
    assert any(
        decision["reason"] == "explicit_edge_fused"
        for decision in result.entity_gate_decisions
    )


def test_dream_cascade_reclusters_new_super_nodes(tmp_path: Path) -> None:
    settings = AppSettings.from_env(
        db_path=tmp_path / "cognitiveos.db",
        memory_output_path=tmp_path / "MEMORY.MD",
        project_root=tmp_path,
    )
    settings.dream_event_threshold = 50
    service = CognitiveOSService(
        settings=settings,
        repository=SQLiteRepository(settings.db_path),
        embedding_provider=CascadeEmbeddingProvider(),
        chat_provider=EmptySummaryChatProvider(),
    )
    service.initialize()
    for node in [
        NodeRecord(
            id="restaurant_a",
            description="Cebu trip restaurant shortlist",
            content="Cebu trip restaurant shortlist",
            embedding=[1.0, 0.0, 0.0],
            metadata={"entities": ["Cebu Trip", "Restaurants"]},
        ),
        NodeRecord(
            id="restaurant_b",
            description="Cebu trip restaurant booking",
            content="Cebu trip restaurant booking",
            embedding=[1.0, 0.0, 0.0],
            metadata={"entities": ["Cebu Trip", "Restaurants"]},
        ),
        NodeRecord(
            id="visa_a",
            description="Cebu trip visa requirement",
            content="Cebu trip visa requirement",
            embedding=[0.0, 1.0, 0.0],
            metadata={"entities": ["Cebu Trip", "Visa"]},
        ),
        NodeRecord(
            id="visa_b",
            description="Cebu trip visa checklist",
            content="Cebu trip visa checklist",
            embedding=[0.0, 1.0, 0.0],
            metadata={"entities": ["Cebu Trip", "Visa"]},
        ),
    ]:
        service.repository.create_node(node, actor="test", action_type="create")

    result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=0,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.6,
        cascade_passes=2,
        cascade_threshold_step=0.15,
    )

    assert result.clusters_created == 3
    assert result.cluster_explanations[0]["leiden_resolution"] == 1.75
    assert any(
        explanation["pass_index"] == 1
        and explanation["decision"] == "eligible_for_cascade_compaction"
        and explanation["leiden_resolution"] == 1.35
        for explanation in result.cluster_explanations
    )


def test_entityless_semantic_neighbors_need_high_similarity() -> None:
    left = NodeRecord(
        id="left",
        description="ordinary runtime memory",
        content="ordinary runtime memory",
    )
    right = NodeRecord(
        id="right",
        description="general tooling memory",
        content="general tooling memory",
    )

    assert not DreamCompiler._can_union_semantic_neighbors(
        left,
        right,
        similarity=0.79,
        entityless_similarity_threshold=0.8,
    )
    assert DreamCompiler._can_union_semantic_neighbors(
        left,
        right,
        similarity=0.8,
        entityless_similarity_threshold=0.8,
    )


def test_entity_gate_uses_primary_boundary_entity_not_incidental_mentions() -> None:
    feedray = NodeRecord(
        id="feedray",
        description="FeedRay repository reset matched the CognitiveOS git config.",
        content="FeedRay repository reset matched the CognitiveOS git config.",
        tags=["feedray"],
        metadata={"entities": ["FeedRay"]},
    )
    cognitiveos = NodeRecord(
        id="cognitiveos",
        description="CognitiveOS release workflow",
        content="CognitiveOS release workflow",
        tags=["cognitiveos"],
        metadata={"entities": ["CognitiveOS"]},
    )

    assert not DreamCompiler._can_union_semantic_neighbors(
        feedray,
        cognitiveos,
        similarity=0.99,
        entityless_similarity_threshold=0.8,
    )


def test_projection_skips_superseded_super_nodes(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path)
    for payload, tags in [
        ("CognitiveOS release workflow and memory runtime", ["project", "cognitiveos"]),
        ("CognitiveOS host bootstrap and MCP memory tooling", ["project", "cognitiveos"]),
    ]:
        service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload=payload,
            tags=tags,
            force=True,
        )

    result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.6,
    )
    super_node = service.repository.get_node(result.super_nodes[0].node_id)
    superseded = super_node.model_copy(
        update={"metadata": {**super_node.metadata, "projection_status": "superseded"}}
    )
    service.repository.overwrite_node(superseded, actor="test", action_type="update")

    memory_text = service.compile_memory_snapshot().read_text(encoding="utf-8")

    assert "Compressed Dream Memory" not in memory_text
    assert "CognitiveOS release workflow" not in memory_text


def test_dream_result_explains_run_and_projection(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path)
    for payload in [
        "CognitiveOS release workflow and memory runtime",
        "CognitiveOS host bootstrap and MCP memory tooling",
    ]:
        service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload=payload,
            tags=["project", "cognitiveos"],
            force=True,
        )

    result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.6,
    )
    super_node = service.repository.get_node(result.super_nodes[0].node_id)

    assert result.effective_config["semantic_threshold"] == 0.6
    assert result.effective_config["entityless_threshold"] == 0.8
    assert result.effective_config["leiden_resolution_start"] == 1.75
    assert len(result.candidate_explanations) == 2
    assert result.cluster_explanations[0]["decision"] == "eligible_for_compaction"
    assert result.entity_gate_decisions[0]["decision"] == "allowed"
    assert result.projected_memory["projection_policy_version"] == "memory-projection-v1"
    assert result.projected_memory["max_projected_super_nodes"] == 5
    assert result.projected_memory["compressed_node_ids"] == [super_node.id]
    assert super_node.metadata["projection_policy_version"] == "memory-projection-v1"
