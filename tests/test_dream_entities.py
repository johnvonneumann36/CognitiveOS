from pathlib import Path

from cognitiveos.config import AppSettings
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.dream import DreamCompiler
from cognitiveos.metadata_shapes import extract_node_entities
from cognitiveos.models import AddPayloadType, NodeRecord
from cognitiveos.service import CognitiveOSService


class SameEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _text in texts]


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
    assert extract_node_entities(
        name=None,
        description="feedray packaging",
        content="",
        tags=["feedray"],
    ) == []
    assert extract_node_entities(
        name=None,
        description="ordinary runtime memory",
        content="",
        tags=[],
    ) == []


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
