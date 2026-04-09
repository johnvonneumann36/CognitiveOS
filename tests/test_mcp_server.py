import asyncio
import json
from pathlib import Path

from cognitiveos.config import AppSettings
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.mcp.server import build_server
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
                    1.0 if "runtime" in lowered else 0.0,
                    1.0 if "tooling" in lowered else 0.0,
                ]
            )
        return embeddings


def _tool_names(*, db_path: Path, profile: str) -> set[str]:
    settings = AppSettings.from_env(db_path=db_path)
    settings.server_profile = profile
    server = build_server(settings)
    tools = asyncio.run(server.list_tools())
    return {tool.name for tool in tools}


def test_host_core_profile_exposes_only_compact_memory_surface(tmp_path: Path) -> None:
    tool_names = _tool_names(db_path=tmp_path / "host-core.db", profile="host-core")

    assert tool_names == {
        "search",
        "read",
        "add",
        "update",
        "link",
        "unlink",
        "dream",
    }


def test_full_profile_exposes_operator_and_bootstrap_tools(tmp_path: Path) -> None:
    tool_names = _tool_names(db_path=tmp_path / "full.db", profile="full")

    assert {"search", "read", "add", "update", "link", "unlink", "dream"} <= tool_names
    assert {"doctor", "providers_test", "list_relationships"} <= tool_names
    assert {"host_bootstrap_status", "submit_host_onboarding", "bootstrap_host"} <= tool_names
    assert "resolve_dream_compaction" not in tool_names


def test_operator_profile_includes_core_tools_but_avoids_duplicate_dream_resolution_tool(
    tmp_path: Path,
) -> None:
    tool_names = _tool_names(db_path=tmp_path / "operator.db", profile="operator")

    assert {"search", "read", "add", "update", "link", "unlink", "dream"} <= tool_names
    assert {"doctor", "providers_test", "list_relationships"} <= tool_names
    assert "list_dream_runs" not in tool_names
    assert "list_dream_compactions" not in tool_names
    assert "set_memory_durability" not in tool_names
    assert "resolve_dream_compaction" not in tool_names


def test_host_core_dream_tool_can_resolve_pending_compaction(tmp_path: Path) -> None:
    db_path = tmp_path / "host-core-resolution.db"
    settings = AppSettings.from_env(db_path=db_path)
    service = CognitiveOSService(
        settings=settings,
        repository=SQLiteRepository(settings.db_path),
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=None,
    )
    service.initialize()

    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime tooling",
        tags=["tech"],
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime",
        tags=["tech"],
        force=True,
        name="Node B",
    )
    assert a.node_id and b.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="supports")
    service.search(keyword="Graph", top_k=5, include_neighbors=1)

    dream_result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.8,
    )
    assert dream_result.status == "awaiting_host_compaction"
    task_id = dream_result.pending_compactions[0].task_id

    settings.server_profile = "host-core"
    server = build_server(settings)
    response = asyncio.run(
        server.call_tool(
            "dream",
            {
                "task_id": task_id,
                "title": "Graph Runtime Cluster",
                "description": "Merged host-side super node",
                "content": "Graph memory runtime tooling and runtime knowledge were merged.",
                "background": False,
            },
        )
    )
    payload = json.loads(response[0].text)
    assert payload["status"] == "success"
    assert payload["task_id"] == task_id
    assert payload["resolution_backend"] == "host_agent"


def test_host_core_dream_tool_can_inspect_status_and_tasks(tmp_path: Path) -> None:
    db_path = tmp_path / "host-core-inspect.db"
    settings = AppSettings.from_env(db_path=db_path)
    service = CognitiveOSService(
        settings=settings,
        repository=SQLiteRepository(settings.db_path),
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=None,
    )
    service.initialize()

    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime tooling",
        tags=["tech"],
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime",
        tags=["tech"],
        force=True,
        name="Node B",
    )
    assert a.node_id and b.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="supports")
    service.search(keyword="Graph", top_k=5, include_neighbors=1)

    dream_result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.8,
    )
    assert dream_result.run_id is not None

    settings.server_profile = "host-core"
    server = build_server(settings)

    status_response = asyncio.run(server.call_tool("dream", {"inspect": "status"}))
    status_payload = json.loads(status_response[0].text)
    assert status_payload["status"] == "success"
    assert "dream_status" in status_payload

    tasks_response = asyncio.run(
        server.call_tool("dream", {"inspect": "tasks", "run_id": dream_result.run_id})
    )
    tasks_payload = json.loads(tasks_response[0].text)
    assert tasks_payload["status"] == "success"
    assert len(tasks_payload["pending_compactions"]) == 1
