from __future__ import annotations

import argparse
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from cognitiveos.config import AppSettings
from cognitiveos.models import AddPayloadType
from cognitiveos.service import CognitiveOSService

HOST_CORE_PROFILE = "host-core"
BOOTSTRAP_PROFILE = "bootstrap"
OPERATOR_PROFILE = "operator"
FULL_PROFILE = "full"
SUPPORTED_SERVER_PROFILES = {
    HOST_CORE_PROFILE,
    BOOTSTRAP_PROFILE,
    OPERATOR_PROFILE,
    FULL_PROFILE,
}


def build_server(settings: AppSettings) -> FastMCP:
    service = CognitiveOSService.from_settings(settings)
    service.initialize()
    profile = (settings.server_profile or HOST_CORE_PROFILE).strip().lower()
    if profile not in SUPPORTED_SERVER_PROFILES:
        raise ValueError(
            "Unsupported MCP profile. Supported profiles: "
            + ", ".join(sorted(SUPPORTED_SERVER_PROFILES))
        )
    expose_host_core = profile in {HOST_CORE_PROFILE, OPERATOR_PROFILE, FULL_PROFILE}
    expose_bootstrap = profile in {BOOTSTRAP_PROFILE, FULL_PROFILE}
    expose_operator = profile in {OPERATOR_PROFILE, FULL_PROFILE}
    mcp = FastMCP(
        settings.server_name,
        instructions=(
            "CognitiveOS is a local-first cognitive graph runtime exposing a compact "
            f"{profile} MCP profile."
        ),
        json_response=True,
    )

    if expose_host_core:
        @mcp.tool()
        def search(
            query: str | None = None,
            keyword: str | None = None,
            top_k: int = 5,
            include_neighbors: int = 1,
            include_evidence: bool = False,
        ) -> list[dict]:
            return [
                result.model_dump()
                for result in service.search(
                    query=query,
                    keyword=keyword,
                    top_k=top_k,
                    include_neighbors=include_neighbors,
                    include_evidence=include_evidence,
                )
            ]

        @mcp.tool()
        def read(ids: list[str], include_content: bool = False) -> dict[str, dict]:
            return {
                node_id: node.model_dump()
                for node_id, node in service.read_nodes(
                    ids,
                    include_content=include_content,
                ).items()
            }

        @mcp.tool()
        def add(
            type: AddPayloadType,
            payload: str,
            tags: list[str] | None = None,
            durability: str | None = None,
            force: bool = False,
            name: str | None = None,
        ) -> dict:
            return service.add_node(
                payload_type=type,
                payload=payload,
                tags=tags or [],
                durability=durability,
                force=force,
                name=name,
            ).model_dump()

        @mcp.tool()
        def update(
            id: str,
            content: str,
            tags: list[str] | None = None,
            durability: str | None = None,
        ) -> dict:
            return service.update_node(
                node_id=id,
                content=content,
                tags=tags or [],
                durability=durability,
            ).model_dump()

        @mcp.tool()
        def link(src_id: str, dst_id: str, relation: str) -> dict:
            receipt = service.link_nodes(
                src_id=src_id,
                dst_id=dst_id,
                relation=relation,
            )
            return receipt.model_dump()

        @mcp.tool()
        def unlink(src_id: str, dst_id: str, relation: str | None = None) -> dict:
            return service.unlink_nodes(
                src_id=src_id,
                dst_id=dst_id,
                relation=relation,
            ).model_dump()

        @mcp.tool()
        def dream(
            output_path: str | None = None,
            window_hours: int = 168,
            min_accesses: int = 2,
            min_cluster_size: int = 2,
            max_candidates: int = 100,
            inspect: str | None = None,
            status_filter: str | None = None,
            run_id: str | None = None,
            limit: int = 20,
            task_id: str | None = None,
            title: str | None = None,
            description: str | None = None,
            content: str | None = None,
            use_heuristic: bool = False,
            background: bool = False,
        ) -> dict:
            if inspect is not None:
                normalized_inspect = inspect.strip().lower()
                if normalized_inspect == "status":
                    return {
                        "status": "success",
                        "dream_status": service.get_dream_status().model_dump(),
                    }
                if normalized_inspect == "runs":
                    return {
                        "status": "success",
                        "runs": [
                            run.model_dump()
                            for run in service.list_dream_runs(
                                status=status_filter,
                                limit=limit,
                            )
                        ],
                    }
                if normalized_inspect == "tasks":
                    return {
                        "status": "success",
                        "pending_compactions": [
                            task.model_dump()
                            for task in service.list_dream_compactions(
                                run_id=run_id,
                                status=status_filter or "pending",
                            )
                        ],
                    }
                raise ValueError("Unsupported dream inspect mode. Use status, runs, or tasks.")
            if task_id is not None:
                return service.resolve_dream_compaction(
                    task_id=task_id,
                    title=title,
                    description=description,
                    content=content,
                    use_heuristic=use_heuristic,
                    background=background,
                ).model_dump()
            result = service.run_dream(
                output_path=Path(output_path) if output_path else None,
                window_hours=window_hours,
                min_accesses=min_accesses,
                min_cluster_size=min_cluster_size,
                max_candidates=max_candidates,
                background=background,
            )
            return result.model_dump()

    if expose_operator:
        @mcp.tool()
        def list_relationships(
            node_id: str,
            relation: str | None = None,
            status: str | None = None,
        ) -> list[dict]:
            return [
                edge.model_dump()
                for edge in service.list_relationships(
                    node_id=node_id,
                    relation=relation,
                    status=status,
                )
            ]

        @mcp.tool()
        def reinforce_relationship(
            src_id: str,
            dst_id: str,
            relation: str,
            delta: float = 0.25,
        ) -> dict:
            return service.reinforce_relationship(
                src_id=src_id,
                dst_id=dst_id,
                relation=relation,
                delta=delta,
            ).model_dump()

        @mcp.tool()
        def prune_relationships(node_id: str | None = None, dry_run: bool = True) -> dict:
            return service.prune_relationships(node_id=node_id, dry_run=dry_run)

        @mcp.tool()
        def pin_memory(node_id: str) -> dict:
            return service.pin_memory(node_id=node_id).model_dump()

        @mcp.tool()
        def unpin_memory(node_id: str) -> dict:
            return service.unpin_memory(node_id=node_id).model_dump()

        @mcp.tool()
        def refresh_source_document(node_id: str) -> dict:
            return service.refresh_source_document(node_id=node_id).model_dump()

        @mcp.tool()
        def reindex_embeddings() -> dict:
            return service.reindex_embeddings()

        @mcp.tool()
        def doctor(check_providers: bool = False) -> dict:
            return service.doctor(check_providers=check_providers)

        @mcp.tool()
        def providers_test() -> dict:
            return service.test_providers()

        @mcp.resource("node://{node_id}")
        def node_resource(node_id: str) -> str:
            node = service.read_nodes([node_id], include_content=True).get(node_id)
            if node is None:
                return "{}"
            return node.model_dump_json(indent=2, exclude_none=True)

    if expose_bootstrap:
        @mcp.tool()
        def host_bootstrap_status(
            host_kind: str = "generic",
            output_dir: str | None = None,
        ) -> dict:
            return service.get_host_bootstrap_status(
                host_kind=host_kind,
                output_dir=Path(output_dir) if output_dir else None,
            ).model_dump()

        @mcp.tool()
        def submit_host_onboarding(
            answers: dict[str, str],
            host_kind: str = "generic",
            output_dir: str | None = None,
        ) -> dict:
            return service.submit_host_onboarding(
                answers=answers,
                host_kind=host_kind,
                output_dir=Path(output_dir) if output_dir else None,
            ).model_dump()

        @mcp.tool()
        def bootstrap_host(
            output_dir: str | None = None,
            host_kind: str = "generic",
            install: bool = False,
        ) -> dict:
            bundle = service.build_host_bootstrap(
                Path(output_dir) if output_dir else None,
                host_kind=host_kind,
                install=install,
            )
            return bundle.model_dump()

    return mcp


def run_mcp_server(
    *,
    settings: AppSettings,
    transport: str,
    host: str,
    port: int,
    path: str,
    profile: str,
) -> None:
    settings.ensure_runtime_paths()
    settings.server_host = host
    settings.server_port = port
    settings.server_path = path
    settings.server_profile = profile
    server = build_server(settings)
    server.settings.host = host
    server.settings.port = port
    server.settings.mount_path = path
    server.settings.sse_path = path
    server.settings.streamable_http_path = path
    server.run(transport=transport)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CognitiveOS MCP server.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Override the SQLite database path.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport to expose.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port.")
    parser.add_argument("--path", default="/mcp", help="HTTP mount path.")
    parser.add_argument(
        "--profile",
        default=HOST_CORE_PROFILE,
        choices=sorted(SUPPORTED_SERVER_PROFILES),
        help="MCP tool profile to expose.",
    )
    args = parser.parse_args()

    run_mcp_server(
        settings=AppSettings.from_env(db_path=args.db_path),
        transport=args.transport,
        host=args.host,
        port=args.port,
        path=args.path,
        profile=args.profile,
    )
