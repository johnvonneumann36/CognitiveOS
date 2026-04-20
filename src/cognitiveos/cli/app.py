from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from cognitiveos.benchmarks.runner import run_benchmark_suite
from cognitiveos.config import AppSettings
from cognitiveos.mcp.server import run_mcp_server
from cognitiveos.models import AddPayloadType
from cognitiveos.service import CognitiveOSService

app = typer.Typer(help="CognitiveOS command line interface.")


def _service(
    db_path: Path | None = None,
    memory_output_path: Path | None = None,
    project_root: Path | None = None,
) -> CognitiveOSService:
    settings = AppSettings.from_env(
        db_path=db_path,
        memory_output_path=memory_output_path,
        project_root=project_root,
    )
    return CognitiveOSService.from_settings(settings)


def _print_json(payload: object) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_key_value_pairs(pairs: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise typer.BadParameter(
                f"Invalid answer '{item}'. Use KEY=VALUE form, for example display_name=Bruce."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise typer.BadParameter(
                f"Invalid answer '{item}'. Both KEY and VALUE must be non-empty."
            )
        parsed[key] = value
    return parsed


@app.command("init-db")
def init_db(
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    service.initialize()
    _print_json({"status": "success", "db_path": str(service.settings.db_path)})


@app.command("add")
def add(
    payload_type: Annotated[AddPayloadType, typer.Option("--type", help="Payload source type.")],
    payload: Annotated[str, typer.Option(help="Raw content, file path, folder path, or URL.")],
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Tag values to apply to the node."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(help="Bypass similarity probing when a provider exists."),
    ] = False,
    durability: Annotated[
        str | None,
        typer.Option(help="Optional durability override: working, durable, pinned, ephemeral."),
    ] = None,
    name: Annotated[str | None, typer.Option(help="Optional node name override.")] = None,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    receipt = service.add_node(
        payload_type=payload_type,
        payload=payload,
        tags=tag or [],
        durability=durability,
        force=force,
        name=name,
    )
    _print_json(receipt.model_dump())


@app.command("read")
def read(
    ids: Annotated[list[str], typer.Argument(help="Target node ids.")],
    include_content: Annotated[bool, typer.Option(help="Return full content payloads.")] = False,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    payload = {
        node_id: node.model_dump()
        for node_id, node in service.read_nodes(ids, include_content=include_content).items()
    }
    _print_json(payload)


@app.command("search")
def search(
    keyword: Annotated[str | None, typer.Option(help="Keyword query for FTS search.")] = None,
    query: Annotated[str | None, typer.Option(help="Semantic query string.")] = None,
    top_k: Annotated[int, typer.Option(help="Number of root nodes to return.")] = 5,
    include_neighbors: Annotated[int, typer.Option(help="Neighbor traversal depth (0-3).")] = 1,
    include_evidence: Annotated[
        bool,
        typer.Option(help="Allow evidence nodes in search results when chunking is enabled."),
    ] = False,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    results = service.search(
        query=query,
        keyword=keyword,
        top_k=top_k,
        include_neighbors=include_neighbors,
        include_evidence=include_evidence,
    )
    _print_json([result.model_dump() for result in results])


@app.command("update")
def update(
    node_id: Annotated[str, typer.Argument(help="Node id to update.")],
    content: Annotated[str, typer.Option(help="New node content.")],
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Replacement tag values.")] = None,
    durability: Annotated[
        str | None,
        typer.Option(help="Optional durability override: working, durable, pinned, ephemeral."),
    ] = None,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    receipt = service.update_node(node_id=node_id, content=content, tags=tag, durability=durability)
    _print_json(receipt.model_dump())


@app.command("link")
def link(
    src_id: Annotated[str, typer.Argument(help="Source node id.")],
    dst_id: Annotated[str, typer.Argument(help="Destination node id.")],
    relation: Annotated[str, typer.Argument(help="Directed edge relation.")],
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    receipt = service.link_nodes(src_id=src_id, dst_id=dst_id, relation=relation)
    _print_json(receipt.model_dump())


@app.command("unlink")
def unlink(
    src_id: Annotated[str, typer.Argument(help="Source node id.")],
    dst_id: Annotated[str, typer.Argument(help="Destination node id.")],
    relation: Annotated[
        str | None,
        typer.Option(help="Optional relation type to remove."),
    ] = None,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(service.unlink_nodes(src_id=src_id, dst_id=dst_id, relation=relation).model_dump())


@app.command("list-relationships")
def list_relationships(
    node_id: Annotated[str, typer.Argument(help="Node id to inspect.")],
    relation: Annotated[
        str | None,
        typer.Option(help="Optional relation filter."),
    ] = None,
    status: Annotated[
        str | None,
        typer.Option(help="Optional relationship status filter."),
    ] = None,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(
        [
            edge.model_dump()
            for edge in service.list_relationships(
                node_id=node_id,
                relation=relation,
                status=status,
            )
        ]
    )


@app.command("reinforce-relationship")
def reinforce_relationship(
    src_id: Annotated[str, typer.Argument(help="Source node id.")],
    dst_id: Annotated[str, typer.Argument(help="Destination node id.")],
    relation: Annotated[str, typer.Argument(help="Relation type.")],
    delta: Annotated[float, typer.Option(help="Strength delta to apply.")] = 0.25,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(
        service.reinforce_relationship(
            src_id=src_id,
            dst_id=dst_id,
            relation=relation,
            delta=delta,
        ).model_dump()
    )


@app.command("prune-relationships")
def prune_relationships(
    node_id: Annotated[
        str | None,
        typer.Option(help="Optional node id scope."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Preview candidates without deleting them."),
    ] = True,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(service.prune_relationships(node_id=node_id, dry_run=dry_run))


@app.command("pin-memory")
def pin_memory(
    node_id: Annotated[str, typer.Argument(help="Node id to pin.")],
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(service.pin_memory(node_id=node_id).model_dump())


@app.command("unpin-memory")
def unpin_memory(
    node_id: Annotated[str, typer.Argument(help="Node id to unpin.")],
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(service.unpin_memory(node_id=node_id).model_dump())


@app.command("refresh-source-document")
def refresh_source_document(
    node_id: Annotated[str, typer.Argument(help="Source document node id to refresh.")],
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(service.refresh_source_document(node_id=node_id).model_dump())


@app.command("dream")
def dream(
    output_path: Annotated[
        Path | None,
        typer.Option(help="Output path for the static memory file."),
    ] = None,
    window_hours: Annotated[int, typer.Option(help="Dream window in hours.")] = 168,
    min_accesses: Annotated[
        int,
        typer.Option(help="Minimum access count for dream candidates."),
    ] = 2,
    min_cluster_size: Annotated[
        int,
        typer.Option(help="Minimum cluster size to consolidate."),
    ] = 2,
    max_candidates: Annotated[
        int,
        typer.Option(help="Maximum nodes to inspect in a dream run."),
    ] = 100,
    inspect: Annotated[
        str | None,
        typer.Option(help="Inspect dream status, runs, or tasks instead of starting a new run."),
    ] = None,
    status_filter: Annotated[
        str | None,
        typer.Option(help="Optional status filter for dream run/task inspection."),
    ] = None,
    run_id: Annotated[
        str | None,
        typer.Option(help="Optional run id filter when inspecting dream tasks."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(help="Maximum number of dream runs to return during inspection."),
    ] = 20,
    task_id: Annotated[
        str | None,
        typer.Option(
            help="Resolve an existing dream compaction task instead of starting a new run."
        ),
    ] = None,
    title: Annotated[
        str | None,
        typer.Option(help="Resolved super-node title when completing a pending compaction task."),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option(
            help="Resolved super-node description when completing a pending compaction task."
        ),
    ] = None,
    content: Annotated[
        str | None,
        typer.Option(help="Resolved super-node content when completing a pending compaction task."),
    ] = None,
    use_heuristic: Annotated[
        bool,
        typer.Option(help="Resolve a pending compaction task with heuristic fallback."),
    ] = False,
    background: Annotated[
        bool,
        typer.Option(
            help=(
                "Queue a dream run in the background, or when resolving with heuristic fallback "
                "queue the compaction task in the background."
            )
        ),
    ] = False,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path, memory_output_path=output_path)
    if inspect is not None:
        normalized_inspect = inspect.strip().lower()
        if normalized_inspect == "status":
            _print_json(
                {
                    "status": "success",
                    "dream_status": service.get_dream_status().model_dump(),
                }
            )
            return
        if normalized_inspect == "runs":
            _print_json(
                {
                    "status": "success",
                    "runs": [
                        run.model_dump()
                        for run in service.list_dream_runs(status=status_filter, limit=limit)
                    ],
                }
            )
            return
        if normalized_inspect == "tasks":
            _print_json(
                {
                    "status": "success",
                    "pending_compactions": [
                        task.model_dump()
                        for task in service.list_dream_compactions(
                            run_id=run_id,
                            status=status_filter or "pending",
                        )
                    ],
                }
            )
            return
        raise typer.BadParameter("Unsupported dream inspect mode. Use status, runs, or tasks.")
    if task_id is not None:
        resolution = service.resolve_dream_compaction(
            task_id=task_id,
            title=title,
            description=description,
            content=content,
            use_heuristic=use_heuristic,
            background=background,
        )
        _print_json(resolution.model_dump())
        return
    result = service.run_dream(
        output_path=output_path,
        window_hours=window_hours,
        min_accesses=min_accesses,
        min_cluster_size=min_cluster_size,
        max_candidates=max_candidates,
        background=background,
    )
    _print_json(result.model_dump())


@app.command("reindex-embeddings")
def reindex_embeddings(
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(service.reindex_embeddings())


@app.command("providers-test")
def providers_test(
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(service.test_providers())


@app.command("doctor")
def doctor(
    check_providers: Annotated[
        bool,
        typer.Option(help="Run live provider smoke tests."),
    ] = False,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path)
    _print_json(service.doctor(check_providers=check_providers))


@app.command("bootstrap-host")
def bootstrap_host(
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Override the bootstrap output directory."),
    ] = None,
    host_kind: Annotated[
        str,
        typer.Option(
            help="Target host kind, for example generic, codex, claude-code, gemini-cli, or cursor."
        ),
    ] = "generic",
    install: Annotated[
        bool,
        typer.Option(help="Install the host mount into project files when supported."),
    ] = False,
    project_root: Annotated[
        Path | None,
        typer.Option(
            help="Project root used for bootstrap artifacts and host install targets."
        ),
    ] = None,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path, project_root=project_root)
    _print_json(
        service.build_host_bootstrap(
            output_dir=output_dir,
            host_kind=host_kind,
            install=install,
        ).model_dump()
    )


@app.command("host-bootstrap-status")
def host_bootstrap_status(
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Override the bootstrap output directory."),
    ] = None,
    host_kind: Annotated[
        str,
        typer.Option(
            help="Target host kind, for example generic, codex, claude-code, gemini-cli, or cursor."
        ),
    ] = "generic",
    project_root: Annotated[
        Path | None,
        typer.Option(
            help="Project root used for bootstrap artifacts and host install targets."
        ),
    ] = None,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path, project_root=project_root)
    _print_json(
        service.get_host_bootstrap_status(
            host_kind=host_kind,
            output_dir=output_dir,
        ).model_dump()
    )


@app.command("submit-host-onboarding")
def submit_host_onboarding(
    answer: Annotated[
        list[str] | None,
        typer.Option(
            "--answer",
            help="Onboarding answer in KEY=VALUE form. Repeat for each question.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Override the bootstrap output directory."),
    ] = None,
    host_kind: Annotated[
        str,
        typer.Option(
            help="Target host kind, for example generic, codex, claude-code, gemini-cli, or cursor."
        ),
    ] = "generic",
    project_root: Annotated[
        Path | None,
        typer.Option(
            help="Project root used for bootstrap artifacts and host install targets."
        ),
    ] = None,
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
) -> None:
    service = _service(db_path=db_path, project_root=project_root)
    _print_json(
        service.submit_host_onboarding(
            answers=_parse_key_value_pairs(answer),
            host_kind=host_kind,
            output_dir=output_dir,
        ).model_dump()
    )


@app.command("benchmark")
def benchmark(
    iterations: Annotated[
        int,
        typer.Option(help="Number of repetitions for each runtime benchmark operation."),
    ] = 5,
    output_path: Annotated[
        Path | None,
        typer.Option(help="Optional path to write the benchmark JSON report."),
    ] = None,
    runtime_dir: Annotated[
        Path | None,
        typer.Option(
            help="Optional directory for the isolated benchmark database, MEMORY.MD, and files."
        ),
    ] = None,
    provider_mode: Annotated[
        str,
        typer.Option(help="Provider mode: fake, env, or none."),
    ] = "fake",
) -> None:
    _print_json(
        run_benchmark_suite(
            iterations=iterations,
            output_path=output_path,
            runtime_dir=runtime_dir,
            provider_mode=provider_mode,
        )
    )


@app.command("mcp")
def mcp(
    transport: Annotated[
        str,
        typer.Option(help="MCP transport: stdio, sse, or streamable-http."),
    ] = "stdio",
    profile: Annotated[
        str,
        typer.Option(help="MCP profile: host-core, bootstrap, operator, or full."),
    ] = "host-core",
    host: Annotated[str, typer.Option(help="HTTP bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="HTTP bind port.")] = 8000,
    path: Annotated[str, typer.Option(help="HTTP mount path.")] = "/mcp",
    db_path: Annotated[Path | None, typer.Option(help="Override the SQLite database path.")] = None,
    memory_output_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Override the MEMORY.MD output path. If omitted, CognitiveOS uses the "
                "shared runtime home or infers it from the db path."
            )
        ),
    ] = None,
    project_root: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Project root used for bootstrap/install targets and background job cwd. "
                "Defaults to the current working directory."
            )
        ),
    ] = None,
) -> None:
    run_mcp_server(
        settings=AppSettings.from_env(
            db_path=db_path,
            memory_output_path=memory_output_path,
            project_root=project_root,
        ),
        transport=transport,
        profile=profile,
        host=host,
        port=port,
        path=path,
    )


def main() -> None:
    app()
