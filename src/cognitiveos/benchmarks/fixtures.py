from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ContentNodeFixture:
    key: str
    name: str
    payload: str
    tags: tuple[str, ...] = ()
    durability: str | None = None


@dataclass(frozen=True, slots=True)
class FileNodeFixture:
    key: str
    filename: str
    name: str
    tags: tuple[str, ...] = ()
    durability: str | None = None


@dataclass(frozen=True, slots=True)
class LinkFixture:
    src: str
    dst: str
    relation: str


@dataclass(frozen=True, slots=True)
class QualityTaskFixture:
    task_id: str
    keyword: str | None = None
    query: str | None = None
    top_k: int = 3
    include_neighbors: int = 1
    include_evidence: bool = False
    expected_root: str | None = None
    expected_linked: tuple[str, ...] = ()
    expected_node_types_present: tuple[str, ...] = ()
    expected_node_types_absent: tuple[str, ...] = ()
    requires_embedding: bool = False


@dataclass(frozen=True, slots=True)
class BenchmarkCorpusFixture:
    name: str
    description: str
    content_nodes: tuple[ContentNodeFixture, ...] = ()
    file_nodes: tuple[FileNodeFixture, ...] = ()
    links: tuple[LinkFixture, ...] = ()
    quality_tasks: tuple[QualityTaskFixture, ...] = ()


def _long_memory_walkthrough() -> str:
    section = (
        "CognitiveOS uses source_document nodes for document-level retrieval. "
        "The long-form walkthrough explains durable document recall, bounded graph expansion, "
        "benchmark baselines, and host memory flows across search, read, add, update, link, "
        "and dream operations. Document summaries and retrieval tags matter when a host needs "
        "fast recall without indexing every lexical detail equally.\n\n"
    )
    return "# Long Memory Walkthrough\n\n" + section * 18


def materialize_fixture_files(base_dir: Path) -> dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "bootstrap_guide": (
            "bootstrap_guide.md",
            (
                "# Bootstrap Guide\n\n"
                "The bootstrap-host command writes host-bootstrap.md, mount-manifest.json, "
                "and mcp-server.json. Codex installation also writes AGENTS.md and "
                ".codex/config.toml so the host can mount CognitiveOS cleanly.\n"
            ),
        ),
        "host_contract_page": (
            "host_contract.html",
            (
                "<html><body><h1>Host Contract</h1>"
                "<p>Hosts should search first, read selected ids second, then add, update, "
                "link, unlink, and dream as needed.</p>"
                "<p>The compact host surface stays focused on operational memory workflows.</p>"
                "</body></html>"
            ),
        ),
        "long_memory_walkthrough": (
            "long_memory_walkthrough.md",
            _long_memory_walkthrough(),
        ),
    }
    paths: dict[str, Path] = {}
    for key, (filename, content) in payloads.items():
        path = base_dir / filename
        path.write_text(content, encoding="utf-8")
        paths[key] = path
    return paths


DEFAULT_BENCHMARK_CORPUS = BenchmarkCorpusFixture(
    name="github_readiness_baseline",
    description=(
        "Deterministic local corpus for GitHub-readiness baseline evaluation across "
        "retrieval, ingestion, host workflows, and dream preparation."
    ),
    content_nodes=(
        ContentNodeFixture(
            key="project_overview",
            name="Project Overview",
            payload=(
                "CognitiveOS is a local-first cognitive graph runtime for host agents. "
                "It exposes installable CLI and MCP surfaces for memory workflows."
            ),
            tags=("project", "runtime"),
        ),
        ContentNodeFixture(
            key="bootstrap_artifacts",
            name="Bootstrap Artifacts",
            payload=(
                "The bootstrap-host command writes host-bootstrap.md, mount-manifest.json, "
                "and mcp-server.json for Codex and other hosts."
            ),
            tags=("bootstrap", "host"),
        ),
        ContentNodeFixture(
            key="codex_mount",
            name="Codex Mount",
            payload=(
                "Codex installation writes AGENTS.md and .codex/config.toml so the host "
                "can mount memory and register CognitiveOS."
            ),
            tags=("codex", "mount"),
        ),
        ContentNodeFixture(
            key="read_write_flow",
            name="Read Write Flow",
            payload=(
                "Hosts should search first, then read selected ids, then add or update "
                "memory, and finally link related facts when the relation matters."
            ),
            tags=("workflow", "host"),
        ),
        ContentNodeFixture(
            key="multi_hop_routing",
            name="Multi Hop Routing",
            payload=(
                "Multi-hop retrieval depends on relation chaining, entity bridges, and "
                "graph-aware scoring across linked memory nodes."
            ),
            tags=("retrieval", "graph"),
        ),
        ContentNodeFixture(
            key="graph_scoring",
            name="Graph Scoring",
            payload=(
                "Graph-aware scoring combines direct relevance with path support, relation "
                "strength, and bounded hop expansion."
            ),
            tags=("retrieval", "scoring"),
        ),
        ContentNodeFixture(
            key="extractor_pipeline",
            name="Extractor Pipeline",
            payload=(
                "The extractor pipeline ingests local files, file URLs, HTTP pages, and "
                "long documents into durable source_document nodes."
            ),
            tags=("extractor", "ingestion"),
        ),
        ContentNodeFixture(
            key="benchmark_story",
            name="Benchmark Story",
            payload=(
                "Benchmark readiness needs deterministic corpora, baseline metrics, and "
                "repeatable local runs before optimization claims are credible."
            ),
            tags=("benchmark", "quality"),
        ),
    ),
    file_nodes=(
        FileNodeFixture(
            key="bootstrap_guide",
            filename="bootstrap_guide.md",
            name="Bootstrap Guide",
            tags=("bootstrap", "docs"),
        ),
        FileNodeFixture(
            key="host_contract_page",
            filename="host_contract.html",
            name="Host Contract Page",
            tags=("host", "docs"),
        ),
        FileNodeFixture(
            key="long_memory_walkthrough",
            filename="long_memory_walkthrough.md",
            name="Long Memory Walkthrough",
            tags=("docs", "longform", "evidence"),
        ),
    ),
    links=(
        LinkFixture(src="bootstrap_artifacts", dst="codex_mount", relation="produces"),
        LinkFixture(src="codex_mount", dst="read_write_flow", relation="supports"),
        LinkFixture(src="multi_hop_routing", dst="graph_scoring", relation="depends_on"),
        LinkFixture(src="extractor_pipeline", dst="benchmark_story", relation="informs"),
        LinkFixture(src="project_overview", dst="benchmark_story", relation="frames"),
        LinkFixture(src="bootstrap_guide", dst="bootstrap_artifacts", relation="documents"),
    ),
    quality_tasks=(
        QualityTaskFixture(
            task_id="keyword_project",
            keyword="installable",
            top_k=3,
            include_neighbors=0,
            expected_root="project_overview",
        ),
        QualityTaskFixture(
            task_id="semantic_read_write_flow",
            query="host search read add update link workflow",
            top_k=3,
            include_neighbors=0,
            expected_root="read_write_flow",
            requires_embedding=True,
        ),
        QualityTaskFixture(
            task_id="hybrid_bootstrap",
            keyword="bootstrap",
            query="Codex install files and bootstrap outputs",
            top_k=3,
            include_neighbors=1,
            expected_root="bootstrap_guide",
            expected_linked=("bootstrap_artifacts",),
            requires_embedding=True,
        ),
        QualityTaskFixture(
            task_id="multi_hop_bootstrap_flow",
            keyword="bootstrap",
            top_k=1,
            include_neighbors=2,
            expected_root="bootstrap_artifacts",
            expected_linked=("codex_mount", "read_write_flow"),
        ),
        QualityTaskFixture(
            task_id="long_document_default_view",
            keyword="Walkthrough",
            top_k=8,
            include_neighbors=0,
            include_evidence=False,
            expected_root="long_memory_walkthrough",
        ),
        QualityTaskFixture(
            task_id="long_document_include_evidence_flag_noop",
            keyword="Walkthrough",
            top_k=8,
            include_neighbors=0,
            include_evidence=True,
            expected_root="long_memory_walkthrough",
        ),
    ),
)
