# CognitiveOS

<p align="center">
  <img src="https://raw.githubusercontent.com/johnvonneumann36/CognitiveOS/main/assets/logo.png" alt="CognitiveOS logo" width="220">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/johnvonneumann36/CognitiveOS/main/assets/banner.png" alt="CognitiveOS banner">
</p>

<p align="center">
  <a href="https://pypi.org/project/cognitiveos/">
    <img src="https://img.shields.io/pypi/v/cognitiveos" alt="PyPI version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/cognitiveos" alt="Python versions">
  <img src="https://img.shields.io/github/license/johnvonneumann36/CognitiveOS" alt="License">
</p>

CognitiveOS is a local-first cognitive graph runtime for agentic environments such as Claude Code, Gemini CLI, Codex, and other MCP-capable hosts.

This repository implements a production-oriented project skeleton for a local-first agent memory runtime, with a focus on:

- installable Python packaging via `pip`
- CLI operations for database lifecycle and graph manipulation
- an installable MCP server runtime
- Docker-based deployment for HTTP MCP serving
- extensible provider and extractor boundaries

> 📦 `v0.1.1` is now available on PyPI: `pip install cognitiveos`

## Start Here

If you want to try CognitiveOS as a package user:

```bash
pip install cognitiveos
cognitiveos init-db
cognitiveos-mcp --transport stdio --profile host-core
```

If you want to work on the repository locally:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Public entry points:

- PyPI: `https://pypi.org/project/cognitiveos/`
- Releases: `https://github.com/johnvonneumann36/CognitiveOS/releases`
- Repository docs: `https://github.com/johnvonneumann36/CognitiveOS/tree/main/docs`

## ✨ Design Goals

CognitiveOS is not intended to be a universal knowledge vault.

Its primary goal is to give an agent host a compact, durable, searchable memory layer that remains useful across sessions without forcing the runtime to mirror every source in full.

The practical goals are:

- preserve durable user preferences, profile facts, goals, lessons, and decisions
- preserve useful source anchors such as files, folders, repositories, and URLs
- make those memories searchable through both keyword and semantic retrieval
- keep the public memory model compact enough that hosts can reason about it reliably
- support long-lived local operation with minimal infrastructure
- keep the host-facing tool surface small and stable

## 🧭 Design Philosophy

Several design choices in this repository follow that goal directly.

Local-first:

- memory lives in a local user-controlled runtime and does not require a hosted backend by default
- SQLite is the default runtime store
- MCP and CLI surfaces are designed to work well on a single machine

Compact public model:

- the runtime prefers a small number of structural node types over many specialized ones
- files are stored as `source_document`
- folders and repositories are stored as `source_collection`
- deep live inspection of codebases, folders, and remote resources remains the host agent's job

Searchability over verbatim chat copying:

- stored memory should be normalized into retrieval-friendly language
- durable facts should be explicit about actor, object, and time when known
- the point is future recall quality, not transcript fidelity

Source anchors over recursive mirroring:

- large repositories and folders should usually be remembered as compact roots
- the runtime stores enough summary and metadata to make the source re-openable
- the host agent can inspect the live source later when detailed exploration is needed

Operational simplicity:

- prefer predictable CLI and MCP semantics over feature sprawl
- prefer bounded metadata over unbounded recursive ingestion
- prefer internal hardening, such as SQLite stability settings and background logs, over expanding host-visible tooling

## ✅ Status

The current milestone is an MVP foundation:

- SQLite-backed node, edge, and audit log storage
- `sqlite-vec` backed vector index with cosine KNN retrieval
- FTS5 keyword search over a unified `search_text` projection with bounded neighbor traversal
- optional multi-provider embeddings with similarity probing and query-only semantic search
- hybrid ranking that merges semantic and FTS results
- content, file, and folder ingestion with durable `source_document` and `source_collection` roots
- chat-backed document descriptions and retrieval-tag generation for file ingestion
- Trafilatura-first remote extraction with typed remote source metadata and snapshot preservation for remote URLs
- root-only folder inspection for repository, media, document, and workspace collection anchors
- update and link operations with audit tracing
- canonical node semantics where `node_type` is structural, `durability` is lifecycle, `tags` are retrieval labels, and `metadata` is factual structured payload
- canonical edge semantics where `strength_score` is the only runtime edge-strength field
- dream execution with access-log-driven clustering, chat-first compaction, host-agent fallback tasks, and `MEMORY.MD` generation
- CLI and MCP entrypoints
- operations commands for health checks, provider smoke tests, and embedding reindexing
- host bootstrap bundle generation for cold-start memory mount and MCP wiring
- generic host bootstrap outputs plus Codex-specific project auto-install via `AGENTS.md` and `.codex/config.toml`
- first-start onboarding questions that seed pinned system-profile memory through structured profile metadata
- background dream execution for runtime-only compaction paths

## 🚀 Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

Initialize the workspace database:

```bash
cognitiveos init-db
```

Add a node:

```bash
cognitiveos add --type content --payload "CognitiveOS stores graph-shaped memory." --tag tech --tag memory --tag graph
```

Add a folder root:

```bash
cognitiveos add --type folder --payload ".\\Pictures\\Japan Trip 2025" --name "Japan Trip 2025 Photos" --tag travel --tag japan --tag photos
```

Add a remote source document and preserve a local snapshot:

```bash
cognitiveos add --type file --payload "https://example.com/article" --name "Example Article" --tag article --tag reference
```

Add a browser-exported local capture with a sidecar manifest:

```bash
cognitiveos add --type file --payload ".\\exports\\article.html"
```

Note:

- use repeated `--tag` for retrieval labels
- use `--name` when the node should keep a stable human-readable title instead of relying on payload-derived defaults
- remote `http` and `https` URLs are preserved by default under `.cognitiveos/snapshots/`, using Markdown for readable text/page captures and binary files for formats such as PDF
- when a page requires login or browser state, use the host's existing browser tools to open the accessible page first; CognitiveOS v1 does not automate browser capture itself
- adding the same remote URL again warns on conflict by default; use `force=true` to refresh that source in place
- host-assisted inputs can be: the final accessible URL, a browser-exported local HTML file, or a host-prepared Markdown file
- for browser-exported local HTML or Markdown, add a sidecar file named `<stem>.cognitiveos-source.json` beside the export to preserve the original remote URL and capture metadata

Context-size warning:

- `include_neighbors`, `include_content`, and `include_evidence` can expand returned context very quickly
- start narrow, then widen only when the initial result is clearly insufficient
- for routine recall, prefer `include_neighbors=0` or `1`, `include_content=false`, and `include_evidence=false`

Search by keyword:

```bash
cognitiveos search --keyword CognitiveOS --top-k 5 --include-neighbors 1
```

Run hybrid search:

```bash
cognitiveos search --query "graph memory runtime" --keyword CognitiveOS --top-k 5
```

Run the MCP server over stdio:

```bash
cognitiveos-mcp --transport stdio --profile host-core
```

Run the MCP server over Streamable HTTP:

```bash
cognitiveos-mcp --transport streamable-http --profile host-core --host 0.0.0.0 --port 8000 --path /mcp
```

## 🔌 Provider Configuration

Embedding and chat providers are optional. When an embedding model is configured:

- `add` performs similarity probing before insert unless `force=true`
- `search --query ...` performs semantic search even without keywords
- `search --query ... --keyword ...` performs hybrid ranking
- missing node embeddings are lazily backfilled

Provider role support:

- `ollama`: chat, embeddings
- `openai`: chat, embeddings
- `gemini`: chat, embeddings
- `anthropic`: chat only
- `local_huggingface`: chat, embeddings, but requires `pip install 'cognitiveos[local-hf]'`

Environment variables:

```bash
COGNITIVEOS_EMBEDDING_PROVIDER_TYPE=ollama
COGNITIVEOS_EMBEDDING_MODEL_NAME=nomic-embed-text
COGNITIVEOS_EMBEDDING_BASE_URL=http://localhost:11434/api
COGNITIVEOS_CHAT_PROVIDER_TYPE=ollama
COGNITIVEOS_CHAT_MODEL_NAME=gemma3
COGNITIVEOS_CHAT_BASE_URL=http://localhost:11434/api
COGNITIVEOS_SIMILARITY_THRESHOLD=0.92
COGNITIVEOS_HYBRID_SEMANTIC_WEIGHT=0.65
COGNITIVEOS_HYBRID_KEYWORD_WEIGHT=0.35
```

Examples by provider:

```bash
# OpenAI
COGNITIVEOS_EMBEDDING_PROVIDER_TYPE=openai
COGNITIVEOS_EMBEDDING_MODEL_NAME=text-embedding-3-small
COGNITIVEOS_EMBEDDING_API_KEY=...
COGNITIVEOS_CHAT_PROVIDER_TYPE=openai
COGNITIVEOS_CHAT_MODEL_NAME=gpt-5-mini
COGNITIVEOS_CHAT_API_KEY=...

# Anthropic chat only
COGNITIVEOS_CHAT_PROVIDER_TYPE=anthropic
COGNITIVEOS_CHAT_MODEL_NAME=claude-3-7-sonnet-latest
COGNITIVEOS_CHAT_API_KEY=...

# Gemini
COGNITIVEOS_EMBEDDING_PROVIDER_TYPE=gemini
COGNITIVEOS_EMBEDDING_MODEL_NAME=text-embedding-004
COGNITIVEOS_EMBEDDING_API_KEY=...
COGNITIVEOS_CHAT_PROVIDER_TYPE=gemini
COGNITIVEOS_CHAT_MODEL_NAME=gemini-2.5-flash
COGNITIVEOS_CHAT_API_KEY=...
```

## 🐳 Docker

Build:

```bash
docker build -t cognitiveos:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 -v $(pwd)/data:/app/data cognitiveos:latest
```

With a local `.env` file:

```bash
cp .env.example .env
docker compose up --build
```

## 🛠 Operations

Health report:

```bash
cognitiveos doctor
```

Live provider smoke test:

```bash
cognitiveos providers-test
```

Rebuild the vector index from current node embedding inputs:

```bash
cognitiveos reindex-embeddings
```

Refresh one `source_document` from its original source path or URI:

```bash
cognitiveos refresh-source-document <node-id>
```

Generate host bootstrap and mount artifacts:

```bash
cognitiveos bootstrap-host
```

This command generates local runtime bootstrap artifacts under `.cognitiveos/bootstrap/`. They are environment-specific and are not intended to be committed to the repository.

Supported host kinds currently include:

- `generic` as the default portable path
- `codex` with project-level auto-install support
- `claude-code`
- `claude-desktop`
- `gemini-cli`
- `cursor`

Inspect host bootstrap status:

```bash
cognitiveos host-bootstrap-status --host-kind generic
```

Submit first-start onboarding answers:

```bash
cognitiveos submit-host-onboarding \
  --answer display_name=Bruce \
  --answer role_team="Sr. Data Engineer" \
  --answer preferred_language=Chinese \
  --answer response_style="Concise, direct, pragmatic" \
  --answer workspace_goal="Build CognitiveOS as a host-facing memory runtime"
```

Install the Codex-specific cold-start mount into project files:

```bash
cognitiveos bootstrap-host --host-kind codex --install
```

Queue a dream run in the background:

```bash
cognitiveos dream --background
```

Inspect current dream status:

```bash
cognitiveos dream --inspect status
```

Inspect recent dream runs:

```bash
cognitiveos dream --inspect runs
```

List pending host dream compaction tasks:

```bash
cognitiveos dream --inspect tasks
```

Resolve one pending dream compaction with host-authored compressed content:

```bash
cognitiveos dream --task-id <task-id> \
  --title "Compressed cluster title" \
  --description "Compressed description" \
  --content "Compressed synthesis"
```

Resolve one pending compaction with heuristic fallback in the background:

```bash
cognitiveos dream --task-id <task-id> --use-heuristic
```

## 🧱 Data Model

Canonical node rules:

- `node_type` carries structural type only
- `durability` carries lifecycle tier only
- `tags` are open retrieval labels only
- `metadata` stores structured factual payload only

Current structural node types:

- `memory`
- `source_document`
- `source_collection`
- `super_node`

Current durability tiers:

- `working`
- `durable`
- `pinned`
- `ephemeral`

Common add parameters:

- `payload`: the primary input body or locator
- `name`: optional human-readable title stored on the node
- `tag`: repeated retrieval labels
- `durability`: optional lifecycle override
- `force`: bypass similarity or duplicate blocking where supported

File ingestion keeps one durable node by default:

- local files keep extracted raw text in `source_document.content`
- remote URLs keep a compact preservation note in `source_document.content`
- `source_document.description` stores a retrieval-oriented description
- `source_document.tags` carries user tags plus generated retrieval tags
- document embeddings are built from `description + tags`
- keyword search does not index full raw document content for `source_document`
- remote URL source kinds include `remote_page`, `remote_document`, `remote_feed_item`, `remote_video`, and `remote_binary`
- remote snapshots are preserved under `.cognitiveos/snapshots/`
- `read --include-content` on a remote `source_document` returns the preserved Markdown snapshot body when available
- binary remote snapshots stay preserved on disk and `read --include-content` falls back to the stored note instead of emitting raw bytes
- browser-exported local HTML / Markdown with a sibling `<stem>.cognitiveos-source.json` manifest is ingested as a remote source instead of a plain `local_file`

Browser-capture sidecar example:

```json
{
  "source_kind": "remote_page",
  "requested_url": "https://example.com/login",
  "final_url": "https://example.com/article",
  "title": "Captured Article",
  "capture_method": "browser_export_html",
  "captured_at": "2026-04-09T10:00:00+08:00",
  "http_status": 200,
  "exported_from": "chrome-devtools"
}
```

Folder ingestion also keeps one durable node by default:

- `source_collection.content` stores a compact root summary, not a recursive file dump
- `source_collection.description` stores a retrieval-oriented collection summary
- `source_collection.metadata.source.ref` stores the normalized folder path
- `source_collection.metadata.collection.class` is one of `repository`, `media_collection`, `document_collection`, or `workspace_bundle`
- folder scanning is `root_only` in v1, with bounded sample entries, file-type counts, and important markers
- folder embeddings are built from `description + tags + compact collection hints`
- `update` revises the stored summary only; it does not rescan the live folder

Canonical edge rules:

- `strength_score` is the only runtime edge-strength field
- edge lifecycle is controlled through `status` values such as `active`, `weak`, and `stale`
- `link` no longer accepts a public `weight` parameter

## 🌙 Dream Policy

Dream becomes due when either condition is met:

- `10` new memory events have accumulated since the last completed dream
- more than `24` hours have passed since the last completed dream and at least `5` new memory events have accumulated since that dream

When a chat model is configured, CognitiveOS auto-triggers dream on the next memory operation after the threshold is met.

When that auto-trigger path can complete entirely inside the runtime, CognitiveOS queues dream in the background instead of blocking the current memory operation.

When the age window is reached but fewer than `5` new events exist, CognitiveOS emits a deferred reminder instead of auto-running dream.

Dream compaction priority is:

- `chat provider`
- `host agent`
- `heuristic fallback`

When no chat model is configured, or when chat compaction fails during a dream run, CognitiveOS returns pending host compaction tasks with source data, a prepared digest, and a prompt for the host agent.

When no chat model is configured and dream is due, CognitiveOS emits a reminder on the next memory operation instead of auto-running dream. The reminder points the host to run `dream`, inspect pending compactions, and submit compressed clusters back through `dream` itself.

For host-core integrations, `dream` is the single public dream entrypoint. It covers:

- running a new dream job
- inspecting current dream status
- inspecting recent runs
- inspecting pending compaction tasks
- resolving one pending compaction task

Heuristic fallback can also run in the background after the host explicitly selects it during compaction resolution.

## 📁 Layout

```text
skills/          repository-local agent skill instructions
src/cognitiveos/
  cli/          CLI entrypoint
  db/           SQLite schema and repository
  extractors/   File and URL extraction pipeline
  mcp/          MCP server entrypoint
  providers/    Provider abstractions
```

## 📝 Notes

- Default runtime paths are user-level and host-agnostic under `~/.cognitiveos/`, so Codex, Claude Code, Gemini CLI, and other hosts can share one CognitiveOS runtime by default.
- When you pass `--db-path`, CognitiveOS infers the shared runtime root from that database path and keeps `MEMORY.MD`, logs, and snapshots anchored to the same runtime.
- Bootstrap artifacts remain project-local by default under `./.cognitiveos/bootstrap/`.
- Override runtime locations explicitly with `COGNITIVEOS_HOME`, `COGNITIVEOS_DB_PATH`, and `COGNITIVEOS_MEMORY_OUTPUT_PATH` when needed.
- `bootstrap-host --host-kind generic` is the default bootstrap path and generates local host guidance under `.cognitiveos/bootstrap/`.
- `bootstrap-host --host-kind codex --install` additionally writes a managed block into project-root `AGENTS.md` and `.codex/config.toml` for Codex-specific auto-mount, and pins `--project-root`, `--db-path`, and `--memory-output-path` in the generated MCP config.
- On first startup, the host should call `get_host_bootstrap_status`; when onboarding is required, ask the user the generated questions and submit them with `submit_host_onboarding` before depending on memory.
- Files under `.cognitiveos/bootstrap/` are runtime-generated local artifacts. Discover them through `host-bootstrap-status` or generate them with `bootstrap-host`; they are not meant to be repository-tracked source files.
- Durable engineering docs in this repository are kept in English.
