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

<p align="center">
  <strong>Local-first cognitive graph runtime for Codex, Claude Code, Gemini CLI, and other MCP-capable hosts.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/cognitiveos/">PyPI</a>
  |
  <a href="https://github.com/johnvonneumann36/CognitiveOS/releases">Releases</a>
  |
  <a href="https://github.com/johnvonneumann36/CognitiveOS/tree/main/docs">Docs</a>
</p>

> [!IMPORTANT]
> `v0.1.3` is live on PyPI: `pip install cognitiveos`
>
> For compact host mounts, prefer `--profile compact-core`.
> `compact-core` exposes `search`, `read`, `add`, `update`, `link`, and `dream`,
> while intentionally omitting `unlink`.

This repository is a production-oriented project skeleton for a local-first agent memory runtime.

At a glance:

| Need | Use | Why |
| --- | --- | --- |
| Install CognitiveOS quickly | `pip install cognitiveos` | Fastest path to a working local runtime |
| Mount into a compact host surface | `cognitiveos-mcp --profile compact-core` | Smaller tool surface, clearer prompts |
| Run a general MCP host | `cognitiveos-mcp --profile host-core` | Full core memory workflow including `unlink` |
| Explore docs and releases | GitHub docs and release pages | Best entry point for release notes and implementation details |

CognitiveOS focuses on:

- installable Python packaging via `pip`
- CLI operations for database lifecycle and graph manipulation
- an installable MCP server runtime
- Docker-based deployment for HTTP MCP serving
- extensible provider and extractor boundaries

## Start Here

### Package Install

Use this when you just want a local runtime:

```bash
pip install cognitiveos
cognitiveos init-db
cognitiveos-mcp --transport stdio --profile host-core
```

### Compact Host Mount

Use this when CognitiveOS is being mounted into a host that benefits from a smaller tool surface:

```bash
cognitiveos-mcp --transport stdio --profile compact-core
```

To make Codex load the repository's memory skill for new conversations, copy the bundled skill into your user-level Codex skills directory:

macOS / Linux:

```bash
mkdir -p ~/.codex/skills
cp -R ./skills/cognitiveos-memory-ops ~/.codex/skills/cognitiveos-memory-ops
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.codex\skills" | Out-Null
Copy-Item -Recurse -Force .\skills\cognitiveos-memory-ops "$HOME\.codex\skills\cognitiveos-memory-ops"
```

### Repository Development

Use this when you are developing CognitiveOS itself:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

### MCP Profile Guide

| Profile | Intended use | Public tools |
| --- | --- | --- |
| `compact-core` | Compact host mounts | `search`, `read`, `add`, `update`, `link`, `dream` |
| `host-core` | General MCP hosts | `search`, `read`, `add`, `update`, `link`, `unlink`, `dream` |
| `operator` | Maintenance / graph operations | core tools plus relationship and provider operations |
| `bootstrap` | Onboarding and host install flows | bootstrap-only tools |
| `full` | Everything | all public tools |

### Public Entry Points

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

The current milestone is an MVP foundation with working runtime, retrieval, and host bootstrap flows.

| Area | Included |
| --- | --- |
| Core runtime | SQLite-backed nodes, edges, audit logs, `sqlite-vec`, FTS5 |
| Retrieval | keyword, semantic, and hybrid ranking with bounded neighbor traversal |
| Ingestion | content, local files, folders, remote URLs, remote snapshot preservation |
| Graph updates | `add`, `update`, `link`, audit tracing, canonical node and edge semantics |
| Dream flow | clustering, chat-first compaction, host fallback tasks, `MEMORY.MD` generation |
| Host integration | CLI, MCP, bootstrap bundle generation, managed host install support |
| Operations | health checks, provider smoke tests, embedding reindexing, background jobs |

Highlights:

- content, file, and folder ingestion use durable `source_document` and `source_collection` roots
- remote extraction is Trafilatura-first and preserves typed source metadata plus snapshots
- folder inspection stays root-only in v1 to keep memory compact and predictable
- first-start onboarding can seed pinned system-profile memory through structured profile metadata
- background dream execution is available for runtime-only compaction paths

## 🚀 Quick Start

### Fast Path

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
cognitiveos init-db
cognitiveos-mcp --transport stdio --profile host-core
```

### Common Commands

| Goal | Command |
| --- | --- |
| Initialize the workspace database | `cognitiveos init-db` |
| Add raw text memory | `cognitiveos add --type content --payload "..." --tag tech` |
| Add a file or URL | `cognitiveos add --type file --payload "..."` |
| Add a folder root | `cognitiveos add --type folder --payload ".\\path"` |
| Search by keyword | `cognitiveos search --keyword CognitiveOS --top-k 5` |
| Search by semantic query | `cognitiveos search --query "graph memory runtime"` |
| Start MCP over stdio | `cognitiveos-mcp --transport stdio --profile host-core` |
| Start compact MCP profile | `cognitiveos-mcp --transport stdio --profile compact-core` |

### Add Memory

```bash
cognitiveos add --type content --payload "CognitiveOS stores graph-shaped memory." --tag tech --tag memory --tag graph
```

### Add Source Anchors

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

> [!TIP]
> Use repeated `--tag` for retrieval labels.
> Use `--name` when the node should keep a stable human-readable title.
> Profile-oriented writes such as identity, communication preferences, workspace context, and engineering preferences are merged into canonical profile nodes instead of creating overlapping duplicates.

Source-handling notes:

- remote `http` and `https` URLs are preserved by default under `.cognitiveos/snapshots/`, using Markdown for readable text/page captures and binary files for formats such as PDF
- when a page requires login or browser state, use the host's existing browser tools to open the accessible page first; CognitiveOS v1 does not automate browser capture itself
- adding the same remote URL again warns on conflict by default; use `force=true` to refresh that source in place
- host-assisted inputs can be: the final accessible URL, a browser-exported local HTML file, or a host-prepared Markdown file
- for browser-exported local HTML or Markdown, add a sidecar file named `<stem>.cognitiveos-source.json` beside the export to preserve the original remote URL and capture metadata

> [!WARNING]
> `include_neighbors`, `include_content`, and `include_evidence` can expand returned context quickly.
> Start narrow, then widen only when the initial result is clearly insufficient.

For routine recall, prefer `include_neighbors=0` or `1`, `include_content=false`, and `include_evidence=false`.

### Search

```bash
cognitiveos search --keyword CognitiveOS --top-k 5 --include-neighbors 1
```

Run hybrid search:

```bash
cognitiveos search --query "graph memory runtime" --keyword CognitiveOS --top-k 5
```

### MCP Transport

```bash
cognitiveos-mcp --transport stdio --profile host-core
```

For compact host mounts, use the reduced tool surface:

```bash
cognitiveos-mcp --transport stdio --profile compact-core
```

Run the MCP server over Streamable HTTP:

```bash
cognitiveos-mcp --transport streamable-http --profile host-core --host 0.0.0.0 --port 8000 --path /mcp
```

## 🔌 Provider Configuration

Embedding and chat providers are optional.

### What Changes When Providers Are Enabled

- `add` performs similarity probing before insert unless `force=true`
- `search --query ...` performs semantic search even without keywords
- `search --query ... --keyword ...` performs hybrid ranking
- missing node embeddings are lazily backfilled

### Provider Support Matrix

| Provider | Chat | Embeddings | Notes |
| --- | --- | --- | --- |
| `ollama` | yes | yes | local-first default option |
| `openai` | yes | yes | hosted API |
| `gemini` | yes | yes | hosted API |
| `anthropic` | yes | no | chat only |
| `local_huggingface` | yes | yes | requires `pip install 'cognitiveos[local-hf]'` |

### Environment Variables

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

### Example Provider Setups

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

### Build

```bash
docker build -t cognitiveos:latest .
```

### Run

```bash
docker run --rm -p 8000:8000 -v $(pwd)/data:/app/data cognitiveos:latest
```

### Use a Local `.env`

```bash
cp .env.example .env
docker compose up --build
```

## 🛠 Operations

### Runtime Checks

| Goal | Command |
| --- | --- |
| Health report | `cognitiveos doctor` |
| Live provider smoke test | `cognitiveos providers-test` |
| Rebuild the vector index | `cognitiveos reindex-embeddings` |
| Refresh one `source_document` | `cognitiveos refresh-source-document <node-id>` |

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

### Host Bootstrap

```bash
cognitiveos bootstrap-host
```

This command generates local runtime bootstrap artifacts under `.cognitiveos/bootstrap/`. They are environment-specific and are not intended to be committed to the repository.

Supported host kinds:

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

Install the managed cold-start mount into project files for a supported host target:

```bash
cognitiveos bootstrap-host --host-kind codex --install
```

Optionally install the bundled skill into Codex so new conversations can recall and save CognitiveOS memory through the documented workflow:

macOS / Linux:

```bash
mkdir -p ~/.codex/skills
cp -R ./skills/cognitiveos-memory-ops ~/.codex/skills/cognitiveos-memory-ops
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force "$HOME\.codex\skills" | Out-Null
Copy-Item -Recurse -Force .\skills\cognitiveos-memory-ops "$HOME\.codex\skills\cognitiveos-memory-ops"
```

### Dream Operations

| Goal | Command |
| --- | --- |
| Queue dream in background | `cognitiveos dream --background` |
| Inspect dream status | `cognitiveos dream --inspect status` |
| Inspect dream runs | `cognitiveos dream --inspect runs` |
| Inspect pending tasks | `cognitiveos dream --inspect tasks` |

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

### Canonical Node Rules

| Field | Meaning |
| --- | --- |
| `node_type` | structural type only |
| `durability` | lifecycle tier only |
| `tags` | open retrieval labels |
| `metadata` | structured factual payload |

### Structural Node Types

- `memory`
- `source_document`
- `source_collection`
- `super_node`

### Durability Tiers

- `working`
- `durable`
- `pinned`
- `ephemeral`

### Common `add` Parameters

- `payload`: the primary input body or locator
- `name`: optional human-readable title stored on the node
- `tag`: repeated retrieval labels
- `durability`: optional lifecycle override
- `force`: bypass similarity or duplicate blocking where supported

### File Ingestion Defaults

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

### Browser-Capture Sidecar Example

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

### Folder Ingestion Defaults

- `source_collection.content` stores a compact root summary, not a recursive file dump
- `source_collection.description` stores a retrieval-oriented collection summary
- `source_collection.metadata.source.ref` stores the normalized folder path
- `source_collection.metadata.collection.class` is one of `repository`, `media_collection`, `document_collection`, or `workspace_bundle`
- folder scanning is `root_only` in v1, with bounded sample entries, file-type counts, and important markers
- folder embeddings are built from `description + tags + compact collection hints`
- `update` revises the stored summary only; it does not rescan the live folder

### Canonical Edge Rules

- `strength_score` is the only runtime edge-strength field
- edge lifecycle is controlled through `status` values such as `active`, `weak`, and `stale`
- `link` no longer accepts a public `weight` parameter

## 🌙 Dream Policy

### Trigger Conditions

| Condition | Dream becomes due when |
| --- | --- |
| Event threshold | `10` new memory events have accumulated since the last completed dream |
| Age threshold | more than `24` hours have passed since the last completed dream and at least `5` new events have accumulated |

### Runtime Behavior

- when a chat model is configured, CognitiveOS auto-triggers dream on the next memory operation after the threshold is met
- when that auto-trigger path can complete entirely inside the runtime, CognitiveOS queues dream in the background instead of blocking the current operation
- when the age window is reached but fewer than `5` new events exist, CognitiveOS emits a deferred reminder instead of auto-running dream

### Compaction Priority

1. `chat provider`
2. `host agent`
3. `heuristic fallback`

### Host Fallback

- when no chat model is configured, or when chat compaction fails during a dream run, CognitiveOS returns pending host compaction tasks with source data, a prepared digest, and a prompt for the host agent
- when no chat model is configured and dream is due, CognitiveOS emits a reminder on the next memory operation instead of auto-running dream
- the reminder points the host to run `dream`, inspect pending compactions, and submit compressed clusters back through `dream`

### Public `dream` Entry Point

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

### Runtime Paths

- default runtime paths are user-level and host-agnostic under `~/.cognitiveos/`
- passing `--db-path` keeps `MEMORY.MD`, logs, and snapshots anchored to the same runtime root
- bootstrap artifacts remain project-local by default under `./.cognitiveos/bootstrap/`
- runtime locations can be overridden with `COGNITIVEOS_HOME`, `COGNITIVEOS_DB_PATH`, and `COGNITIVEOS_MEMORY_OUTPUT_PATH`

### Bootstrap and Install

- `bootstrap-host --host-kind generic` is the default bootstrap path and generates local host guidance under `.cognitiveos/bootstrap/`
- `bootstrap-host --host-kind codex --install` writes a managed block into project-root `AGENTS.md` and `.codex/config.toml`
- the managed install uses the reduced `compact-core` profile and pins `--project-root`, `--db-path`, and `--memory-output-path`; for Codex, the mounted memory file is `~/.codex/MEMORY.MD` instead of a project-root `MEMORY.MD`
- named host bootstrap targets remain registered memory outputs so future snapshot and dream sync can fan out across installed host surfaces
- on first startup, the host should call `get_host_bootstrap_status`; when onboarding is required, ask the generated questions and submit them with `submit_host_onboarding`
- files under `.cognitiveos/bootstrap/` are runtime-generated local artifacts and are not meant to be repository-tracked source files

### Project Conventions

- Durable engineering docs in this repository are kept in English.
