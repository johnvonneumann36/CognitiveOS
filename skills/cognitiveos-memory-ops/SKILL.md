---
name: cognitiveos-memory-ops
description: Use when a host needs to read, save, or update user-related memory in CognitiveOS. Prefer the supported CLI or MCP surface, load only the memory relevant to the current task, and keep only the most important durable preferences, profile facts, working patterns, goals, significant events, and reusable lessons or error workarounds.
---

# CognitiveOS Memory Ops

Use this skill when a host agent needs to operate memory through CognitiveOS rather than editing `MEMORY.MD` or SQLite directly.
## Core Rules
- Treat project-root `MEMORY.MD` as read-only baseline memory.
- For first-time mount or onboarding details, start with `host-bootstrap-status`; if bootstrap artifacts do not exist yet, generate them with `bootstrap-host`, then use `submit-host-onboarding` when required. Use `host_kind=generic` unless you explicitly need a host-specific path such as Codex auto-install. These bootstrap files are runtime-generated local artifacts, not repository-tracked docs.
- Start with `search`, then `read` only the node ids you actually need.
- Keep retrieval narrow by default: `include_neighbors=0` or `1`, `include_content=false`, `include_evidence=false`.
- `include_neighbors > 1`, `include_content=true`, or `include_evidence=true` can expand context very quickly.
- Keep writes high signal: store durable preferences, profile facts, goals, significant events, stable source anchors, and reusable lessons. Skip transient chatter.
- Prefer one compact root anchor for a repository or large folder, not many file-level writes.
- For project changes, migrations, release notes, or refactor milestones, prefer concise `content` memories instead of repeatedly overwriting the project root.

Default MCP command:
```bash
cognitiveos-mcp --transport stdio --profile host-core
```
## Memory Normalization
Write memory in complete, retrieval-friendly language instead of copying ambiguous chat fragments.
- Resolve pronouns into explicit actors when the referent is clear.
- Normalize relative time into an absolute date or bounded timeframe when it can be inferred.
- Replace vague verbs like "changed", "handled", or "had an issue" with the concrete event or state.
- Include the concrete file, folder, repository, URL, team, person, or project name when known.
- Prefer one self-contained sentence over shorthand that depends on surrounding chat.
- Do not invent precision when the referent or date is genuinely unclear.
Examples:
- user said: `她不爱吃香菜`
- store: `The user's mother does not like cilantro.`
- user said: `我昨天生病去医院了`
- store: `On 2026-04-08, the user went to the hospital because of illness.`
- user said: `这个项目上周改了检索逻辑`
- store: `In early April 2026, the project changed its retrieval logic.`
## Write Semantics
### `add --type content`
Use for ordinary memory text: conclusions, preferences, profile facts, working rules, decisions, or concise change summaries.
```bash
cognitiveos add --type content --payload "User prefers concise Chinese responses." --tag preference --tag language
```
### `add --type file`
Use for a local file, `file://` URI, or URL-backed source document.
- Local files stay local-file style sources.
- Remote `http` and `https` URLs preserve a snapshot by default.
- Re-adding the same remote URL normally blocks as a conflict; use `force=true` to refresh in place.
- Use `update` only for manual override of the stored node body/summary, not for re-fetch.
```bash
cognitiveos add --type file --payload ".\\notes\\profile.md" --tag profile
cognitiveos add --type file --payload "https://example.com/article" --name "Example Article" --tag article --tag reference
```
### `add --type folder`
Use for a local folder, repository root, or media/document collection root.
- `payload` must be the folder path itself, not a handwritten description.
- `folder` creates one compact `source_collection` root.
- It is local-path only in v1 and does not recursively import children.
```bash
cognitiveos add --type folder --payload ".\\Pictures\\Japan Trip 2025" --name "Japan Trip 2025 Photos" --tag travel --tag japan --tag photos
```
### Browser-assisted remote capture
If a remote page needs login, browser state, or paywall navigation:
- use the host's existing browser tools first
- then preserve the accessible source through the normal `add --type file` path
- preferred forms are the final accessible URL, a browser-exported local HTML file, or a host-prepared Markdown file
- when preserving browser-exported local HTML or Markdown as a remote source, place a sibling manifest named `<stem>.cognitiveos-source.json` next to the export
### `update`
Use `update` to revise an existing node.
- `content` is overwrite-only, not append.
- `update` on a `source_collection` revises the stored compact summary only; it does not rescan the live folder.
- If `tag` includes `__delete__`, the runtime deletes the node. Use that only for truly incorrect memory.
- `content`: ordinary memory text such as conclusions, preferences, profile facts, working rules, decisions, or concise change summaries.
- `file`: a local file, `file://` URI, or URL-backed source document.
  - local files stay local-file style sources
  - remote `http` and `https` URLs preserve a snapshot by default
  - re-adding the same remote URL normally blocks as a conflict; use `force=true` to refresh in place
  - use `update` only for manual override of the stored node body/summary, not for re-fetch
- `folder`: a local folder, repository root, or media/document collection root.
  - `payload` must be the folder path itself, not a handwritten description
  - `folder` creates one compact `source_collection` root
  - it is local-path only in v1 and does not recursively import children
- Browser-assisted remote capture:
  - use the host's existing browser tools first for login, browser state, or paywall navigation
  - then preserve the accessible source through the normal `add --type file` path
  - preferred forms are the final accessible URL, a browser-exported local HTML file, or a host-prepared Markdown file
  - when preserving browser-exported local HTML or Markdown as a remote source, place a sibling manifest named `<stem>.cognitiveos-source.json` next to the export
- `update` is overwrite-only, not append.
- `update` on a `source_collection` revises the stored compact summary only; it does not rescan the live folder.
- If `tag` includes `__delete__`, the runtime deletes the node. Use that only for truly incorrect memory.
- Use `link` / `unlink` only when the relationship itself matters later.

## Parameters
- `payload`: primary input body or locator.
- `name`: optional stable human-readable title for the stored node.
- `tag`: repeated retrieval labels; include stable keywords people may search later.
- `durability`: `working`, `durable`, `pinned`, or `ephemeral`.
- `force`: bypass similarity or duplicate blocking where supported.
- `relation`: directed edge label such as `informs`, `supports`, or `depends_on`.
- `include_content`: return full node content only when the body is actually needed.
- `include_evidence`: include evidence-style result nodes when that mode exists; usually keep `false`.
Remote snapshot note:
- for remote `source_document`, `include_content=true` returns the preserved Markdown snapshot body when available
- for binary remote snapshots such as PDF, `include_content=true` falls back to the stored note and leaves the preserved file on disk
Durability guide:
- `working`: short-lived task memory
- `durable`: reusable memory that should persist
- `pinned`: always-important memory that should remain highly visible
- `ephemeral`: runtime-only helper state; hosts usually should not choose this
## System / Profile Memory
Use system/profile memory only when the fact should shape future host behavior.
- Typical cases: preferred language, response style, role/team, workspace goal, durable operating constraints.
- Preferred path: `submit-host-onboarding`.
- If you must write it manually, use `add --type content` with stable tags such as `profile` or `bootstrap`.
- Use `pinned` only when the fact must remain strongly visible; otherwise prefer `durable`.
- Do not treat every user fact as system/profile memory.
## Tool Surface
### `search`
Use `search` to find candidates before `read`.
Input:
- `keyword: str | null`
- `query: str | null`
- `top_k: int = 5`
- `include_neighbors: int = 1`
- `include_evidence: bool = false`
```bash
cognitiveos search --query "user language preference" --keyword Chinese --top-k 5
```
### `read`
Use `read` to inspect selected node ids.
- `ids: string[]`
- `include_content: bool = false`
- `type: "content" | "file" | "folder"`
- `payload: str`
- `tag: string[]`
- `force: bool = false`
- `durability: "working" | "durable" | "pinned" | "ephemeral" | null`
- `name: str | null`
- receipt with `status`, `action_taken`, `node_id`
- on block, inspect `reason`, `suggestion`, and `conflicting_nodes`
### `dream`
Use `dream` to inspect status, run consolidation, or resolve one pending task.
- inspect: `dream --inspect status|runs|tasks`
- run: `dream`
- resolve: `dream --task-id ... --title ... --description ... --content ...`
- heuristic fallback: `dream --task-id ... --use-heuristic`

If the runtime emits a dream reminder, continue the main task and queue dream only when the host supports background work.
