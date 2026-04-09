from __future__ import annotations

import hashlib
import logging
import mimetypes
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from cognitiveos.config import AppSettings
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.exceptions import InvalidPayloadError
from cognitiveos.extractors.defaults import DefaultContentExtractor
from cognitiveos.models import ConflictNode
from cognitiveos.metadata_shapes import metadata_source_ref
from cognitiveos.models import PHYSICAL_MAX_NODE_CONTENT_CHARS, NodeRecord, Receipt, SearchResult


logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    def __init__(
        self,
        *,
        settings: AppSettings,
        repository: SQLiteRepository,
        extractor: DefaultContentExtractor,
        default_actor: str,
        summarize_content: Callable[[str], str],
        build_document_profile: Callable[..., dict[str, Any]],
        embed_content: Callable[[str], list[float] | None],
        find_similarity_conflicts: Callable[[list[float]], list[dict[str, Any]]],
        schedule_semantic_neighbor_refresh: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.extractor = extractor
        self.default_actor = default_actor
        self.summarize_content = summarize_content
        self.build_document_profile = build_document_profile
        self.embed_content = embed_content
        self.find_similarity_conflicts = find_similarity_conflicts
        self.schedule_semantic_neighbor_refresh = schedule_semantic_neighbor_refresh
        self._last_source_conflicts: list[NodeRecord] = []

    def add_file_node(
        self,
        *,
        payload: str,
        tags: list[str],
        durability: str,
        force: bool,
        name: str | None,
    ) -> Receipt:
        self._last_source_conflicts = []
        source_document = self.ingest_file_document(
            payload,
            display_name=name,
            tags=tags,
            durability=durability,
            force=force,
            update_mode="source_add",
        )
        if source_document is None:
            return Receipt(
                status="blocked",
                action_taken="none",
                reason=(
                    "A source document with matching source metadata or "
                    "source hash already exists."
                ),
                suggestion=(
                    "Use `update` for a manual override, `refresh-source-document` to "
                    "re-extract the existing node, or re-run `add` with force=true."
                ),
                conflicting_nodes=[
                    ConflictNode(id=node.id, name=node.name, similarity=1.0)
                    for node in self._last_source_conflicts[:3]
                ],
            )
        if self.schedule_semantic_neighbor_refresh is not None:
            self.schedule_semantic_neighbor_refresh(source_document.id)
        last_update_mode = (
            source_document.metadata.get("document_profile", {}).get("last_update_mode")
            if isinstance(source_document.metadata, dict)
            else None
        )
        return Receipt(
            status="success",
            action_taken=(
                "source_document_refreshed"
                if last_update_mode == "source_refresh"
                else "created"
            ),
            node_id=source_document.id,
        )

    def refresh_source_document(self, *, node_id: str) -> Receipt:
        node = self.repository.get_node(node_id)
        if node.node_type != "source_document":
            raise InvalidPayloadError("Source refresh requires a source_document node.")
        source_locator = metadata_source_ref(node.metadata)
        if not source_locator:
            raise InvalidPayloadError("Document source locator is missing; cannot refresh source.")
        generated_tags = list(node.metadata.get("document_profile", {}).get("generated_tags", []))
        generated_lc = {tag.strip().lower() for tag in generated_tags if isinstance(tag, str)}
        user_tags = [tag for tag in node.tags if tag.strip().lower() not in generated_lc]
        source_document = self.ingest_file_document(
            source_locator,
            display_name=node.name,
            tags=user_tags,
            durability=node.durability,
            force=True,
            source_document_id=node.id,
            prior_node=node,
            update_mode="source_refresh",
        )
        if self.schedule_semantic_neighbor_refresh is not None:
            self.schedule_semantic_neighbor_refresh(source_document.id)
        return Receipt(
            status="success",
            action_taken="source_document_refreshed",
            node_id=node.id,
        )

    @staticmethod
    def filter_search_results(
        results: list[SearchResult],
        *,
        top_k: int,
        include_evidence: bool,
    ) -> list[SearchResult]:
        return results[:top_k]

    def ingest_file_document(
        self,
        source_locator: str,
        *,
        display_name: str | None,
        tags: list[str],
        durability: str,
        force: bool,
        source_document_id: str | None = None,
        prior_node: NodeRecord | None = None,
        update_mode: str = "source_add",
    ) -> NodeRecord | None:
        self._last_source_conflicts = []
        extracted = self.extractor.extract(source_locator)
        extracted_content = extracted.content.strip()
        if not extracted_content:
            raise InvalidPayloadError("Content payload cannot be empty.")

        document_tags = list(dict.fromkeys(tags))
        source_metadata = self._build_source_metadata(
            source_locator=source_locator,
            extracted_metadata=extracted.metadata,
            content=extracted_content,
        )
        node_id = source_document_id or f"node_{uuid4().hex}"
        node_name = display_name or extracted.name
        should_probe_conflicts = source_document_id is None and (
            not force or self._is_remote_source_metadata(source_metadata)
        )
        if should_probe_conflicts:
            conflicts = self._find_metadata_conflicts(source_metadata=source_metadata)
            if conflicts:
                self._last_source_conflicts = conflicts
                existing_remote = self._find_remote_same_ref_conflict(
                    conflicts=conflicts,
                    source_metadata=source_metadata,
                )
                if existing_remote is not None and force:
                    source_document_id = existing_remote.id
                    node_id = existing_remote.id
                    prior_node = existing_remote
                    node_name = display_name or existing_remote.name or extracted.name
                    update_mode = "source_refresh"
                else:
                    return None
        if not self._is_remote_source_metadata(source_metadata):
            self._validate_document_content(extracted_content)
        profile = self.build_document_profile(
            content=extracted_content,
            tags=document_tags,
            source_metadata=source_metadata,
            name=node_name,
        )
        effective_tags = profile.get("tags", document_tags)
        description = profile.get("description") or self.summarize_content(extracted_content)
        embedding_input = profile.get("embedding_input") or description
        generated_tags = profile.get("generated_tags", [])
        embedding = self.embed_content(embedding_input)
        stored_content = extracted_content
        if self._is_remote_source_metadata(source_metadata):
            snapshot_metadata = self._write_remote_snapshot(
                node_id=node_id,
                extracted_content=extracted_content,
                source_metadata=source_metadata,
                extracted_metadata=extracted.metadata,
            )
            source_metadata["snapshot"] = snapshot_metadata
            stored_content = self._build_remote_content_note(
                source_metadata=source_metadata,
                snapshot_metadata=snapshot_metadata,
            )

        source_node = NodeRecord(
            id=node_id,
            name=node_name,
            description=description,
            content=stored_content,
            embedding=None,
            tags=effective_tags,
            metadata={
                **source_metadata,
                "document_profile": self._build_document_profile_metadata(
                    generated_tags=generated_tags,
                    prior_node=prior_node,
                    update_mode=update_mode,
                ),
            },
            node_type="source_document",
            durability=durability,
        )
        source_node.embedding = embedding
        self._persist_node(
            source_node,
            action_type="create" if source_document_id is None else "update",
        )
        return source_node

    def _persist_node(self, node: NodeRecord, *, action_type: str) -> None:
        if action_type == "create":
            self.repository.create_node(
                node,
                actor=self.default_actor,
                action_type=action_type,
            )
            return
        self.repository.overwrite_node(
            node,
            actor=self.default_actor,
            action_type=action_type,
        )

    def _build_source_metadata(
        self,
        *,
        source_locator: str,
        extracted_metadata: dict[str, Any],
        content: str,
    ) -> dict[str, Any]:
        source_path = extracted_metadata.get("file_path")
        source_uri = extracted_metadata.get("uri")
        requested_uri = extracted_metadata.get("requested_uri")
        suffix = extracted_metadata.get("suffix") or (
            Path(source_path).suffix if source_path else ""
        )
        mime_type = (
            extracted_metadata.get("media_type")
            or mimetypes.guess_type(source_path or source_locator)[0]
            or "text/plain"
        )
        source_kind = extracted_metadata.get("source_kind") or extracted_metadata.get("source") or (
            "remote_page" if source_uri else "local_file"
        )
        source_ref = source_uri or source_path or source_locator
        metadata = {
            "source": {
                "kind": source_kind,
                "ref": source_ref,
                "hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "mime_type": mime_type,
                **({"title": extracted_metadata.get("title")} if extracted_metadata.get("title") else {}),
                **(
                    {"http_status": extracted_metadata.get("http_status")}
                    if extracted_metadata.get("http_status") is not None
                    else {}
                ),
                **({"etag": extracted_metadata.get("etag")} if extracted_metadata.get("etag") else {}),
                **(
                    {"content_length": extracted_metadata.get("content_length")}
                    if extracted_metadata.get("content_length") is not None
                    else {}
                ),
                **({"requested_ref": requested_uri} if requested_uri else {}),
                **(
                    {"file_name": extracted_metadata.get("file_name")}
                    if extracted_metadata.get("file_name")
                    else {}
                ),
                **(
                    {"modified_at": extracted_metadata.get("modified_at")}
                    if extracted_metadata.get("modified_at")
                    else {}
                ),
            },
            "document": {
                "content_length": len(content),
                "token_estimate": self._estimate_tokens(content),
            },
        }
        if suffix:
            metadata["source"]["suffix"] = suffix
        passthrough = {
            key: value
            for key, value in extracted_metadata.items()
            if key
            not in {
                "source",
                "file_path",
                "uri",
                "requested_uri",
                "media_type",
                "suffix",
                "content_length",
                "title",
                "http_status",
                "etag",
                "snapshot_format",
                "snapshot_suffix",
                "snapshot_bytes",
            }
        }
        metadata.update(passthrough)
        return metadata

    @staticmethod
    def _estimate_tokens(content: str) -> int:
        return max(1, len(content) // 4)

    def _validate_document_content(self, content: str) -> None:
        if len(content) > PHYSICAL_MAX_NODE_CONTENT_CHARS:
            raise InvalidPayloadError(
                "Document content exceeds the physical node content limit. "
                f"Current limit: {PHYSICAL_MAX_NODE_CONTENT_CHARS} characters."
            )

    def _find_metadata_conflicts(self, *, source_metadata: dict[str, Any]) -> list[NodeRecord]:
        source = source_metadata.get("source", {})
        source_kind = source.get("kind")
        source_ref = source.get("ref")
        source_hash = source.get("hash")
        file_name = source.get("file_name")
        modified_at = source.get("modified_at")
        conflicts: list[NodeRecord] = []
        for node in self.repository.list_nodes_by_type(node_type="source_document"):
            node_source = (node.metadata or {}).get("source", {})
            node_source_kind = node_source.get("kind")
            node_ref = node_source.get("ref")
            node_source_hash = node_source.get("hash")
            node_file_name = node_source.get("file_name")
            node_modified_at = node_source.get("modified_at")
            same_ref = bool(source_ref and node_ref and source_ref == node_ref)
            if self._is_remote_source_kind(source_kind):
                if self._is_remote_source_kind(node_source_kind) and same_ref:
                    conflicts.append(node)
                continue
            same_hash = bool(source_hash and node_source_hash and source_hash == node_source_hash)
            same_file_version = bool(
                file_name
                and modified_at
                and node_file_name == file_name
                and node_modified_at == modified_at
            )
            if same_ref or same_hash or same_file_version:
                conflicts.append(node)
        return conflicts

    @staticmethod
    def _is_remote_source_metadata(source_metadata: dict[str, Any]) -> bool:
        source = source_metadata.get("source", {})
        return DocumentIngestionPipeline._is_remote_source_kind(source.get("kind"))

    @staticmethod
    def _is_remote_source_kind(source_kind: str | None) -> bool:
        return bool(source_kind and source_kind.startswith("remote_"))

    @staticmethod
    def _find_remote_same_ref_conflict(
        *,
        conflicts: list[NodeRecord],
        source_metadata: dict[str, Any],
    ) -> NodeRecord | None:
        source_ref = (source_metadata.get("source") or {}).get("ref")
        if not source_ref:
            return None
        for node in conflicts:
            node_source = (node.metadata or {}).get("source", {})
            if (
                DocumentIngestionPipeline._is_remote_source_kind(node_source.get("kind"))
                and node_source.get("ref") == source_ref
            ):
                return node
        return None

    def _write_remote_snapshot(
        self,
        *,
        node_id: str,
        extracted_content: str,
        source_metadata: dict[str, Any],
        extracted_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        source = source_metadata.get("source", {})
        preserved_at = datetime.now(UTC).isoformat()
        capture_method = source_metadata.get("capture_method") or "httpx_trafilatura"
        snapshot_format = extracted_metadata.get("snapshot_format") or "markdown"
        self.settings.snapshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            if snapshot_format == "binary":
                snapshot_suffix = extracted_metadata.get("snapshot_suffix") or ".bin"
                snapshot_path = self.settings.snapshot_dir / f"{node_id}{snapshot_suffix}"
                snapshot_bytes = extracted_metadata.get("snapshot_bytes")
                if not isinstance(snapshot_bytes, (bytes, bytearray)) or not snapshot_bytes:
                    raise InvalidPayloadError("Binary snapshot payload is missing.")
                temp_path = snapshot_path.with_name(f"{snapshot_path.stem}.{uuid4().hex}.tmp")
                temp_path.write_bytes(bytes(snapshot_bytes))
                temp_path.replace(snapshot_path)
                content_hash = hashlib.sha256(bytes(snapshot_bytes)).hexdigest()
            else:
                snapshot_path = self.settings.snapshot_dir / f"{node_id}.md"
                markdown = self._build_snapshot_markdown(
                    requested_ref=source.get("requested_ref") or source.get("ref") or "",
                    final_ref=source.get("ref") or "",
                    preserved_at=preserved_at,
                    mime_type=source.get("mime_type") or "text/plain",
                    modified_at=source.get("modified_at"),
                    capture_method=capture_method,
                    title=source.get("title"),
                    http_status=source.get("http_status"),
                    etag=source.get("etag"),
                    body=extracted_content,
                )
                temp_path = snapshot_path.with_name(f"{snapshot_path.stem}.{uuid4().hex}.tmp")
                temp_path.write_text(markdown, encoding="utf-8")
                temp_path.replace(snapshot_path)
                content_hash = hashlib.sha256(extracted_content.encode("utf-8")).hexdigest()
        except Exception:
            logger.exception(
                "remote_snapshot_write_failed ref=%s format=%s",
                source.get("ref"),
                snapshot_format,
            )
            raise
        return {
            "path": str(snapshot_path.resolve()),
            "format": snapshot_format,
            "preserved_at": preserved_at,
            "content_hash": content_hash,
            "capture_method": capture_method,
        }

    @staticmethod
    def _build_snapshot_markdown(
        *,
        requested_ref: str,
        final_ref: str,
        preserved_at: str,
        mime_type: str,
        modified_at: str | None,
        capture_method: str,
        title: str | None,
        http_status: int | None,
        etag: str | None,
        body: str,
    ) -> str:
        lines = [
            "---",
            f"requested_url: {requested_ref}",
            f"final_url: {final_ref}",
            f"preserved_at: {preserved_at}",
            f"media_type: {mime_type}",
            f"capture_method: {capture_method}",
        ]
        if title:
            lines.append(f"title: {title}")
        if http_status is not None:
            lines.append(f"http_status: {http_status}")
        if etag:
            lines.append(f"etag: {etag}")
        if modified_at:
            lines.append(f"last_modified: {modified_at}")
        lines.extend(["---", "", body.strip(), ""])
        return "\n".join(lines)

    @staticmethod
    def _build_remote_content_note(
        *,
        source_metadata: dict[str, Any],
        snapshot_metadata: dict[str, Any],
    ) -> str:
        source = source_metadata.get("source", {})
        lines = [
            "Remote source snapshot preserved.",
            f"Source kind: {source.get('kind')}",
            f"Title: {source.get('title') or source.get('file_name') or source.get('ref')}",
            f"Requested URL: {source.get('requested_ref') or source.get('ref')}",
            f"Final URL: {source.get('ref')}",
            f"Snapshot path: {snapshot_metadata.get('path')}",
            f"Snapshot format: {snapshot_metadata.get('format')}",
            f"Preserved at: {snapshot_metadata.get('preserved_at')}",
            f"Media type: {source.get('mime_type') or 'text/plain'}",
        ]
        if source.get("modified_at"):
            lines.append(f"Last modified: {source['modified_at']}")
        return "\n".join(lines)

    @staticmethod
    def _build_document_profile_metadata(
        *,
        generated_tags: list[str],
        prior_node: NodeRecord | None,
        update_mode: str,
    ) -> dict[str, Any]:
        normalized_update_mode = DocumentIngestionPipeline._normalize_document_update_mode(
            update_mode
        )
        prior_profile = (
            prior_node.metadata.get("document_profile", {})
            if prior_node is not None and isinstance(prior_node.metadata, dict)
            else {}
        )
        return {
            **prior_profile,
            "generated_tags": generated_tags,
            "sync_status": DocumentIngestionPipeline._sync_status_for_update_mode(
                normalized_update_mode
            ),
            "source_state": DocumentIngestionPipeline._source_state_for_update_mode(
                normalized_update_mode
            ),
            "last_update_mode": normalized_update_mode,
        }

    @staticmethod
    def _normalize_document_update_mode(update_mode: str) -> str:
        if update_mode == "add":
            return "source_add"
        if update_mode == "manual_refresh":
            return "profile_refresh"
        return update_mode

    @staticmethod
    def _sync_status_for_update_mode(update_mode: str) -> str:
        if update_mode == "manual_override":
            return "detached"
        return "source_synced"

    @staticmethod
    def _source_state_for_update_mode(update_mode: str) -> str:
        if update_mode == "manual_override":
            return "detached"
        return "attached"
