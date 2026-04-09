from __future__ import annotations

from typing import Any

LEGACY_NODE_FIELDS = {
    "source",
    "source_path",
    "source_uri",
    "source_locator",
    "source_hash",
    "source_kind",
    "source_ref",
    "mime_type",
    "media_type",
    "suffix",
    "file_name",
    "modified_at",
    "content_length",
    "token_estimate",
    "chunk_index",
    "chunk_count",
    "start_offset",
    "end_offset",
    "section_title",
    "summary_level",
    "summary_node_id",
    "chunk_node_ids",
    "prev_chunk_id",
    "node_type",
}

LEGACY_EDGE_FIELDS = {
    "creation_mode",
    "created_by",
    "reason",
    "redirected_from",
    "redirected_to",
}


def normalize_node_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(metadata or {})
    source_payload = raw.get("source") if isinstance(raw.get("source"), dict) else {}
    document_payload = raw.get("document") if isinstance(raw.get("document"), dict) else {}

    source_kind = (
        source_payload.get("kind")
        or raw.get("source_kind")
        or (raw.get("source") if isinstance(raw.get("source"), str) else None)
    )
    source_ref = (
        source_payload.get("ref")
        or raw.get("source_ref")
        or raw.get("source_locator")
        or raw.get("source_uri")
        or raw.get("source_path")
        or raw.get("uri")
        or raw.get("file_path")
    )
    source_hash = source_payload.get("hash") or raw.get("source_hash")
    mime_type = source_payload.get("mime_type") or raw.get("mime_type") or raw.get("media_type")
    suffix = source_payload.get("suffix") or raw.get("suffix")
    file_name = source_payload.get("file_name") or raw.get("file_name")
    modified_at = source_payload.get("modified_at") or raw.get("modified_at")

    if source_kind or source_ref or source_hash or mime_type or suffix or file_name or modified_at:
        source_payload = {
            **source_payload,
            **({"kind": source_kind} if source_kind else {}),
            **({"ref": source_ref} if source_ref else {}),
            **({"hash": source_hash} if source_hash else {}),
            **({"mime_type": mime_type} if mime_type else {}),
            **({"suffix": suffix} if suffix else {}),
            **({"file_name": file_name} if file_name else {}),
            **({"modified_at": modified_at} if modified_at else {}),
        }

    document_token_estimate = document_payload.get("token_estimate")
    document_content_length = document_payload.get("content_length") or raw.get("content_length")
    if (
        document_token_estimate is None
        and raw.get("token_estimate") is not None
        and raw.get("chunk_index") is None
    ):
        document_token_estimate = raw.get("token_estimate")

    if (
        document_token_estimate is not None
        or document_content_length is not None
    ):
        document_payload = {
            **document_payload,
            **(
                {"token_estimate": document_token_estimate}
                if document_token_estimate is not None
                else {}
            ),
            **(
                {"content_length": document_content_length}
                if document_content_length is not None
                else {}
            ),
        }

    result = {
        key: value
        for key, value in raw.items()
        if key not in LEGACY_NODE_FIELDS and key not in {"document", "chunk"}
    }
    if source_payload:
        result["source"] = source_payload
    if document_payload:
        result["document"] = document_payload
    return result


def normalize_edge_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(metadata or {})
    provenance = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    redirect = raw.get("redirect") if isinstance(raw.get("redirect"), dict) else {}

    creation_mode = provenance.get("creation_mode") or raw.get("creation_mode")
    created_by = provenance.get("created_by") or raw.get("created_by")
    reason = provenance.get("reason") or raw.get("reason")
    redirected_from = redirect.get("from") or raw.get("redirected_from")
    redirected_to = redirect.get("to") or raw.get("redirected_to")

    result = {
        key: value
        for key, value in raw.items()
        if key not in LEGACY_EDGE_FIELDS and key not in {"provenance", "redirect"}
    }
    if creation_mode or created_by or reason:
        result["provenance"] = {
            **({"creation_mode": creation_mode} if creation_mode else {}),
            **({"created_by": created_by} if created_by else {}),
            **({"reason": reason} if reason else {}),
        }
    if redirected_from or redirected_to:
        result["redirect"] = {
            **({"from": redirected_from} if redirected_from else {}),
            **({"to": redirected_to} if redirected_to else {}),
        }
    return result


def metadata_source_ref(metadata: dict[str, Any]) -> str | None:
    normalized = normalize_node_metadata(metadata)
    source = normalized.get("source") or {}
    return source.get("ref")


def metadata_source_kind(metadata: dict[str, Any]) -> str | None:
    normalized = normalize_node_metadata(metadata)
    source = normalized.get("source") or {}
    return source.get("kind")

def metadata_profile_kind(metadata: dict[str, Any]) -> str | None:
    normalized = normalize_node_metadata(metadata)
    profile = normalized.get("profile")
    if isinstance(profile, dict):
        kind = profile.get("kind")
        return str(kind) if kind else None
    return None
