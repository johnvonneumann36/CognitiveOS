from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cognitiveos.exceptions import InvalidPayloadError

REPOSITORY_MARKERS = {
    ".git",
    "pyproject.toml",
    "package.json",
    "cargo.toml",
    "go.mod",
    "pom.xml",
    "requirements.txt",
    "setup.py",
}
REPOSITORY_HINTS = {
    "pyproject.toml": "Python project",
    "requirements.txt": "Python project",
    "setup.py": "Python project",
    "package.json": "JavaScript project",
    "cargo.toml": "Rust project",
    "go.mod": "Go project",
    "pom.xml": "Java project",
}
MEDIA_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".heic",
    ".tif",
    ".tiff",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
}
DOCUMENT_SUFFIXES = {
    ".md",
    ".txt",
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".rst",
    ".html",
    ".htm",
}
SAMPLE_ENTRY_LIMIT = 12
FILE_TYPE_LIMIT = 12


@dataclass(slots=True)
class FolderCollectionProfile:
    normalized_path: str
    display_name: str
    collection_class: str
    top_level_entry_count: int
    sample_entries: list[str]
    file_type_counts: dict[str, int]
    important_markers: list[str]
    content: str
    metadata: dict[str, Any]


def inspect_folder_collection(
    payload: str,
    *,
    name: str | None = None,
) -> FolderCollectionProfile:
    folder_path = Path(payload).expanduser()
    if not folder_path.exists():
        raise InvalidPayloadError(f"Folder does not exist: {payload}")
    if not folder_path.is_dir():
        raise InvalidPayloadError(f"Folder payload must point to a directory: {payload}")

    normalized_path = str(folder_path.resolve())
    entries = sorted(folder_path.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))
    top_level_entry_count = len(entries)
    sample_entries = [entry.name for entry in entries[:SAMPLE_ENTRY_LIMIT]]
    file_type_counts = _build_file_type_counts(entries)
    important_markers = _important_markers(entries)
    collection_class = _classify_collection(entries, important_markers)
    display_name = (name or folder_path.name or folder_path.resolve().name).strip()

    metadata = {
        "source": {
            "kind": "local_folder",
            "ref": normalized_path,
            "file_name": folder_path.name,
        },
        "collection": {
            "class": collection_class,
            "scan_mode": "root_only",
            "scanned_depth": 1,
            "child_anchors": [],
            "top_level_entry_count": top_level_entry_count,
            "sample_entries": sample_entries,
            "file_type_counts": file_type_counts,
            "important_markers": important_markers,
        },
    }

    return FolderCollectionProfile(
        normalized_path=normalized_path,
        display_name=display_name,
        collection_class=collection_class,
        top_level_entry_count=top_level_entry_count,
        sample_entries=sample_entries,
        file_type_counts=file_type_counts,
        important_markers=important_markers,
        content=_build_collection_content(
            normalized_path=normalized_path,
            collection_class=collection_class,
            top_level_entry_count=top_level_entry_count,
            sample_entries=sample_entries,
            file_type_counts=file_type_counts,
            important_markers=important_markers,
        ),
        metadata=metadata,
    )


def _classify_collection(entries: list[Path], important_markers: list[str]) -> str:
    lowered_markers = {marker.lower() for marker in important_markers}
    if lowered_markers & REPOSITORY_MARKERS:
        return "repository"

    media_count = 0
    document_count = 0
    other_count = 0
    for entry in entries:
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix in MEDIA_SUFFIXES:
            media_count += 1
        elif suffix in DOCUMENT_SUFFIXES:
            document_count += 1
        else:
            other_count += 1

    total_files = media_count + document_count + other_count
    if total_files == 0:
        return "workspace_bundle"
    if media_count >= max(document_count, other_count) and media_count * 2 >= total_files:
        return "media_collection"
    if document_count >= max(media_count, other_count) and document_count * 2 >= total_files:
        return "document_collection"
    return "workspace_bundle"


def _build_file_type_counts(entries: list[Path]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        if entry.is_dir():
            key = "dir"
        else:
            suffix = entry.suffix.lower()
            key = suffix[1:] if suffix else "no_extension"
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {key: value for key, value in ordered[:FILE_TYPE_LIMIT]}


def _important_markers(entries: list[Path]) -> list[str]:
    names = {entry.name.lower(): entry.name for entry in entries}
    ordered = [
        names[marker]
        for marker in sorted(REPOSITORY_MARKERS)
        if marker in names
    ]
    if "readme.md" in names and names["readme.md"] not in ordered:
        ordered.append(names["readme.md"])
    if "readme.txt" in names and names["readme.txt"] not in ordered:
        ordered.append(names["readme.txt"])
    return ordered[:SAMPLE_ENTRY_LIMIT]


def _build_collection_content(
    *,
    normalized_path: str,
    collection_class: str,
    top_level_entry_count: int,
    sample_entries: list[str],
    file_type_counts: dict[str, int],
    important_markers: list[str],
) -> str:
    lines = [
        f"Folder root: {normalized_path}",
        f"Collection class: {collection_class}",
        "Scan mode: root_only",
        f"Top-level entries: {top_level_entry_count}",
    ]
    if important_markers:
        lines.append(f"Important markers: {', '.join(important_markers)}")
    if file_type_counts:
        type_summary = ", ".join(
            f"{file_type}={count}" for file_type, count in file_type_counts.items()
        )
        lines.append(f"File type counts: {type_summary}")
    if sample_entries:
        lines.append(f"Sample entries: {', '.join(sample_entries)}")
    return "\n".join(lines)


def collection_hint_text(metadata: dict[str, Any], *, fallback_content: str = "") -> str:
    collection = metadata.get("collection", {}) if isinstance(metadata, dict) else {}
    parts: list[str] = []

    collection_class = collection.get("class")
    if isinstance(collection_class, str) and collection_class:
        parts.append(f"Class: {collection_class}")

    important_markers = [
        marker
        for marker in collection.get("important_markers", [])
        if isinstance(marker, str) and marker.strip()
    ]
    if important_markers:
        parts.append(f"Important markers: {', '.join(important_markers)}")

    sample_entries = [
        entry
        for entry in collection.get("sample_entries", [])
        if isinstance(entry, str) and entry.strip()
    ]
    if sample_entries:
        parts.append(f"Sample entries: {', '.join(sample_entries[:6])}")

    file_type_counts = collection.get("file_type_counts", {})
    if isinstance(file_type_counts, dict) and file_type_counts:
        ordered = sorted(
            (
                (str(file_type), int(count))
                for file_type, count in file_type_counts.items()
                if isinstance(file_type, str) and isinstance(count, int | float)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if ordered:
            parts.append(
                "File types: "
                + ", ".join(f"{file_type}={count}" for file_type, count in ordered[:6])
            )

    if not parts and fallback_content:
        return fallback_content
    return "\n".join(parts)


def repository_hint(important_markers: list[str]) -> str | None:
    for marker in important_markers:
        hint = REPOSITORY_HINTS.get(marker.lower())
        if hint:
            return hint
    return None
