from __future__ import annotations

import json
import logging
import mimetypes
import warnings
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

try:
    import trafilatura
except ImportError:  # pragma: no cover - dependency is optional during transition
    trafilatura = None

from cognitiveos.exceptions import UnsupportedExtractorError
from cognitiveos.extractors.base import ContentExtractor
from cognitiveos.models import ExtractedContent


logger = logging.getLogger(__name__)


class DefaultContentExtractor(ContentExtractor):
    TEXT_SUFFIXES = {
        ".md",
        ".txt",
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".sql",
        ".toml",
        ".ini",
        ".cfg",
    }
    HTML_SUFFIXES = {".html", ".htm"}
    FEED_MEDIA_TYPES = {"application/rss+xml", "application/atom+xml"}
    DOCUMENT_MEDIA_TYPES = {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
        "application/xml",
        "text/xml",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint",
    }
    VIDEO_HOST_KEYWORDS = ("youtube.com", "youtu.be", "bilibili.com", "vimeo.com")
    TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "latin-1")

    def extract(self, uri: str) -> ExtractedContent:
        parsed = urlparse(uri)
        if parsed.scheme in {"http", "https"}:
            return self._extract_http(uri)
        if parsed.scheme == "file":
            return self._extract_local(Path(parsed.path))
        return self._extract_local(Path(uri))

    def _extract_local(self, path: Path) -> ExtractedContent:
        if not path.exists() or not path.is_file():
            raise UnsupportedExtractorError(f"Unsupported or missing local file: {path}")

        suffix = path.suffix.lower()
        if suffix in self.TEXT_SUFFIXES:
            content = self._read_text_file(path)
            capture_method = "file_text"
        elif suffix in self.HTML_SUFFIXES:
            content, capture_method = self._extract_html_content(self._read_text_file(path))
            capture_method = f"file_{capture_method}"
        else:
            raise UnsupportedExtractorError(f"No extractor registered for file type '{suffix}'.")

        browser_capture = self._load_browser_capture_manifest(path)
        if browser_capture is not None:
            return ExtractedContent(
                name=browser_capture.get("title") or path.stem,
                content=content,
                metadata={
                    "source": "remote_uri",
                    "source_kind": browser_capture.get("source_kind", "remote_page"),
                    "uri": browser_capture.get("final_url") or browser_capture.get("requested_url"),
                    "requested_uri": browser_capture.get("requested_url")
                    or browser_capture.get("final_url"),
                    "file_name": browser_capture.get("file_name") or path.name,
                    "title": browser_capture.get("title") or path.stem,
                    "modified_at": browser_capture.get("captured_at")
                    or datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
                    "media_type": browser_capture.get("media_type")
                    or mimetypes.guess_type(str(path))[0]
                    or ("text/html" if suffix in self.HTML_SUFFIXES else "text/markdown"),
                    "content_length": len(path.read_bytes()),
                    "http_status": browser_capture.get("http_status"),
                    "etag": browser_capture.get("etag"),
                    "capture_method": browser_capture.get("capture_method")
                    or f"browser_export_{suffix.lstrip('.') or 'text'}",
                    "browser_capture": {
                        "local_path": str(path.resolve()),
                        **(
                            {"exported_from": browser_capture.get("exported_from")}
                            if browser_capture.get("exported_from")
                            else {}
                        ),
                    },
                },
            )

        return ExtractedContent(
            name=path.stem,
            content=content,
            metadata={
                "source": "local_file",
                "file_path": str(path.resolve()),
                "file_name": path.name,
                "modified_at": datetime.fromtimestamp(
                    path.stat().st_mtime,
                    tz=UTC,
                ).isoformat(),
                "suffix": suffix,
                "content_length": len(content),
                "capture_method": capture_method,
            },
        )

    def _extract_http(self, uri: str) -> ExtractedContent:
        try:
            response = httpx.get(uri, timeout=15.0, follow_redirects=True)
            response.raise_for_status()
        except Exception:
            logger.exception("remote_extract_failed uri=%s", uri)
            raise
        media_type = response.headers.get("content-type", "").split(";")[0].lower()
        final_uri = str(response.url)
        file_name = Path(urlparse(final_uri).path).name or final_uri
        title = file_name
        snapshot_format = "markdown"
        snapshot_suffix = ".md"
        snapshot_bytes: bytes | None = None
        capture_method = "httpx_text"
        source_kind = "remote_document"

        if media_type in {"text/html", "application/xhtml+xml"}:
            html = response.text
            title = self._extract_html_title(html, default=file_name)
            extracted_content, extraction_method = self._extract_html_content(html)
            capture_method = f"httpx_{extraction_method}"
            source_kind = (
                "remote_video" if self._looks_like_video_resource(final_uri, media_type) else "remote_page"
            )
            if source_kind == "remote_video":
                content = self._build_remote_note(
                    kind="video",
                    title=title,
                    final_uri=final_uri,
                    media_type=media_type or "text/html",
                )
            else:
                content = extracted_content
        elif media_type in self.FEED_MEDIA_TYPES:
            xml_text = response.text
            title = self._extract_xml_title(xml_text, default=file_name)
            content = self._xml_to_text(xml_text)
            source_kind = "remote_feed_item"
        elif self._looks_like_document_resource(final_uri, media_type):
            source_kind = "remote_document"
            title = file_name
            if self._is_binary_media_type(media_type):
                content = self._build_remote_note(
                    kind="document",
                    title=title,
                    final_uri=final_uri,
                    media_type=media_type or "application/octet-stream",
                )
                snapshot_format = "binary"
                snapshot_suffix = self._guess_snapshot_suffix(final_uri, media_type)
                snapshot_bytes = response.content
                capture_method = "httpx_binary"
            else:
                content = response.text
        elif self._looks_like_video_resource(final_uri, media_type):
            source_kind = "remote_video"
            title = file_name
            content = self._build_remote_note(
                kind="video",
                title=title,
                final_uri=final_uri,
                media_type=media_type or "application/octet-stream",
            )
            if self._is_binary_media_type(media_type):
                snapshot_format = "binary"
                snapshot_suffix = self._guess_snapshot_suffix(final_uri, media_type)
                snapshot_bytes = response.content
                capture_method = "httpx_binary"
        elif self._is_binary_media_type(media_type):
            source_kind = "remote_binary"
            title = file_name
            content = self._build_remote_note(
                kind="binary",
                title=title,
                final_uri=final_uri,
                media_type=media_type or "application/octet-stream",
            )
            snapshot_format = "binary"
            snapshot_suffix = self._guess_snapshot_suffix(final_uri, media_type)
            snapshot_bytes = response.content
            capture_method = "httpx_binary"
        else:
            content = response.text

        return ExtractedContent(
            name=title,
            content=content,
            metadata={
                "source": "remote_uri",
                "source_kind": source_kind,
                "uri": final_uri,
                "requested_uri": uri,
                "file_name": file_name,
                "title": title,
                "modified_at": response.headers.get("last-modified"),
                "media_type": media_type or "text/plain",
                "content_length": len(response.content),
                "http_status": response.status_code,
                "etag": response.headers.get("etag"),
                "capture_method": capture_method,
                "snapshot_format": snapshot_format,
                "snapshot_suffix": snapshot_suffix,
                **({"snapshot_bytes": snapshot_bytes} if snapshot_bytes is not None else {}),
            },
        )

    def _read_text_file(self, path: Path) -> str:
        for encoding in self.TEXT_ENCODINGS:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")

    def _extract_html_content(self, payload: str) -> tuple[str, str]:
        extracted = self._trafilatura_to_markdown(payload)
        if extracted:
            return extracted, "trafilatura"
        return self._html_to_text(payload), "bs4_fallback"

    @staticmethod
    def _trafilatura_to_markdown(payload: str) -> str | None:
        if trafilatura is None:
            return None
        try:
            result = trafilatura.extract(payload, output_format="markdown")
        except Exception:
            logger.exception("trafilatura_extract_failed")
            return None
        normalized = (result or "").strip()
        return normalized or None

    @staticmethod
    def _html_to_text(payload: str) -> str:
        soup = BeautifulSoup(payload, "html.parser")
        return soup.get_text("\n", strip=True)

    @staticmethod
    def _xml_to_text(payload: str) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(payload, "html.parser")
        return soup.get_text("\n", strip=True)

    @staticmethod
    def _extract_html_title(payload: str, *, default: str) -> str:
        soup = BeautifulSoup(payload, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        if title:
            return title[:200]
        heading = soup.find(["h1", "h2"])
        if heading is not None:
            candidate = heading.get_text(" ", strip=True)
            if candidate:
                return candidate[:200]
        return default

    @staticmethod
    def _extract_xml_title(payload: str, *, default: str) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(payload, "html.parser")
        title = soup.find("title")
        if title is not None:
            candidate = title.get_text(" ", strip=True)
            if candidate:
                return candidate[:200]
        return default

    def _looks_like_document_resource(self, uri: str, media_type: str) -> bool:
        if media_type in self.DOCUMENT_MEDIA_TYPES:
            return True
        suffix = Path(urlparse(uri).path).suffix.lower()
        return suffix in {
            ".pdf",
            ".txt",
            ".md",
            ".csv",
            ".json",
            ".xml",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
        }

    def _looks_like_video_resource(self, uri: str, media_type: str) -> bool:
        host = urlparse(uri).netloc.lower()
        if any(keyword in host for keyword in self.VIDEO_HOST_KEYWORDS):
            return True
        return media_type.startswith("video/")

    @staticmethod
    def _is_binary_media_type(media_type: str) -> bool:
        if not media_type:
            return False
        if media_type.startswith("text/"):
            return False
        return media_type not in {"application/json", "application/xml", "text/xml"}

    @staticmethod
    def _guess_snapshot_suffix(uri: str, media_type: str) -> str:
        suffix = Path(urlparse(uri).path).suffix.lower()
        if suffix:
            return suffix
        mapping = {
            "application/pdf": ".pdf",
            "video/mp4": ".mp4",
            "image/jpeg": ".jpg",
            "image/png": ".png",
        }
        return mapping.get(media_type, ".bin")

    @staticmethod
    def _build_remote_note(
        *,
        kind: str,
        title: str,
        final_uri: str,
        media_type: str,
    ) -> str:
        return (
            f"Remote {kind} source: {title}\n"
            f"URL: {final_uri}\n"
            f"Media type: {media_type}"
        )

    @staticmethod
    def _load_browser_capture_manifest(path: Path) -> dict[str, str | int] | None:
        manifest_path = path.with_name(f"{path.stem}.cognitiveos-source.json")
        if not manifest_path.exists():
            return None
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("browser_capture_manifest_invalid path=%s", manifest_path)
            return None
        if not isinstance(raw, dict):
            return None
        final_url = raw.get("final_url") or raw.get("ref")
        requested_url = raw.get("requested_url") or raw.get("requested_ref") or final_url
        if not final_url and not requested_url:
            return None
        return {
            "source_kind": str(raw.get("source_kind") or "remote_page"),
            "requested_url": str(requested_url) if requested_url else "",
            "final_url": str(final_url or requested_url),
            "title": str(raw.get("title")) if raw.get("title") else "",
            "media_type": str(raw.get("media_type")) if raw.get("media_type") else "",
            "capture_method": str(raw.get("capture_method"))
            if raw.get("capture_method")
            else "",
            "captured_at": str(raw.get("captured_at")) if raw.get("captured_at") else "",
            "etag": str(raw.get("etag")) if raw.get("etag") else "",
            "file_name": str(raw.get("file_name")) if raw.get("file_name") else "",
            "exported_from": str(raw.get("exported_from")) if raw.get("exported_from") else "",
            **(
                {"http_status": int(raw.get("http_status"))}
                if raw.get("http_status") is not None
                else {}
            ),
        }
