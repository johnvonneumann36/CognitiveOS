from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from cognitiveos.config import AppSettings
from cognitiveos.db.connection import open_connection
from cognitiveos.db.repository import SQLiteRepository
from cognitiveos.exceptions import InvalidPayloadError
from cognitiveos.models import AddPayloadType, NodeRecord
from cognitiveos.service import CognitiveOSService


class FakeEmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            embeddings.append(
                [
                    1.0 if "graph" in lowered else 0.0,
                    1.0 if "memory" in lowered else 0.0,
                    1.0 if "profile" in lowered or "preference" in lowered else 0.0,
                    1.0 if "tooling" in lowered or "runtime" in lowered else 0.0,
                ]
            )
        return embeddings


class FakeChatProvider:
    def summarize(self, _content: str) -> str:
        return "synthetic summary"

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        if '"tags"' in system_prompt:
            return '{"tags":["graph","memory","runtime"]}'
        return "synthetic document description"


class BrokenChatProvider:
    def summarize(self, _content: str) -> str:
        raise RuntimeError("chat backend offline")

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("chat backend offline")


def build_service(
    tmp_path: Path,
    *,
    embedding_provider: FakeEmbeddingProvider | None = None,
    chat_provider: FakeChatProvider | None = None,
    dream_event_threshold: int = 50,
    dream_max_age_hours: int = 24,
    dream_age_min_event_count: int = 5,
) -> tuple[CognitiveOSService, AppSettings]:
    settings = AppSettings.from_env(
        db_path=tmp_path / "cognitiveos.db",
        memory_output_path=tmp_path / "MEMORY.MD",
        project_root=tmp_path,
    )
    settings.bootstrap_dir = tmp_path / ".cognitiveos" / "bootstrap"
    settings.background_log_dir = tmp_path / ".cognitiveos" / "logs"
    settings.snapshot_dir = tmp_path / ".cognitiveos" / "snapshots"
    settings.dream_event_threshold = dream_event_threshold
    settings.dream_max_age_hours = dream_max_age_hours
    settings.dream_age_min_event_count = dream_age_min_event_count
    service = CognitiveOSService(
        settings=settings,
        repository=SQLiteRepository(settings.db_path),
        embedding_provider=embedding_provider,
        chat_provider=chat_provider,
    )
    service.initialize()
    return service, settings


def make_http_response(
    *,
    url: str,
    content_type: str,
    text: str,
    last_modified: str | None = None,
) -> httpx.Response:
    headers = {"content-type": content_type}
    if last_modified is not None:
        headers["last-modified"] = last_modified
    return httpx.Response(
        200,
        headers=headers,
        text=text,
        request=httpx.Request("GET", url),
    )


def make_http_binary_response(
    *,
    url: str,
    content_type: str,
    content: bytes,
    last_modified: str | None = None,
    etag: str | None = None,
) -> httpx.Response:
    headers = {"content-type": content_type}
    if last_modified is not None:
        headers["last-modified"] = last_modified
    if etag is not None:
        headers["etag"] = etag
    return httpx.Response(
        200,
        headers=headers,
        content=content,
        request=httpx.Request("GET", url),
    )


def create_system_profile_node(
    service: CognitiveOSService,
    *,
    content: str,
    name: str,
    durability: str = "durable",
) -> str:
    node = NodeRecord(
        id=f"node_profile_{name.lower().replace(' ', '_')}",
        name=name,
        description=service._summarize(content),
        content=content,
        embedding=service._embed_content(content),
        tags=["profile"],
        metadata={"profile": {"kind": "system"}},
        node_type="memory",
        durability=durability,
    )
    service.repository.create_node(
        node,
        actor=service.settings.default_actor,
        action_type="create",
    )
    return node.id


def create_repository_folder(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".git").mkdir()
    (path / "src").mkdir()
    (path / "tests").mkdir()
    (path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (path / "README.md").write_text("# Demo\n", encoding="utf-8")
    return path


def test_service_lifecycle(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path)

    profile_node_id = create_system_profile_node(
        service,
        content="User prefers concise technical answers.",
        name="Communication Preference",
    )
    project = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="CognitiveOS packages a graph memory runtime with CLI and MCP tooling.",
        tags=["tech"],
        name="Project Overview",
    )

    assert project.node_id is not None

    link_receipt = service.link_nodes(
        src_id=profile_node_id,
        dst_id=project.node_id,
        relation="informs",
    )
    assert link_receipt.action_taken == "edge_created"

    results = service.search(keyword="CognitiveOS", top_k=5, include_neighbors=1)
    assert len(results) == 1
    assert results[0].id == project.node_id
    assert results[0].linked_nodes[0].id == profile_node_id

    update_receipt = service.update_node(
        node_id=project.node_id,
        content=(
            "CognitiveOS packages a graph memory runtime with installable CLI, "
            "MCP, and Docker support."
        ),
        tags=["tech", "runtime"],
    )
    assert update_receipt.audit_log_id is not None

    read_result = service.read_nodes([project.node_id], include_content=True)[project.node_id]
    assert "Docker support" in read_result.content
    assert len(read_result.edges) == 1
    assert read_result.tags == ["tech", "runtime"]

    rendered = service.compile_memory_snapshot()
    assert rendered == settings.memory_output_path
    assert "Communication Preference" in rendered.read_text(encoding="utf-8")


def test_add_node_deduplicates_tags_case_insensitively(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)

    receipt = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Tag annotated memory node.",
        tags=["tech", "memory", "Memory", "runtime", "tech", "host"],
        name="Tag Node",
    )
    assert receipt.node_id is not None

    node = service.repository.get_node(receipt.node_id)
    assert node.tags == ["tech", "memory", "runtime", "host"]

    results = service.search(keyword="runtime", top_k=5, include_neighbors=0)
    assert any(result.id == receipt.node_id for result in results)


def test_search_neighbor_expansion_is_cycle_safe_and_batched(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)

    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Alpha graph memory root.",
        tags=["graph"],
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Beta graph memory node.",
        tags=["graph"],
        name="Beta",
    )
    gamma = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Gamma graph memory node.",
        tags=["graph"],
        name="Gamma",
    )
    delta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Delta graph memory node.",
        tags=["graph"],
        name="Delta",
    )

    assert alpha.node_id and beta.node_id and gamma.node_id and delta.node_id

    service.link_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="supports")
    service.link_nodes(src_id=beta.node_id, dst_id=gamma.node_id, relation="supports")
    service.link_nodes(src_id=gamma.node_id, dst_id=alpha.node_id, relation="supports")
    service.link_nodes(src_id=beta.node_id, dst_id=delta.node_id, relation="depends_on")

    results = service.search(keyword="Alpha", top_k=1, include_neighbors=3)

    assert len(results) == 1
    neighbor_hops = {node.id: node.hop for node in results[0].linked_nodes}
    assert neighbor_hops == {
        beta.node_id: 1,
        gamma.node_id: 1,
        delta.node_id: 2,
    }
    assert service.last_runtime_metrics["operation"] == "search"
    assert service.last_runtime_metrics["timings_ms"]["build_results"] >= 0


def test_add_node_async_refreshes_semantic_neighbor_cache(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path, embedding_provider=FakeEmbeddingProvider())

    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Alpha graph memory runtime.",
        tags=["graph"],
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Beta graph memory runtime.",
        tags=["graph"],
        force=True,
        name="Beta",
    )
    gamma = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Gamma tooling preference only.",
        tags=["tooling"],
        name="Gamma",
    )

    assert alpha.node_id and beta.node_id and gamma.node_id
    service._wait_for_background_tasks()

    neighbor_rows = service.repository.list_semantic_neighbors([beta.node_id], min_similarity=0.5)
    assert any(
        node_id == beta.node_id and neighbor_id == alpha.node_id
        for node_id, neighbor_id, _similarity in neighbor_rows
    )
    assert all(neighbor_id != gamma.node_id for _node_id, neighbor_id, _similarity in neighbor_rows)


def test_add_node_accepts_explicit_durability_override(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)

    receipt = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Pinned host instruction memory.",
        tags=["instruction"],
        durability="pinned",
        name="Pinned Instruction",
    )
    assert receipt.node_id is not None

    node = service.repository.get_node(receipt.node_id)
    assert node.durability == "pinned"


def test_update_node_can_change_durability(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)

    receipt = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Working memory to be promoted.",
        tags=["memory"],
        name="Promotable Node",
    )
    assert receipt.node_id is not None

    update_receipt = service.update_node(
        node_id=receipt.node_id,
        content="Working memory promoted to durable.",
        tags=["memory", "promoted"],
        durability="durable",
    )
    assert update_receipt.action_taken == "updated"

    node = service.repository.get_node(receipt.node_id)
    assert node.durability == "durable"
    assert node.tags == ["memory", "promoted"]


def test_add_folder_creates_durable_source_collection(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)
    folder = create_repository_folder(tmp_path / "CognitiveOS")

    receipt = service.add_node(
        payload_type=AddPayloadType.FOLDER,
        payload=str(folder),
        name="CognitiveOS Repository",
        tags=["repository", "memory"],
    )

    assert receipt.status == "success"
    assert receipt.node_id is not None
    node = service.repository.get_node(receipt.node_id)
    assert node.node_type == "source_collection"
    assert node.durability == "durable"
    assert node.name == "CognitiveOS Repository"
    assert node.metadata["source"]["kind"] == "local_folder"
    assert node.metadata["source"]["ref"] == str(folder.resolve())
    assert node.metadata["collection"]["class"] == "repository"
    assert node.metadata["collection"]["scan_mode"] == "root_only"
    assert node.metadata["collection"]["scanned_depth"] == 1
    assert node.metadata["collection"]["child_anchors"] == []
    assert "Folder root:" in node.content
    assert "pyproject.toml" in node.content


def test_add_folder_detects_media_collection(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)
    folder = tmp_path / "Japan Trip 2025"
    folder.mkdir()
    for name in ["IMG_0001.JPG", "IMG_0002.JPG", "clip.mp4", "portrait.png"]:
        (folder / name).write_text("media", encoding="utf-8")
    (folder / "notes.txt").write_text("small note", encoding="utf-8")

    receipt = service.add_node(
        payload_type=AddPayloadType.FOLDER,
        payload=str(folder),
        name="Japan Trip 2025 Photos",
    )

    node = service.repository.get_node(receipt.node_id)
    assert node.metadata["collection"]["class"] == "media_collection"
    assert "media collection" in node.description.lower()


def test_add_folder_falls_back_to_workspace_bundle(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)
    folder = tmp_path / "Workspace"
    folder.mkdir()
    (folder / "docs").mkdir()
    (folder / "script.ps1").write_text("Write-Host hi", encoding="utf-8")
    (folder / "diagram.drawio").write_text("diagram", encoding="utf-8")
    (folder / "archive.zip").write_text("archive", encoding="utf-8")

    receipt = service.add_node(
        payload_type=AddPayloadType.FOLDER,
        payload=str(folder),
    )

    node = service.repository.get_node(receipt.node_id)
    assert node.metadata["collection"]["class"] == "workspace_bundle"


def test_add_remote_url_creates_snapshot_markdown_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/article"
    html = (
        "<html><body><main><h1>Example Article</h1>"
        "<p>Remote body text for preservation.</p></main></body></html>"
    )

    def fake_get(url: str, **_kwargs: object) -> httpx.Response:
        assert url == remote_url
        return make_http_response(
            url="https://example.com/final-article",
            content_type="text/html; charset=utf-8",
            text=html,
            last_modified="Wed, 08 Apr 2026 10:00:00 GMT",
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(
        payload_type=AddPayloadType.FILE,
        payload=remote_url,
        tags=["article"],
        name="Example Article",
    )

    assert receipt.status == "success"
    assert receipt.action_taken == "created"
    node = service.repository.get_node(receipt.node_id)
    assert node.node_type == "source_document"
    assert node.metadata["source"]["kind"] == "remote_page"
    assert node.metadata["source"]["ref"] == "https://example.com/final-article"
    assert node.metadata["source"]["requested_ref"] == remote_url
    assert node.metadata["source"]["mime_type"] == "text/html"
    assert node.metadata["source"]["title"] == "Example Article"
    assert node.metadata["source"]["http_status"] == 200
    assert node.metadata["source"].get("etag") is None
    assert node.metadata["source"]["content_length"] == len(html.encode("utf-8"))
    assert node.metadata["snapshot"]["format"] == "markdown"
    assert node.metadata["snapshot"]["capture_method"] in {
        "httpx_trafilatura",
        "httpx_bs4_fallback",
    }
    snapshot_path = Path(node.metadata["snapshot"]["path"])
    assert snapshot_path == settings.snapshot_dir / f"{node.id}.md"
    assert snapshot_path.exists()
    snapshot_text = snapshot_path.read_text(encoding="utf-8")
    assert "requested_url: https://example.com/article" in snapshot_text
    assert "final_url: https://example.com/final-article" in snapshot_text
    assert "title: Example Article" in snapshot_text
    assert "http_status: 200" in snapshot_text
    assert "Remote body text for preservation." in snapshot_text
    assert node.content.startswith("Remote source snapshot preserved.")


def test_add_remote_url_twice_blocks_without_force(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/article"
    current_html = {
        "value": "<html><body><main><p>First remote version.</p></main></body></html>"
    }

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url="https://example.com/article",
            content_type="text/html; charset=utf-8",
            text=current_html["value"],
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    first = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    current_html["value"] = "<html><body><main><p>Second remote version.</p></main></body></html>"
    second = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)

    assert first.node_id is not None
    assert second.status == "blocked"
    assert second.node_id is None
    assert second.conflicting_nodes[0].id == first.node_id
    read_result = service.read_nodes([first.node_id], include_content=True)[first.node_id]
    assert "First remote version." in read_result.content


def test_add_remote_url_with_force_refreshes_existing_node_in_place(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/article"
    current_html = {
        "value": "<html><body><main><p>First remote version.</p></main></body></html>"
    }

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url="https://example.com/article",
            content_type="text/html; charset=utf-8",
            text=current_html["value"],
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    first = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    current_html["value"] = "<html><body><main><p>Second remote version.</p></main></body></html>"
    second = service.add_node(
        payload_type=AddPayloadType.FILE,
        payload=remote_url,
        force=True,
    )

    assert first.node_id is not None
    assert second.node_id == first.node_id
    assert second.action_taken == "source_document_refreshed"
    read_result = service.read_nodes([first.node_id], include_content=True)[first.node_id]
    assert "Second remote version." in read_result.content


def test_remote_urls_with_identical_content_create_distinct_nodes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    html = "<html><body><main><p>Shared remote content.</p></main></body></html>"

    def fake_get(url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=url,
            content_type="text/html; charset=utf-8",
            text=html,
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    first = service.add_node(payload_type=AddPayloadType.FILE, payload="https://a.example.com/doc")
    second = service.add_node(payload_type=AddPayloadType.FILE, payload="https://b.example.com/doc")

    assert first.node_id is not None
    assert second.node_id is not None
    assert first.node_id != second.node_id


def test_read_remote_source_document_returns_snapshot_body(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/guide"

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=remote_url,
            content_type="text/html; charset=utf-8",
            text="<html><body><main><p>Guide body from snapshot.</p></main></body></html>",
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    result = service.read_nodes([receipt.node_id], include_content=True)[receipt.node_id]

    assert "Guide body from snapshot." in result.content
    assert result.content.startswith("---\nrequested_url:")


def test_remote_snapshot_hash_mismatch_falls_back_to_stored_note(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/hash"

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=remote_url,
            content_type="text/html; charset=utf-8",
            text="<html><body><main><p>Original body.</p></main></body></html>",
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    node = service.repository.get_node(receipt.node_id)
    snapshot_path = Path(node.metadata["snapshot"]["path"])
    snapshot_path.write_text("---\nrequested_url: x\n---\n\nTampered body.\n", encoding="utf-8")

    result = service.read_nodes([receipt.node_id], include_content=True)[receipt.node_id]

    assert result.content.startswith("Remote source snapshot preserved.")
    assert any("content hash mismatch" in notice for notice in result.notices)


def test_missing_remote_snapshot_falls_back_to_stored_note(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/missing"

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=remote_url,
            content_type="text/html; charset=utf-8",
            text="<html><body><main><p>Preserved body.</p></main></body></html>",
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    node = service.repository.get_node(receipt.node_id)
    Path(node.metadata["snapshot"]["path"]).unlink()

    result = service.read_nodes([receipt.node_id], include_content=True)[receipt.node_id]

    assert result.content.startswith("Remote source snapshot preserved.")
    assert any("Preserved snapshot file is unavailable" in notice for notice in result.notices)


def test_remote_keyword_search_does_not_index_preserved_body_only_terms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/search"

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=remote_url,
            content_type="text/html; charset=utf-8",
            text=(
                "<html><body><main><p>HyperUniqueBodyToken appears only in the body.</p>"
                "</main></body></html>"
            ),
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)
    service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)

    results = service.search(keyword="HyperUniqueBodyToken", top_k=5, include_neighbors=0)
    assert results == []


def test_remote_refresh_source_document_updates_snapshot_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/refresh"
    current_html = {"value": "<html><body><main><p>Version one.</p></main></body></html>"}

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=remote_url,
            content_type="text/html; charset=utf-8",
            text=current_html["value"],
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    node = service.repository.get_node(receipt.node_id)
    snapshot_path = Path(node.metadata["snapshot"]["path"])
    first_snapshot = snapshot_path.read_text(encoding="utf-8")

    current_html["value"] = "<html><body><main><p>Version two.</p></main></body></html>"
    refresh_receipt = service.refresh_source_document(node_id=receipt.node_id)

    assert refresh_receipt.action_taken == "source_document_refreshed"
    second_snapshot = snapshot_path.read_text(encoding="utf-8")
    assert "Version one." in first_snapshot
    assert "Version two." in second_snapshot
    assert ".tmp" not in second_snapshot


def test_remote_pdf_creates_binary_snapshot_and_metadata_only_note(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/handbook.pdf"
    pdf_bytes = b"%PDF-1.4\nfake pdf payload"

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_binary_response(
            url=remote_url,
            content_type="application/pdf",
            content=pdf_bytes,
            last_modified="Wed, 08 Apr 2026 10:00:00 GMT",
            etag="W/\"pdf-v1\"",
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    node = service.repository.get_node(receipt.node_id)

    assert node.metadata["source"]["kind"] == "remote_document"
    assert node.metadata["source"]["mime_type"] == "application/pdf"
    assert node.metadata["source"]["etag"] == 'W/"pdf-v1"'
    assert node.metadata["snapshot"]["format"] == "binary"
    assert Path(node.metadata["snapshot"]["path"]) == settings.snapshot_dir / f"{node.id}.pdf"
    assert Path(node.metadata["snapshot"]["path"]).read_bytes() == pdf_bytes
    assert node.content.startswith("Remote source snapshot preserved.")

    read_result = service.read_nodes([receipt.node_id], include_content=True)[receipt.node_id]
    assert read_result.content.startswith("Remote source snapshot preserved.")
    assert any("Binary snapshot preserved" in notice for notice in read_result.notices)


def test_remote_video_page_is_classified_as_remote_video(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://www.youtube.com/watch?v=abc123"

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=remote_url,
            content_type="text/html; charset=utf-8",
            text="<html><head><title>Demo Video</title></head><body><main><p>video page</p></main></body></html>",
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    node = service.repository.get_node(receipt.node_id)

    assert node.metadata["source"]["kind"] == "remote_video"
    assert node.metadata["source"]["title"] == "Demo Video"
    assert node.metadata["snapshot"]["format"] == "markdown"


def test_remote_feed_is_classified_as_feed_item(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/feed.xml"

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=remote_url,
            content_type="application/rss+xml",
            text=(
                "<?xml version='1.0'?><rss><channel><title>Example Feed</title>"
                "<item><title>Item A</title></item></channel></rss>"
            ),
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    node = service.repository.get_node(receipt.node_id)

    assert node.metadata["source"]["kind"] == "remote_feed_item"
    assert node.metadata["source"]["title"] == "Example Feed"


def test_delete_remote_node_removes_snapshot_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    remote_url = "https://example.com/delete"

    def fake_get(_url: str, **_kwargs: object) -> httpx.Response:
        return make_http_response(
            url=remote_url,
            content_type="text/html; charset=utf-8",
            text="<html><body><main><p>Delete me.</p></main></body></html>",
        )

    monkeypatch.setattr("cognitiveos.extractors.defaults.httpx.get", fake_get)

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=remote_url)
    node = service.repository.get_node(receipt.node_id)
    snapshot_path = Path(node.metadata["snapshot"]["path"])
    assert snapshot_path.exists()

    delete_receipt = service.delete_node(node_id=receipt.node_id)

    assert delete_receipt.action_taken == "deleted"
    assert not snapshot_path.exists()


def test_local_html_uses_trafilatura_first_with_fallback(tmp_path: Path, monkeypatch) -> None:
    service, _settings = build_service(tmp_path)
    html_path = tmp_path / "page.html"
    html_path.write_text(
        "<html><body><main><p>Fallback html body.</p></main></body></html>",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "cognitiveos.extractors.defaults.DefaultContentExtractor._trafilatura_to_markdown",
        staticmethod(lambda _payload: None),
    )

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=str(html_path))
    node = service.repository.get_node(receipt.node_id)

    assert "Fallback html body." in node.content
    assert node.metadata["capture_method"] == "file_bs4_fallback"


def test_browser_exported_html_with_sidecar_manifest_is_ingested_as_remote_page(
    tmp_path: Path,
) -> None:
    service, settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    html_path = tmp_path / "article.html"
    html_path.write_text(
        "<html><head><title>Browser Export</title></head><body><main><p>Captured page body.</p></main></body></html>",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "article.cognitiveos-source.json"
    manifest_path.write_text(
        """{
  "source_kind": "remote_page",
  "requested_url": "https://example.com/login",
  "final_url": "https://example.com/article",
  "title": "Captured Article",
  "capture_method": "browser_export_html",
  "captured_at": "2026-04-09T10:00:00+08:00",
  "http_status": 200,
  "exported_from": "chrome-devtools"
}""",
        encoding="utf-8",
    )

    receipt = service.add_node(
        payload_type=AddPayloadType.FILE,
        payload=str(html_path),
        name="Captured Article",
    )
    node = service.repository.get_node(receipt.node_id)

    assert node.metadata["source"]["kind"] == "remote_page"
    assert node.metadata["source"]["requested_ref"] == "https://example.com/login"
    assert node.metadata["source"]["ref"] == "https://example.com/article"
    assert node.metadata["source"]["title"] == "Captured Article"
    assert node.metadata["source"]["http_status"] == 200
    assert node.metadata["snapshot"]["format"] == "markdown"
    assert node.metadata["snapshot"]["capture_method"] == "browser_export_html"
    snapshot_path = Path(node.metadata["snapshot"]["path"])
    assert snapshot_path == settings.snapshot_dir / f"{node.id}.md"
    assert "Captured page body." in snapshot_path.read_text(encoding="utf-8")


def test_browser_exported_markdown_with_sidecar_manifest_is_ingested_as_remote_page(
    tmp_path: Path,
) -> None:
    service, settings = build_service(tmp_path, chat_provider=FakeChatProvider())
    md_path = tmp_path / "article.md"
    md_path.write_text("# Captured Article\n\nReadable markdown body.\n", encoding="utf-8")
    manifest_path = tmp_path / "article.cognitiveos-source.json"
    manifest_path.write_text(
        """{
  "requested_url": "https://example.com/paywalled",
  "final_url": "https://example.com/captured",
  "title": "Readable Capture",
  "capture_method": "browser_export_markdown",
  "captured_at": "2026-04-09T10:30:00+08:00",
  "exported_from": "playwright"
}""",
        encoding="utf-8",
    )

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=str(md_path))
    node = service.repository.get_node(receipt.node_id)

    assert node.metadata["source"]["kind"] == "remote_page"
    assert node.metadata["source"]["requested_ref"] == "https://example.com/paywalled"
    assert node.metadata["source"]["ref"] == "https://example.com/captured"
    assert node.metadata["snapshot"]["capture_method"] == "browser_export_markdown"
    assert Path(node.metadata["snapshot"]["path"]) == settings.snapshot_dir / f"{node.id}.md"


def test_local_file_without_browser_manifest_stays_local_file(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)
    md_path = tmp_path / "notes.md"
    md_path.write_text("ordinary local note", encoding="utf-8")

    receipt = service.add_node(payload_type=AddPayloadType.FILE, payload=str(md_path))
    node = service.repository.get_node(receipt.node_id)

    assert node.metadata["source"]["kind"] == "local_file"


def test_add_folder_blocks_duplicate_source_ref_unless_forced(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)
    folder = create_repository_folder(tmp_path / "CognitiveOS")

    first = service.add_node(payload_type=AddPayloadType.FOLDER, payload=str(folder))
    second = service.add_node(payload_type=AddPayloadType.FOLDER, payload=str(folder))
    third = service.add_node(payload_type=AddPayloadType.FOLDER, payload=str(folder), force=True)

    assert first.status == "success"
    assert second.status == "blocked"
    assert "matching source metadata" in (second.reason or "").lower()
    assert third.status == "success"


def test_folder_search_read_and_update_keep_collection_root_semantics(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path, embedding_provider=FakeEmbeddingProvider())
    folder = create_repository_folder(tmp_path / "CognitiveOS")

    receipt = service.add_node(
        payload_type=AddPayloadType.FOLDER,
        payload=str(folder),
        name="CognitiveOS Repository",
        tags=["repository", "runtime"],
    )
    assert receipt.node_id is not None
    service._wait_for_background_tasks()

    results = service.search(keyword="pyproject", top_k=5, include_neighbors=0)
    assert len(results) == 1
    assert results[0].id == receipt.node_id
    assert results[0].node_type == "source_collection"

    read_result = service.read_nodes([receipt.node_id], include_content=True)[receipt.node_id]
    assert "Sample entries:" in (read_result.content or "")
    assert "src" in (read_result.content or "")

    original = service.repository.get_node(receipt.node_id)
    update_receipt = service.update_node(
        node_id=receipt.node_id,
        content="Repository root for CognitiveOS runtime and MCP tooling.",
        tags=["repository", "runtime", "python"],
    )
    assert update_receipt.action_taken == "updated"

    updated = service.repository.get_node(receipt.node_id)
    assert updated.metadata["collection"]["class"] == "repository"
    assert updated.metadata["collection"]["sample_entries"] == original.metadata["collection"]["sample_entries"]
    assert updated.tags == ["repository", "runtime", "python"]
    assert updated.description
    assert updated.embedding is not None

    memory_text = service.compile_memory_snapshot().read_text(encoding="utf-8")
    assert "## Durable Source Memory" in memory_text
    assert "CognitiveOS Repository" in memory_text


def test_update_node_delete_tag_removes_node_edges_and_vector(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path, embedding_provider=FakeEmbeddingProvider())

    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory alpha runtime.",
        tags=["graph"],
        force=True,
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory beta runtime.",
        tags=["graph"],
        force=True,
        name="Beta",
    )
    assert alpha.node_id and beta.node_id
    service._wait_for_background_tasks()
    service.link_nodes(src_id=alpha.node_id, dst_id=beta.node_id, relation="supports")

    assert service.repository.get_vector_count() == 2
    delete_receipt = service.update_node(
        node_id=alpha.node_id,
        content="",
        tags=["__delete__"],
    )
    assert delete_receipt.action_taken == "deleted"
    assert delete_receipt.edge is not None
    assert delete_receipt.edge["trigger"] == "update_tag"
    assert service.repository.get_node_count() == 1
    assert service.repository.get_vector_count() == 1
    assert service.repository.list_relationships(beta.node_id) == []

    search_results = service.search(keyword="Alpha", top_k=5, include_neighbors=0)
    assert all(result.id != alpha.node_id for result in search_results)


def test_add_node_rejects_invalid_durability(tmp_path: Path) -> None:
    service, _settings = build_service(tmp_path)

    try:
        service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload="Invalid durability node.",
            durability="forever",
            name="Invalid Durability",
        )
    except InvalidPayloadError as exc:
        assert "Invalid durability value" in str(exc)
    else:
        raise AssertionError("Expected add_node to reject invalid durability.")


def test_semantic_search_similarity_block_and_chat_summary(tmp_path: Path) -> None:
    service, _settings = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )

    long_payload = "Graph memory " * 80
    first_receipt = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload=long_payload,
        tags=["tech"],
        name="Semantic Node",
    )
    assert first_receipt.status == "success"
    assert first_receipt.node_id is not None
    stored_first = service.repository.get_node(first_receipt.node_id)
    assert stored_first.description == "synthetic summary"
    assert stored_first.embedding is not None

    blocked_receipt = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory platform",
        tags=["tech"],
        name="Duplicate Candidate",
    )
    assert blocked_receipt.status == "blocked"
    assert blocked_receipt.conflicting_nodes[0].id == first_receipt.node_id

    force_profile_id = create_system_profile_node(
        service,
        content="Profile preference memory",
        name="Profile Node",
    )
    assert force_profile_id.startswith("node_profile_")

    semantic_results = service.search(query="graph memory", top_k=2, include_neighbors=0)
    assert semantic_results[0].id == first_receipt.node_id
    assert semantic_results[0].semantic_score is not None


def test_hybrid_ranking_and_ops_report(tmp_path: Path) -> None:
    service, _settings = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )

    alpha = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime alpha",
        tags=["tech"],
        force=True,
        name="Alpha",
    )
    beta = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory beta",
        tags=["tech"],
        force=True,
        name="Beta",
    )
    gamma = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Keyword only document about CognitiveOS",
        tags=["docs"],
        force=True,
        name="Gamma",
    )
    assert alpha.node_id and beta.node_id and gamma.node_id

    hybrid_results = service.search(
        query="graph runtime",
        keyword="CognitiveOS",
        top_k=3,
        include_neighbors=0,
    )
    assert len(hybrid_results) == 3
    assert hybrid_results[0].score is not None
    assert (
        hybrid_results[0].semantic_score is not None
        or hybrid_results[0].keyword_score is not None
    )

    reindex = service.reindex_embeddings()
    assert reindex["status"] == "success"
    assert reindex["vector_count"] >= 3

    doctor = service.doctor(check_providers=True)
    assert doctor["sqlite_vec_version"]
    assert doctor["node_count"] >= 3
    assert doctor["vector_count"] >= 3
    assert doctor["provider_checks"]["embedding"]["status"] == "success"


def test_dream_creates_super_node_and_redirects_edges(tmp_path: Path) -> None:
    service, settings = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )

    profile_node_id = create_system_profile_node(
        service,
        content="User profile preference memory",
        name="Profile",
    )
    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime tooling",
        tags=["tech"],
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime",
        tags=["tech"],
        force=True,
        name="Node B",
    )
    outside = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="External integration boundary",
        tags=["integration"],
        force=True,
        name="Outside",
    )

    assert profile_node_id and a.node_id and b.node_id and outside.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="supports")
    service.link_nodes(src_id=a.node_id, dst_id=outside.node_id, relation="depends_on")
    service.search(keyword="Graph", top_k=5, include_neighbors=1)
    service.read_nodes([a.node_id, b.node_id], include_content=False)

    dream_result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.8,
    )
    assert dream_result.status == "success"
    assert dream_result.clusters_created >= 1

    super_node_id = dream_result.super_nodes[0].node_id
    super_node = service.repository.get_node(super_node_id)
    assert super_node.node_type == "super_node"
    assert super_node.description == "synthetic summary"

    read_super = service.read_nodes([super_node_id], include_content=True)[super_node_id]
    contains_ids = {edge.dst_id for edge in read_super.edges if edge.src_id == super_node_id}
    assert a.node_id in contains_ids
    assert b.node_id in contains_ids
    assert outside.node_id in contains_ids

    memory_text = settings.memory_output_path.read_text(encoding="utf-8")
    assert "Profile" in memory_text


def test_dream_without_chat_returns_pending_host_compaction(tmp_path: Path) -> None:
    service, settings = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=None,
    )

    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime tooling",
        tags=["tech"],
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime",
        tags=["tech"],
        force=True,
        name="Node B",
    )
    assert a.node_id and b.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="supports")
    service.search(keyword="Graph", top_k=5, include_neighbors=1)

    dream_result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.8,
    )

    assert dream_result.status == "awaiting_host_compaction"
    assert dream_result.clusters_created == 0
    assert len(dream_result.pending_compactions) == 1
    task = dream_result.pending_compactions[0]
    assert task.requested_backend == "host_agent"
    assert task.fallback_backend == "heuristic"
    assert "Chat provider is not configured" in (task.reason or "")
    assert len(service.list_dream_compactions()) == 1


def test_background_dream_with_chat_available_queues_and_completes(tmp_path: Path) -> None:
    service, settings = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=FakeChatProvider(),
    )

    def inline_background_dream(**kwargs: object) -> None:
        service.execute_dream_run(**kwargs)

    service._launch_background_dream = inline_background_dream  # type: ignore[method-assign]

    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime tooling",
        tags=["tech"],
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime",
        tags=["tech"],
        force=True,
        name="Node B",
    )
    assert a.node_id and b.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="supports")
    service.search(keyword="Graph", top_k=5, include_neighbors=1)

    result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.8,
        background=True,
    )
    assert result.status == "queued"
    assert result.run_id is not None

    runs = service.list_dream_runs(limit=5)
    assert runs[0].run_id == result.run_id
    assert runs[0].status == "success"
    assert runs[0].clusters_created >= 1


def test_dream_host_compaction_resolution_creates_super_node(tmp_path: Path) -> None:
    service, settings = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=None,
    )

    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime tooling",
        tags=["tech"],
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime",
        tags=["tech"],
        force=True,
        name="Node B",
    )
    outside = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="External integration boundary",
        tags=["integration"],
        force=True,
        name="Outside",
    )
    assert a.node_id and b.node_id and outside.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="supports")
    service.link_nodes(src_id=a.node_id, dst_id=outside.node_id, relation="depends_on")
    service.search(keyword="Graph", top_k=5, include_neighbors=1)

    dream_result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.8,
    )
    task = dream_result.pending_compactions[0]

    resolution = service.resolve_dream_compaction(
        task_id=task.task_id,
        title="Compressed Graph Runtime Cluster",
        description="Synthetic host-compressed cluster",
        content="Compressed content from host agent",
    )
    assert resolution.status == "success"
    assert resolution.resolution_backend == "host_agent"
    assert resolution.remaining_tasks == 0
    assert resolution.dream_completed is True
    assert resolution.node_id is not None

    super_node = service.repository.get_node(resolution.node_id)
    assert super_node.name == "Compressed Graph Runtime Cluster"
    assert super_node.description == "Synthetic host-compressed cluster"
    assert super_node.metadata["dream_compaction_backend"] == "host_agent"

    read_super = service.read_nodes([resolution.node_id], include_content=True)[resolution.node_id]
    contains_ids = {edge.dst_id for edge in read_super.edges if edge.src_id == resolution.node_id}
    assert a.node_id in contains_ids
    assert b.node_id in contains_ids
    assert outside.node_id in contains_ids
    assert service.list_dream_compactions() == []


def test_dream_chat_error_falls_back_to_host_then_heuristic(tmp_path: Path) -> None:
    service, settings = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=BrokenChatProvider(),
    )

    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime tooling",
        tags=["tech"],
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime",
        tags=["tech"],
        force=True,
        name="Node B",
    )
    assert a.node_id and b.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="supports")
    service.search(keyword="Graph", top_k=5, include_neighbors=1)

    dream_result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.8,
    )
    assert dream_result.status == "awaiting_host_compaction"
    assert "chat provider failed" in dream_result.pending_compactions[0].reason.lower()

    resolution = service.resolve_dream_compaction(
        task_id=dream_result.pending_compactions[0].task_id,
        use_heuristic=True,
        background=False,
    )
    assert resolution.status == "success"
    assert resolution.resolution_backend == "heuristic"
    node = service.repository.get_node(resolution.node_id)
    assert node.metadata["dream_compaction_backend"] == "heuristic"


def test_background_heuristic_compaction_queues_then_completes(tmp_path: Path) -> None:
    service, settings = build_service(
        tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
        chat_provider=BrokenChatProvider(),
    )

    def inline_background_heuristic_compaction(*, task_id: str) -> None:
        service.execute_heuristic_compaction(task_id=task_id)

    service._launch_background_heuristic_compaction = (  # type: ignore[method-assign]
        inline_background_heuristic_compaction
    )

    a = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime tooling",
        tags=["tech"],
        force=True,
        name="Node A",
    )
    b = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Graph memory runtime",
        tags=["tech"],
        force=True,
        name="Node B",
    )
    assert a.node_id and b.node_id
    service.link_nodes(src_id=a.node_id, dst_id=b.node_id, relation="supports")
    service.search(keyword="Graph", top_k=5, include_neighbors=1)

    dream_result = service.run_dream(
        output_path=settings.memory_output_path,
        window_hours=24,
        min_accesses=1,
        min_cluster_size=2,
        max_candidates=20,
        similarity_threshold=0.8,
    )
    task = dream_result.pending_compactions[0]

    resolution = service.resolve_dream_compaction(
        task_id=task.task_id,
        use_heuristic=True,
    )
    assert resolution.status == "queued"
    assert resolution.resolution_backend == "heuristic"

    runs = service.list_dream_runs(limit=5)
    assert runs[0].status == "success"
    assert runs[0].clusters_created >= 1


def test_due_dream_without_chat_returns_reminder(tmp_path: Path) -> None:
    service, _settings = build_service(
        tmp_path,
        dream_event_threshold=2,
    )

    first = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="First event",
        tags=["tech"],
        name="One",
    )
    second = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Second event",
        tags=["tech"],
        name="Two",
    )
    assert not first.notices
    assert second.notices
    assert "Dream reminder" in second.notices[0]


def test_first_memory_event_age_alone_does_not_trigger_dream(tmp_path: Path) -> None:
    service, _settings = build_service(
        tmp_path,
        dream_event_threshold=50,
        dream_max_age_hours=24,
    )

    receipt = service.add_node(
        payload_type=AddPayloadType.CONTENT,
        payload="Only memory event",
        tags=["tech"],
        name="One",
    )
    assert not receipt.notices

    stale_timestamp = (datetime.now(UTC) - timedelta(hours=49)).strftime("%Y-%m-%d %H:%M:%S")
    with open_connection(service.settings.db_path) as connection:
        connection.execute(
            "UPDATE memory_events SET created_at = ?",
            (stale_timestamp,),
        )

    status = service.get_dream_status()
    assert status.event_count_since_last_dream == 1
    assert status.hours_since_last_dream_or_first_event is not None
    assert status.hours_since_last_dream_or_first_event >= 48
    assert status.due is False
    assert status.reasons == []
    assert status.reminder is None


def test_stale_dream_with_insufficient_new_events_only_emits_deferred_reminder(
    tmp_path: Path,
) -> None:
    service, _settings = build_service(
        tmp_path,
        dream_event_threshold=50,
        dream_max_age_hours=24,
        dream_age_min_event_count=5,
    )

    run_id = service.repository.start_dream_run(
        trigger_reason="test",
        auto_triggered=False,
        requires_chat=True,
    )
    service.repository.complete_dream_run(
        run_id,
        status="success",
        candidate_count=0,
        clusters_created=0,
        memory_path=None,
        notes=[],
    )
    for index in range(4):
        service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload=f"Event {index}",
            tags=["tech"],
            name=f"Node {index}",
        )

    stale_timestamp = (datetime.now(UTC) - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")
    with open_connection(service.settings.db_path) as connection:
        connection.execute(
            "UPDATE dream_runs SET completed_at = ? WHERE run_id = ?",
            (stale_timestamp, run_id),
        )

    status = service.get_dream_status()
    assert status.due is False
    assert status.reminder is not None
    assert "only 4 new events accumulated" in status.reminder

    results = service.search(keyword="Event", top_k=5, include_neighbors=0)
    assert results[0].notices
    assert "Dream deferred during search" in results[0].notices[0]


def test_stale_dream_with_five_new_events_becomes_due(tmp_path: Path) -> None:
    service, _settings = build_service(
        tmp_path,
        dream_event_threshold=50,
        dream_max_age_hours=24,
        dream_age_min_event_count=5,
    )

    run_id = service.repository.start_dream_run(
        trigger_reason="test",
        auto_triggered=False,
        requires_chat=True,
    )
    service.repository.complete_dream_run(
        run_id,
        status="success",
        candidate_count=0,
        clusters_created=0,
        memory_path=None,
        notes=[],
    )
    for index in range(5):
        service.add_node(
            payload_type=AddPayloadType.CONTENT,
            payload=f"Event {index}",
            tags=["tech"],
            name=f"Node {index}",
        )

    stale_timestamp = (datetime.now(UTC) - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")
    with open_connection(service.settings.db_path) as connection:
        connection.execute(
            "UPDATE dream_runs SET completed_at = ? WHERE run_id = ?",
            (stale_timestamp, run_id),
        )

    status = service.get_dream_status()
    assert status.due is True
    assert status.reminder is not None
    assert "no chat model is configured" in status.reminder.lower()
    assert any("5 new events" in reason for reason in status.reasons)


def test_build_host_bootstrap_writes_mount_files(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path)
    create_system_profile_node(
        service,
        content="User prefers concise answers",
        name="Profile",
    )

    bundle = service.build_host_bootstrap()
    for path in [
        bundle.memory_path,
        bundle.bootstrap_prompt_path,
        bundle.system_prompt_path,
        bundle.mount_manifest_path,
        bundle.mcp_config_path,
        bundle.onboarding_path,
    ]:
        assert Path(path).exists()

    prompt_text = Path(bundle.bootstrap_prompt_path).read_text(encoding="utf-8")
    assert "Cold-start mount procedure" in prompt_text
    assert str(settings.db_path) in prompt_text
    assert "--profile host-core" in prompt_text
    assert bundle.status.host_kind == "generic"

    system_prompt_text = Path(bundle.system_prompt_path).read_text(encoding="utf-8")
    assert "Parameter recipes:" in system_prompt_text
    assert "type=content" in system_prompt_text
    assert "inspect=status|runs|tasks" in system_prompt_text

    status = service.get_host_bootstrap_status(output_dir=tmp_path / ".cognitiveos" / "bootstrap")
    assert status.host_kind == "generic"
    assert any("only implemented for codex" in notice for notice in status.notices)

    mount_manifest = Path(bundle.mount_manifest_path).read_text(encoding="utf-8")
    assert '"--profile"' in mount_manifest
    assert '"host-core"' in mount_manifest
    assert '"--project-root"' in mount_manifest
    assert '"--memory-output-path"' in mount_manifest

    mcp_config = Path(bundle.mcp_config_path).read_text(encoding="utf-8")
    assert '"--profile"' in mcp_config
    assert '"host-core"' in mcp_config
    assert '"--project-root"' in mcp_config
    assert '"--memory-output-path"' in mcp_config


def test_codex_bootstrap_install_and_onboarding_close_the_loop(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path)
    bootstrap_dir = tmp_path / ".cognitiveos" / "bootstrap"

    initial_status = service.get_host_bootstrap_status(
        host_kind="codex",
        output_dir=bootstrap_dir,
    )
    assert initial_status.first_startup is True
    assert initial_status.needs_onboarding is True
    assert initial_status.installed is False
    assert initial_status.needs_mount is True
    assert len(initial_status.onboarding_questions) == 5

    onboarding_status = service.submit_host_onboarding(
        host_kind="codex",
        output_dir=bootstrap_dir,
        answers={
            "display_name": "Bruce",
            "role_team": "Sr. Data Engineer",
            "preferred_language": "Chinese",
            "response_style": "Concise, direct, pragmatic",
            "workspace_goal": "Build CognitiveOS as a host-facing memory runtime",
        },
    )
    assert onboarding_status.needs_onboarding is False

    memory_text = settings.memory_output_path.read_text(encoding="utf-8")
    assert "Bootstrap Identity" in memory_text
    assert "Bootstrap Communication Preferences" in memory_text

    bundle = service.build_host_bootstrap(
        output_dir=bootstrap_dir,
        host_kind="codex",
        install=True,
    )
    assert bundle.installed is True
    assert bundle.status.installed is True

    agents_path = tmp_path / "AGENTS.md"
    project_config_path = tmp_path / ".codex" / "config.toml"
    bootstrap_prompt_path = Path(bundle.bootstrap_prompt_path)
    assert agents_path.exists()
    assert project_config_path.exists()
    assert bootstrap_prompt_path.exists()
    assert "COGNITIVEOS HOST BOOTSTRAP START" in agents_path.read_text(encoding="utf-8")
    assert "Cold-start mount procedure" in agents_path.read_text(encoding="utf-8")
    assert "reduced `codex-core` profile" in agents_path.read_text(encoding="utf-8")
    assert "search/read/add/update/link/dream" in bootstrap_prompt_path.read_text(
        encoding="utf-8"
    )
    assert "omitting `unlink`" in agents_path.read_text(encoding="utf-8")
    assert "mcp_servers.cognitiveos" in project_config_path.read_text(encoding="utf-8")
    assert "codex-core" in project_config_path.read_text(encoding="utf-8")
    assert "--project-root" in project_config_path.read_text(encoding="utf-8")
    assert "--memory-output-path" in project_config_path.read_text(encoding="utf-8")

    final_status = service.get_host_bootstrap_status(
        host_kind="codex",
        output_dir=bootstrap_dir,
    )
    assert final_status.first_startup is False
    assert final_status.needs_onboarding is False
    assert final_status.installed is True
    assert final_status.needs_mount is False


def test_non_codex_host_kinds_use_generic_bootstrap_flow_without_install(tmp_path: Path) -> None:
    service, _ = build_service(tmp_path)

    for host_kind in ("claude-code", "claude-desktop", "gemini-cli", "cursor"):
        status = service.get_host_bootstrap_status(host_kind=host_kind)
        assert status.host_kind == host_kind.replace("-", "_")
        assert status.installed is False
        assert any("only implemented for codex" in notice for notice in status.notices)


def test_search_requires_query_or_keyword(tmp_path: Path) -> None:
    service, _ = build_service(tmp_path)

    try:
        service.search()
    except InvalidPayloadError as exc:
        assert "query or keyword" in str(exc)
    else:
        raise AssertionError("search() should reject an empty query and keyword.")


def test_sqlite_connection_applies_stability_pragmas(tmp_path: Path) -> None:
    db_path = tmp_path / "pragmas.db"
    with open_connection(db_path) as connection:
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        temp_store = connection.execute("PRAGMA temp_store").fetchone()[0]

    assert busy_timeout == 5000
    assert str(journal_mode).lower() == "wal"
    assert int(synchronous) == 1
    assert int(temp_store) == 2


def test_background_process_uses_project_root_not_current_directory(tmp_path: Path) -> None:
    service, settings = build_service(tmp_path)
    captured: dict[str, object] = {}

    import subprocess

    original_popen = subprocess.Popen

    def fake_popen(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs

        class DummyProcess:
            pass

        return DummyProcess()

    subprocess.Popen = fake_popen  # type: ignore[assignment]
    try:
        service._spawn_background_process(["python", "-m", "cognitiveos.background_jobs"])
    finally:
        subprocess.Popen = original_popen  # type: ignore[assignment]

    assert captured["kwargs"]["cwd"] == str(settings.project_root.resolve())
    stdout_handle = captured["kwargs"]["stdout"]
    stderr_handle = captured["kwargs"]["stderr"]
    assert stdout_handle is stderr_handle
    assert Path(stdout_handle.name).parent == settings.background_log_dir
    assert Path(stdout_handle.name).suffix == ".log"
