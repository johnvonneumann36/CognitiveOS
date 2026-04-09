import pytest

from cognitiveos.config import AppSettings
from cognitiveos.exceptions import UnsupportedProviderError
from cognitiveos.providers.factory import build_chat_provider, build_embedding_provider


def test_factory_rejects_anthropic_embeddings(tmp_path) -> None:
    settings = AppSettings.from_env(
        db_path=tmp_path / "db.sqlite",
        memory_output_path=tmp_path / "MEMORY.MD",
    )
    settings.embedding_provider_type = "anthropic"
    settings.embedding_model_name = "claude-3-7-sonnet-latest"

    with pytest.raises(UnsupportedProviderError):
        build_embedding_provider(settings)


def test_factory_allows_supported_chat_provider_without_instantiation(tmp_path) -> None:
    settings = AppSettings.from_env(
        db_path=tmp_path / "db.sqlite",
        memory_output_path=tmp_path / "MEMORY.MD",
    )
    settings.chat_provider_type = "anthropic"
    settings.chat_model_name = "claude-3-7-sonnet-latest"
    settings.chat_api_key = "test-key"

    provider = build_chat_provider(settings)
    assert provider is not None
