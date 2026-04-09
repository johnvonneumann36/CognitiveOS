from __future__ import annotations

from cognitiveos.config import AppSettings
from cognitiveos.exceptions import UnsupportedProviderError
from cognitiveos.models import ModelProviderConfig, ModelRole
from cognitiveos.providers.anthropic import AnthropicChatProvider
from cognitiveos.providers.base import ChatProvider, EmbeddingProvider
from cognitiveos.providers.gemini import GeminiChatProvider, GeminiEmbeddingProvider
from cognitiveos.providers.local_huggingface import (
    LocalHuggingFaceChatProvider,
    LocalHuggingFaceEmbeddingProvider,
)
from cognitiveos.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from cognitiveos.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider

SUPPORTED_PROVIDER_TYPES = {"ollama", "openai", "anthropic", "gemini", "local_huggingface"}


def build_embedding_provider(settings: AppSettings) -> EmbeddingProvider | None:
    if not settings.has_embedding_provider:
        return None

    provider_type = settings.embedding_provider_type or ""
    if provider_type not in SUPPORTED_PROVIDER_TYPES:
        raise UnsupportedProviderError(
            f"Unsupported embedding provider type: {settings.embedding_provider_type}"
        )
    if provider_type == "anthropic":
        raise UnsupportedProviderError(
            "Anthropic does not provide a native embedding API for this role."
        )
    config = ModelProviderConfig(
        role=ModelRole.EMBEDDING,
        provider_type=provider_type,
        model_name=settings.embedding_model_name or "",
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
    )
    if provider_type == "ollama":
        return OllamaEmbeddingProvider(config)
    if provider_type == "openai":
        return OpenAIEmbeddingProvider(config)
    if provider_type == "gemini":
        return GeminiEmbeddingProvider(config)
    if provider_type == "local_huggingface":
        return LocalHuggingFaceEmbeddingProvider(config)
    raise UnsupportedProviderError(
        f"Unsupported embedding provider type: {settings.embedding_provider_type}"
    )


def build_chat_provider(settings: AppSettings) -> ChatProvider | None:
    if not settings.has_chat_provider:
        return None

    provider_type = settings.chat_provider_type or ""
    if provider_type not in SUPPORTED_PROVIDER_TYPES:
        raise UnsupportedProviderError(
            f"Unsupported chat provider type: {settings.chat_provider_type}"
        )
    config = ModelProviderConfig(
        role=ModelRole.CHAT,
        provider_type=provider_type,
        model_name=settings.chat_model_name or "",
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
    )
    if provider_type == "ollama":
        return OllamaChatProvider(config)
    if provider_type == "openai":
        return OpenAIChatProvider(config)
    if provider_type == "anthropic":
        return AnthropicChatProvider(config)
    if provider_type == "gemini":
        return GeminiChatProvider(config)
    if provider_type == "local_huggingface":
        return LocalHuggingFaceChatProvider(config)
    raise UnsupportedProviderError(
        f"Unsupported chat provider type: {settings.chat_provider_type}"
    )
