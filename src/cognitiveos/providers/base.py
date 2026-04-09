from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from cognitiveos.models import ModelProviderConfig


class EmbeddingProvider(ABC):
    config: ModelProviderConfig

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embeddings for the provided texts."""


class ChatProvider(ABC):
    config: ModelProviderConfig

    @abstractmethod
    def summarize(self, content: str) -> str:
        """Summarize content into a compact description."""

    @abstractmethod
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        """Run a general chat completion and return the text output."""
