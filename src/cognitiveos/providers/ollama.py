from __future__ import annotations

from collections.abc import Sequence

import httpx

from cognitiveos.models import ModelProviderConfig
from cognitiveos.providers.base import ChatProvider, EmbeddingProvider

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/api"


class _OllamaProviderMixin:
    def __init__(self, config: ModelProviderConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _post(self, path: str, payload: dict) -> dict:
        response = httpx.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._headers(),
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()


class OllamaEmbeddingProvider(_OllamaProviderMixin, EmbeddingProvider):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.config.model_name,
            "input": list(texts),
        }
        response = self._post("/embed", payload)
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list):
            return embeddings

        legacy = response.get("embedding")
        if isinstance(legacy, list):
            return [legacy]
        raise ValueError("Ollama embedding response did not include embeddings.")


class OllamaChatProvider(_OllamaProviderMixin, ChatProvider):
    def summarize(self, content: str) -> str:
        return self.complete(
            system_prompt=(
                "You summarize memory nodes. "
                "Return a concise summary in 500 characters or fewer."
            ),
            user_prompt=content,
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model_name,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        }
        response = self._post("/chat", payload)
        message = response.get("message") or {}
        summary = (message.get("content") or "").strip()
        if not summary:
            raise ValueError("Ollama chat response did not include message content.")
        return summary
