from __future__ import annotations

from urllib.parse import urlencode

import httpx

from cognitiveos.models import ModelProviderConfig
from cognitiveos.providers.base import ChatProvider, EmbeddingProvider

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class _GeminiProviderMixin:
    def __init__(self, config: ModelProviderConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or DEFAULT_GEMINI_BASE_URL).rstrip("/")

    def _url(self, path: str) -> str:
        query = urlencode({"key": self.config.api_key or ""})
        return f"{self.base_url}{path}?{query}"

    @staticmethod
    def _headers() -> dict[str, str]:
        return {"Content-Type": "application/json"}


class GeminiEmbeddingProvider(_GeminiProviderMixin, EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            response = httpx.post(
                self._url(f"/models/{self.config.model_name}:embedContent"),
                json={
                    "content": {
                        "parts": [{"text": text}],
                    }
                },
                headers=self._headers(),
                timeout=60.0,
            )
            response.raise_for_status()
            payload = response.json()
            values = payload.get("embedding", {}).get("values", [])
            embeddings.append(values)
        return embeddings


class GeminiChatProvider(_GeminiProviderMixin, ChatProvider):
    def summarize(self, content: str) -> str:
        return self.complete(
            system_prompt="Summarize the provided memory node in 500 characters or fewer.",
            user_prompt=content,
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        response = httpx.post(
            self._url(f"/models/{self.config.model_name}:generateContent"),
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": f"{system_prompt}\n\n{user_prompt}"
                            }
                        ],
                    }
                ]
            },
            headers=self._headers(),
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get("candidates", [])
        for candidate in candidates:
            parts = candidate.get("content", {}).get("parts", [])
            for part in parts:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        raise ValueError("Gemini response did not include text content.")
