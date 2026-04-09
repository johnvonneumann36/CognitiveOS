from __future__ import annotations

import httpx

from cognitiveos.models import ModelProviderConfig
from cognitiveos.providers.base import ChatProvider, EmbeddingProvider

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class _OpenAIProviderMixin:
    def __init__(self, config: ModelProviderConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")

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


class OpenAIEmbeddingProvider(_OpenAIProviderMixin, EmbeddingProvider):
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._post(
            "/embeddings",
            {
                "model": self.config.model_name,
                "input": texts,
            },
        )
        data = response.get("data", [])
        return [item["embedding"] for item in data]


class OpenAIChatProvider(_OpenAIProviderMixin, ChatProvider):
    def summarize(self, content: str) -> str:
        return self.complete(
            system_prompt="Summarize the provided memory node in 500 characters or fewer.",
            user_prompt=content,
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._post(
            "/responses",
            {
                "model": self.config.model_name,
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": system_prompt,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    },
                ],
            },
        )
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        for item in response.get("output", []):
            for part in item.get("content", []):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
        raise ValueError("OpenAI response did not include output text.")
