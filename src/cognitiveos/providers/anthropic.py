from __future__ import annotations

import httpx

from cognitiveos.models import ModelProviderConfig
from cognitiveos.providers.base import ChatProvider

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"


class AnthropicChatProvider(ChatProvider):
    def __init__(self, config: ModelProviderConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or DEFAULT_ANTHROPIC_BASE_URL).rstrip("/")

    def summarize(self, content: str) -> str:
        return self.complete(
            system_prompt="Summarize the provided memory node in 500 characters or fewer.",
            user_prompt=content,
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        response = httpx.post(
            f"{self.base_url}/v1/messages",
            json={
                "model": self.config.model_name,
                "max_tokens": 256,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            headers={
                "content-type": "application/json",
                "x-api-key": self.config.api_key or "",
                "anthropic-version": "2023-06-01",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("content", []):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        raise ValueError("Anthropic response did not include text content.")
