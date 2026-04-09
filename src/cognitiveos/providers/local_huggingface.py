from __future__ import annotations

from cognitiveos.models import ModelProviderConfig
from cognitiveos.providers.base import ChatProvider, EmbeddingProvider


class LocalHuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: ModelProviderConfig) -> None:
        self.config = config
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Local Hugging Face embeddings require the 'local-hf' extra: "
                "pip install 'cognitiveos[local-hf]'"
            ) from exc
        self.model = SentenceTransformer(config.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(texts).tolist()


class LocalHuggingFaceChatProvider(ChatProvider):
    def __init__(self, config: ModelProviderConfig) -> None:
        self.config = config
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "Local Hugging Face chat requires the 'local-hf' extra: "
                "pip install 'cognitiveos[local-hf]'"
            ) from exc

        try:
            self.generator = pipeline(
                "text2text-generation",
                model=config.model_name,
            )
        except Exception:
            self.generator = pipeline(
                "text-generation",
                model=config.model_name,
            )

    def summarize(self, content: str) -> str:
        return self.complete(
            system_prompt="Summarize the following memory node in 500 characters or fewer.",
            user_prompt=content,
        )

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        outputs = self.generator(prompt, max_new_tokens=160, do_sample=False)
        first = outputs[0]
        generated = first.get("generated_text") or first.get("summary_text") or ""
        if isinstance(generated, str) and generated.strip():
            return generated.strip()
        raise ValueError("Local Hugging Face pipeline did not return generated text.")
