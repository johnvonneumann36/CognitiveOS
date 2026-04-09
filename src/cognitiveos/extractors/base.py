from __future__ import annotations

from abc import ABC, abstractmethod

from cognitiveos.models import ExtractedContent


class ContentExtractor(ABC):
    @abstractmethod
    def extract(self, uri: str) -> ExtractedContent:
        """Extract text content and metadata from a local file or remote URI."""
