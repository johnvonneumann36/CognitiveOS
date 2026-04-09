class CognitiveOSError(Exception):
    """Base exception for CognitiveOS."""


class NodeNotFoundError(CognitiveOSError):
    """Raised when a node does not exist."""


class InvalidPayloadError(CognitiveOSError):
    """Raised when an invalid request payload is provided."""


class UnsupportedExtractorError(CognitiveOSError):
    """Raised when no extractor can handle a URI."""


class UnsupportedProviderError(CognitiveOSError):
    """Raised when a configured provider type is unsupported."""


class SimilarityBlockedError(CognitiveOSError):
    """Raised when similarity probing blocks a write."""

    def __init__(self, conflicts: list[dict[str, object]]) -> None:
        super().__init__("High similarity detected.")
        self.conflicts = conflicts
