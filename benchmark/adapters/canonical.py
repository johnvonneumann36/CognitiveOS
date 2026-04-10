from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CanonicalTurn:
    role: str
    content: str


@dataclass(slots=True)
class CanonicalSession:
    session_id: str
    timestamp: str | None
    turns: list[CanonicalTurn] = field(default_factory=list)


@dataclass(slots=True)
class CanonicalSample:
    suite: str
    sample_id: str
    question_id: str
    question: str
    answer: str
    category: str | None
    abstention: bool = False
    sessions: list[CanonicalSession] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
