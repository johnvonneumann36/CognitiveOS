from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppSettings:
    db_path: Path
    memory_output_path: Path
    bootstrap_dir: Path
    background_log_dir: Path
    snapshot_dir: Path
    default_actor: str = "agent"
    server_name: str = "CognitiveOS"
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    server_path: str = "/mcp"
    server_profile: str = "host-core"
    similarity_threshold: float = 0.92
    hybrid_semantic_weight: float = 0.65
    hybrid_keyword_weight: float = 0.35
    search_candidate_cap: int = 20
    search_governance_interval_seconds: int = 300
    search_async_access_logging: bool = True
    semantic_neighbor_k: int = 8
    dream_event_threshold: int = 10
    dream_max_age_hours: int = 24
    dream_age_min_event_count: int = 5
    long_document_token_threshold: int = 1200
    chunk_target_tokens: int = 900
    chunk_overlap_tokens: int = 120
    max_main_document_chars: int = 6000
    max_node_content_chars: int = 12000
    relationship_weak_after_hours: int = 72
    relationship_stale_after_hours: int = 168
    relationship_weak_strength_threshold: float = 0.85
    relationship_stale_strength_threshold: float = 0.35
    relationship_weak_decay_delta: float = 0.2
    relationship_stale_decay_delta: float = 0.35
    relationship_manual_reinforcement_delta: float = 0.5
    relationship_recall_reinforcement_delta: float = 0.1
    relationship_dream_reinforcement_delta: float = 0.2
    embedding_provider_type: str | None = None
    embedding_model_name: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    chat_provider_type: str | None = None
    chat_model_name: str | None = None
    chat_base_url: str | None = None
    chat_api_key: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        db_path: Path | None = None,
        memory_output_path: Path | None = None,
    ) -> AppSettings:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        cwd = Path.cwd()
        resolved_db_path = db_path or Path(
            os.getenv("COGNITIVEOS_DB_PATH", cwd / "data" / "cognitiveos.db")
        )
        resolved_memory_path = memory_output_path or Path(
            os.getenv("COGNITIVEOS_MEMORY_OUTPUT_PATH", cwd / "MEMORY.MD")
        )
        resolved_bootstrap_dir = Path(
            os.getenv("COGNITIVEOS_BOOTSTRAP_DIR", cwd / ".cognitiveos" / "bootstrap")
        )
        resolved_background_log_dir = Path(
            os.getenv(
                "COGNITIVEOS_BACKGROUND_LOG_DIR",
                resolved_bootstrap_dir.parent / "logs",
            )
        )
        resolved_snapshot_dir = Path(
            os.getenv(
                "COGNITIVEOS_SNAPSHOT_DIR",
                resolved_bootstrap_dir.parent / "snapshots",
            )
        )
        return cls(
            db_path=resolved_db_path,
            memory_output_path=resolved_memory_path,
            bootstrap_dir=resolved_bootstrap_dir,
            background_log_dir=resolved_background_log_dir,
            snapshot_dir=resolved_snapshot_dir,
            default_actor=os.getenv("COGNITIVEOS_DEFAULT_ACTOR", "agent"),
            server_name=os.getenv("COGNITIVEOS_SERVER_NAME", "CognitiveOS"),
            server_host=os.getenv("COGNITIVEOS_SERVER_HOST", "127.0.0.1"),
            server_port=int(os.getenv("COGNITIVEOS_SERVER_PORT", "8000")),
            server_path=os.getenv("COGNITIVEOS_SERVER_PATH", "/mcp"),
            server_profile=os.getenv("COGNITIVEOS_SERVER_PROFILE", "host-core"),
            similarity_threshold=float(
                os.getenv("COGNITIVEOS_SIMILARITY_THRESHOLD", "0.92")
            ),
            hybrid_semantic_weight=float(
                os.getenv("COGNITIVEOS_HYBRID_SEMANTIC_WEIGHT", "0.65")
            ),
            hybrid_keyword_weight=float(
                os.getenv("COGNITIVEOS_HYBRID_KEYWORD_WEIGHT", "0.35")
            ),
            search_candidate_cap=int(
                os.getenv("COGNITIVEOS_SEARCH_CANDIDATE_CAP", "20")
            ),
            search_governance_interval_seconds=int(
                os.getenv("COGNITIVEOS_SEARCH_GOVERNANCE_INTERVAL_SECONDS", "300")
            ),
            search_async_access_logging=_env_bool(
                "COGNITIVEOS_SEARCH_ASYNC_ACCESS_LOGGING", True
            ),
            semantic_neighbor_k=int(
                os.getenv("COGNITIVEOS_SEMANTIC_NEIGHBOR_K", "8")
            ),
            dream_event_threshold=int(os.getenv("COGNITIVEOS_DREAM_EVENT_THRESHOLD", "10")),
            dream_max_age_hours=int(os.getenv("COGNITIVEOS_DREAM_MAX_AGE_HOURS", "24")),
            dream_age_min_event_count=int(
                os.getenv("COGNITIVEOS_DREAM_AGE_MIN_EVENT_COUNT", "5")
            ),
            long_document_token_threshold=int(
                os.getenv("COGNITIVEOS_LONG_DOCUMENT_TOKEN_THRESHOLD", "1200")
            ),
            chunk_target_tokens=int(
                os.getenv("COGNITIVEOS_CHUNK_TARGET_TOKENS", "900")
            ),
            chunk_overlap_tokens=int(
                os.getenv("COGNITIVEOS_CHUNK_OVERLAP_TOKENS", "120")
            ),
            max_main_document_chars=int(
                os.getenv("COGNITIVEOS_MAX_MAIN_DOCUMENT_CHARS", "6000")
            ),
            max_node_content_chars=int(
                os.getenv("COGNITIVEOS_MAX_NODE_CONTENT_CHARS", "12000")
            ),
            relationship_weak_after_hours=int(
                os.getenv("COGNITIVEOS_RELATIONSHIP_WEAK_AFTER_HOURS", "72")
            ),
            relationship_stale_after_hours=int(
                os.getenv("COGNITIVEOS_RELATIONSHIP_STALE_AFTER_HOURS", "168")
            ),
            relationship_weak_strength_threshold=float(
                os.getenv("COGNITIVEOS_RELATIONSHIP_WEAK_STRENGTH_THRESHOLD", "0.85")
            ),
            relationship_stale_strength_threshold=float(
                os.getenv("COGNITIVEOS_RELATIONSHIP_STALE_STRENGTH_THRESHOLD", "0.35")
            ),
            relationship_weak_decay_delta=float(
                os.getenv("COGNITIVEOS_RELATIONSHIP_WEAK_DECAY_DELTA", "0.2")
            ),
            relationship_stale_decay_delta=float(
                os.getenv("COGNITIVEOS_RELATIONSHIP_STALE_DECAY_DELTA", "0.35")
            ),
            relationship_manual_reinforcement_delta=float(
                os.getenv("COGNITIVEOS_RELATIONSHIP_MANUAL_REINFORCEMENT_DELTA", "0.5")
            ),
            relationship_recall_reinforcement_delta=float(
                os.getenv("COGNITIVEOS_RELATIONSHIP_RECALL_REINFORCEMENT_DELTA", "0.1")
            ),
            relationship_dream_reinforcement_delta=float(
                os.getenv("COGNITIVEOS_RELATIONSHIP_DREAM_REINFORCEMENT_DELTA", "0.2")
            ),
            embedding_provider_type=os.getenv("COGNITIVEOS_EMBEDDING_PROVIDER_TYPE"),
            embedding_model_name=os.getenv("COGNITIVEOS_EMBEDDING_MODEL_NAME"),
            embedding_base_url=os.getenv("COGNITIVEOS_EMBEDDING_BASE_URL"),
            embedding_api_key=os.getenv("COGNITIVEOS_EMBEDDING_API_KEY"),
            chat_provider_type=os.getenv("COGNITIVEOS_CHAT_PROVIDER_TYPE"),
            chat_model_name=os.getenv("COGNITIVEOS_CHAT_MODEL_NAME"),
            chat_base_url=os.getenv("COGNITIVEOS_CHAT_BASE_URL"),
            chat_api_key=os.getenv("COGNITIVEOS_CHAT_API_KEY"),
        )

    @property
    def has_embedding_provider(self) -> bool:
        return bool(self.embedding_provider_type and self.embedding_model_name)

    @property
    def has_chat_provider(self) -> bool:
        return bool(self.chat_provider_type and self.chat_model_name)

    def ensure_runtime_paths(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.bootstrap_dir.mkdir(parents=True, exist_ok=True)
        self.background_log_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
