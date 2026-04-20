from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


def _env_bool(name: str, default: bool, *, dotenv: dict[str, str | None] | None = None) -> bool:
    raw = os.getenv(name)
    if raw is None and dotenv is not None:
        value = dotenv.get(name)
        raw = value if isinstance(value, str) else None
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_value(name: str, dotenv: dict[str, str | None]) -> str | None:
    raw = os.getenv(name)
    if raw is not None:
        return raw
    value = dotenv.get(name)
    return value if isinstance(value, str) else None


def _resolve_optional_path(
    value: str | Path | None,
    *,
    base_dir: Path,
) -> Path | None:
    if value is None:
        return None
    path = value if isinstance(value, Path) else Path(value)
    path = path.expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _infer_runtime_root(
    *,
    runtime_home: Path | None,
    db_path: Path | None,
    memory_output_path: Path | None,
) -> Path:
    if memory_output_path is not None:
        return memory_output_path.parent
    if db_path is not None:
        if db_path.parent.name == "data":
            return db_path.parent.parent
        return db_path.parent
    if runtime_home is not None:
        return runtime_home
    return Path.home() / ".cognitiveos"


@dataclass(slots=True)
class AppSettings:
    project_root: Path
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
        project_root: Path | None = None,
    ) -> AppSettings:
        resolved_project_root = _resolve_optional_path(project_root, base_dir=Path.cwd())
        if resolved_project_root is None:
            resolved_project_root = Path.cwd()
        resolved_project_root = resolved_project_root.resolve()
        dotenv = dotenv_values(resolved_project_root / ".env")

        explicit_db_path = _resolve_optional_path(db_path, base_dir=resolved_project_root)
        explicit_memory_path = _resolve_optional_path(
            memory_output_path,
            base_dir=resolved_project_root,
        )
        env_runtime_home = _resolve_optional_path(
            _env_value("COGNITIVEOS_HOME", dotenv),
            base_dir=resolved_project_root,
        )

        env_db_path = _resolve_optional_path(
            _env_value("COGNITIVEOS_DB_PATH", dotenv),
            base_dir=resolved_project_root,
        )
        env_memory_path = _resolve_optional_path(
            _env_value("COGNITIVEOS_MEMORY_OUTPUT_PATH", dotenv),
            base_dir=resolved_project_root,
        )
        runtime_root = _infer_runtime_root(
            runtime_home=env_runtime_home,
            db_path=explicit_db_path or env_db_path,
            memory_output_path=explicit_memory_path or env_memory_path,
        )
        resolved_db_path = (
            explicit_db_path or env_db_path or runtime_root / "data" / "cognitiveos.db"
        )
        resolved_memory_path = explicit_memory_path or env_memory_path or runtime_root / "MEMORY.MD"
        env_bootstrap_dir = _resolve_optional_path(
            _env_value("COGNITIVEOS_BOOTSTRAP_DIR", dotenv),
            base_dir=resolved_project_root,
        )
        resolved_bootstrap_dir = (
            env_bootstrap_dir or resolved_project_root / ".cognitiveos" / "bootstrap"
        )
        resolved_background_log_dir = _resolve_optional_path(
            _env_value(
                "COGNITIVEOS_BACKGROUND_LOG_DIR",
                dotenv,
            )
            or str(runtime_root / "logs"),
            base_dir=runtime_root,
        )
        resolved_snapshot_dir = _resolve_optional_path(
            _env_value(
                "COGNITIVEOS_SNAPSHOT_DIR",
                dotenv,
            )
            or str(runtime_root / "snapshots"),
            base_dir=runtime_root,
        )
        return cls(
            project_root=resolved_project_root,
            db_path=resolved_db_path,
            memory_output_path=resolved_memory_path,
            bootstrap_dir=resolved_bootstrap_dir,
            background_log_dir=resolved_background_log_dir,
            snapshot_dir=resolved_snapshot_dir,
            default_actor=_env_value("COGNITIVEOS_DEFAULT_ACTOR", dotenv) or "agent",
            server_name=_env_value("COGNITIVEOS_SERVER_NAME", dotenv) or "CognitiveOS",
            server_host=_env_value("COGNITIVEOS_SERVER_HOST", dotenv) or "127.0.0.1",
            server_port=int(_env_value("COGNITIVEOS_SERVER_PORT", dotenv) or "8000"),
            server_path=_env_value("COGNITIVEOS_SERVER_PATH", dotenv) or "/mcp",
            server_profile=_env_value("COGNITIVEOS_SERVER_PROFILE", dotenv) or "host-core",
            similarity_threshold=float(
                _env_value("COGNITIVEOS_SIMILARITY_THRESHOLD", dotenv) or "0.92"
            ),
            hybrid_semantic_weight=float(
                _env_value("COGNITIVEOS_HYBRID_SEMANTIC_WEIGHT", dotenv) or "0.65"
            ),
            hybrid_keyword_weight=float(
                _env_value("COGNITIVEOS_HYBRID_KEYWORD_WEIGHT", dotenv) or "0.35"
            ),
            search_candidate_cap=int(
                _env_value("COGNITIVEOS_SEARCH_CANDIDATE_CAP", dotenv) or "20"
            ),
            search_governance_interval_seconds=int(
                _env_value("COGNITIVEOS_SEARCH_GOVERNANCE_INTERVAL_SECONDS", dotenv)
                or "300"
            ),
            search_async_access_logging=_env_bool(
                "COGNITIVEOS_SEARCH_ASYNC_ACCESS_LOGGING",
                True,
                dotenv=dotenv,
            ),
            semantic_neighbor_k=int(
                _env_value("COGNITIVEOS_SEMANTIC_NEIGHBOR_K", dotenv) or "8"
            ),
            dream_event_threshold=int(
                _env_value("COGNITIVEOS_DREAM_EVENT_THRESHOLD", dotenv) or "10"
            ),
            dream_max_age_hours=int(
                _env_value("COGNITIVEOS_DREAM_MAX_AGE_HOURS", dotenv) or "24"
            ),
            dream_age_min_event_count=int(
                _env_value("COGNITIVEOS_DREAM_AGE_MIN_EVENT_COUNT", dotenv) or "5"
            ),
            long_document_token_threshold=int(
                _env_value("COGNITIVEOS_LONG_DOCUMENT_TOKEN_THRESHOLD", dotenv) or "1200"
            ),
            chunk_target_tokens=int(
                _env_value("COGNITIVEOS_CHUNK_TARGET_TOKENS", dotenv) or "900"
            ),
            chunk_overlap_tokens=int(
                _env_value("COGNITIVEOS_CHUNK_OVERLAP_TOKENS", dotenv) or "120"
            ),
            max_main_document_chars=int(
                _env_value("COGNITIVEOS_MAX_MAIN_DOCUMENT_CHARS", dotenv) or "6000"
            ),
            max_node_content_chars=int(
                _env_value("COGNITIVEOS_MAX_NODE_CONTENT_CHARS", dotenv) or "12000"
            ),
            relationship_weak_after_hours=int(
                _env_value("COGNITIVEOS_RELATIONSHIP_WEAK_AFTER_HOURS", dotenv) or "72"
            ),
            relationship_stale_after_hours=int(
                _env_value("COGNITIVEOS_RELATIONSHIP_STALE_AFTER_HOURS", dotenv) or "168"
            ),
            relationship_weak_strength_threshold=float(
                _env_value("COGNITIVEOS_RELATIONSHIP_WEAK_STRENGTH_THRESHOLD", dotenv)
                or "0.85"
            ),
            relationship_stale_strength_threshold=float(
                _env_value("COGNITIVEOS_RELATIONSHIP_STALE_STRENGTH_THRESHOLD", dotenv)
                or "0.35"
            ),
            relationship_weak_decay_delta=float(
                _env_value("COGNITIVEOS_RELATIONSHIP_WEAK_DECAY_DELTA", dotenv) or "0.2"
            ),
            relationship_stale_decay_delta=float(
                _env_value("COGNITIVEOS_RELATIONSHIP_STALE_DECAY_DELTA", dotenv) or "0.35"
            ),
            relationship_manual_reinforcement_delta=float(
                _env_value(
                    "COGNITIVEOS_RELATIONSHIP_MANUAL_REINFORCEMENT_DELTA",
                    dotenv,
                )
                or "0.5"
            ),
            relationship_recall_reinforcement_delta=float(
                _env_value(
                    "COGNITIVEOS_RELATIONSHIP_RECALL_REINFORCEMENT_DELTA",
                    dotenv,
                )
                or "0.1"
            ),
            relationship_dream_reinforcement_delta=float(
                _env_value(
                    "COGNITIVEOS_RELATIONSHIP_DREAM_REINFORCEMENT_DELTA",
                    dotenv,
                )
                or "0.2"
            ),
            embedding_provider_type=_env_value("COGNITIVEOS_EMBEDDING_PROVIDER_TYPE", dotenv),
            embedding_model_name=_env_value("COGNITIVEOS_EMBEDDING_MODEL_NAME", dotenv),
            embedding_base_url=_env_value("COGNITIVEOS_EMBEDDING_BASE_URL", dotenv),
            embedding_api_key=_env_value("COGNITIVEOS_EMBEDDING_API_KEY", dotenv),
            chat_provider_type=_env_value("COGNITIVEOS_CHAT_PROVIDER_TYPE", dotenv),
            chat_model_name=_env_value("COGNITIVEOS_CHAT_MODEL_NAME", dotenv),
            chat_base_url=_env_value("COGNITIVEOS_CHAT_BASE_URL", dotenv),
            chat_api_key=_env_value("COGNITIVEOS_CHAT_API_KEY", dotenv),
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
