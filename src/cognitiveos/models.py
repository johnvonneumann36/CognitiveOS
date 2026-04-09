from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


PHYSICAL_MAX_NODE_CONTENT_CHARS = 65535


class AddPayloadType(str, Enum):
    CONTENT = "content"
    FILE = "file"
    FOLDER = "folder"


class ModelRole(str, Enum):
    EMBEDDING = "embedding"
    CHAT = "chat"


class ProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL_HUGGINGFACE = "local_huggingface"


class ModelProviderConfig(BaseModel):
    role: ModelRole
    provider_type: ProviderType
    model_name: str
    api_key: str | None = None
    base_url: str | None = None


class ExtractedContent(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None


class EdgeRecord(BaseModel):
    src_id: str
    dst_id: str
    relation: str
    strength_score: float = 1.0
    durability: str = "durable"
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    last_reinforced_at: str | None = None


class LinkedNode(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    relation: str
    direction: Literal["inbound", "outbound"]
    content_length: int
    hop: int


class SearchResult(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    node_type: str = "memory"
    durability: str = "working"
    score: float | None = None
    semantic_score: float | None = None
    keyword_score: float | None = None
    notices: list[str] = Field(default_factory=list)
    linked_nodes: list[LinkedNode] = Field(default_factory=list)


class ReadNodeResult(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    content: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    node_type: str = "memory"
    durability: str = "working"
    edges: list[EdgeRecord] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    last_reinforced_at: str | None = None
    updated_at: str | None = None
    created_at: str | None = None


class ConflictNode(BaseModel):
    id: str
    name: str | None = None
    similarity: float


class Receipt(BaseModel):
    status: str
    action_taken: str
    node_id: str | None = None
    audit_log_id: str | None = None
    edge: dict[str, Any] | None = None
    reason: str | None = None
    conflicting_nodes: list[ConflictNode] = Field(default_factory=list)
    suggestion: str | None = None
    notices: list[str] = Field(default_factory=list)


class DreamSuperNode(BaseModel):
    node_id: str
    source_node_ids: list[str] = Field(default_factory=list)


class DreamCompactionTask(BaseModel):
    task_id: str
    run_id: str
    status: str
    requested_backend: str
    fallback_backend: str = "heuristic"
    reason: str | None = None
    suggested_title: str | None = None
    suggested_description: str | None = None
    prepared_content: str
    prompt: str
    source_nodes: list[dict[str, Any]] = Field(default_factory=list)


class DreamCompactionResolution(BaseModel):
    status: str
    task_id: str
    run_id: str
    resolution_backend: str
    node_id: str | None = None
    remaining_tasks: int = 0
    dream_completed: bool = False
    memory_path: str | None = None


class DreamRelationshipCleanupPlan(BaseModel):
    src_id: str
    dst_id: str
    relation: str
    current_edge_status: str
    recommended_action: str
    reason: str
    strength_score: float | None = None


class DreamDurabilitySuggestion(BaseModel):
    node_id: str
    current_durability: str
    recommended_durability: str
    reason: str
    confidence: float = 0.0


class DreamRunInfo(BaseModel):
    run_id: str
    status: str
    trigger_reason: str | None = None
    auto_triggered: bool = False
    requires_chat: bool = False
    candidate_count: int = 0
    clusters_created: int = 0
    pending_task_count: int = 0
    memory_path: str | None = None
    notes: list[str] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


class DreamResult(BaseModel):
    status: str
    candidate_node_ids: list[str] = Field(default_factory=list)
    clusters_created: int = 0
    super_nodes: list[DreamSuperNode] = Field(default_factory=list)
    relationship_cleanup_plans: list[DreamRelationshipCleanupPlan] = Field(default_factory=list)
    durability_suggestions: list[DreamDurabilitySuggestion] = Field(default_factory=list)
    pending_compactions: list[DreamCompactionTask] = Field(default_factory=list)
    memory_path: str | None = None
    run_id: str | None = None
    trigger_reason: str | None = None
    auto_triggered: bool = False
    notices: list[str] = Field(default_factory=list)


class DreamStatus(BaseModel):
    due: bool
    event_count_since_last_dream: int = 0
    last_dream_completed_at: str | None = None
    hours_since_last_dream_or_first_event: float | None = None
    reasons: list[str] = Field(default_factory=list)
    reminder: str | None = None


class HostOnboardingQuestion(BaseModel):
    id: str
    prompt: str
    guidance: str | None = None
    required: bool = True
    example: str | None = None


class HostBootstrapStatus(BaseModel):
    host_kind: str
    first_startup: bool = False
    onboarding_completed: bool = False
    needs_onboarding: bool = False
    installed: bool = False
    needs_mount: bool = False
    installed_at: str | None = None
    memory_path: str | None = None
    bootstrap_prompt_path: str | None = None
    system_prompt_path: str | None = None
    mount_manifest_path: str | None = None
    mcp_config_path: str | None = None
    onboarding_path: str | None = None
    host_instruction_path: str | None = None
    host_project_config_path: str | None = None
    system_prompt_block: str | None = None
    onboarding_questions: list[HostOnboardingQuestion] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)


class BootstrapBundle(BaseModel):
    host_kind: str
    memory_path: str
    bootstrap_prompt_path: str
    system_prompt_path: str
    mount_manifest_path: str
    mcp_config_path: str
    onboarding_path: str
    host_instruction_path: str | None = None
    host_project_config_path: str | None = None
    installed: bool = False
    status: HostBootstrapStatus


class HostOnboardingSubmission(BaseModel):
    host_kind: str = "generic"
    answers: dict[str, str] = Field(default_factory=dict)


class NodeRecord(BaseModel):
    id: str
    name: str | None = None
    description: str
    content: str
    embedding: list[float] | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    node_type: str = "memory"
    durability: str = "working"
    last_reinforced_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if len(value) > 800:
            raise ValueError("description must be 800 characters or fewer")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if len(value) > PHYSICAL_MAX_NODE_CONTENT_CHARS:
            raise ValueError(
                f"content must be {PHYSICAL_MAX_NODE_CONTENT_CHARS} characters or fewer"
            )
        return value
