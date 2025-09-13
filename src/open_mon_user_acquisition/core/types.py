"""Core types and enumerations for OpenMonetization-UserAcquisition."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


class TaskStatus(str, Enum):
    """Enumeration of possible task statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """Enumeration of possible workflow statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChannelType(str, Enum):
    """Enumeration of user acquisition channel types."""
    PAID_SEARCH = "paid_search"
    ORGANIC_SEARCH = "organic_search"
    SOCIAL_MEDIA = "social_media"
    EMAIL_MARKETING = "email_marketing"
    CONTENT_MARKETING = "content_marketing"
    AFFILIATE = "affiliate"
    PARTNERSHIP = "partnership"
    DIRECT = "direct"
    REFERRAL = "referral"


class MetricType(str, Enum):
    """Enumeration of performance metric types."""
    IMPRESSIONS = "impressions"
    CLICKS = "clicks"
    CONVERSIONS = "conversions"
    REVENUE = "revenue"
    COST = "cost"
    CPA = "cpa"  # Cost Per Acquisition
    CAC = "cac"  # Customer Acquisition Cost
    LTV = "ltv"  # Lifetime Value
    ROI = "roi"  # Return on Investment
    RETENTION_RATE = "retention_rate"
    CHURN_RATE = "churn_rate"


class LLMBackendType(str, Enum):
    """Enumeration of supported LLM backend types."""
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class TaskSpec:
    """Specification for a task to be executed."""
    id: str
    name: str
    agent_type: str
    description: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    priority: int = 1


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowInstance(BaseModel):
    """Represents a running instance of a workflow."""
    id: str
    name: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    tasks: List[TaskSpec] = Field(default_factory=list)
    results: Dict[str, TaskResult] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class MetricData(BaseModel):
    """Represents a single metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: datetime = Field(default_factory=datetime.now)
    channel: Optional[ChannelType] = None
    campaign: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Standardized response from LLM backends."""
    content: str
    usage: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    """Represents a message in an LLM conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextData(BaseModel):
    """Context data passed between agents and workflows."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    campaign_id: Optional[str] = None
    channel: Optional[ChannelType] = None
    metrics: Dict[str, MetricData] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    previous_results: Dict[str, Any] = Field(default_factory=dict)
