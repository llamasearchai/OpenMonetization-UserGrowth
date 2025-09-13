"""Core module for OpenMonetization-UserAcquisition.

This module contains the fundamental building blocks of the OMUA system including
types, interfaces, and base classes for agents, workflows, and backends.
"""

from .types import (
    TaskStatus,
    WorkflowStatus,
    ChannelType,
    MetricType,
    LLMBackendType,
    TaskSpec,
    TaskResult,
    WorkflowInstance,
    MetricData,
    LLMResponse,
    LLMMessage,
    ContextData,
)

from .interfaces import (
    AgentInterface,
    LLMBackendInterface,
    StorageBackendInterface,
    ConfigInterface,
    PluginInterface,
    OrchestratorInterface,
)

__all__ = [
    # Types
    "TaskStatus",
    "WorkflowStatus",
    "ChannelType",
    "MetricType",
    "LLMBackendType",
    "TaskSpec",
    "TaskResult",
    "WorkflowInstance",
    "MetricData",
    "LLMResponse",
    "LLMMessage",
    "ContextData",

    # Interfaces
    "AgentInterface",
    "LLMBackendInterface",
    "StorageBackendInterface",
    "ConfigInterface",
    "PluginInterface",
    "OrchestratorInterface",
]
