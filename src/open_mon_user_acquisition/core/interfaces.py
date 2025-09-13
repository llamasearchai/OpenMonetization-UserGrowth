"""Core interfaces for the OpenMonetization-UserAcquisition system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime

from .types import (
    TaskSpec,
    TaskResult,
    WorkflowInstance,
    ContextData,
    LLMResponse,
    LLMMessage,
)


class AgentInterface(Protocol):
    """Protocol for agent implementations."""

    @property
    def name(self) -> str:
        """Return the agent name."""
        ...

    @property
    def description(self) -> str:
        """Return the agent description."""
        ...

    async def plan(self, context: ContextData) -> List[TaskSpec]:
        """Plan tasks based on the given context."""
        ...

    async def execute(self, task: TaskSpec, context: ContextData) -> TaskResult:
        """Execute a specific task."""
        ...


class LLMBackendInterface(ABC):
    """Abstract base class for LLM backend implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available and configured."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Conduct a chat conversation."""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the backend connection is working."""
        pass


class StorageBackendInterface(ABC):
    """Abstract base class for storage backend implementations."""

    @abstractmethod
    async def save_workflow(self, workflow: WorkflowInstance) -> None:
        """Save a workflow instance."""
        pass

    @abstractmethod
    async def load_workflow(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Load a workflow instance by ID."""
        pass

    @abstractmethod
    async def list_workflows(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[WorkflowInstance]:
        """List workflow instances with optional filtering."""
        pass

    @abstractmethod
    async def save_metric(self, metric_name: str, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a metric measurement."""
        pass

    @abstractmethod
    async def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve metrics with optional filtering."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage backend."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend connection."""
        pass


class ConfigInterface(ABC):
    """Abstract base class for configuration management."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        pass

    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if a configuration key exists."""
        pass

    @abstractmethod
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from a file."""
        pass

    @abstractmethod
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a file."""
        pass


class PluginInterface(ABC):
    """Abstract base class for plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the plugin version."""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass


class OrchestratorInterface(ABC):
    """Abstract base class for workflow orchestrators."""

    @abstractmethod
    async def create_workflow(
        self,
        name: str,
        tasks: List[TaskSpec],
        context: ContextData
    ) -> WorkflowInstance:
        """Create a new workflow instance."""
        pass

    @abstractmethod
    async def execute_workflow(self, workflow_id: str) -> WorkflowInstance:
        """Execute a workflow instance."""
        pass

    @abstractmethod
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Get the status of a workflow instance."""
        pass

    @abstractmethod
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        pass

    @abstractmethod
    async def list_active_workflows(self) -> List[WorkflowInstance]:
        """List all active workflows."""
        pass
