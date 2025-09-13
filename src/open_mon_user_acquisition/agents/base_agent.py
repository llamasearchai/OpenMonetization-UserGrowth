"""Base agent class with common functionality for all OMUA agents."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..core.interfaces import AgentInterface
from ..core.types import TaskSpec, TaskResult, ContextData, TaskStatus
from ..llm import LLMFallbackManager

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all OMUA agents with common LLM functionality."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_backend: Optional[LLMFallbackManager] = None
    ):
        """Initialize the base agent.

        Args:
            name: Agent name identifier
            description: Human-readable description
            llm_backend: LLM backend for AI-powered planning/execution
        """
        self._name = name
        self._description = description
        self.llm_backend = llm_backend

    @property
    def name(self) -> str:
        """Return the agent name."""
        return self._name

    @property
    def description(self) -> str:
        """Return the agent description."""
        return self._description

    @abstractmethod
    async def plan(self, context: ContextData) -> List[TaskSpec]:
        """Plan tasks based on the given context."""
        pass

    @abstractmethod
    async def execute(self, task: TaskSpec, context: ContextData) -> TaskResult:
        """Execute a specific task."""
        pass

    async def _generate_with_llm(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate text using the LLM backend.

        Args:
            prompt: The prompt to send to the LLM
            context: Additional context data
            temperature: LLM temperature setting
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        if not self.llm_backend:
            raise RuntimeError(f"LLM backend not available for agent {self.name}")

        # Enhance prompt with context if provided
        enhanced_prompt = prompt
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            enhanced_prompt = f"Context:\n{context_str}\n\n{prompt}"

        try:
            response = await self.llm_backend.generate(
                prompt=enhanced_prompt,
                options={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed for agent {self.name}: {e}")
            raise

    async def _create_task_result(
        self,
        task: TaskSpec,
        result_data: Dict[str, Any],
        status: TaskStatus = TaskStatus.COMPLETED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """Create a standardized task result.

        Args:
            task: The task that was executed
            result_data: The result data
            status: Task execution status
            metadata: Additional metadata

        Returns:
            TaskResult instance
        """
        from ..core.types import TaskResult

        combined_metadata = {
            "agent": self.name,
            "execution_timestamp": datetime.now().isoformat(),
            "task_type": task.name,
        }

        if metadata:
            combined_metadata.update(metadata)

        return TaskResult(
            task_id=task.id,
            status=status,
            result=result_data,
            metadata=combined_metadata,
            executed_at=datetime.now()
        )

    def _create_task_spec(
        self,
        task_id: str,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 1,
        timeout_seconds: Optional[int] = None
    ) -> TaskSpec:
        """Create a standardized task specification.

        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            description: Task description
            parameters: Task execution parameters
            dependencies: List of task IDs this task depends on
            priority: Task priority (higher = more important)
            timeout_seconds: Task timeout in seconds

        Returns:
            TaskSpec instance
        """
        return TaskSpec(
            id=task_id,
            name=name,
            agent_type=self.name,
            description=description,
            parameters=parameters or {},
            dependencies=dependencies or [],
            priority=priority,
            timeout_seconds=timeout_seconds
        )
