"""Main workflow orchestrator that coordinates all system components."""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..core.types import ContextData, WorkflowInstance
from ..core.interfaces import (
    OrchestratorInterface,
    StorageBackendInterface,
    LLMBackendInterface,
)
from ..config import ConfigManager
from ..llm import LLMFallbackManager
from ..storage import SQLiteStorageBackend
from .task_scheduler import TaskScheduler
from .workflow_engine import WorkflowEngine

logger = logging.getLogger(__name__)


class WorkflowOrchestrator(OrchestratorInterface):
    """Main orchestrator that coordinates workflow execution and system components."""

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        storage: Optional[StorageBackendInterface] = None,
        llm_backend: Optional[LLMBackendInterface] = None,
    ):
        """Initialize the workflow orchestrator.

        Args:
            config: Configuration manager. If None, creates default.
            storage: Storage backend. If None, creates SQLite backend.
            llm_backend: LLM backend. If None, creates fallback manager.
        """
        self.config = config or ConfigManager()
        self.storage = storage or SQLiteStorageBackend()
        self.llm_backend = llm_backend or LLMFallbackManager()

        # Initialize components
        self.task_scheduler = TaskScheduler(self.config)
        self.workflow_engine = WorkflowEngine(
            self.config,
            self.storage,
            self.task_scheduler,
        )

        self._initialized = False
        self._running = False

    async def initialize(self) -> None:
        """Initialize all orchestrator components."""
        if self._initialized:
            return

        logger.info("Initializing workflow orchestrator...")

        # Initialize storage
        await self.storage.initialize()

        # Register default agents
        await self._register_default_agents()

        # Start workflow monitoring
        await self.workflow_engine.start_monitoring()

        # Validate LLM backend
        if not await self.llm_backend.validate_connection():
            logger.warning("No LLM backends are available. Some features may not work.")

        self._initialized = True
        logger.info("Workflow orchestrator initialized successfully")

    async def _register_default_agents(self) -> None:
        """Register the default set of agents."""
        from ..agents import UserAcquisitionAgent, PaidSearchAgent, SocialMediaAgent

        logger.info("Registering default agents...")

        # Create and register agents
        agents = [
            UserAcquisitionAgent(llm_backend=self.llm_backend),
            PaidSearchAgent(llm_backend=self.llm_backend),
            SocialMediaAgent(llm_backend=self.llm_backend),
        ]

        for agent in agents:
            self.register_agent(agent.name, agent)

        logger.info(f"Registered {len(agents)} default agents successfully")

    async def shutdown(self) -> None:
        """Shutdown all orchestrator components."""
        if not self._initialized:
            return

        logger.info("Shutting down workflow orchestrator...")

        # Stop workflow monitoring
        await self.workflow_engine.stop_monitoring()

        # Shutdown task scheduler
        await self.task_scheduler.shutdown()

        # Close storage
        await self.storage.close()

        self._initialized = False
        logger.info("Workflow orchestrator shutdown complete")

    async def create_workflow(
        self,
        name: str,
        tasks: List,
        context: ContextData
    ) -> WorkflowInstance:
        """Create a new workflow instance.

        Args:
            name: Name of the workflow.
            tasks: List of tasks to execute (ignored, tasks are planned by agents).
            context: Execution context.

        Returns:
            Created workflow instance.
        """
        await self._ensure_initialized()

        # For now, create workflow with all registered agents
        # In a more advanced implementation, this could be configurable
        workflow = await self.workflow_engine.create_workflow(
            name=name,
            context=context,
        )

        logger.info(f"Created workflow: {workflow.id}")
        return workflow

    async def execute_workflow(self, workflow_id: str) -> WorkflowInstance:
        """Execute a workflow instance.

        Args:
            workflow_id: ID of the workflow to execute.

        Returns:
            Updated workflow instance.
        """
        await self._ensure_initialized()

        workflow = await self.workflow_engine.execute_workflow(workflow_id)

        logger.info(f"Executed workflow: {workflow_id} with status: {workflow.status}")
        return workflow

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Get the status of a workflow instance.

        Args:
            workflow_id: ID of the workflow.

        Returns:
            Workflow instance if found, None otherwise.
        """
        await self._ensure_initialized()

        return await self.workflow_engine.get_workflow_status(workflow_id)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel.

        Returns:
            True if workflow was cancelled, False otherwise.
        """
        await self._ensure_initialized()

        cancelled = await self.workflow_engine.cancel_workflow(workflow_id)

        if cancelled:
            logger.info(f"Cancelled workflow: {workflow_id}")
        else:
            logger.warning(f"Failed to cancel workflow: {workflow_id}")

        return cancelled

    async def list_active_workflows(self) -> List[WorkflowInstance]:
        """List all active workflows.

        Returns:
            List of active workflow instances.
        """
        await self._ensure_initialized()

        return await self.workflow_engine.list_active_workflows()

    async def _ensure_initialized(self) -> None:
        """Ensure the orchestrator is initialized."""
        if not self._initialized:
            await self.initialize()

    # Additional convenience methods

    async def create_and_execute_workflow(
        self,
        name: str,
        context: ContextData,
    ) -> WorkflowInstance:
        """Create and immediately execute a workflow.

        Args:
            name: Name of the workflow.
            context: Execution context.

        Returns:
            Executed workflow instance.
        """
        await self._ensure_initialized()

        # Create the workflow
        workflow = await self.create_workflow(name, [], context)

        # Execute it
        executed_workflow = await self.execute_workflow(workflow.id)

        return executed_workflow

    async def get_system_status(self) -> Dict[str, any]:
        """Get the overall system status.

        Returns:
            Dictionary containing system status information.
        """
        await self._ensure_initialized()

        status = {
            "initialized": self._initialized,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "storage": {
                    "available": True,
                    "type": type(self.storage).__name__,
                },
                "llm_backend": {
                    "available": await self.llm_backend.validate_connection(),
                    "type": type(self.llm_backend).__name__,
                },
                "task_scheduler": {
                    "active": True,
                    "running_tasks": len(self.task_scheduler.get_running_tasks()),
                },
            },
        }

        # Add workflow metrics
        try:
            workflow_metrics = await self.workflow_engine.get_workflow_metrics()
            status["workflow_metrics"] = workflow_metrics
        except Exception as e:
            logger.error(f"Failed to get workflow metrics: {e}")
            status["workflow_metrics"] = {"error": str(e)}

        return status

    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old workflow and metric data.

        Args:
            days_to_keep: Number of days of data to keep.

        Returns:
            Dictionary with cleanup statistics.
        """
        await self._ensure_initialized()

        cleanup_stats = {
            "workflows_cleaned": 0,
            "metrics_cleaned": 0,
            "errors": [],
        }

        try:
            # Clean up old workflows from memory
            workflows_cleaned = await self.workflow_engine.cleanup_completed_workflows(
                max_age_hours=days_to_keep * 24
            )
            cleanup_stats["workflows_cleaned"] = workflows_cleaned
        except Exception as e:
            cleanup_stats["errors"].append(f"Workflow cleanup failed: {e}")

        # Clean up old data from storage
        if hasattr(self.storage, 'cleanup_old_data'):
            try:
                storage_cleaned = await self.storage.cleanup_old_data(days_to_keep)
                cleanup_stats["storage_cleaned"] = storage_cleaned
            except Exception as e:
                cleanup_stats["errors"].append(f"Storage cleanup failed: {e}")

        logger.info(f"Data cleanup completed: {cleanup_stats}")
        return cleanup_stats

    def register_agent(self, agent_type: str, agent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent_type: Type identifier for the agent.
            agent: Agent instance.
        """
        self.workflow_engine.register_agent(agent_type, agent)
        logger.info(f"Registered agent with orchestrator: {agent_type}")

    def get_available_agents(self) -> List[str]:
        """Get list of available agent types.

        Returns:
            List of available agent type identifiers.
        """
        return list(self.workflow_engine._agents.keys())
