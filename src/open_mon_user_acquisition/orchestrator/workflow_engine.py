"""Workflow engine for managing workflow lifecycle and execution."""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from uuid import uuid4

from ..core.types import (
    TaskSpec,
    TaskResult,
    WorkflowInstance,
    WorkflowStatus,
    ContextData,
)
from ..core.interfaces import AgentInterface, StorageBackendInterface
from ..config import ConfigManager
from .task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)


class WorkflowEngine:
    """Engine for managing workflow lifecycle and execution."""

    def __init__(
        self,
        config: ConfigManager,
        storage: StorageBackendInterface,
        scheduler: TaskScheduler,
    ):
        """Initialize the workflow engine.

        Args:
            config: Configuration manager.
            storage: Storage backend for persistence.
            scheduler: Task scheduler for execution.
        """
        self.config = config
        self.storage = storage
        self.scheduler = scheduler

        self._active_workflows: Dict[str, WorkflowInstance] = {}
        self._agents: Dict[str, AgentInterface] = {}

        # Background task for workflow monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    def register_agent(self, agent_type: str, agent: AgentInterface) -> None:
        """Register an agent for workflow execution.

        Args:
            agent_type: Type identifier for the agent.
            agent: Agent instance.
        """
        self._agents[agent_type] = agent
        self.scheduler.register_agent(agent_type, agent)
        logger.info(f"Registered agent in workflow engine: {agent_type}")

    async def create_workflow(
        self,
        name: str,
        context: ContextData,
        agent_types: Optional[List[str]] = None,
    ) -> WorkflowInstance:
        """Create a new workflow instance.

        Args:
            name: Name of the workflow.
            context: Initial context for the workflow.
            agent_types: Optional list of agent types to use.

        Returns:
            Created workflow instance.
        """
        workflow_id = str(uuid4())

        # Plan tasks using available agents
        tasks = []
        if agent_types:
            for agent_type in agent_types:
                agent = self._agents.get(agent_type)
                if agent:
                    agent_tasks = await agent.plan(context)
                    tasks.extend(agent_tasks)
        else:
            # Use all available agents
            for agent in self._agents.values():
                agent_tasks = await agent.plan(context)
                tasks.extend(agent_tasks)

        # Assign unique IDs to tasks if they don't already have one
        for i, task in enumerate(tasks):
            if not task.id:
                task.id = f"{workflow_id}_task_{i}"

        workflow = WorkflowInstance(
            id=workflow_id,
            name=name,
            status=WorkflowStatus.PENDING,
            tasks=tasks,
            created_at=datetime.now(),
            metadata={"agent_types": agent_types or list(self._agents.keys())},
        )

        # Store the workflow in active workflows (don't save to storage yet)
        self._active_workflows[workflow_id] = workflow

        logger.info(f"Created workflow: {workflow_id} ({name}) with {len(tasks)} tasks")
        return workflow

    async def execute_workflow(self, workflow_id: str) -> WorkflowInstance:
        """Execute a workflow instance.

        Args:
            workflow_id: ID of the workflow to execute.

        Returns:
            Updated workflow instance.

        Raises:
            ValueError: If workflow is not found or already running.
        """
        # Get the workflow
        workflow = self._active_workflows.get(workflow_id)
        workflow_from_storage = False

        if not workflow:
            # Try to load from storage
            workflow = await self.storage.load_workflow(workflow_id)
            workflow_from_storage = True
            if not workflow:
                raise ValueError(f"Workflow not found: {workflow_id}")

        if workflow.status == WorkflowStatus.RUNNING:
            raise ValueError(f"Workflow is already running: {workflow_id}")

        # Update workflow status
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()

        # Save workflow to storage when execution starts
        await self.storage.save_workflow(workflow)

        self._active_workflows[workflow_id] = workflow

        logger.info(f"Starting workflow execution: {workflow_id}")

        try:
            # Execute tasks
            context = ContextData(
                workflow_id=workflow_id,
                metadata={"workflow_name": workflow.name},
            )

            task_results = await self.scheduler.execute_tasks_concurrent(
                workflow.tasks,
                context,
                workflow_id,
            )

            # Update workflow with results
            workflow.results = task_results
            workflow.completed_at = datetime.now()

            # Determine final status
            failed_tasks = [r for r in task_results.values() if r.status == "failed"]
            if failed_tasks:
                workflow.status = WorkflowStatus.FAILED
                logger.error(f"Workflow {workflow_id} failed with {len(failed_tasks)} failed tasks")
            else:
                workflow.status = WorkflowStatus.COMPLETED
                logger.info(f"Workflow {workflow_id} completed successfully")

        except Exception as e:
            logger.error(f"Workflow {workflow_id} execution failed: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()

        # Save final state
        await self.storage.save_workflow(workflow)
        self._active_workflows[workflow_id] = workflow

        return workflow

    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Get the status of a workflow instance.

        Args:
            workflow_id: ID of the workflow.

        Returns:
            Workflow instance if found, None otherwise.
        """
        # Check active workflows first
        workflow = self._active_workflows.get(workflow_id)
        if workflow:
            return workflow

        # Load from storage
        return await self.storage.load_workflow(workflow_id)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow.

        Args:
            workflow_id: ID of the workflow to cancel.

        Returns:
            True if workflow was cancelled, False if not found or not running.
        """
        workflow = self._active_workflows.get(workflow_id)
        if not workflow:
            return False

        if workflow.status != WorkflowStatus.RUNNING:
            return False

        # Update workflow status
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.now()
        await self.storage.save_workflow(workflow)
        self._active_workflows[workflow_id] = workflow

        logger.info(f"Cancelled workflow: {workflow_id}")
        return True

    async def list_active_workflows(self) -> List[WorkflowInstance]:
        """List all active (non-completed) workflows.

        Returns:
            List of active workflow instances.
        """
        active_statuses = [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]
        active_workflows = [
            w for w in self._active_workflows.values()
            if w.status in active_statuses
        ]

        # Also check storage for any workflows that might have been missed
        stored_workflows = await self.storage.list_workflows(limit=100)
        for workflow in stored_workflows:
            if (workflow.status in active_statuses and
                workflow.id not in self._active_workflows):
                active_workflows.append(workflow)
                self._active_workflows[workflow.id] = workflow

        return active_workflows

    async def cleanup_completed_workflows(self, max_age_hours: int = 24) -> int:
        """Clean up old completed workflows from memory.

        Args:
            max_age_hours: Maximum age in hours for completed workflows to keep in memory.

        Returns:
            Number of workflows cleaned up.
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for workflow_id, workflow in self._active_workflows.items():
            if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED] and
                workflow.completed_at and workflow.completed_at < cutoff_time):
                to_remove.append(workflow_id)

        for workflow_id in to_remove:
            del self._active_workflows[workflow_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old workflows from memory")

        return len(to_remove)

    async def start_monitoring(self) -> None:
        """Start the background workflow monitoring task."""
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_workflows())
            logger.info("Started workflow monitoring")

    async def stop_monitoring(self) -> None:
        """Stop the background workflow monitoring task."""
        if self._monitor_task:
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._monitor_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._monitor_task.cancel()
            self._monitor_task = None
            logger.info("Stopped workflow monitoring")

    async def _monitor_workflows(self) -> None:
        """Background task to monitor workflow health and cleanup."""
        cleanup_interval = self.config.get("workflow.cleanup_interval", 86400)  # 24 hours
        monitor_interval = 60  # 1 minute

        while not self._shutdown_event.is_set():
            try:
                # Clean up old workflows
                await self.cleanup_completed_workflows()

                # Check for stuck workflows (running too long)
                await self._check_stuck_workflows()

                # Wait or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=monitor_interval
                )

            except asyncio.TimeoutError:
                continue  # Continue monitoring
            except Exception as e:
                logger.error(f"Error in workflow monitoring: {e}")
                await asyncio.sleep(monitor_interval)

    async def _check_stuck_workflows(self) -> None:
        """Check for workflows that may be stuck and need attention."""
        max_runtime_hours = self.config.get("workflow.max_runtime_hours", 24)
        cutoff_time = datetime.now() - timedelta(hours=max_runtime_hours)

        stuck_workflows = []
        for workflow in self._active_workflows.values():
            if (workflow.status == WorkflowStatus.RUNNING and
                workflow.started_at and workflow.started_at < cutoff_time):
                stuck_workflows.append(workflow)

        for workflow in stuck_workflows:
            logger.warning(f"Workflow {workflow.id} has been running for more than {max_runtime_hours} hours")
            # Could implement automatic cancellation or alerting here

    async def get_workflow_metrics(self) -> Dict[str, int]:
        """Get workflow execution metrics.

        Returns:
            Dictionary of workflow metrics.
        """
        metrics = {
            "active_workflows": len([w for w in self._active_workflows.values()
                                   if w.status == WorkflowStatus.RUNNING]),
            "pending_workflows": len([w for w in self._active_workflows.values()
                                    if w.status == WorkflowStatus.PENDING]),
            "completed_workflows": len([w for w in self._active_workflows.values()
                                      if w.status == WorkflowStatus.COMPLETED]),
            "failed_workflows": len([w for w in self._active_workflows.values()
                                   if w.status == WorkflowStatus.FAILED]),
        }

        # Add storage-based metrics
        try:
            stats = await self.storage.get_workflow_stats()
            metrics.update(stats)
        except Exception as e:
            logger.error(f"Failed to get storage metrics: {e}")

        return metrics
