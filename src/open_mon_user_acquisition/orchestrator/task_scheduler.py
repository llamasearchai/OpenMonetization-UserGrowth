"""Task scheduler for managing concurrent task execution."""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from ..core.types import TaskSpec, TaskResult, TaskStatus, ContextData
from ..core.interfaces import AgentInterface
from ..config import ConfigManager

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Scheduler for managing concurrent task execution with dependencies."""

    def __init__(
        self,
        config: ConfigManager,
        max_concurrent_tasks: int = 10,
        thread_pool_size: int = 4,
    ):
        """Initialize the task scheduler.

        Args:
            config: Configuration manager instance.
            max_concurrent_tasks: Maximum number of tasks to execute concurrently.
            thread_pool_size: Size of the thread pool for CPU-bound tasks.
        """
        self.config = config
        self.max_concurrent_tasks = max_concurrent_tasks
        self.thread_pool_size = thread_pool_size

        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._executor = ThreadPoolExecutor(max_workers=thread_pool_size, thread_name_prefix="task-executor")

        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._completed_tasks: Dict[str, TaskResult] = {}
        self._task_dependencies: Dict[str, Set[str]] = {}

        self._agents: Dict[str, AgentInterface] = {}

    def register_agent(self, agent_type: str, agent: AgentInterface) -> None:
        """Register an agent for task execution.

        Args:
            agent_type: Type identifier for the agent.
            agent: Agent instance.
        """
        self._agents[agent_type] = agent
        logger.info(f"Registered agent: {agent_type}")

    def unregister_agent(self, agent_type: str) -> None:
        """Unregister an agent.

        Args:
            agent_type: Type identifier for the agent to remove.
        """
        if agent_type in self._agents:
            del self._agents[agent_type]
            logger.info(f"Unregistered agent: {agent_type}")

    async def execute_task(
        self,
        task: TaskSpec,
        context: ContextData,
        workflow_id: str
    ) -> TaskResult:
        """Execute a single task.

        Args:
            task: Task specification to execute.
            context: Execution context.
            workflow_id: ID of the parent workflow.

        Returns:
            Task execution result.
        """
        async with self._semaphore:
            return await self._execute_task_internal(task, context, workflow_id)

    async def _execute_task_internal(
        self,
        task: TaskSpec,
        context: ContextData,
        workflow_id: str
    ) -> TaskResult:
        """Internal task execution logic.

        Args:
            task: Task specification to execute.
            context: Execution context.
            workflow_id: ID of the parent workflow.

        Returns:
            Task execution result.
        """
        task_id = task.id
        logger.info(f"Starting task execution: {task_id} ({task.name})")

        start_time = datetime.now()
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            started_at=start_time,
        )

        try:
            # Get the appropriate agent
            agent = self._agents.get(task.agent_type)
            if not agent:
                raise RuntimeError(f"No agent registered for type: {task.agent_type}")

            # Check timeout
            timeout = task.timeout_seconds or self.config.get("workflow.default_timeout", 3600)

            # Execute the task with retries and timeout
            max_retries = task.retry_count or self.config.get("workflow.retry_attempts", 3)
            for attempt in range(max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        agent.execute(task, context),
                        timeout=timeout
                    )
                    task_result = result
                    task_result.started_at = start_time
                    task_result.completed_at = datetime.now()
                    break
                except asyncio.TimeoutError:
                    if attempt == max_retries:
                        logger.error(f"Task {task_id} timed out after {timeout} seconds after {max_retries + 1} attempts")
                        task_result.status = TaskStatus.FAILED
                        task_result.error = f"Task timed out after {timeout} seconds"
                        task_result.completed_at = datetime.now()
                    else:
                        wait_time = (2 ** attempt) + (attempt * 0.1)  # Exponential backoff with jitter
                        logger.warning(f"Task {task_id} attempt {attempt + 1} timed out, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Task {task_id} failed with error: {e} after {max_retries + 1} attempts")
                        task_result.status = TaskStatus.FAILED
                        task_result.error = str(e)
                        task_result.completed_at = datetime.now()
                    else:
                        wait_time = (2 ** attempt) + (attempt * 0.1)
                        logger.warning(f"Task {task_id} attempt {attempt + 1} failed, retrying in {wait_time}s")
                        await asyncio.sleep(wait_time)

        except Exception as e:
            logger.error(f"Failed to execute task {task_id}: {e}")
            task_result.status = TaskStatus.FAILED
            task_result.error = str(e)
            task_result.completed_at = datetime.now()

        # Store the result
        self._completed_tasks[task_id] = task_result

        duration = (task_result.completed_at - start_time).total_seconds() if task_result.completed_at else 0
        logger.info(f"Task {task_id} completed in {duration:.2f}s with status: {task_result.status}")

        return task_result

    async def execute_tasks_concurrent(
        self,
        tasks: List[TaskSpec],
        context: ContextData,
        workflow_id: str
    ) -> Dict[str, TaskResult]:
        """Execute multiple tasks concurrently, respecting dependencies.

        Args:
            tasks: List of tasks to execute.
            context: Execution context.
            workflow_id: ID of the parent workflow.

        Returns:
            Dictionary mapping task IDs to their results.
        """
        # Build dependency graph
        self._build_dependency_graph(tasks)

        # Execute tasks in dependency order
        results = {}
        pending_tasks = {task.id: task for task in tasks}

        while pending_tasks:
            # Find tasks with satisfied dependencies
            ready_tasks = []
            for task_id, task in pending_tasks.items():
                dependencies = self._task_dependencies.get(task_id, set())
                if all(dep_id in results for dep_id in dependencies):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Circular dependency or unsatisfied dependencies
                unsatisfied = {}
                for task_id, deps in self._task_dependencies.items():
                    if task_id in pending_tasks:
                        unsatisfied_deps = deps - set(results.keys())
                        if unsatisfied_deps:
                            unsatisfied[task_id] = unsatisfied_deps

                raise RuntimeError(f"Unresolved task dependencies: {unsatisfied}")

            # Execute ready tasks concurrently
            execution_tasks = [
                self.execute_task(task, context, workflow_id)
                for task in ready_tasks
            ]

            # Wait for all ready tasks to complete
            completed_results = await asyncio.gather(*execution_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(completed_results):
                task = ready_tasks[i]
                if isinstance(result, Exception):
                    # Task failed with exception
                    error_result = TaskResult(
                        task_id=task.id,
                        status=TaskStatus.FAILED,
                        error=str(result),
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                    )
                    results[task.id] = error_result
                else:
                    results[task.id] = result

                # Remove from pending
                del pending_tasks[task.id]

        return results

    def _build_dependency_graph(self, tasks: List[TaskSpec]) -> None:
        """Build the dependency graph for tasks.

        Args:
            tasks: List of tasks to analyze.
        """
        self._task_dependencies.clear()

        for task in tasks:
            self._task_dependencies[task.id] = set(task.dependencies)

    def get_running_tasks(self) -> List[str]:
        """Get list of currently running task IDs.

        Returns:
            List of running task IDs.
        """
        return list(self._running_tasks.keys())

    def get_completed_tasks(self, workflow_id: Optional[str] = None) -> Dict[str, TaskResult]:
        """Get completed tasks, optionally filtered by workflow.

        Args:
            workflow_id: Optional workflow ID to filter by.

        Returns:
            Dictionary of completed task results.
        """
        if workflow_id is None:
            return self._completed_tasks.copy()

        # Filter by workflow_id if provided
        filtered = {}
        for task_id, result in self._completed_tasks.items():
            if hasattr(result, 'workflow_id') and result.workflow_id == workflow_id:
                filtered[task_id] = result

        return filtered

    def clear_completed_tasks(self, workflow_id: Optional[str] = None) -> None:
        """Clear completed tasks from memory.

        Args:
            workflow_id: Optional workflow ID to filter by.
        """
        if workflow_id is None:
            self._completed_tasks.clear()
        else:
            # Remove tasks for specific workflow
            to_remove = []
            for task_id, result in self._completed_tasks.items():
                if hasattr(result, 'workflow_id') and result.workflow_id == workflow_id:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._completed_tasks[task_id]

    async def shutdown(self) -> None:
        """Shutdown the task scheduler."""
        # Cancel all running tasks
        for task in self._running_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)

        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        logger.info("Task scheduler shutdown complete")
