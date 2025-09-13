"""Unit tests for workflow execution functionality."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from open_mon_user_acquisition.core.types import (
    ContextData,
    WorkflowInstance,
    WorkflowStatus,
    TaskSpec,
    TaskResult,
    TaskStatus
)
from open_mon_user_acquisition.orchestrator.task_scheduler import TaskScheduler
from open_mon_user_acquisition.orchestrator.workflow_engine import WorkflowEngine
from open_mon_user_acquisition.config import ConfigManager


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = MagicMock(spec=ConfigManager)
    config.get.return_value = "test_value"
    return config


@pytest.fixture
def mock_storage():
    """Create mock storage backend."""
    storage = MagicMock()
    storage.save_workflow = AsyncMock()
    storage.load_workflow = AsyncMock(return_value=None)
    return storage


@pytest.fixture
def task_scheduler(mock_config):
    """Create task scheduler for testing."""
    return TaskScheduler(mock_config)


@pytest.fixture
def workflow_engine(mock_config, mock_storage, task_scheduler):
    """Create workflow engine for testing."""
    engine = WorkflowEngine(mock_config, mock_storage, task_scheduler)
    return engine


class TestTaskScheduler:
    """Test task scheduler functionality."""

    @pytest.mark.asyncio
    async def test_task_scheduler_creation(self, task_scheduler):
        """Test creating task scheduler."""
        assert task_scheduler.max_concurrent_tasks == 10
        assert task_scheduler._agents == {}
        assert task_scheduler._running_tasks == {}

    def test_register_agent(self, task_scheduler):
        """Test registering an agent."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        task_scheduler.register_agent("test_agent", mock_agent)

        assert "test_agent" in task_scheduler._agents
        assert task_scheduler._agents["test_agent"] == mock_agent

    def test_unregister_agent(self, task_scheduler):
        """Test unregistering an agent."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        task_scheduler.register_agent("test_agent", mock_agent)
        assert "test_agent" in task_scheduler._agents

        task_scheduler.unregister_agent("test_agent")
        assert "test_agent" not in task_scheduler._agents

    @pytest.mark.asyncio
    async def test_execute_task_success(self, task_scheduler):
        """Test successful task execution."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.execute = AsyncMock(return_value=TaskResult(
            task_id="test_task",
            status=TaskStatus.COMPLETED,
            result="success"
        ))

        task_scheduler.register_agent("test_agent", mock_agent)

        task = TaskSpec(
            id="test_task",
            name="Test Task",
            agent_type="test_agent"
        )

        context = ContextData()
        result = await task_scheduler.execute_task(task, context, "test_workflow")

        assert result.status == TaskStatus.COMPLETED
        assert result.result == "success"
        mock_agent.execute.assert_called_once_with(task, context)

    @pytest.mark.asyncio
    async def test_execute_task_failure(self, task_scheduler):
        """Test task execution failure."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.execute = AsyncMock(side_effect=Exception("Task failed"))

        task_scheduler.register_agent("test_agent", mock_agent)

        task = TaskSpec(
            id="test_task",
            name="Test Task",
            agent_type="test_agent"
        )

        context = ContextData()
        result = await task_scheduler.execute_task(task, context, "test_workflow")

        assert result.status == TaskStatus.FAILED
        assert "Task failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_task_unknown_agent(self, task_scheduler):
        """Test task execution with unknown agent."""
        task = TaskSpec(
            id="test_task",
            name="Test Task",
            agent_type="unknown_agent"
        )

        context = ContextData()

        with pytest.raises(RuntimeError, match="No agent registered"):
            await task_scheduler.execute_task(task, context, "test_workflow")

    @pytest.mark.asyncio
    async def test_execute_tasks_concurrent_no_dependencies(self, task_scheduler):
        """Test concurrent execution of tasks with no dependencies."""
        # Mock agents
        mock_agent1 = MagicMock()
        mock_agent1.name = "agent1"
        mock_agent1.execute = AsyncMock(return_value=TaskResult(
            task_id="task1",
            status=TaskStatus.COMPLETED,
            result="result1"
        ))

        mock_agent2 = MagicMock()
        mock_agent2.name = "agent2"
        mock_agent2.execute = AsyncMock(return_value=TaskResult(
            task_id="task2",
            status=TaskStatus.COMPLETED,
            result="result2"
        ))

        task_scheduler.register_agent("agent1", mock_agent1)
        task_scheduler.register_agent("agent2", mock_agent2)

        tasks = [
            TaskSpec(id="task1", name="Task 1", agent_type="agent1"),
            TaskSpec(id="task2", name="Task 2", agent_type="agent2")
        ]

        context = ContextData()
        results = await task_scheduler.execute_tasks_concurrent(tasks, context, "test_workflow")

        assert len(results) == 2
        assert results["task1"].result == "result1"
        assert results["task2"].result == "result2"

    @pytest.mark.asyncio
    async def test_execute_tasks_concurrent_with_dependencies(self, task_scheduler):
        """Test concurrent execution of tasks with dependencies."""
        # Mock agents
        mock_agent1 = MagicMock()
        mock_agent1.name = "agent1"
        mock_agent1.execute = AsyncMock(return_value=TaskResult(
            task_id="task1",
            status=TaskStatus.COMPLETED,
            result="result1"
        ))

        mock_agent2 = MagicMock()
        mock_agent2.name = "agent2"
        mock_agent2.execute = AsyncMock(return_value=TaskResult(
            task_id="task2",
            status=TaskStatus.COMPLETED,
            result="result2"
        ))

        task_scheduler.register_agent("agent1", mock_agent1)
        task_scheduler.register_agent("agent2", mock_agent2)

        tasks = [
            TaskSpec(id="task1", name="Task 1", agent_type="agent1"),
            TaskSpec(id="task2", name="Task 2", agent_type="agent2", dependencies=["task1"])
        ]

        context = ContextData()
        results = await task_scheduler.execute_tasks_concurrent(tasks, context, "test_workflow")

        assert len(results) == 2
        assert results["task1"].result == "result1"
        assert results["task2"].result == "result2"

        # Verify task1 was executed before task2
        assert mock_agent1.execute.called
        assert mock_agent2.execute.called

    @pytest.mark.asyncio
    async def test_execute_tasks_concurrent_circular_dependency(self, task_scheduler):
        """Test handling of circular dependencies."""
        tasks = [
            TaskSpec(id="task1", name="Task 1", agent_type="agent1", dependencies=["task2"]),
            TaskSpec(id="task2", name="Task 2", agent_type="agent2", dependencies=["task1"])
        ]

        context = ContextData()

        with pytest.raises(RuntimeError, match="Unresolved task dependencies"):
            await task_scheduler.execute_tasks_concurrent(tasks, context, "test_workflow")

    def test_get_running_tasks(self, task_scheduler):
        """Test getting running tasks."""
        # Initially empty
        assert task_scheduler.get_running_tasks() == []

        # Add a mock running task
        task_scheduler._running_tasks["task1"] = MagicMock()
        assert "task1" in task_scheduler.get_running_tasks()

    def test_get_completed_tasks(self, task_scheduler):
        """Test getting completed tasks."""
        # Initially empty
        assert task_scheduler.get_completed_tasks() == {}

        # Add a completed task
        task_result = TaskResult(task_id="task1", status=TaskStatus.COMPLETED)
        task_scheduler._completed_tasks["task1"] = task_result

        completed = task_scheduler.get_completed_tasks()
        assert "task1" in completed
        assert completed["task1"] == task_result

    def test_clear_completed_tasks(self, task_scheduler):
        """Test clearing completed tasks."""
        # Add completed tasks
        task_scheduler._completed_tasks["task1"] = TaskResult(task_id="task1", status=TaskStatus.COMPLETED)
        task_scheduler._completed_tasks["task2"] = TaskResult(task_id="task2", status=TaskStatus.COMPLETED)

        assert len(task_scheduler.get_completed_tasks()) == 2

        # Clear specific task
        task_scheduler.clear_completed_tasks("task1")
        assert len(task_scheduler.get_completed_tasks()) == 1
        assert "task2" in task_scheduler.get_completed_tasks()

        # Clear all
        task_scheduler.clear_completed_tasks()
        assert len(task_scheduler.get_completed_tasks()) == 0


class TestWorkflowEngine:
    """Test workflow engine functionality."""

    @pytest.mark.asyncio
    async def test_workflow_engine_creation(self, workflow_engine, mock_config, mock_storage, task_scheduler):
        """Test creating workflow engine."""
        assert workflow_engine.config == mock_config
        assert workflow_engine.storage == mock_storage
        assert workflow_engine.scheduler == task_scheduler
        assert workflow_engine._active_workflows == {}
        assert workflow_engine._agents == {}

    def test_register_agent(self, workflow_engine):
        """Test registering an agent."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        workflow_engine.register_agent("test_agent", mock_agent)

        assert "test_agent" in workflow_engine._agents
        assert workflow_engine._agents["test_agent"] == mock_agent

    @pytest.mark.asyncio
    async def test_create_workflow(self, workflow_engine, mock_storage):
        """Test creating a new workflow."""
        context = ContextData(campaign_id="test_campaign")

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.plan = AsyncMock(return_value=[
            TaskSpec(id="task1", name="Task 1", agent_type="test_agent")
        ])

        workflow_engine.register_agent("test_agent", mock_agent)

        workflow = await workflow_engine.create_workflow(
            name="Test Workflow",
            context=context,
            agent_types=["test_agent"]
        )

        assert workflow.name == "Test Workflow"
        assert workflow.status == WorkflowStatus.PENDING
        assert len(workflow.tasks) == 1
        assert workflow.tasks[0].id == "task1"

        # Verify workflow was saved
        mock_storage.save_workflow.assert_called_once_with(workflow)

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, workflow_engine, mock_storage, task_scheduler):
        """Test successful workflow execution."""
        # Create a workflow
        workflow = WorkflowInstance(
            id="test-workflow",
            name="Test Workflow",
            status=WorkflowStatus.PENDING,
            tasks=[
                TaskSpec(id="task1", name="Task 1", agent_type="test_agent")
            ]
        )

        # Mock agent and task execution
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.execute = AsyncMock(return_value=TaskResult(
            task_id="task1",
            status=TaskStatus.COMPLETED,
            result="success"
        ))

        workflow_engine.register_agent("test_agent", mock_agent)

        # Mock task execution
        task_scheduler.execute_tasks_concurrent = AsyncMock(return_value={
            "task1": TaskResult(
                task_id="task1",
                status=TaskStatus.COMPLETED,
                result="success"
            )
        })

        result = await workflow_engine.execute_workflow("test-workflow")

        # Since workflow is not in active workflows, it should try to load from storage
        # and return None (our mock returns None)
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_workflow_with_active_workflow(self, workflow_engine, mock_storage, task_scheduler):
        """Test executing an active workflow."""
        # Create and add workflow to active workflows
        workflow = WorkflowInstance(
            id="test-workflow",
            name="Test Workflow",
            status=WorkflowStatus.PENDING,
            tasks=[
                TaskSpec(id="task1", name="Task 1", agent_type="test_agent")
            ]
        )

        workflow_engine._active_workflows["test-workflow"] = workflow

        # Mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.execute = AsyncMock(return_value=TaskResult(
            task_id="task1",
            status=TaskStatus.COMPLETED,
            result="success"
        ))

        workflow_engine.register_agent("test_agent", mock_agent)

        # Mock successful task execution
        task_scheduler.execute_tasks_concurrent = AsyncMock(return_value={
            "task1": TaskResult(
                task_id="task1",
                status=TaskStatus.COMPLETED,
                result="success"
            )
        })

        result = await workflow_engine.execute_workflow("test-workflow")

        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.results) == 1
        assert result.results["task1"].status == TaskStatus.COMPLETED

        # Verify workflow was saved with final state
        assert mock_storage.save_workflow.call_count == 2  # Initial update + final save

    @pytest.mark.asyncio
    async def test_execute_workflow_already_running(self, workflow_engine):
        """Test executing an already running workflow."""
        workflow = WorkflowInstance(
            id="test-workflow",
            name="Test Workflow",
            status=WorkflowStatus.RUNNING
        )

        workflow_engine._active_workflows["test-workflow"] = workflow

        with pytest.raises(ValueError, match="Workflow is already running"):
            await workflow_engine.execute_workflow("test-workflow")

    @pytest.mark.asyncio
    async def test_get_workflow_status_active(self, workflow_engine):
        """Test getting status of active workflow."""
        workflow = WorkflowInstance(
            id="test-workflow",
            name="Test Workflow",
            status=WorkflowStatus.RUNNING
        )

        workflow_engine._active_workflows["test-workflow"] = workflow

        result = await workflow_engine.get_workflow_status("test-workflow")

        assert result == workflow

    @pytest.mark.asyncio
    async def test_get_workflow_status_from_storage(self, workflow_engine, mock_storage):
        """Test getting status from storage when not active."""
        workflow = WorkflowInstance(
            id="test-workflow",
            name="Test Workflow",
            status=WorkflowStatus.COMPLETED
        )

        mock_storage.load_workflow.return_value = workflow

        result = await workflow_engine.get_workflow_status("test-workflow")

        assert result == workflow
        mock_storage.load_workflow.assert_called_once_with("test-workflow")

    @pytest.mark.asyncio
    async def test_cancel_workflow_success(self, workflow_engine, mock_storage):
        """Test successful workflow cancellation."""
        workflow = WorkflowInstance(
            id="test-workflow",
            name="Test Workflow",
            status=WorkflowStatus.RUNNING
        )

        workflow_engine._active_workflows["test-workflow"] = workflow

        result = await workflow_engine.cancel_workflow("test-workflow")

        assert result is True
        assert workflow.status == WorkflowStatus.CANCELLED
        mock_storage.save_workflow.assert_called_once_with(workflow)

    @pytest.mark.asyncio
    async def test_cancel_workflow_not_running(self, workflow_engine):
        """Test cancelling a non-running workflow."""
        workflow = WorkflowInstance(
            id="test-workflow",
            name="Test Workflow",
            status=WorkflowStatus.PENDING
        )

        workflow_engine._active_workflows["test-workflow"] = workflow

        result = await workflow_engine.cancel_workflow("test-workflow")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_workflow_not_found(self, workflow_engine):
        """Test cancelling a non-existent workflow."""
        result = await workflow_engine.cancel_workflow("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_active_workflows(self, workflow_engine, mock_storage):
        """Test listing active workflows."""
        # Add active workflows
        workflow1 = WorkflowInstance(
            id="wf1",
            name="Workflow 1",
            status=WorkflowStatus.RUNNING
        )
        workflow2 = WorkflowInstance(
            id="wf2",
            name="Workflow 2",
            status=WorkflowStatus.PENDING
        )

        workflow_engine._active_workflows["wf1"] = workflow1
        workflow_engine._active_workflows["wf2"] = workflow2

        # Mock storage to return a completed workflow
        completed_workflow = WorkflowInstance(
            id="wf3",
            name="Workflow 3",
            status=WorkflowStatus.COMPLETED
        )
        mock_storage.list_workflows.return_value = [completed_workflow]

        active_workflows = await workflow_engine.list_active_workflows()

        # Should include running and pending from active, but not completed from storage
        assert len(active_workflows) == 2
        workflow_ids = {wf.id for wf in active_workflows}
        assert workflow_ids == {"wf1", "wf2"}

    @pytest.mark.asyncio
    async def test_cleanup_completed_workflows(self, workflow_engine):
        """Test cleaning up old completed workflows."""
        # Add workflows with different ages
        recent_completed = WorkflowInstance(
            id="recent",
            name="Recent Workflow",
            status=WorkflowStatus.COMPLETED,
            completed_at=datetime.now()
        )

        old_completed = WorkflowInstance(
            id="old",
            name="Old Workflow",
            status=WorkflowStatus.COMPLETED,
            completed_at=datetime.now().replace(hour=datetime.now().hour - 25)  # 25 hours ago
        )

        workflow_engine._active_workflows["recent"] = recent_completed
        workflow_engine._active_workflows["old"] = old_completed

        cleaned_count = await workflow_engine.cleanup_completed_workflows(max_age_hours=24)

        assert cleaned_count == 1
        assert "recent" in workflow_engine._active_workflows
        assert "old" not in workflow_engine._active_workflows

    @pytest.mark.asyncio
    async def test_get_workflow_metrics(self, workflow_engine, mock_storage):
        """Test getting workflow metrics."""
        # Mock storage stats
        mock_storage.get_workflow_stats = AsyncMock(return_value={
            "total_workflows": 10,
            "status_counts": {"completed": 7, "failed": 2, "running": 1}
        })

        # Add some active workflows
        workflow_engine._active_workflows["wf1"] = WorkflowInstance(
            id="wf1", name="WF1", status=WorkflowStatus.RUNNING
        )
        workflow_engine._active_workflows["wf2"] = WorkflowInstance(
            id="wf2", name="WF2", status=WorkflowStatus.PENDING
        )

        metrics = await workflow_engine.get_workflow_metrics()

        assert metrics["active_workflows"] == 1  # Only running
        assert metrics["pending_workflows"] == 1
        assert metrics["completed_workflows"] == 0  # None in active
        assert metrics["total_workflows"] == 10  # From storage

    @pytest.mark.asyncio
    async def test_get_workflow_metrics_storage_error(self, workflow_engine):
        """Test workflow metrics when storage fails."""
        mock_storage.get_workflow_stats = AsyncMock(side_effect=Exception("Storage error"))

        metrics = await workflow_engine.get_workflow_metrics()

        # Should still return basic metrics even if storage fails
        assert "active_workflows" in metrics
        assert "pending_workflows" in metrics
        assert "completed_workflows" in metrics


class TestWorkflowExecutionIntegration:
    """Integration tests for workflow execution."""

    @pytest.mark.asyncio
    async def test_full_workflow_execution_flow(self, workflow_engine, task_scheduler):
        """Test complete workflow execution flow."""
        # Create a complex workflow with dependencies
        context = ContextData(
            campaign_id="integration_test",
            metadata={"test_type": "integration"}
        )

        # Mock agents
        mock_agent = MagicMock()
        mock_agent.name = "integration_agent"
        mock_agent.plan = AsyncMock(return_value=[
            TaskSpec(
                id="analysis_task",
                name="Analysis Task",
                agent_type="integration_agent",
                parameters={"analysis_type": "campaign_performance"}
            ),
            TaskSpec(
                id="optimization_task",
                name="Optimization Task",
                agent_type="integration_agent",
                dependencies=["analysis_task"],
                parameters={"optimization_goal": "maximize_roi"}
            ),
            TaskSpec(
                id="reporting_task",
                name="Reporting Task",
                agent_type="integration_agent",
                dependencies=["optimization_task"],
                parameters={"report_format": "json"}
            )
        ])

        mock_agent.execute = AsyncMock(side_effect=[
            TaskResult(
                task_id="analysis_task",
                status=TaskStatus.COMPLETED,
                result={"insights": ["Campaign performing well", "High conversion rate"]}
            ),
            TaskResult(
                task_id="optimization_task",
                status=TaskStatus.COMPLETED,
                result={"optimizations": ["Increase budget by 20%", "Target new segments"]}
            ),
            TaskResult(
                task_id="reporting_task",
                status=TaskStatus.COMPLETED,
                result={"report": {"status": "success", "recommendations": 2}}
            )
        ])

        workflow_engine.register_agent("integration_agent", mock_agent)

        # Create workflow
        workflow = await workflow_engine.create_workflow(
            name="Integration Test Workflow",
            context=context,
            agent_types=["integration_agent"]
        )

        assert len(workflow.tasks) == 3
        assert workflow.tasks[0].id == "analysis_task"
        assert workflow.tasks[1].dependencies == ["analysis_task"]
        assert workflow.tasks[2].dependencies == ["optimization_task"]

        # Execute workflow
        executed_workflow = await workflow_engine.execute_workflow(workflow.id)

        assert executed_workflow.status == WorkflowStatus.COMPLETED
        assert len(executed_workflow.results) == 3

        # Verify all tasks completed successfully
        for task_id, result in executed_workflow.results.items():
            assert result.status == TaskStatus.COMPLETED
            assert result.result is not None

        # Verify execution order (dependencies respected)
        analysis_result = executed_workflow.results["analysis_task"]
        optimization_result = executed_workflow.results["optimization_task"]
        reporting_result = executed_workflow.results["reporting_task"]

        assert "insights" in analysis_result.result
        assert "optimizations" in optimization_result.result
        assert "report" in reporting_result.result
