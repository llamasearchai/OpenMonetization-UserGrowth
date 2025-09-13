"""Unit tests for workflow test_orchestrator."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from open_mon_user_acquisition.orchestrator import WorkflowOrchestrator
from open_mon_user_acquisition.orchestrator.task_scheduler import TaskScheduler
from open_mon_user_acquisition.orchestrator.workflow_engine import WorkflowEngine
from open_mon_user_acquisition.core.types import (
    ContextData,
    WorkflowInstance,
    WorkflowStatus,
    TaskSpec,
    TaskResult,
    TaskStatus
)
from open_mon_user_acquisition.config import ConfigManager


@pytest.fixture
def mock_config():
    """Create mock configuration manager."""
    config = MagicMock(spec=ConfigManager)
    config.get.return_value = "default_value"
    return config


@pytest.fixture
def mock_storage():
    """Create mock storage backend."""
    storage = MagicMock()
    storage.initialize = AsyncMock()
    storage.close = AsyncMock()
    storage.save_workflow = AsyncMock()
    storage.load_workflow = AsyncMock(return_value=None)
    storage.list_workflows = AsyncMock(return_value=[])
    return storage


@pytest.fixture
def mock_llm_backend():
    """Create mock LLM backend."""
    llm = MagicMock()
    llm.validate_connection = AsyncMock(return_value=True)
    llm.generate = AsyncMock(return_value=MagicMock(content="Test response"))
    return llm


# Fixture now defined in conftest.py

class TestWorkflowOrchestratorInitialization:
    """Test test_orchestrator initialization."""

    @pytest.mark.asyncio
    async def test_test_orchestrator_creation(self, mock_config, mock_storage, mock_llm_backend):
        """Test creating test_orchestrator instance."""
        orch = WorkflowOrchestrator(
            config=mock_config,
            storage=mock_storage,
            llm_backend=mock_llm_backend
        )

        assert orch.config == mock_config
        assert orch.storage == mock_storage
        assert orch.llm_backend == mock_llm_backend
        assert orch._initialized is False

    @pytest.mark.asyncio
    async def test_test_orchestrator_initialization(self, test_orchestrator, mock_storage, mock_llm_backend):
        """Test test_orchestrator initialization."""
        # Verify initialization calls
        mock_storage.initialize.assert_called_once()
        mock_llm_backend.validate_connection.assert_called_once()

        assert test_orchestrator._initialized is True

    @pytest.mark.asyncio
    async def test_test_orchestrator_double_initialization(self, test_orchestrator, mock_storage):
        """Test that double initialization doesn't cause issues."""
        # Second initialization should be a no-op
        await test_orchestrator.initialize()

        # Should still only be called once
        mock_storage.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_orchestrator_shutdown(self, test_orchestrator, mock_storage):
        """Test test_orchestrator shutdown."""
        await test_orchestrator.shutdown()

        mock_storage.close.assert_called_once()
        assert test_orchestrator._initialized is False

    @pytest.mark.asyncio
    async def test_test_orchestrator_double_shutdown(self, test_orchestrator, mock_storage):
        """Test that double shutdown doesn't cause issues."""
        await test_orchestrator.shutdown()
        await test_orchestrator.shutdown()

        # Should still only be called once
        mock_storage.close.assert_called_once()


class TestWorkflowOrchestratorWorkflowManagement:
    """Test workflow management operations."""

    @pytest.mark.asyncio
    async def test_create_workflow(self, test_orchestrator, mock_storage):
        """Test creating a new workflow."""
        context = ContextData(campaign_id="test_campaign")
        mock_workflow = WorkflowInstance(
            id="test-workflow-123",
            name="Test Workflow",
            status=WorkflowStatus.PENDING
        )

        # Mock the workflow engine
        test_orchestrator.workflow_engine.create_workflow = AsyncMock(return_value=mock_workflow)

        result = await test_orchestrator.create_workflow(
            name="Test Workflow",
            tasks=[],
            context=context
        )

        assert result == mock_workflow
        test_orchestrator.workflow_engine.create_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_workflow(self, test_orchestrator):
        """Test executing a workflow."""
        mock_workflow = WorkflowInstance(
            id="test-workflow-123",
            name="Test Workflow",
            status=WorkflowStatus.COMPLETED
        )

        test_orchestrator.workflow_engine.execute_workflow = AsyncMock(return_value=mock_workflow)

        result = await test_orchestrator.execute_workflow("test-workflow-123")

        assert result == mock_workflow
        test_orchestrator.workflow_engine.execute_workflow.assert_called_once_with("test-workflow-123")

    @pytest.mark.asyncio
    async def test_get_workflow_status(self, test_orchestrator):
        """Test getting workflow status."""
        mock_workflow = WorkflowInstance(
            id="test-workflow-123",
            name="Test Workflow",
            status=WorkflowStatus.RUNNING
        )

        test_orchestrator.workflow_engine.get_workflow_status = AsyncMock(return_value=mock_workflow)

        result = await test_orchestrator.get_workflow_status("test-workflow-123")

        assert result == mock_workflow
        test_orchestrator.workflow_engine.get_workflow_status.assert_called_once_with("test-workflow-123")

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, test_orchestrator):
        """Test cancelling a workflow."""
        test_orchestrator.workflow_engine.cancel_workflow = AsyncMock(return_value=True)

        result = await test_orchestrator.cancel_workflow("test-workflow-123")

        assert result is True
        test_orchestrator.workflow_engine.cancel_workflow.assert_called_once_with("test-workflow-123")

    @pytest.mark.asyncio
    async def test_list_active_workflows(self, test_orchestrator):
        """Test listing active workflows."""
        mock_workflows = [
            WorkflowInstance(id="wf1", name="Workflow 1", status=WorkflowStatus.RUNNING),
            WorkflowInstance(id="wf2", name="Workflow 2", status=WorkflowStatus.PENDING)
        ]

        test_orchestrator.workflow_engine.list_active_workflows = AsyncMock(return_value=mock_workflows)

        result = await test_orchestrator.list_active_workflows()

        assert result == mock_workflows
        test_orchestrator.workflow_engine.list_active_workflows.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_and_execute_workflow(self, test_orchestrator):
        """Test create and execute workflow convenience method."""
        context = ContextData(campaign_id="test_campaign")

        mock_created_workflow = WorkflowInstance(
            id="test-workflow-123",
            name="Test Workflow",
            status=WorkflowStatus.PENDING
        )

        mock_executed_workflow = WorkflowInstance(
            id="test-workflow-123",
            name="Test Workflow",
            status=WorkflowStatus.COMPLETED
        )

        test_orchestrator.workflow_engine.create_workflow = AsyncMock(return_value=mock_created_workflow)
        test_orchestrator.workflow_engine.execute_workflow = AsyncMock(return_value=mock_executed_workflow)

        result = await test_orchestrator.create_and_execute_workflow(
            name="Test Workflow",
            context=context
        )

        assert result == mock_executed_workflow
        test_orchestrator.workflow_engine.create_workflow.assert_called_once()
        test_orchestrator.workflow_engine.execute_workflow.assert_called_once_with("test-workflow-123")


class TestWorkflowOrchestratorSystemStatus:
    """Test system status and monitoring."""

    @pytest.mark.asyncio
    async def test_get_system_status(self, test_orchestrator, mock_storage):
        """Test getting system status."""
        # Mock workflow metrics
        test_orchestrator.workflow_engine.get_workflow_metrics = AsyncMock(return_value={
            "active_workflows": 2,
            "completed_workflows": 5
        })

        status = await test_orchestrator.get_system_status()

        assert status["initialized"] is True
        assert "timestamp" in status
        assert "components" in status
        assert "workflow_metrics" in status
        assert status["workflow_metrics"]["active_workflows"] == 2

    @pytest.mark.asyncio
    async def test_get_system_status_with_errors(self, test_orchestrator):
        """Test system status when workflow metrics fail."""
        test_orchestrator.workflow_engine.get_workflow_metrics = AsyncMock(side_effect=Exception("Test error"))

        status = await test_orchestrator.get_system_status()

        assert status["initialized"] is True
        assert "error" in status["workflow_metrics"]

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, test_orchestrator):
        """Test cleaning up old data."""
        cleanup_stats = {"workflows_cleaned": 3, "metrics_cleaned": 10}
        test_orchestrator.workflow_engine.cleanup_completed_workflows = AsyncMock(return_value=3)

        result = await test_orchestrator.cleanup_old_data(days_to_keep=30)

        assert result["workflows_cleaned"] == 3
        test_orchestrator.workflow_engine.cleanup_completed_workflows.assert_called_once()


class TestWorkflowOrchestratorAgentManagement:
    """Test agent registration and management."""

    @pytest.mark.asyncio
    async def test_register_agent(self, test_orchestrator):
        """Test registering an agent."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"

        # Mock the workflow engine register_agent method
        with patch.object(test_orchestrator.workflow_engine, 'register_agent') as mock_engine_register:

            test_orchestrator.register_agent("test_agent", mock_agent)

            # Verify agent was registered in the workflow engine
            mock_engine_register.assert_called_once_with("test_agent", mock_agent)

    @pytest.mark.asyncio
    async def test_get_available_agents(self, test_orchestrator):
        """Test getting available agents."""
        test_orchestrator.workflow_engine._agents = {"agent1": MagicMock(), "agent2": MagicMock()}

        agents = test_orchestrator.get_available_agents()

        assert set(agents) == {"agent1", "agent2"}


class TestWorkflowOrchestratorErrorHandling:
    """Test error handling in test_orchestrator."""

    @pytest.mark.asyncio
    async def test_operations_before_initialization(self):
        """Test operations work before explicit initialization (auto-init)."""
        orch = WorkflowOrchestrator()

        # Should work because orchestrator auto-initializes
        workflow = await orch.create_workflow("test", [], ContextData())
        assert workflow is not None
        assert workflow.name == "test"

    @pytest.mark.asyncio
    async def test_workflow_creation_failure(self, test_orchestrator):
        """Test handling workflow creation failure."""
        test_orchestrator.workflow_engine.create_workflow = AsyncMock(side_effect=Exception("Creation failed"))

        with pytest.raises(Exception, match="Creation failed"):
            await test_orchestrator.create_workflow("Test", [], ContextData())

    @pytest.mark.asyncio
    async def test_workflow_execution_failure(self, test_orchestrator):
        """Test handling workflow execution failure."""
        test_orchestrator.workflow_engine.execute_workflow = AsyncMock(side_effect=Exception("Execution failed"))

        with pytest.raises(Exception, match="Execution failed"):
            await test_orchestrator.execute_workflow("test-id")

    @pytest.mark.asyncio
    async def test_storage_initialization_failure(self, mock_config, mock_llm_backend):
        """Test handling storage initialization failure."""
        mock_storage = MagicMock()
        mock_storage.initialize = AsyncMock(side_effect=Exception("Storage init failed"))

        orch = WorkflowOrchestrator(
            config=mock_config,
            storage=mock_storage,
            llm_backend=mock_llm_backend
        )

        with pytest.raises(Exception, match="Storage init failed"):
            await orch.initialize()

    @pytest.mark.asyncio
    async def test_llm_validation_failure(self, test_orchestrator, mock_llm_backend):
        """Test handling LLM backend validation failure."""
        mock_llm_backend.validate_connection = AsyncMock(side_effect=Exception("LLM validation failed"))

        # Should not raise exception, just log warning
        await test_orchestrator.initialize()

        assert test_orchestrator._initialized is True


class TestWorkflowOrchestratorConcurrency:
    """Test concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_workflow_operations(self, test_orchestrator):
        """Test running multiple workflow operations concurrently."""
        async def create_workflow_operation(i):
            context = ContextData(campaign_id=f"campaign_{i}")
            return await test_orchestrator.create_workflow(
                name=f"Workflow {i}",
                tasks=[],
                context=context
            )

        # Mock the workflow engine to handle concurrent calls
        test_orchestrator.workflow_engine.create_workflow = AsyncMock(side_effect=[
            WorkflowInstance(id=f"wf_{i}", name=f"Workflow {i}", status=WorkflowStatus.PENDING)
            for i in range(3)
        ])

        # Run multiple operations concurrently
        tasks = [create_workflow_operation(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(isinstance(wf, WorkflowInstance) for wf in results)
        assert test_orchestrator.workflow_engine.create_workflow.call_count == 3

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.open_mon_user_acquisition.orchestrator.task_scheduler import TaskScheduler
from src.open_mon_user_acquisition.core.types import TaskSpec, TaskStatus, ContextData
from src.open_mon_user_acquisition.config import ConfigManager

@pytest.mark.asyncio
async def test_task_retry_on_failure():
    # Mock config with retries
    config = MagicMock()
    config.get.return_value = 2  # 2 retries

    scheduler = TaskScheduler(config)

    # Mock agent that fails first 2 times, succeeds on 3rd
    from open_mon_user_acquisition.core.types import TaskResult, TaskStatus

    agent = AsyncMock()
    agent.execute.side_effect = [
        Exception("First failure"),
        Exception("Second failure"),
        TaskResult(
            task_id="test_task",
            status=TaskStatus.COMPLETED,
            result={"result": "Success"}
        )  # Success on third attempt
    ]

    # Register mock agent
    scheduler.register_agent("test_agent", agent)

    # Create task
    task = TaskSpec(
        id="test_task",
        name="Test Task",
        agent_type="test_agent",
        retry_count=2
    )

    context = ContextData()
    workflow_id = "test_workflow"

    # Execute task
    result = await scheduler.execute_task(task, context, workflow_id)

    # Verify success after retries
    assert result.status == TaskStatus.COMPLETED
    assert result.result == {"result": "Success"}
    assert agent.execute.call_count == 3  # Called 3 times (1 success + 2 failures)

    # Verify config was called
    config.get.assert_called_with("workflow.retry_attempts", 3)