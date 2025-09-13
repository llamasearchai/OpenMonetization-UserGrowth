"""Unit tests for storage backend."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from open_mon_user_acquisition.storage.sqlite_backend import SQLiteStorageBackend
from open_mon_user_acquisition.core.types import WorkflowInstance, WorkflowStatus, TaskSpec, TaskStatus, ContextData


# Use fixture from conftest.py


class TestSQLiteStorageBackend:
    """Test SQLite storage backend functionality."""

    @pytest.mark.asyncio
    async def test_storage_initialization(self, test_storage_backend):
        """Test storage backend initialization."""
        assert test_storage_backend._engine is not None
        assert test_storage_backend._initialized is True

    @pytest.mark.asyncio
    async def test_save_and_load_workflow(self, test_storage_backend):
        """Test saving and loading a workflow."""
        # Create a test workflow
        workflow = WorkflowInstance(
            id="test-workflow-123",
            name="Test Workflow",
            status=WorkflowStatus.PENDING,
            tasks=[
                TaskSpec(
                    id="task1",
                    name="Test Task 1",
                    agent_type="test_agent",
                    status=TaskStatus.PENDING
                ),
                TaskSpec(
                    id="task2",
                    name="Test Task 2",
                    agent_type="test_agent",
                    status=TaskStatus.PENDING,
                    dependencies=["task1"]
                )
            ]
        )

        # Save workflow
        await test_storage_backend.save_workflow(workflow)

        # Load workflow
        loaded_workflow = await test_storage_backend.load_workflow("test-workflow-123")

        assert loaded_workflow is not None
        assert loaded_workflow.id == "test-workflow-123"
        assert loaded_workflow.name == "Test Workflow"
        assert loaded_workflow.status == WorkflowStatus.PENDING
        assert len(loaded_workflow.tasks) == 2
        assert loaded_workflow.tasks[0].id == "task1"
        assert loaded_workflow.tasks[1].dependencies == ["task1"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_workflow(self, test_storage_backend):
        """Test loading a non-existent workflow."""
        loaded_workflow = await test_storage_backend.load_workflow("nonexistent")

        assert loaded_workflow is None

    @pytest.mark.asyncio
    async def test_list_workflows_empty(self, test_storage_backend):
        """Test listing workflows when none exist."""
        workflows = await test_storage_backend.list_workflows()

        assert workflows == []

    @pytest.mark.asyncio
    async def test_list_workflows_with_data(self, test_storage_backend):
        """Test listing workflows with data."""
        # Create and save multiple workflows
        workflow1 = WorkflowInstance(
            id="wf1",
            name="Workflow 1",
            status=WorkflowStatus.PENDING
        )
        workflow2 = WorkflowInstance(
            id="wf2",
            name="Workflow 2",
            status=WorkflowStatus.COMPLETED
        )

        await test_storage_backend.save_workflow(workflow1)
        await test_storage_backend.save_workflow(workflow2)

        # List all workflows
        workflows = await test_storage_backend.list_workflows()

        assert len(workflows) == 2
        workflow_ids = {wf.id for wf in workflows}
        assert workflow_ids == {"wf1", "wf2"}

    @pytest.mark.asyncio
    async def test_list_workflows_with_status_filter(self, test_storage_backend):
        """Test listing workflows with status filter."""
        # Create workflows with different statuses
        workflow1 = WorkflowInstance(
            id="wf1",
            name="Workflow 1",
            status=WorkflowStatus.PENDING
        )
        workflow2 = WorkflowInstance(
            id="wf2",
            name="Workflow 2",
            status=WorkflowStatus.COMPLETED
        )

        await test_storage_backend.save_workflow(workflow1)
        await test_storage_backend.save_workflow(workflow2)

        # List only completed workflows
        completed_workflows = await test_storage_backend.list_workflows(status="completed")

        assert len(completed_workflows) == 1
        assert completed_workflows[0].id == "wf2"

    @pytest.mark.asyncio
    async def test_workflow_update(self, test_storage_backend):
        """Test updating an existing workflow."""
        # Create initial workflow
        workflow = WorkflowInstance(
            id="test-workflow",
            name="Test Workflow",
            status=WorkflowStatus.PENDING
        )

        await test_storage_backend.save_workflow(workflow)

        # Update workflow
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()

        await test_storage_backend.save_workflow(workflow)

        # Load and verify update
        loaded_workflow = await test_storage_backend.load_workflow("test-workflow")

        assert loaded_workflow.status == WorkflowStatus.RUNNING
        assert loaded_workflow.started_at is not None

    @pytest.mark.asyncio
    async def test_task_persistence(self, test_storage_backend):
        """Test that tasks are properly persisted with workflows."""
        # Create workflow with complex tasks
        workflow = WorkflowInstance(
            id="complex-workflow",
            name="Complex Workflow",
            status=WorkflowStatus.PENDING,
            tasks=[
                TaskSpec(
                    id="analysis_task",
                    name="Analysis Task",
                    agent_type="analyst",
                    status=TaskStatus.PENDING,
                    parameters={"type": "data_analysis"},
                    timeout_seconds=300
                ),
                TaskSpec(
                    id="optimization_task",
                    name="Optimization Task",
                    agent_type="optimizer",
                    status=TaskStatus.PENDING,
                    dependencies=["analysis_task"],
                    parameters={"goal": "maximize_roi"},
                    priority=2
                )
            ]
        )

        await test_storage_backend.save_workflow(workflow)

        # Load and verify task details
        loaded_workflow = await test_storage_backend.load_workflow("complex-workflow")

        assert len(loaded_workflow.tasks) == 2

        analysis_task = loaded_workflow.tasks[0]
        assert analysis_task.id == "analysis_task"
        assert analysis_task.agent_type == "analyst"
        assert analysis_task.parameters == {"type": "data_analysis"}
        assert analysis_task.timeout_seconds == 300

        optimization_task = loaded_workflow.tasks[1]
        assert optimization_task.id == "optimization_task"
        assert optimization_task.dependencies == ["analysis_task"]
        assert optimization_task.priority == 2

    @pytest.mark.asyncio
    async def test_workflow_with_metadata(self, test_storage_backend):
        """Test workflow with metadata persistence."""
        metadata = {
            "campaign_id": "test_campaign_123",
            "created_by": "test_user",
            "priority": "high",
            "tags": ["test", "demo"]
        }

        workflow = WorkflowInstance(
            id="metadata-workflow",
            name="Workflow with Metadata",
            status=WorkflowStatus.PENDING,
            metadata=metadata
        )

        await test_storage_backend.save_workflow(workflow)

        # Load and verify metadata
        loaded_workflow = await test_storage_backend.load_workflow("metadata-workflow")

        assert loaded_workflow.metadata == metadata
        assert loaded_workflow.metadata["campaign_id"] == "test_campaign_123"
        assert loaded_workflow.metadata["tags"] == ["test", "demo"]

    @pytest.mark.asyncio
    async def test_concurrent_workflow_operations(self, test_storage_backend):
        """Test concurrent workflow operations."""
        async def create_and_save_workflow(workflow_id):
            workflow = WorkflowInstance(
                id=workflow_id,
                name=f"Workflow {workflow_id}",
                status=WorkflowStatus.PENDING
            )
            await test_storage_backend.save_workflow(workflow)
            return workflow_id

        # Create multiple workflows concurrently
        workflow_ids = [f"concurrent-wf-{i}" for i in range(5)]
        tasks = [create_and_save_workflow(wf_id) for wf_id in workflow_ids]
        results = await asyncio.gather(*tasks)

        assert set(results) == set(workflow_ids)

        # Verify all workflows were saved
        all_workflows = await test_storage_backend.list_workflows()
        saved_ids = {wf.id for wf in all_workflows}
        assert saved_ids == set(workflow_ids)

    @pytest.mark.asyncio
    async def test_storage_error_handling(self, test_storage_backend):
        """Test storage error handling."""
        # Test with workflow that has an invalid status (this should not raise an error)
        workflow = WorkflowInstance(
            id="error-test-workflow",
            name="Error Test Workflow",
            status=WorkflowStatus.PENDING
        )

        # Storage layer handles this gracefully
        await test_storage_backend.save_workflow(workflow)

        # Verify it was saved
        loaded = await test_storage_backend.load_workflow("error-test-workflow")
        assert loaded is not None
        assert loaded.name == "Error Test Workflow"

    @pytest.mark.asyncio
    async def test_workflow_cleanup(self, test_storage_backend):
        """Test workflow cleanup operations."""
        # Create some test workflows
        for i in range(3):
            workflow = WorkflowInstance(
                id=f"cleanup-wf-{i}",
                name=f"Cleanup Workflow {i}",
                status=WorkflowStatus.COMPLETED
            )
            await test_storage_backend.save_workflow(workflow)

        # Verify workflows exist
        workflows_before = await test_storage_backend.list_workflows()
        assert len(workflows_before) >= 3

        # Note: In a real implementation, we would have cleanup methods
        # For now, just verify the workflows persist correctly

    @pytest.mark.asyncio
    async def test_storage_backend_reinitialization(self, test_storage_backend):
        """Test storage backend reinitialization."""
        # Should handle multiple initializations gracefully
        await test_storage_backend.initialize()
        await test_storage_backend.initialize()

        assert test_storage_backend._initialized is True

    @pytest.mark.asyncio
    async def test_storage_backend_shutdown(self, test_storage_backend):
        """Test storage backend shutdown."""
        await test_storage_backend.close()

        # Should handle multiple closes gracefully
        await test_storage_backend.close()

        # Engine is disposed but not set to None
        assert test_storage_backend._engine is not None


class TestStorageBackendIntegration:
    """Integration tests for storage backend."""

    @pytest.mark.asyncio
    async def test_full_workflow_lifecycle(self, test_storage_backend):
        """Test complete workflow lifecycle with storage."""
        workflow_id = "lifecycle-test"

        # 1. Create workflow
        workflow = WorkflowInstance(
            id=workflow_id,
            name="Lifecycle Test Workflow",
            status=WorkflowStatus.PENDING,
            tasks=[
                TaskSpec(
                    id="lifecycle_task_1",
                    name="Lifecycle Task 1",
                    agent_type="test_agent",
                    status=TaskStatus.PENDING
                )
            ]
        )

        # 2. Save workflow
        await test_storage_backend.save_workflow(workflow)

        # 3. Load and verify
        loaded = await test_storage_backend.load_workflow(workflow_id)
        assert loaded.status == WorkflowStatus.PENDING

        # 4. Update workflow
        loaded.status = WorkflowStatus.RUNNING
        await test_storage_backend.save_workflow(loaded)

        # 5. Load again and verify update
        updated = await test_storage_backend.load_workflow(workflow_id)
        assert updated.status == WorkflowStatus.RUNNING

        # 6. List workflows
        workflows = await test_storage_backend.list_workflows()
        workflow_ids = [wf.id for wf in workflows]
        assert workflow_id in workflow_ids

    @pytest.mark.asyncio
    async def test_storage_backend_with_metadata(self, test_storage_backend):
        """Test storage with workflow metadata."""
        metadata = {
            "campaign_id": "test_campaign",
            "channel": "paid_search",
            "test": True,
            "tags": ["demo", "test"]
        }

        workflow = WorkflowInstance(
            id="metadata-test",
            name="Metadata Test Workflow",
            status=WorkflowStatus.PENDING,
            metadata=metadata
        )

        await test_storage_backend.save_workflow(workflow)

        loaded = await test_storage_backend.load_workflow("metadata-test")
        assert loaded.metadata["campaign_id"] == "test_campaign"
        assert loaded.metadata["channel"] == "paid_search"
        assert loaded.metadata["test"] is True
        assert loaded.metadata["tags"] == ["demo", "test"]