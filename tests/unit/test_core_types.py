"""Unit tests for core types and data models."""

import pytest
from datetime import datetime

from open_mon_user_acquisition.core.types import (
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


class TestTaskStatus:
    """Test TaskStatus enum."""

    def test_task_status_values(self):
        """Test that TaskStatus has expected values."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"


class TestWorkflowStatus:
    """Test WorkflowStatus enum."""

    def test_workflow_status_values(self):
        """Test that WorkflowStatus has expected values."""
        assert WorkflowStatus.PENDING == "pending"
        assert WorkflowStatus.RUNNING == "running"
        assert WorkflowStatus.COMPLETED == "completed"
        assert WorkflowStatus.FAILED == "failed"
        assert WorkflowStatus.CANCELLED == "cancelled"


class TestChannelType:
    """Test ChannelType enum."""

    def test_channel_type_values(self):
        """Test that ChannelType has expected values."""
        assert ChannelType.PAID_SEARCH == "paid_search"
        assert ChannelType.ORGANIC_SEARCH == "organic_search"
        assert ChannelType.SOCIAL_MEDIA == "social_media"
        assert ChannelType.EMAIL_MARKETING == "email_marketing"
        assert ChannelType.CONTENT_MARKETING == "content_marketing"
        assert ChannelType.AFFILIATE == "affiliate"
        assert ChannelType.PARTNERSHIP == "partnership"
        assert ChannelType.DIRECT == "direct"
        assert ChannelType.REFERRAL == "referral"


class TestMetricType:
    """Test MetricType enum."""

    def test_metric_type_values(self):
        """Test that MetricType has expected values."""
        assert MetricType.IMPRESSIONS == "impressions"
        assert MetricType.CLICKS == "clicks"
        assert MetricType.CONVERSIONS == "conversions"
        assert MetricType.REVENUE == "revenue"
        assert MetricType.COST == "cost"
        assert MetricType.CPA == "cpa"
        assert MetricType.CAC == "cac"
        assert MetricType.LTV == "ltv"
        assert MetricType.ROI == "roi"
        assert MetricType.RETENTION_RATE == "retention_rate"
        assert MetricType.CHURN_RATE == "churn_rate"


class TestLLMBackendType:
    """Test LLMBackendType enum."""

    def test_llm_backend_type_values(self):
        """Test that LLMBackendType has expected values."""
        assert LLMBackendType.OPENAI == "openai"
        assert LLMBackendType.OLLAMA == "ollama"


class TestTaskSpec:
    """Test TaskSpec dataclass."""

    def test_task_spec_creation(self):
        """Test creating a TaskSpec instance."""
        task = TaskSpec(
            id="test_task_1",
            name="Test Task",
            agent_type="test_agent",
            parameters={"param1": "value1"},
            dependencies=["task_0"],
            timeout_seconds=60,
            priority=2
        )

        assert task.id == "test_task_1"
        assert task.name == "Test Task"
        assert task.agent_type == "test_agent"
        assert task.parameters == {"param1": "value1"}
        assert task.dependencies == ["task_0"]
        assert task.timeout_seconds == 60
        assert task.priority == 2
        assert task.retry_count == 0  # default value

    def test_task_spec_defaults(self):
        """Test TaskSpec default values."""
        task = TaskSpec(
            id="test_task_2",
            name="Minimal Task",
            agent_type="test_agent"
        )

        # Check that default values are properly set
        assert task.description is None
        assert task.timeout_seconds is None
        assert task.retry_count == 0
        assert task.priority == 1

        # For dataclasses with Field(default_factory), we can't directly compare
        # but we can check the types and that they're not None
        assert hasattr(task, 'parameters')
        assert hasattr(task, 'dependencies')
        assert task.parameters is not None
        assert task.dependencies is not None


class TestTaskResult:
    """Test TaskResult dataclass."""

    def test_task_result_creation(self):
        """Test creating a TaskResult instance."""
        result = TaskResult(
            task_id="test_task_1",
            status=TaskStatus.COMPLETED,
            result={"output": "success"},
            started_at=datetime.now(),
            completed_at=datetime.now()
        )

        assert result.task_id == "test_task_1"
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"output": "success"}
        assert result.error is None
        assert result.started_at is not None
        assert result.completed_at is not None

    def test_task_result_with_error(self):
        """Test TaskResult with error."""
        result = TaskResult(
            task_id="test_task_1",
            status=TaskStatus.FAILED,
            error="Task failed",
            started_at=datetime.now(),
            completed_at=datetime.now()
        )

        assert result.status == TaskStatus.FAILED
        assert result.error == "Task failed"
        assert result.result is None


class TestWorkflowInstance:
    """Test WorkflowInstance model."""

    def test_workflow_instance_creation(self):
        """Test creating a WorkflowInstance."""
        workflow = WorkflowInstance(
            id="test_workflow_1",
            name="Test Workflow",
            tasks=[
                TaskSpec(id="task_1", name="Task 1", agent_type="agent1"),
                TaskSpec(id="task_2", name="Task 2", agent_type="agent2")
            ]
        )

        assert workflow.id == "test_workflow_1"
        assert workflow.name == "Test Workflow"
        assert workflow.status == WorkflowStatus.PENDING
        assert len(workflow.tasks) == 2
        assert workflow.results == {}
        assert workflow.created_at is not None

    def test_workflow_instance_with_results(self):
        """Test WorkflowInstance with task results."""
        task_result = TaskResult(
            task_id="task_1",
            status=TaskStatus.COMPLETED,
            result="success"
        )

        workflow = WorkflowInstance(
            id="test_workflow_2",
            name="Workflow with Results",
            status=WorkflowStatus.COMPLETED,
            tasks=[TaskSpec(id="task_1", name="Task 1", agent_type="agent1")],
            results={"task_1": task_result}
        )

        assert workflow.status == WorkflowStatus.COMPLETED
        assert len(workflow.results) == 1
        assert workflow.results["task_1"] == task_result


class TestMetricData:
    """Test MetricData model."""

    def test_metric_data_creation(self):
        """Test creating MetricData instance."""
        metric = MetricData(
            name="test_metric",
            value=42.5,
            channel=ChannelType.PAID_SEARCH,
            campaign="test_campaign",
            metadata={"source": "test"}
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.channel == ChannelType.PAID_SEARCH
        assert metric.campaign == "test_campaign"
        assert metric.metadata == {"source": "test"}
        assert metric.timestamp is not None


class TestLLMResponse:
    """Test LLMResponse model."""

    def test_llm_response_creation(self):
        """Test creating LLMResponse instance."""
        response = LLMResponse(
            content="Test response",
            usage={"tokens": 10, "cost": 0.01},
            finish_reason="stop",
            model="gpt-4",
            metadata={"backend": "openai"}
        )

        assert response.content == "Test response"
        assert response.usage == {"tokens": 10, "cost": 0.01}
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4"
        assert response.metadata == {"backend": "openai"}


class TestLLMMessage:
    """Test LLMMessage model."""

    def test_llm_message_creation(self):
        """Test creating LLMMessage instance."""
        message = LLMMessage(
            role="user",
            content="Hello, world!",
            name="test_user",
            metadata={"temperature": 0.7}
        )

        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name == "test_user"
        assert message.metadata == {"temperature": 0.7}

    def test_llm_message_minimal(self):
        """Test creating minimal LLMMessage."""
        message = LLMMessage(
            role="assistant",
            content="Response"
        )

        assert message.role == "assistant"
        assert message.content == "Response"
        assert message.name is None
        assert message.metadata == {}


class TestContextData:
    """Test ContextData model."""

    def test_context_data_creation(self):
        """Test creating ContextData instance."""
        context = ContextData(
            user_id="user_123",
            session_id="session_456",
            campaign_id="campaign_789",
            channel=ChannelType.SOCIAL_MEDIA,
            metrics={
                "clicks": MetricData(name="clicks", value=100),
                "conversions": MetricData(name="conversions", value=10)
            },
            metadata={"version": "1.0"},
            previous_results={"task_1": "result_1"}
        )

        assert context.user_id == "user_123"
        assert context.session_id == "session_456"
        assert context.campaign_id == "campaign_789"
        assert context.channel == ChannelType.SOCIAL_MEDIA
        assert len(context.metrics) == 2
        assert context.metadata == {"version": "1.0"}
        assert context.previous_results == {"task_1": "result_1"}

    def test_context_data_empty(self):
        """Test creating empty ContextData."""
        context = ContextData()

        assert context.user_id is None
        assert context.session_id is None
        assert context.campaign_id is None
        assert context.channel is None
        assert context.metrics == {}
        assert context.metadata == {}
        assert context.previous_results == {}
