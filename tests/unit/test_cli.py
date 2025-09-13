"""Unit tests for CLI interface."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typer.testing import CliRunner

from open_mon_user_acquisition.cli.main import app
from open_mon_user_acquisition.orchestrator import WorkflowOrchestrator


@pytest.fixture
def runner():
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator for CLI testing."""
    mock_orch = MagicMock(spec=WorkflowOrchestrator)

    # Mock the async methods
    mock_orch.get_system_status = AsyncMock(return_value={
        "initialized": True,
        "timestamp": "2025-01-01T12:00:00",
        "components": {
            "storage": {"available": True, "type": "SQLiteStorageBackend"},
            "llm_backend": {"available": True, "type": "LLMFallbackManager"},
            "task_scheduler": {"active": True, "running_tasks": 0}
        },
        "workflow_metrics": {
            "active_workflows": 0,
            "pending_workflows": 0,
            "completed_workflows": 0,
            "failed_workflows": 0,
            "status_counts": {},
            "total_workflows": 0,
            "recent_workflows": 0
        }
    })

    mock_orch.list_active_workflows = AsyncMock(return_value=[])
    mock_orch.create_workflow = AsyncMock()
    mock_orch.execute_workflow = AsyncMock()
    mock_orch.get_workflow_status = AsyncMock(return_value=None)
    mock_orch.cancel_workflow = AsyncMock(return_value=True)
    mock_orch.cleanup_old_data = AsyncMock(return_value={"deleted": 0})

    return mock_orch


class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "OpenMonetization-UserAcquisition" in result.output
        assert "status" in result.output
        assert "workflow" in result.output
        assert "config" in result.output

    def test_cli_version(self, runner):
        """Test CLI version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "OpenMonetization-UserAcquisition" in result.output
        assert "0.1.0" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cli_status_command(self, mock_get_orch, runner, mock_orchestrator):
        """Test status command."""
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "System Status" in result.output
        assert "Available" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cli_status_json_command(self, mock_get_orch, runner, mock_orchestrator):
        """Test status command with JSON output."""
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["status", "--json"])
        assert result.exit_code == 0
        assert '"initialized": true' in result.output
        assert '"components"' in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cli_workflow_list_empty(self, mock_get_orch, runner, mock_orchestrator):
        """Test workflow list command with no workflows."""
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "list"])
        assert result.exit_code == 0
        assert "No active workflows found" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cli_workflow_create_missing_name(self, mock_get_orch, runner, mock_orchestrator):
        """Test workflow create command without name."""
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "create"])
        assert result.exit_code != 0
        assert "Error: --name is required" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cli_workflow_status_missing_id(self, mock_get_orch, runner, mock_orchestrator):
        """Test workflow status command without ID."""
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "status"])
        assert result.exit_code != 0
        assert "Error: --id is required" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cli_workflow_execute_missing_id(self, mock_get_orch, runner, mock_orchestrator):
        """Test workflow execute command without ID."""
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "execute"])
        assert result.exit_code != 0
        assert "Error: --id is required" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cli_workflow_cancel_missing_id(self, mock_get_orch, runner, mock_orchestrator):
        """Test workflow cancel command without ID."""
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "cancel"])
        assert result.exit_code != 0
        assert "Error: --id is required" in result.output

    def test_cli_config_show(self, runner):
        """Test config show command."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "Current Configuration" in result.output
        assert "OpenMonetization-UserAcquisition" in result.output

    def test_cli_config_validate_valid(self, runner):
        """Test config validate command with valid config."""
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0
        assert "Configuration is valid" in result.output

    def test_cli_config_create_default_missing_file(self, runner):
        """Test config create-default without file parameter."""
        result = runner.invoke(app, ["config", "create-default"])
        assert result.exit_code != 0
        assert "Error: --file is required" in result.output

    def test_cli_cleanup_missing_confirmation(self, runner):
        """Test cleanup command without confirmation."""
        result = runner.invoke(app, ["cleanup"])
        assert result.exit_code == 0  # Should not fail, just inform user

    def test_cli_completion_invalid_shell(self, runner):
        """Test completion command with invalid shell."""
        result = runner.invoke(app, ["completion", "invalid"])
        assert result.exit_code != 0
        assert "Unsupported shell" in result.output

    def test_cli_completion_valid_shell(self, runner):
        """Test completion command with valid shell."""
        result = runner.invoke(app, ["completion", "bash"])
        assert result.exit_code == 0
        assert "OMUA completion for bash" in result.output


class TestCLIWorkflowCommands:
    """Test CLI workflow-specific commands."""

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_workflow_create_success(self, mock_get_orch, runner, mock_orchestrator):
        """Test successful workflow creation."""
        from open_mon_user_acquisition.core.types import WorkflowInstance

        mock_workflow = WorkflowInstance(
            id="test-workflow-123",
            name="Test Workflow",
            status="pending"
        )
        mock_orchestrator.create_workflow.return_value = mock_workflow
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "create", "--name", "Test Workflow"])
        assert result.exit_code == 0
        assert "Created workflow: test-workflow-123" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_workflow_execute_success(self, mock_get_orch, runner, mock_orchestrator):
        """Test successful workflow execution."""
        from open_mon_user_acquisition.core.types import WorkflowInstance

        mock_workflow = WorkflowInstance(
            id="test-workflow-123",
            name="Test Workflow",
            status="completed"
        )
        mock_orchestrator.execute_workflow.return_value = mock_workflow
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "execute", "--id", "test-workflow-123"])
        assert result.exit_code == 0
        assert "Executed workflow: test-workflow-123" in result.output
        assert "completed" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_workflow_status_not_found(self, mock_get_orch, runner, mock_orchestrator):
        """Test workflow status for non-existent workflow."""
        mock_orchestrator.get_workflow_status.return_value = None
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "status", "--id", "nonexistent"])
        assert result.exit_code != 0
        assert "Workflow not found" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_workflow_cancel_success(self, mock_get_orch, runner, mock_orchestrator):
        """Test successful workflow cancellation."""
        mock_orchestrator.cancel_workflow.return_value = True
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "cancel", "--id", "test-workflow-123"])
        assert result.exit_code == 0
        assert "Cancelled workflow: test-workflow-123" in result.output

    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_workflow_cancel_failed(self, mock_get_orch, runner, mock_orchestrator):
        """Test failed workflow cancellation."""
        mock_orchestrator.cancel_workflow.return_value = False
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["workflow", "cancel", "--id", "test-workflow-123"])
        assert result.exit_code == 0
        assert "Could not cancel workflow" in result.output


class TestCLIConfigCommands:
    """Test CLI configuration commands."""

    def test_config_show_output(self, runner):
        """Test config show output format."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "app_name" in result.output or "OpenMonetization-UserAcquisition" in result.output

    def test_config_unknown_action(self, runner):
        """Test config command with unknown action."""
        result = runner.invoke(app, ["config", "unknown"])
        assert result.exit_code != 0
        assert "Unknown action" in result.output

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("yaml.dump")
    def test_config_create_default_yaml(self, mock_yaml_dump, mock_open, mock_mkdir, runner, tmp_path):
        """Test creating default YAML configuration."""
        config_file = tmp_path / "test_config.yaml"

        result = runner.invoke(app, ["config", "create-default", "--file", str(config_file)])
        assert result.exit_code == 0
        assert "Created default configuration file" in result.output

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.dump")
    def test_config_create_default_json(self, mock_json_dump, mock_open, mock_mkdir, runner, tmp_path):
        """Test creating default JSON configuration."""
        config_file = tmp_path / "test_config.json"

        result = runner.invoke(app, ["config", "create-default", "--file", str(config_file)])
        assert result.exit_code == 0
        assert "Created default configuration file" in result.output


class TestCLICleanupCommands:
    """Test CLI cleanup commands."""

    @patch("typer.confirm")
    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cleanup_with_confirmation(self, mock_get_orch, mock_confirm, runner, mock_orchestrator):
        """Test cleanup command with user confirmation."""
        mock_confirm.return_value = True
        mock_orchestrator.cleanup_old_data.return_value = {"workflows_cleaned": 5}
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["cleanup", "--days", "30"])
        assert result.exit_code == 0
        assert "Cleanup completed" in result.output

    @patch("typer.confirm")
    @patch("open_mon_user_acquisition.cli.main.get_orchestrator")
    def test_cleanup_cancelled(self, mock_get_orch, mock_confirm, runner, mock_orchestrator):
        """Test cleanup command when user cancels."""
        mock_confirm.return_value = False
        mock_get_orch.return_value = mock_orchestrator

        result = runner.invoke(app, ["cleanup", "--days", "30"])
        assert result.exit_code == 0
        assert "Cleanup cancelled" in result.output
