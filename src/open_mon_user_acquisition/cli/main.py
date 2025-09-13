"""Main CLI application for OpenMonetization-UserAcquisition."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from ..config import ConfigManager
from ..orchestrator import WorkflowOrchestrator
from ..core.types import ContextData
from . import commands

# Initialize console for rich output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="omua",
    help="OpenMonetization-UserAcquisition - Enterprise user acquisition strategy platform",
    add_completion=False,
)

# Global orchestrator instance
_orchestrator: Optional[WorkflowOrchestrator] = None


def get_orchestrator() -> WorkflowOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        config = ConfigManager()
        _orchestrator = WorkflowOrchestrator(config)
    return _orchestrator


@app.callback()
def main_callback(
    ctx: typer.Context,
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug mode",
    ),
):
    """OpenMonetization-UserAcquisition CLI."""
    # Store options in context for subcommands
    ctx.obj = {
        "config_file": config_file,
        "verbose": verbose,
        "debug": debug,
    }


@app.command()
def status(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output status in JSON format",
    ),
):
    """Show system status and health information."""
    async def _status():
        try:
            orchestrator = get_orchestrator()
            system_status = await orchestrator.get_system_status()

            if json_output:
                console.print_json(data=system_status)
            else:
                # Rich formatted output
                status_table = Table(title="System Status")
                status_table.add_column("Component", style="cyan")
                status_table.add_column("Status", style="green")
                status_table.add_column("Details", style="yellow")

                for component, info in system_status["components"].items():
                    status = "[green]✓ Available[/green]" if info["available"] else "[red]✗ Unavailable[/red]"
                    details = info.get("type", "N/A")
                    if component == "task_scheduler" and "running_tasks" in info:
                        details += f" ({info['running_tasks']} running tasks)"
                    status_table.add_row(component, status, details)

                console.print(status_table)

                # Workflow metrics
                if "workflow_metrics" in system_status:
                    metrics = system_status["workflow_metrics"]
                    metrics_table = Table(title="Workflow Metrics")
                    metrics_table.add_column("Metric", style="cyan")
                    metrics_table.add_column("Value", style="magenta")

                    for metric, value in metrics.items():
                        metrics_table.add_row(metric.replace("_", " ").title(), str(value))

                    console.print(metrics_table)

        except Exception as e:
            if json_output:
                console.print_json(data={"error": str(e)})
            else:
                console.print(f"[red]Error getting system status: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_status())


@app.command()
def workflow(
    ctx: typer.Context,
    action: str = typer.Argument(
        ...,
        help="Action to perform: create, execute, status, cancel, list",
    ),
    workflow_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Workflow ID for status/cancel actions",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Workflow name for create action",
    ),
    context_file: Optional[str] = typer.Option(
        None,
        "--context",
        "-f",
        help="Path to JSON file containing workflow context",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format",
    ),
):
    """Manage workflows."""
    async def _workflow():
        try:
            orchestrator = get_orchestrator()

            if action == "create":
                if not name:
                    console.print("[red]Error: --name is required for create action[/red]")
                    raise typer.Exit(1)

                # Load context
                context = ContextData()
                if context_file:
                    if not Path(context_file).exists():
                        console.print(f"[red]Error: Context file not found: {context_file}[/red]")
                        raise typer.Exit(1)

                    with open(context_file, "r") as f:
                        context_data = json.load(f)
                        context = ContextData(**context_data)

                workflow = await orchestrator.create_workflow(name, [], context)

                if json_output:
                    console.print_json(data=workflow.model_dump())
                else:
                    console.print(f"[green]Created workflow:[/green] {workflow.id}")
                    console.print(f"[cyan]Name:[/cyan] {workflow.name}")
                    console.print(f"[cyan]Status:[/cyan] {workflow.status}")
                    console.print(f"[cyan]Tasks:[/cyan] {len(workflow.tasks)}")

            elif action == "execute":
                if not workflow_id:
                    console.print("[red]Error: --id is required for execute action[/red]")
                    raise typer.Exit(1)

                workflow = await orchestrator.execute_workflow(workflow_id)

                if json_output:
                    console.print_json(data=workflow.model_dump())
                else:
                    status_color = {
                        "completed": "green",
                        "failed": "red",
                        "running": "yellow",
                        "pending": "blue",
                        "cancelled": "magenta",
                    }.get(workflow.status, "white")

                    console.print(f"[green]Executed workflow:[/green] {workflow.id}")
                    console.print(f"[cyan]Status:[/cyan] [{status_color}]{workflow.status}[/{status_color}]")

                    if workflow.completed_at and workflow.started_at:
                        duration = workflow.completed_at - workflow.started_at
                        console.print(f"[cyan]Duration:[/cyan] {duration.total_seconds():.2f}s")

            elif action == "status":
                if not workflow_id:
                    console.print("[red]Error: --id is required for status action[/red]")
                    raise typer.Exit(1)

                workflow = await orchestrator.get_workflow_status(workflow_id)
                if not workflow:
                    console.print(f"[red]Workflow not found: {workflow_id}[/red]")
                    raise typer.Exit(1)

                if json_output:
                    console.print_json(data=workflow.model_dump())
                else:
                    status_color = {
                        "completed": "green",
                        "failed": "red",
                        "running": "yellow",
                        "pending": "blue",
                        "cancelled": "magenta",
                    }.get(workflow.status, "white")

                    console.print(Panel.fit(
                        f"[bold]Workflow: {workflow.id}[/bold]\n"
                        f"[cyan]Name:[/cyan] {workflow.name}\n"
                        f"[cyan]Status:[/cyan] [{status_color}]{workflow.status}[/{status_color}]\n"
                        f"[cyan]Created:[/cyan] {workflow.created_at}\n"
                        f"[cyan]Tasks:[/cyan] {len(workflow.tasks)}",
                        title="Workflow Status"
                    ))

            elif action == "cancel":
                if not workflow_id:
                    console.print("[red]Error: --id is required for cancel action[/red]")
                    raise typer.Exit(1)

                cancelled = await orchestrator.cancel_workflow(workflow_id)

                if cancelled:
                    console.print(f"[green]Cancelled workflow: {workflow_id}[/green]")
                else:
                    console.print(f"[yellow]Could not cancel workflow: {workflow_id}[/yellow]")

            elif action == "list":
                workflows = await orchestrator.list_active_workflows()

                if json_output:
                    workflow_data = [w.model_dump() for w in workflows]
                    console.print_json(data={"workflows": workflow_data})
                else:
                    if not workflows:
                        console.print("[yellow]No active workflows found[/yellow]")
                        return

                    table = Table(title="Active Workflows")
                    table.add_column("ID", style="cyan", no_wrap=True)
                    table.add_column("Name", style="white")
                    table.add_column("Status", style="green")
                    table.add_column("Created", style="blue")
                    table.add_column("Tasks", style="yellow", justify="right")

                    for workflow in workflows:
                        status_color = {
                            "completed": "green",
                            "failed": "red",
                            "running": "yellow",
                            "pending": "blue",
                            "cancelled": "magenta",
                        }.get(workflow.status, "white")

                        table.add_row(
                            workflow.id[:8] + "...",
                            workflow.name,
                            f"[{status_color}]{workflow.status}[/{status_color}]",
                            workflow.created_at.strftime("%Y-%m-%d %H:%M") if workflow.created_at else "N/A",
                            str(len(workflow.tasks)),
                        )

                    console.print(table)

            else:
                console.print(f"[red]Unknown action: {action}[/red]")
                console.print("[cyan]Available actions: create, execute, status, cancel, list[/cyan]")
                raise typer.Exit(1)

        except Exception as e:
            if json_output:
                console.print_json(data={"error": str(e)})
            else:
                console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_workflow())


@app.command()
def config(
    ctx: typer.Context,
    action: str = typer.Argument(
        ...,
        help="Action to perform: show, validate, create-default",
    ),
    file_path: Optional[str] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to configuration file",
    ),
):
    """Manage configuration."""
    try:
        config_manager = ConfigManager()

        if action == "show":
            config_path = file_path or ctx.obj.get("config_file")
            if config_path:
                config_manager.load_from_file(config_path)

            config_data = config_manager.get_all()

            console.print(Panel.fit(
                "[bold cyan]Current Configuration[/bold cyan]\n" +
                json.dumps(config_data, indent=2, default=str),
                title="Configuration"
            ))

        elif action == "validate":
            config_path = file_path or ctx.obj.get("config_file")
            if config_path:
                config_manager.load_from_file(config_path)

            errors = config_manager.validate_config()

            if errors:
                console.print("[red]Configuration validation failed:[/red]")
                for error in errors:
                    console.print(f"  • {error}")
                raise typer.Exit(1)
            else:
                console.print("[green]Configuration is valid[/green]")

        elif action == "create-default":
            if not file_path:
                console.print("[red]Error: --file is required for create-default action[/red]")
                raise typer.Exit(1)

            config_manager.create_default_config(file_path)
            console.print(f"[green]Created default configuration file: {file_path}[/green]")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("[cyan]Available actions: show, validate, create-default[/cyan]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def cleanup(
    ctx: typer.Context,
    days: int = typer.Option(
        30,
        "--days",
        "-d",
        help="Number of days of data to keep",
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
):
    """Clean up old workflow and metric data."""
    async def _cleanup():
        try:
            if not confirm:
                console.print(f"[yellow]This will delete data older than {days} days.[/yellow]")
                if not typer.confirm("Are you sure you want to continue?"):
                    console.print("[cyan]Cleanup cancelled[/cyan]")
                    return

            orchestrator = get_orchestrator()
            stats = await orchestrator.cleanup_old_data(days)

            console.print("[green]Cleanup completed:[/green]")
            for key, value in stats.items():
                if key != "errors":
                    console.print(f"  • {key.replace('_', ' ').title()}: {value}")

            if stats.get("errors"):
                console.print("[red]Errors during cleanup:[/red]")
                for error in stats["errors"]:
                    console.print(f"  • {error}")

        except Exception as e:
            console.print(f"[red]Error during cleanup: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_cleanup())


@app.command()
def ab(
    ctx: typer.Context,
    action: str = typer.Argument(
        ...,
        help="Action to perform: create, start, stop, list, status, stats",
    ),
    experiment_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Experiment ID for status/stats actions",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Experiment name for create action",
    ),
    campaign_id: Optional[str] = typer.Option(
        None,
        "--campaign",
        "-c",
        help="Campaign ID for create/list actions",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-f",
        help="Configuration file path for create action",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Experiment description for create action",
    ),
    primary_metric: str = typer.Option(
        "conversion_rate",
        "--metric",
        "-m",
        help="Primary metric for create action",
    ),
    sample_size: int = typer.Option(
        1000,
        "--sample-size",
        "-s",
        help="Target sample size for create action",
    ),
    status_filter: Optional[str] = typer.Option(
        None,
        "--status",
        help="Status filter for list action (draft, running, paused, completed, cancelled)",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format",
    ),
):
    """Manage A/B testing experiments."""
    async def _ab():
        try:
            if action == "create":
                if not name or not campaign_id or not config_file:
                    console.print("[red]Error: --name, --campaign, and --config are required for create action[/red]")
                    raise typer.Exit(1)

                await commands.ab_create(
                    name=name,
                    campaign_id=campaign_id,
                    config_file=config_file,
                    description=description,
                    primary_metric=primary_metric,
                    target_sample_size=sample_size,
                    json_output=json_output
                )

            elif action == "start":
                if not experiment_id:
                    console.print("[red]Error: --id is required for start action[/red]")
                    raise typer.Exit(1)

                await commands.ab_start(experiment_id, json_output)

            elif action == "stop":
                if not experiment_id:
                    console.print("[red]Error: --id is required for stop action[/red]")
                    raise typer.Exit(1)

                await commands.ab_stop(experiment_id, json_output)

            elif action == "list":
                await commands.ab_list(campaign_id, status_filter, json_output)

            elif action == "status":
                if not experiment_id:
                    console.print("[red]Error: --id is required for status action[/red]")
                    raise typer.Exit(1)

                await commands.ab_status(experiment_id, json_output)

            elif action == "stats":
                if not experiment_id:
                    console.print("[red]Error: --id is required for stats action[/red]")
                    raise typer.Exit(1)

                await commands.ab_stats(experiment_id, json_output)

            else:
                console.print(f"[red]Unknown action: {action}[/red]")
                console.print("[cyan]Available actions: create, start, stop, list, status, stats[/cyan]")
                raise typer.Exit(1)

        except Exception as e:
            if json_output:
                console.print_json(data={"error": str(e)})
            else:
                console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_ab())


@app.command()
def version():
    """Show version information."""
    from .. import __version__
    console.print(f"[bold cyan]OpenMonetization-UserAcquisition[/bold cyan] v{__version__}")
    console.print("[dim]Enterprise user acquisition strategy platform[/dim]")


# Add completion for bash/zsh/fish
@app.command()
def completion(shell: str):
    """Generate shell completion script."""
    if shell not in ["bash", "zsh", "fish"]:
        console.print(f"[red]Unsupported shell: {shell}[/red]")
        console.print("[cyan]Supported shells: bash, zsh, fish[/cyan]")
        raise typer.Exit(1)

    completion_script = f"""
# Add this to your ~/.{shell}rc file:
# eval "$(omua completion {shell})"

# OMUA completion for {shell}
"""

    console.print(completion_script)


if __name__ == "__main__":
    app()
