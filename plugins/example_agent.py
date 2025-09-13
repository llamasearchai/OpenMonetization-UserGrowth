"""Example custom agent plugin for OMUA.

This plugin demonstrates how to create custom agents that can be loaded
dynamically by the OMUA system.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from open_mon_user_acquisition.core.interfaces import AgentInterface
from open_mon_user_acquisition.core.types import TaskSpec, ContextData, TaskStatus
from open_mon_user_acquisition.plugins import PluginInterface

logger = logging.getLogger(__name__)


class ExampleAgent(AgentInterface, PluginInterface):
    """Example custom agent that can be loaded as a plugin."""

    def __init__(self, custom_parameter: str = "default"):
        """Initialize the example agent."""
        self._custom_parameter = custom_parameter
        self._name = "example_agent"
        self._description = f"Example custom agent (param: {custom_parameter})"

    @property
    def name(self) -> str:
        """Return the agent name."""
        return self._name

    @property
    def description(self) -> str:
        """Return the agent description."""
        return self._description

    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"

    @property
    def type(self) -> str:
        """Plugin type."""
        return "agent"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        logger.info(f"Initializing example agent plugin with config: {config}")
        if "custom_parameter" in config:
            self._custom_parameter = config["custom_parameter"]
            self._description = f"Example custom agent (param: {self._custom_parameter})"

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        logger.info("Shutting down example agent plugin")

    async def plan(self, context: ContextData) -> List[TaskSpec]:
        """Plan tasks for user acquisition using custom logic."""
        logger.info(f"{self.name}: Planning tasks with custom parameter: {self._custom_parameter}")

        # Create tasks based on the custom parameter
        tasks = []

        if "analysis" in self._custom_parameter:
            tasks.append(TaskSpec(
                id=f"{self.name}_analysis_task",
                name="Custom Analysis Task",
                agent_type=self.name,
                description=f"Perform {self._custom_parameter} analysis",
                parameters={
                    "analysis_type": self._custom_parameter,
                    "context_campaign": context.campaign_id
                }
            ))

        if "optimization" in self._custom_parameter:
            tasks.append(TaskSpec(
                id=f"{self.name}_optimization_task",
                name="Custom Optimization Task",
                agent_type=self.name,
                description=f"Perform {self._custom_parameter} optimization",
                parameters={
                    "optimization_type": self._custom_parameter,
                    "target_metrics": ["conversion_rate", "roi"]
                },
                dependencies=[f"{self.name}_analysis_task"] if tasks else []
            ))

        # Always add a summary task
        tasks.append(TaskSpec(
            id=f"{self.name}_summary_task",
            name="Custom Summary Task",
            agent_type=self.name,
            description="Generate summary of custom agent activities",
            parameters={
                "custom_parameter_used": self._custom_parameter,
                "tasks_performed": len(tasks)
            },
            dependencies=[t.id for t in tasks[:-1]] if len(tasks) > 1 else []
        ))

        logger.info(f"{self.name}: Planned {len(tasks)} custom tasks")
        return tasks

    async def execute(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute a task using custom logic."""
        logger.info(f"{self.name}: Executing task '{task.name}' with custom parameter: {self._custom_parameter}")

        try:
            if "analysis" in task.name.lower():
                return await self._execute_analysis_task(task, context)
            elif "optimization" in task.name.lower():
                return await self._execute_optimization_task(task, context)
            elif "summary" in task.name.lower():
                return await self._execute_summary_task(task, context)
            else:
                return await self._execute_generic_task(task, context)

        except Exception as e:
            logger.error(f"{self.name}: Task execution failed: {e}")
            # Return a properly formatted error result
            from open_mon_user_acquisition.core.types import TaskResult
            return TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                result={"error": str(e)},
                metadata={"execution_error": True, "custom_parameter": self._custom_parameter}
            )

    async def _execute_analysis_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute custom analysis task."""
        from open_mon_user_acquisition.core.types import TaskResult

        analysis_result = {
            "custom_analysis": True,
            "parameter_used": self._custom_parameter,
            "insights": [
                f"Custom analysis using parameter: {self._custom_parameter}",
                f"Campaign context: {context.campaign_id}",
                "Identified key opportunities based on custom logic",
                "Generated actionable recommendations"
            ],
            "metrics_analyzed": ["conversion_rate", "acquisition_cost", "lifetime_value"],
            "custom_score": len(self._custom_parameter) * 10  # Example custom scoring
        }

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            result=analysis_result,
            metadata={
                "analysis_type": "custom",
                "custom_parameter": self._custom_parameter,
                "execution_time": 1.5
            }
        )

    async def _execute_optimization_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute custom optimization task."""
        from open_mon_user_acquisition.core.types import TaskResult

        optimization_result = {
            "custom_optimization": True,
            "parameter_used": self._custom_parameter,
            "optimizations_applied": [
                f"Applied {self._custom_parameter} optimization strategy",
                "Optimized channel allocation based on custom logic",
                "Implemented automated bidding adjustments",
                "Set up custom performance monitoring"
            ],
            "expected_improvements": {
                "conversion_rate": "+15%",
                "cost_efficiency": "+12%",
                "roi": "+8%"
            },
            "custom_optimization_score": len(self._custom_parameter) * 15
        }

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            result=optimization_result,
            metadata={
                "optimization_type": "custom",
                "custom_parameter": self._custom_parameter,
                "execution_time": 2.0
            }
        )

    async def _execute_summary_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute custom summary task."""
        from open_mon_user_acquisition.core.types import TaskResult

        summary_result = {
            "custom_summary": True,
            "agent_name": self.name,
            "custom_parameter": self._custom_parameter,
            "campaign_processed": context.campaign_id,
            "summary_stats": {
                "tasks_completed": task.parameters.get("tasks_performed", 0),
                "custom_score": len(self._custom_parameter) * 25,
                "processing_time": 0.5
            },
            "recommendations": [
                f"Continue using {self._custom_parameter} for future campaigns",
                "Monitor performance metrics closely",
                "Consider A/B testing with different custom parameters"
            ]
        }

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            result=summary_result,
            metadata={
                "summary_type": "custom_agent",
                "custom_parameter": self._custom_parameter,
                "execution_time": 0.5
            }
        )

    async def _execute_generic_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute generic custom task."""
        from open_mon_user_acquisition.core.types import TaskResult

        generic_result = {
            "custom_execution": True,
            "task_name": task.name,
            "custom_parameter": self._custom_parameter,
            "context_info": {
                "campaign_id": context.campaign_id,
                "channel": context.channel,
                "has_metadata": context.metadata is not None
            },
            "custom_processing": f"Processed using {self._custom_parameter} logic"
        }

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            result=generic_result,
            metadata={
                "execution_type": "custom_generic",
                "custom_parameter": self._custom_parameter,
                "execution_time": 1.0
            }
        )
