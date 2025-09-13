#!/usr/bin/env python3
"""Demo script for OpenMonetization-UserAcquisition system."""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from open_mon_user_acquisition.config import ConfigManager
from open_mon_user_acquisition.orchestrator import WorkflowOrchestrator
from open_mon_user_acquisition.core.types import ContextData, TaskSpec
from open_mon_user_acquisition.llm import LLMFallbackManager
from open_mon_user_acquisition.storage import SQLiteStorageBackend


class DemoAgent:
    """Simple demo agent for testing the system."""

    def __init__(self, name: str):
        self._name = name
        self._description = f"Demo agent: {name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def plan(self, context):
        """Plan tasks for the demo."""
        print(f"🤖 {self.name}: Planning tasks based on context...")

        tasks = [
            TaskSpec(
                id=f"{self.name}_task_1",
                name=f"{self.name} Analysis Task",
                agent_type=self.name,
                description=f"Analyze {context.campaign_id or 'campaign'} data",
                parameters={
                    "analysis_type": "user_acquisition",
                    "channel": context.channel or "mixed"
                }
            ),
            TaskSpec(
                id=f"{self.name}_task_2",
                name=f"{self.name} Optimization Task",
                agent_type=self.name,
                description="Optimize acquisition strategy",
                parameters={
                    "optimization_goal": "maximize_roi",
                    "budget_constraint": 10000
                },
                dependencies=[f"{self.name}_task_1"]
            )
        ]

        print(f"📋 {self.name}: Planned {len(tasks)} tasks")
        return tasks

    async def execute(self, task, context):
        """Execute a task for the demo."""
        from open_mon_user_acquisition.core.types import TaskResult, TaskStatus

        print(f"⚡ {self.name}: Executing task '{task.name}'...")

        # Simulate some processing time
        await asyncio.sleep(0.5)

        # Simulate task result
        if "analysis" in task.name.lower():
            result = {
                "insights": [
                    "Channel performance varies by 40%",
                    "Mobile traffic shows 25% higher conversion",
                    "Email marketing has best LTV/CAC ratio"
                ],
                "recommendations": [
                    "Increase mobile ad spend by 30%",
                    "Optimize email campaigns for higher engagement",
                    "Test new landing page variants"
                ]
            }
        else:
            result = {
                "optimizations": [
                    "Allocated $6,000 to high-performing channels",
                    "Created 3 new campaign variants for testing",
                    "Set up automated bid optimization"
                ],
                "expected_impact": {
                    "roi_improvement": "+15%",
                    "cost_reduction": "-8%",
                    "conversion_increase": "+22%"
                }
            }

        print(f"✅ {self.name}: Task '{task.name}' completed successfully")

        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            result=result,
            metadata={"agent": self.name, "execution_time": 0.5}
        )


async def main():
    """Run the OMUA demo."""
    print("OpenMonetization-UserAcquisition Demo")
    print("=" * 50)

    # Create demo context
    context = ContextData(
        user_id="demo_user_123",
        session_id="demo_session_456",
        campaign_id="demo_campaign_789",
        channel="paid_search",
        metadata={
            "demo": True,
            "version": "1.0",
            "objectives": ["increase_conversion_rate", "optimize_cac", "improve_retention"]
        }
    )

    print("Demo Context:")
    print(f"  Campaign: {context.campaign_id}")
    print(f"  Channel: {context.channel}")
    print(f"  Objectives: {', '.join(context.metadata['objectives'])}")
    print()

    # Initialize components
    print("Initializing system components...")

    config = ConfigManager()
    storage = SQLiteStorageBackend()
    llm_backend = LLMFallbackManager()

    orchestrator = WorkflowOrchestrator(
        config=config,
        storage=storage,
        llm_backend=llm_backend
    )

    await orchestrator.initialize()

    print("System initialized with LLM-powered agents")
    print()

    # Create and execute workflow
    print("Creating user acquisition optimization workflow...")

    workflow = await orchestrator.create_workflow(
        name="Demo User Acquisition Campaign",
        tasks=[],  # Tasks will be planned by agents
        context=context
    )

    print(f"Workflow created: {workflow.id}")
    print(f"   Name: {workflow.name}")
    print(f"   Tasks: {len(workflow.tasks)}")
    print(f"   Status: {workflow.status}")
    print()

    # For now, just create the workflow without executing it
    # to test if creation works properly
    print("Workflow creation successful!")
    print("Skipping execution for now to test creation flow.")
    print()

    # Show workflow details
    print("Workflow Details:")
    for task in workflow.tasks:
        print(f"  - Task: {task.id}")
        print(f"    Name: {task.name}")
        print(f"    Agent: {task.agent_type}")
        print(f"    Status: {task.status}")
        print(f"    Dependencies: {task.dependencies}")
        print()

    # Show system status
    print("Current System Status:")
    status = await orchestrator.get_system_status()

    print(f"  Initialized: {status['initialized']}")
    print(f"  Active Workflows: {status['workflow_metrics']['active_workflows']}")
    print(f"  Pending Workflows: {status['workflow_metrics']['pending_workflows']}")
    print(f"  LLM Backend Available: {status['components']['llm_backend']['available']}")
    print(f"  Storage Available: {status['components']['storage']['available']}")
    print()

    # Cleanup
    await orchestrator.shutdown()

    print("Demo completed successfully!")
    print("Thank you for trying OpenMonetization-UserAcquisition!")
    print()
    print("Next steps:")
    print("  - Try the CLI: omua --help")
    print("  - Run tests: pytest")
    print("  - Check the README for more examples")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
