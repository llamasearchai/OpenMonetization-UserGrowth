"""User Acquisition Agent - LLM-powered agent for planning and executing user acquisition campaigns."""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_agent import BaseAgent
from ..core.types import TaskSpec, ContextData, TaskStatus

logger = logging.getLogger(__name__)


class UserAcquisitionAgent(BaseAgent):
    """LLM-powered agent for user acquisition strategy planning and execution."""

    def __init__(self, llm_backend=None):
        """Initialize the user acquisition agent."""
        super().__init__(
            name="user_acquisition_agent",
            description="LLM-powered agent for user acquisition strategy planning and optimization",
            llm_backend=llm_backend
        )

    async def plan(self, context: ContextData) -> List[TaskSpec]:
        """Plan user acquisition tasks based on context using LLM.

        Args:
            context: Execution context containing campaign information

        Returns:
            List of planned tasks
        """
        logger.info(f"{self.name}: Planning user acquisition strategy for campaign {context.campaign_id}")

        # Create LLM prompt for planning
        planning_prompt = self._create_planning_prompt(context)

        try:
            # Get LLM response for task planning
            llm_response = await self._generate_with_llm(
                prompt=planning_prompt,
                context=self._context_to_dict(context),
                temperature=0.3,  # Lower temperature for more consistent planning
                max_tokens=1500
            )

            # Parse LLM response into tasks
            tasks = self._parse_planning_response(llm_response, context)

            logger.info(f"{self.name}: Planned {len(tasks)} user acquisition tasks")
            return tasks

        except Exception as e:
            logger.error(f"{self.name}: Failed to plan tasks: {e}")
            # Fallback to basic tasks if LLM fails
            return self._create_fallback_tasks(context)

    async def execute(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute a user acquisition task using LLM.

        Args:
            task: Task specification to execute
            context: Execution context

        Returns:
            Task execution result
        """
        logger.info(f"{self.name}: Executing task '{task.name}'")

        try:
            if "analysis" in task.name.lower():
                return await self._execute_analysis_task(task, context)
            elif "optimization" in task.name.lower():
                return await self._execute_optimization_task(task, context)
            elif "strategy" in task.name.lower():
                return await self._execute_strategy_task(task, context)
            elif "campaign" in task.name.lower():
                return await self._execute_campaign_task(task, context)
            else:
                # Generic task execution
                return await self._execute_generic_task(task, context)

        except Exception as e:
            logger.error(f"{self.name}: Task execution failed: {e}")
            return await self._create_task_result(
                task=task,
                result_data={"error": str(e)},
                status=TaskStatus.FAILED,
                metadata={"execution_error": True}
            )

    def _create_planning_prompt(self, context: ContextData) -> str:
        """Create a planning prompt for the LLM."""
        return f"""You are an expert user acquisition strategist. Based on the following context, plan a comprehensive user acquisition campaign strategy.

Context:
- Campaign ID: {context.campaign_id or 'N/A'}
- Channel: {context.channel or 'multi-channel'}
- User ID: {context.user_id or 'N/A'}
- Session ID: {context.session_id or 'N/A'}
- Metadata: {json.dumps(context.metadata, indent=2) if context.metadata else 'None'}

Your task is to plan a series of actionable tasks for optimizing user acquisition. Consider:

1. **Market Analysis**: Analyze target audience, competitors, and market conditions
2. **Channel Strategy**: Determine optimal channel mix and budget allocation
3. **Content Strategy**: Plan compelling content and messaging
4. **Campaign Optimization**: Set up A/B testing and performance monitoring
5. **Attribution Modeling**: Track and optimize conversion paths

Respond with a JSON array of tasks, where each task has:
- id: unique identifier
- name: descriptive name
- description: detailed description of what the task does
- parameters: execution parameters as key-value pairs
- dependencies: array of task IDs this task depends on (empty array if none)
- priority: 1-5 (5 being highest priority)

Example response format:
[
  {{
    "id": "market_analysis",
    "name": "Market Analysis",
    "description": "Analyze target market and competition",
    "parameters": {{"depth": "comprehensive"}},
    "dependencies": [],
    "priority": 5
  }},
  {{
    "id": "channel_optimization",
    "name": "Channel Optimization",
    "description": "Optimize channel mix and budget allocation",
    "parameters": {{"budget": 10000, "channels": ["paid_search", "social", "email"]}},
    "dependencies": ["market_analysis"],
    "priority": 4
  }}
]

Plan 3-6 specific, actionable tasks for this user acquisition campaign:"""

    def _parse_planning_response(self, llm_response: str, context: ContextData) -> List[TaskSpec]:
        """Parse LLM planning response into TaskSpec objects."""
        try:
            # Try to extract JSON from response
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON array found in response")

            json_str = llm_response[json_start:json_end]
            task_data = json.loads(json_str)

            tasks = []
            for item in task_data:
                task = self._create_task_spec(
                    task_id=item.get('id', f"{self.name}_{len(tasks)}"),
                    name=item.get('name', 'Unnamed Task'),
                    description=item.get('description', ''),
                    parameters=item.get('parameters', {}),
                    dependencies=item.get('dependencies', []),
                    priority=item.get('priority', 1)
                )
                tasks.append(task)

            return tasks

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM planning response: {e}")
            return self._create_fallback_tasks(context)

    def _create_fallback_tasks(self, context: ContextData) -> List[TaskSpec]:
        """Create fallback tasks when LLM planning fails."""
        logger.info(f"{self.name}: Using fallback task planning")

        tasks = [
            self._create_task_spec(
                task_id=f"{self.name}_analysis_{context.campaign_id}",
                name="User Acquisition Analysis",
                description="Analyze current user acquisition performance and identify opportunities",
                parameters={
                    "analysis_type": "comprehensive",
                    "focus_areas": ["channel_performance", "audience_behavior", "conversion_funnel"]
                },
                priority=4
            ),
            self._create_task_spec(
                task_id=f"{self.name}_strategy_{context.campaign_id}",
                name="Acquisition Strategy Development",
                description="Develop comprehensive user acquisition strategy based on analysis",
                parameters={
                    "strategy_type": "multi_channel",
                    "timeframe": "quarterly",
                    "budget_focus": True
                },
                dependencies=[f"{self.name}_analysis_{context.campaign_id}"],
                priority=5
            ),
            self._create_task_spec(
                task_id=f"{self.name}_optimization_{context.campaign_id}",
                name="Campaign Optimization",
                description="Optimize existing campaigns and implement improvements",
                parameters={
                    "optimization_goals": ["increase_conversion", "reduce_cac", "improve_retention"],
                    "testing_framework": "ab_testing"
                },
                dependencies=[f"{self.name}_strategy_{context.campaign_id}"],
                priority=3
            )
        ]

        return tasks

    async def _execute_analysis_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute an analysis task."""
        analysis_prompt = f"""You are a user acquisition analyst. Analyze the following campaign context and provide insights.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide a comprehensive analysis including:
1. Current performance assessment
2. Key opportunities and challenges
3. Competitive landscape insights
4. Recommended focus areas

Format your response as JSON with keys: assessment, opportunities, challenges, recommendations."""

        llm_response = await self._generate_with_llm(
            prompt=analysis_prompt,
            temperature=0.2,  # Lower temperature for analytical tasks
            max_tokens=1200
        )

        try:
            analysis_result = json.loads(llm_response)
        except json.JSONDecodeError:
            analysis_result = {
                "assessment": llm_response,
                "opportunities": ["Further analysis needed"],
                "challenges": ["Data parsing issues"],
                "recommendations": ["Retry with better context"]
            }

        return await self._create_task_result(
            task=task,
            result_data=analysis_result,
            metadata={"analysis_type": task.parameters.get("analysis_type", "general")}
        )

    async def _execute_optimization_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute an optimization task."""
        optimization_prompt = f"""You are a user acquisition optimizer. Based on the campaign context, provide specific optimization recommendations.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide concrete optimization recommendations including:
1. Channel reallocations with expected impact
2. Creative and messaging improvements
3. Technical optimizations
4. Testing recommendations

Format as JSON with keys: channel_optimization, creative_improvements, technical_optimizations, testing_recommendations."""

        llm_response = await self._generate_with_llm(
            prompt=optimization_prompt,
            temperature=0.3,
            max_tokens=1000
        )

        try:
            optimization_result = json.loads(llm_response)
        except json.JSONDecodeError:
            optimization_result = {
                "channel_optimization": ["Review current channel performance"],
                "creative_improvements": ["Test new creative variations"],
                "technical_optimizations": ["Optimize landing pages"],
                "testing_recommendations": ["Implement A/B testing framework"]
            }

        return await self._create_task_result(
            task=task,
            result_data=optimization_result,
            metadata={"optimization_focus": task.parameters.get("optimization_goals", [])}
        )

    async def _execute_strategy_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute a strategy development task."""
        strategy_prompt = f"""You are a user acquisition strategist. Develop a comprehensive acquisition strategy.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Develop a strategy including:
1. Target audience segmentation
2. Channel strategy and budget allocation
3. Content and messaging strategy
4. Timeline and milestones
5. Success metrics and KPIs

Format as JSON with keys: audience_segments, channel_strategy, content_strategy, timeline, success_metrics."""

        llm_response = await self._generate_with_llm(
            prompt=strategy_prompt,
            temperature=0.4,
            max_tokens=1400
        )

        try:
            strategy_result = json.loads(llm_response)
        except json.JSONDecodeError:
            strategy_result = {
                "audience_segments": ["General audience targeting"],
                "channel_strategy": ["Multi-channel approach"],
                "content_strategy": ["Value-driven messaging"],
                "timeline": ["Q1: Research, Q2: Execution, Q3: Optimization"],
                "success_metrics": ["CAC < $50", "LTV > $200", "Conversion > 3%"]
            }

        return await self._create_task_result(
            task=task,
            result_data=strategy_result,
            metadata={"strategy_type": task.parameters.get("strategy_type", "comprehensive")}
        )

    async def _execute_campaign_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute a campaign management task."""
        campaign_prompt = f"""You are a campaign manager. Plan and execute specific campaign tactics.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide specific campaign execution details including:
1. Campaign structure and variants
2. Targeting specifications
3. Creative requirements
4. Budget allocation details
5. Launch timeline

Format as JSON with keys: campaign_structure, targeting, creative_specs, budget_allocation, timeline."""

        llm_response = await self._generate_with_llm(
            prompt=campaign_prompt,
            temperature=0.5,
            max_tokens=1200
        )

        try:
            campaign_result = json.loads(llm_response)
        except json.JSONDecodeError:
            campaign_result = {
                "campaign_structure": ["Primary campaign with 3 variants"],
                "targeting": ["Interest-based targeting"],
                "creative_specs": ["Multiple ad formats"],
                "budget_allocation": ["$5000 paid search, $3000 social, $2000 email"],
                "timeline": ["Week 1: Launch, Week 2-4: Optimization"]
            }

        return await self._create_task_result(
            task=task,
            result_data=campaign_result,
            metadata={"campaign_type": "acquisition"}
        )

    async def _execute_generic_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute a generic task using LLM."""
        generic_prompt = f"""Execute the following user acquisition task based on the context.

Task: {task.name}
Description: {task.description}
Parameters: {json.dumps(task.parameters, indent=2)}

Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Provide a comprehensive response with actionable results."""

        llm_response = await self._generate_with_llm(
            prompt=generic_prompt,
            temperature=0.4,
            max_tokens=1000
        )

        result_data = {
            "task_execution": "completed",
            "response": llm_response,
            "parameters_used": task.parameters
        }

        return await self._create_task_result(
            task=task,
            result_data=result_data,
            metadata={"execution_type": "generic"}
        )

    def _context_to_dict(self, context: ContextData) -> Dict[str, Any]:
        """Convert ContextData to dictionary for LLM prompts."""
        return {
            "campaign_id": context.campaign_id,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "channel": context.channel,
            "metadata": context.metadata or {}
        }
