"""Paid Search Agent - Specialized agent for paid search advertising campaigns."""

import json
import logging
from typing import List, Dict, Any, Optional

from .base_agent import BaseAgent
from ..core.types import TaskSpec, ContextData, TaskStatus

logger = logging.getLogger(__name__)


class PaidSearchAgent(BaseAgent):
    """Specialized agent for paid search advertising (Google Ads, Bing Ads, etc.)."""

    def __init__(self, llm_backend=None):
        """Initialize the paid search agent."""
        super().__init__(
            name="paid_search_agent",
            description="Specialized agent for paid search advertising optimization and management",
            llm_backend=llm_backend
        )

    async def plan(self, context: ContextData) -> List[TaskSpec]:
        """Plan paid search optimization tasks based on context."""
        logger.info(f"{self.name}: Planning paid search strategy for campaign {context.campaign_id}")

        # Check if this is relevant for the current context
        if context.channel and "paid" not in context.channel.lower() and "search" not in context.channel.lower():
            # Return minimal tasks if not focused on paid search
            return self._create_minimal_tasks(context)

        planning_prompt = self._create_planning_prompt(context)

        try:
            llm_response = await self._generate_with_llm(
                prompt=planning_prompt,
                context=self._context_to_dict(context),
                temperature=0.3,
                max_tokens=1500
            )

            tasks = self._parse_planning_response(llm_response, context)
            logger.info(f"{self.name}: Planned {len(tasks)} paid search tasks")
            return tasks

        except Exception as e:
            logger.error(f"{self.name}: Failed to plan tasks: {e}")
            return self._create_fallback_tasks(context)

    async def execute(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute a paid search task."""
        logger.info(f"{self.name}: Executing task '{task.name}'")

        try:
            if "keyword" in task.name.lower():
                return await self._execute_keyword_task(task, context)
            elif "bid" in task.name.lower() or "budget" in task.name.lower():
                return await self._execute_bid_management_task(task, context)
            elif "creative" in task.name.lower() or "ad" in task.name.lower():
                return await self._execute_ad_creative_task(task, context)
            elif "audience" in task.name.lower() or "targeting" in task.name.lower():
                return await self._execute_targeting_task(task, context)
            else:
                return await self._execute_generic_task(task, context)

        except Exception as e:
            logger.error(f"{self.name}: Task execution failed: {e}")
            return await self._create_task_result(
                task=task,
                result_data={"error": str(e)},
                status=TaskStatus.FAILED
            )

    def _create_planning_prompt(self, context: ContextData) -> str:
        """Create planning prompt for paid search optimization."""
        return f"""You are a paid search advertising expert. Plan a comprehensive paid search campaign strategy.

Context:
- Campaign ID: {context.campaign_id or 'N/A'}
- Channel: {context.channel or 'paid_search'}
- Objectives: {context.metadata.get('objectives', []) if context.metadata else []}

Plan actionable paid search tasks including:
1. Keyword research and expansion
2. Bid management and budget optimization
3. Ad creative optimization
4. Audience targeting improvements
5. Performance monitoring and reporting

Consider current paid search best practices:
- Long-tail keyword targeting
- Quality Score optimization
- Ad extensions utilization
- Negative keyword management
- Conversion tracking setup
- A/B testing for ad variations

Respond with JSON array of tasks with id, name, description, parameters, dependencies, priority."""

    def _parse_planning_response(self, llm_response: str, context: ContextData) -> List[TaskSpec]:
        """Parse LLM planning response into TaskSpec objects."""
        try:
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']') + 1
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

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._create_fallback_tasks(context)

    def _create_minimal_tasks(self, context: ContextData) -> List[TaskSpec]:
        """Create minimal paid search tasks when not the primary focus."""
        return [
            self._create_task_spec(
                task_id=f"{self.name}_monitor_{context.campaign_id}",
                name="Paid Search Performance Monitoring",
                description="Monitor paid search campaign performance and identify optimization opportunities",
                parameters={"monitoring_focus": "key_metrics"},
                priority=2
            )
        ]

    def _create_fallback_tasks(self, context: ContextData) -> List[TaskSpec]:
        """Create fallback tasks when LLM planning fails."""
        return [
            self._create_task_spec(
                task_id=f"{self.name}_keyword_research_{context.campaign_id}",
                name="Keyword Research and Analysis",
                description="Research and analyze keywords for paid search campaigns",
                parameters={
                    "research_type": "comprehensive",
                    "target_volume": "high_medium",
                    "competition_level": "medium"
                },
                priority=4
            ),
            self._create_task_spec(
                task_id=f"{self.name}_bid_optimization_{context.campaign_id}",
                name="Bid Management Optimization",
                description="Optimize bids and budget allocation across keywords and campaigns",
                parameters={
                    "optimization_goal": "maximize_conversions",
                    "budget_constraint": True,
                    "automated_bidding": True
                },
                dependencies=[f"{self.name}_keyword_research_{context.campaign_id}"],
                priority=5
            ),
            self._create_task_spec(
                task_id=f"{self.name}_ad_creative_{context.campaign_id}",
                name="Ad Creative Optimization",
                description="Create and optimize ad creatives for better performance",
                parameters={
                    "creative_types": ["responsive_search", "text_ads"],
                    "testing_variants": 3,
                    "focus_metrics": ["ctr", "conversion_rate"]
                },
                priority=3
            )
        ]

    async def _execute_keyword_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute keyword research and analysis task."""
        keyword_prompt = f"""You are a keyword research expert for paid search. Analyze and recommend keywords for this campaign.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide keyword recommendations including:
1. Primary keywords with search volume and competition
2. Long-tail keyword opportunities
3. Negative keywords to exclude
4. Keyword grouping and theme identification
5. Estimated CPC and conversion potential

Format as JSON with keys: primary_keywords, long_tail_opportunities, negative_keywords, keyword_groups, cpc_estimates."""

        llm_response = await self._generate_with_llm(
            prompt=keyword_prompt,
            temperature=0.2,
            max_tokens=1200
        )

        try:
            keyword_result = json.loads(llm_response)
        except json.JSONDecodeError:
            keyword_result = {
                "primary_keywords": ["user acquisition", "customer acquisition"],
                "long_tail_opportunities": ["best user acquisition strategies 2024"],
                "negative_keywords": ["free", "cheap"],
                "keyword_groups": ["strategy", "tactics", "tools"],
                "cpc_estimates": {"high": 2.50, "medium": 1.20, "low": 0.50}
            }

        return await self._create_task_result(
            task=task,
            result_data=keyword_result,
            metadata={"keyword_count": len(keyword_result.get("primary_keywords", []))}
        )

    async def _execute_bid_management_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute bid management and budget optimization task."""
        bid_prompt = f"""You are a paid search bid management expert. Optimize bidding strategy and budget allocation.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide bid optimization recommendations including:
1. Bid strategy recommendations (manual CPC, automated bidding, etc.)
2. Budget allocation across campaigns and ad groups
3. Bid adjustments by device, location, and time
4. Performance targets and thresholds
5. Budget pacing recommendations

Format as JSON with keys: bid_strategy, budget_allocation, bid_adjustments, performance_targets, budget_pacing."""

        llm_response = await self._generate_with_llm(
            prompt=bid_prompt,
            temperature=0.3,
            max_tokens=1000
        )

        try:
            bid_result = json.loads(llm_response)
        except json.JSONDecodeError:
            bid_result = {
                "bid_strategy": "Target CPA with automated bidding",
                "budget_allocation": {"campaign_a": "60%", "campaign_b": "40%"},
                "bid_adjustments": {"mobile": "+20%", "desktop": "-10%"},
                "performance_targets": {"target_cpa": 25.00, "target_roas": 3.0},
                "budget_pacing": "Accelerated delivery for first 3 days, then even pacing"
            }

        return await self._create_task_result(
            task=task,
            result_data=bid_result,
            metadata={"optimization_type": "bid_management"}
        )

    async def _execute_ad_creative_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute ad creative optimization task."""
        creative_prompt = f"""You are a paid search ad creative expert. Create and optimize ad copy for better performance.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide ad creative recommendations including:
1. Headline variations (30 characters max each)
2. Description variations (90 characters max each)
3. Call-to-action recommendations
4. Ad extension suggestions
5. A/B testing recommendations

Format as JSON with keys: headlines, descriptions, calls_to_action, extensions, testing_recommendations."""

        llm_response = await self._generate_with_llm(
            prompt=creative_prompt,
            temperature=0.6,  # Higher creativity for ad copy
            max_tokens=1200
        )

        try:
            creative_result = json.loads(llm_response)
        except json.JSONDecodeError:
            creative_result = {
                "headlines": ["Optimize User Acquisition", "Grow Your User Base Fast", "Scale User Acquisition"],
                "descriptions": ["Expert strategies for user acquisition success", "Proven tactics to acquire more users"],
                "calls_to_action": ["Learn More", "Get Started", "Download Guide"],
                "extensions": ["sitelink", "callout", "structured_snippet"],
                "testing_recommendations": ["Test 3 headline variations", "A/B test CTA buttons"]
            }

        return await self._create_task_result(
            task=task,
            result_data=creative_result,
            metadata={"creative_variants": len(creative_result.get("headlines", []))}
        )

    async def _execute_targeting_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute audience targeting optimization task."""
        targeting_prompt = f"""You are an audience targeting expert for paid search. Optimize audience targeting for better campaign performance.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide targeting optimization recommendations including:
1. Geographic targeting adjustments
2. Device and platform targeting
3. Audience segment identification
4. Demographic targeting suggestions
5. Custom audience creation ideas

Format as JSON with keys: geographic_targeting, device_targeting, audience_segments, demographic_targeting, custom_audiences."""

        llm_response = await self._generate_with_llm(
            prompt=targeting_prompt,
            temperature=0.3,
            max_tokens=1000
        )

        try:
            targeting_result = json.loads(llm_response)
        except json.JSONDecodeError:
            targeting_result = {
                "geographic_targeting": ["Target top 50 US markets", "Expand to Canada"],
                "device_targeting": ["+20% mobile bid adjustment", "Focus on smartphones"],
                "audience_segments": ["B2B decision makers", "marketing professionals"],
                "demographic_targeting": ["Age 25-54", "Income $75k+"],
                "custom_audiences": ["Website visitors", "Email list subscribers"]
            }

        return await self._create_task_result(
            task=task,
            result_data=targeting_result,
            metadata={"targeting_type": "comprehensive"}
        )

    async def _execute_generic_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute generic paid search task."""
        generic_prompt = f"""Execute this paid search advertising task.

Task: {task.name}
Description: {task.description}
Parameters: {json.dumps(task.parameters, indent=2)}

Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Provide actionable recommendations and implementation steps."""

        llm_response = await self._generate_with_llm(
            prompt=generic_prompt,
            temperature=0.4,
            max_tokens=1000
        )

        result_data = {
            "task_execution": "completed",
            "recommendations": llm_response,
            "implementation_steps": ["Review recommendations", "Implement changes", "Monitor performance"]
        }

        return await self._create_task_result(
            task=task,
            result_data=result_data,
            metadata={"channel": "paid_search"}
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
