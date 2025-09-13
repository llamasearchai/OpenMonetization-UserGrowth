"""Social Media Agent - Specialized agent for social media advertising and organic growth."""

import json
import logging
from typing import List, Dict, Any, Optional

from .base_agent import BaseAgent
from ..core.types import TaskSpec, ContextData, TaskStatus

logger = logging.getLogger(__name__)


class SocialMediaAgent(BaseAgent):
    """Specialized agent for social media advertising and organic growth strategies."""

    def __init__(self, llm_backend=None):
        """Initialize the social media agent."""
        super().__init__(
            name="social_media_agent",
            description="Specialized agent for social media advertising and organic growth optimization",
            llm_backend=llm_backend
        )

    async def plan(self, context: ContextData) -> List[TaskSpec]:
        """Plan social media optimization tasks based on context."""
        logger.info(f"{self.name}: Planning social media strategy for campaign {context.campaign_id}")

        # Check if this is relevant for the current context
        if context.channel and "social" not in context.channel.lower() and "facebook" not in context.channel.lower():
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
            logger.info(f"{self.name}: Planned {len(tasks)} social media tasks")
            return tasks

        except Exception as e:
            logger.error(f"{self.name}: Failed to plan tasks: {e}")
            return self._create_fallback_tasks(context)

    async def execute(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute a social media task."""
        logger.info(f"{self.name}: Executing task '{task.name}'")

        try:
            if "content" in task.name.lower() or "creative" in task.name.lower():
                return await self._execute_content_task(task, context)
            elif "audience" in task.name.lower() or "targeting" in task.name.lower():
                return await self._execute_targeting_task(task, context)
            elif "campaign" in task.name.lower() or "ad" in task.name.lower():
                return await self._execute_campaign_task(task, context)
            elif "organic" in task.name.lower() or "growth" in task.name.lower():
                return await self._execute_organic_growth_task(task, context)
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
        """Create planning prompt for social media optimization."""
        return f"""You are a social media marketing expert. Plan a comprehensive social media acquisition strategy.

Context:
- Campaign ID: {context.campaign_id or 'N/A'}
- Channel: {context.channel or 'social_media'}
- Objectives: {context.metadata.get('objectives', []) if context.metadata else []}

Plan actionable social media tasks including:
1. Content strategy and creative development
2. Audience targeting and segmentation
3. Paid campaign structure and optimization
4. Organic growth and engagement strategies
5. Cross-platform campaign management
6. Performance monitoring and analytics

Consider current social media best practices:
- Platform-specific content optimization
- Lookalike audience creation
- Video content utilization
- Influencer partnership opportunities
- Community building and engagement
- Multi-platform campaign coordination

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
        """Create minimal social media tasks when not the primary focus."""
        return [
            self._create_task_spec(
                task_id=f"{self.name}_monitor_{context.campaign_id}",
                name="Social Media Performance Monitoring",
                description="Monitor social media campaign performance and engagement metrics",
                parameters={"monitoring_focus": "engagement_and_conversions"},
                priority=2
            )
        ]

    def _create_fallback_tasks(self, context: ContextData) -> List[TaskSpec]:
        """Create fallback tasks when LLM planning fails."""
        return [
            self._create_task_spec(
                task_id=f"{self.name}_content_strategy_{context.campaign_id}",
                name="Social Content Strategy Development",
                description="Develop compelling content strategy for social media platforms",
                parameters={
                    "platforms": ["facebook", "instagram", "twitter", "linkedin"],
                    "content_types": ["video", "carousel", "stories", "static"],
                    "posting_frequency": "daily"
                },
                priority=4
            ),
            self._create_task_spec(
                task_id=f"{self.name}_audience_targeting_{context.campaign_id}",
                name="Audience Targeting Optimization",
                description="Optimize audience targeting across social platforms",
                parameters={
                    "targeting_methods": ["interests", "behaviors", "lookalike_audiences"],
                    "audience_overlap_analysis": True,
                    "retargeting_setup": True
                },
                priority=5
            ),
            self._create_task_spec(
                task_id=f"{self.name}_paid_campaign_{context.campaign_id}",
                name="Paid Social Campaign Setup",
                description="Set up and optimize paid social media campaigns",
                parameters={
                    "campaign_objectives": ["conversions", "traffic", "engagement"],
                    "budget_allocation": "platform_specific",
                    "a_b_testing": True
                },
                dependencies=[f"{self.name}_audience_targeting_{context.campaign_id}"],
                priority=4
            ),
            self._create_task_spec(
                task_id=f"{self.name}_organic_growth_{context.campaign_id}",
                name="Organic Growth Strategy",
                description="Implement organic growth tactics and community building",
                parameters={
                    "growth_tactics": ["user_generated_content", "influencer_partnerships", "community_engagement"],
                    "engagement_goals": ["increase_followers", "boost_interaction_rate"],
                    "community_building": True
                },
                priority=3
            )
        ]

    async def _execute_content_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute content strategy and creative development task."""
        content_prompt = f"""You are a social media content strategist. Create compelling content strategies and creative concepts.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide content strategy recommendations including:
1. Platform-specific content themes and formats
2. Creative concepts and campaign ideas
3. Content calendar recommendations
4. Visual design guidelines
5. Copywriting frameworks and messaging

Format as JSON with keys: content_themes, creative_concepts, content_calendar, design_guidelines, messaging_framework."""

        llm_response = await self._generate_with_llm(
            prompt=content_prompt,
            temperature=0.7,  # Higher creativity for content
            max_tokens=1400
        )

        try:
            content_result = json.loads(llm_response)
        except json.JSONDecodeError:
            content_result = {
                "content_themes": ["User success stories", "Industry insights", "Product tutorials"],
                "creative_concepts": ["Before/after transformations", "Day in the life series"],
                "content_calendar": ["Monday: Educational content", "Wednesday: Engagement posts"],
                "design_guidelines": ["Consistent brand colors", "Mobile-first design"],
                "messaging_framework": ["Problem-solution-benefit structure", "Storytelling approach"]
            }

        return await self._create_task_result(
            task=task,
            result_data=content_result,
            metadata={"content_pieces": len(content_result.get("content_themes", []))}
        )

    async def _execute_targeting_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute audience targeting optimization task."""
        targeting_prompt = f"""You are a social media targeting expert. Optimize audience targeting for maximum campaign effectiveness.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide targeting optimization recommendations including:
1. Interest-based targeting strategies
2. Behavioral targeting opportunities
3. Lookalike audience creation
4. Custom audience development
5. Cross-platform audience coordination

Format as JSON with keys: interest_targeting, behavioral_targeting, lookalike_audiences, custom_audiences, cross_platform_coordination."""

        llm_response = await self._generate_with_llm(
            prompt=targeting_prompt,
            temperature=0.3,
            max_tokens=1200
        )

        try:
            targeting_result = json.loads(llm_response)
        except json.JSONDecodeError:
            targeting_result = {
                "interest_targeting": ["Technology enthusiasts", "Business growth topics"],
                "behavioral_targeting": ["Recent website visitors", "Content engagers"],
                "lookalike_audiences": ["Top 1% converters", "High-value customers"],
                "custom_audiences": ["Email subscribers", "Past purchasers"],
                "cross_platform_coordination": ["Unified audience naming", "Consistent targeting criteria"]
            }

        return await self._create_task_result(
            task=task,
            result_data=targeting_result,
            metadata={"targeting_segments": len(targeting_result.get("interest_targeting", []))}
        )

    async def _execute_campaign_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute paid social campaign setup and optimization task."""
        campaign_prompt = f"""You are a paid social campaign manager. Design and optimize paid social media campaigns.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide campaign setup and optimization recommendations including:
1. Campaign structure and objectives
2. Budget allocation across platforms
3. Ad format recommendations
4. Bidding and optimization strategies
5. Performance tracking setup

Format as JSON with keys: campaign_structure, budget_allocation, ad_formats, bidding_strategy, tracking_setup."""

        llm_response = await self._generate_with_llm(
            prompt=campaign_prompt,
            temperature=0.4,
            max_tokens=1200
        )

        try:
            campaign_result = json.loads(llm_response)
        except json.JSONDecodeError:
            campaign_result = {
                "campaign_structure": ["Awareness campaign", "Consideration campaign", "Conversion campaign"],
                "budget_allocation": {"facebook": "40%", "instagram": "35%", "linkedin": "25%"},
                "ad_formats": ["Carousel ads", "Video ads", "Lead generation forms"],
                "bidding_strategy": ["Lowest cost for objective", "Target cost bidding"],
                "tracking_setup": ["Facebook Pixel", "Conversion API", "Custom conversion events"]
            }

        return await self._create_task_result(
            task=task,
            result_data=campaign_result,
            metadata={"campaign_objectives": len(campaign_result.get("campaign_structure", []))}
        )

    async def _execute_organic_growth_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute organic growth and community building task."""
        organic_prompt = f"""You are a social media growth expert. Develop organic growth strategies and community building tactics.

Campaign Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Task Parameters:
{json.dumps(task.parameters, indent=2)}

Provide organic growth recommendations including:
1. Community engagement strategies
2. Influencer and partnership opportunities
3. User-generated content campaigns
4. Viral content creation tactics
5. Community management approaches

Format as JSON with keys: engagement_strategies, partnership_opportunities, ugc_campaigns, viral_tactics, community_management."""

        llm_response = await self._generate_with_llm(
            prompt=organic_prompt,
            temperature=0.6,
            max_tokens=1200
        )

        try:
            organic_result = json.loads(llm_response)
        except json.JSONDecodeError:
            organic_result = {
                "engagement_strategies": ["Daily Q&A sessions", "Poll and quiz posts", "Live video streams"],
                "partnership_opportunities": ["Industry influencers", "Complementary brands"],
                "ugc_campaigns": ["User story features", "Contest and giveaways", "Hashtag challenges"],
                "viral_tactics": ["Trending topic integration", "Emotional storytelling", "Shareable content"],
                "community_management": ["24/7 response time", "Content moderation guidelines", "Community guidelines"]
            }

        return await self._create_task_result(
            task=task,
            result_data=organic_result,
            metadata={"growth_tactics": len(organic_result.get("engagement_strategies", []))}
        )

    async def _execute_generic_task(self, task: TaskSpec, context: ContextData) -> Any:
        """Execute generic social media task."""
        generic_prompt = f"""Execute this social media marketing task.

Task: {task.name}
Description: {task.description}
Parameters: {json.dumps(task.parameters, indent=2)}

Context:
{json.dumps(self._context_to_dict(context), indent=2)}

Provide actionable social media recommendations and implementation steps."""

        llm_response = await self._generate_with_llm(
            prompt=generic_prompt,
            temperature=0.4,
            max_tokens=1000
        )

        result_data = {
            "task_execution": "completed",
            "recommendations": llm_response,
            "platforms_focused": ["facebook", "instagram", "twitter", "linkedin"],
            "implementation_steps": ["Plan content calendar", "Create assets", "Schedule posts", "Monitor engagement"]
        }

        return await self._create_task_result(
            task=task,
            result_data=result_data,
            metadata={"channel": "social_media"}
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
