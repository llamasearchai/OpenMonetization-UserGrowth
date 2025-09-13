"""A/B Testing Framework for campaign optimization."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4
import json
import statistics

from ..core.types import ContextData
from ..storage import SQLiteStorageBackend

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(Enum):
    """Type of experiment variant."""
    CONTROL = "control"
    TREATMENT = "treatment"


@dataclass
class Variant:
    """A single variant in an A/B test."""
    id: str
    name: str
    variant_type: VariantType
    configuration: Dict[str, Any]
    traffic_percentage: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    sample_size: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def update_metrics(self, metric_name: str, value: float, count: int = 1) -> None:
        """Update metrics for this variant."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = {"values": [], "count": 0, "sum": 0.0}

        metric_data = self.metrics[metric_name]
        metric_data["values"].append(value)
        metric_data["count"] += count
        metric_data["sum"] += value * count

    def get_metric_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Get statistical summary for a metric."""
        if metric_name not in self.metrics:
            return None

        metric_data = self.metrics[metric_name]
        values = metric_data["values"]

        if not values:
            return None

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "total": metric_data["sum"]
        }


@dataclass
class Experiment:
    """An A/B testing experiment."""
    id: str
    name: str
    description: str
    campaign_id: str
    status: ExperimentStatus
    variants: List[Variant]
    primary_metric: str
    secondary_metrics: List[str] = field(default_factory=list)
    target_sample_size: int = 1000
    confidence_level: float = 0.95
    minimum_effect_size: float = 0.05
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    winner_variant_id: Optional[str] = None
    statistical_significance: Optional[float] = None

    def start_experiment(self) -> None:
        """Start the experiment."""
        if self.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment with status: {self.status}")

        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now()
        logger.info(f"Started experiment: {self.id}")

    def pause_experiment(self) -> None:
        """Pause the running experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot pause experiment with status: {self.status}")

        self.status = ExperimentStatus.PAUSED
        logger.info(f"Paused experiment: {self.id}")

    def resume_experiment(self) -> None:
        """Resume a paused experiment."""
        if self.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Cannot resume experiment with status: {self.status}")

        self.status = ExperimentStatus.RUNNING
        logger.info(f"Resumed experiment: {self.id}")

    def complete_experiment(self, winner_variant_id: Optional[str] = None) -> None:
        """Complete the experiment."""
        if self.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            raise ValueError(f"Cannot complete experiment with status: {self.status}")

        self.status = ExperimentStatus.COMPLETED
        self.completed_at = datetime.now()
        self.winner_variant_id = winner_variant_id
        logger.info(f"Completed experiment: {self.id}")

    def cancel_experiment(self) -> None:
        """Cancel the experiment."""
        if self.status in [ExperimentStatus.COMPLETED, ExperimentStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel experiment with status: {self.status}")

        self.status = ExperimentStatus.CANCELLED
        self.completed_at = datetime.now()
        logger.info(f"Cancelled experiment: {self.id}")

    def get_control_variant(self) -> Optional[Variant]:
        """Get the control variant."""
        for variant in self.variants:
            if variant.variant_type == VariantType.CONTROL:
                return variant
        return None

    def get_treatment_variants(self) -> List[Variant]:
        """Get all treatment variants."""
        return [v for v in self.variants if v.variant_type == VariantType.TREATMENT]

    def get_variant_by_id(self, variant_id: str) -> Optional[Variant]:
        """Get a variant by ID."""
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        return None

    def record_conversion(self, variant_id: str, metric_name: str = "conversion", value: float = 1.0) -> None:
        """Record a conversion for a variant."""
        variant = self.get_variant_by_id(variant_id)
        if not variant:
            raise ValueError(f"Variant not found: {variant_id}")

        variant.update_metrics(metric_name, value)
        variant.sample_size += 1

    def should_stop_experiment(self) -> Tuple[bool, Optional[str], Optional[float]]:
        """Check if experiment should be stopped based on statistical significance."""
        if self.status != ExperimentStatus.RUNNING:
            return False, None, None

        control = self.get_control_variant()
        treatments = self.get_treatment_variants()

        if not control or not treatments:
            return False, None, None

        # Check if we have minimum sample size
        total_sample = sum(v.sample_size for v in self.variants)
        if total_sample < self.target_sample_size:
            return False, None, None

        # Simple statistical significance check (in real implementation, use proper statistical tests)
        control_rate = control.get_metric_stats(self.primary_metric)
        if not control_rate:
            return False, None, None

        control_mean = control_rate["mean"]

        for treatment in treatments:
            treatment_stats = treatment.get_metric_stats(self.primary_metric)
            if not treatment_stats:
                continue

            treatment_mean = treatment_stats["mean"]

            # Calculate relative improvement
            if control_mean > 0:
                improvement = (treatment_mean - control_mean) / control_mean

                # Check if improvement exceeds minimum effect size
                if abs(improvement) >= self.minimum_effect_size:
                    # Simple significance check based on sample size and variance
                    # In production, use proper statistical tests (t-test, chi-square, etc.)
                    control_std = control_rate.get("std_dev", 0)
                    treatment_std = treatment_stats.get("std_dev", 0)

                    # Basic significance approximation
                    if control_std > 0 and treatment_std > 0:
                        # Simplified z-score approximation
                        se = ((control_std ** 2 / control.sample_size) +
                              (treatment_std ** 2 / treatment.sample_size)) ** 0.5
                        if se > 0:
                            z_score = abs(treatment_mean - control_mean) / se
                            # Approximate p-value (rough estimate)
                            significance = min(1.0, 2 * (1 - self._normal_cdf(z_score)))

                            if significance <= (1 - self.confidence_level):
                                winner = treatment.id if treatment_mean > control_mean else control.id
                                return True, winner, significance

        return False, None, None

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal cumulative distribution function."""
        # Abramowitz and Stegun approximation
        a1 =  0.254829592
        a2 = -0.284496736
        a3 =  1.421413741
        a4 = -1.453152027
        a5 =  1.061405429
        p  =  0.3275911

        sign = 1 if x >= 0 else -1
        x = abs(x) / (2 ** 0.5)

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp() if hasattr((-x * x), 'exp') else 1

        return 0.5 * (1 + sign * y)

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment results."""
        summary = {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "primary_metric": self.primary_metric,
            "total_sample_size": sum(v.sample_size for v in self.variants),
            "target_sample_size": self.target_sample_size,
            "variants": []
        }

        for variant in self.variants:
            variant_summary = {
                "id": variant.id,
                "name": variant.name,
                "type": variant.variant_type.value,
                "traffic_percentage": variant.traffic_percentage,
                "sample_size": variant.sample_size,
                "metrics": {}
            }

            for metric_name in [self.primary_metric] + self.secondary_metrics:
                stats = variant.get_metric_stats(metric_name)
                if stats:
                    variant_summary["metrics"][metric_name] = stats

            summary["variants"].append(variant_summary)

        if self.winner_variant_id:
            summary["winner"] = self.winner_variant_id

        if self.statistical_significance is not None:
            summary["statistical_significance"] = self.statistical_significance

        return summary


class ABTestingManager:
    """Manager for A/B testing experiments."""

    def __init__(self, storage: Optional[SQLiteStorageBackend] = None):
        """Initialize the A/B testing manager."""
        self.storage = storage or SQLiteStorageBackend()
        self._experiments: Dict[str, Experiment] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the A/B testing manager."""
        if self._initialized:
            return

        await self.storage.initialize()
        await self._load_experiments()
        self._initialized = True
        logger.info("A/B Testing manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the A/B testing manager."""
        if not self._initialized:
            return

        await self._save_experiments()
        self._initialized = False
        logger.info("A/B Testing manager shutdown")

    async def create_experiment(
        self,
        name: str,
        description: str,
        campaign_id: str,
        variants_config: List[Dict[str, Any]],
        primary_metric: str,
        secondary_metrics: Optional[List[str]] = None,
        target_sample_size: int = 1000,
        confidence_level: float = 0.95,
        minimum_effect_size: float = 0.05
    ) -> Experiment:
        """Create a new A/B testing experiment."""
        await self._ensure_initialized()

        # Validate variants configuration
        if len(variants_config) < 2:
            raise ValueError("Experiment must have at least 2 variants")

        control_count = sum(1 for v in variants_config if v.get("type") == "control")
        if control_count != 1:
            raise ValueError("Experiment must have exactly one control variant")

        # Check traffic percentage totals
        total_traffic = sum(v.get("traffic_percentage", 0) for v in variants_config)
        if abs(total_traffic - 100.0) > 0.1:
            raise ValueError("Variant traffic percentages must total 100%")

        # Create variants
        variants = []
        for config in variants_config:
            variant = Variant(
                id=str(uuid4()),
                name=config["name"],
                variant_type=VariantType(config["type"]),
                configuration=config.get("configuration", {}),
                traffic_percentage=config["traffic_percentage"]
            )
            variants.append(variant)

        # Create experiment
        experiment = Experiment(
            id=str(uuid4()),
            name=name,
            description=description,
            campaign_id=campaign_id,
            status=ExperimentStatus.DRAFT,
            variants=variants,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or [],
            target_sample_size=target_sample_size,
            confidence_level=confidence_level,
            minimum_effect_size=minimum_effect_size
        )

        self._experiments[experiment.id] = experiment
        await self._save_experiment(experiment)

        logger.info(f"Created experiment: {experiment.id} - {name}")
        return experiment

    async def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment."""
        await self._ensure_initialized()

        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        experiment.start_experiment()
        await self._save_experiment(experiment)
        return experiment

    async def pause_experiment(self, experiment_id: str) -> Experiment:
        """Pause a running experiment."""
        await self._ensure_initialized()

        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        experiment.pause_experiment()
        await self._save_experiment(experiment)
        return experiment

    async def stop_experiment(self, experiment_id: str) -> Experiment:
        """Stop an experiment and determine winner."""
        await self._ensure_initialized()

        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        should_stop, winner_id, significance = experiment.should_stop_experiment()

        if should_stop:
            experiment.complete_experiment(winner_id)
            experiment.statistical_significance = significance
        else:
            experiment.cancel_experiment()

        await self._save_experiment(experiment)
        return experiment

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        await self._ensure_initialized()
        return self._experiments.get(experiment_id)

    async def list_experiments(
        self,
        campaign_id: Optional[str] = None,
        status: Optional[ExperimentStatus] = None
    ) -> List[Experiment]:
        """List experiments with optional filtering."""
        await self._ensure_initialized()

        experiments = list(self._experiments.values())

        if campaign_id:
            experiments = [e for e in experiments if e.campaign_id == campaign_id]

        if status:
            experiments = [e for e in experiments if e.status == status]

        return experiments

    async def record_event(
        self,
        experiment_id: str,
        variant_id: str,
        metric_name: str,
        value: float = 1.0
    ) -> None:
        """Record an event/conversion for an experiment variant."""
        await self._ensure_initialized()

        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")

        if experiment.status != ExperimentStatus.RUNNING:
            return  # Silently ignore events for non-running experiments

        experiment.record_conversion(variant_id, metric_name, value)
        await self._save_experiment(experiment)

        # Check if experiment should be stopped
        should_stop, winner_id, significance = experiment.should_stop_experiment()
        if should_stop:
            experiment.complete_experiment(winner_id)
            experiment.statistical_significance = significance
            await self._save_experiment(experiment)
            logger.info(f"Experiment {experiment_id} completed automatically. Winner: {winner_id}")

    async def get_variant_for_user(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[Variant]:
        """Get the variant assignment for a user (deterministic based on user ID)."""
        await self._ensure_initialized()

        experiment = self._experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None

        # Simple deterministic assignment based on user ID hash
        # In production, use a proper hashing algorithm for consistency
        user_hash = hash(user_id) % 100

        cumulative_percentage = 0.0
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage
            if user_hash < cumulative_percentage * 100:
                return variant

        # Fallback to first variant
        return experiment.variants[0] if experiment.variants else None

    async def get_experiment_results(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive results for an experiment."""
        await self._ensure_initialized()

        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return None

        return experiment.get_experiment_summary()

    async def _ensure_initialized(self) -> None:
        """Ensure the manager is initialized."""
        if not self._initialized:
            await self.initialize()

    async def _load_experiments(self) -> None:
        """Load experiments from storage."""
        try:
            # In a real implementation, this would load from database
            # For now, experiments are kept in memory
            pass
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")

    async def _save_experiment(self, experiment: Experiment) -> None:
        """Save an experiment to storage."""
        try:
            # In a real implementation, this would save to database
            # For now, experiments are kept in memory
            pass
        except Exception as e:
            logger.error(f"Failed to save experiment {experiment.id}: {e}")

    async def _save_experiments(self) -> None:
        """Save all experiments to storage."""
        # Implementation would save all experiments
        pass
