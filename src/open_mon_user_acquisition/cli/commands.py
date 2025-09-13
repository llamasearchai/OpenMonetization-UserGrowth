"""Additional CLI commands for OpenMonetization-UserAcquisition."""

import json
from typing import Optional, List, Any
from pathlib import Path

from ..experiments import ABTestingManager

# Global A/B testing manager instance
_ab_testing_manager: Optional[ABTestingManager] = None


def get_ab_testing_manager() -> ABTestingManager:
    """Get or create the global A/B testing manager instance."""
    global _ab_testing_manager
    if _ab_testing_manager is None:
        _ab_testing_manager = ABTestingManager()
    return _ab_testing_manager


async def init_ab_testing():
    """Initialize the A/B testing manager."""
    manager = get_ab_testing_manager()
    await manager.initialize()
    return manager


async def ab_create(
    name: str,
    campaign_id: str,
    config_file: str,
    description: Optional[str] = None,
    primary_metric: str = "conversion_rate",
    target_sample_size: int = 1000,
    json_output: bool = False
) -> None:
    """Create a new A/B testing experiment."""
    try:
        manager = await init_ab_testing()

        # Load configuration from file
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, 'r') as f:
            config = json.load(f)

        variants_config = config.get("variants", [])
        if not variants_config:
            raise ValueError("Configuration must include 'variants' array")

        experiment = await manager.create_experiment(
            name=name,
            description=description or f"A/B test for campaign {campaign_id}",
            campaign_id=campaign_id,
            variants_config=variants_config,
            primary_metric=primary_metric,
            secondary_metrics=config.get("secondary_metrics", []),
            target_sample_size=target_sample_size,
            confidence_level=config.get("confidence_level", 0.95),
            minimum_effect_size=config.get("minimum_effect_size", 0.05)
        )

        if json_output:
            print(json.dumps(experiment.get_experiment_summary(), indent=2))
        else:
            print(f"✅ Created experiment: {experiment.id}")
            print(f"   Name: {experiment.name}")
            print(f"   Campaign: {experiment.campaign_id}")
            print(f"   Variants: {len(experiment.variants)}")
            print(f"   Primary Metric: {experiment.primary_metric}")
            print(f"   Target Sample Size: {experiment.target_sample_size}")

    except Exception as e:
        print(f"❌ Error creating experiment: {e}")
        raise


async def ab_start(experiment_id: str, json_output: bool = False) -> None:
    """Start an A/B testing experiment."""
    try:
        manager = await init_ab_testing()
        experiment = await manager.start_experiment(experiment_id)

        if json_output:
            print(json.dumps(experiment.get_experiment_summary(), indent=2))
        else:
            print(f"✅ Started experiment: {experiment.id}")
            print(f"   Status: {experiment.status.value}")

    except Exception as e:
        print(f"❌ Error starting experiment: {e}")
        raise


async def ab_stop(experiment_id: str, json_output: bool = False) -> None:
    """Stop an A/B testing experiment."""
    try:
        manager = await init_ab_testing()
        experiment = await manager.stop_experiment(experiment_id)

        if json_output:
            print(json.dumps(experiment.get_experiment_summary(), indent=2))
        else:
            print(f"✅ Stopped experiment: {experiment.id}")
            print(f"   Status: {experiment.status.value}")
            if experiment.winner_variant_id:
                winner = experiment.get_variant_by_id(experiment.winner_variant_id)
                if winner:
                    print(f"   Winner: {winner.name} ({winner.id})")

    except Exception as e:
        print(f"❌ Error stopping experiment: {e}")
        raise


async def ab_list(
    campaign_id: Optional[str] = None,
    status: Optional[str] = None,
    json_output: bool = False
) -> None:
    """List A/B testing experiments."""
    try:
        manager = await init_ab_testing()

        from ..experiments.ab_testing import ExperimentStatus
        status_filter = ExperimentStatus(status) if status else None

        experiments = await manager.list_experiments(
            campaign_id=campaign_id,
            status=status_filter
        )

        if json_output:
            experiment_summaries = [exp.get_experiment_summary() for exp in experiments]
            print(json.dumps({"experiments": experiment_summaries}, indent=2))
        else:
            if not experiments:
                print("No experiments found")
                return

            print(f"📊 Found {len(experiments)} experiments:")
            print()
            for exp in experiments:
                print(f"ID: {exp.id}")
                print(f"Name: {exp.name}")
                print(f"Campaign: {exp.campaign_id}")
                print(f"Status: {exp.status.value}")
                print(f"Variants: {len(exp.variants)}")
                print(f"Primary Metric: {exp.primary_metric}")
                print(f"Sample Size: {sum(v.sample_size for v in exp.variants)}/{exp.target_sample_size}")
                if exp.winner_variant_id:
                    winner = exp.get_variant_by_id(exp.winner_variant_id)
                    if winner:
                        print(f"Winner: {winner.name}")
                print()

    except Exception as e:
        print(f"❌ Error listing experiments: {e}")
        raise


async def ab_status(experiment_id: str, json_output: bool = False) -> None:
    """Get the status of an A/B testing experiment."""
    try:
        manager = await init_ab_testing()
        experiment = await manager.get_experiment(experiment_id)

        if not experiment:
            print(f"❌ Experiment not found: {experiment_id}")
            return

        if json_output:
            print(json.dumps(experiment.get_experiment_summary(), indent=2))
        else:
            print(f"📊 Experiment: {experiment.id}")
            print(f"Name: {experiment.name}")
            print(f"Description: {experiment.description}")
            print(f"Campaign: {experiment.campaign_id}")
            print(f"Status: {experiment.status.value}")
            print(f"Primary Metric: {experiment.primary_metric}")
            print(f"Target Sample Size: {experiment.target_sample_size}")
            print(f"Current Sample Size: {sum(v.sample_size for v in experiment.variants)}")
            print(f"Created: {experiment.created_at}")

            if experiment.started_at:
                print(f"Started: {experiment.started_at}")
            if experiment.completed_at:
                print(f"Completed: {experiment.completed_at}")

            print()
            print("Variants:")
            for variant in experiment.variants:
                print(f"  - {variant.name} ({variant.id})")
                print(f"    Type: {variant.variant_type.value}")
                print(f"    Traffic: {variant.traffic_percentage}%")
                print(f"    Sample Size: {variant.sample_size}")
                if variant.metrics:
                    for metric_name, stats in variant.metrics.items():
                        if stats["values"]:
                            mean = sum(stats["values"]) / len(stats["values"])
                            print(f"    {metric_name}: {mean:.3f} (n={len(stats['values'])})")
                print()

    except Exception as e:
        print(f"❌ Error getting experiment status: {e}")
        raise


async def ab_stats(experiment_id: str, json_output: bool = False) -> None:
    """Get statistics for an A/B testing experiment."""
    try:
        manager = await init_ab_testing()
        results = await manager.get_experiment_results(experiment_id)

        if not results:
            print(f"❌ Experiment not found: {experiment_id}")
            return

        if json_output:
            print(json.dumps(results, indent=2))
        else:
            print(f"📈 Experiment Statistics: {results['id']}")
            print(f"Name: {results['name']}")
            print(f"Status: {results['status']}")
            print(f"Primary Metric: {results['primary_metric']}")
            print(f"Sample Size: {results['total_sample_size']}/{results['target_sample_size']}")
            print()

            print("Variant Performance:")
            for variant in results["variants"]:
                print(f"  📊 {variant['name']} ({variant['type']})")
                print(f"     Traffic: {variant['traffic_percentage']}%")
                print(f"     Sample Size: {variant['sample_size']}")

                if variant["metrics"]:
                    for metric_name, stats in variant["metrics"].items():
                        print(f"     {metric_name}:")
                        print(f"       Mean: {stats['mean']:.3f}")
                        print(f"       Count: {stats['count']}")
                        if stats['std_dev'] > 0:
                            print(f"       Std Dev: {stats['std_dev']:.3f}")
                print()

            if "winner" in results:
                winner_variant = next((v for v in results["variants"] if v["id"] == results["winner"]), None)
                if winner_variant:
                    print(f"🏆 Winner: {winner_variant['name']}")

            if "statistical_significance" in results:
                print(f"📊 Statistical Significance: {results['statistical_significance']:.3f}")

    except Exception as e:
        print(f"❌ Error getting experiment statistics: {e}")
        raise


__all__ = [
    "ab_create",
    "ab_start",
    "ab_stop",
    "ab_list",
    "ab_status",
    "ab_stats"
]
