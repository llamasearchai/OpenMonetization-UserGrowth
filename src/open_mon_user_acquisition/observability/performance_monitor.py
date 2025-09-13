"""Performance monitoring and alerting for the application."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime, timedelta

from .logging_config import get_logger
from .metrics_collector import MetricsCollector

logger = get_logger(__name__)


@dataclass
class PerformanceAlert:
    """Represents a performance alert."""
    alert_type: str
    message: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class PerformanceMonitor:
    """Monitors application performance and generates alerts."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize the performance monitor.

        Args:
            metrics_collector: Metrics collector instance.
            alert_thresholds: Dictionary of alert thresholds by metric name.
        """
        self.metrics = metrics_collector
        self.alert_thresholds = alert_thresholds or self._get_default_thresholds()

        self._alert_handlers: List[Callable[[PerformanceAlert], Awaitable[None]]] = []
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Alert state
        self._active_alerts: Dict[str, PerformanceAlert] = {}
        self._alert_cooldowns: Dict[str, datetime] = {}

    def _get_default_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Get default alert thresholds."""
        return {
            "system.cpu_percent": {
                "warning": 80,
                "critical": 95,
                "operator": "gt",
            },
            "system.memory_percent": {
                "warning": 85,
                "critical": 95,
                "operator": "gt",
            },
            "system.disk_percent": {
                "warning": 90,
                "critical": 98,
                "operator": "gt",
            },
            "workflow.duration": {
                "warning": 300,  # 5 minutes
                "critical": 1800,  # 30 minutes
                "operator": "gt",
            },
            "llm.response_time": {
                "warning": 30,  # 30 seconds
                "critical": 120,  # 2 minutes
                "operator": "gt",
            },
        }

    def add_alert_handler(self, handler: Callable[[PerformanceAlert], Awaitable[None]]) -> None:
        """Add an alert handler function.

        Args:
            handler: Async function that takes a PerformanceAlert.
        """
        self._alert_handlers.append(handler)

    async def start_monitoring(self) -> None:
        """Start the performance monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            logger.info("Started performance monitoring")

    async def stop_monitoring(self) -> None:
        """Stop the performance monitoring."""
        if self._monitoring_task:
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._monitoring_task.cancel()
            self._monitoring_task = None
            logger.info("Stopped performance monitoring")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_thresholds()
                await self._cleanup_expired_alerts()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _check_thresholds(self) -> None:
        """Check metric thresholds and generate alerts."""
        for metric_name, thresholds in self.alert_thresholds.items():
            await self._check_metric_threshold(metric_name, thresholds)

    async def _check_metric_threshold(self, metric_name: str, thresholds: Dict[str, Any]) -> None:
        """Check a specific metric against its thresholds.

        Args:
            metric_name: Name of the metric to check.
            thresholds: Threshold configuration.
        """
        # Get current metric value
        if metric_name.startswith("system.") or metric_name.startswith("process."):
            # System/process metrics are gauges
            value = self.metrics.get_gauge_value(metric_name)
        else:
            # Other metrics, get latest value from time series
            points = self.metrics.get_metric_values(metric_name, limit=1)
            value = points[-1].value if points else None

        if value is None:
            return

        operator = thresholds.get("operator", "gt")
        warning_threshold = thresholds.get("warning")
        critical_threshold = thresholds.get("critical")

        alert_type = None
        severity = None

        if operator == "gt":
            if critical_threshold and value > critical_threshold:
                alert_type = "critical"
                severity = "critical"
            elif warning_threshold and value > warning_threshold:
                alert_type = "warning"
                severity = "medium"
        elif operator == "lt":
            if critical_threshold and value < critical_threshold:
                alert_type = "critical"
                severity = "critical"
            elif warning_threshold and value < warning_threshold:
                alert_type = "warning"
                severity = "medium"

        if alert_type:
            alert_key = f"{metric_name}_{alert_type}"
            await self._generate_alert(
                alert_key,
                f"{metric_name} is {alert_type}: {value} {operator} {thresholds.get(alert_type + '_threshold' if alert_type == 'warning' else 'critical', warning_threshold or critical_threshold)}",
                severity,
                {
                    "metric_name": metric_name,
                    "value": value,
                    "threshold": thresholds.get(alert_type + "_threshold" if alert_type == "warning" else "critical", warning_threshold or critical_threshold),
                    "operator": operator,
                }
            )

    async def _generate_alert(
        self,
        alert_key: str,
        message: str,
        severity: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Generate and handle an alert.

        Args:
            alert_key: Unique key for the alert.
            message: Alert message.
            severity: Alert severity.
            metadata: Alert metadata.
        """
        # Check cooldown
        now = datetime.now()
        last_alert = self._alert_cooldowns.get(alert_key)
        if last_alert and (now - last_alert) < timedelta(minutes=5):
            return  # Still in cooldown

        alert = PerformanceAlert(
            alert_type=alert_key,
            message=message,
            severity=severity,
            metadata=metadata
        )

        self._active_alerts[alert_key] = alert
        self._alert_cooldowns[alert_key] = now

        logger.warning(f"Performance alert: {message}", extra=metadata)

        # Notify handlers
        for handler in self._alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    async def _cleanup_expired_alerts(self) -> None:
        """Clean up expired alerts."""
        now = datetime.now()
        expired_keys = []

        for alert_key, alert in self._active_alerts.items():
            # Auto-resolve alerts after 1 hour
            if (now - alert.timestamp) > timedelta(hours=1):
                expired_keys.append(alert_key)

        for key in expired_keys:
            del self._active_alerts[key]

    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get list of currently active alerts.

        Returns:
            List of active performance alerts.
        """
        return list(self._active_alerts.values())

    def resolve_alert(self, alert_key: str) -> bool:
        """Manually resolve an alert.

        Args:
            alert_key: Key of the alert to resolve.

        Returns:
            True if alert was resolved, False if not found.
        """
        if alert_key in self._active_alerts:
            del self._active_alerts[alert_key]
            logger.info(f"Manually resolved alert: {alert_key}")
            return True
        return False

    @asynccontextmanager
    async def measure_performance(self, operation_name: str, **tags):
        """Context manager to measure performance of an operation.

        Args:
            operation_name: Name of the operation being measured.
            **tags: Additional tags for the metrics.
        """
        start_time = time.time()
        start_memory = None

        # Try to get memory usage
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            pass

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time

            # Record timing
            self.metrics.record_timing(operation_name, duration, tags)

            # Record memory usage if available
            if start_memory is not None:
                try:
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = end_memory - start_memory
                    self.metrics.record_histogram(
                        f"{operation_name}.memory_delta_mb",
                        memory_delta,
                        tags
                    )
                except Exception:
                    pass

            logger.debug(
                f"Performance measurement for {operation_name}",
                duration=f"{duration:.4f}s",
                **tags
            )

    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report.

        Returns:
            Dictionary containing performance metrics and alerts.
        """
        metrics_summary = self.metrics.get_all_metrics_summary()

        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": metrics_summary,
            "active_alerts": [
                {
                    "type": alert.alert_type,
                    "message": alert.message,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata,
                }
                for alert in self.get_active_alerts()
            ],
            "system_health": self._calculate_system_health(),
        }

        return report

    def _calculate_system_health(self) -> str:
        """Calculate overall system health based on alerts.

        Returns:
            Health status string.
        """
        alerts = self.get_active_alerts()

        critical_alerts = [a for a in alerts if a.severity == "critical"]
        warning_alerts = [a for a in alerts if a.severity == "medium"]

        if critical_alerts:
            return "critical"
        elif warning_alerts:
            return "warning"
        else:
            return "healthy"
