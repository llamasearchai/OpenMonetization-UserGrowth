"""Metrics collection and reporting for the application."""

import asyncio
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

from ..config import ConfigManager
from ..core.interfaces import StorageBackendInterface
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric measurement."""
    name: str
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        storage: Optional[StorageBackendInterface] = None,
        collection_interval: int = 60,
        max_points_per_metric: int = 1000,
    ):
        """Initialize the metrics collector.

        Args:
            config: Configuration manager instance.
            storage: Storage backend for persistence.
            collection_interval: Interval in seconds for collecting metrics.
            max_points_per_metric: Maximum number of points to keep per metric.
        """
        self.config = config
        self.storage = storage
        self.collection_interval = collection_interval
        self.max_points_per_metric = max_points_per_metric

        # In-memory storage for metrics
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))

        # Counters for various metrics
        self._counters: Dict[str, int] = defaultdict(int)

        # Gauges for current values
        self._gauges: Dict[str, Union[int, float]] = {}

        # Histograms for distributions
        self._histograms: Dict[str, List[Union[int, float]]] = defaultdict(list)

        # Background collection task
        self._collection_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Thread safety
        self._lock = threading.Lock()

    async def start_collection(self) -> None:
        """Start the background metrics collection."""
        if self._collection_task is None:
            self._collection_task = asyncio.create_task(self._collect_metrics_loop())
            logger.info("Started metrics collection")

    async def stop_collection(self) -> None:
        """Stop the background metrics collection."""
        if self._collection_task:
            self._shutdown_event.set()
            try:
                await asyncio.wait_for(self._collection_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._collection_task.cancel()
            self._collection_task = None
            logger.info("Stopped metrics collection")

    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name.
            value: Value to increment by.
            tags: Optional tags for the metric.
        """
        with self._lock:
            self._counters[name] += value

        self._record_point(name, value, tags or {}, {"type": "counter"})

    def set_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric.

        Args:
            name: Metric name.
            value: Value to set.
            tags: Optional tags for the metric.
        """
        with self._lock:
            self._gauges[name] = value

        self._record_point(name, value, tags or {}, {"type": "gauge"})

    def record_histogram(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram metric.

        Args:
            name: Metric name.
            value: Value to record.
            tags: Optional tags for the metric.
        """
        with self._lock:
            self._histograms[name].append(value)

        self._record_point(name, value, tags or {}, {"type": "histogram"})

    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric.

        Args:
            name: Metric name.
            duration: Duration in seconds.
            tags: Optional tags for the metric.
        """
        self.record_histogram(f"{name}.duration", duration, tags)

    def _record_point(
        self,
        name: str,
        value: Union[int, float],
        tags: Dict[str, str],
        metadata: Dict[str, Any]
    ) -> None:
        """Record a metric point.

        Args:
            name: Metric name.
            value: Metric value.
            tags: Metric tags.
            metadata: Additional metadata.
        """
        point = MetricPoint(
            name=name,
            value=value,
            tags=tags,
            metadata=metadata
        )

        with self._lock:
            self._metrics[name].append(point)

        # Persist to storage if available
        if self.storage:
            asyncio.create_task(self._persist_metric_point(point))

    async def _persist_metric_point(self, point: MetricPoint) -> None:
        """Persist a metric point to storage.

        Args:
            point: Metric point to persist.
        """
        try:
            await self.storage.save_metric(
                metric_name=point.name,
                value=point.value,
                metadata={
                    "timestamp": point.timestamp,
                    "tags": point.tags,
                    **point.metadata
                }
            )
        except Exception as e:
            logger.error(f"Failed to persist metric {point.name}: {e}")

    def get_metric_values(self, name: str, limit: Optional[int] = None) -> List[MetricPoint]:
        """Get values for a specific metric.

        Args:
            name: Metric name.
            limit: Maximum number of points to return.

        Returns:
            List of metric points.
        """
        with self._lock:
            points = list(self._metrics[name])

        if limit:
            points = points[-limit:]

        return points

    def get_counter_value(self, name: str) -> int:
        """Get the current value of a counter.

        Args:
            name: Counter name.

        Returns:
            Current counter value.
        """
        with self._lock:
            return self._counters[name]

    def get_gauge_value(self, name: str) -> Optional[Union[int, float]]:
        """Get the current value of a gauge.

        Args:
            name: Gauge name.

        Returns:
            Current gauge value, or None if not set.
        """
        with self._lock:
            return self._gauges.get(name)

    def get_histogram_stats(self, name: str) -> Dict[str, Union[int, float]]:
        """Get statistics for a histogram metric.

        Args:
            name: Histogram name.

        Returns:
            Dictionary with histogram statistics.
        """
        with self._lock:
            values = self._histograms[name]

        if not values:
            return {"count": 0}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
        }

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Dictionary with metrics summary.
        """
        summary = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {},
            "time_series_metrics": {},
        }

        with self._lock:
            # Histogram summaries
            for name, values in self._histograms.items():
                if values:
                    summary["histograms"][name] = {
                        "count": len(values),
                        "latest": values[-1] if values else None,
                    }

            # Time series metrics
            for name, points in self._metrics.items():
                if points:
                    latest = points[-1]
                    summary["time_series_metrics"][name] = {
                        "count": len(points),
                        "latest_value": latest.value,
                        "latest_timestamp": latest.timestamp.isoformat(),
                    }

        return summary

    async def _collect_metrics_loop(self) -> None:
        """Background loop for collecting system metrics."""
        while not self._shutdown_event.is_set():
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        try:
            import psutil
        except ImportError:
            # psutil not available, skip system metrics
            return

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.set_gauge("system.cpu_percent", cpu_percent)

        # Memory usage
        memory = psutil.virtual_memory()
        self.set_gauge("system.memory_percent", memory.percent)
        self.set_gauge("system.memory_used_mb", memory.used / 1024 / 1024)

        # Disk usage
        disk = psutil.disk_usage('/')
        self.set_gauge("system.disk_percent", disk.percent)
        self.set_gauge("system.disk_used_gb", disk.used / 1024 / 1024 / 1024)

        # Process info
        process = psutil.Process()
        self.set_gauge("process.cpu_percent", process.cpu_percent())
        self.set_gauge("process.memory_mb", process.memory_info().rss / 1024 / 1024)
        self.set_gauge("process.threads", process.num_threads())

    def clear_metrics(self, name: Optional[str] = None) -> None:
        """Clear metrics data.

        Args:
            name: Specific metric name to clear, or None to clear all.
        """
        with self._lock:
            if name:
                if name in self._metrics:
                    self._metrics[name].clear()
                if name in self._counters:
                    self._counters[name] = 0
                if name in self._gauges:
                    del self._gauges[name]
                if name in self._histograms:
                    self._histograms[name].clear()
            else:
                self._metrics.clear()
                self._counters.clear()
                self._gauges.clear()
                self._histograms.clear()
