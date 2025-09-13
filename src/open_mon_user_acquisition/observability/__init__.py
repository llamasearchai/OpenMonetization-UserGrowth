"""Observability layer for OpenMonetization-UserAcquisition.

This module provides structured logging, metrics collection, and monitoring
capabilities for the OMUA system.
"""

from .logging_config import setup_logging, get_logger
from .metrics_collector import MetricsCollector
from .performance_monitor import PerformanceMonitor

__all__ = [
    "setup_logging",
    "get_logger",
    "MetricsCollector",
    "PerformanceMonitor",
]
