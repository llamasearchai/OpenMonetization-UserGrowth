import pytest
from unittest.mock import MagicMock, patch
from src.open_mon_user_acquisition.observability.logging_config import setup_logging, get_logger, LoggerMixin
from src.open_mon_user_acquisition.observability.metrics_collector import MetricsCollector, MetricPoint
from src.open_mon_user_acquisition.observability.performance_monitor import PerformanceMonitor, PerformanceAlert
from src.open_mon_user_acquisition.config import ConfigManager
from src.open_mon_user_acquisition.storage import SQLiteStorageBackend
from src.open_mon_user_acquisition.core.types import WorkflowInstance, WorkflowStatus, TaskSpec

class TestLoggingConfig:
    @pytest.fixture
    def temp_log_file(self):
        import tempfile
        return tempfile.mktemp(suffix=".log")
    
    def test_setup_logging(self, temp_log_file):
        config = MagicMock()
        config.get.return_value = False  # Not JSON
        setup_logging(config, level="DEBUG", log_file=temp_log_file)
        
        logger = get_logger("test")
        logger.info("Test log message", param="value")
        
        assert logger is not None
        
        # Verify structlog is configured
        assert "param" in logger._logger._context

def test_metrics_collector():
    config = MagicMock()
    storage = MagicMock()
    collector = MetricsCollector(config, storage)
    
    # Test counter
    collector.increment_counter("test.counter")
    assert collector.get_counter_value("test.counter") == 1
    
    # Test gauge
    collector.set_gauge("test.gauge", 50.5)
    assert collector.get_gauge_value("test.gauge") == 50.5
    
    # Test histogram
    collector.record_histogram("test.histogram", 10.0)
    stats = collector.get_histogram_stats("test.histogram")
    assert stats["count"] == 1
    assert stats["min"] == 10.0
    
    # Test timing
    collector.record_timing("test.timing", 0.5)
    stats = collector.get_histogram_stats("test.timing.duration")
    assert stats["count"] == 1
    
    # Test persistence mock
    storage.save_metric.assert_called()

@pytest.mark.asyncio
async def test_performance_monitor():
    config = MagicMock()
    metrics = MagicMock()
    
    monitor = PerformanceMonitor(metrics)
    
    # Add handler
    async def mock_handler(alert):
        pass
    monitor.add_alert_handler(mock_handler)
    
    # Mock metrics for threshold
    metrics.get_gauge_value.return_value = 100  # Critical threshold
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Check thresholds
    await monitor._check_thresholds()
    
    # Verify alert generated
    alerts = monitor.get_active_alerts()
    assert len(alerts) > 0
    
    # Stop monitoring
    await monitor.stop_monitoring()
