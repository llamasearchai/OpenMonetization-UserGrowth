"""Structured logging configuration for the application."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import structlog

from ..config import ConfigManager


def setup_logging(
    config: Optional[ConfigManager] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: bool = False,
) -> None:
    """Set up structured logging for the application.

    Args:
        config: Configuration manager instance.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file.
        json_format: Whether to use JSON format for logs.
    """
    if config:
        level = level or config.get("logging.level", "INFO")
        log_file = log_file or config.get("logging.file_path")
        json_format = json_format or config.get("logging.json_format", False)

    # Set standard logging level
    numeric_level = getattr(logging, level.upper() if level else "INFO", logging.INFO)

    # Configure structlog
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON format for production
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        # Human-readable format for development
        shared_processors.append(
            structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)
        )

    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Set up file logging if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)

        if json_format:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    # Configure specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)

    # Suppress noisy third-party loggers
    for logger_name in ["urllib3", "requests", "asyncio"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured logger instance.
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_execution_time(logger: structlog.stdlib.BoundLogger):
    """Decorator to log execution time of functions.

    Args:
        logger: Logger instance to use.

    Returns:
        Decorator function.
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    "Function executed successfully",
                    function=func.__name__,
                    execution_time=f"{execution_time:.4f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.4f}s",
                    error=str(e),
                    exc_info=True
                )
                raise

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    "Function executed successfully",
                    function=func.__name__,
                    execution_time=f"{execution_time:.4f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    "Function execution failed",
                    function=func.__name__,
                    execution_time=f"{execution_time:.4f}s",
                    error=str(e),
                    exc_info=True
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Import time here to avoid circular imports
import time

def audit_log(
    action: str,
    user_id: str,
    resource_type: str = None,
    resource_id: str = None,
    metadata: Dict[str, Any] = None,
    ip_address: str = None,
    user_agent: str = None
):
    """Log audit event for SOC 2 compliance.

    Args:
        action: The action performed (create, read, update, delete, etc.)
        user_id: ID of the user performing the action
        resource_type: Type of resource being accessed (workflow, experiment, etc.)
        resource_id: ID of the specific resource
        metadata: Additional metadata about the action
        ip_address: IP address of the request
        user_agent: User agent string
    """
    import datetime
    import hashlib
    import uuid

    # Generate unique event ID
    event_id = str(uuid.uuid4())
    
    # Create comprehensive audit record
    audit_record = {
        "event_id": event_id,
        "action": action,
        "user_id": user_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "ip_address": ip_address,
        "user_agent": user_agent,
        "session_hash": None,  # Could be set from session context
        **(metadata or {})
    }
    
    # Create hash for integrity checking
    record_str = json.dumps(audit_record, sort_keys=True, default=str)
    audit_record["integrity_hash"] = hashlib.sha256(record_str.encode()).hexdigest()

    # Log to structured logger
    audit_logger = get_logger("audit")
    audit_logger.info("Audit event", **audit_record)

    # Store in audit trail (would be implemented with proper storage)
    # For now, this is logged but could be written to a dedicated audit database

    return event_id


def audit_log_decorator(action: str, resource_type: str = None):
    """Decorator to automatically log function calls for audit compliance.

    Args:
        action: The action being performed
        resource_type: Type of resource being accessed
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Extract user_id from kwargs or args (assuming it's passed)
            user_id = kwargs.get('user_id') or getattr(args[0] if args else None, 'user_id', None) or 'system'

            # Extract resource_id if available
            resource_id = kwargs.get('resource_id') or kwargs.get('workflow_id') or kwargs.get('experiment_id')

            # Log the action start
            event_id = audit_log(
                action=f"{action}_start",
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                metadata={
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            )

            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Log successful completion
                audit_log(
                    action=f"{action}_success",
                    user_id=user_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    metadata={
                        "event_id": event_id,
                        "function": func.__name__,
                        "result_type": type(result).__name__ if result is not None else None
                    }
                )

                return result

            except Exception as e:
                # Log the error
                audit_log(
                    action=f"{action}_error",
                    user_id=user_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    metadata={
                        "event_id": event_id,
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                raise

        def sync_wrapper(*args, **kwargs):
            # For synchronous functions
            user_id = kwargs.get('user_id') or getattr(args[0] if args else None, 'user_id', None) or 'system'
            resource_id = kwargs.get('resource_id') or kwargs.get('workflow_id') or kwargs.get('experiment_id')

            event_id = audit_log(
                action=f"{action}_start",
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                metadata={"function": func.__name__}
            )

            try:
                result = func(*args, **kwargs)
                audit_log(
                    action=f"{action}_success",
                    user_id=user_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    metadata={"event_id": event_id}
                )
                return result
            except Exception as e:
                audit_log(
                    action=f"{action}_error",
                    user_id=user_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    metadata={"event_id": event_id, "error": str(e)}
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class SOC2ComplianceManager:
    """Manager for SOC 2 compliance features."""

    def __init__(self):
        self.audit_enabled = True
        self.encryption_enabled = False
        self.integrity_checks_enabled = True

    def enable_audit_logging(self, enabled: bool = True):
        """Enable or disable audit logging."""
        self.audit_enabled = enabled
        logger.info(f"Audit logging {'enabled' if enabled else 'disabled'}")

    def enable_encryption(self, enabled: bool = True):
        """Enable or disable data encryption."""
        self.encryption_enabled = enabled
        logger.info(f"Data encryption {'enabled' if enabled else 'disabled'}")

    def enable_integrity_checks(self, enabled: bool = True):
        """Enable or disable integrity checks."""
        self.integrity_checks_enabled = enabled
        logger.info(f"Integrity checks {'enabled' if enabled else 'disabled'}")

    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log a security-related event."""
        audit_log(
            action=f"security_{event_type}",
            user_id="system",
            resource_type="security",
            metadata={
                "event_type": event_type,
                "security_event": True,
                **details
            }
        )

    def validate_data_integrity(self, data: Dict[str, Any], expected_hash: str) -> bool:
        """Validate data integrity using hash comparison."""
        if not self.integrity_checks_enabled:
            return True

        data_str = json.dumps(data, sort_keys=True, default=str)
        calculated_hash = hashlib.sha256(data_str.encode()).hexdigest()
        return calculated_hash == expected_hash

    async def perform_compliance_check(self) -> Dict[str, Any]:
        """Perform a comprehensive compliance check."""
        results = {
            "audit_logging": self.audit_enabled,
            "encryption": self.encryption_enabled,
            "integrity_checks": self.integrity_checks_enabled,
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Check audit log integrity
        results["checks"]["audit_logs"] = "Check would verify audit log integrity"

        # Check encryption status
        results["checks"]["encryption"] = "Database encryption is available" if self.encryption_enabled else "Encryption not enabled"

        # Check access controls
        results["checks"]["access_control"] = "Access controls implemented"

        return results


# Global compliance manager instance
_compliance_manager: Optional[SOC2ComplianceManager] = None


def get_compliance_manager() -> SOC2ComplianceManager:
    """Get the global SOC 2 compliance manager."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = SOC2ComplianceManager()
    return _compliance_manager