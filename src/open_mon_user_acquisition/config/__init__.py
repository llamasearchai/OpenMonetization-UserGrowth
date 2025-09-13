"""Configuration management for OpenMonetization-UserAcquisition.

This module provides centralized configuration management with support for
YAML/JSON files and environment variables.
"""

from .config_manager import ConfigManager
from .settings import Settings

__all__ = [
    "ConfigManager",
    "Settings",
]
