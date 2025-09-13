"""Configuration manager implementation."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.interfaces import ConfigInterface
from .settings import Settings


class ConfigManager(ConfigInterface):
    """Configuration manager with file and environment variable support."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the configuration manager.

        Args:
            settings: Settings instance. If None, loads default settings.
        """
        self.settings = settings or Settings()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (dot-separated for nested access).
            default: Default value if key is not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self.settings.dict()

        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key: Configuration key (dot-separated for nested access).
            value: Value to set.
        """
        keys = key.split(".")
        config_dict = self.settings.dict()

        # Navigate to the parent of the target key
        target = config_dict
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        # Set the value
        target[keys[-1]] = value

        # Update the settings object
        self.settings = Settings(**config_dict)

    def has(self, key: str) -> bool:
        """Check if a configuration key exists.

        Args:
            key: Configuration key (dot-separated for nested access).

        Returns:
            True if key exists, False otherwise.
        """
        return self.get(key, "__NOT_FOUND__") != "__NOT_FOUND__"

    def load_from_file(self, file_path: str) -> None:
        """Load configuration from a file.

        Args:
            file_path: Path to the configuration file.
        """
        self.settings = Settings.from_file(file_path)

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a file.

        Args:
            file_path: Path to save the configuration file.
        """
        self.settings.to_file(file_path)

    def load_from_env(self, prefix: str = "") -> None:
        """Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix to filter by.
        """
        # Environment variables are already handled by Pydantic
        # This method exists for interface compatibility
        self.settings = Settings()

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values as a flat dictionary.

        Returns:
            Dictionary of all configuration values.
        """
        return self.settings.dict()

    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary.

        Args:
            config_dict: Dictionary of configuration values.
        """
        current_config = self.settings.dict()
        current_config.update(config_dict)
        self.settings = Settings(**current_config)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a configuration section as a dictionary.

        Args:
            section: Section name.

        Returns:
            Dictionary of section configuration values.
        """
        config_dict = self.settings.dict()
        return config_dict.get(section, {})

    def set_section(self, section: str, values: Dict[str, Any]) -> None:
        """Set a configuration section from a dictionary.

        Args:
            section: Section name.
            values: Dictionary of section values.
        """
        config_dict = self.settings.dict()
        config_dict[section] = values
        self.settings = Settings(**config_dict)

    def create_default_config(self, file_path: str) -> None:
        """Create a default configuration file.

        Args:
            file_path: Path where to create the configuration file.
        """
        default_settings = Settings()
        default_settings.to_file(file_path)

    def get_data_directory(self) -> Path:
        """Get the application data directory.

        Returns:
            Path to the application data directory.
        """
        return self.settings.get_data_directory()

    def get_log_directory(self) -> Path:
        """Get the application log directory.

        Returns:
            Path to the application log directory.
        """
        return self.settings.get_log_directory()

    def get_database_path(self) -> Path:
        """Get the database file path.

        Returns:
            Path to the database file.
        """
        return self.settings.get_database_path()

    def ensure_directories_exist(self) -> None:
        """Ensure that all required directories exist."""
        self.get_data_directory().mkdir(parents=True, exist_ok=True)
        self.get_log_directory().mkdir(parents=True, exist_ok=True)

    def validate_config(self) -> List[str]:
        """Validate the current configuration.

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []

        # Check required settings
        if not self.settings.llm.openai_api_key and not any([
            self.settings.llm.ollama_base_url
        ]):
            errors.append("No LLM backend configured. Set OPENAI_API_KEY or ensure Ollama is running.")

        # Check database path is writable
        if self.settings.database.path:
            db_path = Path(self.settings.database.path)
            try:
                db_path.parent.mkdir(parents=True, exist_ok=True)
                # Try to create a temporary file to test write permissions
                test_file = db_path.parent / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                errors.append(f"Database path is not writable: {e}")

        # Check log directory is writable
        log_dir = self.get_log_directory()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            test_file = log_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            errors.append(f"Log directory is not writable: {e}")

        return errors
