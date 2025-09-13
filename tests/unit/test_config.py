"""Unit tests for configuration management."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from open_mon_user_acquisition.config import ConfigManager, Settings
from open_mon_user_acquisition.config.settings import LLMSettings, DatabaseSettings


class TestSettings:
    """Test Settings configuration model."""

    def test_default_settings_creation(self):
        """Test creating settings with defaults."""
        settings = Settings()
        assert settings.app_name == "OpenMonetization-UserAcquisition"
        assert settings.version == "0.1.0"
        assert settings.debug is False
        assert settings.llm.openai_model == "gpt-4"
        assert settings.database.echo is False

    def test_custom_llm_settings(self):
        """Test custom LLM settings."""
        llm_settings = LLMSettings(
            openai_api_key="test_key",
            openai_model="gpt-3.5-turbo",
            ollama_base_url="http://custom:8080",
            fallback_order=["ollama", "openai"]
        )

        assert llm_settings.openai_api_key == "test_key"
        assert llm_settings.openai_model == "gpt-3.5-turbo"
        assert llm_settings.ollama_base_url == "http://custom:8080"
        assert llm_settings.fallback_order == ["ollama", "openai"]

    def test_database_settings(self):
        """Test database settings."""
        db_settings = DatabaseSettings(
            url="sqlite:///test.db",
            echo=True
        )

        assert db_settings.url == "sqlite:///test.db"
        assert db_settings.echo is True

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env_key", "DEBUG": "true"})
    def test_settings_from_env(self):
        """Test loading settings from environment variables."""
        settings = Settings()
        assert settings.llm.openai_api_key == "env_key"
        assert settings.debug is True

    def test_get_data_directory_debug(self):
        """Test data directory in debug mode."""
        settings = Settings(debug=True)
        data_dir = settings.get_data_directory()
        assert "data" in str(data_dir)

    def test_get_data_directory_production(self):
        """Test data directory in production mode."""
        settings = Settings(debug=False)
        data_dir = settings.get_data_directory()
        assert ".open_mon_user_acquisition" in str(data_dir)

    def test_get_log_directory(self):
        """Test log directory path."""
        settings = Settings()
        log_dir = settings.get_log_directory()
        assert "logs" in str(log_dir)

    def test_get_database_path_with_custom(self):
        """Test database path with custom path."""
        settings = Settings(database=DatabaseSettings(path="/custom/path.db"))
        db_path = settings.get_database_path()
        assert str(db_path) == "/custom/path.db"

    def test_get_database_path_default(self):
        """Test database path with default."""
        settings = Settings()
        db_path = settings.get_database_path()
        assert "data.db" in str(db_path)

    def test_settings_to_from_file_yaml(self):
        """Test saving and loading settings from YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = Path(temp_dir) / "config.yaml"

            # Create settings and save to file
            original_settings = Settings(
                debug=True,
                llm=LLMSettings(openai_api_key="test_key")
            )
            original_settings.to_file(str(yaml_file))

            # Load settings from file
            loaded_settings = Settings.from_file(str(yaml_file))

            assert loaded_settings.debug is True
            assert loaded_settings.llm.openai_api_key == "test_key"

    def test_settings_to_from_file_json(self):
        """Test saving and loading settings from JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            json_file = Path(temp_dir) / "config.json"

            # Create settings and save to file
            original_settings = Settings(debug=True)
            original_settings.to_file(str(json_file))

            # Load settings from file
            loaded_settings = Settings.from_file(str(json_file))

            assert loaded_settings.debug is True

    def test_settings_invalid_file_format(self):
        """Test loading settings from unsupported file format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_file = Path(temp_dir) / "config.txt"
            txt_file.write_text("invalid format")

            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                Settings.from_file(str(txt_file))

    def test_settings_missing_file(self):
        """Test loading settings from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Settings.from_file("non_existent.yaml")


class TestConfigManager:
    """Test ConfigManager implementation."""

    def test_config_manager_creation(self):
        """Test creating a config manager."""
        config = ConfigManager()
        assert config is not None

    def test_config_manager_with_custom_settings(self):
        """Test config manager with custom settings."""
        custom_settings = Settings(debug=True)
        config = ConfigManager(custom_settings)
        assert config.settings.debug is True

    def test_get_existing_config(self):
        """Test getting existing configuration value."""
        settings = Settings(debug=True)
        config = ConfigManager(settings)

        assert config.get("debug") is True
        assert config.get("app_name") == "OpenMonetization-UserAcquisition"

    def test_get_nonexistent_config(self):
        """Test getting non-existent configuration value."""
        config = ConfigManager()
        assert config.get("nonexistent") is None
        assert config.get("nonexistent", "default") == "default"

    def test_get_nested_config(self):
        """Test getting nested configuration values."""
        config = ConfigManager()
        assert config.get("llm.openai_model") == "gpt-4"
        assert config.get("database.echo") is False

    def test_set_config_value(self):
        """Test setting configuration values."""
        config = ConfigManager()

        config.set("debug", True)
        assert config.get("debug") is True

        config.set("llm.openai_model", "gpt-3.5-turbo")
        assert config.get("llm.openai_model") == "gpt-3.5-turbo"

    def test_has_config_key(self):
        """Test checking if configuration key exists."""
        config = ConfigManager()

        assert config.has("debug") is True
        assert config.has("nonexistent") is False
        assert config.has("llm.openai_model") is True

    def test_get_all_config(self):
        """Test getting all configuration values."""
        config = ConfigManager()
        all_config = config.get_all()

        assert isinstance(all_config, dict)
        assert "app_name" in all_config
        assert "debug" in all_config
        assert "llm" in all_config

    def test_get_section(self):
        """Test getting configuration section."""
        config = ConfigManager()
        llm_section = config.get_section("llm")

        assert isinstance(llm_section, dict)
        assert "openai_model" in llm_section

    def test_set_section(self):
        """Test setting configuration section."""
        config = ConfigManager()

        new_llm_config = {
            "openai_model": "gpt-3.5-turbo",
            "openai_api_key": "new_key"
        }

        config.set_section("llm", new_llm_config)

        assert config.get("llm.openai_model") == "gpt-3.5-turbo"
        assert config.get("llm.openai_api_key") == "new_key"

    def test_update_from_dict(self):
        """Test updating configuration from dictionary."""
        config = ConfigManager()

        updates = {
            "debug": True,
            "llm": {
                "openai_model": "gpt-3.5-turbo"
            }
        }

        config.update_from_dict(updates)

        assert config.get("debug") is True
        assert config.get("llm.openai_model") == "gpt-3.5-turbo"

    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = ConfigManager()
        errors = config.validate_config()

        # Should have no errors for basic valid config
        assert isinstance(errors, list)

    def test_ensure_directories_exist(self):
        """Test ensuring directories exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a custom settings that points to temp directory
            settings = Settings()
            settings._data_directory = Path(temp_dir)
            config = ConfigManager(settings)

            config.ensure_directories_exist()

            assert (Path(temp_dir) / "logs").exists()

    def test_config_directory_paths(self):
        """Test configuration directory path methods."""
        config = ConfigManager()

        data_dir = config.get_data_directory()
        log_dir = config.get_log_directory()
        db_path = config.get_database_path()

        assert isinstance(data_dir, Path)
        assert isinstance(log_dir, Path)
        assert isinstance(db_path, Path)

        assert "logs" in str(log_dir)
        assert "data.db" in str(db_path)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env_key"})
    def test_load_from_env(self):
        """Test loading configuration from environment."""
        config = ConfigManager()

        # This should reload from environment
        config.load_from_env()

        # The ConfigManager doesn't have a direct load_from_env method
        # but the Settings object handles environment variables automatically
        assert config.settings.llm.openai_api_key == "env_key"
