"""Pydantic settings model for application configuration."""

from pathlib import Path
from typing import List, Optional

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """Settings for LLM backends."""

    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(None, env="OPENAI_BASE_URL")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    openai_timeout: int = Field(30, env="OPENAI_TIMEOUT")

    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama2", env="OLLAMA_MODEL")
    ollama_timeout: int = Field(60, env="OLLAMA_TIMEOUT")

    fallback_order: List[str] = Field(["openai", "ollama"], env="LLM_FALLBACK_ORDER")


class DatabaseSettings(BaseSettings):
    """Settings for database storage."""

    url: Optional[str] = Field(None, env="DATABASE_URL")
    path: Optional[str] = Field(None, env="DATABASE_PATH")
    echo: bool = Field(False, env="DATABASE_ECHO")


class LoggingSettings(BaseSettings):
    """Settings for logging configuration."""

    level: str = Field("INFO", env="LOG_LEVEL")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    file_path: Optional[str] = Field(None, env="LOG_FILE_PATH")
    max_file_size: int = Field(10 * 1024 * 1024, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")


class MetricsSettings(BaseSettings):
    """Settings for metrics collection."""

    enabled: bool = Field(True, env="METRICS_ENABLED")
    collection_interval: int = Field(60, env="METRICS_COLLECTION_INTERVAL")  # seconds
    retention_days: int = Field(30, env="METRICS_RETENTION_DAYS")


class WorkflowSettings(BaseSettings):
    """Settings for workflow execution."""

    max_concurrent_workflows: int = Field(10, env="MAX_CONCURRENT_WORKFLOWS")
    default_timeout: int = Field(3600, env="DEFAULT_WORKFLOW_TIMEOUT")  # 1 hour
    retry_attempts: int = Field(3, env="WORKFLOW_RETRY_ATTEMPTS")
    cleanup_interval: int = Field(86400, env="WORKFLOW_CLEANUP_INTERVAL")  # 24 hours


class SecuritySettings(BaseSettings):
    """Settings for security and safety."""

    allow_arbitrary_code: bool = Field(False, env="ALLOW_ARBITRARY_CODE")
    max_prompt_length: int = Field(10000, env="MAX_PROMPT_LENGTH")
    rate_limit_requests: int = Field(100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, env="RATE_LIMIT_WINDOW")  # seconds


class Settings(BaseSettings):
    """Main application settings."""

    # Application info
    app_name: str = "OpenMonetization-UserAcquisition"
    version: str = "0.1.0"
    debug: bool = Field(False, env="DEBUG")

    # Configuration file paths
    config_file: Optional[str] = Field(None, env="CONFIG_FILE")

    # Component settings
    llm: LLMSettings = LLMSettings()
    database: DatabaseSettings = DatabaseSettings()
    logging: LoggingSettings = LoggingSettings()
    metrics: MetricsSettings = MetricsSettings()
    workflow: WorkflowSettings = WorkflowSettings()
    security: SecuritySettings = SecuritySettings()

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @classmethod
    def from_file(cls, file_path: str) -> "Settings":
        """Load settings from a YAML or JSON file.

        Args:
            file_path: Path to the configuration file.

        Returns:
            Settings instance loaded from file.
        """
        import yaml
        import json

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                config_data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")

        return cls(**config_data)

    def to_file(self, file_path: str) -> None:
        """Save settings to a YAML or JSON file.

        Args:
            file_path: Path where to save the configuration file.
        """
        import yaml
        import json

        path = Path(file_path)
        config_data = self.dict()

        with open(path, "w", encoding="utf-8") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            elif path.suffix.lower() == ".json":
                json.dump(config_data, f, indent=2, sort_keys=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    def get_data_directory(self) -> Path:
        """Get the application data directory.

        Returns:
            Path to the application data directory.
        """
        if self.debug:
            return Path.cwd() / "data"
        else:
            return Path.home() / ".open_mon_user_acquisition"

    def get_log_directory(self) -> Path:
        """Get the application log directory.

        Returns:
            Path to the application log directory.
        """
        return self.get_data_directory() / "logs"

    def get_database_path(self) -> Path:
        """Get the database file path.

        Returns:
            Path to the database file.
        """
        if self.database.path:
            return Path(self.database.path)
        else:
            return self.get_data_directory() / "data.db"
