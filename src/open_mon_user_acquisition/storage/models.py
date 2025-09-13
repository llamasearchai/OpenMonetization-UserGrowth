"""SQLAlchemy models for the storage layer."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import json

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey,
    JSON,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class WorkflowModel(Base):
    """SQLAlchemy model for workflow instances."""

    __tablename__ = "workflows"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)

    # Relationships
    tasks = relationship("TaskModel", back_populates="workflow", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata_,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowModel":
        """Create model from dictionary."""
        # Handle datetime strings
        for field in ["created_at", "started_at", "completed_at"]:
            if data.get(field):
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field])
                # If it's already a datetime object, keep it as is

        # Rename metadata field
        if "metadata" in data:
            data["metadata_"] = data.pop("metadata")

        # Remove tasks and results from data as they're handled separately
        tasks_data = data.pop("tasks", [])
        data.pop("results", {})  # Remove results as it's not stored in the model

        workflow = cls(**data)
        workflow.tasks = [TaskModel.from_dict(task_data) for task_data in tasks_data]

        return workflow


class TaskModel(Base):
    """SQLAlchemy model for tasks."""

    __tablename__ = "tasks"

    id = Column(String, primary_key=True)
    workflow_id = Column(String, ForeignKey("workflows.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    agent_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    parameters = Column(JSON, default=dict)
    dependencies = Column(JSON, default=list)  # List of task IDs
    timeout_seconds = Column(Integer, nullable=True)
    retry_count = Column(Integer, default=0)
    priority = Column(Integer, default=1)
    metadata_ = Column("metadata", JSON, default=dict)

    # Relationships
    workflow = relationship("WorkflowModel", back_populates="tasks")

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "priority": self.priority,
            "metadata": self.metadata_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskModel":
        """Create model from dictionary."""
        # Handle datetime strings
        for field in ["started_at", "completed_at"]:
            if data.get(field):
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field])
                # If it's already a datetime object, keep it as is

        # Rename metadata field
        if "metadata" in data:
            data["metadata_"] = data.pop("metadata")

        return cls(**data)


class MetricModel(Base):
    """SQLAlchemy model for metrics."""

    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    channel = Column(String, nullable=True)
    campaign = Column(String, nullable=True)
    metadata_ = Column("metadata", JSON, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "channel": self.channel,
            "campaign": self.campaign,
            "metadata": self.metadata_,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricModel":
        """Create model from dictionary."""
        # Handle datetime string
        if data.get("timestamp"):
            if isinstance(data["timestamp"], str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            # If it's already a datetime object, keep it as is

        # Rename metadata field
        if "metadata" in data:
            data["metadata_"] = data.pop("metadata")

        return cls(**data)


def create_engine_and_session(database_url: str):
    """Create SQLAlchemy engine and session factory.

    Args:
        database_url: Database connection URL.

    Returns:
        Tuple of (engine, session_factory).
    """
    # Special handling for SQLite to allow async operations
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
    else:
        engine = create_engine(database_url, echo=False)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return engine, SessionLocal
