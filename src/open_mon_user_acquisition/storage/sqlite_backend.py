"""SQLite storage backend implementation using SQLAlchemy."""

import asyncio
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path

from sqlalchemy import and_, delete, desc, func
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, selectinload
from sqlalchemy.future import select

from ..core.interfaces import StorageBackendInterface
from ..core.types import WorkflowInstance, MetricData
from .models import (
    WorkflowModel,
    TaskModel,
    MetricModel,
    Base,
    create_engine_and_session,
)


class SQLiteStorageBackend(StorageBackendInterface):
    """SQLite storage backend implementation with SQLAlchemy ORM and optional encryption."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        database_path: Optional[str] = None,
        echo: bool = False,
        encryption_enabled: bool = False,
        encryption_key: Optional[str] = None,
    ):
        """Initialize the SQLite storage backend.

        Args:
            database_url: Full database URL. If provided, overrides database_path.
            database_path: Path to SQLite database file. Defaults to user data directory.
            echo: Whether to echo SQL statements for debugging.
            encryption_enabled: Enable database encryption (requires pysqlcipher3).
            encryption_key: Encryption passphrase for the database.
        """
        if database_url:
            self.database_url = database_url
        elif database_path:
            self.database_path = Path(database_path)
            self.database_url = f"sqlite:///{self.database_path.absolute()}"
        else:
            # Default to user data directory
            data_dir = Path.home() / ".open_mon_user_acquisition"
            data_dir.mkdir(exist_ok=True)
            self.database_path = data_dir / "data.db"
            self.database_url = f"sqlite:///{self.database_path.absolute()}"

        self.echo = echo
        self.encryption_enabled = encryption_enabled
        self.encryption_key = encryption_key
        self._engine = None
        self._async_session = None
        self._initialized = False

        # Validate encryption configuration
        if encryption_enabled and not encryption_key:
            raise ValueError("Encryption key is required when encryption is enabled")

    async def initialize(self) -> None:
        """Initialize the storage backend and create tables."""
        if self._initialized:
            return

        # Handle encrypted database URL
        db_url = self.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
        connect_args = {"check_same_thread": False}

        if self.encryption_enabled:
            try:
                # Use pysqlcipher3 for encrypted SQLite databases
                import pysqlcipher3
                # Modify URL for SQLCipher (note: aiosqlite doesn't support SQLCipher directly)
                # This would require a custom approach or different async SQLite driver
                db_url = db_url.replace("sqlite+aiosqlite:///", "sqlite+pysqlcipher3:///")
                connect_args["passphrase"] = self.encryption_key
                connect_args["cipher_compatibility"] = 3  # Use SQLCipher 3 compatibility
            except ImportError:
                raise RuntimeError(
                    "pysqlcipher3 is required for database encryption. "
                    "Install with: pip install pysqlcipher3"
                )

        # Create async engine for SQLite
        self._engine = create_async_engine(
            db_url,
            echo=self.echo,
            connect_args=connect_args,
        )

        # Create tables
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Create async session factory
        self._async_session = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

        self._initialized = True

    async def close(self) -> None:
        """Close the storage backend connection."""
        if self._engine:
            await self._engine.dispose()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the backend is initialized."""
        if not self._initialized:
            await self.initialize()

    async def save_workflow(self, workflow: WorkflowInstance) -> None:
        """Save a workflow instance.

        Args:
            workflow: The workflow instance to save.
        """
        await self._ensure_initialized()

        async with self._async_session() as session:
            # Convert to model
            workflow_data = workflow.model_dump()

            # Extract tasks before creating workflow model (from_dict pops the tasks key)
            tasks_data = workflow_data.pop("tasks", [])

            workflow_model = WorkflowModel.from_dict(workflow_data)

            # Check if workflow exists
            existing = await session.get(WorkflowModel, workflow.id)
            if existing:
                # Update existing workflow
                for key, value in workflow_data.items():
                    if key == "tasks":
                        continue  # Handle tasks separately
                    elif key == "metadata":
                        setattr(existing, "metadata_", value)
                    elif hasattr(existing, key):
                        setattr(existing, key, value)

                # Use merge to handle task updates/inserts
                for task_data in tasks_data:
                    task_model = TaskModel.from_dict(task_data)
                    task_model.workflow_id = workflow.id
                    # Use merge to handle both insert and update cases
                    await session.merge(task_model)
            else:
                # Create new workflow
                session.add(workflow_model)

                # Add tasks for new workflow
                for task_data in tasks_data:
                    task_model = TaskModel.from_dict(task_data)
                    task_model.workflow_id = workflow.id
                    await session.merge(task_model)

            await session.commit()

    async def load_workflow(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Load a workflow instance by ID.

        Args:
            workflow_id: The ID of the workflow to load.

        Returns:
            The workflow instance if found, None otherwise.
        """
        await self._ensure_initialized()

        async with self._async_session() as session:
            result = await session.execute(
                select(WorkflowModel).options(selectinload(WorkflowModel.tasks)).where(WorkflowModel.id == workflow_id)
            )
            workflow_model = result.scalar_one_or_none()

            if workflow_model:
                # Ensure tasks are loaded
                await session.refresh(workflow_model, ["tasks"])
                workflow_data = workflow_model.to_dict()
                return WorkflowInstance(**workflow_data)

            return None

    async def list_workflows(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[WorkflowInstance]:
        """List workflow instances with optional filtering.

        Args:
            status: Filter by workflow status.
            limit: Maximum number of workflows to return.
            offset: Number of workflows to skip.

        Returns:
            List of workflow instances.
        """
        await self._ensure_initialized()

        async with self._async_session() as session:
            query = select(WorkflowModel).options(selectinload(WorkflowModel.tasks))

            if status:
                query = query.where(WorkflowModel.status == status)

            query = query.order_by(desc(WorkflowModel.created_at)).limit(limit).offset(offset)

            result = await session.execute(query)
            workflow_models = result.scalars().all()

            workflows = []
            for model in workflow_models:
                # Tasks should already be loaded by selectinload
                workflow_data = model.to_dict()
                workflows.append(WorkflowInstance(**workflow_data))

            return workflows

    async def save_metric(self, metric_name: str, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a metric measurement.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            metadata: Additional metadata for the metric.
        """
        await self._ensure_initialized()

        # Extract specific fields from metadata
        channel = metadata.get("channel")
        campaign = metadata.get("campaign")
        timestamp = metadata.get("timestamp", datetime.now())

        # Remove extracted fields from metadata
        clean_metadata = {k: v for k, v in metadata.items()
                         if k not in ["channel", "campaign", "timestamp"]}

        metric_model = MetricModel(
            name=metric_name,
            value=float(value),
            timestamp=timestamp,
            channel=channel,
            campaign=campaign,
            metadata_=clean_metadata,
        )

        async with self._async_session() as session:
            session.add(metric_model)
            await session.commit()

    async def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve metrics with optional filtering.

        Args:
            metric_name: Filter by metric name.
            start_time: Filter metrics after this time.
            end_time: Filter metrics before this time.
            limit: Maximum number of metrics to return.

        Returns:
            List of metric dictionaries.
        """
        await self._ensure_initialized()

        async with self._async_session() as session:
            query = select(MetricModel)

            if metric_name:
                query = query.where(MetricModel.name == metric_name)

            if start_time:
                query = query.where(MetricModel.timestamp >= start_time)

            if end_time:
                query = query.where(MetricModel.timestamp <= end_time)

            query = query.order_by(desc(MetricModel.timestamp)).limit(limit)

            result = await session.execute(query)
            metric_models = result.scalars().all()

            metrics = []
            for model in metric_models:
                metric_data = model.to_dict()
                metrics.append(metric_data)

            return metrics

    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics.

        Returns:
            Dictionary containing workflow statistics.
        """
        await self._ensure_initialized()

        async with self._async_session() as session:
            # Count workflows by status
            status_counts = await session.execute(
                select(WorkflowModel.status, func.count(WorkflowModel.id))
                .group_by(WorkflowModel.status)
            )

            stats = {"status_counts": dict(status_counts.all())}

            # Get total workflow count
            total_count = await session.execute(
                select(func.count(WorkflowModel.id))
            )
            stats["total_workflows"] = total_count.scalar()

            # Get recent workflow count (last 24 hours)
            yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            recent_count = await session.execute(
                select(func.count(WorkflowModel.id))
                .where(WorkflowModel.created_at >= yesterday)
            )
            stats["recent_workflows"] = recent_count.scalar()

            return stats

    async def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old workflow and metric data.

        Args:
            days_to_keep: Number of days of data to keep.

        Returns:
            Number of records deleted.
        """
        await self._ensure_initialized()

        cutoff_date = datetime.now()
        cutoff_date = cutoff_date.replace(
            day=cutoff_date.day - days_to_keep,
            hour=0, minute=0, second=0, microsecond=0
        )

        deleted_count = 0

        async with self._async_session() as session:
            # Delete old workflows (and their tasks via cascade)
            workflow_result = await session.execute(
                select(WorkflowModel).where(WorkflowModel.created_at < cutoff_date)
            )
            old_workflows = workflow_result.scalars().all()

            for workflow in old_workflows:
                await session.delete(workflow)
                deleted_count += 1

            # Delete old metrics
            metric_result = await session.execute(
                select(MetricModel).where(MetricModel.timestamp < cutoff_date)
            )
            old_metrics = metric_result.scalars().all()

            for metric in old_metrics:
                await session.delete(metric)
                deleted_count += 1

            await session.commit()

        return deleted_count
