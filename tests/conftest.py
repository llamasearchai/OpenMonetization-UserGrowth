"""Shared test configuration and fixtures."""

import asyncio
import pytest
import pytest_asyncio

from open_mon_user_acquisition.config import ConfigManager, Settings
from open_mon_user_acquisition.storage.sqlite_backend import SQLiteStorageBackend
from open_mon_user_acquisition.llm.fallback_manager import LLMFallbackManager
from open_mon_user_acquisition.orchestrator import WorkflowOrchestrator


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Create mock configuration manager."""
    config = ConfigManager()
    return config


@pytest.fixture
def mock_storage():
    """Create mock storage backend."""
    storage = SQLiteStorageBackend()
    return storage


@pytest.fixture
def mock_llm_backend():
    """Create mock LLM backend."""
    llm = LLMFallbackManager()
    return llm


@pytest_asyncio.fixture
async def test_orchestrator(mock_config, mock_storage, mock_llm_backend):
    """Create test orchestrator instance."""
    orch = WorkflowOrchestrator(
        config=mock_config,
        storage=mock_storage,
        llm_backend=mock_llm_backend
    )
    await orch.initialize()
    try:
        yield orch
    finally:
        await orch.shutdown()


@pytest_asyncio.fixture
async def test_storage_backend():
    """Create storage backend for testing."""
    backend = SQLiteStorageBackend()
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.close()


@pytest.fixture(autouse=True)
async def cleanup_database(test_storage_backend):
    """Clean up database between tests."""
    # This runs before each test
    from open_mon_user_acquisition.storage.models import WorkflowModel, TaskModel
    from sqlalchemy import delete

    async with test_storage_backend._async_session() as session:
        # Delete all data from tables
        await session.execute(delete(WorkflowModel))
        await session.execute(delete(TaskModel))
        await session.commit()