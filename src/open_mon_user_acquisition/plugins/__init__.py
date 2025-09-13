"""Plugin system for extensible backends and agents."""

import importlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
from abc import ABC, abstractmethod

from ..core.interfaces import AgentInterface, LLMBackendInterface, StorageBackendInterface

logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """Base interface for all plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """Plugin type (agent, llm_backend, storage_backend)."""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass


class PluginManager:
    """Manages plugin loading, registration, and lifecycle."""

    def __init__(self):
        self._plugins: Dict[str, PluginInterface] = {}
        self._agent_plugins: Dict[str, Type[AgentInterface]] = {}
        self._llm_backend_plugins: Dict[str, Type[LLMBackendInterface]] = {}
        self._storage_backend_plugins: Dict[str, Type[StorageBackendInterface]] = {}
        self._plugin_dirs: List[Path] = []

    def add_plugin_directory(self, directory: Path) -> None:
        """Add a directory to search for plugins."""
        if directory.exists() and directory.is_dir():
            self._plugin_dirs.append(directory)
            logger.info(f"Added plugin directory: {directory}")
        else:
            logger.warning(f"Plugin directory does not exist: {directory}")

    async def load_plugins(self) -> None:
        """Load all plugins from configured directories."""
        for plugin_dir in self._plugin_dirs:
            await self._load_plugins_from_directory(plugin_dir)

    async def _load_plugins_from_directory(self, plugin_dir: Path) -> None:
        """Load plugins from a specific directory."""
        try:
            # Look for Python files that might contain plugins
            for py_file in plugin_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                module_name = py_file.stem
                try:
                    # Import the module
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Look for plugin classes in the module
                        await self._register_plugins_from_module(module, module_name)

                except Exception as e:
                    logger.error(f"Failed to load plugin from {py_file}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to load plugins from directory {plugin_dir}: {e}")

    async def _register_plugins_from_module(self, module, module_name: str) -> None:
        """Register plugin classes found in a module."""
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue

            attr = getattr(module, attr_name)
            if not isinstance(attr, type):
                continue

            # Check if it's a plugin class
            if (issubclass(attr, PluginInterface) and attr != PluginInterface and
                hasattr(attr, 'name') and hasattr(attr, 'version') and hasattr(attr, 'type')):

                try:
                    # Check plugin type from class attributes
                    plugin_type = getattr(attr, 'type', None)
                    if isinstance(plugin_type, property):
                        # If type is a property, create a temporary instance to get it
                        temp_instance = attr()
                        plugin_type = temp_instance.type
                        version = temp_instance.version
                    else:
                        # Fallback for class-level attributes
                        version = getattr(attr, 'version', 'unknown')

                    # Register based on type
                    if plugin_type == "agent":
                        self._agent_plugins[attr_name] = attr
                        logger.info(f"Registered agent plugin: {attr_name} v{version}")

                    elif plugin_type == "llm_backend":
                        self._llm_backend_plugins[attr_name] = attr
                        logger.info(f"Registered LLM backend plugin: {attr_name} v{version}")

                    elif plugin_type == "storage_backend":
                        self._storage_backend_plugins[attr_name] = attr
                        logger.info(f"Registered storage backend plugin: {attr_name} v{version}")

                    else:
                        logger.warning(f"Unknown plugin type '{plugin_type}' for {attr_name}")

                except Exception as e:
                    logger.error(f"Failed to register plugin {attr_name}: {e}")

    def get_agent_plugins(self) -> Dict[str, Type[AgentInterface]]:
        """Get all registered agent plugins."""
        return self._agent_plugins.copy()

    def get_llm_backend_plugins(self) -> Dict[str, Type[LLMBackendInterface]]:
        """Get all registered LLM backend plugins."""
        return self._llm_backend_plugins.copy()

    def get_storage_backend_plugins(self) -> Dict[str, Type[StorageBackendInterface]]:
        """Get all registered storage backend plugins."""
        return self._storage_backend_plugins.copy()

    def create_agent_instance(self, plugin_name: str, **kwargs) -> Optional[AgentInterface]:
        """Create an instance of an agent plugin."""
        if plugin_name not in self._agent_plugins:
            logger.error(f"Agent plugin not found: {plugin_name}")
            return None

        try:
            agent_class = self._agent_plugins[plugin_name]
            return agent_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create agent instance {plugin_name}: {e}")
            return None

    def create_llm_backend_instance(self, plugin_name: str, **kwargs) -> Optional[LLMBackendInterface]:
        """Create an instance of an LLM backend plugin."""
        if plugin_name not in self._llm_backend_plugins:
            logger.error(f"LLM backend plugin not found: {plugin_name}")
            return None

        try:
            backend_class = self._llm_backend_plugins[plugin_name]
            return backend_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create LLM backend instance {plugin_name}: {e}")
            return None

    def create_storage_backend_instance(self, plugin_name: str, **kwargs) -> Optional[StorageBackendInterface]:
        """Create an instance of a storage backend plugin."""
        if plugin_name not in self._storage_backend_plugins:
            logger.error(f"Storage backend plugin not found: {plugin_name}")
            return None

        try:
            backend_class = self._storage_backend_plugins[plugin_name]
            return backend_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create storage backend instance {plugin_name}: {e}")
            return None

    async def initialize_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Initialize a specific plugin."""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            logger.error(f"Plugin not found for initialization: {plugin_name}")
            return False

        try:
            await plugin.initialize(config)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
            return False

    async def shutdown_plugin(self, plugin_name: str) -> bool:
        """Shutdown a specific plugin."""
        plugin = self._plugins.get(plugin_name)
        if not plugin:
            logger.error(f"Plugin not found for shutdown: {plugin_name}")
            return False

        try:
            await plugin.shutdown()
            return True
        except Exception as e:
            logger.error(f"Failed to shutdown plugin {plugin_name}: {e}")
            return False

    async def shutdown_all_plugins(self) -> None:
        """Shutdown all loaded plugins."""
        for plugin_name, plugin in self._plugins.items():
            try:
                await plugin.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown plugin {plugin_name}: {e}")

        self._plugins.clear()
        logger.info("All plugins shutdown")


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get or create the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def setup_plugin_system(plugin_dirs: Optional[List[str]] = None) -> PluginManager:
    """Set up the plugin system with default and custom directories."""
    manager = get_plugin_manager()

    # Add default plugin directories
    default_dirs = [
        Path.home() / ".omua" / "plugins",
        Path.cwd() / "plugins",
    ]

    for dir_path in default_dirs:
        manager.add_plugin_directory(dir_path)

    # Add custom directories
    if plugin_dirs:
        for dir_path in plugin_dirs:
            manager.add_plugin_directory(Path(dir_path))

    return manager
