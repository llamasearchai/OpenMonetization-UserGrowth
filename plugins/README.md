# OMUA Plugin System

The OpenMonetization-UserAcquisition (OMUA) plugin system allows you to extend the platform with custom agents, LLM backends, and storage backends without modifying the core codebase.

## Plugin Types

### Agent Plugins
Custom agents that can plan and execute user acquisition tasks. Agents must implement both `AgentInterface` and `PluginInterface`.

### LLM Backend Plugins
Custom LLM providers (OpenAI, Anthropic, local models, etc.). Must implement `LLMBackendInterface` and `PluginInterface`.

### Storage Backend Plugins
Custom storage solutions (PostgreSQL, MongoDB, etc.). Must implement `StorageBackendInterface` and `PluginInterface`.

## Creating a Plugin

### 1. Agent Plugin Example

```python
from open_mon_user_acquisition.core.interfaces import AgentInterface
from open_mon_user_acquisition.core.types import TaskSpec, ContextData, TaskResult, TaskStatus
from open_mon_user_acquisition.plugins import PluginInterface

class MyCustomAgent(AgentInterface, PluginInterface):
    """My custom user acquisition agent."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self._name = "my_custom_agent"
        self._description = "Custom agent for specialized acquisition tasks"

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def type(self) -> str:
        return "agent"

    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin."""
        if "api_key" in config:
            self.api_key = config["api_key"]

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass

    async def plan(self, context: ContextData) -> List[TaskSpec]:
        """Plan tasks based on context."""
        # Your custom planning logic here
        return [
            TaskSpec(
                id=f"{self.name}_task_1",
                name="Custom Task",
                agent_type=self.name,
                description="Execute custom logic",
                parameters={"custom_param": "value"}
            )
        ]

    async def execute(self, task: TaskSpec, context: ContextData) -> TaskResult:
        """Execute a task."""
        # Your custom execution logic here
        return TaskResult(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            result={"custom_result": "success"},
            metadata={"execution_time": 1.0}
        )
```

### 2. LLM Backend Plugin Example

```python
from open_mon_user_acquisition.core.interfaces import LLMBackendInterface
from open_mon_user_acquisition.core.types import LLMResponse, LLMMessage
from open_mon_user_acquisition.plugins import PluginInterface

class MyCustomLLM(LLMBackendInterface, PluginInterface):
    """Custom LLM backend implementation."""

    @property
    def name(self) -> str:
        return "my_custom_llm"

    @property
    def is_available(self) -> bool:
        return True

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def type(self) -> str:
        return "llm_backend"

    async def initialize(self, config: Dict[str, Any]) -> None:
        self.api_key = config.get("api_key")

    async def shutdown(self) -> None:
        pass

    async def generate(self, prompt: str, options: Optional[Dict[str, Any]] = None) -> LLMResponse:
        # Your LLM generation logic here
        return LLMResponse(content="Generated response", usage={"tokens": 100})

    async def chat(self, messages: List[LLMMessage], options: Optional[Dict[str, Any]] = None) -> LLMResponse:
        # Your chat logic here
        return LLMResponse(content="Chat response", usage={"tokens": 50})

    async def validate_connection(self) -> bool:
        return True
```

## Plugin Installation

### 1. Plugin Directories
Plugins are automatically loaded from:
- `~/.omua/plugins/` (user plugins)
- `./plugins/` (project plugins)
- Custom directories specified in configuration

### 2. File Structure
```
plugins/
├── my_agent.py          # Plugin file
├── my_llm_backend.py    # Another plugin
└── README.md           # Documentation
```

### 3. Loading Plugins
Plugins are automatically loaded when the OMUA system starts. You can also manually load them:

```python
from open_mon_user_acquisition.plugins import setup_plugin_system

# Setup with custom directories
manager = setup_plugin_system(['/path/to/custom/plugins'])

# Load all plugins
await manager.load_plugins()

# Get available plugins
agent_plugins = manager.get_agent_plugins()
llm_plugins = manager.get_llm_backend_plugins()

# Create instances
agent = manager.create_agent_instance('MyCustomAgent', api_key='...')
llm = manager.create_llm_backend_instance('MyCustomLLM', api_key='...')
```

## Plugin Configuration

Plugins can be configured via the main OMUA configuration file:

```yaml
plugins:
  my_custom_agent:
    api_key: "your-api-key-here"
    custom_setting: "value"

  my_custom_llm:
    model: "gpt-4"
    temperature: 0.7
```

## Best Practices

1. **Error Handling**: Always handle errors gracefully in your plugins
2. **Async/Await**: Use async methods for I/O operations
3. **Type Hints**: Provide comprehensive type annotations
4. **Documentation**: Document your plugin's purpose and usage
5. **Versioning**: Use semantic versioning for your plugins
6. **Testing**: Test your plugins thoroughly before deployment

## Example Plugin: Email Marketing Agent

See `example_agent.py` for a complete working example of a custom agent plugin that demonstrates:
- Custom planning logic
- Task execution with different types
- Configuration handling
- Proper error handling
- Result formatting

## Troubleshooting

### Plugin Not Loading
- Check that your plugin class inherits from the correct interfaces
- Ensure the plugin file is in a plugin directory
- Check the logs for loading errors

### Plugin Not Working
- Verify the `type` property returns the correct plugin type
- Check that all required methods are implemented
- Ensure proper error handling in your plugin

### Import Errors
- Make sure all imports are available in the plugin environment
- Check that your plugin doesn't have circular dependencies
