"""Unit tests for LLM backend implementations."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import HTTPStatusError, RequestError

from open_mon_user_acquisition.llm.openai_backend import OpenAIBackend
from open_mon_user_acquisition.llm.ollama_backend import OllamaBackend
from open_mon_user_acquisition.llm.fallback_manager import LLMFallbackManager
from open_mon_user_acquisition.core.types import LLMMessage, LLMResponse


class TestOpenAIBackend:
    """Test OpenAI backend implementation."""

    def test_backend_initialization(self):
        """Test OpenAI backend initialization."""
        backend = OpenAIBackend(api_key="test_key")
        assert backend.name == "openai"
        assert backend.is_available is True
        assert backend.api_key == "test_key"

    def test_backend_initialization_no_key(self):
        """Test OpenAI backend without API key."""
        with patch.dict("os.environ", {}, clear=True):
            backend = OpenAIBackend()
            assert backend.is_available is False
            assert backend.api_key is None

    def test_backend_initialization_with_env_key(self):
        """Test OpenAI backend with environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env_key"}):
            backend = OpenAIBackend()
            assert backend.is_available is True
            assert backend.api_key == "env_key"

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful text generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.id = "test_response_id"
        mock_response.created = 1234567890
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        backend = OpenAIBackend(api_key="test_key")
        backend._client = mock_client

        response = await backend.generate("Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "Generated text"
        assert response.finish_reason == "stop"
        assert response.model == "gpt-4"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        assert response.usage["total_tokens"] == 30
        assert response.metadata["backend"] == "openai"

    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        """Test generation with API error."""
        import openai
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=openai.APIError("API Error"))

        backend = OpenAIBackend(api_key="test_key")
        backend._client = mock_client

        with pytest.raises(RuntimeError, match="OpenAI API error"):
            await backend.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_unavailable_backend(self):
        """Test generation with unavailable backend."""
        backend = OpenAIBackend()  # No API key

        with pytest.raises(RuntimeError, match="OpenAI backend is not configured"):
            await backend.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Test successful chat conversation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Chat response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        backend = OpenAIBackend(api_key="test_key")
        backend._client = mock_client

        messages = [
            LLMMessage(role="user", content="Hello"),
            LLMMessage(role="assistant", content="Hi there")
        ]

        response = await backend.chat(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Chat response"
        assert response.usage["total_tokens"] == 40

    @pytest.mark.asyncio
    async def test_validate_connection_success(self):
        """Test successful connection validation."""
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=MagicMock())

        backend = OpenAIBackend(api_key="test_key")
        backend._client = mock_client

        result = await backend.validate_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_connection_failure(self):
        """Test connection validation failure."""
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(side_effect=Exception("Connection failed"))

        backend = OpenAIBackend(api_key="test_key")
        backend._client = mock_client

        result = await backend.validate_connection()
        assert result is False


class TestOllamaBackend:
    """Test Ollama backend implementation."""

    def test_backend_initialization(self):
        """Test Ollama backend initialization."""
        backend = OllamaBackend()
        assert backend.name == "ollama"
        assert backend.is_available is True  # Ollama is considered available by default
        assert backend.base_url == "http://localhost:11434"

    def test_backend_initialization_custom_url(self):
        """Test Ollama backend with custom URL."""
        backend = OllamaBackend(base_url="http://custom:8080")
        assert backend.base_url == "http://custom:8080"

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful text generation with Ollama."""
        mock_response_data = {
            "response": "Generated text from Ollama",
            "done": True,
            "model": "llama2",
            "eval_count": 50,
            "eval_duration": 1000000,
            "total_duration": 2000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 500000,
            "done_reason": "stop"
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        backend = OllamaBackend()
        backend._client = mock_client

        response = await backend.generate("Test prompt")

        assert isinstance(response, LLMResponse)
        assert response.content == "Generated text from Ollama"
        assert response.model == "llama2"
        assert response.finish_reason == "stop"
        assert response.metadata["backend"] == "ollama"

    @pytest.mark.asyncio
    async def test_generate_model_not_found(self):
        """Test generation with model not found error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(side_effect=HTTPStatusError(
            "404 Not Found",
            request=MagicMock(),
            response=MagicMock(status_code=404)
        ))
        mock_client.post = AsyncMock(return_value=mock_response)

        backend = OllamaBackend()
        backend._client = mock_client

        with pytest.raises(RuntimeError, match="Model .* not found in Ollama"):
            await backend.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_connection_error(self):
        """Test generation with connection error."""
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=RequestError("Connection failed"))

        backend = OllamaBackend()
        backend._client = mock_client

        with pytest.raises(RuntimeError, match="Failed to connect to Ollama server"):
            await backend.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """Test successful chat with Ollama."""
        mock_response_data = {
            "message": {"content": "Chat response from Ollama"},
            "done": True,
            "model": "llama2",
            "eval_count": 30,
            "eval_duration": 800000,
            "total_duration": 1500000,
            "prompt_eval_count": 8,
            "prompt_eval_duration": 300000,
            "done_reason": "stop"
        }

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value=mock_response_data)
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        backend = OllamaBackend()
        backend._client = mock_client

        messages = [LLMMessage(role="user", content="Hello")]

        response = await backend.chat(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Chat response from Ollama"
        assert response.metadata["backend"] == "ollama"

    @pytest.mark.asyncio
    async def test_validate_connection_success(self):
        """Test successful connection validation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        backend = OllamaBackend()
        backend._client = mock_client

        result = await backend.validate_connection()
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_connection_failure(self):
        """Test connection validation failure."""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection failed"))

        backend = OllamaBackend()
        backend._client = mock_client

        result = await backend.validate_connection()
        assert result is False


class TestLLMFallbackManager:
    """Test LLM fallback manager."""

    def test_manager_initialization_default(self):
        """Test fallback manager with default backends."""
        manager = LLMFallbackManager()
        assert manager.name == "fallback_manager"
        assert len(manager.backends) == 2
        assert "openai" in manager.backend_map
        assert "ollama" in manager.backend_map
        assert manager.fallback_order == ["openai", "ollama"]

    def test_manager_initialization_custom(self):
        """Test fallback manager with custom backends."""
        mock_backend1 = MagicMock()
        mock_backend1.name = "backend1"
        mock_backend2 = MagicMock()
        mock_backend2.name = "backend2"

        manager = LLMFallbackManager([mock_backend1, mock_backend2])
        assert len(manager.backends) == 2
        assert manager.backend_map["backend1"] == mock_backend1
        assert manager.backend_map["backend2"] == mock_backend2

    def test_manager_initialization_invalid_fallback_order(self):
        """Test fallback manager with invalid fallback order."""
        mock_backend = MagicMock()
        mock_backend.name = "backend1"

        with pytest.raises(ValueError, match="Backend .* not found in backends"):
            LLMFallbackManager([mock_backend], ["nonexistent"])

    @pytest.mark.asyncio
    async def test_generate_with_fallback(self):
        """Test generation with fallback to second backend."""
        # First backend fails
        failing_backend = MagicMock()
        failing_backend.name = "failing"
        failing_backend.is_available = True
        failing_backend.validate_connection = AsyncMock(return_value=True)
        failing_backend.generate = AsyncMock(side_effect=Exception("First backend failed"))

        # Second backend succeeds
        success_backend = MagicMock()
        success_backend.name = "success"
        success_backend.is_available = True
        success_backend.validate_connection = AsyncMock(return_value=True)
        success_backend.generate = AsyncMock(return_value=MagicMock(
            content="Success response",
            metadata={}
        ))

        manager = LLMFallbackManager([failing_backend, success_backend])

        response = await manager.generate("Test prompt")

        assert response.content == "Success response"
        assert response.metadata["used_backend"] == "success"
        assert response.metadata["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_generate_all_backends_fail(self):
        """Test generation when all backends fail."""
        failing_backend1 = MagicMock()
        failing_backend1.name = "fail1"
        failing_backend1.is_available = True
        failing_backend1.validate_connection = AsyncMock(return_value=True)
        failing_backend1.generate = AsyncMock(side_effect=Exception("Backend 1 failed"))

        failing_backend2 = MagicMock()
        failing_backend2.name = "fail2"
        failing_backend2.is_available = True
        failing_backend2.validate_connection = AsyncMock(return_value=True)
        failing_backend2.generate = AsyncMock(side_effect=Exception("Backend 2 failed"))

        manager = LLMFallbackManager([failing_backend1, failing_backend2])

        with pytest.raises(RuntimeError, match="All LLM backends failed"):
            await manager.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_generate_no_available_backends(self):
        """Test generation with no available backends."""
        unavailable_backend = MagicMock()
        unavailable_backend.name = "unavailable"
        unavailable_backend.is_available = False

        manager = LLMFallbackManager([unavailable_backend])

        with pytest.raises(RuntimeError, match="No LLM backends are available"):
            await manager.generate("Test prompt")

    @pytest.mark.asyncio
    async def test_validate_connection(self):
        """Test connection validation."""
        available_backend = MagicMock()
        available_backend.name = "available"
        available_backend.is_available = True
        available_backend.validate_connection = AsyncMock(return_value=True)

        manager = LLMFallbackManager([available_backend])

        result = await manager.validate_connection()
        assert result is True

    def test_get_backend_status(self):
        """Test getting backend status."""
        backend1 = MagicMock()
        backend1.name = "backend1"
        backend1.is_available = True

        backend2 = MagicMock()
        backend2.name = "backend2"
        backend2.is_available = False

        manager = LLMFallbackManager([backend1, backend2])

        status = manager.get_backend_status()

        assert status["backend1"]["is_available"] is True
        assert status["backend2"]["is_available"] is False

    def test_add_backend(self):
        """Test adding a backend."""
        manager = LLMFallbackManager([])
        new_backend = MagicMock()
        new_backend.name = "new_backend"

        manager.add_backend(new_backend)

        assert "new_backend" in manager.backend_map
        assert new_backend in manager.backends

    def test_add_duplicate_backend(self):
        """Test adding a duplicate backend."""
        existing_backend = MagicMock()
        existing_backend.name = "existing"
        manager = LLMFallbackManager([existing_backend])

        duplicate_backend = MagicMock()
        duplicate_backend.name = "existing"

        with pytest.raises(ValueError, match="Backend .* already exists"):
            manager.add_backend(duplicate_backend)

    def test_remove_backend(self):
        """Test removing a backend."""
        backend = MagicMock()
        backend.name = "to_remove"
        manager = LLMFallbackManager([backend])

        manager.remove_backend("to_remove")

        assert "to_remove" not in manager.backend_map
        assert backend not in manager.backends

    def test_remove_nonexistent_backend(self):
        """Test removing a nonexistent backend."""
        manager = LLMFallbackManager([])

        with pytest.raises(ValueError, match="Backend .* not found"):
            manager.remove_backend("nonexistent")
