"""Ollama LLM backend implementation."""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx

from ..core.interfaces import LLMBackendInterface
from ..core.types import LLMResponse, LLMMessage


class OllamaBackend(LLMBackendInterface):
    """Ollama LLM backend implementation for local LLM inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize the Ollama backend.

        Args:
            base_url: Base URL for the Ollama server.
            model: Default model to use for generation.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "ollama"

    @property
    def is_available(self) -> bool:
        """Check if the backend is available and configured."""
        # For Ollama, we consider it available if we can create a client
        # The actual connection validation happens in validate_connection
        return True

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate text from a prompt using Ollama.

        Args:
            prompt: The text prompt to generate from.
            options: Additional options for generation.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            Exception: If the generation fails after retries.
        """
        client = await self._get_client()
        options = options or {}

        model = options.get("model", self.default_model)
        temperature = options.get("temperature", 0.7)
        top_p = options.get("top_p", 0.9)
        top_k = options.get("top_k", 40)
        num_predict = options.get("max_tokens", 1000)
        repeat_penalty = options.get("repeat_penalty", 1.1)
        stream = options.get("stream", False)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty,
            }
        }

        url = f"{self.base_url}/api/generate"

        for attempt in range(self.max_retries + 1):
            try:
                start_time = datetime.now()

                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()

                data = response.json()
                content = data.get("response", "")

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                usage = {
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                    "total_duration": data.get("total_duration", 0),
                    "duration_seconds": duration,
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                }

                return LLMResponse(
                    content=content,
                    usage=usage,
                    finish_reason=data.get("done_reason"),
                    model=data.get("model"),
                    metadata={
                        "backend": self.name,
                        "done": data.get("done", False),
                        "context": data.get("context", []),
                    }
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise RuntimeError(f"Model '{model}' not found in Ollama. Please pull the model first.") from e
                elif attempt == self.max_retries:
                    raise RuntimeError(f"Ollama API error: {e}") from e
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed to connect to Ollama server at {self.base_url}: {e}") from e
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed to generate with Ollama: {e}") from e

            # Wait before retrying
            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError("Failed to generate after all retries")

    async def chat(
        self,
        messages: List[LLMMessage],
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Conduct a chat conversation using Ollama.

        Args:
            messages: List of messages in the conversation.
            options: Additional options for the chat.

        Returns:
            LLMResponse containing the response and metadata.

        Raises:
            Exception: If the chat fails after retries.
        """
        client = await self._get_client()
        options = options or {}

        model = options.get("model", self.default_model)
        temperature = options.get("temperature", 0.7)
        top_p = options.get("top_p", 0.9)
        top_k = options.get("top_k", 40)
        num_predict = options.get("max_tokens", 1000)
        repeat_penalty = options.get("repeat_penalty", 1.1)
        stream = options.get("stream", False)

        # Convert LLMMessage objects to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty,
            }
        }

        url = f"{self.base_url}/api/chat"

        for attempt in range(self.max_retries + 1):
            try:
                start_time = datetime.now()

                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()

                data = response.json()
                content = data.get("message", {}).get("content", "")

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                usage = {
                    "eval_count": data.get("eval_count", 0),
                    "eval_duration": data.get("eval_duration", 0),
                    "total_duration": data.get("total_duration", 0),
                    "duration_seconds": duration,
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                }

                return LLMResponse(
                    content=content,
                    usage=usage,
                    finish_reason=data.get("done_reason"),
                    model=data.get("model"),
                    metadata={
                        "backend": self.name,
                        "done": data.get("done", False),
                    }
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise RuntimeError(f"Model '{model}' not found in Ollama. Please pull the model first.") from e
                elif attempt == self.max_retries:
                    raise RuntimeError(f"Ollama API error: {e}") from e
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed to connect to Ollama server at {self.base_url}: {e}") from e
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"Failed to chat with Ollama: {e}") from e

            # Wait before retrying
            if attempt < self.max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError("Failed to chat after all retries")

    async def validate_connection(self) -> bool:
        """Validate that the backend connection is working.

        Returns:
            True if the connection is valid, False otherwise.
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return True
        except Exception:
            return False
