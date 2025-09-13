"""OpenAI LLM backend implementation."""

import os
from typing import Any, Dict, List, Optional
from datetime import datetime
import openai
from openai import AsyncOpenAI

from ..core.interfaces import LLMBackendInterface
from ..core.types import LLMResponse, LLMMessage


class OpenAIBackend(LLMBackendInterface):
    """OpenAI LLM backend implementation using the official OpenAI SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the OpenAI backend.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable.
            base_url: Custom API base URL for compatibility with other providers.
            model: Default model to use for generation.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.default_model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[AsyncOpenAI] = None

    @property
    def name(self) -> str:
        """Return the backend name."""
        return "openai"

    @property
    def is_available(self) -> bool:
        """Check if the backend is available and configured."""
        return self.api_key is not None

    async def _get_client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: The text prompt to generate from.
            options: Additional options for generation.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            Exception: If the generation fails after retries.
        """
        if not self.is_available:
            raise RuntimeError("OpenAI backend is not configured. Set OPENAI_API_KEY environment variable.")

        client = await self._get_client()
        options = options or {}

        model = options.get("model", self.default_model)
        temperature = options.get("temperature", 0.7)
        max_tokens = options.get("max_tokens", 1000)
        top_p = options.get("top_p", 1.0)
        frequency_penalty = options.get("frequency_penalty", 0.0)
        presence_penalty = options.get("presence_penalty", 0.0)

        try:
            start_time = datetime.now()

            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            choice = response.choices[0]
            content = choice.message.content or ""

            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "duration_seconds": duration,
            }

            return LLMResponse(
                content=content,
                usage=usage,
                finish_reason=choice.finish_reason,
                model=response.model,
                metadata={
                    "backend": self.name,
                    "request_id": getattr(response, "id", None),
                    "created": getattr(response, "created", None),
                }
            )

        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate with OpenAI: {e}") from e

    async def chat(
        self,
        messages: List[LLMMessage],
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Conduct a chat conversation.

        Args:
            messages: List of messages in the conversation.
            options: Additional options for the chat.

        Returns:
            LLMResponse containing the response and metadata.

        Raises:
            Exception: If the chat fails after retries.
        """
        if not self.is_available:
            raise RuntimeError("OpenAI backend is not configured. Set OPENAI_API_KEY environment variable.")

        client = await self._get_client()
        options = options or {}

        model = options.get("model", self.default_model)
        temperature = options.get("temperature", 0.7)
        max_tokens = options.get("max_tokens", 1000)
        top_p = options.get("top_p", 1.0)
        frequency_penalty = options.get("frequency_penalty", 0.0)
        presence_penalty = options.get("presence_penalty", 0.0)

        # Convert LLMMessage objects to OpenAI format
        openai_messages = []
        for msg in messages:
            message_dict = {"role": msg.role, "content": msg.content}
            if msg.name:
                message_dict["name"] = msg.name
            openai_messages.append(message_dict)

        try:
            start_time = datetime.now()

            response = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            choice = response.choices[0]
            content = choice.message.content or ""

            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "duration_seconds": duration,
            }

            return LLMResponse(
                content=content,
                usage=usage,
                finish_reason=choice.finish_reason,
                model=response.model,
                metadata={
                    "backend": self.name,
                    "request_id": getattr(response, "id", None),
                    "created": getattr(response, "created", None),
                }
            )

        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to chat with OpenAI: {e}") from e

    async def validate_connection(self) -> bool:
        """Validate that the backend connection is working.

        Returns:
            True if the connection is valid, False otherwise.
        """
        if not self.is_available:
            return False

        try:
            client = await self._get_client()
            # Try a simple models list call to validate the connection
            await client.models.list()
            return True
        except Exception:
            return False
