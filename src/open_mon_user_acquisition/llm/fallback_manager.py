"""LLM fallback manager for graceful backend switching."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..core.interfaces import LLMBackendInterface
from ..core.types import LLMResponse, LLMMessage
from .openai_backend import OpenAIBackend
from .ollama_backend import OllamaBackend

logger = logging.getLogger(__name__)


class LLMFallbackManager(LLMBackendInterface):
    """Manager that handles fallback between multiple LLM backends."""

    def __init__(
        self,
        backends: Optional[List[LLMBackendInterface]] = None,
        fallback_order: Optional[List[str]] = None,
    ):
        """Initialize the fallback manager.

        Args:
            backends: List of LLM backends to manage. If None, creates default OpenAI and Ollama backends.
            fallback_order: Order of backend names to try. If None, uses ["openai", "ollama"].
        """
        if backends is None:
            self.backends = [
                OpenAIBackend(),
                OllamaBackend(),
            ]
        else:
            self.backends = backends

        self.fallback_order = fallback_order or ["openai", "ollama"]
        self.backend_map = {backend.name: backend for backend in self.backends}

        # Validate fallback order
        for backend_name in self.fallback_order:
            if backend_name not in self.backend_map:
                raise ValueError(f"Backend '{backend_name}' specified in fallback_order but not found in backends")

    @property
    def name(self) -> str:
        """Return the manager name."""
        return "fallback_manager"

    @property
    def is_available(self) -> bool:
        """Check if at least one backend is available."""
        return any(backend.is_available for backend in self.backends)

    async def _get_available_backend(self) -> Optional[LLMBackendInterface]:
        """Get the first available backend according to fallback order.

        Returns:
            The first available backend, or None if none are available.
        """
        for backend_name in self.fallback_order:
            backend = self.backend_map.get(backend_name)
            if backend and backend.is_available:
                # Double-check availability with a connection test
                try:
                    if await backend.validate_connection():
                        return backend
                except Exception as e:
                    logger.warning(f"Backend {backend_name} failed connection validation: {e}")
                    continue

        return None

    async def generate(
        self,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate text using the first available backend.

        Args:
            prompt: The text prompt to generate from.
            options: Additional options for generation.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            RuntimeError: If no backends are available.
        """
        backend = await self._get_available_backend()
        if backend is None:
            raise RuntimeError("No LLM backends are available. Please check your configuration.")

        try:
            logger.info(f"Using backend: {backend.name}")
            response = await backend.generate(prompt, options)

            # Add backend info to metadata
            if response.metadata is None:
                response.metadata = {}
            response.metadata["used_backend"] = backend.name

            return response

        except Exception as e:
            logger.error(f"Backend {backend.name} failed: {e}")

            # Try next backend in fallback order
            remaining_backends = [b for b in self.fallback_order if b != backend.name]
            for backend_name in remaining_backends:
                next_backend = self.backend_map.get(backend_name)
                if next_backend and next_backend.is_available:
                    try:
                        logger.info(f"Falling back to backend: {next_backend.name}")
                        response = await next_backend.generate(prompt, options)

                        # Add backend info to metadata
                        if response.metadata is None:
                            response.metadata = {}
                        response.metadata["used_backend"] = next_backend.name
                        response.metadata["fallback_used"] = True

                        return response

                    except Exception as fallback_error:
                        logger.error(f"Fallback backend {backend_name} also failed: {fallback_error}")
                        continue

            # All backends failed
            raise RuntimeError(f"All LLM backends failed. Last error: {e}") from e

    async def chat(
        self,
        messages: List[LLMMessage],
        options: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Conduct a chat conversation using the first available backend.

        Args:
            messages: List of messages in the conversation.
            options: Additional options for the chat.

        Returns:
            LLMResponse containing the response and metadata.

        Raises:
            RuntimeError: If no backends are available.
        """
        backend = await self._get_available_backend()
        if backend is None:
            raise RuntimeError("No LLM backends are available. Please check your configuration.")

        try:
            logger.info(f"Using backend: {backend.name}")
            response = await backend.chat(messages, options)

            # Add backend info to metadata
            if response.metadata is None:
                response.metadata = {}
            response.metadata["used_backend"] = backend.name

            return response

        except Exception as e:
            logger.error(f"Backend {backend.name} failed: {e}")

            # Try next backend in fallback order
            remaining_backends = [b for b in self.fallback_order if b != backend.name]
            for backend_name in remaining_backends:
                next_backend = self.backend_map.get(backend_name)
                if next_backend and next_backend.is_available:
                    try:
                        logger.info(f"Falling back to backend: {next_backend.name}")
                        response = await next_backend.chat(messages, options)

                        # Add backend info to metadata
                        if response.metadata is None:
                            response.metadata = {}
                        response.metadata["used_backend"] = next_backend.name
                        response.metadata["fallback_used"] = True

                        return response

                    except Exception as fallback_error:
                        logger.error(f"Fallback backend {backend_name} also failed: {fallback_error}")
                        continue

            # All backends failed
            raise RuntimeError(f"All LLM backends failed. Last error: {e}") from e

    async def validate_connection(self) -> bool:
        """Validate that at least one backend connection is working.

        Returns:
            True if at least one backend connection is valid, False otherwise.
        """
        backend = await self._get_available_backend()
        return backend is not None

    def get_backend_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all backends.

        Returns:
            Dictionary mapping backend names to their status information.
        """
        status = {}
        for backend in self.backends:
            status[backend.name] = {
                "is_available": backend.is_available,
                "name": backend.name,
            }
        return status

    def add_backend(self, backend: LLMBackendInterface) -> None:
        """Add a new backend to the manager.

        Args:
            backend: The backend to add.
        """
        if backend.name in self.backend_map:
            raise ValueError(f"Backend with name '{backend.name}' already exists")

        self.backends.append(backend)
        self.backend_map[backend.name] = backend

    def remove_backend(self, backend_name: str) -> None:
        """Remove a backend from the manager.

        Args:
            backend_name: Name of the backend to remove.
        """
        if backend_name not in self.backend_map:
            raise ValueError(f"Backend with name '{backend_name}' not found")

        backend = self.backend_map[backend_name]
        self.backends.remove(backend)
        del self.backend_map[backend_name]

        # Remove from fallback order if present
        if backend_name in self.fallback_order:
            self.fallback_order.remove(backend_name)
