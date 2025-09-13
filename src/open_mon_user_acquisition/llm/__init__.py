"""LLM backend implementations for OpenMonetization-UserAcquisition.

This module provides implementations for various LLM backends including OpenAI
and Ollama, with automatic fallback and graceful degradation.
"""

from .openai_backend import OpenAIBackend
from .ollama_backend import OllamaBackend
from .fallback_manager import LLMFallbackManager

__all__ = [
    "OpenAIBackend",
    "OllamaBackend",
    "LLMFallbackManager",
]
