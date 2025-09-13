"""OpenMonetization-UserAcquisition - Enterprise user acquisition strategy platform.

A comprehensive Python framework for user acquisition strategy, experimentation,
and optimization in fintech/crypto contexts with LLM-powered agents and
multi-channel performance marketing capabilities.
"""

from .core import *
from .config import *
from .llm import *
from .storage import *
from .orchestrator import *
from .cli import *

__version__ = "0.1.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",

    # Main components
    "WorkflowOrchestrator",
    "ConfigManager",
    "LLMFallbackManager",
    "SQLiteStorageBackend",

    # CLI
    "app",
]
