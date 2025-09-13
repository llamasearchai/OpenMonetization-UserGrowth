"""CLI interface for OpenMonetization-UserAcquisition.

This module provides the command-line interface for interacting with the OMUA system.
"""

from .main import app
from .commands import *

__all__ = [
    "app",
]
