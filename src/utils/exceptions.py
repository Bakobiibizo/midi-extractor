"""
Custom exceptions for the MIDI Generator application.

This module defines custom exceptions to provide more meaningful error messages
and better error handling throughout the application.
"""
from typing import Optional, Dict, Any
import sys
import traceback

class MIDIGeneratorError(Exception):
    """Base exception class for all MIDI Generator specific exceptions."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message

class ConfigurationError(MIDIGeneratorError):
    """Raised when there is an error in the application configuration."""
    pass

class DatasetError(MIDIGeneratorError):
    """Raised when there is an error related to dataset handling."""
    pass

class ModelError(MIDIGeneratorError):
    """Raised when there is an error related to model operations."""
    pass

def format_exception(e: Exception) -> str:
    """Format an exception with its traceback for logging."""
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    return ''.join(tb_lines)
