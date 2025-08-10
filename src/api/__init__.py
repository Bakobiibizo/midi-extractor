"""
MIDI Generator API Module

This module provides REST API capabilities for audio-to-MIDI transcription.
"""

from .api_server import app, run_server

__all__ = ['app', 'run_server']
