#!/usr/bin/env python3
"""
MIDI Generator - Main Entry Point

This module serves as the main entry point for the MIDI Generator application.
It initializes the application, sets up logging, and handles any uncaught exceptions.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Any

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from src.cli import main as cli_main
from src.utils.logging import get_logger, print_startup_banner
from src.utils.exceptions import (
    MIDIGeneratorError,
    ConfigurationError,
    DatasetError,
    ModelError,
    format_exception
)

# Configure logger
logger = get_logger(__name__)

def setup_environment() -> None:
    """Set up the runtime environment.
    
    This function performs any necessary environment setup, such as:
    - Setting environment variables
    - Verifying required dependencies
    - Creating necessary directories
    """
    try:
        # Create necessary directories if they don't exist
        Path('checkpoints').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        
        # Show startup banner
        print_startup_banner("MIDI Generator", "1.0.0")
        
        # Log environment information
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
    except Exception as e:
        logger.error("Failed to set up environment")
        logger.debug("Environment setup error details:", exc_info=True)
        raise ConfigurationError(f"Environment setup failed: {str(e)}") from e

def main(args: Optional[list] = None) -> int:
    """Main entry point for the application.
    
    Args:
        args: Command line arguments. If None, uses sys.argv[1:]
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        # Set up the environment
        setup_environment()
        
        # Run the CLI with the provided arguments
        return cli_main()
        
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        return 130  # Standard exit code for SIGINT
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 2
        
    except DatasetError as e:
        logger.error(f"Dataset error: {str(e)}")
        return 3
        
    except ModelError as e:
        logger.error(f"Model error: {str(e)}")
        return 4
        
    except MIDIGeneratorError as e:
        logger.error(f"Application error: {str(e)}")
        return 1
        
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        logger.debug("Unexpected error details:", exc_info=True)
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # This is a last resort error handler
        logger.critical(f"Unhandled exception in main thread: {str(e)}")
        logger.debug("Critical error details:", exc_info=True)
        sys.exit(1)