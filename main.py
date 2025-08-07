#!/usr/bin/env python3
"""
MIDI Generator - Main Entry Point

This module serves as the main entry point for the MIDI Generator application.
It initializes the application, sets up logging, and handles any uncaught exceptions.
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports
from src.cli import main as cli_main
from src.utils.exceptions import (
    MIDIGeneratorError,
    ConfigurationError,
    DatasetError,
    ModelError,
    format_exception
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('midi_generator.log')
    ]
)
logger = logging.getLogger(__name__)

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
        
        # Log environment information
        logger.info("=" * 80)
        logger.info(f"MIDI Generator - Starting up")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.critical("Failed to set up environment", exc_info=True)
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
        print("\nOperation cancelled by user")
        return 130  # Standard exit code for SIGINT
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        print(f"\nConfiguration error: {str(e)}", file=sys.stderr)
        return 2
        
    except DatasetError as e:
        logger.error(f"Dataset error: {str(e)}")
        print(f"\nDataset error: {str(e)}", file=sys.stderr)
        return 3
        
    except ModelError as e:
        logger.error(f"Model error: {str(e)}")
        print(f"\nModel error: {str(e)}", file=sys.stderr)
        return 4
        
    except MIDIGeneratorError as e:
        logger.error(f"Application error: {str(e)}")
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1
        
    except Exception as e:
        logger.critical("Unexpected error", exc_info=True)
        print("\nAn unexpected error occurred. Please check the logs for details.", file=sys.stderr)
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # This is a last resort error handler
        print("\nA critical error occurred. Please check the logs for details.", file=sys.stderr)
        logger.critical("Unhandled exception in main thread", exc_info=True)
        sys.exit(1)