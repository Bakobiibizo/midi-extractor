"""
Command Line Interface for MIDI Generator.

This module provides a command-line interface for various MIDI generation tasks,
including dataset indexing, filtering, visualization, and model training.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from pydantic import ValidationError

# Local imports
from datasets.stem_indexer import DataSet, TrackData, get_dataset_index, filter_index
from datasets.slakh_stem_dataset import SlakhStemDataset
from train.run_trainer import run_trainer
from utils.visualize_sample import visualize_sample
from utils.exceptions import (
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


def print_usage(parser: argparse.ArgumentParser, error: Optional[str] = None) -> None:
    """Print usage information and available commands with optional error message.
    
    Args:
        parser: Argument parser instance
        error: Optional error message to display before usage information
    """
    if error:
        print(f"\nError: {error}", file=sys.stderr)
    
    print("\nMIDI Generator - Generate and process MIDI music")
    print("\nAvailable commands:")
    print("  index     - Index dataset and create metadata")
    print("  filter    - Filter indexed dataset based on audio quality")
    print("  visualize - Visualize samples from the dataset")
    print("  train     - Train the model")
    print("\nUse 'python -m midi_generator.cli <command> -h' for help on a specific command")
    parser.print_help()


def parse_args():
    """Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
        
    Raises:
        ConfigurationError: If there are issues with the provided arguments
    """
    parser = argparse.ArgumentParser(
        description="MIDI Generator - Generate and process MIDI music",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Index command
    index_parser = subparsers.add_parser(
        'index',
        help='Index dataset and create metadata',
        description='Create an index of audio files and their metadata.'
    )
    index_parser.add_argument(
        'dataset',
        type=str,
        help='Path to dataset directory containing audio files'
    )
    index_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON path for the index file',
        default=None
    )
    index_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for detailed logging'
    )

    # Filter command
    filter_parser = subparsers.add_parser(
        'filter',
        help='Filter indexed dataset',
        description='Filter the dataset index based on audio quality metrics.'
    )
    filter_parser.add_argument(
        'index_file',
        type=str,
        help='Path to the index JSON file to filter'
    )
    filter_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output JSON path for the filtered index',
        default=None
    )
    filter_parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=-35.0,
        help='Silence threshold in dB. Files with more silence than this will be filtered out.'
    )
    filter_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for detailed logging'
    )

    # Visualize command
    viz_parser = subparsers.add_parser(
        'visualize',
        help='Visualize dataset samples',
        description='Generate visualizations of audio and MIDI data.'
    )
    viz_parser.add_argument(
        'index_file',
        type=str,
        help='Path to the index JSON file to visualize'
    )
    viz_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='samples',
        help='Output directory for generated visualizations'
    )
    viz_parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=5,
        help='Number of samples to visualize (default: 5)'
    )
    viz_parser.add_argument(
        '--max-len',
        type=int,
        default=1000,
        help='Maximum number of frames to visualize (default: 1000)'
    )
    viz_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for detailed logging'
    )

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        'index_file',
        type=str,
        help='Path to index JSON file containing training data'
    )
    train_parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Number of samples per batch (default: 8)'
    )
    train_parser.add_argument(
        '--clip-seconds', '-c',
        type=float,
        default=10.0,
        help='Length of audio clips in seconds (default: 10.0)'
    )
    train_parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    train_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='checkpoints',
        help='Directory to save model checkpoints (default: checkpoints/)'
    )
    train_parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (default: cuda if available, else cpu)'
    )
    train_parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    train_parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Fraction of data to use for validation (default: 0.1)'
    )
    train_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for detailed logging'
    )

    try:
        args = parser.parse_args()
        return args
    except Exception as e:
        raise ConfigurationError(f"Invalid command line arguments: {str(e)}")

def command_index(args) -> int:
    """Handle index command
    
    Args:
        args: Command line arguments with 'dataset' and 'debug' attributes
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
        
    Raises:
        DatasetError: If there are issues with the dataset directory or indexing process
        ConfigurationError: If there are issues with the output path
    """
    try:
        dataset_path = Path(args.dataset)
        
        # Validate dataset directory
        if not dataset_path.exists():
            raise DatasetError(f"Dataset directory not found: {dataset_path}")
        if not dataset_path.is_dir():
            raise DatasetError(f"Dataset path is not a directory: {dataset_path}")
            
        logger.info(f"Indexing dataset at: {dataset_path}")
        
        # Determine output path
        output_path = args.output
        if output_path is None:
            output_path = dataset_path.parent / f"{dataset_path.name}_index.json"
            logger.info(f"No output path specified, using: {output_path}")
            
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Perform the indexing
        try:
            dataset_index = get_dataset_index(dataset_path)
            
            # Save the index
            with open(output_path, 'w') as f:
                json.dump(dataset_index.model_dump(), f, indent=2)
                
            logger.info(f"Successfully indexed {len(dataset_index.tracks)} tracks to {output_path}")
            return 0
            
        except Exception as e:
            raise DatasetError(f"Failed to index dataset: {str(e)}") from e
            
    except Exception as e:
        if not isinstance(e, MIDIGeneratorError):
            raise DatasetError(f"Error during dataset indexing: {str(e)}") from e
        raise


def command_filter(args) -> int:
    """Handle filter command
    
    Args:
        args: Command line arguments with 'index_file', 'threshold', and 'output' attributes
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
        
    Raises:
        DatasetError: If there are issues with the input index file or filtering process
        ConfigurationError: If there are issues with the output path
    """
    try:
        index_path = Path(args.index_file)
        
        # Validate input file
        if not index_path.exists():
            raise DatasetError(f"Index file not found: {index_path}")
            
        logger.info(f"Filtering index: {index_path}")
        logger.debug(f"Using silence threshold: {args.threshold}dB")
        
        try:
            with open(index_path, 'r') as f:
                dataset = json.loads(f.read())
                
            # Validate dataset structure
            if not isinstance(dataset, dict) or 'tracks' not in dataset:
                raise DatasetError("Invalid index format: 'tracks' key not found")
                
            # Filter the dataset
            filtered_dataset = filter_index(dataset, silence_threshold=args.threshold)
            
            # Determine output path
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = index_path.parent / f"{index_path.stem}_filtered{index_path.suffix}"
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the filtered dataset
            with open(output_path, 'w') as f:
                json.dump(filtered_dataset, f, indent=2)
                
            logger.info(f"Filtered dataset saved to: {output_path}")
            logger.info(f"Original tracks: {len(dataset['tracks'])}, Filtered tracks: {len(filtered_dataset['tracks'])}")
            
            return 0
            
        except json.JSONDecodeError as e:
            raise DatasetError(f"Failed to parse JSON file: {index_path}") from e
        except Exception as e:
            raise DatasetError(f"Error during filtering: {str(e)}") from e
            
    except Exception as e:
        if not isinstance(e, MIDIGeneratorError):
            raise DatasetError(f"Error in filter command: {str(e)}") from e
        raise


def command_visualize(args) -> int:
    """Handle visualize command
    
    Args:
        args: Command line arguments with 'index_file' and 'output_dir' attributes
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
        
    Raises:
        DatasetError: If there are issues with the input index file or visualization process
        ConfigurationError: If there are issues with the output directory
    """
    try:
        index_path = Path(args.index_file)
        output_dir = Path(args.output_dir)
        
        # Validate input file
        if not index_path.exists():
            raise DatasetError(f"Index file not found: {index_path}")
            
        logger.info(f"Generating visualizations from: {index_path}")
        logger.info(f"Output directory: {output_dir}")
        
        try:
            with open(index_path, 'r') as f:
                dataset = json.load(f)
                
            # Validate dataset structure
            if not isinstance(dataset, dict) or 'tracks' not in dataset:
                raise DatasetError("Invalid index format: 'tracks' key not found")
                
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Visualize each track
            total_tracks = len(dataset['tracks'])
            logger.info(f"Generating visualizations for {total_tracks} tracks...")
            
            success_count = 0
            for i, (track_id, track) in enumerate(dataset['tracks'].items(), 1):
                try:
                    logger.debug(f"Visualizing track {i}/{total_tracks}: {track_id}")
                    visualize_sample(track, output_dir)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to visualize track {track_id}: {str(e)}")
                    continue
                    
            logger.info(f"Successfully generated {success_count}/{total_tracks} visualizations")
            logger.info(f"Visualizations saved to: {output_dir}")
            
            if success_count == 0:
                raise DatasetError("Failed to generate any visualizations. Check logs for details.")
                
            return 0
            
        except json.JSONDecodeError as e:
            raise DatasetError(f"Failed to parse JSON file: {index_path}") from e
        except Exception as e:
            raise DatasetError(f"Error during visualization: {str(e)}") from e
            
    except Exception as e:
        if not isinstance(e, MIDIGeneratorError):
            raise DatasetError(f"Error in visualize command: {str(e)}") from e
        raise


def command_train(args) -> int:
    """Handle train command
    
    Args:
        args: Command line arguments with 'index_file', 'batch_size', 'clip_seconds', 'epochs', 'output_dir', 'device', and 'lr' attributes
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
        
    Raises:
        DatasetError: If there are issues with the input index file or training process
        ConfigurationError: If there are issues with the output directory or training configuration
    """
    try:
        index_file = Path(args.index_file)
        if not index_file.exists():
            raise DatasetError(f"Index file not found: {index_file}")
            
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training with configuration:")
        print(f"  Dataset: {index_file}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Clip length: {args.clip_seconds}s")
        print(f"  Epochs: {args.epochs}")
        print(f"  Device: {args.device}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Output directory: {output_dir}")
        
        dataset_name = "babyslakh" if "baby" in index_file.name else "slakh"
        try:
            run_trainer(
                json_path=index_file,
                batch_size=args.batch_size,
                clip_seconds=args.clip_seconds,
                num_epochs=args.epochs,
                device=args.device,
                lr=args.lr,
                output_dir=output_dir,
                dataset=dataset_name
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            return 1
        except Exception as e:
            raise DatasetError(f"Error during training: {str(e)}") from e
            
        return 0
        
    except Exception as e:
        if not isinstance(e, MIDIGeneratorError):
            raise DatasetError(f"Error in train command: {str(e)}") from e
        raise


def main() -> int:
    """Main entry point for the command-line interface.
    
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
    try:
        args = parse_args()
        
        # Configure logging based on debug flag
        log_level = logging.DEBUG if getattr(args, 'debug', False) else logging.INFO
        logger.setLevel(log_level)
        
        # Log command execution
        logger.info(f"Executing command: {args.command}")
        
        # Dispatch to appropriate command handler
        command_handlers = {
            'index': command_index,
            'filter': command_filter,
            'visualize': command_visualize,
            'train': command_train
        }
        
        if args.command in command_handlers:
            return command_handlers[args.command](args)
        else:
            raise ConfigurationError(f"Unknown command: {args.command}")
            
    except MIDIGeneratorError as e:
        logger.error(f"Error: {str(e)}")
        if getattr(args, 'debug', False) and hasattr(e, '__traceback__'):
            logger.debug(format_exception(e))
        return 1
    except Exception as e:
        logger.critical("An unexpected error occurred", exc_info=True)
        print(f"\nA critical error occurred: {str(e)}\n", file=sys.stderr)
        print("Please run with --debug for more details or report this issue.", file=sys.stderr)
        return 2

if __name__ == "__main__":
    import sys
    sys.exit(main())