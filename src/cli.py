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

# Rich imports for enhanced CLI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm
from rich import print as rprint

# Local imports
from datasets.stem_indexer import DataSet, TrackData, get_dataset_index, filter_index
from datasets.slakh_stem_dataset import SlakhStemDataset
from train.run_trainer import run_trainer
from utils.visualize_sample import visualize_sample
from utils.logging import get_logger, console, set_debug_mode
from utils.exceptions import (
    MIDIGeneratorError,
    ConfigurationError,
    DatasetError,
    ModelError,
    format_exception
)

# Configure logger
logger = get_logger(__name__)


def show_rich_subcommand_help(command_name: str, parser: argparse.ArgumentParser, title: str, description: str, examples: list):
    """Display Rich-formatted help for a subcommand."""
    # Title panel
    title_text = f"🎵 MIDI Generator - {title}"
    console.print(Panel.fit(title_text, border_style="cyan"))
    console.print()
    
    # Description
    console.print(f"[bold cyan]Description:[/bold cyan]")
    console.print(f"  {description}")
    console.print()
    
    # Arguments table
    table = Table(title="Arguments & Options", show_header=True, header_style="bold cyan")
    table.add_column("Argument", style="bold green", width=25)
    table.add_column("Type", style="yellow", width=12)
    table.add_column("Description", style="white")
    
    # Add arguments from parser
    for action in parser._actions:
        if action.dest == 'help':
            continue
        
        arg_name = action.dest
        if action.option_strings:
            arg_name = ", ".join(action.option_strings)
        elif hasattr(action, 'metavar') and action.metavar:
            arg_name = action.metavar
        
        arg_type = "flag" if action.nargs == 0 else "string"
        if hasattr(action, 'type') and action.type:
            if action.type == int:
                arg_type = "integer"
            elif action.type == float:
                arg_type = "float"
        
        description = action.help or "No description available"
        if action.default and action.default != argparse.SUPPRESS:
            description += f" (default: {action.default})"
        
        table.add_row(arg_name, arg_type, description)
    
    console.print(table)
    console.print()
    
    # Examples
    if examples:
        console.print("[bold cyan]Examples:[/bold cyan]")
        for example in examples:
            console.print(f"  [dim]$[/dim] [bold]{example}[/bold]")
        console.print()
    
    # Usage tip
    tip_text = "💡 [bold cyan]Tip:[/bold cyan] Use [bold]--debug[/bold] flag for verbose logging and detailed error information"
    console.print(Panel(tip_text, border_style="blue", title="Help"))


def print_usage(parser: argparse.ArgumentParser, error: Optional[str] = None) -> None:
    """Print usage information and available commands with optional error message.
    
    Args:
        parser: Argument parser instance
        error: Optional error message to display before usage information
    """
    if error:
        console.print(f"[bold red]Error:[/bold red] {error}")
        console.print()
    
    # Create title panel
    title = Text("🎵 MIDI Generator", style="bold magenta")
    subtitle = Text("Generate and process MIDI music with AI", style="italic cyan")
    console.print(Panel.fit(f"{title}\n{subtitle}", border_style="magenta"))
    console.print()
    
    # Create commands table
    table = Table(title="Available Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="bold green", width=12)
    table.add_column("Description", style="white")
    
    table.add_row("index", "📁 Index dataset and create metadata for audio files")
    table.add_row("filter", "🔍 Filter indexed dataset based on audio quality metrics")
    table.add_row("visualize", "📊 Generate visualizations and samples from the dataset")
    table.add_row("train", "🚀 Train the AI model on your dataset")
    table.add_row("transcribe", "🎼 Convert audio files to MIDI using trained model")
    table.add_row("api", "🌐 Start REST API server for audio-to-MIDI transcription")
    
    console.print(table)
    console.print()
    
    # Add usage tip
    tip_text = "💡 [bold cyan]Tip:[/bold cyan] Use [bold]uv run src/cli.py <command> -h[/bold] for detailed help on any command"
    console.print(Panel(tip_text, border_style="blue", title="Help"))


class SilentArgumentParser(argparse.ArgumentParser):
    """Custom ArgumentParser that suppresses default error output."""
    
    def error(self, message):
        """Override error method to suppress default argparse error output."""
        # Don't print anything - just raise SystemExit
        # Our custom error handling will catch this and show Rich formatting
        raise SystemExit(2)

def parse_args():
    """Parse and validate command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
        
    Raises:
        ConfigurationError: If there are issues with the provided arguments
    """
    parser = SilentArgumentParser(
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
        '--dataset',
        type=str,
        choices=['babyslakh', 'slakh'],
        default='babyslakh',
        help='Dataset type to use (default: babyslakh)'
    )
    train_parser.add_argument(
        '--mixed-precision',
        action='store_true',
        default=True,
        help='Enable mixed precision training for faster GPU utilization (default: True)'
    )
    train_parser.add_argument(
        '--no-mixed-precision',
        action='store_true',
        help='Disable mixed precision training'
    )
    train_parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=1,
        help='Number of gradient accumulation steps (default: 1)'
    )
    train_parser.add_argument(
        '--compile-model',
        action='store_true',
        default=True,
        help='Compile model for faster training (PyTorch 2.0+, default: True)'
    )
    train_parser.add_argument(
        '--no-compile',
        action='store_true',
        help='Disable model compilation'
    )
    train_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for detailed logging'
    )

    # Transcribe command
    transcribe_parser = subparsers.add_parser(
        'transcribe',
        help='Convert audio to MIDI',
        description='Transcribe audio files to MIDI using the trained model.'
    )
    transcribe_parser.add_argument(
        'audio_file',
        type=str,
        help='Path to audio file to transcribe'
    )
    transcribe_parser.add_argument(
        'model_path',
        type=str,
        help='Path to trained model checkpoint'
    )
    transcribe_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output MIDI file path (default: same as input with .mid extension)'
    )
    transcribe_parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference (default: cuda if available, else cpu)'
    )
    transcribe_parser.add_argument(
        '--onset-threshold',
        type=float,
        default=0.5,
        help='Onset detection threshold (0.0-1.0, default: 0.5)'
    )
    transcribe_parser.add_argument(
        '--frame-threshold',
        type=float,
        default=0.5,
        help='Frame detection threshold (0.0-1.0, default: 0.5)'
    )
    transcribe_parser.add_argument(
        '--velocity-scale',
        type=float,
        default=127.0,
        help='Velocity scaling factor (1.0-127.0, default: 127.0)'
    )
    transcribe_parser.add_argument(
        '--min-note-duration',
        type=float,
        default=0.05,
        help='Minimum note duration in seconds (default: 0.05)'
    )
    transcribe_parser.add_argument(
        '--clip-length',
        type=float,
        default=4.0,
        help='Audio clip length for processing in seconds (default: 4.0)'
    )
    transcribe_parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help='Overlap between clips (0.0-0.9, default: 0.5)'
    )
    transcribe_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for detailed logging'
    )

    # API command
    api_parser = subparsers.add_parser(
        'api',
        help='Start API server',
        description='Start the REST API server for audio-to-MIDI transcription.'
    )
    api_parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind the server to (default: 0.0.0.0)'
    )
    api_parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port to bind the server to (default: 8000)'
    )
    api_parser.add_argument(
        '--model-path', '--midi-model-path', '-m',
        dest='model_path',
        type=str,
        help='Path to trained model checkpoint (auto-detects latest if not provided)'
    )
    api_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for detailed logging'
    )

    # Check if help is requested for a specific subcommand
    if len(sys.argv) >= 2 and sys.argv[-1] in ['-h', '--help']:
        command = sys.argv[1] if len(sys.argv) >= 2 else None
        
        # Define Rich help content for each command
        rich_help_content = {
            'index': {
                'title': '📁 Dataset Indexing',
                'description': 'Scan a directory of audio files and create a comprehensive metadata index. This process analyzes audio properties, extracts features, and creates a JSON index file for use with other commands.',
                'examples': [
                    'uv run src/cli.py index /path/to/audio/dataset',
                    'uv run src/cli.py index /path/to/dataset --output my_index.json',
                    'uv run src/cli.py index /path/to/dataset --debug'
                ]
            },
            'filter': {
                'title': '🔍 Dataset Filtering',
                'description': 'Filter an existing dataset index based on audio quality metrics such as silence detection. This helps remove low-quality or problematic audio files from your training dataset.',
                'examples': [
                    'uv run src/cli.py filter dataset_index.json',
                    'uv run src/cli.py filter dataset_index.json --threshold -30.0',
                    'uv run src/cli.py filter dataset_index.json --output filtered_dataset.json --debug'
                ]
            },
            'visualize': {
                'title': '📊 Dataset Visualization',
                'description': 'Generate visual representations of your audio and MIDI data. This creates spectrograms, waveforms, and other visualizations to help you understand your dataset before training.',
                'examples': [
                    'uv run src/cli.py visualize dataset_index.json',
                    'uv run src/cli.py visualize dataset_index.json --num-samples 10',
                    'uv run src/cli.py visualize dataset_index.json --output-dir visualizations --max-len 2000'
                ]
            },
            'train': {
                'title': '🚀 Model Training',
                'description': 'Train the AI model on your prepared dataset. This process uses deep learning to learn patterns from your audio data and create a model capable of generating new MIDI sequences.',
                'examples': [
                    'uv run src/cli.py train dataset_index.json',
                    'uv run src/cli.py train dataset_index.json --epochs 50 --batch-size 16',
                    'uv run src/cli.py train dataset_index.json --device cuda --learning-rate 0.001 --output-dir my_model'
                ]
            },
            'transcribe': {
                'title': '🎼 Audio Transcription',
                'description': 'Convert audio files to MIDI using the trained model. This process analyzes audio content and generates corresponding MIDI note sequences.',
                'examples': [
                    'uv run src/cli.py transcribe audio.wav model.pth',
                    'uv run src/cli.py transcribe audio.wav model.pth --output song.mid',
                    'uv run src/cli.py transcribe audio.wav model.pth --onset-threshold 0.3 --clip-length 6.0'
                ]
            },
            'api': {
                'title': '🌐 API Server',
                'description': 'Start a REST API server for audio-to-MIDI transcription. This allows you to send audio files via HTTP requests and receive MIDI files back.',
                'examples': [
                    'uv run src/cli.py api',
                    'uv run src/cli.py api --port 8080 --model-path model.pth',
                    'uv run src/cli.py api --host localhost --port 3000'
                ]
            }
        }
        
        if command in rich_help_content:
            # Get the subparser for this command
            subparser = None
            for action in parser._subparsers._actions:
                if isinstance(action, argparse._SubParsersAction):
                    subparser = action.choices.get(command)
                    break
            
            if subparser:
                help_info = rich_help_content[command]
                show_rich_subcommand_help(
                    command, 
                    subparser, 
                    help_info['title'], 
                    help_info['description'], 
                    help_info['examples']
                )
                sys.exit(0)
    
    try:
        args = parser.parse_args()
        return args
    except SystemExit as e:
        # If no command was provided, show our enhanced usage
        if len(sys.argv) == 1:
            print_usage(parser)
            sys.exit(0)
        
        # Check if a command was provided but has argument errors (missing args or invalid syntax)
        if len(sys.argv) >= 2 and sys.argv[1] in ['index', 'filter', 'visualize', 'train']:
            command = sys.argv[1]
            
            # Define Rich help content for each command (same as above)
            rich_help_content = {
                'index': {
                    'title': '📁 Dataset Indexing',
                    'description': 'Scan a directory of audio files and create a comprehensive metadata index. This process analyzes audio properties, extracts features, and creates a JSON index file for use with other commands.',
                    'examples': [
                        'uv run src/cli.py index /path/to/audio/dataset',
                        'uv run src/cli.py index /path/to/dataset --output my_index.json',
                        'uv run src/cli.py index /path/to/dataset --debug'
                    ]
                },
                'filter': {
                    'title': '🔍 Dataset Filtering',
                    'description': 'Filter an existing dataset index based on audio quality metrics such as silence detection. This helps remove low-quality or problematic audio files from your training dataset.',
                    'examples': [
                        'uv run src/cli.py filter dataset_index.json',
                        'uv run src/cli.py filter dataset_index.json --threshold -30.0',
                        'uv run src/cli.py filter dataset_index.json --output filtered_dataset.json --debug'
                    ]
                },
                'visualize': {
                    'title': '📊 Dataset Visualization',
                    'description': 'Generate visual representations of your audio and MIDI data. This creates spectrograms, waveforms, and other visualizations to help you understand your dataset before training.',
                    'examples': [
                        'uv run src/cli.py visualize dataset_index.json',
                        'uv run src/cli.py visualize dataset_index.json --num-samples 10',
                        'uv run src/cli.py visualize dataset_index.json --output-dir visualizations --max-len 2000'
                    ]
                },
                'train': {
                    'title': '🚀 Model Training',
                    'description': 'Train the AI model on your prepared dataset. This process uses deep learning to learn patterns from your audio data and create a model capable of generating new MIDI sequences.',
                    'examples': [
                        'uv run src/cli.py train dataset_index.json',
                        'uv run src/cli.py train dataset_index.json --epochs 50 --batch-size 16',
                        'uv run src/cli.py train dataset_index.json --device cuda --learning-rate 0.001 --output-dir my_model'
                    ]
                }
            }
            
            if command in rich_help_content:
                # Get the subparser for this command
                subparser = None
                for action in parser._subparsers._actions:
                    if isinstance(action, argparse._SubParsersAction):
                        subparser = action.choices.get(command)
                        break
                
                if subparser:
                    help_info = rich_help_content[command]
                    
                    # Show specific error message with details
                    error_msg = str(e)
                    
                    # Try to extract specific parameter information from the error
                    specific_error = ""
                    if "unrecognized arguments:" in error_msg:
                        # Extract the unrecognized argument
                        unrecognized = error_msg.split("unrecognized arguments:")[1].strip()
                        specific_error = f"Unrecognized parameter: [bold yellow]{unrecognized}[/bold yellow]"
                    elif "the following arguments are required:" in error_msg:
                        # Extract required arguments
                        required = error_msg.split("the following arguments are required:")[1].strip()
                        specific_error = f"Missing required parameter: [bold yellow]{required}[/bold yellow]"
                    elif "invalid choice:" in error_msg:
                        # Extract invalid choice information
                        choice_info = error_msg.split("invalid choice:")[1].strip()
                        specific_error = f"Invalid choice: [bold yellow]{choice_info}[/bold yellow]"
                    else:
                        specific_error = f"[dim]{error_msg}[/dim]"
                    
                    # Create a formatted error panel
                    error_panel = Panel(
                        f"[bold red]❌ Invalid arguments for '{command}' command[/bold red]\n\n{specific_error}",
                        title="[bold red]Error[/bold red]",
                        border_style="red",
                        padding=(1, 2)
                    )
                    console.print(error_panel)
                    console.print()
                    
                    # Then show the Rich help
                    show_rich_subcommand_help(
                        command, 
                        subparser, 
                        help_info['title'], 
                        help_info['description'], 
                        help_info['examples']
                    )
                    sys.exit(1)
        
        # Re-raise for other argument parsing errors
        raise
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
        
        # Show indexing header
        console.print(Panel.fit(
            "📁 [bold cyan]Dataset Indexing[/bold cyan]\n"
            "Creating metadata index for audio files",
            border_style="cyan"
        ))
        console.print()
        
        # Validate dataset directory
        if not dataset_path.exists():
            raise DatasetError(f"Dataset directory not found: {dataset_path}")
        if not dataset_path.is_dir():
            raise DatasetError(f"Dataset path is not a directory: {dataset_path}")
            
        console.print(f"📂 [bold]Dataset path:[/bold] {dataset_path}")
        
        # Determine output path
        output_path = args.output
        if output_path is None:
            output_path = dataset_path.parent / f"{dataset_path.name}_index.json"
            console.print(f"📄 [bold]Output path:[/bold] {output_path} [dim](auto-generated)[/dim]")
        else:
            console.print(f"📄 [bold]Output path:[/bold] {output_path}")
            
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Perform the indexing with progress indication
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🔍 Scanning and indexing audio files...", total=None)
            
            try:
                dataset_index = get_dataset_index(dataset_path)
                progress.update(task, description="💾 Saving index file...")
                
                # Save the index
                with open(output_path, 'w') as f:
                    json.dump(dataset_index.model_dump(), f, indent=2)
                    
                progress.update(task, description="✅ Indexing completed!")
                
            except Exception as e:
                progress.update(task, description="❌ Indexing failed!")
                raise DatasetError(f"Failed to index dataset: {str(e)}") from e
        
        # Show success summary
        console.print()
        success_table = Table(show_header=False, box=None, padding=(0, 1))
        success_table.add_column("Icon", width=3)
        success_table.add_column("Info")
        
        success_table.add_row("✅", f"[bold green]Successfully indexed {len(dataset_index.tracks)} tracks[/bold green]")
        success_table.add_row("📁", f"Index saved to: [bold]{output_path}[/bold]")
        success_table.add_row("📊", f"File size: [bold]{output_path.stat().st_size / 1024:.1f} KB[/bold]")
        
        console.print(Panel(success_table, title="[bold green]Indexing Complete[/bold green]", border_style="green"))
        return 0
            
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
        
        # Show filtering header
        console.print(Panel.fit(
            "🔍 [bold cyan]Dataset Filtering[/bold cyan]\n"
            "Filtering dataset based on audio quality metrics",
            border_style="cyan"
        ))
        console.print()
        
        # Validate input file
        if not index_path.exists():
            raise DatasetError(f"Index file not found: {index_path}")
            
        console.print(f"📄 [bold]Input file:[/bold] {index_path}")
        console.print(f"🔇 [bold]Silence threshold:[/bold] {args.threshold}dB")
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
            console.print(f"💾 [bold]Output file:[/bold] {output_path}")
        else:
            output_path = index_path.parent / f"{index_path.stem}_filtered{index_path.suffix}"
            console.print(f"💾 [bold]Output file:[/bold] {output_path} [dim](auto-generated)[/dim]")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Perform filtering with progress indication
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("📖 Loading dataset index...", total=None)
            
            try:
                with open(index_path, 'r') as f:
                    dataset = json.loads(f.read())
                    
                # Validate dataset structure
                if not isinstance(dataset, dict) or 'tracks' not in dataset:
                    raise DatasetError("Invalid index format: 'tracks' key not found")
                
                progress.update(task, description="🔍 Filtering tracks by quality metrics...")
                
                # Filter the dataset
                filtered_dataset = filter_index(dataset, silence_threshold=args.threshold)
                
                progress.update(task, description="💾 Saving filtered dataset...")
                
                # Save the filtered dataset
                with open(output_path, 'w') as f:
                    json.dump(filtered_dataset, f, indent=2)
                    
                progress.update(task, description="✅ Filtering completed!")
                
            except json.JSONDecodeError as e:
                progress.update(task, description="❌ Failed to parse JSON!")
                raise DatasetError(f"Failed to parse JSON file: {index_path}") from e
            except Exception as e:
                progress.update(task, description="❌ Filtering failed!")
                raise DatasetError(f"Error during filtering: {str(e)}") from e
        
        # Show success summary
        console.print()
        original_count = len(dataset['tracks'])
        filtered_count = len(filtered_dataset['tracks'])
        removed_count = original_count - filtered_count
        
        success_table = Table(show_header=False, box=None, padding=(0, 1))
        success_table.add_column("Icon", width=3)
        success_table.add_column("Info")
        
        success_table.add_row("📊", f"[bold]Original tracks:[/bold] {original_count}")
        success_table.add_row("✅", f"[bold green]Filtered tracks:[/bold green] {filtered_count}")
        success_table.add_row("🗑️", f"[bold red]Removed tracks:[/bold red] {removed_count}")
        success_table.add_row("📁", f"Filtered dataset saved to: [bold]{output_path}[/bold]")
        success_table.add_row("📊", f"File size: [bold]{output_path.stat().st_size / 1024:.1f} KB[/bold]")
        
        console.print(Panel(success_table, title="[bold green]Filtering Complete[/bold green]", border_style="green"))
        return 0
            
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
        
        # Show visualization header
        console.print(Panel.fit(
            "📊 [bold cyan]Dataset Visualization[/bold cyan]\n"
            "Generating visual representations of audio and MIDI data",
            border_style="cyan"
        ))
        console.print()
        
        # Validate input file
        if not index_path.exists():
            raise DatasetError(f"Index file not found: {index_path}")
            
        console.print(f"📄 [bold]Input file:[/bold] {index_path}")
        console.print(f"📁 [bold]Output directory:[/bold] {output_dir}")
        console.print(f"🔢 [bold]Max samples:[/bold] {getattr(args, 'num_samples', 5)}")
        console.print(f"📏 [bold]Max length:[/bold] {getattr(args, 'max_len', 1000)} frames")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and validate dataset
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            load_task = progress.add_task("📖 Loading dataset index...", total=None)
            
            try:
                with open(index_path, 'r') as f:
                    dataset = json.load(f)
                    
                # Validate dataset structure
                if not isinstance(dataset, dict) or 'tracks' not in dataset:
                    raise DatasetError("Invalid index format: 'tracks' key not found")
                
                total_tracks = len(dataset['tracks'])
                progress.update(load_task, description=f"✅ Loaded {total_tracks} tracks", completed=100, total=100)
                
                # Limit the number of tracks to visualize if specified
                tracks_to_visualize = list(dataset['tracks'].items())
                if hasattr(args, 'num_samples') and args.num_samples:
                    tracks_to_visualize = tracks_to_visualize[:args.num_samples]
                    total_tracks = len(tracks_to_visualize)
                
                # Visualize each track with progress bar
                viz_task = progress.add_task("🎨 Generating visualizations...", total=total_tracks)
                
                success_count = 0
                failed_tracks = []
                
                for i, (track_id, track) in enumerate(tracks_to_visualize):
                    try:
                        progress.update(viz_task, description=f"🎨 Visualizing: {track_id[:20]}...")
                        visualize_sample(track, output_dir, max_len=getattr(args, 'max_len', 1000))
                        success_count += 1
                    except Exception as e:
                        failed_tracks.append((track_id, str(e)))
                        if getattr(args, 'debug', False):
                            console.print(f"[yellow]⚠️  Failed to visualize {track_id}: {str(e)}[/yellow]")
                    
                    progress.update(viz_task, advance=1)
                
                progress.update(viz_task, description="✅ Visualization completed!")
                
            except json.JSONDecodeError as e:
                progress.update(load_task, description="❌ Failed to parse JSON!")
                raise DatasetError(f"Failed to parse JSON file: {index_path}") from e
            except Exception as e:
                progress.update(load_task, description="❌ Visualization failed!")
                raise DatasetError(f"Error during visualization: {str(e)}") from e
        
        # Show success summary
        console.print()
        success_table = Table(show_header=False, box=None, padding=(0, 1))
        success_table.add_column("Icon", width=3)
        success_table.add_column("Info")
        
        success_table.add_row("✅", f"[bold green]Successfully visualized:[/bold green] {success_count}/{total_tracks} tracks")
        if failed_tracks:
            success_table.add_row("⚠️", f"[bold yellow]Failed tracks:[/bold yellow] {len(failed_tracks)}")
        success_table.add_row("📁", f"Visualizations saved to: [bold]{output_dir}[/bold]")
        
        # Calculate output directory size
        try:
            total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
            success_table.add_row("📊", f"Total output size: [bold]{total_size / 1024 / 1024:.1f} MB[/bold]")
        except:
            pass
        
        panel_style = "green" if success_count > 0 else "yellow"
        panel_title = "[bold green]Visualization Complete[/bold green]" if success_count > 0 else "[bold yellow]Visualization Completed with Warnings[/bold yellow]"
        
        console.print(Panel(success_table, title=panel_title, border_style=panel_style))
        
        # Show failed tracks if any and debug is enabled
        if failed_tracks and getattr(args, 'debug', False):
            console.print()
            console.print("[bold yellow]Failed Tracks Details:[/bold yellow]")
            for track_id, error in failed_tracks[:5]:  # Show first 5 failures
                console.print(f"  • [red]{track_id}[/red]: {error}")
            if len(failed_tracks) > 5:
                console.print(f"  ... and {len(failed_tracks) - 5} more (use --debug for full list)")
        
        if success_count == 0:
            raise DatasetError("Failed to generate any visualizations. Check logs for details.")
                
        return 0
            
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
        
        # Show training header
        console.print(Panel.fit(
            "🚀 [bold cyan]Model Training[/bold cyan]\n"
            "Training AI model on your prepared dataset",
            border_style="cyan"
        ))
        console.print()
        
        # Validate input file
        if not index_file.exists():
            raise DatasetError(f"Index file not found: {index_file}")
            
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Show training configuration
        config_table = Table(title="Training Configuration", show_header=True, header_style="bold cyan")
        config_table.add_column("Parameter", style="bold green", width=20)
        config_table.add_column("Value", style="white")
        
        config_table.add_row("📄 Dataset", str(index_file))
        config_table.add_row("📦 Batch Size", str(args.batch_size))
        config_table.add_row("⏱️ Clip Length", f"{args.clip_seconds}s")
        config_table.add_row("🔄 Epochs", str(args.epochs))
        config_table.add_row("🖥️ Device", str(args.device))
        config_table.add_row("📈 Learning Rate", str(getattr(args, 'lr', getattr(args, 'learning_rate', 'N/A'))))
        config_table.add_row("💾 Output Directory", str(output_dir))
        
        # Add validation split if available
        if hasattr(args, 'val_split'):
            config_table.add_row("📊 Validation Split", f"{args.val_split:.1%}")
        
        # Add GPU optimization settings
        mixed_precision = getattr(args, 'mixed_precision', True) and not getattr(args, 'no_mixed_precision', False)
        compile_model = getattr(args, 'compile_model', True) and not getattr(args, 'no_compile', False)
        grad_accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
        
        config_table.add_row("⚡ Mixed Precision", "✅ Enabled" if mixed_precision else "❌ Disabled")
        config_table.add_row("🚀 Model Compilation", "✅ Enabled" if compile_model else "❌ Disabled")
        if grad_accum_steps > 1:
            config_table.add_row("🔄 Gradient Accumulation", f"{grad_accum_steps} steps")
        
        console.print(config_table)
        console.print()
        
        # Use specified dataset type
        dataset_name = args.dataset
        console.print(f"🎵 [bold]Dataset type detected:[/bold] {dataset_name}")
        console.print()
        
        # Confirm before starting training
        if not Confirm.ask("🚀 [bold cyan]Start training?[/bold cyan]", default=True):
            console.print("[yellow]Training cancelled by user[/yellow]")
            return 0
        
        console.print()
        console.print("🚀 [bold green]Starting training...[/bold green]")
        console.print("[dim]Note: Training progress will be shown by the trainer module[/dim]")
        console.print()
        
        try:
            # Get learning rate from args (handle both --lr and --learning-rate)
            learning_rate = getattr(args, 'lr', getattr(args, 'learning_rate', 1e-4))
            
            # Handle GPU optimization flags
            mixed_precision = getattr(args, 'mixed_precision', True) and not getattr(args, 'no_mixed_precision', False)
            compile_model = getattr(args, 'compile_model', True) and not getattr(args, 'no_compile', False)
            grad_accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
            
            run_trainer(
                json_path=index_file,
                batch_size=args.batch_size,
                clip_seconds=args.clip_seconds,
                num_epochs=args.epochs,
                device=args.device,
                lr=learning_rate,
                output_dir=output_dir,
                dataset=dataset_name,
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=grad_accum_steps,
                compile_model=compile_model
            )
            
            # Show completion message
            console.print()
            completion_table = Table(show_header=False, box=None, padding=(0, 1))
            completion_table.add_column("Icon", width=3)
            completion_table.add_column("Info")
            
            completion_table.add_row("✅", "[bold green]Training completed successfully![/bold green]")
            completion_table.add_row("📁", f"Model checkpoints saved to: [bold]{output_dir}[/bold]")
            
            # Try to get output directory size
            try:
                total_size = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
                completion_table.add_row("📊", f"Total model size: [bold]{total_size / 1024 / 1024:.1f} MB[/bold]")
            except:
                pass
            
            console.print(Panel(completion_table, title="[bold green]Training Complete[/bold green]", border_style="green"))
            
        except KeyboardInterrupt:
            console.print()
            console.print("[yellow]⚠️  Training interrupted by user[/yellow]")
            console.print("[dim]Partial checkpoints may have been saved to the output directory[/dim]")
            return 1
        except Exception as e:
            console.print()
            console.print(f"[bold red]❌ Training failed:[/bold red] {str(e)}")
            if getattr(args, 'debug', False):
                console.print("\n[dim]Debug traceback:[/dim]")
                console.print_exception()
            raise DatasetError(f"Error during training: {str(e)}") from e
            
        return 0
        
    except Exception as e:
        if not isinstance(e, MIDIGeneratorError):
            raise DatasetError(f"Error in train command: {str(e)}") from e
        raise


def handle_transcribe_command(args):
    """Handle the transcribe command for audio-to-MIDI conversion."""
    from .inference.inference import create_inference_engine
    
    try:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]🎵 MIDI Generator - 🎼 Audio Transcription[/bold cyan]",
            border_style="cyan"
        ))
        
        # Validate inputs
        audio_path = Path(args.audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        output_path = Path(args.output) if args.output else audio_path.with_suffix('.mid')
        
        # Display configuration
        config_table = Table(title="Transcription Configuration", show_header=False, box=box.ROUNDED)
        config_table.add_column("Setting", style="bold blue", width=25)
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Audio file", str(audio_path))
        config_table.add_row("Model checkpoint", str(model_path))
        config_table.add_row("Output MIDI", str(output_path))
        config_table.add_row("Device", args.device)
        config_table.add_row("Onset threshold", str(args.onset_threshold))
        config_table.add_row("Frame threshold", str(args.frame_threshold))
        config_table.add_row("Clip length", f"{args.clip_length}s")
        
        console.print(config_table)
        console.print()
        
        # Create inference engine
        with console.status("[bold green]Initializing inference engine..."):
            engine = create_inference_engine(
                model_path=str(model_path),
                device=args.device,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                velocity_scale=args.velocity_scale,
                min_note_duration=args.min_note_duration,
                clip_length=args.clip_length,
                overlap=args.overlap
            )
        
        # Run transcription
        console.print("[bold green]🎵 Starting audio transcription...[/bold green]")
        midi = engine.transcribe_audio(audio_path, output_path)
        
        # Display results
        num_notes = len(midi.instruments[0].notes) if midi.instruments else 0
        duration = midi.get_end_time()
        
        results_table = Table(title="Transcription Results", show_header=False, box=box.ROUNDED)
        results_table.add_column("Metric", style="bold blue", width=20)
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Notes generated", str(num_notes))
        results_table.add_row("Duration", f"{duration:.2f}s")
        results_table.add_row("Output file", str(output_path))
        
        console.print(results_table)
        console.print(f"[bold green]✅ Transcription completed successfully![/bold green]")
        
        return 0
        
    except Exception as e:
        console.print(f"[bold red]❌ Transcription failed:[/bold red] {str(e)}")
        if getattr(args, 'debug', False):
            console.print("\n[dim]Debug traceback:[/dim]")
            console.print_exception()
        return 1


def handle_api_command(args):
    """Handle the API server command."""
    from .api.api_server import run_server
    
    try:
        console.print()
        console.print(Panel.fit(
            "[bold magenta]🚀 MIDI Generator - 🌐 API Server[/bold magenta]",
            border_style="magenta"
        ))
        
        # Validate model path if provided
        if args.model_path:
            model_path = Path(args.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Display server configuration
        config_table = Table(title="API Server Configuration")
        config_table.add_column("Setting", style="bold blue", width=20)
        config_table.add_column("Value", style="green")
        
        config_table.add_row("Host", args.host)
        config_table.add_row("Port", str(args.port))
        config_table.add_row("Model path", args.model_path or "Auto-detect latest checkpoint")
        
        console.print(config_table)
        console.print()
        console.print(f"[bold green]🌐 Starting API server at http://{args.host}:{args.port}[/bold green]")
        console.print(f"[dim]📖 API documentation: http://{args.host}:{args.port}/docs[/dim]")
        console.print(f"[dim]🔍 Health check: http://{args.host}:{args.port}/health[/dim]")
        console.print()
        
        # Run server
        run_server(args.host, args.port, args.model_path)
        
        return 0
        
    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]⚠️  API server stopped by user[/yellow]")
        return 0
    except Exception as e:
        console.print(f"[bold red]❌ API server failed:[/bold red] {str(e)}")
        if getattr(args, 'debug', False):
            console.print("\n[dim]Debug traceback:[/dim]")
            console.print_exception()
        return 1
        
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
        debug_mode = getattr(args, 'debug', False)
        set_debug_mode(debug_mode)
        if debug_mode:
            logger.set_level('DEBUG')
        
        # Show welcome message for debug mode
        if getattr(args, 'debug', False):
            logger.debug("Debug mode enabled - verbose logging active")
        
        # Dispatch to appropriate command handler
        command_handlers = {
            'index': command_index,
            'filter': command_filter,
            'visualize': command_visualize,
            'train': command_train,
            'transcribe': handle_transcribe_command,
            'api': handle_api_command
        }
        
        if args.command in command_handlers:
            return command_handlers[args.command](args)
        else:
            raise ConfigurationError(f"Unknown command: {args.command}")
            

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
        return 130  # Standard exit code for SIGINT
    except MIDIGeneratorError as e:
        logger.error(str(e))
        # Try to get debug flag, but handle case where args wasn't parsed
        try:
            debug_mode = getattr(args, 'debug', False)
        except NameError:
            debug_mode = '--debug' in sys.argv
            
        if debug_mode and hasattr(e, '__traceback__'):
            logger.debug("Debug traceback:", exc_info=True)
        return 1
    except Exception as e:
        logger.critical(str(e))
        logger.warning("This appears to be an unexpected error. Please consider reporting this issue.")
        
        # Try to get debug flag, but handle case where args wasn't parsed
        try:
            debug_mode = getattr(args, 'debug', False)
        except NameError:
            debug_mode = '--debug' in sys.argv
            
        if debug_mode:
            logger.debug("Full traceback:", exc_info=True)
        else:
            logger.info("Run with --debug for detailed error information.")
        return 2

if __name__ == "__main__":
    import sys
    sys.exit(main())