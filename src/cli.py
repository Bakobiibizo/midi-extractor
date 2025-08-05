import argparse
import torch
from pathlib import Path
from typing import Optional

from datasets.stem_indexer import get_dataset_index, filter_index
from datasets.slakh_stem_dataset import SlakhStemDataset
from train.run_trainer import run_trainer
from utils.visualize_sample import visualize_sample


def parse_args():
    parser = argparse.ArgumentParser(description="MIDI Extractor - Process audio and extract MIDI information")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Index command
    index_parser = subparsers.add_parser('index', help='Index dataset and create metadata')
    index_parser.add_argument('dataset', type=str, help='Path to dataset directory')
    index_parser.add_argument('--output', '-o', type=str, help='Output JSON path', default=None)
    index_parser.add_argument('--debug', action='store_true', help='Enable debug output')

    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter indexed dataset')
    filter_parser.add_argument('index_file', type=str, help='Path to index JSON file')
    filter_parser.add_argument('--output', '-o', type=str, help='Output JSON path', default=None)
    filter_parser.add_argument('--threshold', '-t', type=float, default=-35.0,
                             help='Silence threshold in dB')
    filter_parser.add_argument('--debug', action='store_true', help='Enable debug output')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize dataset samples')
    viz_parser.add_argument('index_file', type=str, help='Path to index JSON file')
    viz_parser.add_argument('--output-dir', '-o', type=str, default='samples',
                          help='Output directory for visualizations')
    viz_parser.add_argument('--num-samples', '-n', type=int, default=5,
                          help='Number of samples to visualize')
    viz_parser.add_argument('--max-len', type=int, default=1000,
                          help='Maximum number of frames to visualize')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('index_file', type=str, help='Path to index JSON file')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    train_parser.add_argument('--clip-seconds', type=float, default=10.0,
                            help='Length of audio clips in seconds')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--output-dir', '-o', type=str, default='checkpoints',
                            help='Output directory for checkpoints')
    train_parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                            help='Device to use for training')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    return parser.parse_args()

def command_index(args):
    """Handle index command"""
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist")
        return 1
    
    print(f"Indexing dataset at {dataset_path}...")
    output_path = get_dataset_index(dataset_path, debug=args.debug)
    print(f"Index saved to {output_path}")
    return 0

def command_filter(args):
    """Handle filter command"""
    index_file = Path(args.index_file)
    if not index_file.exists():
        print(f"Error: Index file {index_file} does not exist")
        return 1
    
    output_path = args.output or index_file.parent / "filtered_dataset.json"
    print(f"Filtering index {index_file}...")
    filtered_path = filter_index(
        index_file, 
        output_path=output_path,
        threshold_db=args.threshold,
        debug=args.debug
    )
    print(f"Filtered index saved to {filtered_path}")
    return 0

def command_visualize(args):
    """Handle visualize command"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    index_file = Path(args.index_file)
    if not index_file.exists():
        print(f"Error: Index file {index_file} does not exist")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset from {index_file}...")
    dataset = SlakhStemDataset(index_file)
    
    print(f"Generating {args.num_samples} sample visualizations...")
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        output_path = output_dir / f"sample_{i:03d}.png"
        visualize_sample(sample, output_path=output_path, max_len=args.max_len)
        print(f"Saved visualization to {output_path}")
    
    return 0

def command_train(args):
    """Handle train command"""
    index_file = Path(args.index_file)
    if not index_file.exists():
        print(f"Error: Index file {index_file} does not exist")
        return 1
    
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
        print(f"Error during training: {str(e)}")
        return 1
    
    return 0

def main():
    args = parse_args()
    
    command_handlers = {
        'index': command_index,
        'filter': command_filter,
        'visualize': command_visualize,
        'train': command_train,
    }
    
    return command_handlers[args.command](args)

if __name__ == "__main__":
    import sys
    sys.exit(main())