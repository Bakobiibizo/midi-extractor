"""
Training script for the MIDI Generator model.

This module handles the training pipeline including data loading, model initialization,
and the training loop execution.
"""
import json
import os
from pathlib import Path
from typing import Literal, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import DataLoader, random_split, Dataset

# Local imports
from models.interpretable_transcription import TranscriptionModel
from datasets.slakh_stem_dataset import SlakhStemDataset
from datasets.baby_slakh_stem_dataset import BabySlakhStemDataset
from train.trainer import Trainer
from utils.logging import get_logger
from utils.exceptions import (
    MIDIGeneratorError,
    DatasetError,
    ModelError,
    ConfigurationError,
    format_exception
)

# Configure logger
logger = get_logger(__name__, 'training.log')

DatasetType = Literal["babyslakh", "slakh"]

def _validate_json_file(json_path: Path) -> Dict[str, Any]:
    """Validate and load the JSON dataset file.
    
    Args:
        json_path: Path to the JSON dataset file
        
    Returns:
        dict: The loaded JSON data
        
    Raises:
        DatasetError: If the file is invalid or malformed
    """
    try:
        with open(json_path, 'r') as f:
            dataset = json.load(f)
            
        if not isinstance(dataset, dict):
            raise DatasetError(f"Invalid dataset format: expected a dictionary, got {type(dataset).__name__}")
            
        if 'tracks' not in dataset or not isinstance(dataset['tracks'], dict):
            raise DatasetError("Invalid dataset format: 'tracks' key missing or not a dictionary")
            
        return dataset
        
    except json.JSONDecodeError as e:
        raise DatasetError(f"Failed to parse JSON file: {str(e)}") from e
    except Exception as e:
        raise DatasetError(f"Error loading dataset: {str(e)}") from e


def create_dataloaders(
    json_path: str, 
    batch_size: int = 16, 
    clip_seconds: float = 10.0, 
    val_split: float = 0.1, 
    dataset_type: DatasetType = "babyslakh"
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        json_path: Path to the dataset JSON file
        batch_size: Number of samples per batch
        clip_seconds: Length of audio clips in seconds
        val_split: Fraction of data to use for validation
        dataset_type: Type of dataset ("babyslakh" or "slakh")
        
    Returns:
        Tuple containing train and validation DataLoader instances
        
    Raises:
        DatasetError: If there are issues with the dataset
        ConfigurationError: If the configuration is invalid
    """
    try:
        json_path = Path(json_path)
        if not json_path.exists():
            raise DatasetError(f"Dataset file not found: {json_path}")
            
        # Load and validate dataset
        dataset_data = _validate_json_file(json_path)
        logger.info(f"Loaded dataset with {len(dataset_data['tracks'])} tracks")
        
        # Initialize the appropriate dataset class
        dataset: Dataset
        if dataset_type == "babyslakh":
            dataset = BabySlakhStemDataset(str(json_path), clip_seconds=clip_seconds)
        elif dataset_type == "slakh":
            dataset = SlakhStemDataset(str(json_path), clip_seconds=clip_seconds)
        else:
            raise ConfigurationError(f"Invalid dataset type: {dataset_type}")
        
        if len(dataset) == 0:
            raise DatasetError("Dataset is empty after loading")
            
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Validate split configuration
        if not 0 < val_split < 1:
            raise ConfigurationError(f"Validation split must be between 0 and 1, got {val_split}")
            
        # Ensure we have enough samples for at least one batch in validation
        min_val_samples = max(2, batch_size)
        if len(dataset) < min_val_samples * 2:
            raise DatasetError(
                f"Insufficient samples in dataset. Need at least {min_val_samples * 2} "
                f"samples, but only found {len(dataset)}"
            )
            
        # Calculate split sizes
        val_size = min(min_val_samples, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        
        logger.info(f"Splitting dataset: {train_size} training, {val_size} validation samples")
        
        # Perform the split with a fixed random seed for reproducibility
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} validation samples")
        
        # Configure dataloaders
        loader_args = {
            'batch_size': batch_size,
            'num_workers': min(4, batch_size),  # Don't use more workers than batch size
            'pin_memory': torch.cuda.is_available(),
        }
        
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **loader_args
        )
        
        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            drop_last=False,
            **loader_args
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        if not isinstance(e, MIDIGeneratorError):
            raise DatasetError(f"Failed to create dataloaders: {str(e)}") from e
        raise

def run_trainer(
    json_path: str,
    batch_size: int = 16,
    clip_seconds: float = 10.0,
    num_epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lr: float = 1e-4,
    output_dir: str = "checkpoints",
    val_split: float = 0.2,
    dataset: str = "babyslakh",
    mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    compile_model: bool = True
) -> None:
    """Run the training process.
    
    Args:
        json_path: Path to the dataset JSON file
        batch_size: Number of samples per batch
        clip_seconds: Length of audio clips in seconds
        num_epochs: Number of training epochs
        device: Device to use for training ("cuda" or "cpu")
        lr: Learning rate
        output_dir: Directory to save model checkpoints
        val_split: Fraction of data to use for validation
        dataset: Type of dataset ("babyslakh" or "slakh")
        mixed_precision: Enable mixed precision training for faster GPU utilization
        gradient_accumulation_steps: Number of gradient accumulation steps
        compile_model: Compile model for faster training (PyTorch 2.0+)
        val_split: Fraction of data to use for validation
        dataset: Type of dataset ("babyslakh" or "slakh")
        
    Raises:
        ConfigurationError: If the training configuration is invalid
        DatasetError: If there are issues with the dataset
        ModelError: If there are issues with model training
    """
    try:
        # Validate inputs
        if batch_size < 1:
            raise ConfigurationError(f"Batch size must be positive, got {batch_size}")
        if clip_seconds <= 0:
            raise ConfigurationError(f"Clip seconds must be positive, got {clip_seconds}")
        if num_epochs < 1:
            raise ConfigurationError(f"Number of epochs must be positive, got {num_epochs}")
        if not 0 < lr < 1:
            raise ConfigurationError(f"Learning rate must be between 0 and 1, got {lr}")
        if not 0 < val_split < 1:
            raise ConfigurationError(f"Validation split must be between 0 and 1, got {val_split}")
            
        # Set up output directory
        output_dir = Path(output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                raise ConfigurationError(f"No write permission for output directory: {output_dir}")
        except OSError as e:
            raise ConfigurationError(f"Failed to create output directory {output_dir}: {str(e)}") from e
            
        logger.section(
            "🎵 Training Configuration",
            f"Dataset: {json_path}\n"
            f"Dataset type: {dataset}\n"
            f"Batch size: {batch_size}\n"
            f"Clip length: {clip_seconds}s\n"
            f"Epochs: {num_epochs}\n"
            f"Learning rate: {lr}\n"
            f"Validation split: {val_split}\n"
            f"Device: {device}\n"
            f"Output directory: {output_dir.absolute()}"
        )
        
        # Create dataloaders
        logger.progress_section("Creating dataloaders", "Setting up training and validation data...")
        train_loader, val_loader = create_dataloaders(
            json_path=json_path,
            batch_size=batch_size,
            clip_seconds=clip_seconds,
            val_split=val_split,
            dataset_type=dataset
        )
        
        # Initialize model
        logger.progress_section("Initializing model", "Setting up neural network architecture...")
        try:
            model = TranscriptionModel()
            model = model.to(device)
            logger.success(f"Model initialized on device: {device}")
        except Exception as e:
            raise ModelError(f"Failed to initialize model: {str(e)}") from e
        
        # Initialize trainer
        logger.progress_section("Initializing trainer", "Setting up training pipeline...")
        try:
            trainer = Trainer(
                json_path=json_path,
                batch_size=batch_size,
                clip_seconds=clip_seconds,
                num_epochs=num_epochs,
                device=device,
                lr=lr,
                output_dir=str(output_dir),
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                mixed_precision=mixed_precision,
                gradient_accumulation_steps=gradient_accumulation_steps,
                compile_model=compile_model
            )
        except Exception as e:
            raise ModelError(f"Failed to initialize trainer: {str(e)}") from e
        
        # Start training
        logger.progress_section("Starting training", "Beginning neural network training...")
        try:
            trainer.run(num_epochs=num_epochs)
            logger.success("Training completed successfully!")
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            raise
        except Exception as e:
            raise ModelError(f"Training failed: {str(e)}") from e
            
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.debug("Training error details:", exc_info=True)
        if not isinstance(e, MIDIGeneratorError):
            raise ModelError(f"Unexpected error during training: {str(e)}") from e
        raise
