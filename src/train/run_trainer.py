import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split

from models.interpretable_transcription import TranscriptionModel
from datasets.slakh_stem_dataset import SlakhStemDataset, BabyslakhStemDataset
from train.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def create_dataloaders(json_path, batch_size=16, clip_seconds=10.0, val_split=0.1, dataset: "babyslakh" | "slakh" = "babyslakh"):
    """Create train and validation dataloaders."""
    if dataset == "babyslakh":
        dataset = BabyslakhStemDataset(json_path, clip_seconds=clip_seconds)
    elif dataset == "slakh":
        dataset = SlakhStemDataset(json_path, clip_seconds=clip_seconds)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    try:
        # Load and verify dataset
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        
        if 'tracks' not in dataset or not isinstance(dataset['tracks'], dict):
            raise ValueError("Invalid dataset format: 'tracks' key missing or not a dictionary")
        
        logger.info(f"Loaded dataset with {len(dataset['tracks'])} tracks")
        
        # Create dataset
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty after loading")
        
        # Ensure we have enough samples for at least one batch in validation
        min_val_samples = max(2, batch_size)  # At least 2 samples or batch_size, whichever is larger
        
        if len(dataset) < min_val_samples * 2:  # Need at least 2 batches worth of data
            raise ValueError(f"Not enough samples in dataset. Need at least {min_val_samples * 2} samples, got {len(dataset)}")
            
        # Calculate split sizes
        val_size = min(min_val_samples, int(len(dataset) * val_split))
        train_size = len(dataset) - val_size
        
        logger.info(f"Splitting dataset: {train_size} training, {val_size} validation samples")
        
        # Perform the split
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Split dataset: {len(train_dataset)} training, {len(val_dataset)} validation samples")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        if 'dataset' in locals():
            logger.error(f"Dataset length: {len(dataset)}")
        raise

def run_trainer(
    json_path: str,
    batch_size: int = 16,
    clip_seconds: float = 10.0,
    num_epochs: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lr: float = 1e-4,
    output_dir: str = "checkpoints",
    val_split: float = 0.2
):
    """Run the training process."""
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            json_path=json_path,
            batch_size=batch_size,
            clip_seconds=clip_seconds,
            val_split=val_split
        )
        
        # Initialize model
        logger.info("Initializing model...")
        model = TranscriptionModel()
        model = model.to(device)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            json_path=json_path,
            batch_size=batch_size,
            clip_seconds=clip_seconds,
            num_epochs=num_epochs,
            device=device,
            lr=lr,
            output_dir=output_dir,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.run(num_epochs=num_epochs)
        
    except Exception as e:
        logger.exception("Error in run_trainer:")
        raise
