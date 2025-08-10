import torch
from torch.utils.data import DataLoader
from pathlib import Path
from .slakh_stem_dataset import SlakhStemDataset


def collate_fn(batch):
    """
    Collate function for SlakhStemDataset that pads sequences to the same length.
    
    Args:
        batch: List of samples from SlakhStemDataset
        
    Returns:
        Dict containing batched and padded tensors with shapes:
        - spec: [B, n_mels, T]  (spectrogram)
        - onset: [B, T, 128]    (onset labels)
        - frame: [B, T, 128]    (frame labels)
        - velocity: [B, T, 128] (velocity labels)
        - lengths: [B]          (original sequence lengths)
        - mask: [B, T]          (padding mask)
        - meta: List[dict]      (metadata)
    """
    # Get max sequence length (T) in batch
    max_len = max(sample["spec"].shape[1] for sample in batch)
    
    def pad_tensor(tensor, target_len, dim):
        """Pad tensor along specified dimension to target length"""
        pad_size = list(tensor.shape)
        pad_size[dim] = target_len - tensor.shape[dim]
        if pad_size[dim] <= 0:
            return tensor
        pad = torch.zeros(*pad_size, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=dim)
    
    # Initialize batch tensors
    specs = []
    onsets = []
    frames = []
    velocities = []
    lengths = []
    
    for sample in batch:
        # Get original lengths
        seq_len = sample["onset"].shape[0]
        lengths.append(seq_len)
        
        # Pad tensors
        spec = pad_tensor(sample["spec"], max_len, dim=1)  # [n_mels, T] -> [n_mels, max_len]
        onset = pad_tensor(sample["onset"], max_len, dim=0)  # [T, 128] -> [max_len, 128]
        frame = pad_tensor(sample["frame"], max_len, dim=0)  # [T, 128] -> [max_len, 128]
        velocity = pad_tensor(sample["velocity"], max_len, dim=0)  # [T, 128] -> [max_len, 128]
        
        specs.append(spec)
        onsets.append(onset)
        frames.append(frame)
        velocities.append(velocity)
    
    # Stack all tensors
    try:
        specs = torch.stack(specs)  # [B, n_mels, max_len]
        onsets = torch.stack(onsets)  # [B, max_len, 128]
        frames = torch.stack(frames)  # [B, max_len, 128]
        velocities = torch.stack(velocities)  # [B, max_len, 128]
        lengths = torch.tensor(lengths, dtype=torch.long)  # [B]
        
        # Create attention masks (1 for real data, 0 for padding)
        masks = torch.arange(max_len).expand(len(batch), max_len) < lengths.unsqueeze(1)  # [B, max_len]
        masks = masks.float()
        
        # Prepare metadata
        metadata = [{
            "track": sample["track"],
            "stem_id": sample["stem_id"],
            "program": sample["program"],
            "wav_path": sample["wav_path"],
            "midi_path": sample["midi_path"],
        } for sample in batch]
        
        batch_dict = {
            "spec": specs,         # [B, n_mels, T]
            "onset": onsets,       # [B, T, 128]
            "frame": frames,       # [B, T, 128]
            "velocity": velocities, # [B, T, 128]
            "lengths": lengths,    # [B]
            "mask": masks,         # [B, T]
            "meta": metadata       # List[dict]
        }
        
        return batch_dict
        
    except Exception as e:
        import logging
        logging.error(f"Error in collate_fn: {str(e)}")
        logging.error(f"Input batch sizes: {[b['spec'].shape for b in batch]}")
        logging.error(f"Padded shapes - specs: {[s.shape for s in specs]}")
        raise


def create_dataloader(
    json_path: str | Path,
    batch_size: int = 8,
    clip_seconds: float = 10.0,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader for the SlakhStemDataset.
    
    Args:
        json_path: Path to the dataset JSON file
        batch_size: Batch size for the DataLoader
        clip_seconds: Length of audio clips in seconds
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        
    Returns:
        DataLoader instance
    """
    dataset = SlakhStemDataset(json_path, clip_seconds=clip_seconds)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_train_val_dataloaders(
    json_path: str | Path,
    train_split: float = 0.8,
    batch_size: int = 8,
    clip_seconds: float = 10.0,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders with a random split.
    
    Args:
        json_path: Path to the dataset JSON file
        train_split: Fraction of data to use for training (0.0 to 1.0)
        batch_size: Batch size for both DataLoaders
        clip_seconds: Length of audio clips in seconds
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    dataset = SlakhStemDataset(json_path, clip_seconds=clip_seconds)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    # Create random split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


# Example usage functions
def test_dataloader(json_path: str | Path, batch_size: int = 4):
    """
    Test function to verify the dataloader works correctly.
    
    Args:
        json_path: Path to the dataset JSON file
        batch_size: Batch size for testing
    """
    print(f"Testing dataloader with {json_path}")
    
    dataloader = create_dataloader(
        json_path, 
        batch_size=batch_size, 
        num_workers=0,  # Use 0 for debugging
        shuffle=False
    )
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test first batch
    batch = next(iter(dataloader))
    
    print("\nBatch shapes:")
    print(f"  spec: {batch['spec'].shape}")
    print(f"  onset: {batch['onset'].shape}")
    print(f"  frame: {batch['frame'].shape}")
    print(f"  velocity: {batch['velocity'].shape}")
    print(f"  lengths: {batch['lengths'].shape}")
    print(f"  mask: {batch['mask'].shape}")
    print(f"  metadata: {len(batch['meta'])} items")
    
    print("\nFirst sample metadata:")
    for key, value in batch['meta'][0].items():
        print(f"  {key}: {value}")
    
    print(f"\nSequence lengths in batch: {batch['lengths'].tolist()}")
    
    return batch


if __name__ == "__main__":
    # Test the dataloader
    test_json = "datasets/babyslakh_16k/filtered_dataset.json"
    test_dataloader(test_json)
