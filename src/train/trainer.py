import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
import logging
from typing import Dict, Tuple, Optional

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

class Trainer:
    def __init__(
        self, 
        json_path: str,
        batch_size: int,
        clip_seconds: float,
        num_epochs: int,
        device: str,
        lr: float,
        output_dir: str,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader
    ):
        self.json_path = json_path
        self.batch_size = batch_size
        self.clip_seconds = clip_seconds
        self.num_epochs = num_epochs
        self.device = device
        self.lr = lr
        self.output_dir = output_dir
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.max_grad_norm = 1.0

    def _log_grad_norms(self):
        """Log gradient norms for debugging"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        logger.debug(f'Gradient norm: {total_norm:.6f}')

    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate input batch for NaNs/Infs and proper shapes"""
        required_tensors = ['spec', 'onset', 'frame']
        
        for key in required_tensors:
            if key not in batch:
                raise ValueError(f'Batch is missing required key: {key}')
                
        for key, value in batch.items():
            # Skip non-tensor values
            if not isinstance(value, torch.Tensor):
                logger.debug(f'Skipping validation for non-tensor key: {key}')
                continue
                
            # Check for NaN/Inf values
            if value.dtype.is_floating_point or value.dtype.is_complex:
                if torch.isnan(value).any():
                    raise ValueError(f'Batch contains NaN values in {key}')
                if torch.isinf(value).any():
                    raise ValueError(f'Batch contains Inf values in {key}')
                
                # Check tensor dimensions (only for numeric tensors)
                if key == 'spec' and len(value.shape) != 3:  # [batch, n_mels, time]
                    raise ValueError(f'Expected spec shape [batch, n_mels, time], got {value.shape}')
                elif key in ['onset', 'frame', 'velocity'] and len(value.shape) != 3:  # [batch, time, pitch]
                    raise ValueError(f'Expected {key} shape [batch, time, pitch], got {value.shape}')

    def _process_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process a single batch and return loss and metrics
        
        Args:
            batch: Dictionary containing:
                - spec: [B, n_mels, T] spectrogram
                - onset: [B, T, 128] onset labels
                - frame: [B, T, 128] frame labels
                - velocity: [B, T, 128] velocity labels
                - lengths: [B] original sequence lengths
                - mask: [B, T] padding mask
                - meta: List[dict] metadata
                
        Returns:
            Tuple of (loss, metrics_dict)
        """
        try:
            # Debug: Log input shapes
            logger.debug(f"Input batch shapes: { {k: tuple(v.shape) if isinstance(v, torch.Tensor) else f'list[{len(v)}]' for k, v in batch.items() if k != 'meta'} }")
                
            # Extract and move tensors to device
            spec = batch['spec'].to(self.device)               # [B, 128, T]
            onset_labels = batch['onset'].to(self.device)      # [B, T, 128]
            frame_labels = batch['frame'].to(self.device)      # [B, T, 128]
    
            # Log input shape
            logger.debug(f"Spec shape before model: {spec.shape}")
    
            # Forward pass through model
            outputs = self.model(spec)
            onset_logits = outputs["onset"]     # [B, T, 128]
            frame_logits = outputs["frame"]     # [B, T, 128]
    
            # Loss calculation
            onset_loss = F.binary_cross_entropy(onset_logits, onset_labels, reduction='mean')
            frame_loss = F.binary_cross_entropy(frame_logits, frame_labels, reduction='mean')
            loss = onset_loss + frame_loss
    
            # Metrics
            with torch.no_grad():
                onset_preds = (onset_logits > 0.5).float()
                frame_preds = (frame_logits > 0.5).float()
                onset_acc = (onset_preds == onset_labels).float().mean()
                frame_acc = (frame_preds == frame_labels).float().mean()
    
            metrics = {
                'loss': loss.item(),
                'onset_loss': onset_loss.item(),
                'frame_loss': frame_loss.item(),
                'onset_acc': onset_acc.item(),
                'frame_acc': frame_acc.item()
            }
    
            return loss, metrics

        except Exception as e:
            logger.error(f'Error in batch processing: {str(e)}')
            logger.error(f'Batch stats - spec: {spec.shape}, onset: {onset_labels.shape}, frame: {frame_labels.shape}')
            if 'onset_logits' in locals():
                logger.error(f'Onset logits stats: mean={onset_logits.mean().item()}, std={onset_logits.std().item()}, min={onset_logits.min().item()}, max={onset_logits.max().item()}')
            if 'frame_logits' in locals():
                logger.error(f'Frame logits stats: mean={frame_logits.mean().item()}, std={frame_logits.std().item()}, min={frame_logits.min().item()}, max={frame_logits.max().item()}')
            raise

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_metrics = {
            'loss': 0.0,
            'onset_loss': 0.0,
            'frame_loss': 0.0,
            'onset_acc': 0.0,
            'frame_acc': 0.0
        }
        
        # Get number of batches, ensuring it's at least 1
        num_batches = max(1, len(self.train_loader))
        processed_batches = 0
        
        # Initialize progress bar with total=None for dynamic updates
        progress_bar = tqdm(desc=f'Train Epoch {epoch}', leave=False)
        
        try:
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # Skip empty batches
                    if batch is None or len(batch) == 0:
                        logger.warning(f'Skipping empty batch {batch_idx}')
                        continue
                        
                    # Validate input batch
                    try:
                        self._validate_batch(batch)
                    except Exception as e:
                        logger.error(f'Batch validation failed: {str(e)}')
                        continue
                    
                    # Process batch and calculate loss
                    loss, metrics = self._process_batch(batch)
                    
                    # Skip if loss is NaN or inf
                    if not torch.isfinite(loss):
                        logger.error(f'Invalid loss value: {loss}')
                        continue
                    
                    # Update metrics
                    for k, v in metrics.items():
                        total_metrics[k] += v
                    processed_batches += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'o_acc': f"{metrics['onset_acc']:.2%}",
                        'f_acc': f"{metrics['frame_acc']:.2%}"
                    })
                    progress_bar.update(1)
                    
                except Exception as e:
                    logger.error(f'Error in batch {batch_idx}: {str(e)}')
                    if batch is not None:
                        logger.error(f'Batch stats: { {k: v.shape if hasattr(v, "shape") else str(v)[:50] + "..." for k, v in batch.items()} }')
                    continue
            
            if processed_batches == 0:
                logger.warning('No valid batches processed in this epoch')
                return total_metrics
                
            # Calculate average metrics
            avg_metrics = {k: v / processed_batches for k, v in total_metrics.items()}
            
            logger.info(
                f'Train Epoch {epoch} - '
                f'Processed {processed_batches}/{num_batches} batches - '
                f'Loss: {avg_metrics["loss"]:.4f}, '
                f'Onset Loss: {avg_metrics["onset_loss"]:.4f}, '
                f'Frame Loss: {avg_metrics["frame_loss"]:.4f}, '
                f'Onset Acc: {avg_metrics["onset_acc"]:.2%}, '
                f'Frame Acc: {avg_metrics["frame_acc"]:.2%}'
            )
            
            return avg_metrics
            
        except Exception as e:
            logger.exception(f'Unexpected error in train_epoch: {str(e)}')
            # Return current metrics if we have any, otherwise raise
            if processed_batches > 0:
                return {k: v / processed_batches for k, v in total_metrics.items()}
            raise

    def validate(self, epoch: int = None) -> Dict[str, float]:
        """Run validation on the validation set."""
        self.model.eval()
        total_metrics = {
            'loss': 0.0,
            'onset_loss': 0.0,
            'frame_loss': 0.0,
            'onset_acc': 0.0,
            'frame_acc': 0.0
        }
        processed_batches = 0
    
        try:
            with torch.no_grad():
                progress_desc = f'Validation Epoch {epoch}' if epoch is not None else 'Validation'
                progress_bar = tqdm(self.val_loader, desc=progress_desc, leave=False)
    
                for batch_idx, batch in enumerate(progress_bar):
                    if batch is None or len(batch) == 0:
                        logger.warning(f'Skipping empty validation batch {batch_idx}')
                        continue
                    
                    try:
                        self._validate_batch(batch)
    
                        # Move data to device
                        spec = batch['spec'].to(self.device)  # [B, 128, T]
                        onset_labels = batch['onset'].to(self.device)  # [B, T, 128]
                        frame_labels = batch['frame'].to(self.device)  # [B, T, 128]
    
                        # Forward pass (model handles unsqueeze internally)
                        outputs = self.model(spec)
                        onset_logits = outputs["onset"]  # [B, T, 128]
                        frame_logits = outputs["frame"]  # [B, T, 128]
    
                        # Losses
                        onset_loss = F.binary_cross_entropy(onset_logits, onset_labels, reduction='mean')
                        frame_loss = F.binary_cross_entropy(frame_logits, frame_labels, reduction='mean')
                        loss = onset_loss + frame_loss
    
                        # Accuracy
                        onset_preds = (onset_logits > 0.5).float()
                        frame_preds = (frame_logits > 0.5).float()
                        onset_acc = (onset_preds == onset_labels).float().mean()
                        frame_acc = (frame_preds == frame_labels).float().mean()
    
                        batch_metrics = {
                            'loss': loss.item(),
                            'onset_loss': onset_loss.item(),
                            'frame_loss': frame_loss.item(),
                            'onset_acc': onset_acc.item(),
                            'frame_acc': frame_acc.item()
                        }
    
                        for k, v in batch_metrics.items():
                            total_metrics[k] += v
                        processed_batches += 1
    
                        progress_bar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'o_acc': f"{onset_acc.item():.2%}",
                            'f_acc': f"{frame_acc.item():.2%}"
                        })
    
                    except Exception as e:
                        logger.error(f'Error in validation batch {batch_idx}: {str(e)}')
                        logger.error(f'Batch stats: { {k: v.shape if hasattr(v, "shape") else str(v)[:50]+"..." for k, v in batch.items()} }')
                        continue
                    
            if processed_batches == 0:
                logger.warning('No valid batches in validation')
                return total_metrics
    
            avg_metrics = {k: v / processed_batches for k, v in total_metrics.items()}
            logger.info(
                f'Validation - Processed {processed_batches} batches - '
                f'Loss: {avg_metrics["loss"]:.4f}, '
                f'Onset Loss: {avg_metrics["onset_loss"]:.4f}, '
                f'Frame Loss: {avg_metrics["frame_loss"]:.4f}, '
                f'Onset Acc: {avg_metrics["onset_acc"]:.2%}, '
                f'Frame Acc: {avg_metrics["frame_acc"]:.2%}'
            )
            return avg_metrics
    

        except Exception as e:
            logger.exception(f'Unexpected error in validation: {str(e)}')
            if processed_batches > 0:
                return {k: v / processed_batches for k, v in total_metrics.items()}
            return total_metrics
        
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'batch_size': self.batch_size,
                'clip_seconds': self.clip_seconds,
                'lr': self.lr
            }
        }
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_path)
            logger.info(f'Saved best model to {best_path}')

    def run(self, num_epochs: Optional[int] = None) -> None:
        """Run training loop"""
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        best_val_loss = float('inf')
        
        try:
            for epoch in range(1, num_epochs + 1):
                logger.info(f'Starting epoch {epoch}/{num_epochs}')
                
                # Train for one epoch
                train_metrics = self.train_epoch(epoch)
                
                # Validate
                val_metrics = self.validate(epoch)
                
                # Skip if validation failed
                if not val_metrics:
                    logger.warning(f'Skipping checkpoint save due to validation failure in epoch {epoch}')
                    continue
                
                # Save checkpoint
                self._save_checkpoint(epoch)
                
                # Save best model
                current_val_loss = val_metrics.get('loss', float('inf'))
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    self._save_checkpoint(epoch, is_best=True)
                    logger.info(f'New best model saved with validation loss: {best_val_loss:.6f}')
                
                logger.info(
                    f'Epoch {epoch} - '
                    f'Train Loss: {train_metrics.get("loss", float("nan")):.6f}, '
                    f'Val Loss: {current_val_loss:.6f}, '
                    f'Best Val Loss: {best_val_loss:.6f}'
                )
                
        except KeyboardInterrupt:
            logger.info('Training interrupted by user')
            raise
        except Exception as e:
            logger.exception('Error during training:')
            raise
        finally:
            logger.info('Training completed')
