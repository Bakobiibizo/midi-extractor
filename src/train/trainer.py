import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Local imports
from utils.logging import get_logger, training_progress, get_gpu_info, get_system_info, save_summary_to_file

# Configure logger
logger = get_logger(__name__, 'training.log')

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
        val_loader: torch.utils.data.DataLoader,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        compile_model: bool = True
    ):
        self.json_path = json_path
        self.batch_size = batch_size
        self.clip_seconds = clip_seconds
        self.num_epochs = num_epochs
        self.device = device
        self.lr = lr
        self.output_dir = output_dir
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # GPU Optimizations
        self.model = model.to(device)
        
        # Model compilation for faster training (PyTorch 2.0+)
        if compile_model and hasattr(torch, 'compile'):
            logger.info("🚀 Compiling model for faster training...")
            self.model = torch.compile(self.model)
            logger.success("Model compiled successfully!")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.max_grad_norm = 1.0
        
        # Mixed precision training
        self.scaler = GradScaler() if mixed_precision and device.startswith('cuda') else None
        if self.scaler:
            logger.info("⚡ Mixed precision training enabled for faster GPU utilization")
        
        # Training statistics
        self.training_start_time = None
        self.epoch_losses = []
        self.val_losses = []
        self.best_epoch = 0
        self.total_batches_processed = 0

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
            onset_loss = F.binary_cross_entropy_with_logits(onset_logits, onset_labels, reduction='mean')
            frame_loss = F.binary_cross_entropy_with_logits(frame_logits, frame_labels, reduction='mean')
            loss = onset_loss + frame_loss
    
            # Apply sigmoid to get probabilities for accuracy calculation
            onset_probs = torch.sigmoid(onset_logits)
            frame_probs = torch.sigmoid(frame_logits)
            onset_preds = (onset_probs > 0.5).float()
            frame_preds = (frame_probs > 0.5).float()
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

    def train_epoch(self, epoch: int, progress_manager=None) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_onset_loss = 0.0
        total_frame_loss = 0.0
        total_onset_acc = 0.0
        total_frame_acc = 0.0
        processed_batches = 0
        
        # Initialize progress tracking if provided
        if progress_manager:
            progress_manager.start_epoch(epoch, len(self.train_loader))
        
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
                    
                    # Process batch with mixed precision and gradient accumulation
                    if self.scaler and self.mixed_precision:
                        # Mixed precision forward pass
                        with autocast():
                            loss, metrics = self._process_batch(batch)
                        
                        # Scale loss for gradient accumulation
                        scaled_loss = loss / self.gradient_accumulation_steps
                        
                        # Backward pass with gradient scaling
                        self.scaler.scale(scaled_loss).backward()
                        
                        # Update weights every gradient_accumulation_steps
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            # Unscale gradients and clip
                            self.scaler.unscale_(self.optimizer)
                            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            
                            # Optimizer step with scaling
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        # Standard precision training
                        loss, metrics = self._process_batch(batch)
                        
                        # Scale loss for gradient accumulation
                        scaled_loss = loss / self.gradient_accumulation_steps
                        scaled_loss.backward()
                        
                        # Update weights every gradient_accumulation_steps
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    
                    # Skip if loss is NaN or inf
                    if not torch.isfinite(loss):
                        logger.error(f'Invalid loss value: {loss}')
                        continue
                    
                    # Update metrics
                    total_loss += metrics['loss']
                    total_onset_loss += metrics['onset_loss']
                    total_frame_loss += metrics['frame_loss']
                    total_onset_acc += metrics['onset_acc']
                    total_frame_acc += metrics['frame_acc']
                    processed_batches += 1
                    
                    # Update progress tracking
                    if progress_manager:
                        progress_manager.update_batch(
                            advance=1,
                            description=f"[green]Epoch {epoch} - Loss: {loss.item():.4f}, O_Acc: {metrics['onset_acc']:.1%}, F_Acc: {metrics['frame_acc']:.1%}"
                        )
                    
                    # Update total batch counter
                    self.total_batches_processed += 1
                    
                except Exception as e:
                    logger.error(f'Error in batch {batch_idx}: {str(e)}')
                    if batch is not None:
                        logger.debug(f'Batch stats: { {k: v.shape if hasattr(v, "shape") else str(v)[:50] + "..." for k, v in batch.items()} }')
                    continue
            
            if processed_batches == 0:
                logger.warning('No valid batches in training epoch')
                return {'loss': float('inf'), 'onset_loss': float('inf'), 'frame_loss': float('inf'), 'onset_acc': 0.0, 'frame_acc': 0.0}
        
            avg_metrics = {
                'loss': total_loss / processed_batches,
                'onset_loss': total_onset_loss / processed_batches,
                'frame_loss': total_frame_loss / processed_batches,
                'onset_acc': total_onset_acc / processed_batches,
                'frame_acc': total_frame_acc / processed_batches
            }
        
            logger.debug(
                f'Training - Processed {processed_batches} batches - '
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

    def validate(self, epoch: int = None, progress_manager=None) -> Dict[str, float]:
        """Run validation on the validation set."""
        self.model.eval()
        total_metrics = {'loss': 0.0, 'onset_loss': 0.0, 'frame_loss': 0.0, 'onset_acc': 0.0, 'frame_acc': 0.0}
        processed_batches = 0
        
        # Initialize validation progress tracking if provided
        if progress_manager:
            progress_manager.start_validation(len(self.val_loader))
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
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
                        onset_loss = F.binary_cross_entropy_with_logits(onset_logits, onset_labels, reduction='mean')
                        frame_loss = F.binary_cross_entropy_with_logits(frame_logits, frame_labels, reduction='mean')
                        loss = onset_loss + frame_loss
        
                        # Accuracy
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
        
                        # Accumulate metrics
                        for key, value in metrics.items():
                            total_metrics[key] += value
                        processed_batches += 1
                        
                        # Update validation progress
                        if progress_manager:
                            progress_manager.update_validation(
                                advance=1,
                                description=f"[yellow]Validation - Loss: {metrics['loss']:.4f}"
                            )
        
                    except Exception as e:
                        logger.error(f'Error processing validation batch {batch_idx}: {str(e)}')
                        if epoch is not None:
                            logger.debug(f'Batch stats: { {k: v.shape if hasattr(v, "shape") else str(v)[:50]+"..." for k, v in batch.items()} }')
                            continue
                        
                if processed_batches == 0:
                    logger.warning('No valid batches in validation')
                    return total_metrics

                avg_metrics = {k: v / processed_batches for k, v in total_metrics.items()}
                logger.debug(
                    f'Validation - Processed {processed_batches} batches - '
                    f'Loss: {avg_metrics["loss"]:.4f}, '
                    f'Onset Loss: {avg_metrics["onset_loss"]:.4f}, '
                    f'Frame Loss: {avg_metrics["frame_loss"]:.4f}, '
                    f'Onset Acc: {avg_metrics["onset_acc"]:.2%}, '
                    f'Frame Acc: {avg_metrics["frame_acc"]:.2%}'
                )
                return avg_metrics

        except Exception as e:
            logger.error(f'Unexpected error in validation: {str(e)}')
            logger.debug('Validation error details:', exc_info=True)
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
        logger.debug(f'Saved checkpoint to {checkpoint_path}')
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pt')
            torch.save(self.model.state_dict(), best_path)
            logger.success(f'New best model saved to {best_path}')

    def _generate_training_summary(self, num_epochs: int, best_val_loss: float, 
                                 final_train_metrics: Dict[str, float], 
                                 final_val_metrics: Dict[str, float],
                                 interrupted: bool = False) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        training_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        # Format training time
        hours, remainder = divmod(int(training_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        
        # Calculate model size
        output_dir = Path(self.output_dir)
        model_size = "N/A"
        checkpoint_count = 0
        best_model_path = "N/A"
        
        if output_dir.exists():
            try:
                # Count checkpoints
                checkpoint_files = list(output_dir.glob("checkpoint_epoch_*.pt"))
                checkpoint_count = len(checkpoint_files)
                
                # Get best model info
                best_model_file = output_dir / "best_model.pt"
                if best_model_file.exists():
                    best_model_path = str(best_model_file)
                    model_size_bytes = best_model_file.stat().st_size
                    if model_size_bytes > 1024 * 1024:  # MB
                        model_size = f"{model_size_bytes / (1024 * 1024):.1f} MB"
                    else:  # KB
                        model_size = f"{model_size_bytes / 1024:.1f} KB"
            except Exception:
                pass
        
        # Generate recommendations
        recommendations = []
        
        if interrupted:
            recommendations.append("Training was interrupted - consider resuming from the last checkpoint")
        
        if best_val_loss == float('inf'):
            recommendations.append("No valid validation loss recorded - check your dataset and model")
        elif len(self.val_losses) > 1:
            # Check for overfitting
            recent_val_losses = self.val_losses[-3:] if len(self.val_losses) >= 3 else self.val_losses
            if len(recent_val_losses) > 1 and all(recent_val_losses[i] >= recent_val_losses[i-1] for i in range(1, len(recent_val_losses))):
                recommendations.append("Validation loss is increasing - consider early stopping or reducing learning rate")
            
            # Check for underfitting
            final_train_loss = final_train_metrics.get('loss', float('inf'))
            if final_train_loss > 1.0:
                recommendations.append("High training loss - consider training for more epochs or adjusting hyperparameters")
        
        # Check accuracy
        final_val_acc = (final_val_metrics.get('onset_acc', 0) + final_val_metrics.get('frame_acc', 0)) / 2
        if final_val_acc < 0.5:
            recommendations.append("Low validation accuracy - consider adjusting model architecture or hyperparameters")
        elif final_val_acc > 0.9:
            recommendations.append("Excellent validation accuracy achieved!")
        
        if not recommendations:
            recommendations.append("Training completed successfully - model is ready for inference")
        
        # Collect GPU and system information
        gpu_info = get_gpu_info()
        system_info = get_system_info()
        
        # Create the summary dictionary
        summary = {
            'total_epochs': len(self.epoch_losses) if interrupted else num_epochs,
            'training_time': time_str,
            'total_batches': self.total_batches_processed,
            'device': self.device,
            'final_metrics': {
                'train_loss': final_train_metrics.get('loss', 0),
                'val_loss': final_val_metrics.get('loss', 0),
                'best_val_loss': best_val_loss,
                'train_acc': (final_train_metrics.get('onset_acc', 0) + final_train_metrics.get('frame_acc', 0)) / 2,
                'val_acc': final_val_acc
            },
            'gpu_info': gpu_info,
            'system_info': system_info,
            'output_info': {
                'output_dir': str(output_dir),
                'best_model_path': best_model_path,
                'model_size': model_size,
                'checkpoint_count': checkpoint_count
            },
            'dataset_info': {
                'dataset_path': str(self.json_path),  # Convert to string for Rich rendering
                'train_samples': len(self.train_loader.dataset) if self.train_loader else 'N/A',
                'val_samples': len(self.val_loader.dataset) if self.val_loader else 'N/A',
                'batch_size': self.batch_size
            },
            'recommendations': recommendations,
            'interrupted': interrupted
        }
        
        # Save summary to log file in output directory
        try:
            log_file_path = save_summary_to_file(summary, output_dir)
            summary['log_file'] = log_file_path
            logger.info(f"Training summary saved to: {log_file_path}")
        except Exception as e:
            logger.warning(f"Failed to save training summary to file: {str(e)}")
            summary['log_file'] = 'N/A'
        
        return summary

    def run(self, num_epochs: Optional[int] = None) -> None:
        """Run training loop"""
        if num_epochs is None:
            num_epochs = self.num_epochs
        
        best_val_loss = float('inf')
        self.training_start_time = time.time()
        
        # Use the unified progress tracking system
        with training_progress(logger, num_epochs) as progress_manager:
            try:
                for epoch in range(1, num_epochs + 1):
                    logger.progress_section(f'Epoch {epoch}/{num_epochs}', 'Training neural network...')
                    
                    # Train for one epoch
                    train_metrics = self.train_epoch(epoch, progress_manager)
                    
                    # Validate
                    val_metrics = self.validate(epoch, progress_manager)
                    
                    # Skip if validation failed
                    if not val_metrics:
                        logger.warning(f'Skipping checkpoint save due to validation failure in epoch {epoch}')
                        continue
                    
                    # Record metrics for summary
                    self.epoch_losses.append(train_metrics.get('loss', float('inf')))
                    self.val_losses.append(val_metrics.get('loss', float('inf')))
                    
                    # Save checkpoint
                    self._save_checkpoint(epoch)
                    
                    # Save best model
                    current_val_loss = val_metrics.get('loss', float('inf'))
                    is_best = current_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = current_val_loss
                        self.best_epoch = epoch
                        self._save_checkpoint(epoch, is_best=True)
                    
                    # Update progress with epoch completion
                    progress_manager.finish_epoch(
                        epoch, 
                        train_metrics.get('loss', float('nan')), 
                        current_val_loss, 
                        is_best
                    )
                
                # Generate and show training summary
                training_summary = self._generate_training_summary(
                    num_epochs, best_val_loss, train_metrics, val_metrics
                )
                progress_manager.finish_training(training_summary)
                    
            except KeyboardInterrupt:
                logger.warning('Training interrupted by user')
                # Still show summary for partial training
                if self.epoch_losses:
                    training_summary = self._generate_training_summary(
                        len(self.epoch_losses), best_val_loss, 
                        {'loss': self.epoch_losses[-1] if self.epoch_losses else float('inf')},
                        {'loss': self.val_losses[-1] if self.val_losses else float('inf')},
                        interrupted=True
                    )
                    progress_manager.finish_training(training_summary)
                raise
            except Exception as e:
                logger.error(f'Error during training: {str(e)}')
                logger.debug('Training error details:', exc_info=True)
                raise
