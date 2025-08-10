"""
Unified logging system for the MIDI Generator application.

This module provides a consistent logging interface across the entire codebase,
matching the CLI's Rich-based logging style with enhanced user experience.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
import logging

# Try to import GPU monitoring utilities
try:
    import torch
    import psutil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False

# Global console instance
console = Console()

# Global progress instance for training
_global_progress: Optional[Progress] = None

class MIDIGeneratorLogger:
    """Enhanced logger with Rich formatting and progress tracking capabilities."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.log_file = log_file or 'midi_generator.log'
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Set up the logger with Rich handler and file handler."""
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Rich handler for console output
        rich_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            show_time=False
        )
        rich_handler.setFormatter(logging.Formatter('%(message)s'))
        
        # File handler for persistent logging
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        self.logger.addHandler(rich_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def set_level(self, level: Union[int, str]) -> None:
        """Set the logging level."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with Rich formatting."""
        self.logger.info(message, **kwargs)
    
    def success(self, message: str, **kwargs) -> None:
        """Log success message with green checkmark."""
        self.logger.info(f"[bold green]✅ {message}[/bold green]", **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with yellow warning icon."""
        self.logger.warning(f"[bold yellow]⚠️  {message}[/bold yellow]", **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with red X icon."""
        self.logger.error(f"[bold red]❌ {message}[/bold red]", **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with red explosion icon."""
        self.logger.critical(f"[bold red]💥 {message}[/bold red]", **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(f"[dim]{message}[/dim]", **kwargs)
    
    def section(self, title: str, message: Optional[str] = None) -> None:
        """Log a section header with formatting."""
        panel_content = title
        if message:
            panel_content = f"{title}\n{message}"
        
        console.print(Panel.fit(panel_content, border_style="cyan", title="[bold cyan]Info[/bold cyan]"))
    
    def progress_section(self, title: str, description: Optional[str] = None) -> None:
        """Log a progress section header."""
        content = f"[bold cyan]{title}[/bold cyan]"
        if description:
            content += f"\n[dim]{description}[/dim]"
        console.print(content)
    
    def exception(self, message: str, exc_info: bool = True) -> None:
        """Log exception with Rich traceback."""
        self.logger.exception(f"[bold red]💥 {message}[/bold red]", exc_info=exc_info)


class TrainingProgressManager:
    """Manages training progress bars with Rich formatting."""
    
    def __init__(self, logger: MIDIGeneratorLogger):
        self.logger = logger
        self.progress: Optional[Progress] = None
        self.epoch_task = None
        self.batch_task = None
        self.val_task = None
    
    def start_training(self, total_epochs: int) -> None:
        """Start the training progress display."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False
        )
        
        self.progress.start()
        self.epoch_task = self.progress.add_task(
            "[bold blue]Training Progress", 
            total=total_epochs
        )
        
        global _global_progress
        _global_progress = self.progress
    
    def start_epoch(self, epoch: int, total_batches: int) -> None:
        """Start progress tracking for an epoch."""
        if self.progress and self.batch_task is not None:
            self.progress.remove_task(self.batch_task)
        
        if self.progress:
            self.batch_task = self.progress.add_task(
                f"[green]Epoch {epoch} - Training",
                total=total_batches
            )
    
    def update_batch(self, advance: int = 1, **kwargs) -> None:
        """Update batch progress."""
        if self.progress and self.batch_task is not None:
            self.progress.update(self.batch_task, advance=advance, **kwargs)
    
    def start_validation(self, total_batches: int) -> None:
        """Start validation progress tracking."""
        if self.progress and self.val_task is not None:
            self.progress.remove_task(self.val_task)
        
        if self.progress:
            self.val_task = self.progress.add_task(
                "[yellow]Validation",
                total=total_batches
            )
    
    def update_validation(self, advance: int = 1, **kwargs) -> None:
        """Update validation progress."""
        if self.progress and self.val_task is not None:
            self.progress.update(self.val_task, advance=advance, **kwargs)
    
    def finish_epoch(self, epoch: int, train_loss: float, val_loss: float, is_best: bool = False) -> None:
        """Finish epoch and update progress."""
        if self.progress:
            # Clean up batch and validation tasks
            if self.batch_task is not None:
                self.progress.remove_task(self.batch_task)
                self.batch_task = None
            if self.val_task is not None:
                self.progress.remove_task(self.val_task)
                self.val_task = None
            
            # Update epoch progress
            if self.epoch_task is not None:
                status = "🏆 Best!" if is_best else "✅ Done"
                description = f"Epoch {epoch} - Train: {train_loss:.4f}, Val: {val_loss:.4f} {status}"
                self.progress.update(
                    self.epoch_task, 
                    advance=1,
                    description=f"[bold blue]{description}"
                )
    
    def finish_training(self, training_summary: Optional[Dict[str, Any]] = None) -> None:
        """Finish training and clean up progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None
        
        global _global_progress
        _global_progress = None
        
        # Show training summary if provided
        if training_summary:
            self.show_training_summary(training_summary)
        else:
            # Show basic completion message
            self.logger.success("Training completed successfully!")
    
    def show_training_summary(self, summary: Dict[str, Any]) -> None:
        """Display a comprehensive training summary."""
        
        # Create summary table
        summary_table = Table(show_header=False, box=None, padding=(0, 2))
        summary_table.add_column("Metric", style="bold cyan", width=25)
        summary_table.add_column("Value", style="white")
        
        # Training overview
        summary_table.add_row("🎯 Total Epochs", str(summary.get('total_epochs', 'N/A')))
        summary_table.add_row("⏱️  Training Time", summary.get('training_time', 'N/A'))
        summary_table.add_row("📊 Total Batches", str(summary.get('total_batches', 'N/A')))
        summary_table.add_row("🔥 Device Used", summary.get('device', 'N/A'))
        
        # Add separator
        summary_table.add_row("", "")
        
        # Performance metrics
        final_metrics = summary.get('final_metrics', {})
        if final_metrics:
            summary_table.add_row("📈 Final Train Loss", f"{final_metrics.get('train_loss', 0):.4f}")
            summary_table.add_row("📉 Final Val Loss", f"{final_metrics.get('val_loss', 0):.4f}")
            summary_table.add_row("🎯 Best Val Loss", f"{final_metrics.get('best_val_loss', 0):.4f}")
            summary_table.add_row("🎪 Train Accuracy", f"{final_metrics.get('train_acc', 0):.1%}")
            summary_table.add_row("✅ Val Accuracy", f"{final_metrics.get('val_acc', 0):.1%}")
        
        # Add separator
        summary_table.add_row("", "")
        
        # GPU Information
        gpu_info = summary.get('gpu_info', {})
        if gpu_info and gpu_info.get('available'):
            summary_table.add_row("🎮 GPU Name", gpu_info.get('name', 'N/A'))
            summary_table.add_row("🔥 GPU Memory Used", gpu_info.get('memory_used', 'N/A'))
            summary_table.add_row("💾 GPU Memory Total", gpu_info.get('memory_total', 'N/A'))
            summary_table.add_row("📊 GPU Usage", gpu_info.get('memory_percent', 'N/A'))
            
            # Add separator
            summary_table.add_row("", "")
        
        # System Information
        system_info = summary.get('system_info', {})
        if system_info:
            summary_table.add_row("⚡ CPU Usage", system_info.get('cpu_percent', 'N/A'))
            summary_table.add_row("🧠 System Memory", f"{system_info.get('memory_used', 'N/A')} / {system_info.get('memory_total', 'N/A')}")
            
            # Add separator
            summary_table.add_row("", "")
        
        # File information
        output_info = summary.get('output_info', {})
        if output_info:
            # Convert Path objects to strings for Rich rendering
            output_dir = output_info.get('output_dir', 'N/A')
            if hasattr(output_dir, '__fspath__'):  # Check if it's a Path-like object
                output_dir = str(output_dir)
            
            best_model_path = output_info.get('best_model_path', 'N/A')
            if hasattr(best_model_path, '__fspath__'):
                best_model_path = str(best_model_path)
            
            summary_table.add_row("📁 Output Directory", output_dir)
            summary_table.add_row("💾 Best Model", best_model_path)
            summary_table.add_row("📦 Model Size", output_info.get('model_size', 'N/A'))
            summary_table.add_row("🔢 Total Checkpoints", str(output_info.get('checkpoint_count', 'N/A')))
            
            # Show log file location
            log_file = summary.get('log_file', 'N/A')
            if log_file != 'N/A':
                if hasattr(log_file, '__fspath__'):
                    log_file = str(log_file)
                summary_table.add_row("📄 Summary Log", log_file)
        
        # Dataset information
        dataset_info = summary.get('dataset_info', {})
        if dataset_info:
            summary_table.add_row("", "")
            summary_table.add_row("📚 Dataset", dataset_info.get('dataset_path', 'N/A'))
            summary_table.add_row("🔢 Train Samples", str(dataset_info.get('train_samples', 'N/A')))
            summary_table.add_row("🔍 Val Samples", str(dataset_info.get('val_samples', 'N/A')))
            summary_table.add_row("⚡ Batch Size", str(dataset_info.get('batch_size', 'N/A')))
        
        # Create the main panel
        console.print()
        console.print(Panel(
            summary_table,
            title="[bold green]🎵 Training Summary[/bold green]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Show recommendations or next steps
        recommendations = summary.get('recommendations', [])
        if recommendations:
            console.print()
            rec_text = "\n".join([f"• {rec}" for rec in recommendations])
            console.print(Panel(
                rec_text,
                title="[bold blue]💡 Recommendations[/bold blue]",
                border_style="blue"
            ))
        
        console.print()
        self.logger.success("Training completed successfully!")


@contextmanager
def training_progress(logger: MIDIGeneratorLogger, total_epochs: int):
    """Context manager for training progress tracking."""
    progress_manager = TrainingProgressManager(logger)
    try:
        progress_manager.start_training(total_epochs)
        yield progress_manager
    finally:
        progress_manager.finish_training()


def get_logger(name: str, log_file: Optional[str] = None) -> MIDIGeneratorLogger:
    """Get a configured logger instance."""
    return MIDIGeneratorLogger(name, log_file)


def set_debug_mode(enabled: bool = True) -> None:
    """Enable or disable debug mode for all loggers."""
    level = logging.DEBUG if enabled else logging.INFO
    
    # Update all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        if hasattr(logger, 'setLevel'):
            logger.setLevel(level)


def print_startup_banner(app_name: str = "MIDI Generator", version: str = "1.0.0") -> None:
    """Print a startup banner."""
    banner_text = f"🎵 {app_name} v{version}"
    console.print(Panel.fit(banner_text, border_style="cyan", title="[bold cyan]Welcome[/bold cyan]"))
    console.print()


def print_section_header(title: str, description: Optional[str] = None) -> None:
    """Print a section header."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if description:
        content += f"\n[dim]{description}[/dim]"
    console.print(Panel(content, border_style="blue"))
    console.print()


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information and memory usage."""
    gpu_info = {
        'available': False,
        'name': 'N/A',
        'memory_used': 'N/A',
        'memory_total': 'N/A',
        'memory_percent': 'N/A',
        'utilization': 'N/A'
    }
    
    if not GPU_MONITORING_AVAILABLE:
        return gpu_info
    
    try:
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['name'] = torch.cuda.get_device_name(0)
            
            # Get memory info
            memory_used = torch.cuda.memory_allocated(0)
            memory_total = torch.cuda.max_memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            
            # Format memory usage
            memory_used_mb = memory_used / (1024 ** 2)
            memory_reserved_mb = memory_reserved / (1024 ** 2)
            
            gpu_info['memory_used'] = f"{memory_used_mb:.1f} MB"
            gpu_info['memory_reserved'] = f"{memory_reserved_mb:.1f} MB"
            
            # Try to get total GPU memory (this might not always work)
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_memory_mb = total_memory / (1024 ** 2)
                gpu_info['memory_total'] = f"{total_memory_mb:.1f} MB"
                gpu_info['memory_percent'] = f"{(memory_reserved / total_memory) * 100:.1f}%"
            except:
                gpu_info['memory_total'] = f"{memory_reserved_mb:.1f} MB (reserved)"
                gpu_info['memory_percent'] = "N/A"
                
    except Exception as e:
        # If there's any error, just return default values
        pass
    
    return gpu_info


def get_system_info() -> Dict[str, Any]:
    """Get system information including CPU and memory usage."""
    system_info = {
        'cpu_percent': 'N/A',
        'memory_percent': 'N/A',
        'memory_used': 'N/A',
        'memory_total': 'N/A'
    }
    
    if not GPU_MONITORING_AVAILABLE:
        return system_info
    
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        system_info['cpu_percent'] = f"{cpu_percent:.1f}%"
        
        # Memory usage
        memory = psutil.virtual_memory()
        system_info['memory_percent'] = f"{memory.percent:.1f}%"
        system_info['memory_used'] = f"{memory.used / (1024 ** 3):.1f} GB"
        system_info['memory_total'] = f"{memory.total / (1024 ** 3):.1f} GB"
        
    except Exception:
        pass
    
    return system_info


def save_summary_to_file(summary: Dict[str, Any], output_dir: Path) -> str:
    """Save training summary to a Rich-formatted log file."""
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_summary_{timestamp}.log"
    log_path = output_dir / log_filename
    
    # Create a file console for Rich formatting
    with open(log_path, 'w', encoding='utf-8') as f:
        file_console = Console(file=f, width=80, legacy_windows=False)
        
        # Write header
        file_console.print("=" * 80)
        file_console.print(f"MIDI Generator Training Summary")
        file_console.print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        file_console.print("=" * 80)
        file_console.print()
        
        # Training Overview
        file_console.print("[bold cyan]Training Overview[/bold cyan]")
        file_console.print(f"Total Epochs: {summary.get('total_epochs', 'N/A')}")
        file_console.print(f"Training Time: {summary.get('training_time', 'N/A')}")
        file_console.print(f"Total Batches: {summary.get('total_batches', 'N/A')}")
        file_console.print(f"Device Used: {summary.get('device', 'N/A')}")
        file_console.print()
        
        # Performance Metrics
        final_metrics = summary.get('final_metrics', {})
        if final_metrics:
            file_console.print("[bold cyan]Performance Metrics[/bold cyan]")
            file_console.print(f"Final Train Loss: {final_metrics.get('train_loss', 0):.4f}")
            file_console.print(f"Final Val Loss: {final_metrics.get('val_loss', 0):.4f}")
            file_console.print(f"Best Val Loss: {final_metrics.get('best_val_loss', 0):.4f}")
            file_console.print(f"Train Accuracy: {final_metrics.get('train_acc', 0):.1%}")
            file_console.print(f"Val Accuracy: {final_metrics.get('val_acc', 0):.1%}")
            file_console.print()
        
        # GPU Information
        gpu_info = summary.get('gpu_info', {})
        if gpu_info and gpu_info.get('available'):
            file_console.print("[bold cyan]GPU Information[/bold cyan]")
            file_console.print(f"GPU Name: {gpu_info.get('name', 'N/A')}")
            file_console.print(f"Memory Used: {gpu_info.get('memory_used', 'N/A')}")
            file_console.print(f"Memory Reserved: {gpu_info.get('memory_reserved', 'N/A')}")
            file_console.print(f"Memory Total: {gpu_info.get('memory_total', 'N/A')}")
            file_console.print(f"Memory Usage: {gpu_info.get('memory_percent', 'N/A')}")
            file_console.print()
        
        # System Information
        system_info = summary.get('system_info', {})
        if system_info:
            file_console.print("[bold cyan]System Information[/bold cyan]")
            file_console.print(f"CPU Usage: {system_info.get('cpu_percent', 'N/A')}")
            file_console.print(f"System Memory: {system_info.get('memory_used', 'N/A')} / {system_info.get('memory_total', 'N/A')} ({system_info.get('memory_percent', 'N/A')})")
            file_console.print()
        
        # Output Information
        output_info = summary.get('output_info', {})
        if output_info:
            file_console.print("[bold cyan]Output Information[/bold cyan]")
            file_console.print(f"Output Directory: {output_info.get('output_dir', 'N/A')}")
            file_console.print(f"Best Model: {output_info.get('best_model_path', 'N/A')}")
            file_console.print(f"Model Size: {output_info.get('model_size', 'N/A')}")
            file_console.print(f"Total Checkpoints: {output_info.get('checkpoint_count', 'N/A')}")
            file_console.print()
        
        # Dataset Information
        dataset_info = summary.get('dataset_info', {})
        if dataset_info:
            file_console.print("[bold cyan]Dataset Information[/bold cyan]")
            file_console.print(f"Dataset: {dataset_info.get('dataset_path', 'N/A')}")
            file_console.print(f"Train Samples: {dataset_info.get('train_samples', 'N/A')}")
            file_console.print(f"Val Samples: {dataset_info.get('val_samples', 'N/A')}")
            file_console.print(f"Batch Size: {dataset_info.get('batch_size', 'N/A')}")
            file_console.print()
        
        # Recommendations
        recommendations = summary.get('recommendations', [])
        if recommendations:
            file_console.print("[bold cyan]Recommendations[/bold cyan]")
            for rec in recommendations:
                file_console.print(f"• {rec}")
            file_console.print()
        
        # Footer
        file_console.print("=" * 80)
        if summary.get('interrupted', False):
            file_console.print("[yellow]Training was interrupted[/yellow]")
        else:
            file_console.print("[green]Training completed successfully[/green]")
        file_console.print("=" * 80)
    
    return str(log_path)
