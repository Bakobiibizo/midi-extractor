# MIDI Generator

A production-grade PyTorch pipeline for automatic music transcription and MIDI generation from audio datasets. Features a unified Rich-based CLI, comprehensive training system, and GPU monitoring.

## Features

- 🎵 **Dataset Processing**: Index, filter, and visualize Slakh dataset stems
- 🔊 **Audio Processing**: Convert audio to mel spectrograms with configurable parameters
- 🎹 **MIDI Processing**: Extract onset, frame, and velocity targets from MIDI files
- 🚀 **Production Training**: Complete training pipeline with progress tracking
- 🧠 **Model Architecture**: CNN + LSTM/Transformer for music transcription
- 🎯 **Multi-Target Learning**: Simultaneous onset, frame, and velocity prediction
- 📊 **Rich CLI**: Beautiful command-line interface with progress bars and logging
- 🎮 **GPU Monitoring**: Real-time GPU usage and system resource tracking
- 📄 **Training Summaries**: Comprehensive training reports with recommendations

## Installation

```bash
# Clone the repository
git clone https://github.com/Bakobiibizo/midi-extractor.git
cd midi-extractor

# Install dependencies using uv
uv sync
```

## Quick Start

### 1. Dataset Indexing

Create an index of your Slakh dataset:

```bash
# Index a dataset directory
uv run src/cli.py index datasets/babyslakh_16k

# With custom output file
uv run src/cli.py index datasets/babyslakh_16k --output-file my_dataset.json
```

This will:
- Index all stems matching target instruments (piano, bass, synth)
- Generate a comprehensive dataset index file
- Display progress with Rich formatting

### 2. Dataset Filtering

Filter out silent audio and empty MIDI files:

```bash
# Filter dataset with default threshold (-40dB)
uv run src/cli.py filter dataset_index.json

# Custom threshold and output
uv run src/cli.py filter dataset_index.json --threshold -35 --output filtered_data.json
```

### 3. Dataset Visualization

Visualize samples from your dataset:

```bash
# Visualize random samples
uv run src/cli.py visualize dataset_index.json

# Specific number of samples with custom output
uv run src/cli.py visualize dataset_index.json --num-samples 10 --output-dir visualizations
```

### 4. Model Training

Train your model with comprehensive monitoring:

```bash
# Quick validation run (recommended first)
uv run src/cli.py train dataset_index.json --epochs 5 --batch-size 16 --dataset babyslakh

# Production training
uv run src/cli.py train dataset_index.json --epochs 100 --batch-size 32 --lr 3e-5 --dataset slakh

# GPU-optimized training (for RTX A6000 or similar)
uv run src/cli.py train dataset_index.json --epochs 100 --batch-size 64 --clip-seconds 20.0 --lr 2e-5
```

**Training Features:**
- 📊 Real-time progress bars with Rich formatting
- 🎮 GPU memory usage monitoring
- 🧠 System resource tracking
- 📄 Comprehensive training summaries saved to log files
- 💡 Intelligent recommendations (overfitting detection, etc.)
- 🏆 Automatic best model checkpointing

## CLI Commands

### Available Commands

| Command | Description | Example |
|---------|-------------|----------|
| `index` | Index dataset directory | `uv run src/cli.py index datasets/babyslakh_16k` |
| `filter` | Filter dataset by audio/MIDI quality | `uv run src/cli.py filter dataset.json --threshold -35` |
| `visualize` | Create sample visualizations | `uv run src/cli.py visualize dataset.json --num-samples 5` |
| `train` | Train the model | `uv run src/cli.py train dataset.json --epochs 50` |

### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--batch-size` | Samples per batch | 8 | 16-64 (depending on GPU) |
| `--clip-seconds` | Audio clip length | 10.0 | 10.0-20.0 seconds |
| `--epochs` | Training epochs | 10 | 50-100 |
| `--lr` | Learning rate | 1e-4 | 1e-4 to 5e-5 |
| `--device` | Training device | cuda | cuda (if available) |
| `--dataset` | Dataset type | babyslakh | babyslakh or slakh |
| `--output-dir` | Checkpoint directory | checkpoints | Custom path |
| `--val-split` | Validation fraction | 0.1 | 0.1-0.2 |

### Debug Mode

Enable detailed logging and error information:

```bash
uv run src/cli.py train dataset.json --debug
```

## Project Structure

```
src/
├── datasets/
│   ├── stem_indexer.py          # Dataset indexing and filtering
│   ├── slakh_stem_dataset.py    # PyTorch Dataset implementation
│   ├── baby_slakh_stem_dataset.py # BabySlakh Dataset implementation
│   └── visualize_sample.py      # Visualization tools
├── models/
│   └── interpretable_transcription.py # Model architecture
├── train/
│   ├── trainer.py               # Training loop and logic
│   └── run_trainer.py           # Training orchestration
├── utils/
│   ├── logging.py               # Unified Rich logging system
│   └── exceptions.py            # Custom exceptions
└── cli.py                       # Rich-based command-line interface
```

## Dataset Format

The pipeline expects Slakh-format datasets with:

```
Track00001/
├── metadata.yaml     # Instrument metadata
├── stems/           # Individual instrument audio files
│   ├── S01.wav
│   └── S02.wav
└── MIDI/            # Corresponding MIDI files
    ├── S01.mid
    └── S02.mid
```

## Configuration

### Audio Processing
- **Sample Rate**: 22,050 Hz
- **Mel Bins**: 128
- **FFT Size**: 1024
- **Hop Length**: 256 samples (~86.1 Hz frame rate)

### Target Instruments
- **Piano**: GM programs 0-7 (Acoustic & Electric Pianos)
- **Bass**: GM programs 32-39 (Bass instruments)
- **Synth**: GM programs 80-87 (Synth Leads)

### Filtering Thresholds
- **Audio Silence**: -35 dB (configurable)
- **MIDI Empty**: No notes in any instrument

## Training Output

### Training Summary

After training completes, you'll get a comprehensive summary including:

- 🎯 **Training Metrics**: Final losses, accuracies, best model performance
- ⏱️ **Timing Information**: Total training time, epochs completed
- 🎮 **GPU Usage**: Memory utilization, GPU name and specs
- 🧠 **System Resources**: CPU and memory usage during training
- 📁 **Output Files**: Model checkpoints, log files, directory info
- 💡 **Recommendations**: Overfitting detection, training suggestions

### Log Files

Training summaries are automatically saved as Rich-formatted log files:

```
checkpoints/your_run/
├── best_model.pt                    # Best performing model
├── checkpoint_epoch_N.pt            # Regular checkpoints
└── training_summary_YYYYMMDD_HHMMSS.log  # Detailed training log
```

## Development

### Using the Python API

```python
from src.datasets.slakh_stem_dataset import SlakhStemDataset
from src.train.trainer import Trainer
from src.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Create dataset
dataset = SlakhStemDataset("dataset.json", clip_seconds=10.0)

# Initialize trainer
trainer = Trainer(
    json_path="dataset.json",
    batch_size=16,
    num_epochs=50,
    device="cuda"
)

# Start training
trainer.run()
```

### Visualizing Samples

```bash
# Use the CLI for easy visualization
uv run src/cli.py visualize dataset.json --num-samples 5 --output-dir viz
```

## Model Architecture

The model follows a proven architecture for automatic music transcription:

```
[Mel Spectrogram: B × 128 × T]
           ↓
    [CNN Feature Extraction]
           ↓
  [BiLSTM/Transformer Encoder]
           ↓
    [Multi-Head Outputs]
    ├── Onset Head (Binary)
    ├── Frame Head (Binary)
    └── Velocity Head (Regression)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on the [BabySlakh Dataset](https://zenodo.org/records/4603870)
- Inspired by my frustration at not having a good midi extractor
- Uses PyTorch for deep learning components