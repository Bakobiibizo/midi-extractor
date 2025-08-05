# MIDI Extractor

A production-grade PyTorch pipeline for extracting and processing MIDI data from audio datasets, with support for automatic music transcription using deep learning.

## Features

- 🎵 **Dataset Indexing**: Automatically index and filter Slakh dataset stems
- 🔊 **Audio Processing**: Convert audio to mel spectrograms with configurable parameters
- 🎹 **MIDI Processing**: Extract onset, frame, and velocity targets from MIDI files
- 🚀 **Production DataLoader**: Batching, padding, attention masks, and metadata passthrough
- 🧠 **Model Architecture**: CNN + LSTM/Transformer for music transcription
- 🎯 **Multi-Target Learning**: Simultaneous onset, frame, and velocity prediction

## Installation

```bash
# Clone the repository
git clone https://github.com/Bakobiibizo/midi-extractor.git
cd midi-extractor

# Install dependencies using uv
uv sync
```

## Quick Start

### 1. Index and Filter Dataset

```bash
# Process Slakh dataset and create filtered index
uv run src/cli.py --dataset datasets/babyslakh_16k
```

This will:
- Index all stems matching target instruments (piano, bass, synth)
- Filter out silent audio and empty MIDI files
- Generate `dataset.json` and `filtered_dataset.json`
- Create sample visualizations

### 2. Use the DataLoader

```python
from src.datasets.dataloader import create_dataloader, create_train_val_dataloaders

# Simple dataloader
dataloader = create_dataloader(
    "datasets/babyslakh_16k/filtered_dataset.json",
    batch_size=8,
    clip_seconds=10.0
)

# Train/validation split
train_loader, val_loader = create_train_val_dataloaders(
    "datasets/babyslakh_16k/filtered_dataset.json",
    train_split=0.8,
    batch_size=8
)

# Iterate through batches
for batch in train_loader:
    specs = batch["spec"]        # [B, n_mels, T]
    onsets = batch["onset"]      # [B, T, 128]
    frames = batch["frame"]      # [B, T, 128]
    velocities = batch["velocity"] # [B, T, 128]
    masks = batch["mask"]        # [B, T] - attention masks
    metadata = batch["meta"]     # List of track info
```

### 3. Model Training (Coming Soon)

```python
from src.model.InterpretableTranscription import InterpretableTranscription
from src.model.InterpretableTranscription.loss import loss_fn

# Initialize model
model = InterpretableTranscription()

# Training loop
for batch in train_loader:
    outputs = model(batch["spec"])
    loss, loss_dict = loss_fn(outputs, {
        "onset": batch["onset"],
        "frame": batch["frame"],
        "velocity": batch["velocity"]
    })
```

## Project Structure

```
src/
├── datasets/
│   ├── stem_indexer.py          # Dataset indexing and filtering
│   ├── slakh_stem_dataset.py    # PyTorch Dataset implementation
│   ├── dataloader.py            # Production DataLoader utilities
│   └── visualize_sample.py      # Visualization tools
├── model/
│   └── InterpretableTranscription/
│       ├── interpretable_transcription.py  # Model architecture
│       └── loss.py              # Loss functions
└── cli.py                       # Command-line interface
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

## Development

### Testing the DataLoader

```bash
# Test dataloader functionality
uv run -m src.datasets.dataloader
```

### Visualizing Samples

```python
from src.datasets.slakh_stem_dataset import SlakhStemDataset
from src.datasets.visualize_sample import visualize_sample

dataset = SlakhStemDataset("datasets/babyslakh_16k/filtered_dataset.json")
sample = dataset[0]
visualize_sample(sample, output_path="sample.png")
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