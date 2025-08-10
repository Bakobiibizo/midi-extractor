#!/usr/bin/env python
"""
Visualize a single WAV file using the same preprocessing as training.

Computes MelSpectrogram + AmplitudeToDB with training-aligned parameters and
saves a PNG image.

Usage:
  uv run examples/visualize_wav.py examples/example.wav --output examples/example_mel.png --max-len 1000
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import sys
from pathlib import Path as _Path

# Robust import: allow running this script directly from examples/
try:
    from src.audio.preprocessing import load_waveform, build_mel_transforms
except ModuleNotFoundError:
    # Add project root to sys.path
    _ROOT = _Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.audio.preprocessing import load_waveform, build_mel_transforms

# Training-aligned defaults (kept in sync with training/inference)
DEFAULTS = {
    "sample_rate": 22050,
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 128,
    "f_min": 30.0,
    "f_max": 11000.0,
    "top_db": 80.0,
}


def compute_mel(path: Path, max_len: int) -> np.ndarray:
    wav = load_waveform(path, DEFAULTS["sample_rate"])  # [1, T]
    mel_t, amp2db = build_mel_transforms(
        sample_rate=DEFAULTS["sample_rate"],
        n_fft=DEFAULTS["n_fft"],
        hop_length=DEFAULTS["hop_length"],
        n_mels=DEFAULTS["n_mels"],
        f_min=DEFAULTS["f_min"],
        f_max=DEFAULTS["f_max"],
        top_db=DEFAULTS["top_db"],
    )

    with torch.no_grad():
        mel = mel_t(wav)  # [n_mels, T]
        mel_db = amp2db(mel)  # [n_mels, T]

    mel_db = mel_db.squeeze(0).cpu().numpy()  # [n_mels, T]

    # Optionally trim time dimension
    if mel_db.shape[1] > max_len:
        mel_db = mel_db[:, :max_len]
    return mel_db


def save_png(mel_db: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4), dpi=150)
    # Display with lower origin (low freq at bottom), similar to librosa
    vmin = np.percentile(mel_db, 5)
    vmax = np.percentile(mel_db, 95)
    plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    plt.colorbar(label='dB')
    plt.xlabel('Frames')
    plt.ylabel('Mel bins')
    plt.title('Mel Spectrogram (dB)')
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Visualize a single WAV as a mel spectrogram (training-aligned)")
    ap.add_argument("input", help="Path to WAV file")
    ap.add_argument("--output", default="examples/example_mel.png", help="Output PNG path")
    ap.add_argument("--max-len", type=int, default=1000, help="Max frames to visualize")
    args = ap.parse_args()

    mel_db = compute_mel(Path(args.input), args.max_len)
    save_png(mel_db, Path(args.output))
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
