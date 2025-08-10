"""
Shared audio preprocessing utilities for training and inference.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import torch
import torchaudio


def load_waveform(path: Path | str, target_sr: int) -> torch.Tensor:
    """Load audio as mono float32 tensor at target sample rate.
    Returns tensor shape [1, T].
    """
    wav, sr = torchaudio.load(str(path))  # [C, T]
    # Convert to mono by mean if multi-channel
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    # Ensure contiguous float32
    return wav.to(torch.float32)


def build_mel_transforms(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: float,
    top_db: float,
) -> Tuple[torchaudio.transforms.MelSpectrogram, torchaudio.transforms.AmplitudeToDB]:
    """Factory returning MelSpectrogram and AmplitudeToDB matching training."""
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        center=True,
        power=2.0,
        norm=None,
        mel_scale="htk",
    )
    amp2db = torchaudio.transforms.AmplitudeToDB(top_db=top_db)
    return mel, amp2db
