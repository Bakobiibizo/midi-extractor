import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_sample(sample, output_path="datasets/babyslakh_16k/sample.png", max_len=1000):
    spec = sample["spec"].numpy()
    onset = sample["onset"].numpy()
    pitch_range = np.arange(128)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(spec[:, :max_len], aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Mel Spectrogram with Onsets")
    ax.set_ylabel("Mel bins")
    ax.set_xlabel("Frames")

    # overlay onsets
    on_x, on_y = np.where(onset[:max_len] > 0)
    N_MELS = 128
    ax.scatter(on_x, on_y * (N_MELS / 128), c="cyan", s=10, alpha=0.7, label="Onsets")

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved spectrogram to {output_path}")
    print(f"Track: {sample.get('track')}")
    print(f"Stem ID: {sample.get('stem_id')}")
    print(f"Program: {sample.get('program')}")
    print(f"WAV: {sample.get('wav_path')}")
    print(f"MIDI: {sample.get('midi_path')}")
    print("\n")
