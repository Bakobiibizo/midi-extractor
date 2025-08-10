#!/usr/bin/env python
"""
Visualize a single sample's spectrogram and (optional) onset targets.

Supports inputs:
- .pt  (torch.save'd dict with keys like 'spec', 'onset')
- .npz (numpy savez with arrays 'spec', 'onset' if available)
- .npy (numpy array for spec only; creates empty onset)

Usage:
  uv run examples/visualize_single.py path/to/sample.pt --output examples/sample.png --max-len 1000
"""
import argparse
from pathlib import Path
import numpy as np

try:
    import torch
except Exception:
    torch = None

from src.utils.visualize_sample import visualize_sample


def load_sample(path: Path):
    ext = path.suffix.lower()
    if ext == '.pt':
        if torch is None:
            raise RuntimeError("PyTorch not available to load .pt file")
        obj = torch.load(path)
        if not isinstance(obj, dict) or 'spec' not in obj:
            raise ValueError(".pt file must contain a dict with key 'spec'")
        # Ensure numpy arrays
        sample = {}
        for k, v in obj.items():
            if hasattr(v, 'numpy'):
                try:
                    sample[k] = v.numpy()
                except Exception:
                    sample[k] = np.array(v)
            else:
                sample[k] = np.array(v)
        return sample
    elif ext == '.npz':
        data = np.load(path)
        sample = {k: data[k] for k in data.files}
        return sample
    elif ext == '.npy':
        spec = np.load(path)
        # Build minimal sample
        time_len = spec.shape[1] if spec.ndim == 2 else spec.shape[-1]
        onset = np.zeros((time_len, 128), dtype=np.float32)
        return {"spec": spec, "onset": onset}
    else:
        raise ValueError(f"Unsupported input extension: {ext}")


def main():
    ap = argparse.ArgumentParser(description="Visualize a single saved sample")
    ap.add_argument("input", help="Path to sample (.pt, .npz, or .npy for spec)")
    ap.add_argument("--output", default="examples/sample.png", help="Output image path")
    ap.add_argument("--max-len", type=int, default=1000, help="Max time frames to visualize")
    args = ap.parse_args()

    sample = load_sample(Path(args.input))

    # Ensure expected keys
    if "spec" not in sample:
        raise ValueError("Input must provide a 'spec' array")
    if "onset" not in sample:
        # create empty onset if missing
        t = sample["spec"].shape[1]
        sample["onset"] = np.zeros((t, 128), dtype=np.float32)

    visualize_sample(sample, output_path=args.output, max_len=args.max_len)
    print(f"Saved visualization to: {args.output}")


if __name__ == "__main__":
    main()
