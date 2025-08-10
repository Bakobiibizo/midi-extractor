import torch
import torchaudio
import pretty_midi
import numpy as np
import json
import yaml
from pathlib import Path
from torch.utils.data import Dataset

SR = 22050
N_MELS = 128
N_FFT = 1024
HOP = 256
FPS = SR / HOP

mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=30.0, f_max=11000.0
)
amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

def midi_to_frame_targets(pm: pretty_midi.PrettyMIDI, n_frames: int):
    onset = np.zeros((n_frames, 128), dtype=np.float32)
    frame = np.zeros((n_frames, 128), dtype=np.float32)
    velocity = np.zeros((n_frames, 128), dtype=np.float32)

    def t_to_f(t): return int(np.clip(np.round(t * FPS), 0, n_frames - 1))

    for inst in pm.instruments:
        for note in inst.notes:
            f_on = t_to_f(note.start)
            f_off = max(f_on + 1, t_to_f(note.end))
            pitch = note.pitch
            onset[f_on, pitch] = 1.0
            frame[f_on:f_off, pitch] = 1.0
            velocity[f_on, pitch] = note.velocity / 127.0

    return onset, frame, velocity

class SlakhStemDataset(Dataset):
    def __init__(self, json_path: str | Path, clip_seconds=10.0):
        """Initialize dataset from JSON index file.
        
        Args:
            json_path: Path to the JSON index file created by the indexer
            clip_seconds: Length of audio clips in seconds
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Extract tracks from the JSON index
        self.tracks = data.get("tracks", [])
        if isinstance(self.tracks, dict):
            self.tracks = list(self.tracks.values())  # Convert dict to list if needed
            
        self.clip_frames = int(clip_seconds * FPS)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        item = self.tracks[idx]

        wav, sr = torchaudio.load(item["wav_path"])
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav.mean(dim=0, keepdim=True)

        spec = amp_to_db(mel_spec(wav)).squeeze(0)
        n_frames = spec.shape[1]

        pm = pretty_midi.PrettyMIDI(item["midi_path"])
        onset, frame, velocity = midi_to_frame_targets(pm, n_frames)

        if n_frames > self.clip_frames:
            start = np.random.randint(0, n_frames - self.clip_frames)
        else:
            start = 0
        end = min(start + self.clip_frames, n_frames)

        spec = spec[:, start:end]
        onset = torch.from_numpy(onset[start:end])
        frame = torch.from_numpy(frame[start:end])
        velocity = torch.from_numpy(velocity[start:end])

        return {
            "spec": spec,
            "onset": onset,
            "frame": frame,
            "velocity": velocity,
            "track": item["track"],
            "stem_id": item["stem_id"],
            "program": item["program"],
            "wav_path": item["wav_path"],
            "midi_path": item["midi_path"],
        }
