import os
import yaml
from pathlib import Path
from typing import List

# Map General MIDI programs to categories
TARGET_PROGRAMS = {
    "piano": list(range(0, 8)),         # Acoustic + Electric Pianos
    "bass": list(range(32, 40)),        # Finger, Picked, Fretless, etc
    "synth": list(range(80, 88)),       # Synth Leads
}

def load_metadata(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_index(
    root: Path,
    include_types: List[str] = ["piano", "bass", "synth"]
):
    items = []
    instrument_programs = sum((TARGET_PROGRAMS[t] for t in include_types), [])

    for track_dir in sorted(root.glob("Track*")):
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists():
            continue
        metadata = load_metadata(meta_path)
        for stem_id, inst_meta in metadata["stems"].items():
            program = inst_meta["program_num"]
            if program in instrument_programs:
                wav_path = track_dir / "stems" / f"{stem_id}.wav"
                midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
                if wav_path.exists() and midi_path.exists():
                    items.append({
                        "wav_path": str(wav_path),
                        "midi_path": str(midi_path),
                        "program": program,
                        "track": track_dir.name,
                        "stem_id": stem_id
                    })
    return items
