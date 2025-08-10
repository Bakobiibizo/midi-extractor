import yaml
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel
import torch
import torchaudio
import pretty_midi


# Map General MIDI programs to categories
TARGET_PROGRAMS = {
    "piano": list(range(0, 8)),         # Acoustic + Electric Pianos
    "bass": list(range(32, 40)),        # Finger, Picked, Fretless, etc
    "synth": list(range(80, 88)),       # Synth Leads
}

INSTRUMENT_PROGRAMS = set(sum(TARGET_PROGRAMS.values(), []))

def load_metadata(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

class TrackData(BaseModel):
    wav_path: str
    midi_path: str
    program: int
    track: str
    stem_id: str
    metadata: dict | None = None

class DataSet(BaseModel):
    tracks: dict[str, TrackData] | None = None
    metadata: dict | None = None

def build_index(
    root: Path,
    include_types: List[str] = ["piano", "bass", "synth"]
):
    items = []
    instrument_programs = sum((TARGET_PROGRAMS[t] for t in include_types), [])

    # Search for Track* directories recursively
    # This handles both flat structures (babyslakh) and nested structures (slakh2100)
    track_dirs = []
    
    # First check if there are Track* directories directly in root
    direct_tracks = list(root.glob("Track*"))
    if direct_tracks:
        track_dirs.extend(direct_tracks)
    else:
        # If no direct tracks, search recursively in subdirectories
        # This handles Slakh2100 structure with train/, test/, validation/ subdirs
        track_dirs.extend(root.glob("*/Track*"))
        track_dirs.extend(root.glob("**/Track*"))  # Even deeper nesting if needed

    for track_dir in sorted(track_dirs):
        if not track_dir.is_dir():
            continue
            
        meta_path = track_dir / "metadata.yaml"
        if not meta_path.exists():
            continue

        metadata = load_metadata(meta_path)

        for stem_id, inst_meta in metadata.get("stems", {}).items():
            program = inst_meta.get("program_num")
            if program in instrument_programs:
                # Check for both .wav and .flac audio formats
                wav_path = track_dir / "stems" / f"{stem_id}.wav"
                flac_path = track_dir / "stems" / f"{stem_id}.flac"
                midi_path = track_dir / "MIDI" / f"{stem_id}.mid"

                # Use whichever audio format exists
                audio_path = None
                if wav_path.exists():
                    audio_path = wav_path
                elif flac_path.exists():
                    audio_path = flac_path

                if audio_path and midi_path.exists():
                    items.append({
                        "wav_path": str(audio_path),  # Keep the field name for compatibility
                        "midi_path": str(midi_path),
                        "program": program,
                        "track": track_dir.name,
                        "stem_id": stem_id,
                        "metadata": metadata["stems"].get(stem_id)
                    })

    return items

def get_dataset_index(input_path: Path | str) -> DataSet:
    if isinstance(input_path, str):
        input_path = Path(input_path)

    items = build_index(input_path)
    dataset_data = {}

    for item in items:
        track_data = TrackData(**item)
        dataset_data[track_data.track] = track_data

    dataset = DataSet(
        tracks=dataset_data,
        metadata={
            "dataset": input_path.name,
            "num_tracks": len(dataset_data),
            "filter_instruments": list(TARGET_PROGRAMS.keys())
        }
    )
    
    output_path = input_path / "dataset.json"

    with open(output_path, "w") as f:
        f.write(dataset.model_dump_json(indent=2))

    return dataset

def get_db(wav_path):
    wav, sr = torchaudio.load(wav_path)
    wav = wav.mean(dim=0)
    rms = torch.sqrt(torch.mean(wav ** 2))
    return 20 * torch.log10(rms + 1e-8)

def is_audio_blank(wav_path, threshold_db=-40.0, debug=False):
    wav, sr = torchaudio.load(wav_path)
    if sr != 22050:
        wav = torchaudio.functional.resample(wav, sr, 22050)
    wav = wav.mean(dim=0)  # convert to mono

    db = get_db(wav_path)
    if debug:
        print(f"DB: {db.item()}")
    return db.item() < threshold_db


def is_midi_blank(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        return all(len(inst.notes) == 0 for inst in midi.instruments)
    except Exception:
        return True  # if corrupted, treat as blank


def filter_index(dataset_path: Path, output_path: Path = None, threshold_db=-40.0, debug=False):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    filtered_tracks = {}

    for track_name, track_data in dataset["tracks"].items():
        program = track_data["program"]
        if program not in INSTRUMENT_PROGRAMS:
            continue
        
        if is_audio_blank(track_data["wav_path"], threshold_db=threshold_db, debug=debug):
            if debug:
                print(f"[SKIP] Silent audio @ {get_db(track_data['wav_path']):.2f} dB → {track_data['wav_path']}")
            continue
        
        if is_midi_blank(track_data["midi_path"]):
            if debug:
                print(f"[SKIP] Empty MIDI → {track_data['midi_path']}")
            continue
        
        filtered_tracks[track_name] = track_data

    filtered_dataset = {
        "tracks": filtered_tracks,
        "metadata": {
            "source": str(dataset_path.name),
            "num_tracks": len(filtered_tracks),
            "filter_programs": sorted(list(INSTRUMENT_PROGRAMS)),
            "filter_instruments": list(TARGET_PROGRAMS.keys()),
        }
    }

    if output_path is None:
        output_path = dataset_path.parent / "filtered_dataset.json"

    with open(output_path, "w") as f:
        json.dump(filtered_dataset, f, indent=2)

    print(f"[✓] Saved filtered dataset with {len(filtered_tracks)} tracks to {output_path}")
    return output_path


if __name__ == "__main__":
    input_path = Path("datasets/babyslakh_16k/Track00007/stems/S08.wav")
    result = is_audio_blank(input_path, threshold_db=-40.0, debug=True)
    print(result)