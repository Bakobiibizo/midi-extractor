#!/usr/bin/env python
import argparse
import itertools
import json
import time
from pathlib import Path
import requests
import pretty_midi


def transcribe_once(server: str, audio: Path, params: dict) -> tuple[str, Path]:
    base = server.rstrip('/')
    transcribe_url = f"{base}/transcribe"
    with open(audio, 'rb') as f:
        files = {"audio_file": (audio.name, f, "application/octet-stream")}
        r = requests.post(transcribe_url, files=files, params=params, timeout=180)
    r.raise_for_status()
    job_id = r.json()["job_id"]

    status_url = f"{base}/status/{job_id}"
    while True:
        s = requests.get(status_url, timeout=120)
        s.raise_for_status()
        sj = s.json()
        if sj.get("status") in {"completed", "failed"}:
            break
        time.sleep(0.5)

    dl_url = f"{base}/download/{job_id}"
    out = Path("examples") / f"{job_id}.mid"
    resp = requests.get(dl_url, timeout=180)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    return job_id, out


def count_notes(midi_path: Path) -> int:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    return sum(len(i.notes) for i in pm.instruments)


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep thresholds and report note counts")
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument("--server", default="http://localhost:8989")
    args = ap.parse_args()

    audio = Path(args.audio)

    onset_vals = [0.06, 0.08, 0.10, 0.12, 0.15]
    frame_on_vals = [0.15, 0.18, 0.22, 0.26, 0.30]
    frame_off_scales = [0.5, 0.6, 0.7, 0.8]
    min_dur_vals = [0.06, 0.08, 0.10, 0.12, 0.15]

    results = []
    for onset, fon, scale, md in itertools.product(onset_vals, frame_on_vals, frame_off_scales, min_dur_vals):
        params = {
            "onset_threshold": onset,
            "frame_threshold": fon,
            "frame_threshold_off": round(fon * scale, 3),
            "min_note_duration": md,
        }
        try:
            job_id, out = transcribe_once(args.server, audio, params)
            n = count_notes(out)
            results.append({"job_id": job_id, "notes": n, **params})
            print(f"notes={n} onset={onset:.2f} frame_on={fon:.2f} frame_off={fon*scale:.2f} min_dur={md:.2f} -> {out}")
        except Exception as e:
            print(f"error: {e} params={params}")

    # sort by closeness to 16 notes
    results.sort(key=lambda r: abs(r["notes"] - 16))
    best = results[:10]
    print("\nTop candidates (closest to 16 notes):")
    for r in best:
        print(json.dumps(r, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
