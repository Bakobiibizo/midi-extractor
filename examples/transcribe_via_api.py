#!/usr/bin/env python
import argparse
import json
import time
from pathlib import Path
import sys

import requests


def main() -> int:
    ap = argparse.ArgumentParser(description="Transcribe audio via API with optional thresholds")
    ap.add_argument("audio", help="Path to audio file")
    ap.add_argument("--server", default="http://localhost:8989", help="API server base URL")
    ap.add_argument("--profile", type=str, default=None, help="Decoding profile (e.g., bass, piano)")
    ap.add_argument("--onset-threshold", type=float, default=None)
    ap.add_argument("--frame-threshold", type=float, default=None)
    ap.add_argument("--frame-threshold-off", type=float, default=None)
    ap.add_argument("--min-note-duration", type=float, default=None)
    ap.add_argument("--clip-length", type=float, default=None)
    ap.add_argument("--overlap", type=float, default=None)
    # Evaluator-style decoding knobs
    ap.add_argument("--smooth-window", type=int, default=None)
    ap.add_argument("--min-ioi", type=float, default=None)
    ap.add_argument("--merge-gap", type=float, default=None)
    ap.add_argument("--pitch-min", type=int, default=None)
    ap.add_argument("--pitch-max", type=int, default=None)
    ap.add_argument("--enable-fallback-segmentation", action="store_true")
    args = ap.parse_args()

    base = args.server.rstrip("/")
    transcribe_url = f"{base}/transcribe"

    params = {}
    if args.onset_threshold is not None:
        params["onset_threshold"] = args.onset_threshold
    if args.frame_threshold is not None:
        params["frame_threshold"] = args.frame_threshold
    if args.frame_threshold_off is not None:
        params["frame_threshold_off"] = args.frame_threshold_off
    if args.min_note_duration is not None:
        params["min_note_duration"] = args.min_note_duration
    if args.clip_length is not None:
        params["clip_length"] = args.clip_length
    if args.overlap is not None:
        params["overlap"] = args.overlap
    if args.smooth_window is not None:
        params["smooth_window"] = args.smooth_window
    if args.min_ioi is not None:
        params["min_ioi"] = args.min_ioi
    if args.merge_gap is not None:
        params["merge_gap"] = args.merge_gap
    if args.pitch_min is not None:
        params["pitch_min"] = args.pitch_min
    if args.pitch_max is not None:
        params["pitch_max"] = args.pitch_max
    if args.enable_fallback_segmentation:
        params["enable_fallback_segmentation"] = True
    if args.profile is not None:
        params["profile"] = args.profile

    with open(args.audio, "rb") as f:
        files = {"audio_file": (Path(args.audio).name, f, "application/octet-stream")}
        r = requests.post(transcribe_url, files=files, params=params, timeout=120)

    r.raise_for_status()
    data = r.json()
    job_id = data["job_id"]
    print("Job:", job_id)

    # poll
    status_url = f"{base}/status/{job_id}"
    while True:
        s = requests.get(status_url, timeout=60)
        s.raise_for_status()
        sj = s.json()
        print("status:", sj.get("status"), sj.get("progress"))
        if sj.get("status") in {"completed", "failed"}:
            break
        time.sleep(1.0)

    if sj.get("status") != "completed":
        print("Failed:", json.dumps(sj, indent=2))
        return 1

    # download
    dl_url = f"{base}/download/{job_id}"
    out = Path("examples") / f"{job_id}.mid"
    resp = requests.get(dl_url, timeout=120)
    resp.raise_for_status()
    out.write_bytes(resp.content)
    print("Saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
