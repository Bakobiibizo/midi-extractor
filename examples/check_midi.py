#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Any, Dict

try:
    import pretty_midi
except Exception as e:
    print(f"Failed to import pretty_midi: {e}", file=sys.stderr)
    sys.exit(2)


def analyze_midi(path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": path,
        "exists": os.path.exists(path),
        "size_bytes": os.path.getsize(path) if os.path.exists(path) else 0,
    }
    if not info["exists"]:
        return info
    try:
        pm = pretty_midi.PrettyMIDI(path)
        instrument_stats = []
        total_notes = 0
        total_cc = 0
        for inst in pm.instruments:
            n_notes = len(inst.notes)
            n_cc = len(inst.control_changes)
            instrument_stats.append({
                "program": inst.program,
                "name": getattr(inst, "name", ""),
                "is_drum": inst.is_drum,
                "notes": n_notes,
                "control_changes": n_cc,
            })
            total_notes += n_notes
            total_cc += n_cc

        tempi, times = pm.get_tempo_changes()
        info.update({
            "resolution": pm.resolution,
            "instruments": len(pm.instruments),
            "total_notes": total_notes,
            "total_control_changes": total_cc,
            "instrument_stats": instrument_stats,
            "tempo_changes": len(tempi),
            "time_signature_changes": len(pm.time_signature_changes),
            "duration_seconds": pm.get_end_time(),
        })
    except Exception as e:
        info["load_error"] = f"{type(e).__name__}: {e}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Check MIDI file contents and print stats")
    ap.add_argument("midi_path", help="Path to .mid/.midi file")
    ap.add_argument("--json", action="store_true", help="Output JSON only")
    ap.add_argument("--fail-empty", action="store_true", help="Exit non-zero if no notes detected")
    args = ap.parse_args()

    stats = analyze_midi(args.midi_path)
    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"Path: {stats['path']}")
        print(f"Exists: {stats['exists']}")
        print(f"Size (bytes): {stats.get('size_bytes')}")
        if 'load_error' in stats:
            print(f"Load error: {stats['load_error']}")
            return 2
        print(f"Resolution (TPB): {stats.get('resolution')}")
        print(f"Instruments: {stats.get('instruments')}")
        print(f"Total notes: {stats.get('total_notes')}")
        print(f"Total CC: {stats.get('total_control_changes')}")
        print(f"Tempo changes: {stats.get('tempo_changes')}")
        print(f"Time signature changes: {stats.get('time_signature_changes')}")
        print(f"Duration (s): {stats.get('duration_seconds'):.3f}")
        print("Instrument breakdown:")
        for i, ist in enumerate(stats.get('instrument_stats', [])):
            print(f"  [{i}] program={ist['program']} name='{ist['name']}' drums={ist['is_drum']} notes={ist['notes']} cc={ist['control_changes']}")

    if args.fail_empty and (stats.get('total_notes', 0) == 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
