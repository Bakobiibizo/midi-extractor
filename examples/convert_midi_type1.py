#!/usr/bin/env python
import argparse
from pathlib import Path
import mido


def convert_to_type1(src: Path, dst: Path) -> None:
    mid = mido.MidiFile(filename=str(src))
    # Force type 1
    mid.type = 1
    mid.save(str(dst))


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a MIDI file to SMF Type 1")
    ap.add_argument("input_midi", help="Path to input .mid/.midi")
    ap.add_argument("output_midi", nargs="?", help="Path to output .mid (default: <input>_type1.mid)")
    args = ap.parse_args()

    inp = Path(args.input_midi)
    out = Path(args.output_midi) if args.output_midi else inp.with_name(inp.stem + "_type1.mid")
    convert_to_type1(inp, out)
    print("Wrote:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
