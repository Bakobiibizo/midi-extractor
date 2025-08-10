#!/usr/bin/env python
"""
Inference Tuner UI (Tkinter)

- Load and persist an audio file
- Adjust decoding parameters (dials)
- Send to running API server, poll status, download MIDI
- Display MIDI check information inline

Run:
  uv run python examples/inference_tuner_ui.py --api http://127.0.0.1:8989

Make sure the API server is running, e.g.:
  uv run python -m src.cli api --port 8989 --model-path checkpoints/best_model.pt
"""
import argparse
import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import json

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import requests
import pretty_midi as pm

API_DEFAULT = "http://127.0.0.1:8989"


@dataclass
class Params:
    profile: str = "bass"
    smooth_window: int = 5
    onset_threshold: float = 0.04
    frame_threshold: float = 0.12
    frame_threshold_off: float = 0.10
    merge_gap: float = 0.01
    min_note_duration: float = 0.10
    min_ioi: float = 0.24
    pitch_min: int = 31
    pitch_max: int = 50
    enable_fallback_segmentation: bool = True


class InferenceTunerUI(tk.Tk):
    def __init__(self, api_base: str):
        super().__init__()
        self.title("MIDI Generator – Inference Tuner")
        self.geometry("920x720")
        self.api_base = api_base.rstrip("/")
        self.audio_path: Path | None = None
        self.last_job_id: str | None = None

        self.params = Params()
        self._build_widgets()

    def _build_widgets(self):
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top: File + API
        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        ttk.Label(top, text="API Base:").pack(side=tk.LEFT)
        self.api_var = tk.StringVar(value=self.api_base)
        api_entry = ttk.Entry(top, textvariable=self.api_var, width=40)
        api_entry.pack(side=tk.LEFT, padx=5)

        self.file_var = tk.StringVar(value="<no file selected>")
        ttk.Button(top, text="Load WAV", command=self.on_load_file).pack(side=tk.LEFT, padx=10)
        ttk.Label(top, textvariable=self.file_var).pack(side=tk.LEFT, padx=5)

        # Middle: Controls
        controls = ttk.LabelFrame(root, text="Decoding Dials")
        controls.pack(fill=tk.X, pady=10)

        # Profile
        ttk.Label(controls, text="Profile").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.profile_var = tk.StringVar(value=self.params.profile)
        ttk.Combobox(controls, textvariable=self.profile_var, values=["bass", "piano", "melody"], width=10).grid(row=0, column=1, padx=5, pady=5)

        # Numeric helpers
        def add_float(row, col, label, varname, from_, to_, step, init):
            ttk.Label(controls, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=5)
            v = tk.DoubleVar(value=init)
            setattr(self, varname, v)
            s = ttk.Scale(controls, from_=from_, to=to_, orient=tk.HORIZONTAL, value=init, command=lambda _=None, vv=v: vv.set(float(s.get())))
            s.grid(row=row, column=col+1, sticky=tk.EW, padx=5)
            s.configure(length=220)
            # Show value
            val_lbl = ttk.Label(controls, textvariable=v, width=6)
            val_lbl.grid(row=row, column=col+2, padx=5)
            return v

        def add_int(row, col, label, varname, from_, to_, init):
            ttk.Label(controls, text=label).grid(row=row, column=col, sticky=tk.W, padx=5, pady=5)
            v = tk.IntVar(value=init)
            setattr(self, varname, v)
            s = ttk.Scale(controls, from_=from_, to=to_, orient=tk.HORIZONTAL, value=init, command=lambda _=None, vv=v: vv.set(int(round(s.get()))))
            s.grid(row=row, column=col+1, sticky=tk.EW, padx=5)
            s.configure(length=220)
            val_lbl = ttk.Label(controls, textvariable=v, width=6)
            val_lbl.grid(row=row, column=col+2, padx=5)
            return v

        colA, colB = 0, 3
        self.smooth_window = add_int(1, colA, "smooth_window", "smooth_window_var", 1, 15, self.params.smooth_window)
        self.onset_threshold = add_float(1, colB, "onset_threshold", "onset_threshold_var", 0.0, 1.0, 0.01, self.params.onset_threshold)
        self.frame_threshold = add_float(2, colB, "frame_threshold", "frame_threshold_var", 0.0, 1.0, 0.01, self.params.frame_threshold)
        self.frame_threshold_off = add_float(3, colB, "frame_threshold_off", "frame_threshold_off_var", 0.0, 1.0, 0.01, self.params.frame_threshold_off)
        self.merge_gap = add_float(4, colB, "merge_gap", "merge_gap_var", 0.0, 0.2, 0.005, self.params.merge_gap)
        self.min_note_duration = add_float(2, colA, "min_note_duration", "min_note_duration_var", 0.01, 0.5, 0.01, self.params.min_note_duration)
        self.min_ioi = add_float(3, colA, "min_ioi", "min_ioi_var", 0.05, 0.6, 0.01, self.params.min_ioi)
        self.pitch_min = add_int(4, colA, "pitch_min", "pitch_min_var", 0, 127, self.params.pitch_min)
        self.pitch_max = add_int(5, colA, "pitch_max", "pitch_max_var", 0, 127, self.params.pitch_max)

        # Checkbox
        self.fallback_var = tk.BooleanVar(value=self.params.enable_fallback_segmentation)
        ttk.Checkbutton(controls, text="enable_fallback_segmentation", variable=self.fallback_var).grid(row=5, column=colB, sticky=tk.W, padx=5, pady=5)

        # Run/Status
        run_row = ttk.Frame(root)
        run_row.pack(fill=tk.X, pady=5)
        ttk.Button(run_row, text="Run", command=self.on_run).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(run_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        # Output
        out = ttk.LabelFrame(root, text="MIDI Check Output")
        out.pack(fill=tk.BOTH, expand=True)
        self.output_text = ScrolledText(out, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def on_load_file(self):
        f = filedialog.askopenfilename(title="Select audio file", filetypes=[("Audio", ".wav .mp3 .flac .m4a .aac"), ("All", "*.*")])
        if f:
            self.audio_path = Path(f)
            self.file_var.set(str(self.audio_path))

    def on_run(self):
        if not self.audio_path or not self.audio_path.exists():
            messagebox.showerror("No file", "Please load an audio file first.")
            return
        self.status_var.set("Submitting...")
        self.output_text.delete("1.0", tk.END)
        threading.Thread(target=self._run_job, daemon=True).start()

    def _params_dict(self):
        return {
            "profile": self.profile_var.get(),
            "smooth_window": int(self.smooth_window.get()),
            "onset_threshold": float(self.onset_threshold.get()),
            "frame_threshold": float(self.frame_threshold.get()),
            "frame_threshold_off": float(self.frame_threshold_off.get()),
            "merge_gap": float(self.merge_gap.get()),
            "min_note_duration": float(self.min_note_duration.get()),
            "min_ioi": float(self.min_ioi.get()),
            "pitch_min": int(self.pitch_min.get()),
            "pitch_max": int(self.pitch_max.get()),
            "enable_fallback_segmentation": bool(self.fallback_var.get()),
        }

    def _run_job(self):
        try:
            api = self.api_var.get().rstrip("/")
            files = {"audio_file": (self.audio_path.name, open(self.audio_path, "rb"), "application/octet-stream")}
            params = self._params_dict()
            # Submit transcription: params go in query string, file in multipart
            r = requests.post(f"{api}/transcribe", files=files, params=params, timeout=120)
            r.raise_for_status()
            job = r.json()
            job_id = job.get("job_id")
            if not job_id:
                raise RuntimeError(f"Unexpected response: {job}")
            self.last_job_id = job_id
            self.status_var.set(f"Job {job_id} submitted")

            # Poll
            midi_path = None
            for _ in range(600):  # up to ~60s
                time.sleep(0.1)
                s = requests.get(f"{api}/status/{job_id}")
                s.raise_for_status()
                st = s.json()
                self.status_var.set(f"{st.get('status')} {st.get('progress')}")
                if st.get("status") == "completed":
                    # Download MIDI
                    dl = requests.get(f"{api}/download/{job_id}")
                    dl.raise_for_status()
                    out_dir = Path("examples")
                    out_dir.mkdir(exist_ok=True)
                    midi_path = out_dir / f"{job_id}.mid"
                    with open(midi_path, "wb") as f:
                        f.write(dl.content)
                    break
                if st.get("status") == "failed":
                    raise RuntimeError(st.get("error") or st)

            if not midi_path or not midi_path.exists():
                raise RuntimeError("No MIDI downloaded – timeout or server error")

            # Check MIDI
            stats = self._check_midi(midi_path)
            self._display_stats(self.audio_path, midi_path, stats)
            self.status_var.set("Done")
        except Exception as e:
            self.status_var.set("Error")
            messagebox.showerror("Error", str(e))

    def _check_midi(self, midi_path: Path) -> dict:
        m = pm.PrettyMIDI(str(midi_path))
        notes = sum(len(inst.notes) for inst in m.instruments)
        insts = [
            {
                "program": inst.program,
                "name": getattr(inst, "name", ""),
                "drums": inst.is_drum,
                "notes": len(inst.notes),
                "cc": len(inst.control_changes),
            }
            for inst in m.instruments
        ]
        duration = m.get_end_time() if m else 0.0
        return {
            "path": str(midi_path),
            "exists": midi_path.exists(),
            "instruments": len(m.instruments),
            "total_notes": notes,
            "total_cc": sum(i["cc"] for i in insts),
            "duration_s": duration,
            "instrument_breakdown": insts,
        }

    def _display_stats(self, audio_path: Path, midi_path: Path, stats: dict):
        self.output_text.insert(tk.END, f"Audio: {audio_path}\n")
        self.output_text.insert(tk.END, f"MIDI:  {midi_path}\n\n")
        self.output_text.insert(tk.END, json.dumps(stats, indent=2))
        self.output_text.insert(tk.END, "\n")
        self.output_text.see(tk.END)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=API_DEFAULT, help="API base URL (e.g., http://127.0.0.1:8989)")
    args = ap.parse_args()

    app = InferenceTunerUI(api_base=args.api)
    app.mainloop()


if __name__ == "__main__":
    main()
