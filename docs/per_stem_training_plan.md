# Per-Stem Training Plan and Implementation Notes

This document summarizes our decisions and the implementation plan to train separate models per stem type in `midi-extractor`.

## Findings

- __Existing pipeline__: The repo already includes dataset indexing, dataset loaders, a training loop, and inference.
  - Indexer: `src/datasets/stem_indexer.py`
  - Dataset: `src/datasets/slakh_stem_dataset.py`
  - Trainer: `src/train/run_trainer.py`, `src/train/trainer.py`
  - Inference: `src/inference/inference.py`
- __Current stem handling__: `TARGET_PROGRAMS` in `src/datasets/stem_indexer.py` maps GM program ranges to categories (currently `piano`, `bass`, `synth`). The indexer produces a JSON consumed by `SlakhStemDataset`.
- __Feasibility__: Per-stem training fits the codebase with minimal changes: generate per-stem JSON indices and train a checkpoint per stem.

## Confirmed Stem Taxonomy (current scope)

- __bass__ (GM 32–39)
- __piano/keys__ (GM 0–7 pianos; optionally include organs GM 16–23 if we decide to treat as keys)
- __synth_lead__ (GM 80–87)
- __synth_pad_fx__ (GM 88–103)
- __guitar__ (GM 24–31)
- __strings_ensemble__ (GM 40–51)
- __drums__ (core kit; `is_drum=True`, non-cymbal)
- __high_percs__ (cymbals/hi-hats/crashes/risers; subset of `is_drum=True`)
- __other__ (fallback for anything not mapped)
- __vocals__ (placeholder only; no training yet, but keep the hook for future)

## Musical Roles (for loss/aug policy)

Add a role tag to each stem in the index metadata: `role ∈ {melodic, harmonic, percussive}`

- __melodic__: `synth_lead`, `bass` (bass can be dual-role; configurable)
- __harmonic__: `piano/keys`, `guitar`, `strings_ensemble`, `synth_pad_fx`
- __percussive__: `drums`, `high_percs`

Use role to adjust:
- Loss weighting (e.g., onset/pitch weight ↑ for melodic; frame/voicing continuity ↑ for harmonic; onset emphasis ↑ for percussive)
- Augmentations (e.g., time-stretch caution on percussive; pitch-shift rules on harmonic vs melodic)

## Indexer Changes (`src/datasets/stem_indexer.py`)

- __Extend `TARGET_PROGRAMS`__:
  - `guitar`: GM 24–31
  - `strings_ensemble`: GM 40–51
  - `synth_pad_fx`: GM 88–103
  - Keep `synth_lead` as GM 80–87 (split from existing `synth`)
- __Drums handling__:
  - Read `inst_meta.get("is_drum")` from `metadata.yaml` and collect drum stems separately from GM programs.
  - Split `drums` vs `high_percs` using a heuristic on the audio file:
    - High percs likely: higher spectral centroid/rolloff, longer noisy tails; lower transient-to-sustain ratio for cymbals.
    - Core drums: strong transient energy in LF/MF bands; shorter decays (kick/snare/toms/claps).
- __Metadata additions__ per track in the JSON index:
  - `metadata.is_drum: bool` (if available)
  - `metadata.role: "melodic" | "harmonic" | "percussive"`
- __Filtering__: `filter_index()` remains compatible; it will accept the expanded `TARGET_PROGRAMS` and `INSTRUMENT_PROGRAMS` and skip silence/empty MIDI as before.

## Per-Stem Datasets

For each stem `S`, we will produce:
- `<dataset_root>/dataset_S.json` (built with `include_types=[S]` and drum split logic)
- `<dataset_root>/filtered_dataset_S.json` via `filter_index()`

These JSONs are consumed by `SlakhStemDataset` unchanged; additional metadata fields are additive and backward-compatible.

## Training Orchestration

Two options (we can implement either; CLI is preferred):

- __Option A: Orchestration script__ `src/train/train_per_stem.py`
  - Generates per-stem indices from a dataset root
  - Applies filtering
  - Trains each stem sequentially with configurable hyperparams
  - Saves to `models/<stem>/...`

- __Option B: CLI subcommand__ in `src/cli.py` (preferred)
  - `train-per-stem --dataset-root <path> --stems piano,bass,... --epochs 100 --batch-size 64 --clip-seconds 20.0 --outdir models` 
  - Internally calls the same indexing/filtering/training routines.

Both will:
- Log summaries per stem (Rich logs; reuse `Trainer`’s summary export in `src/train/trainer.py`).
- Persist best checkpoints to `models/<stem>/best.pt` (or similar naming).

## Inference Routing (`src/inference/inference.py`)

- Add a small resolver that maps `instrument_name` (or requested stem) to the corresponding checkpoint directory.
- Fallback order: exact stem → role-compatible model → `other`.
- Keep a placeholder for `vocals` so a future checkpoint can be dropped in without code changes.

## Example Commands (A6000-friendly)

Assuming `uv` is available and dataset root is `datasets/slakh2100` (or `datasets/babyslakh_16k`):

```bash
# 1) Build and filter per-stem indices (CLI to be added)
uv run src/cli.py build-index --dataset-root datasets/slakh2100 --stems piano,bass,synth_lead,synth_pad_fx,guitar,strings_ensemble,drums,high_percs,other
uv run src/cli.py filter-index --dataset-root datasets/slakh2100 --stems piano,bass,synth_lead,synth_pad_fx,guitar,strings_ensemble,drums,high_percs,other

# 2) Train per stem (sequential). For RTX A6000, larger batch and longer clips are OK.
uv run src/cli.py train filtered_dataset_piano.json --epochs 100 --batch-size 64 --clip-seconds 20.0 --lr 2e-5 --dataset slakh
uv run src/cli.py train filtered_dataset_bass.json --epochs 100 --batch-size 64 --clip-seconds 20.0 --lr 2e-5 --dataset slakh
# ... repeat for each stem

# (Planned) One-shot orchestrated training
uv run src/cli.py train-per-stem --dataset-root datasets/slakh2100 \
  --stems piano,bass,synth_lead,synth_pad_fx,guitar,strings_ensemble,drums,high_percs,other \
  --epochs 100 --batch-size 64 --clip-seconds 20.0 --lr 2e-5 --outdir models
```

Notes:
- If using BabySlakh: `--dataset babyslakh` and consider smaller `--batch-size`.
- You can profile GPU memory in `Trainer` via the existing GPU/system info utilities (`src/train/trainer.py`).

## Directory Layout (expected)

```
models/
  piano/
    best.pt
  bass/
    best.pt
  synth_lead/
    best.pt
  synth_pad_fx/
    best.pt
  guitar/
    best.pt
  strings_ensemble/
    best.pt
  drums/
    best.pt
  high_percs/
    best.pt
  other/
    best.pt
```

## Vocals (Future)

- Keep `vocals` in the taxonomy but skip indexing/training until we have data (e.g., MUSDB18-HQ stems, separate melody annotation).
- When data is available, add a `vocals` dataset builder and train a dedicated checkpoint; inference routing already supports the slot.

## Implementation Checklist

- __Indexer__
  - [ ] Expand `TARGET_PROGRAMS` with new categories and ranges
  - [ ] Add `is_drum` handling and implement `drums` vs `high_percs` heuristic
  - [ ] Add `role` to per-track metadata
  - [ ] Ensure `filter_index()` remains compatible

- __CLI / Orchestration__
  - [ ] Add `train-per-stem` subcommand (preferred) or a `train_per_stem.py` orchestration script
  - [ ] Generate per-stem JSONs and filtered JSONs
  - [ ] Sequentially train each stem; export best checkpoints and summaries

- __Inference__
  - [ ] Model resolver by stem name (with fallbacks), directory structure as above
  - [ ] Keep placeholder for `vocals`

## Defaults to Start With (tunable)

- Epochs: 100
- Batch size: 64 (A6000), 32 (smaller GPUs)
- Clip length: 20.0s
- LR: 2e-5 to 3e-5
- Loss weighting: per-role adjustments (to be finalized after first baselines)

---

Prepared for switching to the dual A6000 machine. Once this doc lands, the next PR will implement the indexer changes and the `train-per-stem` CLI, then we can kick off training.
