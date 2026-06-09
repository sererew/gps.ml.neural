# gps.ml.neural

Experimental project for filtering noisy GPS tracks with classical filters,
neural networks, and slow-drift diagnostics.

The original Java/Maven idea is preserved under `java-initial/`, but the active
work is now Python.

## Current Goal

Given a noisy GPS recording, produce a cleaner track that can be used to compute
route metrics such as:

- horizontal distance
- elevation gain/loss
- point-level deviation against a clean reference pattern

The current best local model is `v3`. It predicts residual corrections over
normalized deltas:

```text
filtered_delta = noisy_delta + predicted_residual
```

The main open problem is accumulated low-frequency drift. Local point error can
look acceptable while the integrated track still drifts hundreds of meters.

## Repository Layout

```text
data/                 Input and preprocessed datasets
docs/                 Notes, handoffs, and remote training commands
java-initial/         Archived initial Java project
models/               Trained model artifacts
python/               Active Python code
  archive/            Superseded Python experiments
  filters/            Individual classical/NN filters
  pipeline/           Main preprocessing/training/evaluation pipeline
  tools/              Diagnostics and utility scripts
results/              Generated outputs, diagnostics, plots, reports
```

`results/` should be treated as generated output. Do not commit it unless there
is a specific reason.

## Python Setup

Create and activate an environment, then install dependencies:

```bash
cd gps.ml.neural
python -m venv .venv
.venv\Scripts\activate
pip install -r python/requirements.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
pip install -r python/requirements.txt
```

TensorFlow is used for neural models. `h5py` is required because some `.keras`
models may need a weight-loading fallback across Keras versions.

## Active Pipeline

The active pipeline scripts are:

```text
python/pipeline/1_resample_recordings.py
python/pipeline/2_generate_consensus_track.py
python/pipeline/3_align_patterns_times.py
python/pipeline/4_resample_patterns.py
python/pipeline/5_generate_input_dataset_v2.py
python/pipeline/6_train_neural_network_v3.py
python/pipeline/6b_train_slow_drift_corrector_v2.py
python/pipeline/6b_train_slow_drift_corrector_v2_xy.py
python/pipeline/6b_train_slow_drift_corrector_v2_1_xy.py
python/pipeline/7_apply_all_filters.py
python/pipeline/8_compare_tracks.py
```

Typical local training/evaluation commands:

```bash
# Fast v3 training check
python -X utf8 python/pipeline/6_train_neural_network_v3.py --fast --seed 42

# Full v3 training
python -X utf8 python/pipeline/6_train_neural_network_v3.py --seed 42

# Evaluate the frozen v3 model
python -X utf8 python/tools/evaluate_v3_model.py
```

## Slow-Drift Work

Slow drift is the current research focus.

Important tools:

```text
python/tools/diagnose_slow_drift.py
python/tools/diagnose_slow_drift_pattern.py
python/tools/evaluate_oracle_slow_correction.py
python/tools/diagnose_parametric_slow_drift.py
```

Useful commands:

```bash
# Diagnose slow/fast error split
python -X utf8 python/tools/diagnose_slow_drift.py --smooth_window 1800 --plots

# Check pattern/coherence of slow drift
python -X utf8 python/tools/diagnose_slow_drift_pattern.py --smooth_window 1800 --plots

# Oracle slow correction upper bound
python -X utf8 python/tools/evaluate_oracle_slow_correction.py --smooth_window 1800 --plots 5

# Parametric slow-drift complexity
python -X utf8 python/tools/diagnose_parametric_slow_drift.py ^
  --split test ^
  --smooth_window 1800 ^
  --control_points 2,3,5,8 ^
  --control_radius 90 ^
  --interpolation cubic ^
  --plots 5 ^
  --seed 42
```

On Linux/macOS, replace `^` line continuations with `\`.

## Main Findings So Far

The complete `v3` model improves local filtering but does not solve drift:

```text
MAE total:          0.5281 m -> 0.4182 m
RMS drift:          523.37 m -> 311.18 m
Length diff:       1319.32 m -> 563.32 m
```

Slow drift dominates the accumulated error. With a 1800-point smoothing window,
the slow component explains most of the XY and Z drift energy.

An oracle correction using the clean pattern shows a high ceiling:

```text
Baseline RMS XY drift:  ~311 m
Oracle RMS XY drift:     ~70 m
```

The most important discovery is that the slow drift is low-dimensional. On the
test split, approximating the oracle slow error with a few XY control points gave:

```text
baseline v3:            RMS XY 425.75 m
moving-average oracle:  RMS XY  58.80 m
5 cubic control points: RMS XY  65.06 m
8 cubic control points: RMS XY  59.10 m
```

This means:

```text
The hard part is not representing the slow correction.
The hard part is estimating those control points without the clean pattern.
```

## Current Experimental Branches

Second-stage slow-drift correctors on top of frozen `v3`:

- `6b_train_slow_drift_corrector_v2.py`: XYZ, 8 control points.
- `6b_train_slow_drift_corrector_v2_xy.py`: XY only, same capacity as v2.
- `6b_train_slow_drift_corrector_v2_1_xy.py`: smaller XY-only variant.

The learned correctors are useful experiments but do not yet infer slow drift
reliably. They remain far from the oracle.

## Handoff Notes

For a fuller technical summary of the slow-drift investigation, see:

```text
docs/2026-06-09_slow_drift_handoff.md
```

Recommended next steps:

1. Measure X, Y, and Z slow-drift complexity separately.
2. Compare learned control-point magnitudes against oracle control-point magnitudes.
3. Try direct control-point regression instead of dense slow-error regression.
4. Keep `v3` as the local correction baseline.
5. Consider anchors or extra information: GPS accuracy/HDOP, track closure,
   route consensus, map matching, DEM/barometer, or controlled synthetic
   slow-drift augmentation.
