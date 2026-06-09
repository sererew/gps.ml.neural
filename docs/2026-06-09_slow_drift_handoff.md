# 2026-06-09 - Slow Drift Investigation Handoff

## Context

This project processes noisy GPS tracks and learns filters against clean reference
patterns. The current production-level model is `v3`, trained by:

```bash
python -X utf8 python/pipeline/6_train_neural_network_v3.py --seed 42
```

The `v3` model predicts residual delta corrections:

```text
filtered_delta = noisy_delta + predicted_residual
```

It works reasonably well at local point level, but accumulated position drift is
still too large for the final goal.

## Project State After This Session

Active pipeline files:

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

Archived experimental versions:

```text
python/archive/6_train_neural_network_v4.py
python/archive/6b_train_slow_drift_corrector_v1.py
```

Important diagnostic tools added:

```text
python/tools/evaluate_v3_model.py
python/tools/diagnose_slow_drift.py
python/tools/diagnose_slow_drift_pattern.py
python/tools/evaluate_oracle_slow_correction.py
python/tools/diagnose_parametric_slow_drift.py
```

Generated outputs are under `results/` and should not be committed unless there
is an explicit reason.

## v3 Findings

The complete `v3` model improved over fast training, but accumulated drift
remains significant.

Reported comparison against fast training:

```text
MAE total:              0.5281 m -> 0.4182 m
Mean final drift:     337.14 m  -> 286.96 m
RMS drift:            523.37 m  -> 311.18 m
Length diff:         1319.32 m  -> 563.32 m
Relative length diff: 16.48%    -> 6.93%
```

The important conclusion is that the local MAE can be acceptable while the
integrated track still drifts hundreds of meters.

## Slow Drift Diagnosis

The slow component dominates the accumulated error. With a smoothing window of
1800 points, the diagnostic showed approximately:

```text
XY RMS error:              311.178 m
XY RMS slow:               296.591 m
XY RMS fast:                69.987 m
XY slow energy / total:      0.908
XY slow energy share:        0.947

Z RMS error:                34.155 m
Z RMS slow:                 30.603 m
Z RMS fast:                 10.452 m
Z slow energy / total:       0.803
Z slow energy share:         0.896
```

This means the main failure mode is not local fast noise. It is low-frequency
accumulated drift.

## Oracle Result

An oracle correction was tested by computing:

```text
slow_error = moving_average(pos_filtered - pos_true, 1800)
pos_corrected = pos_filtered - slow_error
```

This is not usable in production because it uses the clean pattern, but it
measures the upper bound.

Result:

```text
Baseline RMS XY drift:       311.18 m
Oracle RMS XY drift:          69.99 m
Improvement:                 ~77.5%

Baseline mean final XY:      286.96 m
Oracle mean final XY:         51.92 m

Baseline RMS Z:               34.15 m
Oracle RMS Z:                 10.45 m
```

Conclusion: the slow component is worth correcting. The ceiling is high.

## v4 Attempt

`6_train_neural_network_v4.py` added contextual features:

```text
dx, dy, dz, t_norm, distance_norm, absolute_t_norm
```

It helped only modestly:

```text
MAE total:              0.5281 -> 0.5233 m
Mean final XY drift:   337.14 -> 299.15 m
RMS XY drift:          523.37 -> 440.80 m
Length diff:          1319.32 -> 1210.90 m
```

Conclusion: augmented features alone do not solve slow drift.

## 6b Slow Corrector Attempts

The 6b family is a second-stage corrector on top of frozen `v3`.

### v2

Architecture:

```text
recording = sequence of windows
each window: 3600 x 9
Conv1D window encoder
GRU over window summaries
8 x 3 global control points
interpolate control points
subtract predicted slow error from v3 positions
```

Complete run:

```text
Baseline RMS XY drift:      511.03 m
Corrected RMS XY drift:     484.31 m
Oracle RMS XY drift:         66.51 m

Baseline final XY drift:    453.15 m
Corrected final XY drift:   432.41 m
Oracle final XY drift:       35.56 m

Baseline RMS Z drift:        36.60 m
Corrected RMS Z drift:       33.74 m
Oracle RMS Z drift:          10.25 m
```

Training overfit: best validation was around epoch 11, then train loss kept
improving while validation got worse.

### v2_xy Same Capacity

Same architecture capacity as v2, but predicts only XY:

```text
8 x 2 control points
Z unchanged
```

Fast result:

```text
Baseline RMS XY:       403.80 m
Corrected RMS XY:      405.53 m
Oracle RMS XY:          67.89 m

Baseline final XY:     364.54 m
Corrected final XY:    362.88 m

Baseline length diff:  594.40 m
Corrected length diff: 560.78 m
Oracle length diff:    557.94 m
```

Conclusion: separating Z did not improve XY drift, but it did improve length
difference almost to the oracle.

### v2.1 XY Small

Reduced model:

```text
5 x 2 control points
smaller Conv1D/GRU/Dense
Z unchanged
```

Fast result:

```text
Baseline RMS XY:       403.80 m
Corrected RMS XY:      413.02 m
Baseline final XY:     364.54 m
Corrected final XY:    377.82 m
Baseline length diff:  594.40 m
Corrected length diff: 612.58 m
```

Conclusion: this reduced version is not useful as-is.

## Main Discovery: Parametric Slow Drift Is Simple

The most important result came from:

```bash
python -X utf8 python/tools/diagnose_parametric_slow_drift.py \
  --split test \
  --smooth_window 1800 \
  --control_points 2,3,5,8 \
  --control_radius 90 \
  --seed 42
```

This diagnostic uses the clean pattern to compute the oracle slow error, then
approximates that slow error with a small number of XY control points.

Linear interpolation result on `test`:

```text
baseline v3:            RMS XY 425.75 m
moving average oracle:  RMS XY  58.80 m
2 control points:       RMS XY 170.22 m
3 control points:       RMS XY  92.87 m
5 control points:       RMS XY  68.96 m
8 control points:       RMS XY  61.66 m
```

Then cubic interpolation was tested:

```bash
python -X utf8 python/tools/diagnose_parametric_slow_drift.py \
  --split test \
  --smooth_window 1800 \
  --control_points 2,3,5,8 \
  --control_radius 90 \
  --interpolation cubic \
  --plots 5 \
  --seed 42
```

Cubic result:

```text
2 control points:       RMS XY 170.22 m
3 control points:       RMS XY  95.16 m
5 control points:       RMS XY  65.06 m
8 control points:       RMS XY  59.10 m
moving average oracle:  RMS XY  58.80 m
```

Conclusion:

```text
The slow drift is smooth and low-dimensional.
It can be represented very well with 5-8 XY control points.
The hard problem is not representing the correction.
The hard problem is estimating those control points without the clean pattern.
```

This is the key handoff point.

## Current Interpretation

The project should not keep chasing a larger neural network blindly. The data
suggests:

```text
1. v3 handles local correction reasonably.
2. The remaining main error is slow drift.
3. Slow drift is simple to represent.
4. The learned 6b models do not infer the slow drift reliably from current features.
5. More data may help, but the real issue may be lack of an anchor.
```

Potential anchors or future sources of information:

```text
GPS accuracy / HDOP if available
track closure constraints
map matching
DEM or barometric constraints for Z
repeated tracks or route consensus
synthetic slow-drift augmentation
external landmarks or known start/end constraints
```

## Recommended Next Steps

1. Measure X, Y, and Z complexity separately.

   The current parametric diagnostic approximates XY jointly. It would be useful
   to report residuals for X and Y separately, and then repeat for Z.

2. Add a control-point magnitude diagnostic.

   Compare:

   ```text
   predicted control point magnitude from 6b
   oracle control point magnitude
   ```

   This will confirm whether learned models are predicting corrections that are
   too small or too averaged.

3. Consider a direct control-point regression target.

   Instead of training against reconstructed dense slow error, train directly
   against oracle control points. This may be a cleaner supervised target.

4. Keep v3 as the local correction baseline.

   Do not replace v3 yet. Treat slow correction as a second-stage experimental
   branch.

5. Defer GAN/data augmentation until the control-point target is better defined.

   If synthetic augmentation is used, generate known smooth slow drift over real
   clean tracks and train/evaluate against real held-out tracks only.

## Short Version For A New Collaborator

We have a local GPS denoising model (`v3`) that improves point-level deltas but
still accumulates large low-frequency drift. Diagnostics show that this slow
drift dominates the error and can be corrected almost completely by an oracle.

The major discovery is that the slow drift is not complex: 5-8 XY control points,
especially with cubic interpolation, represent it almost as well as a flexible
moving-average oracle. Neural second-stage correctors tried so far do not infer
those points reliably from current inputs.

The next useful work is therefore not "make the network bigger". It is to find a
way to estimate a small set of slow-drift control points, possibly with better
features, constraints, route anchors, or synthetic augmentation.
