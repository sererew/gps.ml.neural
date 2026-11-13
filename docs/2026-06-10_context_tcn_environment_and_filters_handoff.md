# 2026-06-10 - Context TCN, Environment Alignment, and Filter Comparison

## Context

This handoff covers the work done after the earlier
`2026-06-10_pattern_anchor_and_nonbike_handoff.md` note.

The session continued from the pattern-anchor experiment and moved back toward
the main neural-filter question:

```text
Can a different architecture, with contextual per-timestep features, reduce the
drift problem enough to become useful as a GPX filter?
```

The short answer is:

```text
context_tcn_v1 greatly reduces training-time drift metrics, but as a real GPX
filter it only reaches the classic-filter cluster. It is not yet better than
simple smoothing filters.
```

## Active Environment Decision

The project now treats the modern TensorFlow/Keras stack as the active stack:

```text
Python:      3.11.x
TensorFlow:  2.21.0
Keras:       3.14.1
```

This was necessary because the A40 produced `.keras` models using Keras 3. The
previous local environment was TensorFlow/Keras 2.15 and could not deserialize
some A40 models, especially `context_tcn_v1`.

The active dependency file is now:

```text
python/requirements.txt
```

Exact lock files were also added:

```text
python/requirements-context-tf221-win-lock.txt
python/requirements-context-tf221-a40-lock.txt
```

The first lock captures the local Windows environment. The second captures the
A40/Linux GPU environment, including CUDA packages.

Usage notes were documented in:

```text
docs/2026-06-10_python_environment.md
```

Local virtual environments are ignored with:

```text
.venv*/
```

## Remote A40 Script

`python/tools/remote_a40.ps1` was updated to support the context models and the
modern environment.

Current behavior:

```text
default remote:       gpu
default remote dir:   /home/alb/gps.ml.neural
python:               python3.11
remote venv:          venv_context_v1_py311
default training:     context_tcn_v1
```

Supported training values:

```text
context_v1
context_tcn_v1
```

The script now installs the A40 lock:

```text
python/requirements-context-tf221-a40-lock.txt
```

It also fetches only final models:

```text
models/model_final_<training>.keras
models/model_final_<training>.weights.h5
```

## Model Artifact Policy

`model_best_*` was removed from the active training flow.

Reason:

```text
EarlyStopping(..., restore_best_weights=True)
```

is used in the active training scripts. Therefore the model in memory at the end
of training already contains the best validation weights. The final artifact is
the one filters should consume:

```text
model_final_*.keras
```

This removes duplicate model noise from `models/`.

Active training scripts touched:

```text
python/pipeline/6_train_neural_network_v3.py
python/pipeline/6_train_neural_network_context_v1.py
python/pipeline/6_train_neural_network_context_tcn_v1.py
```

## Context Dataset

A new context dataset generator was created:

```text
python/pipeline/5_generate_input_dataset_context_v1.py
```

It writes:

```text
data/input_context_v1
```

Input features include base deltas plus local context:

```text
dx
dy
dz
xy_speed
heading_cos
heading_sin
slope_clipped
rolling_xy_speed_mean_30
rolling_xy_speed_std_30
rolling_pause_ratio_30
rolling_turn_abs_mean_30
rolling_tortuosity_30
rolling_xy_speed_mean_180
rolling_pause_ratio_180
rolling_tortuosity_180
```

Labels remain:

```text
dx
dy
dz
```

Dataset generation produced:

```text
train: 722 windows
val:    71 windows
test:   98 windows
```

## Context Models

Two context training scripts are active:

```text
python/pipeline/6_train_neural_network_context_v1.py
python/pipeline/6_train_neural_network_context_tcn_v1.py
```

`context_v1` is an LSTM model using the contextual features.

`context_tcn_v1` is a temporal convolutional network with dilated convolution
blocks:

```text
dilations: 1,2,4,8,16,32,64,128
filters:   64
kernel:    5
```

Both scripts now save:

```text
models/model_final_<tag>.keras
models/model_final_<tag>.weights.h5
```

## Training Results

The A40 full `context_v1` run reported:

```text
epochs:          100/100
time:            3.77 min
MAE total:       0.3820 m
Mean final drift: 379.9241 m
RMS drift:        363.2280 m
Length diff:      361.6284 m
```

Interpretation:

```text
The contextual LSTM improves local MAE and length difference, but still
accumulates too much drift.
```

The A40 full `context_tcn_v1` run reported:

```text
best epoch:        8
epochs trained:   23/100
time:             2.79 min
MAE total:        0.4538 m
Mean final drift: 51.7342 m
RMS drift:        126.2387 m
Length diff:      668.0806 m
Relative final drift: 0.64%
Relative RMS drift:   1.55%
```

Interpretation:

```text
TCN worsens local MAE versus context_v1, but greatly reduces integrated drift
during training evaluation.
```

This made it worth testing as a real GPX filter.

## Context Filters

Two new GPX filters were added:

```text
python/filters/7_nn_context_v1_filter.py
python/filters/7_nn_context_tcn_v1_filter.py
```

They appear in step 7 as:

```text
nn_context_v1
nn_context_tcn_v1
```

`7_nn_context_v1_filter.py` contains the shared contextual feature-building and
normalization path.

`7_nn_context_tcn_v1_filter.py` wraps the context filter with TCN defaults and
can reconstruct the TCN architecture if a weights fallback is needed.

An accidental unused import was removed from:

```text
python/filters/7_nn_filter.py
```

The removed import was:

```text
from accelerate.commands.menu import input
```

It was not used and broke the fresh modern environment.

## Step 7 Context Filter Run

The context filters were applied to all preprocessed tracks:

```bash
python -X utf8 python/pipeline/7_apply_all_filters.py --filtros nn_context_v1,nn_context_tcn_v1 --overwrite
```

Run result:

```text
510/510 GPX outputs generated
```

The step 7 runner was also updated so subprocess filter execution uses UTF-8:

```text
sys.executable -X utf8
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
encoding="utf-8"
errors="replace"
```

This avoids console decoding failures on Windows when filenames contain
non-ASCII characters.

## Step 8 Comparison

The global comparison was rerun after adding the context filters:

```bash
python -X utf8 python/pipeline/8_compare_tracks.py --max-workers 8
```

Result:

```text
3016 total tasks
3004 successful comparisons
12 failed comparisons
```

The 12 failures are the same underlying issue repeated across filters:

```text
4a_3a_12_I_W_resampled_*.gpx
```

Reason:

```text
No overlapping time range found.
```

The Excel report is:

```text
results/evaluation/track_comparison_results.xlsx
```

The report contains:

```text
Track_Comparison
Filter_Summary
Trimming_Analysis
```

## Comparison Summary

Final global summary after all fixes:

```text
filter_name           tracks  mean_point_deviation_avg
nn_pattern_anchor       210   14.952287
gaussian                254   29.183433
moving_average          254   29.184264
triangular_weighted     254   29.187377
median                  254   29.192797
savgol                  254   29.195003
identity                254   29.195699
kalman                  254   29.217480
nn_context_tcn_v1       254   29.479597
exponential             254   30.120298
nn_context_v1           254   112.591313
nn                       254   129.237949
```

Interpretation:

```text
nn_pattern_anchor is still the best, but it is an oracle-style experiment.
nn_context_tcn_v1 is no longer disastrous; it lands near the classic filters.
nn_context_v1 and the original nn filter remain bad as real GPX filters.
```

The important negative result:

```text
The TCN architecture fixes much of the training drift metric, but it does not
yet beat simple classic smoothing when applied to full GPX tracks.
```

## Triangular Weighted Bug

`triangular_weighted` initially appeared catastrophically bad:

```text
mean_length_deviation:    about 1,220,218 m
mean_point_deviation_avg: about 379.63 m
```

Cause:

```python
np.convolve(data, weights, mode="same")
```

Using `mode="same"` pads with zeros. That is invalid for latitude/longitude and
created huge artificial jumps at the first and last points.

Example observed:

```text
original first point:  43.3854516596, -5.8208586741
filtered first point:  28.92363394,   -3.88057267
```

Fix:

```python
padded_data = np.pad(data, (half_window, half_window), mode="edge")
filtered_data = np.convolve(padded_data, weights, mode="valid")
```

After regenerating all triangular outputs:

```text
255/255 GPX outputs generated
```

After re-comparison:

```text
mean_length_deviation:    274.603337 m
mean_point_deviation_avg: 29.187377 m
```

So `triangular_weighted` is not bad; it was broken at the edges.

## Current Working Interpretation

The project now has three meaningful neural conclusions:

```text
1. v3 and context_v1 can improve local residual metrics but accumulate drift.
2. context_tcn_v1 greatly reduces drift in training evaluation, but as a GPX
   filter it only matches classic filters.
3. sparse pattern anchors can correct drift very well, but that is still an
   oracle-style method unless user-provided anchors or another non-oracle source
   are introduced.
```

Classic filters remain surprisingly strong under the current comparison metric.

The next likely work should be one of:

```text
1. Inspect why context_tcn_v1 training drift does not translate into better GPX
   comparison.
2. Continue the anchor idea with user-provided waypoints instead of pattern
   oracle anchors.
3. Revisit evaluation by modality, especially walking/run versus bicycle.
4. Improve neural targets/losses so the network optimizes accumulated trajectory
   quality, not only local residual quality.
```

## Git and Output Notes

Generated outputs remain under:

```text
results/
```

and should not be committed unless explicitly requested.

Model files under `models/` may appear as local changes after training or model
fetching. Do not commit them unless explicitly requested.

The `6b` slow-drift scripts were moved to `python/archive/` by the user and are
considered discarded for now.
