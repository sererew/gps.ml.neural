# 2026-06-11 - Context TCN v2/v3, Cascade Experiment, and Anchor Baselines

## Context

This handoff covers the work done after:

```text
docs/2026-06-10_context_tcn_environment_and_filters_handoff.md
```

The session continued from the discovery that:

```text
context_tcn_v1 strongly improves training-time drift metrics, but as a GPX
filter it only reaches the classic-filter cluster.
```

The main question became:

```text
Can we keep the TCN drift advantage while recovering the better local MAE of
context_v1?
```

Several architecture variants were tested. The short answer is:

```text
Not yet. The simple context_tcn_v1 remains the strongest neural model for drift.
context_v1 remains the strongest neural model for local MAE and length. Explicit
slow-curve branches and the two-network cascade did not improve the tradeoff.
```

## Remote A40 Runner

`python/tools/remote_a40.ps1` was extended beyond the first context models.

Supported training values now include:

```text
context_v1
context_tcn_v1
context_tcn_v2
context_tcn_v3
context_cascade_v1
```

A new action was added:

```text
run-fetch
```

This runs training and fetches outputs without re-copying the input dataset.
That matters because `data/input_context_v1` is already on the A40 and does not
need to be uploaded for every experiment.

A Windows/PowerShell to Linux/bash line-ending issue was also fixed. The remote
command is now cleaned before bash executes it:

```powershell
$UnixCommand | ssh $Remote "tr -d '\r' | bash -s"
```

Without this, the remote log file could be created with a trailing carriage
return in its name, for example:

```text
training_context_tcn_v3_complete_a40.log\r
```

and then the fetch step could not find it.

## Model Artifact Policy

The active training scripts now save only final artifacts:

```text
models/model_final_<tag>.keras
models/model_final_<tag>.weights.h5
```

Reason:

```text
EarlyStopping(..., restore_best_weights=True)
```

means the model in memory at the end of training already contains the best
validation weights. Keeping both `best` and `final` was redundant noise.

For remote A40 runs, fetched artifacts are first copied under:

```text
models/a40/
```

and then copied to:

```text
models/
```

so the local filters can use the latest fetched models directly.

## Context Dataset

All new models use the same context dataset:

```text
data/input_context_v1
```

Input features remain:

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

The important conceptual point is that the rolling features expose local context
to each timestep. They are partly redundant with deltas, but they can make useful
patterns easier for the network to learn.

## Existing Reference Models

### context_v1

`context_v1` is the best current model for local residual quality.

A40 full result:

```text
epochs trained:        100
MAE total:             0.3820 m
MAE XY step:           0.8207 m
Mean final drift:      379.9 m
RMS drift:             363.2 m
Length diff:           361.6 m
```

Interpretation:

```text
Good local cleanup and good length behavior, but too much integrated drift.
```

### context_tcn_v1

`context_tcn_v1` is the best current model for drift.

Architecture summary:

```text
Conv1D 64 kernel 1
LayerNorm
ReLU
TCN blocks in series with dilations 1,2,4,8,16,32,64,128
Dense 64
Dropout 0.2
Dense 3 -> residual
```

A40 full result:

```text
epochs trained:        23
MAE total:             0.4538 m
MAE XY step:           0.9295 m
Mean final drift:       51.7 m
RMS drift:             126.2 m
Length diff:           668.1 m
```

Interpretation:

```text
Worse local MAE than context_v1, but much better accumulated drift.
```

## context_tcn_v2

New script:

```text
python/pipeline/6_train_neural_network_context_tcn_v2.py
```

New filter:

```text
python/filters/7_nn_context_tcn_v2_filter.py
```

Architecture idea:

```text
one model
two additive branches
fast branch + slow TCN branch -> total residual
```

The loss combined:

```text
residual MAE
trajectory loss using cumsum(noisy_delta + residual_pred)
slow smoothness loss using diff(slow_residual)
```

Config:

```text
lambda_traj:        0.001
lambda_slow_smooth: 0.01
```

A40 full result:

```text
epochs trained:        100
MAE total:             0.4367 m
MAE XY step:           0.9080 m
Mean final drift:      281.3 m
RMS drift:             249.6 m
Length diff:           583.7 m
```

Interpretation:

```text
It improves MAE compared with context_tcn_v1, but loses most of the drift
advantage. It does not solve the tradeoff.
```

## context_tcn_v3

New script:

```text
python/pipeline/6_train_neural_network_context_tcn_v3.py
```

New filter:

```text
python/filters/7_nn_context_tcn_v3_filter.py
```

Architecture idea:

```text
one model
two explicit outputs
fast residual output
slow cumulative curve output
```

Inference logic:

```text
total_residual = fast_residual + diff(slow_curve)
```

The first version used two losses without balancing. That caused the slow curve
loss to dominate because it is measured on an accumulated signal with a larger
scale than per-step residuals.

The active version uses Keras multi-output loss weights:

```python
loss=[residual_mae_loss, slow_curve_mae_loss]
loss_weights=[1.0, 0.02]
```

A40 full result after weighting:

```text
epochs trained:        100
MAE total:             0.4329 m
MAE XY step:           0.8959 m
Mean final drift:      431.1 m
RMS drift:             332.8 m
Length diff:           468.2 m
```

Interpretation:

```text
The training became healthy after weighting, but the explicit slow-curve output
still did not correct drift. It is better than v2 for MAE/length, but worse for
drift.
```

## context_cascade_v1

New script:

```text
python/pipeline/6_train_neural_network_context_cascade_v1.py
```

New filter:

```text
python/filters/7_nn_context_cascade_v1_filter.py
```

This tested two separate networks instead of two branches in one model.

### Fast model

Input:

```text
3600 x 15 context features
```

Architecture:

```text
Conv1D 64 kernel 5
LayerNorm
ReLU
Conv1D 64 kernel 3
LayerNorm
ReLU
Dense 32
Dense 3 -> fast_residual
```

Target:

```text
fast_target = total_residual - diff(slow_curve)
```

So Fast tries to learn the high-frequency part of the residual.

### Slow model

Input:

```text
15 original context features
+ 3 fast_residual
+ 3 fast_delta
= 21 channels
```

Architecture:

```text
Conv1D 64 kernel 1
LayerNorm
ReLU
TCN blocks in series with dilations 1,2,4,8,16,32,64,128
Dense 64
Dropout 0.2
Dense 3 -> slow_error_curve
```

Target:

```text
slow_target = cumsum(fast_delta - clean_delta)
```

Inference:

```text
fast_delta = noisy_delta + fast_residual
filtered_delta = fast_delta - diff(slow_curve_pred)
```

A40 full result:

```text
fast epochs trained:   100
slow epochs trained:    28
slow best epoch:        13
MAE total:             0.9707 m
MAE XY step:           1.9166 m
Mean final drift:      410.3 m
RMS drift:             329.9 m
Length diff:          2599.1 m
```

Interpretation:

```text
The separated-loss idea did not fix the slow correction. The Fast model is
reasonable, but Slow injects too much variation when converted back with diff(),
which explodes trajectory length.
```

A later manual fast-only check on the test split indicated that Fast alone is
close to context_v1 locally:

```text
MAE total:             about 0.3901 m
MAE XY step:           about 0.8311 m
Mean final drift:      about 397.2 m
RMS drift:             about 321.3 m
Length diff:           about 473.7 m
```

That supports the diagnosis:

```text
The cascade failure is mainly the Slow stage, not the Fast stage.
```

## Training Metric Comparison

Training/test split metrics from the current neural experiments:

```text
model                   MAE total   MAE XY   final drift   RMS drift   length diff
context_v1              0.3820 m   0.8207 m    379.9 m      363.2 m      361.6 m
cascade_fast_only       0.3901 m   0.8311 m    397.2 m      321.3 m      473.7 m
v3                      0.4182 m      n/a      287.0 m      311.2 m      563.3 m
context_tcn_v3          0.4329 m   0.8959 m    431.1 m      332.8 m      468.2 m
context_tcn_v2          0.4367 m   0.9080 m    281.3 m      249.6 m      583.7 m
context_tcn_v1          0.4538 m   0.9295 m     51.7 m      126.2 m      668.1 m
context_cascade_v1      0.9707 m   1.9166 m    410.3 m      329.9 m     2599.1 m
```

Current interpretation:

```text
Best MAE:         context_v1
Best drift:       context_tcn_v1
Best length diff: context_v1
Worst failure:    context_cascade_v1 slow stage
```

## GPX Filter Comparison Baseline

The latest global GPX comparison report is:

```text
results/evaluation/track_comparison_results.xlsx
```

Relevant `Filter_Summary` values:

```text
filter_name           total_tracks   mean_point_deviation_avg   mean_length_deviation
nn_pattern_anchor          210              14.952287 m              -74.185843 m
gaussian                   254              29.183433 m              247.411015 m
moving_average             254              29.184264 m              252.255023 m
triangular_weighted        254              29.187377 m              274.603337 m
median                     254              29.192797 m              308.063178 m
savgol                     254              29.195003 m              340.469167 m
identity                   254              29.195699 m              363.000060 m
kalman                     254              29.217480 m              405.698711 m
nn_context_tcn_v1          254              29.479597 m              435.810519 m
exponential                254              30.120298 m              174.941149 m
nn_context_v1              254             112.591313 m             -305.075202 m
nn                         254             129.237949 m             -275.861374 m
```

Interpretation:

```text
nn_pattern_anchor is still the best GPX result, but it is an oracle experiment.
nn_context_tcn_v1 is the best non-oracle neural GPX filter so far, but it only
lands near the classic smoothing filters.
```

## Anchor Reference

The strongest anchor diagnostic remains:

```text
results/diagnostics/pattern_anchor_correction_v3_test_8perhour_min8_cubic/summary.json
```

Configuration:

```text
split:            test
anchors_per_hour: 8
min_anchors:      8
interpolation:    cubic
recordings:       43
```

Key metrics:

```text
method                  RMS XY      final XY      length diff
baseline                476.3 m      366.9 m       806.9 m
moving_average_oracle    62.6 m       33.6 m       774.7 m
pattern_anchor           55.2 m        0.0 m       665.5 m
```

Interpretation:

```text
Sparse anchor correction is very powerful against drift. The unresolved problem
is how to obtain equivalent anchors without using the clean pattern.
```

## Architecture Lessons

Important conceptual notes from the session:

```text
Conv1D operates along the time axis. Channels are features per timestep.
TCN blocks are executed in series, not in parallel.
cumsum integrates deltas into a trajectory-like accumulated signal.
diff converts an accumulated curve back into per-step corrections.
```

The key negative lesson:

```text
Predicting an explicit slow accumulated curve is hard. Small local errors in the
curve can become large per-step artifacts after diff(), especially when used as
a correction signal.
```

The key positive lesson:

```text
TCN receptive field helps with drift. context_tcn_v1 is simple, but it remains
the clearest evidence that architecture can reduce integrated drift.
```

## Current Working Hypotheses

Current hypotheses after these experiments:

```text
1. Local MAE and accumulated drift are partly competing objectives.
2. context_v1 learns local residual cleanup well but does not control long drift.
3. context_tcn_v1 controls drift better but sacrifices local point accuracy.
4. Explicit slow-curve supervision needs a better target or regularization before
   it can be useful.
5. The two-network cascade needs a safer Slow stage; the current one overcorrects
   and inflates length.
6. Anchor correction is still the best evidence that slow drift is correctable,
   but it remains oracle-only.
```

## Files Added or Changed

New active training scripts:

```text
python/pipeline/6_train_neural_network_context_tcn_v2.py
python/pipeline/6_train_neural_network_context_tcn_v3.py
python/pipeline/6_train_neural_network_context_cascade_v1.py
```

New active filters:

```text
python/filters/7_nn_context_tcn_v2_filter.py
python/filters/7_nn_context_tcn_v3_filter.py
python/filters/7_nn_context_cascade_v1_filter.py
```

Remote runner touched:

```text
python/tools/remote_a40.ps1
```

Generated models are present under:

```text
models/
models/a40/
```

Generated results are present under:

```text
results/
```

Do not commit generated `results/` outputs or model files unless explicitly
requested.

## Useful Commands

Run a full A40 training without re-uploading the dataset:

```powershell
cd C:\Users\alb\git\gps.ml.neural\python\tools
.\remote_a40.ps1 -Training=context_tcn_v1 -Action=run-fetch
```

Other supported training values:

```text
context_v1
context_tcn_v1
context_tcn_v2
context_tcn_v3
context_cascade_v1
```

Apply the current TCN v1 GPX filter:

```bash
python -X utf8 python/pipeline/7_apply_all_filters.py --filtros nn_context_tcn_v1 --overwrite
```

Run the comparison:

```bash
python -X utf8 python/pipeline/8_compare_tracks.py --max-workers 8
```

## Suggested Next Step

The cleanest next experiment is probably not more loss mixing.

Most promising options:

```text
1. Improve context_tcn_v1 directly, because it is the only neural model with a
   strong drift result.
2. Add movement-mode features or split training by modality, because bicycle and
   walking/caco tracks appear to have different noise distributions.
3. Revisit anchors as a second-stage correction with non-oracle sources, such as
   user waypoints.
4. Design a safer slow-stage target that cannot inject high-frequency artifacts
   through diff().
```

For now, the best practical summary is:

```text
context_v1 cleans locally.
context_tcn_v1 controls drift.
pattern_anchor proves sparse slow correction can work.
No tested v2/v3/cascade architecture has combined all three properties yet.
```
