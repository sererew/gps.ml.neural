# 2026-06-10 - Pattern Anchor Experiment and Non-Bike Evaluation

## Context

This session continued the investigation of whether a neural filter can correct
noisy GPS recordings against clean reference patterns.

The main finding from the previous session still holds: the `v3` neural model
can reduce local residual error, but it tends to accumulate low-frequency drift.
That drift can dominate the final track even when point-level MAE looks
reasonable.

Today focused on a practical question:

```text
Can sparse, pattern-derived control points correct the slow drift introduced by
the neural filter?
```

This is an oracle-style experiment, because the control points are derived from
the clean pattern. It is not a production filter yet. Its purpose is to measure
whether sparse anchors are enough to recover a useful track after the neural
filter.

## Files Changed

Main code touched during the session:

```text
python/filters/7_nn_filter.py
python/filters/7_nn_pattern_anchor_filter.py
python/pipeline/8_compare_tracks.py
```

Related diagnostic tools present in the working tree:

```text
python/tools/diagnose_parametric_slow_drift_channels.py
python/tools/evaluate_pattern_anchor_correction.py
```

Generated outputs remain under `results/` and should not be committed unless
explicitly requested.

## Pattern Anchor Filter

A new experimental slow-correction mode was added to the neural filter:

```text
--slow-correction pattern-anchor
```

It is also exposed through a wrapper:

```text
python/filters/7_nn_pattern_anchor_filter.py
```

The wrapper makes the filter visible as:

```text
nn_pattern_anchor
```

The correction flow is:

```text
1. Run the existing v3 neural filter.
2. Load the clean pattern for the same family.
3. Select sparse anchor timesteps from the pattern timeline.
4. Estimate the error between neural output and pattern at those anchors.
5. Interpolate that sparse error curve over the valid track interval.
6. Subtract the interpolated slow error from the neural output.
```

This tests whether sparse control points are enough to undo the slow drift.

## Current Pattern Anchor Defaults

The current wrapper defaults are:

```text
slow_correction:        pattern-anchor
anchors_per_hour:       8
min_anchors:            8
anchor_channels:        x,y,z
anchor_interpolation:   cubic
anchor_trim_to_pattern: true
anchor_edge_skip:       180 points
anchor_edge_blend:      30 points
```

At 1 Hz:

```text
anchor_edge_skip = 180 points = 3 minutes
anchor_edge_blend = 30 points = 30 seconds
```

The edge skip is important because the beginning and end of some families are
not reliable anchor zones. Using those borders directly created large artificial
offsets and diagonal jumps.

The tradeoff is that trimming/blending can affect accumulated altitude metrics,
especially when there is a height step near the beginning or end.

## Alignment Findings

The first version of the experiment aligned pattern and recording by array
index. That failed on some families because the pattern and recording can have a
time offset.

The filter was changed to prefer timestamp-based alignment:

```text
use common timestamps when available
fallback to index alignment only when timestamps are missing
```

This fixed one class of obvious mismatch, but family 17 exposed a deeper edge
case.

In `17_11_6_E_B`, the first common timestamp still had a very large spatial
offset:

```text
recording-pattern at first common timestamp: about 92.9 m
global median recording-pattern offset:      about 3 m
```

So the issue was not only filename or timestamp matching. The first minutes of
that family are anomalous for this correction method.

The current mitigation is:

```text
trim to the common pattern timeline
avoid selecting anchors near the first/last 180 points
blend correction near the edges
```

## Bicycle Passes

A confusing point was corrected during the session.

The bicycle passes are:

```text
12, 13, 16, 17
```

These are pass numbers, not track numbers.

The working hypothesis is that bicycle recordings behave differently because
the higher speed spreads GPS error over larger point-to-point distances. The GPS
noise is therefore less dominant relative to movement. Walking and walk/run
tracks contain more varied artifacts: rebounds, low-speed noise, stops, point
clusters, and low-frequency drift.

This suggests that mixing movement modalities may be hurting the neural model.
The problem may not be a single "GPS noise" distribution.

## Step 7 Run

All filters were applied to the non-bike pass set:

```text
1,2,3,4,4a,4b,4c,4d,5,6,7,8,9,10,11,14,15,15a,15b,15c,15d
```

The bike passes were excluded:

```text
12,13,16,17
```

Command used:

```bash
python -X utf8 python/pipeline/7_apply_all_filters.py --pasadas "1,2,3,4,4a,4b,4c,4d,5,6,7,8,9,10,11,14,15,15a,15b,15c,15d" --overwrite
```

The planned workload was:

```text
1980 filter-track combinations
```

A post-run count confirmed that every filter produced 198 GPX outputs for the
selected pass set:

```text
exponential          198
gaussian             198
identity             198
kalman               198
median               198
moving_average       198
nn                   198
nn_pattern_anchor    198
savgol               198
triangular_weighted  198
```

## Step 8 Optimization

`python/pipeline/8_compare_tracks.py` was optimized because the comparison step
was taking too long.

Changes:

```text
added --filtros
added --max-workers
cached pattern data per worker
replaced geopy.geodesic point loops with UTM + NumPy vectorized distances
```

Measured examples:

```text
nn_pattern_anchor, pass 17:       about 3.9 s
nn_pattern_anchor, passes 6/10/17: about 7.4 s
all filters, passes 6/10/17:      about 39.7 s
all filters, non-bike set:        about 4.7 min
```

The UTM vectorization may slightly change decimal-level distance metrics versus
geodesic calculations, but it should preserve the meaningful ranking.

## Step 8 Non-Bike Evaluation

Command used:

```bash
python -X utf8 python/pipeline/8_compare_tracks.py --pasadas "1,2,3,4,4a,4b,4c,4d,5,6,7,8,9,10,11,14,15,15a,15b,15c,15d" --output results/reports/track_comparison_excluding_bike_12_13_16_17_all_filters.xlsx
```

Result:

```text
total tasks:   1980
successful:    1970
missing/fail:    10
```

The 10 failures correspond to pass `4a`, where the comparison reported no
temporal overlap. It affected all filters equally.

Output report:

```text
results/reports/track_comparison_excluding_bike_12_13_16_17_all_filters.xlsx
```

## Summary Metrics

Mean point deviation on the non-bike evaluation:

```text
filter_name          tracks  mean_point_deviation_avg_m
nn_pattern_anchor       197                       12.503
gaussian                197                       30.206
moving_average          197                       30.207
median                  197                       30.214
savgol                  197                       30.216
identity                197                       30.217
kalman                  197                       30.231
exponential             197                       30.543
nn                      197                       96.646
triangular_weighted     197                      373.995
```

Other notable values:

```text
nn_pattern_anchor mean length deviation:   -29.929 m
nn_pattern_anchor mean gain deviation:     -13.399 m
nn_pattern_anchor mean loss deviation:      -7.235 m
nn std point deviation:                     41.758 m
nn_pattern_anchor std point deviation:      12.806 m
```

`triangular_weighted` remains unstable in this report, with extreme length
deviation. It should not be interpreted as a competitive filter in the current
state.

## Per-Pass Notes

The oracle anchor correction improved every evaluated non-bike pass compared to
the base neural filter.

Approximate mean point deviation:

```text
pass   nn_pattern_anchor   classical_family_around   nn
1                 11.23                   24.92       66.91
10                 8.98                   30.42       45.64
11                11.10                   43.23       65.84
14                 9.35                   21.37       39.50
15                 7.04                   25.54       97.93
15a                6.17                    8.63       67.45
15b                9.15                   21.23       64.27
15c                6.71                   12.01       66.21
15d                7.38                   10.10       75.75
2                 12.86                   27.37       54.34
3                  8.86                   23.70       31.96
4                  4.33                   14.88       86.66
4b                 4.59                    9.92       36.35
4c                 3.48                    6.58       54.06
4d                 5.10                    7.56      114.47
5                 26.97                   60.50      301.53
6                 32.74                   50.28      363.57
7                 11.54                   25.32       52.52
8                  9.27                   25.23       43.89
9                  9.08                   24.19       47.27
```

The strongest conclusion is not that this is a usable final filter. The
strongest conclusion is that sparse slow-drift control points are powerful.

## Interpretation

The non-bike evaluation supports three conclusions:

```text
1. Base nn v3 is still not good enough as a GPX filter.
2. Sparse anchors can correct most of the damaging drift in this experiment.
3. Movement modality matters and should probably be modeled or split.
```

`nn_pattern_anchor` is an oracle filter because it uses the clean pattern. It
should be treated as a feasibility test for future approaches such as:

```text
manual user waypoints
automatically detected anchor candidates
pseudo-pattern generation
bootstrapping more training pairs
specialized artifact filters
```

The result suggests that if real anchors can be supplied or estimated with
enough quality, a second-stage correction may be practical.

## Open Questions

Important follow-ups:

```text
Should bike and non-bike tracks be trained/evaluated separately?
Can movement mode be inferred and used as an input feature?
Can anchors be supplied by the user as waypoints instead of using the pattern?
How many anchors per hour are actually needed?
Should X/Y and Z be corrected with different anchor strategies?
Can pseudo-patterns be generated well enough for bootstrapping 6b training?
Should artifacts be handled by specialized filters before or after nn?
```

Candidate artifact-specific filters:

```text
rebound reduction
low-frequency drift correction
stationary point cluster cleanup
subsampling/path-shape stabilization
mode-specific filtering for walking, walk/run, and bicycle
```

## Current Caution

The anchor experiment works well as an oracle, but the current implementation
depends on the pattern. It should not be interpreted as solving the original
production problem yet.

The useful discovery is narrower and stronger:

```text
The neural output contains correctable slow drift, and sparse control points can
remove much of it.
```

That makes the anchor/pseudo-pattern direction worth keeping.

## Useful Commands

Apply all filters to the current non-bike set:

```bash
python -X utf8 python/pipeline/7_apply_all_filters.py --pasadas "1,2,3,4,4a,4b,4c,4d,5,6,7,8,9,10,11,14,15,15a,15b,15c,15d" --overwrite
```

Evaluate all filters on the same set:

```bash
python -X utf8 python/pipeline/8_compare_tracks.py --pasadas "1,2,3,4,4a,4b,4c,4d,5,6,7,8,9,10,11,14,15,15a,15b,15c,15d" --output results/reports/track_comparison_excluding_bike_12_13_16_17_all_filters.xlsx
```

Evaluate only the anchor filter:

```bash
python -X utf8 python/pipeline/8_compare_tracks.py --pasadas "1,2,3,4,4a,4b,4c,4d,5,6,7,8,9,10,11,14,15,15a,15b,15c,15d" --filtros nn_pattern_anchor --output results/reports/track_comparison_excluding_bike_12_13_16_17_anchor_only.xlsx
```

