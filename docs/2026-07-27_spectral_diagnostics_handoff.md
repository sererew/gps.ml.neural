# 2026-07-27 - Spectral Diagnostics and Current Direction Handoff

## Purpose of This Session

This session shifted the evaluation from aggregate track metrics to a spectral
view of the error. The question was not to isolate a predefined "slow" or
"fast" component, but to measure where the error energy is and what every
filter actually does to it.

The error is always measured against the aligned clean pattern.

```text
error = filtered recording - aligned pattern
```

The analysis covers 0 to 0.5 Hz, the usable range for the 1 Hz recordings.
XY is represented as the sum of the X and Y PSDs. Z is reported separately.

## Tools and Outputs

Main diagnostic tool:

```text
python/tools/diagnose_error_spectrum.py
```

It produces per-recording spectrograms, band-power CSV files, and a summary
table with a common color scale across filters and channels.

Aggregate sweep tool:

```text
python/tools/sweep_error_spectrum_one_per_family.py
```

This selects one recording per family, applies the requested filters when
needed, runs the spectral diagnostic for each family, and aggregates the
results. The selection can be supplied from a previous `selection.csv` so
comparisons use exactly the same recordings.

Current aggregate output including the classic plus anchor experiment:

```text
results/diagnostics/error_spectrum_one_per_family_classic_anchor/
```

Important files in that directory:

```text
band_power.csv
band_power_summary.csv
band_ratio_to_raw.png
selection.csv
```

Generated `results/` files remain ignored by Git.

## Frequency Bands

The current bands are:

```text
0-0.0005 Hz
0.0005-0.001 Hz
0.001-0.003 Hz
0.003-0.01 Hz
0.01-0.05 Hz
0.05-0.15 Hz
0.15-0.5 Hz
```

They were chosen to separate the very slow components visible in the data,
while retaining the higher-frequency region where conventional smoothers act.

## How to Read `band_ratio_to_raw.png`

Each filter row contains `log10(filtered band power / raw band power)`.

```text
 0.0  = unchanged band energy
-1.0  = 10x reduction
 1.0  = 10x amplification
 2.0  = 100x amplification
```

The first row, `raw energy %`, is not a ratio. It reports the percentage of
the raw error energy belonging to each band.

The rightmost `total error %` column reports the integrated error energy after
the filter, relative to the raw error energy of that channel:

```text
100%  = raw error energy
 50%  = half the raw error energy remains
200%  = twice the raw error energy remains
```

This total is calculated from integrated PSD energy, not by summing the band
ratios. Therefore it remains meaningful if the band partition changes.

## Main Measurement: Error Energy Is Low Frequency

For the one-recording-per-family aggregate, raw XY error energy was:

```text
band                 raw XY energy share
0-0.0005 Hz                    22.7%
0.0005-0.001 Hz                25.3%
0.001-0.003 Hz                 19.8%
0.003-0.01 Hz                  19.9%
0.01-0.05 Hz                   10.9%
0.05-0.15 Hz                    1.2%
0.15-0.5 Hz                     0.2%
```

About 88% of XY error energy is below 0.01 Hz.

Raw Z error energy was even more concentrated at low frequency:

```text
band                 raw Z energy share
0-0.0005 Hz                    60.2%
0.0005-0.001 Hz                33.7%
0.001-0.003 Hz                  3.8%
0.003-0.01 Hz                   1.6%
0.01-0.05 Hz                    0.7%
0.05-0.15 Hz                    0.0%
0.15-0.5 Hz                     0.0%
```

About 94% of Z error energy is in the first two bands.

## Filter Findings

### Classical Filters

Gaussian, moving average, median, Savgol, and similar filters produce only
small changes to the total error. They act most clearly in the two highest
bands, but those bands contain little of the raw error energy.

This is not evidence that classical filtering is useless. It is evidence that
the present parameterization does not remove the dominant low-frequency error.

### Base Neural Filter (`nn`)

The important correction to earlier interpretation is:

```text
The neural filter does not merely fail to remove low-frequency error.
It introduces or amplifies low-frequency XY error very strongly.
```

In the aggregate XY table, the base `nn` had approximately:

```text
0-0.0005 Hz:       10^3.9 times raw band energy
0.0005-0.001 Hz:   10^3.8 times raw band energy
0.001-0.003 Hz:    10^2.4 times raw band energy
0.003-0.01 Hz:     10^1.3 times raw band energy
total XY energy:   about 20.4k% of raw
```

This is why aggregate point metrics or trajectory metrics can be misleading:
they did not expose that the dominant low-frequency XY error was being made
far worse by the neural output.

### Pattern Anchors Are Oracle Corrections

Pattern anchors compute a correction from the aligned clean pattern. They are
not available in a production filter that receives only a new GPS recording.

They remain useful as a diagnostic because they show how much could be removed
if a trustworthy external reference were available.

`nn_pattern_anchor` removes much of the damage caused by `nn`, but it is not a
general solution. In XY it still leaves about 304.2% of raw total error energy
in the aggregate and amplifies several low-frequency bands.

### Classic Filter Plus Pattern Anchor

Two separate oracle wrappers were added:

```text
python/filters/7_gaussian_pattern_anchor_filter.py
python/filters/7_moving_average_pattern_anchor_filter.py
```

Shared implementation:

```text
python/filters/classic_pattern_anchor_filter.py
python/filters/pattern_anchor_common.py
```

These wrappers intentionally have no dependency on `7_nn_filter.py`.

The flow is:

```text
raw recording -> classical local filter -> pattern anchor correction
```

This was tested because the prior `nn -> anchor` pipe confounded the anchor
result with damage introduced by the neural stage.

Aggregate results:

```text
filter                         total XY energy    total Z energy
gaussian + pattern anchor             95.3%              3.4%
moving average + pattern anchor       95.3%              3.4%
```

Interpretation:

```text
XY: classic plus anchor only removes about 4.7% of total error energy.
    It avoids the neural low-frequency explosion, but does not solve XY.

Z:  classic plus anchor removes about 96.6% of total error energy.
    It is unexpectedly effective, despite shifting some residual energy into
    mid-low bands.
```

This strongly suggests that Z has a simpler externally constrainable component
than XY. It does not make anchors a deployable solution by itself because the
clean pattern is still used as an oracle.

## Current Technical Conclusion

The project objective remains to remove as much noise as possible across all
frequency bands. It is not acceptable to define success as merely smoothing
high-frequency noise.

The current data and experiments support these narrower conclusions:

```text
1. Most error energy is in low-frequency bands.
2. Current classical filters mainly affect high-frequency bands that contain
   little of the total error energy.
3. The current neural filter is harmful in the dominant low-frequency XY bands.
4. The apparent value of nn_pattern_anchor partly comes from undoing damage
   introduced by nn itself.
5. Pattern anchors are highly effective in Z and only marginally helpful in XY
   when combined with a classical filter.
```

The present position-only neural architecture has not demonstrated a reliable
way to infer the missing low-frequency XY correction. More layers, larger
windows, or another variation of the same architecture should not be started
without a new, testable source of information or a clearly different prior.

## Scope Discipline

Several possible directions were discussed: repeated-route effects,
satellite geometry, atmosphere, receiver state, IMU, map matching, and DEM.

Do not treat these as current pipeline features:

```text
- The available dataset does not contain IMU measurements.
- Satellite selection and internal receiver calculations are not observable.
- Atmospheric and receiver-state explanations are plausible but not usable
  inputs in the current data.
- Learning a correction specific to one known route is not general GPS
  denoising. It would be a separate route-specific product.
```

Repeated recordings of a route may still be useful as a diagnostic of what is
stable within or across sessions, but this is not yet a justification for a new
model. It must not be allowed to create another branch of loosely connected
experiments.

## Recommended State on Resume

1. Keep the spectral diagnostic as the primary evaluation guardrail. Any new
   filter must be compared against raw per band and must not hide low-frequency
   amplification behind aggregate metrics.

2. Do not start another neural architecture using only the existing GPS inputs
   until there is a concrete falsifiable hypothesis about what extra information
   or prior makes low-frequency XY correction identifiable.

3. Treat map matching for XY and DEM or other elevation references for Z as
   possible separate investigations, only if they can be introduced as real
   inputs available at filtering time.

4. Preserve the distinction between an oracle experiment and a deployable
   filter in all future reports and plots.

## Short Version for a New Collaborator

Spectral analysis revealed that the difficult error is overwhelmingly low
frequency. The base neural filter (`nn`) makes that dominant low-frequency XY
error dramatically worse. Classical filters affect mostly high-frequency error
that contributes little to the total.

Using the clean pattern as sparse anchors is an oracle test. It shows a strong
path for Z correction but only marginal XY improvement when used after a
classical filter. The prior interpretation that anchors solved the neural
problem was too optimistic: they mainly repair error introduced by `nn`.

The project should pause neural architecture iteration and use the spectral
diagnostic as the standard acceptance test. The next meaningful direction must
introduce a real source of information or an explicit, valid external prior.
