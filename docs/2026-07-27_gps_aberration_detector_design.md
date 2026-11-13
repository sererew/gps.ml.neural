# 2026-07-27 - GPS Aberration Detector Notes

## Current Purpose

Distance to OSM alone cannot distinguish:

1. A real path that is missing from the map.
2. A temporary GPS aberration away from the real route.

The current detector finds off-map intervals and classifies only clear cases.
Prediction uses the recording and OSM. The clean pattern is optional and is
used only by the diagnostic script to evaluate the result.

Reusable logic is kept in one module:

```text
python/gps_aberration_detector.py
```

The CLI, plots, files, and pattern-based evaluation remain in:

```text
python/tools/diagnose_gps_aberrations.py
```

This separation is enough for now. Split the detector further only when a real
second implementation or a substantially larger set of evidence requires it.

## Evidence Used Today

For every interval without map support:

```text
- duration
- map support before and after the interval
- maximum and median distance to OSM
- raw XY path length
- shortest OSM route between the entry and exit
- ratio between OSM and raw path lengths
```

The classifier returns `gps_aberration`, `off_map`, or `uncertain`. Abstaining
is important because some unmapped paths and smooth GPS displacements are
fundamentally ambiguous from position and map data alone.

## Possible Extensions

Add these as independent measurements when an observed case justifies them:

### XY Shape and Motion

```text
- complete distance-to-map profile and its rate of change
- heading agreement with candidate OSM edges
- continuity of the mapped route before and after the event
- displacement versus travelled distance
- speed, acceleration, jerk, and turn rate
- jumps, frozen points, loops, U-turns, and self-intersections
- residual against robust smoothers at several time scales
```

### Map Context

```text
- number of plausible nearby edges and best-versus-second score
- lattice ambiguity
- local graph density to choose lattice width
- road, footway, private, railway, bridge, tunnel, and barrier semantics
- buildings, water, or other implausible locations
- indications that OSM coverage is incomplete
```

Map context is evidence, not truth: pedestrians may walk anywhere and OSM can
be incomplete.

### Time and State

```text
- evidence before, during, and after the event
- separate ON_MAP, OFF_MAP, ABERRATION, and UNCERTAIN states
- asymmetric entry and exit conditions
- merge or split nearby events
- event confidence and point-level reliability
- offline look-ahead before deciding
```

### Height

Z is currently ignored. It may add independent evidence:

```text
- abrupt elevation jumps
- implausible vertical speed or acceleration
- slow elevation drift
- local Z residual against a robust smoother
- start/end elevation mismatch on a closed loop
- consistency between elevation change and horizontal slope
- agreement with a DEM when available
- XY and Z disturbances occurring at the same time
```

XY and Z should not share one threshold. A vertical event may be independent
of the horizontal error. DEM checks must account for bridges, tunnels, and
terrain resolution. Start/end drift may also be barometric rather than GPS.

### Metadata, If It Becomes Available

```text
- reported horizontal or vertical accuracy
- HDOP, VDOP, or PDOP
- fix type, satellite count, and timestamp gaps
- source of altitude, such as GNSS or barometer
```

The current data does not provide enough receiver internals to rely on these.

### Multiple Recordings, for Evaluation

```text
- disagreement between simultaneous devices
- consensus between recordings
- recurrence at the same place on different days
- device-specific failure patterns
```

These signals are useful for labels and diagnostics but are not normally
available when processing one new production recording.

## Likely Future Event Types

```text
- abrupt XY jump
- smooth multipath excursion
- slow XY drift
- real unmapped path
- dropout or interpolated gap
- repeated or frozen positions
- implausible loop or backtrack
- altitude spike
- slow altitude drift
- combined XY/Z aberration
```

Do not implement this taxonomy in advance. Add a type after finding examples
and a useful measurement that separates it.

## Next Sensible Work

1. Preserve family 7 as a known GPS-aberration regression case.
2. Preserve family 4 as a set of real off-map trail cases.
3. Add one new evidence measurement at a time and compare decision rate,
   precision, and recall.
4. Build a labelled event set before considering a learned classifier.
5. Keep correction policy outside the detector: detection says what may have
   happened; map matching decides whether to use OSM, raw GPS, or a blend.
6. Evaluate a mild classical smoother after detector-guided map matching. It
   may remove high-frequency energy introduced by map projection or transition
   blending. Accept it only if it reduces those high bands without increasing
   low-frequency energy or worsening route geometry.
