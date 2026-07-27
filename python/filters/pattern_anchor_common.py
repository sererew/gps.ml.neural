#!/usr/bin/env python3
"""Common GPX and pattern-anchor utilities for experimental filters."""

from __future__ import annotations

from pathlib import Path

import gpxpy
import gpxpy.gpx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import CubicSpline


CHANNEL_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALL_CHANNELS = ("x", "y", "z")


def parse_gpx(gpx_path):
    """Parse a GPX track into a DataFrame with lat, lon, ele, and time columns."""
    print(f"Loading GPX from {gpx_path}...")
    with open(gpx_path, "r", encoding="utf-8") as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(
                    {
                        "lat": point.latitude,
                        "lon": point.longitude,
                        "ele": point.elevation if point.elevation is not None else 0.0,
                        "time": point.time,
                    }
                )

    if not points:
        raise ValueError(f"No trackpoints found in {gpx_path}")

    df = pd.DataFrame(points)
    print(f"Loaded {len(df)} points from {gpx_path}")
    return df


def create_gpx(lat, lon, ele, time=None, output_path="filtered_track.gpx"):
    """Write a GPX track from coordinate arrays."""
    print(f"Creating GPX with {len(lat)} points...")
    gpx = gpxpy.gpx.GPX()
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    for i in range(len(lat)):
        point_time = time[i] if time is not None and i < len(time) and pd.notna(time[i]) else None
        point = gpxpy.gpx.GPXTrackPoint(
            latitude=float(lat[i]),
            longitude=float(lon[i]),
            elevation=float(ele[i]),
            time=point_time,
        )
        gpx_segment.points.append(point)

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(gpx.to_xml())

    print(f"Filtered track saved to {output_path}")


def setup_projection(lat_center, lon_center):
    """Create a WGS84-to-UTM transformer for the given track area."""
    utm_zone = int((lon_center + 180) / 6) + 1
    hemisphere = "north" if lat_center >= 0 else "south"
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    print(f"Using UTM Zone {utm_zone}{hemisphere[0].upper()} projection")
    return transformer


def latlon_to_meters(lat, lon, transformer, lat_ref=None, lon_ref=None):
    """Convert latitude and longitude to projected metric coordinates."""
    x_utm, y_utm = transformer.transform(lon, lat)
    if lat_ref is not None and lon_ref is not None:
        x_ref, y_ref = transformer.transform(lon_ref, lat_ref)
        x_utm = x_utm - x_ref
        y_utm = y_utm - y_ref
    return x_utm, y_utm


def meters_to_latlon(x, y, transformer, lat_ref=None, lon_ref=None):
    """Convert projected metric coordinates back to latitude and longitude."""
    if lat_ref is not None and lon_ref is not None:
        x_ref, y_ref = transformer.transform(lon_ref, lat_ref)
        x = x + x_ref
        y = y + y_ref
    lon, lat = transformer.transform(x, y, direction="INVERSE")
    return lat, lon


def parse_channels(value):
    """Parse a comma-separated channel list."""
    channels = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(channels) - set(ALL_CHANNELS))
    if unknown:
        raise ValueError(f"Unknown anchor channels: {unknown}. Use any of: {', '.join(ALL_CHANNELS)}")
    return channels


def infer_pattern_path(input_gpx):
    """Infer the aligned pattern path from a preprocessed recording path."""
    input_path = Path(input_gpx)
    family = input_path.parent.name
    pattern_path = input_path.parent / f"{family}_aligned_pattern_resampled.gpx"
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found for {input_gpx}: {pattern_path}")
    return pattern_path


def choose_anchor_indices(n_points, duration_s, anchors_per_hour, min_anchors, max_anchors, edge_skip_points=0):
    """Choose anchor indices uniformly over the common track-pattern timeline."""
    if n_points <= 0:
        return np.asarray([], dtype=np.int64)

    first_index = int(max(0, edge_skip_points))
    last_index = int(min(n_points - 1, n_points - 1 - edge_skip_points))
    if last_index <= first_index:
        first_index = 0
        last_index = n_points - 1

    duration_hours = max(float(duration_s) / 3600.0, 0.0)
    anchor_count = int(round(duration_hours * anchors_per_hour))
    anchor_count = max(anchor_count, min_anchors)
    if max_anchors and max_anchors > 0:
        anchor_count = min(anchor_count, max_anchors)
    anchor_count = max(2, min(anchor_count, last_index - first_index + 1))

    return np.unique(np.linspace(first_index, last_index, anchor_count).round().astype(np.int64))


def anchor_values(error, anchor_indices, channel_idx, radius):
    """Return exact or local-mean error values at anchor indices."""
    values = np.zeros(len(anchor_indices), dtype=np.float64)
    for i, idx in enumerate(anchor_indices):
        if radius <= 0:
            values[i] = error[idx, channel_idx]
            continue
        start = max(0, int(idx) - radius)
        end = min(len(error), int(idx) + radius + 1)
        values[i] = float(np.mean(error[start:end, channel_idx]))
    return values


def interpolate_values(indices, values, n_points, interpolation):
    """Interpolate sparse anchor values over all timesteps with clamped edges."""
    if n_points <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(indices) == 0:
        return np.zeros(n_points, dtype=np.float64)
    if len(indices) == 1:
        return np.full(n_points, values[0], dtype=np.float64)

    target = np.arange(n_points, dtype=np.float64)
    indices_float = indices.astype(np.float64)
    clipped_target = np.clip(target, indices_float[0], indices_float[-1])

    if interpolation == "cubic" and len(indices) >= 3:
        spline = CubicSpline(indices_float, values, bc_type="natural")
        curve = spline(clipped_target)
    else:
        curve = np.interp(clipped_target, indices_float, values)

    curve[target < indices_float[0]] = values[0]
    curve[target > indices_float[-1]] = values[-1]
    return curve


def apply_pattern_anchor_correction(
    filtered_df,
    pattern_df,
    anchors_per_hour=8.0,
    min_anchors=8,
    max_anchors=0,
    anchor_error_radius=0,
    anchor_channels="x,y,z",
    anchor_interpolation="cubic",
    anchor_edge_blend_points=30,
    anchor_trim_to_pattern=False,
    anchor_edge_skip_points=0,
):
    """Apply an oracle slow correction using sparse pattern anchors."""
    print("Applying experimental pattern-anchor slow correction...")
    channels = parse_channels(anchor_channels)

    track_indices = None
    pattern_indices = None
    if (
        "time" in filtered_df.columns
        and "time" in pattern_df.columns
        and filtered_df["time"].notna().any()
        and pattern_df["time"].notna().any()
    ):
        track_times = pd.to_datetime(filtered_df["time"], utc=True, errors="coerce")
        pattern_times = pd.to_datetime(pattern_df["time"], utc=True, errors="coerce")
        track_time_df = pd.DataFrame({"time": track_times, "track_idx": np.arange(len(filtered_df))}).dropna()
        pattern_time_df = pd.DataFrame({"time": pattern_times, "pattern_idx": np.arange(len(pattern_df))}).dropna()
        common = pd.merge(track_time_df, pattern_time_df, on="time", how="inner").sort_values("track_idx")
        if len(common) >= 2:
            track_indices = common["track_idx"].to_numpy(dtype=np.int64)
            pattern_indices = common["pattern_idx"].to_numpy(dtype=np.int64)

    if track_indices is None or pattern_indices is None:
        n_common = min(len(filtered_df), len(pattern_df))
        if n_common < 2:
            raise ValueError("Need at least two common points for pattern-anchor correction")
        track_indices = np.arange(n_common, dtype=np.int64)
        pattern_indices = np.arange(n_common, dtype=np.int64)
        print(f"Pattern-anchor alignment: positional fallback with {n_common} common points")
    else:
        print(f"Pattern-anchor alignment: matched {len(track_indices)} points by timestamp")

    combined_lat = pd.concat([filtered_df["lat"], pattern_df["lat"]])
    combined_lon = pd.concat([filtered_df["lon"], pattern_df["lon"]])
    transformer = setup_projection(combined_lat.mean(), combined_lon.mean())
    lat_ref = filtered_df["lat"].iloc[0]
    lon_ref = filtered_df["lon"].iloc[0]

    x_filt, y_filt = latlon_to_meters(filtered_df["lat"], filtered_df["lon"], transformer, lat_ref, lon_ref)
    z_filt = filtered_df["ele"].to_numpy(dtype=np.float64)

    x_pattern, y_pattern = latlon_to_meters(pattern_df["lat"], pattern_df["lon"], transformer, lat_ref, lon_ref)
    z_pattern = pattern_df["ele"].to_numpy(dtype=np.float64)

    filtered_pos = np.column_stack([x_filt, y_filt, z_filt])
    pattern_pos = np.column_stack([x_pattern, y_pattern, z_pattern])
    matched_error = filtered_pos[track_indices] - pattern_pos[pattern_indices]

    if "time" in filtered_df.columns and filtered_df["time"].notna().any():
        t0 = filtered_df["time"].iloc[int(track_indices[0])]
        t1 = filtered_df["time"].iloc[int(track_indices[-1])]
        try:
            duration_s = max((t1 - t0).total_seconds(), 0.0)
        except Exception:
            duration_s = float(len(track_indices) - 1)
    else:
        duration_s = float(len(track_indices) - 1)

    anchor_pair_indices = choose_anchor_indices(
        len(track_indices),
        duration_s,
        anchors_per_hour,
        min_anchors,
        max_anchors,
        anchor_edge_skip_points,
    )
    anchor_track_indices = track_indices[anchor_pair_indices]
    print(
        f"Pattern anchors: {len(anchor_track_indices)} anchors over {duration_s / 3600.0:.2f} h "
        f"({anchors_per_hour:g}/h, min {min_anchors})"
    )

    correction = np.zeros((len(filtered_df), 3), dtype=np.float64)
    for channel in channels:
        channel_idx = CHANNEL_TO_INDEX[channel]
        values = anchor_values(matched_error, anchor_pair_indices, channel_idx, anchor_error_radius)
        curve = interpolate_values(anchor_track_indices, values, len(filtered_df), anchor_interpolation)
        first_anchor = int(anchor_track_indices[0])
        if first_anchor > 0:
            curve[:first_anchor] = 0.0

        if anchor_edge_blend_points > 0:
            blend_end = min(len(curve), first_anchor + int(anchor_edge_blend_points) + 1)
            if blend_end > first_anchor:
                weights = np.linspace(0.0, 1.0, blend_end - first_anchor)
                curve[first_anchor:blend_end] *= weights
        correction[:, channel_idx] = curve

    corrected_pos = filtered_pos - correction
    lat_corr, lon_corr = meters_to_latlon(corrected_pos[:, 0], corrected_pos[:, 1], transformer, lat_ref, lon_ref)

    corrected_df = filtered_df.copy()
    corrected_df["lat"] = lat_corr
    corrected_df["lon"] = lon_corr
    corrected_df["ele"] = corrected_pos[:, 2]
    if anchor_trim_to_pattern:
        corrected_df = corrected_df.iloc[track_indices].copy().reset_index(drop=True)
    return corrected_df
