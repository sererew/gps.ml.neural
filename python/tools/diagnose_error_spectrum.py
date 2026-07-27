#!/usr/bin/env python3
"""
Diagnose the frequency content of track error against the reference pattern.

This is a spike tool. It compares raw resampled recordings and, when present,
filtered GPX outputs from results/filtered/<filter>/<pass>/ against the aligned
resampled pattern for one pass. It writes spectrogram plots and band-power CSVs.
The XY channel is the combined horizontal power PSD(X) + PSD(Y), not the PSD of
the positive XY norm.
"""

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import spectrogram

try:
    from pyproj import Transformer
except ImportError:
    Transformer = None


CHANNELS = ("x", "y", "z", "xy")
DEFAULT_BANDS = "0:0.005,0.005:0.02,0.02:0.10,0.10:0.25,0.25:0.50"


class LocalMetricProjection:
    """Small-area equirectangular projection fallback when pyproj is unavailable."""

    def __init__(self, ref_lat, ref_lon):
        self.ref_lat_rad = math.radians(ref_lat)
        self.ref_lon = ref_lon
        self.ref_lat = ref_lat
        self.earth_radius_m = 6371008.8

    def transform(self, lon_values, lat_values):
        lon_values = np.asarray(lon_values, dtype=np.float64)
        lat_values = np.asarray(lat_values, dtype=np.float64)
        x_values = (
            np.radians(lon_values - self.ref_lon)
            * math.cos(self.ref_lat_rad)
            * self.earth_radius_m
        )
        y_values = np.radians(lat_values - self.ref_lat) * self.earth_radius_m
        return x_values, y_values


def parse_gpx(gpx_path):
    """Read a GPX track into a DataFrame with lat, lon, ele, and time columns."""
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    namespace = {"gpx": "http://www.topografix.com/GPX/1/1"}
    if root.tag.startswith("{"):
        namespace = {"gpx": root.tag.split("}")[0][1:]}

    rows = []
    for point in root.findall(".//gpx:trkpt", namespace):
        ele_element = point.find("gpx:ele", namespace)
        time_element = point.find("gpx:time", namespace)
        rows.append(
            {
                "lat": float(point.get("lat")),
                "lon": float(point.get("lon")),
                "ele": float(ele_element.text) if ele_element is not None else 0.0,
                "time": time_element.text if time_element is not None else None,
            }
        )

    if not rows:
        raise ValueError(f"No track points found in {gpx_path}")

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    return df.dropna(subset=["time"]).reset_index(drop=True)


def setup_projection(lat_values, lon_values):
    """Create a local UTM projection centered on the data."""
    ref_lat = float(np.nanmean(lat_values))
    ref_lon = float(np.nanmean(lon_values))
    if Transformer is None:
        return LocalMetricProjection(ref_lat, ref_lon)
    zone = int((ref_lon + 180.0) / 6.0) + 1
    epsg = 32600 + zone if ref_lat >= 0.0 else 32700 + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def elapsed_seconds(times, origin):
    """Convert timestamps to seconds from a common origin."""
    return (times - origin).dt.total_seconds().to_numpy(dtype=np.float64)


def cumulative_distance_xy(x_values, y_values):
    """Compute cumulative horizontal distance in meters."""
    if len(x_values) == 0:
        return np.asarray([], dtype=np.float64)
    if len(x_values) == 1:
        return np.asarray([0.0], dtype=np.float64)
    dx = np.diff(x_values)
    dy = np.diff(y_values)
    steps = np.sqrt(dx * dx + dy * dy)
    return np.concatenate([[0.0], np.cumsum(steps)])


def align_track_to_pattern(pattern_df, track_df):
    """
    Project pattern and track to meters and interpolate the track to pattern times.

    Returns a dict with time, distance, and signed error channels.
    """
    common_start = max(pattern_df["time"].min(), track_df["time"].min())
    common_end = min(pattern_df["time"].max(), track_df["time"].max())
    if common_end <= common_start:
        raise ValueError("No common time range between pattern and track")

    pattern_common = pattern_df[(pattern_df["time"] >= common_start) & (pattern_df["time"] <= common_end)].copy()
    track_common = track_df[(track_df["time"] >= common_start) & (track_df["time"] <= common_end)].copy()
    if len(pattern_common) < 4 or len(track_common) < 4:
        raise ValueError("Not enough common points for spectral analysis")

    lat_values = np.concatenate([pattern_common["lat"].to_numpy(), track_common["lat"].to_numpy()])
    lon_values = np.concatenate([pattern_common["lon"].to_numpy(), track_common["lon"].to_numpy()])
    transformer = setup_projection(lat_values, lon_values)

    pattern_x, pattern_y = transformer.transform(
        pattern_common["lon"].to_numpy(dtype=np.float64),
        pattern_common["lat"].to_numpy(dtype=np.float64),
    )
    track_x, track_y = transformer.transform(
        track_common["lon"].to_numpy(dtype=np.float64),
        track_common["lat"].to_numpy(dtype=np.float64),
    )

    origin = common_start
    pattern_t = elapsed_seconds(pattern_common["time"], origin)
    track_t = elapsed_seconds(track_common["time"], origin)

    interp_x = np.interp(pattern_t, track_t, track_x)
    interp_y = np.interp(pattern_t, track_t, track_y)
    interp_z = np.interp(pattern_t, track_t, track_common["ele"].to_numpy(dtype=np.float64))
    pattern_z = pattern_common["ele"].to_numpy(dtype=np.float64)

    error_x = interp_x - pattern_x
    error_y = interp_y - pattern_y
    error_z = interp_z - pattern_z

    return {
        "time_seconds": pattern_t,
        "distance_m": cumulative_distance_xy(pattern_x, pattern_y),
        "errors": {
            "x": error_x,
            "y": error_y,
            "z": error_z,
        },
        "n_points": int(len(pattern_t)),
        "common_start": str(common_start),
        "common_end": str(common_end),
    }


def parse_bands(value):
    """Parse comma-separated frequency bands such as 0:0.005,0.005:0.02."""
    bands = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        low_text, high_text = item.split(":", maxsplit=1)
        low = float(low_text)
        high = float(high_text)
        if low < 0.0 or high <= low:
            raise ValueError(f"Invalid frequency band: {item}")
        bands.append((low, high))
    if not bands:
        raise ValueError("At least one frequency band is required")
    return bands


def compute_spectrogram(values, sample_rate, nperseg, overlap):
    """Compute a PSD spectrogram with no detrending."""
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0)
    effective_nperseg = min(int(nperseg), len(values))
    if effective_nperseg < 8:
        raise ValueError("Signal is too short for spectrogram")
    noverlap = int(round(effective_nperseg * overlap))
    noverlap = max(0, min(noverlap, effective_nperseg - 1))

    freqs, times, power = spectrogram(
        values,
        fs=sample_rate,
        window="hann",
        nperseg=effective_nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    return freqs, times, power


def summarize_bands(freqs, power, bands):
    """Summarize spectrogram power by frequency bands."""
    if len(freqs) > 1:
        df = float(np.median(np.diff(freqs)))
    else:
        df = 1.0

    total_power = float(np.nanmean(np.sum(power, axis=0) * df))
    rows = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high if high < freqs.max() else freqs <= high)
        band_power = float(np.nanmean(np.sum(power[mask, :], axis=0) * df)) if np.any(mask) else 0.0
        rows.append(
            {
                "f_low_hz": low,
                "f_high_hz": high,
                "band_power": band_power,
                "total_power": total_power,
                "band_pct": band_power / total_power * 100.0 if total_power > 0.0 else 0.0,
            }
        )
    return rows


def safe_name(value):
    """Return a filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def find_pattern(preprocessed_dir, pasada):
    """Find the aligned resampled pattern for one pass."""
    path = preprocessed_dir / pasada / f"{pasada}_aligned_pattern_resampled.gpx"
    if not path.exists():
        raise FileNotFoundError(f"Pattern not found: {path}")
    return path


def find_raw_recordings(preprocessed_dir, pasada):
    """Find raw resampled recordings for one pass."""
    pass_dir = preprocessed_dir / pasada
    recordings = {}
    for path in sorted(pass_dir.glob("*_resampled.gpx")):
        name = path.name
        if "pattern" in name or "aligned_pattern" in name:
            continue
        recordings[path.stem] = path
    return recordings


def filtered_base_name(path, filter_name):
    """Recover the raw recording stem from a filtered GPX filename."""
    suffix = f"_{filter_name}_filtered"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def find_filtered_recordings(filtered_dir, pasada, filters):
    """Find filtered GPX outputs for the selected pass and filters."""
    by_filter = {}
    if not filtered_dir.exists():
        return by_filter

    for filter_name in filters:
        filter_pass_dir = filtered_dir / filter_name / pasada
        if not filter_pass_dir.exists():
            continue
        rows = {}
        for path in sorted(filter_pass_dir.glob("*_filtered.gpx")):
            rows[filtered_base_name(path, filter_name)] = path
        if rows:
            by_filter[filter_name] = rows
    return by_filter


def select_recordings(raw_recordings, selected):
    """Select recordings by exact stem, comma list, or all when empty."""
    if not selected:
        return raw_recordings
    wanted = {item.strip() for item in selected.split(",") if item.strip()}
    missing = sorted(wanted - set(raw_recordings.keys()))
    if missing:
        raise ValueError(f"Unknown recordings: {missing}")
    return {name: raw_recordings[name] for name in sorted(wanted)}


def build_comparisons(raw_recordings, filtered_recordings, include_raw):
    """Build comparable variants per recording."""
    comparisons = {}
    for recording, raw_path in raw_recordings.items():
        variants = {}
        if include_raw:
            variants["raw"] = raw_path
        for filter_name, rows in filtered_recordings.items():
            if recording in rows:
                variants[filter_name] = rows[recording]
        comparisons[recording] = variants
    return comparisons


def plot_recording(recording, analyses, x_axis, output_path, clip_percentiles=None, title_suffix=""):
    """Plot one recording as rows=variants and columns=channels."""
    if not analyses:
        return

    variants = list(analyses.keys())
    n_rows = len(variants)
    n_cols = len(CHANNELS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, max(2.8, 2.35 * n_rows)),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )

    values = []
    for item in analyses.values():
        for channel in CHANNELS:
            power = item["spectra"][channel]["power"]
            values.append(np.log10(power + 1e-12))
    merged = np.concatenate([value.ravel() for value in values])
    if clip_percentiles is None:
        vmin = float(np.nanmin(merged))
        vmax = float(np.nanmax(merged))
        scale_note = "absolute scale"
    else:
        low, high = clip_percentiles
        vmin = float(np.nanpercentile(merged, low))
        vmax = float(np.nanpercentile(merged, high))
        scale_note = f"{low:g}-{high:g} percentile scale"

    for row_idx, variant in enumerate(variants):
        item = analyses[variant]
        for col_idx, channel in enumerate(CHANNELS):
            ax = axes[row_idx][col_idx]
            spec = item["spectra"][channel]
            x_values = spec["times"] / 60.0
            x_label = "Time (min)"
            if x_axis == "distance":
                x_values = np.interp(spec["times"], item["time_seconds"], item["distance_m"]) / 1000.0
                x_label = "Distance (km)"

            log_power = np.log10(spec["power"] + 1e-12)
            mesh = ax.pcolormesh(
                x_values,
                spec["freqs"],
                log_power,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            if row_idx == 0:
                ax.set_title(channel.upper())
            if col_idx == 0:
                ax.set_ylabel(f"{variant}\nFrequency (Hz)")
            if row_idx == n_rows - 1:
                ax.set_xlabel(x_label)
            ax.set_ylim(0.0, 0.5)
            ax.grid(False)

        cbar = fig.colorbar(mesh, ax=axes[row_idx, :], fraction=0.015, pad=0.01)
        cbar.set_label("log10 PSD")

    fig.suptitle(f"Error spectrum - {recording} - {scale_note}{title_suffix}", y=0.995)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_raw_ratios(summary_df):
    """Add ratio-to-raw columns for each recording/channel/band."""
    raw = summary_df[summary_df["variant"] == "raw"][
        ["recording", "channel", "band_label", "band_power"]
    ].rename(columns={"band_power": "raw_band_power"})
    merged = summary_df.merge(raw, on=["recording", "channel", "band_label"], how="left")
    merged["ratio_to_raw"] = np.where(
        merged["raw_band_power"] > 0.0,
        merged["band_power"] / merged["raw_band_power"],
        np.nan,
    )
    return merged


def plot_band_ratio_summary(summary_df, output_path, channels=("xy", "z")):
    """Plot band-power ratios and total residual error energy by channel."""
    ratio_col = "ratio_to_raw" if "ratio_to_raw" in summary_df.columns else "mean_ratio_to_raw"
    pct_col = "band_pct" if "band_pct" in summary_df.columns else "mean_band_pct"
    total_power_col = "total_power" if "total_power" in summary_df.columns else "mean_total_power"
    all_rows = summary_df[summary_df["channel"].isin(channels)].copy()
    rows = summary_df[
        (summary_df["variant"] != "raw")
        & (summary_df["channel"].isin(channels))
        & summary_df[ratio_col].notna()
    ].copy()
    if rows.empty:
        return

    label_aliases = {
        "moving_average": "mov_avg",
        "triangular_weighted": "triangular",
        "gaussian_pattern_anchor": "gaussian+anchor",
        "moving_average_pattern_anchor": "mov_avg+anchor",
        "nn_pattern_anchor": "nn+anchor",
    }
    variants = list(dict.fromkeys(rows["variant"].tolist()))
    bands = list(dict.fromkeys(all_rows["band_label"].tolist()))
    columns = bands + ["total\nerror %"]
    row_labels = ["raw energy %"] + [label_aliases.get(variant, variant) for variant in variants]
    fig, axes = plt.subplots(
        1,
        len(channels),
        figsize=(4.6 * len(channels), max(3.5, 0.45 * len(row_labels))),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#f2f2f2")

    def format_percent(value):
        if abs(value) >= 1000.0:
            return f"{value / 1000.0:.1f}k"
        return f"{value:.1f}"

    for ax, channel in zip(axes[0], channels):
        matrix = np.full((len(row_labels), len(columns)), np.nan, dtype=np.float64)
        channel_rows = rows[rows["channel"] == channel]
        raw_rows = all_rows[(all_rows["variant"] == "raw") & (all_rows["channel"] == channel)]
        raw_pct_values = np.full(len(bands), np.nan, dtype=np.float64)
        total_error_pct_values = np.full(len(row_labels), np.nan, dtype=np.float64)
        raw_total_power = raw_rows[total_power_col].mean()
        if np.isfinite(raw_total_power) and raw_total_power > 0.0:
            total_error_pct_values[0] = 100.0
            matrix[0, -1] = 4.0
        for j, band in enumerate(bands):
            subset = raw_rows[raw_rows["band_label"] == band]
            if not subset.empty:
                raw_pct = float(subset[pct_col].mean())
                raw_pct_values[j] = raw_pct
                matrix[0, j] = -1.0 + 5.0 * np.clip(raw_pct / 100.0, 0.0, 1.0)
        for i, variant in enumerate(variants):
            variant_rows = channel_rows[channel_rows["variant"] == variant]
            variant_total_power = variant_rows[total_power_col].mean()
            if np.isfinite(raw_total_power) and raw_total_power > 0.0 and np.isfinite(variant_total_power):
                total_error_pct = 100.0 * variant_total_power / raw_total_power
                total_error_pct_values[i + 1] = total_error_pct
                matrix[i + 1, -1] = -1.0 + 5.0 * np.clip(total_error_pct / 100.0, 0.0, 1.0)
            for j, band in enumerate(bands):
                subset = channel_rows[
                    (channel_rows["variant"] == variant)
                    & (channel_rows["band_label"] == band)
                ]
                if not subset.empty:
                    ratio = float(subset[ratio_col].mean())
                    matrix[i + 1, j] = np.log10(max(ratio, 1e-12))

        mesh = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-1.0, vmax=4.0)
        ax.set_title(channel.upper())
        ax.set_xticks(np.arange(len(columns)))
        ax.set_xticklabels(columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for j, raw_pct in enumerate(raw_pct_values):
            if np.isfinite(raw_pct):
                text_color = "white" if matrix[0, j] < -0.6 or matrix[0, j] > 3.2 else "black"
                ax.text(j, 0, f"{raw_pct:.1f}", ha="center", va="center", fontsize=8, color=text_color)
        for i in range(1, len(row_labels)):
            for j in range(len(bands)):
                value = matrix[i, j]
                if np.isfinite(value):
                    text_color = "white" if value < -0.6 or value > 3.2 else "black"
                    ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color=text_color)
        for i, total_error_pct in enumerate(total_error_pct_values):
            if np.isfinite(total_error_pct):
                value = matrix[i, -1]
                text_color = "white" if value < -0.6 or value > 3.2 else "black"
                ax.text(
                    len(columns) - 1,
                    i,
                    format_percent(total_error_pct),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )

    cbar = fig.colorbar(mesh, ax=axes[0, :], fraction=0.04, pad=0.03)
    cbar.set_label("Band cells: log10 ratio to raw; percent cells: 0 blue, 100 red")
    fig.suptitle("Band-power ratio to raw and residual total error")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectral diagnostic of GPS track error against the aligned pattern",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pasada", default="17", help="Pass/family to analyze")
    parser.add_argument(
        "--recordings",
        default="17_11_13_J_B_resampled",
        help="Comma-separated raw recording stems. Empty means all recordings in the pass.",
    )
    parser.add_argument("--preprocessed_dir", default="data/preprocessed", help="Preprocessed data directory")
    parser.add_argument("--filtered_dir", default="results/filtered", help="Filtered GPX directory")
    parser.add_argument(
        "--filters",
        default="identity,gaussian,kalman,nn,nn_pattern_anchor",
        help="Comma-separated filtered outputs to compare when present",
    )
    parser.add_argument("--no_raw", action="store_true", help="Do not include raw recordings")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="Sampling rate in Hz")
    parser.add_argument("--nperseg", type=int, default=1024, help="STFT window size in samples")
    parser.add_argument("--overlap", type=float, default=0.75, help="STFT overlap fraction")
    parser.add_argument("--bands", default=DEFAULT_BANDS, help="Frequency bands as low:high comma list")
    parser.add_argument("--x_axis", choices=["time", "distance"], default="time", help="Plot X axis")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path.cwd()
    preprocessed_dir = repo_root / args.preprocessed_dir
    filtered_dir = repo_root / args.filtered_dir
    output_dir = Path(
        args.output_dir or repo_root / "results" / "diagnostics" / f"error_spectrum_{args.pasada}"
    )
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    bands = parse_bands(args.bands)
    filter_names = [item.strip() for item in args.filters.split(",") if item.strip()]

    print("=== ERROR SPECTRUM DIAGNOSTIC ===")
    print(f"Pass: {args.pasada}")
    print(f"Recordings: {args.recordings or 'all'}")
    print(f"Filters: {filter_names}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"STFT window: {args.nperseg} samples")
    print(f"Output: {output_dir}")

    pattern_path = find_pattern(preprocessed_dir, args.pasada)
    raw_recordings = select_recordings(find_raw_recordings(preprocessed_dir, args.pasada), args.recordings)
    filtered_recordings = find_filtered_recordings(filtered_dir, args.pasada, filter_names)
    comparisons = build_comparisons(raw_recordings, filtered_recordings, include_raw=not args.no_raw)

    pattern_df = parse_gpx(pattern_path)
    summary_rows = []
    metadata = {
        "pasada": args.pasada,
        "pattern": str(pattern_path),
        "sample_rate_hz": args.sample_rate,
        "nperseg": args.nperseg,
        "overlap": args.overlap,
        "bands": [{"f_low_hz": low, "f_high_hz": high} for low, high in bands],
        "recordings": {},
        "missing_filters": {},
    }

    for recording, variants in comparisons.items():
        print(f"\nRecording: {recording}")
        analyses = {}
        present_variants = sorted(variants.keys())
        metadata["recordings"][recording] = present_variants
        metadata["missing_filters"][recording] = [
            name for name in filter_names if name not in present_variants
        ]
        if metadata["missing_filters"][recording]:
            print(f"  Missing filtered outputs: {metadata['missing_filters'][recording]}")

        for variant, path in variants.items():
            try:
                aligned = align_track_to_pattern(pattern_df, parse_gpx(path))
            except Exception as exc:
                print(f"  {variant}: skipped ({exc})")
                continue

            spectra = {}
            for channel in ("x", "y", "z"):
                freqs, times, power = compute_spectrogram(
                    aligned["errors"][channel],
                    args.sample_rate,
                    args.nperseg,
                    args.overlap,
                )
                spectra[channel] = {"freqs": freqs, "times": times, "power": power}

            spectra["xy"] = {
                "freqs": spectra["x"]["freqs"],
                "times": spectra["x"]["times"],
                "power": spectra["x"]["power"] + spectra["y"]["power"],
            }

            for channel in CHANNELS:
                spec = spectra[channel]
                for band_row in summarize_bands(spec["freqs"], spec["power"], bands):
                    band_label = f"{band_row['f_low_hz']:.6g}-{band_row['f_high_hz']:.6g}"
                    summary_rows.append(
                        {
                            "recording": recording,
                            "variant": variant,
                            "channel": channel,
                            "band_label": band_label,
                            **band_row,
                            "n_points": aligned["n_points"],
                            "common_start": aligned["common_start"],
                            "common_end": aligned["common_end"],
                        }
                    )

            analyses[variant] = {
                "spectra": spectra,
                "time_seconds": aligned["time_seconds"],
                "distance_m": aligned["distance_m"],
            }
            print(f"  {variant}: {aligned['n_points']} common points")

        if analyses:
            plot_path = plots_dir / f"{safe_name(recording)}_spectrogram.png"
            detail_plot_path = plots_dir / f"{safe_name(recording)}_spectrogram_detail.png"
            plot_recording(recording, analyses, args.x_axis, plot_path)
            plot_recording(
                recording,
                analyses,
                args.x_axis,
                detail_plot_path,
                clip_percentiles=(2, 98),
                title_suffix=" (detail)",
            )
            print(f"  Plot: {plot_path}")
            print(f"  Detail plot: {detail_plot_path}")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = add_raw_ratios(summary_df)
        summary_df.to_csv(output_dir / "band_power.csv", index=False)

        aggregate = (
            summary_df.groupby(["variant", "channel", "band_label", "f_low_hz", "f_high_hz"], dropna=False)
            .agg(
                recordings=("recording", "nunique"),
                mean_band_power=("band_power", "mean"),
                mean_total_power=("total_power", "mean"),
                mean_band_pct=("band_pct", "mean"),
                mean_ratio_to_raw=("ratio_to_raw", "mean"),
            )
            .reset_index()
        )
        aggregate.to_csv(output_dir / "band_power_summary.csv", index=False)
        plot_band_ratio_summary(aggregate, output_dir / "band_ratio_to_raw.png")

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== DONE ===")
    print(f"Band power CSV: {output_dir / 'band_power.csv'}")
    print(f"Band summary CSV: {output_dir / 'band_power_summary.csv'}")
    print(f"Band ratio plot: {output_dir / 'band_ratio_to_raw.png'}")
    print(f"Metadata: {output_dir / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
