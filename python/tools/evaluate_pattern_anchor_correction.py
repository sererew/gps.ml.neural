#!/usr/bin/env python3
"""
Evaluate slow-drift correction from sparse pattern-derived anchors.

This is a diagnostic upper bound for a future workflow where a user provides
real anchor waypoints. Here, anchors are sampled automatically from the clean
pattern at known timesteps, so the experiment measures how much sparse anchors
can correct v3 accumulated drift.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.interpolate import CubicSpline

from diagnose_parametric_slow_drift import (
    FEATURES,
    aggregate_recording,
    denormalize_deltas,
    length_xy,
    load_manifest,
    load_norm_stats,
    load_split_arrays,
    load_v3_model,
    moving_average,
)


CHANNEL_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALL_CHANNELS = ("x", "y", "z")


def parse_channels(value):
    """Parse a comma-separated channel list."""
    channels = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(channels) - set(ALL_CHANNELS))
    if unknown:
        raise ValueError(f"Unknown channels: {unknown}. Use any of: {', '.join(ALL_CHANNELS)}")
    return channels


def rms(values):
    """Return root mean square."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))


def metrics_for_positions(pos_pred, pos_true):
    """Compute accumulated-position metrics."""
    if len(pos_pred) == 0:
        return {
            "n_points": 0,
            "duration_s": 0.0,
            "rms_x_m": 0.0,
            "rms_y_m": 0.0,
            "rms_z_m": 0.0,
            "rms_xy_m": 0.0,
            "rms_xyz_m": 0.0,
            "final_abs_x_m": 0.0,
            "final_abs_y_m": 0.0,
            "final_abs_z_m": 0.0,
            "final_xy_m": 0.0,
            "final_xyz_m": 0.0,
            "length_diff_xy_m": 0.0,
            "true_length_xy_m": 0.0,
        }

    error = pos_pred - pos_true
    xy = np.linalg.norm(error[:, :2], axis=1)
    xyz = np.linalg.norm(error, axis=1)
    final = error[-1]
    true_length = length_xy(pos_true)
    return {
        "n_points": int(len(pos_pred)),
        "rms_x_m": rms(error[:, 0]),
        "rms_y_m": rms(error[:, 1]),
        "rms_z_m": rms(error[:, 2]),
        "rms_xy_m": rms(xy),
        "rms_xyz_m": rms(xyz),
        "final_abs_x_m": float(abs(final[0])),
        "final_abs_y_m": float(abs(final[1])),
        "final_abs_z_m": float(abs(final[2])),
        "final_xy_m": float(np.linalg.norm(final[:2])),
        "final_xyz_m": float(np.linalg.norm(final)),
        "length_diff_xy_m": abs(length_xy(pos_pred) - true_length),
        "true_length_xy_m": true_length,
    }


def aggregate_metric_rows(rows, method):
    """Aggregate per-recording metrics for one method."""
    subset = [row for row in rows if row["method"] == method]
    if not subset:
        return {}

    total_points = sum(row["n_points"] for row in subset)

    def weighted_rms(field):
        if total_points <= 0:
            return 0.0
        energy = sum((row[field] ** 2) * row["n_points"] for row in subset)
        return float(np.sqrt(energy / total_points))

    mean_true_length = float(np.mean([row["true_length_xy_m"] for row in subset]))
    result = {
        "recordings": len(subset),
        "n_points": int(total_points),
        "mean_anchor_count": float(np.mean([row.get("anchor_count", 0) for row in subset])),
        "mean_duration_hours": float(np.mean([row.get("duration_hours", 0.0) for row in subset])),
        "mean_true_length_xy_m": mean_true_length,
        "mean_length_diff_xy_m": float(np.mean([row["length_diff_xy_m"] for row in subset])),
        "mean_final_xy_m": float(np.mean([row["final_xy_m"] for row in subset])),
        "mean_final_xyz_m": float(np.mean([row["final_xyz_m"] for row in subset])),
    }
    for field in ["rms_x_m", "rms_y_m", "rms_z_m", "rms_xy_m", "rms_xyz_m"]:
        result[field] = weighted_rms(field)
    result["rms_xy_pct"] = result["rms_xy_m"] / mean_true_length * 100.0 if mean_true_length > 0 else 0.0
    result["length_diff_pct"] = (
        result["mean_length_diff_xy_m"] / mean_true_length * 100.0 if mean_true_length > 0 else 0.0
    )
    return result


def choose_anchor_indices(n_points, duration_s, anchors_per_hour, min_anchors, max_anchors):
    """Choose anchor indices uniformly over the recording timeline."""
    if n_points <= 0:
        return np.asarray([], dtype=np.int64)

    duration_hours = max(float(duration_s) / 3600.0, 0.0)
    anchor_count = int(round(duration_hours * anchors_per_hour))
    anchor_count = max(anchor_count, min_anchors)
    if max_anchors and max_anchors > 0:
        anchor_count = min(anchor_count, max_anchors)
    anchor_count = max(2, min(anchor_count, n_points))

    return np.unique(np.linspace(0, n_points - 1, anchor_count).round().astype(np.int64))


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
    """Interpolate sparse values over all timesteps."""
    if n_points <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(indices) == 0:
        return np.zeros(n_points, dtype=np.float64)
    if len(indices) == 1:
        return np.full(n_points, values[0], dtype=np.float64)

    target = np.arange(n_points, dtype=np.float64)
    if interpolation == "cubic" and len(indices) >= 3:
        spline = CubicSpline(indices.astype(np.float64), values, bc_type="natural")
        return spline(target)
    return np.interp(target, indices.astype(np.float64), values)


def build_anchor_correction(error, anchor_indices, channels, interpolation, anchor_error_radius):
    """Build a dense slow-error correction curve from sparse anchors."""
    correction = np.zeros_like(error, dtype=np.float64)
    anchor_rows = []

    for channel in channels:
        channel_idx = CHANNEL_TO_INDEX[channel]
        values = anchor_values(error, anchor_indices, channel_idx, anchor_error_radius)
        correction[:, channel_idx] = interpolate_values(anchor_indices, values, len(error), interpolation)
        for anchor_order, (idx, value) in enumerate(zip(anchor_indices, values)):
            anchor_rows.append(
                {
                    "anchor_index": int(anchor_order),
                    "sample_index": int(idx),
                    "t_norm": float(idx / max(len(error) - 1, 1)),
                    "channel": channel,
                    "anchor_error_m": float(value),
                }
            )

    return correction, anchor_rows


def write_csv(path, rows, fieldnames):
    """Write rows to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(value):
    """Return a filename-safe string."""
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def plot_recording(output_path, recording, pos_clean, method_positions, anchor_indices, channels):
    """Plot per-recording residual errors and XY trajectories."""
    time = np.arange(len(pos_clean), dtype=np.float64)
    method_errors = {name: positions - pos_clean for name, positions in method_positions.items()}

    fig, axes = plt.subplots(2 + len(channels), 1, figsize=(13, 4 + 2.6 * len(channels)), sharex=False)

    ax = axes[0]
    for name, error in method_errors.items():
        ax.plot(time, np.linalg.norm(error[:, :2], axis=1), label=name)
    for idx in anchor_indices:
        ax.axvline(idx, color="black", alpha=0.12, linewidth=0.8)
    ax.set_title(f"Anchor correction residuals - {recording}")
    ax.set_ylabel("XY error (m)")
    ax.grid(True)
    ax.legend()

    for axis_offset, channel in enumerate(channels, start=1):
        channel_idx = CHANNEL_TO_INDEX[channel]
        ax = axes[axis_offset]
        for name, error in method_errors.items():
            ax.plot(time, error[:, channel_idx], label=name)
        ax.scatter(
            anchor_indices,
            method_errors["baseline"][anchor_indices, channel_idx],
            s=18,
            color="black",
            label="anchors",
            zorder=5,
        )
        ax.axhline(0.0, color="black", alpha=0.25, linewidth=0.8)
        ax.set_ylabel(f"{channel.upper()} error (m)")
        ax.grid(True)
        ax.legend()

    ax = axes[-1]
    ax.plot(pos_clean[:, 0], pos_clean[:, 1], label="pattern", color="black", linewidth=2.0)
    ax.plot(method_positions["baseline"][:, 0], method_positions["baseline"][:, 1], label="baseline", alpha=0.75)
    ax.plot(
        method_positions["pattern_anchor"][:, 0],
        method_positions["pattern_anchor"][:, 1],
        label="pattern_anchor",
        alpha=0.9,
    )
    ax.scatter(pos_clean[anchor_indices, 0], pos_clean[anchor_indices, 1], s=22, color="black", label="anchors")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(output_path, summary, methods):
    """Plot global summary metrics by method."""
    metric_names = ["rms_xy_m", "rms_xyz_m", "mean_final_xy_m", "mean_length_diff_xy_m"]
    titles = ["RMS XY", "RMS XYZ", "Mean final XY", "Mean length diff XY"]
    values = [[summary["methods"][method][metric] for method in methods] for metric in metric_names]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    x = np.arange(len(methods))
    for ax, title, metric_values in zip(axes, titles, values):
        ax.bar(x, metric_values, color=["#8f8f8f", "#4c78a8", "#59a14f"])
        ax.set_title(title)
        ax.set_ylabel("Meters")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate sparse pattern-anchor slow correction for v3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--v3_model", default="models/model_best_v3.keras", help="Frozen v3 model path")
    parser.add_argument("--smooth_window", type=int, default=1800, help="Moving-average oracle window")
    parser.add_argument("--anchors_per_hour", type=float, default=8.0, help="Anchor density per recording hour")
    parser.add_argument("--min_anchors", type=int, default=8, help="Minimum anchors per recording")
    parser.add_argument("--max_anchors", type=int, default=0, help="Maximum anchors per recording; 0 means no cap")
    parser.add_argument("--anchor_error_radius", type=int, default=0, help="Local radius for anchor error averaging")
    parser.add_argument("--channels", default="x,y,z", help="Comma-separated channels to correct")
    parser.add_argument(
        "--interpolation",
        default="cubic",
        choices=["linear", "cubic"],
        help="Interpolation mode between anchors",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="v3 prediction batch size")
    parser.add_argument("--plots", type=int, default=5, help="Number of worst baseline recordings to plot")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    channels = parse_channels(args.channels)
    output_dir = Path(
        args.output_dir
        or (
            f"results/diagnostics/pattern_anchor_correction_v3_{args.split}"
            f"_{args.anchors_per_hour:g}perhour_min{args.min_anchors}_{args.interpolation}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== PATTERN ANCHOR CORRECTION DIAGNOSTIC ===")
    print(f"Split: {args.split}")
    print(f"Channels: {channels}")
    print(f"Anchors per hour: {args.anchors_per_hour:g}")
    print(f"Minimum anchors: {args.min_anchors}")
    print(f"Maximum anchors: {args.max_anchors if args.max_anchors else 'none'}")
    print(f"Anchor error radius: {args.anchor_error_radius}")
    print(f"Interpolation: {args.interpolation}")
    print(f"Smooth window oracle: {args.smooth_window}")

    manifest = load_manifest(args.data_root, args.split)
    norm_stats = load_norm_stats(args.data_root)
    x_norm, y_norm, masks = load_split_arrays(manifest)

    sequence_length = x_norm.shape[1]
    v3_model = load_v3_model(args.v3_model, sequence_length, len(FEATURES))
    residual_pred = v3_model.predict(x_norm, batch_size=args.batch_size, verbose=1)
    filtered_norm = x_norm + residual_pred
    filtered_meters = denormalize_deltas(filtered_norm, norm_stats)
    clean_meters = denormalize_deltas(y_norm, norm_stats)

    rows = []
    anchor_rows = []
    plot_candidates = []
    grouped = manifest.reset_index().sort_values(["grabacion", "t_start"]).groupby("grabacion", sort=False)

    for recording, group in grouped:
        rec = aggregate_recording(group, filtered_meters, clean_meters, masks)
        if rec is None:
            continue

        pos_filtered = rec["pos_filtered"]
        pos_clean = rec["pos_clean"]
        times = rec["times"]
        duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        duration_hours = duration_s / 3600.0 if duration_s > 0 else 0.0
        error = pos_filtered - pos_clean

        anchor_indices = choose_anchor_indices(
            len(error),
            duration_s,
            args.anchors_per_hour,
            args.min_anchors,
            args.max_anchors,
        )
        anchor_correction, recording_anchor_rows = build_anchor_correction(
            error,
            anchor_indices,
            channels,
            args.interpolation,
            args.anchor_error_radius,
        )

        moving_correction = np.zeros_like(error)
        slow_error = moving_average(error, args.smooth_window)
        for channel in channels:
            channel_idx = CHANNEL_TO_INDEX[channel]
            moving_correction[:, channel_idx] = slow_error[:, channel_idx]

        methods = {
            "baseline": pos_filtered,
            "moving_average_oracle": pos_filtered - moving_correction,
            "pattern_anchor": pos_filtered - anchor_correction,
        }

        base_meta = {
            "recording": recording,
            "pasada": str(group["pasada"].iloc[0]),
            "pattern": str(group["pattern"].iloc[0]),
            "anchor_count": int(len(anchor_indices)),
            "duration_hours": float(duration_hours),
        }
        for method, positions in methods.items():
            rows.append({**base_meta, "method": method, **metrics_for_positions(positions, pos_clean)})

        baseline_metrics = metrics_for_positions(pos_filtered, pos_clean)
        plot_candidates.append(
            (
                baseline_metrics["rms_xyz_m"],
                recording,
                pos_clean,
                methods,
                anchor_indices,
            )
        )

        for anchor_row in recording_anchor_rows:
            idx = int(anchor_row["sample_index"])
            anchor_rows.append(
                {
                    **base_meta,
                    **anchor_row,
                    "time_s": float(times[idx]),
                    "filtered_x_m": float(pos_filtered[idx, 0]),
                    "filtered_y_m": float(pos_filtered[idx, 1]),
                    "filtered_z_m": float(pos_filtered[idx, 2]),
                    "pattern_x_m": float(pos_clean[idx, 0]),
                    "pattern_y_m": float(pos_clean[idx, 1]),
                    "pattern_z_m": float(pos_clean[idx, 2]),
                }
            )

    methods = ["baseline", "moving_average_oracle", "pattern_anchor"]
    summary = {
        "split": args.split,
        "v3_model": args.v3_model,
        "channels": channels,
        "smooth_window": args.smooth_window,
        "anchors_per_hour": args.anchors_per_hour,
        "min_anchors": args.min_anchors,
        "max_anchors": args.max_anchors,
        "anchor_error_radius": args.anchor_error_radius,
        "interpolation": args.interpolation,
        "recordings": len({row["recording"] for row in rows}),
        "methods": {method: aggregate_metric_rows(rows, method) for method in methods},
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.plots > 0:
        plot_summary(output_dir / "summary_metrics.png", summary, methods)
        plot_candidates.sort(reverse=True, key=lambda item: item[0])
        for _, recording, pos_clean, method_positions, anchor_indices in plot_candidates[: args.plots]:
            plot_recording(
                plots_dir / f"{safe_filename(recording)}.png",
                recording,
                pos_clean,
                method_positions,
                anchor_indices,
                channels,
            )

    write_csv(
        output_dir / "per_recording.csv",
        rows,
        [
            "recording",
            "pasada",
            "pattern",
            "method",
            "anchor_count",
            "duration_hours",
            "n_points",
            "rms_x_m",
            "rms_y_m",
            "rms_z_m",
            "rms_xy_m",
            "rms_xyz_m",
            "final_abs_x_m",
            "final_abs_y_m",
            "final_abs_z_m",
            "final_xy_m",
            "final_xyz_m",
            "length_diff_xy_m",
            "true_length_xy_m",
        ],
    )
    write_csv(
        output_dir / "pattern_anchor_points.csv",
        anchor_rows,
        [
            "recording",
            "pasada",
            "pattern",
            "anchor_count",
            "duration_hours",
            "anchor_index",
            "sample_index",
            "time_s",
            "t_norm",
            "channel",
            "anchor_error_m",
            "filtered_x_m",
            "filtered_y_m",
            "filtered_z_m",
            "pattern_x_m",
            "pattern_y_m",
            "pattern_z_m",
        ],
    )

    print("\n=== SUMMARY ===")
    for method in methods:
        metrics = summary["methods"][method]
        print(
            f"{method:22s} "
            f"RMS X {metrics['rms_x_m']:8.2f} m | "
            f"RMS Y {metrics['rms_y_m']:8.2f} m | "
            f"RMS Z {metrics['rms_z_m']:8.2f} m | "
            f"RMS XY {metrics['rms_xy_m']:8.2f} m | "
            f"RMS XYZ {metrics['rms_xyz_m']:8.2f} m | "
            f"Final XY {metrics['mean_final_xy_m']:8.2f} m | "
            f"Anchors {metrics['mean_anchor_count']:5.1f}"
        )

    print(f"\nSummary saved to: {summary_path}")
    print(f"Per-recording CSV saved to: {output_dir / 'per_recording.csv'}")
    print(f"Pattern anchor points saved to: {output_dir / 'pattern_anchor_points.csv'}")
    if args.plots > 0:
        print(f"Summary plot saved to: {output_dir / 'summary_metrics.png'}")
        print(f"Recording plots saved to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
