#!/usr/bin/env python3
"""
Diagnose parametric slow-drift complexity independently for X, Y, and Z.

This tool is diagnostic only. It uses the clean pattern to build oracle slow
errors, then approximates each selected channel with a small number of control
points. It also exports the oracle control-point values that can be used later
as a direct supervised target.
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


def interpolate_control_values(control_values, n_points, interpolation):
    """Interpolate one-dimensional control values to all timesteps."""
    control_t = np.linspace(0.0, 1.0, len(control_values))
    target_t = np.linspace(0.0, 1.0, n_points)

    if interpolation == "cubic" and len(control_values) >= 3:
        spline = CubicSpline(control_t, control_values, bc_type="natural")
        return spline(target_t)

    return np.interp(target_t, control_t, control_values)


def control_values_for_channel(slow_error, channel_idx, n_control_points, radius):
    """Sample oracle slow error around fixed control points for one channel."""
    n_points = len(slow_error)
    if n_points == 0:
        return (
            np.zeros(n_control_points, dtype=np.float64),
            np.zeros(n_control_points, dtype=np.int64),
            np.zeros(n_control_points, dtype=np.float64),
        )

    control_indices = np.linspace(0, n_points - 1, n_control_points).round().astype(int)
    control_t = np.linspace(0.0, 1.0, n_control_points)
    values = np.zeros(n_control_points, dtype=np.float64)

    for i, idx in enumerate(control_indices):
        start = max(0, idx - radius)
        end = min(n_points, idx + radius + 1)
        values[i] = np.mean(slow_error[start:end, channel_idx])

    return values, control_indices, control_t


def parametric_slow_channels(slow_error, channels, n_control_points, radius, interpolation):
    """Approximate selected slow-error channels with independent control curves."""
    curves = np.zeros_like(slow_error, dtype=np.float64)
    control_rows = []

    for channel in channels:
        channel_idx = CHANNEL_TO_INDEX[channel]
        values, indices, control_t = control_values_for_channel(
            slow_error,
            channel_idx,
            n_control_points,
            radius,
        )
        curves[:, channel_idx] = interpolate_control_values(values, len(slow_error), interpolation)

        for control_idx, (sample_idx, t_norm, value) in enumerate(zip(indices, control_t, values)):
            control_rows.append(
                {
                    "channel": channel,
                    "n_control_points": n_control_points,
                    "control_index": control_idx,
                    "sample_index": int(sample_idx),
                    "t_norm": float(t_norm),
                    "control_value_m": float(value),
                }
            )

    return curves, control_rows


def rms(values):
    """Return root mean square."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    return float(np.sqrt(np.mean(values**2)))


def metrics_for_positions(pos_pred, pos_true):
    """Compute combined and per-channel accumulated-position metrics."""
    if len(pos_pred) == 0:
        return {
            "n_points": 0,
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


def channel_metrics(error, correction, channel):
    """Compute one-channel residual metrics after applying a correction curve."""
    channel_idx = CHANNEL_TO_INDEX[channel]
    residual = error[:, channel_idx] - correction[:, channel_idx]
    return {
        "channel": channel,
        "n_points": int(len(residual)),
        "rms_m": rms(residual),
        "final_abs_m": float(abs(residual[-1])) if len(residual) else 0.0,
        "mean_abs_m": float(np.mean(np.abs(residual))) if len(residual) else 0.0,
    }


def aggregate_metric_rows(rows, method):
    """Aggregate combined per-recording metrics for one method."""
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


def aggregate_channel_rows(rows, method, channel):
    """Aggregate one-channel metrics for one method."""
    subset = [row for row in rows if row["method"] == method and row["channel"] == channel]
    if not subset:
        return {}
    total_points = sum(row["n_points"] for row in subset)
    energy = sum((row["rms_m"] ** 2) * row["n_points"] for row in subset)
    return {
        "recordings": len(subset),
        "n_points": int(total_points),
        "rms_m": float(np.sqrt(energy / total_points)) if total_points > 0 else 0.0,
        "mean_abs_m": float(np.mean([row["mean_abs_m"] for row in subset])),
        "mean_final_abs_m": float(np.mean([row["final_abs_m"] for row in subset])),
    }


def plot_recording(output_path, recording, error, slow_error, param_curves, channels):
    """Save signed slow-drift plots for one recording."""
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3.0 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for ax, channel in zip(axes, channels):
        channel_idx = CHANNEL_TO_INDEX[channel]
        ax.plot(error[:, channel_idx], label=f"{channel} error", alpha=0.6)
        ax.plot(slow_error[:, channel_idx], label=f"{channel} moving-average oracle")
        for label, curve in param_curves.items():
            ax.plot(curve[:, channel_idx], label=label)
        ax.set_ylabel("Meters")
        ax.grid(True)
        ax.legend()

    axes[0].set_title(f"Parametric slow drift by channel - {recording}")
    axes[-1].set_xlabel("Timestep")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows, fieldnames):
    """Write rows to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose per-channel parametric approximations of v3 slow drift",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--v3_model", default="models/model_best_v3.keras", help="Frozen v3 model path")
    parser.add_argument("--smooth_window", type=int, default=1800, help="Moving-average oracle window")
    parser.add_argument("--control_points", default="2,3,5,8", help="Comma-separated control point counts")
    parser.add_argument("--control_radius", type=int, default=90, help="Radius around each control point")
    parser.add_argument("--channels", default="x,y,z", help="Comma-separated channels to correct")
    parser.add_argument(
        "--interpolation",
        default="cubic",
        choices=["linear", "cubic"],
        help="Interpolation mode between control points",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="v3 prediction batch size")
    parser.add_argument("--plots", type=int, default=5, help="Number of recording plots to save")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    channels = parse_channels(args.channels)
    control_points = [int(value.strip()) for value in args.control_points.split(",") if value.strip()]
    output_dir = Path(
        args.output_dir
        or (
            f"results/diagnostics/parametric_slow_drift_channels_v3_{args.split}"
            f"_w{args.smooth_window}_{args.interpolation}"
        )
    )
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== PARAMETRIC SLOW DRIFT CHANNEL DIAGNOSTIC ===")
    print(f"Split: {args.split}")
    print(f"Smooth window: {args.smooth_window}")
    print(f"Control points: {control_points}")
    print(f"Control radius: {args.control_radius}")
    print(f"Channels: {channels}")
    print(f"Interpolation: {args.interpolation}")

    manifest = load_manifest(args.data_root, args.split)
    norm_stats = load_norm_stats(args.data_root)
    x_norm, y_norm, masks = load_split_arrays(manifest)

    sequence_length = x_norm.shape[1]
    v3_model = load_v3_model(args.v3_model, sequence_length, len(FEATURES))
    residual_pred = v3_model.predict(x_norm, batch_size=args.batch_size, verbose=1)
    filtered_norm = x_norm + residual_pred
    filtered_meters = denormalize_deltas(filtered_norm, norm_stats)
    clean_meters = denormalize_deltas(y_norm, norm_stats)

    combined_rows = []
    channel_rows = []
    control_rows = []
    plot_candidates = []

    grouped = manifest.reset_index().sort_values(["grabacion", "t_start"]).groupby("grabacion", sort=False)
    for recording, group in grouped:
        rec = aggregate_recording(group, filtered_meters, clean_meters, masks)
        if rec is None:
            continue

        pos_filtered = rec["pos_filtered"]
        pos_clean = rec["pos_clean"]
        error = pos_filtered - pos_clean
        slow_error = moving_average(error, args.smooth_window)

        zero_correction = np.zeros_like(slow_error)
        baseline_metrics = metrics_for_positions(pos_filtered, pos_clean)
        combined_rows.append({"recording": recording, "method": "baseline", **baseline_metrics})
        for channel in channels:
            channel_rows.append(
                {"recording": recording, "method": "baseline", **channel_metrics(error, zero_correction, channel)}
            )

        moving_correction = np.zeros_like(slow_error)
        for channel in channels:
            moving_correction[:, CHANNEL_TO_INDEX[channel]] = slow_error[:, CHANNEL_TO_INDEX[channel]]
        moving_corrected = pos_filtered - moving_correction
        combined_rows.append(
            {
                "recording": recording,
                "method": "moving_average_oracle",
                **metrics_for_positions(moving_corrected, pos_clean),
            }
        )
        for channel in channels:
            channel_rows.append(
                {
                    "recording": recording,
                    "method": "moving_average_oracle",
                    **channel_metrics(error, moving_correction, channel),
                }
            )

        param_curves_for_plot = {}
        for k in control_points:
            correction, controls = parametric_slow_channels(
                slow_error,
                channels,
                k,
                args.control_radius,
                args.interpolation,
            )
            corrected = pos_filtered - correction
            method = f"control_{k}"
            combined_rows.append({"recording": recording, "method": method, **metrics_for_positions(corrected, pos_clean)})
            for channel in channels:
                channel_rows.append(
                    {"recording": recording, "method": method, **channel_metrics(error, correction, channel)}
                )
            for row in controls:
                control_rows.append({"recording": recording, **row})
            param_curves_for_plot[method] = correction

        plot_candidates.append((baseline_metrics["rms_xyz_m"], recording, error, slow_error, param_curves_for_plot))

    methods = ["baseline", "moving_average_oracle"] + [f"control_{k}" for k in control_points]
    summary = {
        "split": args.split,
        "v3_model": args.v3_model,
        "smooth_window": args.smooth_window,
        "control_points": control_points,
        "control_radius": args.control_radius,
        "channels": channels,
        "interpolation": args.interpolation,
        "recordings": int(manifest["grabacion"].nunique()),
        "methods": {method: aggregate_metric_rows(combined_rows, method) for method in methods},
        "channels_by_method": {
            channel: {method: aggregate_channel_rows(channel_rows, method, channel) for method in methods}
            for channel in channels
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    write_csv(
        output_dir / "per_recording.csv",
        combined_rows,
        [
            "recording",
            "method",
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
        output_dir / "per_recording_channel.csv",
        channel_rows,
        ["recording", "method", "channel", "n_points", "rms_m", "final_abs_m", "mean_abs_m"],
    )
    write_csv(
        output_dir / "oracle_control_points.csv",
        control_rows,
        [
            "recording",
            "channel",
            "n_control_points",
            "control_index",
            "sample_index",
            "t_norm",
            "control_value_m",
        ],
    )

    plot_candidates.sort(reverse=True, key=lambda item: item[0])
    for _, recording, error, slow_error, param_curves in plot_candidates[: args.plots]:
        safe_name = str(recording).replace("/", "_").replace("\\", "_")
        plot_recording(plots_dir / f"{safe_name}.png", recording, error, slow_error, param_curves, channels)

    print("\n=== COMBINED SUMMARY ===")
    for method in methods:
        metrics = summary["methods"][method]
        print(
            f"{method:22s} "
            f"RMS X {metrics['rms_x_m']:8.2f} m | "
            f"RMS Y {metrics['rms_y_m']:8.2f} m | "
            f"RMS Z {metrics['rms_z_m']:8.2f} m | "
            f"RMS XY {metrics['rms_xy_m']:8.2f} m | "
            f"RMS XYZ {metrics['rms_xyz_m']:8.2f} m"
        )

    print("\n=== CHANNEL SUMMARY ===")
    for channel in channels:
        print(f"\nChannel {channel}:")
        for method in methods:
            metrics = summary["channels_by_method"][channel][method]
            print(
                f"  {method:22s} "
                f"RMS {metrics['rms_m']:8.2f} m | "
                f"Mean abs {metrics['mean_abs_m']:8.2f} m | "
                f"Final abs {metrics['mean_final_abs_m']:8.2f} m"
            )

    print(f"\nSummary saved to: {summary_path}")
    print(f"Per-recording CSV saved to: {output_dir / 'per_recording.csv'}")
    print(f"Per-channel CSV saved to: {output_dir / 'per_recording_channel.csv'}")
    print(f"Oracle control points saved to: {output_dir / 'oracle_control_points.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
