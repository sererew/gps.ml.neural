#!/usr/bin/env python3
"""
Diagnose whether v3 slow drift can be explained by simple parametric curves.

This tool is diagnostic only. It uses the clean pattern to build oracle slow
errors, then approximates those errors with a small number of XY control points.
The goal is to measure complexity, not to train a production model.
"""

import argparse
import csv
import json
import random
import tempfile
import zipfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.interpolate import CubicSpline
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Masking
from tensorflow.keras.models import Sequential


FEATURES = ["dx", "dy", "dz"]


def create_v3_model(sequence_length, n_features):
    """Create the residual v3 architecture for loading frozen weights."""
    return Sequential(
        [
            Input(shape=(sequence_length, n_features), name="input_layer"),
            Masking(mask_value=0.0, name="masking"),
            LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.0, name="lstm"),
            Dense(64, activation="relu", name="dense"),
            Dropout(0.2, name="dropout"),
            Dense(n_features, activation="linear", name="dense_1"),
        ]
    )


def load_v3_model(model_path, sequence_length, n_features):
    """Load v3 model, with a fallback for Keras version incompatibilities."""
    model_path = Path(model_path)
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        print(f"Direct v3 load failed, trying weight fallback: {exc.__class__.__name__}")

    model = create_v3_model(sequence_length, n_features)
    model(np.zeros((1, sequence_length, n_features), dtype=np.float32))

    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_path = Path(tmp_dir) / "model.weights.h5"
        with zipfile.ZipFile(model_path, "r") as archive:
            archive.extract("model.weights.h5", tmp_dir)
        with h5py.File(weights_path, "r") as weights:
            model.get_layer("lstm").set_weights(
                [
                    weights["layers\\lstm\\cell"]["vars"]["0"][()],
                    weights["layers\\lstm\\cell"]["vars"]["1"][()],
                    weights["layers\\lstm\\cell"]["vars"]["2"][()],
                ]
            )
            model.get_layer("dense").set_weights(
                [
                    weights["layers\\dense"]["vars"]["0"][()],
                    weights["layers\\dense"]["vars"]["1"][()],
                ]
            )
            model.get_layer("dense_1").set_weights(
                [
                    weights["layers\\dense_1"]["vars"]["0"][()],
                    weights["layers\\dense_1"]["vars"]["1"][()],
                ]
            )
    return model


def normalize_manifest_paths(manifest_df):
    """Normalize manifest path separators for cross-platform use."""
    for col in ["slice_path", "label_path", "mask_path"]:
        manifest_df[col] = manifest_df[col].fillna("").astype(str).str.replace("\\", "/", regex=False)


def load_manifest(data_root, split):
    """Load one split manifest."""
    manifest = pd.read_csv(Path(data_root) / split / f"manifest_{split}.csv")
    normalize_manifest_paths(manifest)
    return manifest


def load_norm_stats(data_root):
    """Load normalization statistics."""
    with open(Path(data_root) / "norm_stats_train.json", "r") as f:
        return json.load(f)


def load_split_arrays(manifest):
    """Load normalized noisy deltas, clean deltas, and masks for a manifest."""
    x_list = []
    y_list = []
    mask_list = []

    for _, row in manifest.iterrows():
        slice_data = pd.read_csv(Path(row["slice_path"]))
        label_data = pd.read_csv(Path(row["label_path"]))
        mask_data = pd.read_csv(Path(row["mask_path"]))
        x_list.append(slice_data[FEATURES].values)
        y_list.append(label_data[FEATURES].values)
        mask_list.append(mask_data["mask"].values)

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(mask_list, dtype=np.float32),
    )


def denormalize_deltas(values, norm_stats):
    """Convert normalized deltas back to meters."""
    meters = values.copy()
    for i, feature in enumerate(FEATURES):
        meters[..., i] = values[..., i] * norm_stats["std"][feature] + norm_stats["mean"][feature]
    return meters


def moving_average(values, window):
    """Centered moving average with partial windows at the edges."""
    if window <= 1 or len(values) == 0:
        return values.copy()
    window = int(min(window, len(values)))
    return pd.DataFrame(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def aggregate_recording(group, filtered_meters, clean_meters, masks):
    """Aggregate overlapping windows into one recording timeline."""
    time_parts = []
    filtered_parts = []
    clean_parts = []

    for _, row in group.iterrows():
        window_idx = int(row["index"])
        valid = masks[window_idx].astype(bool)
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            continue
        timestamps = float(row["t_start"]) + np.arange(n_valid, dtype=np.float64)
        time_parts.append(timestamps)
        filtered_parts.append(filtered_meters[window_idx, valid])
        clean_parts.append(clean_meters[window_idx, valid])

    if not time_parts:
        return None

    all_times = np.concatenate(time_parts)
    all_filtered = np.concatenate(filtered_parts, axis=0)
    all_clean = np.concatenate(clean_parts, axis=0)
    order = np.argsort(all_times)
    all_times = all_times[order]
    all_filtered = all_filtered[order]
    all_clean = all_clean[order]

    unique_times, inverse = np.unique(all_times, return_inverse=True)
    filtered_unique = np.zeros((len(unique_times), 3), dtype=np.float64)
    clean_unique = np.zeros((len(unique_times), 3), dtype=np.float64)
    counts = np.zeros(len(unique_times), dtype=np.float64)

    for idx, inv in enumerate(inverse):
        filtered_unique[inv] += all_filtered[idx]
        clean_unique[inv] += all_clean[idx]
        counts[inv] += 1.0

    filtered_unique /= counts[:, None]
    clean_unique /= counts[:, None]

    return {
        "times": unique_times,
        "filtered_deltas": filtered_unique,
        "clean_deltas": clean_unique,
        "pos_filtered": np.cumsum(filtered_unique, axis=0),
        "pos_clean": np.cumsum(clean_unique, axis=0),
    }


def interpolate_control_points(control_points, n_points, interpolation):
    """Interpolate control points to all timesteps."""
    control_t = np.linspace(0.0, 1.0, len(control_points))
    target_t = np.linspace(0.0, 1.0, n_points)

    if interpolation == "cubic" and len(control_points) >= 3:
        spline_x = CubicSpline(control_t, control_points[:, 0], bc_type="natural")
        spline_y = CubicSpline(control_t, control_points[:, 1], bc_type="natural")
        return np.column_stack([spline_x(target_t), spline_y(target_t)])

    return np.column_stack(
        [
            np.interp(target_t, control_t, control_points[:, 0]),
            np.interp(target_t, control_t, control_points[:, 1]),
        ]
    )


def parametric_slow_xy(slow_error, n_control_points, radius, interpolation):
    """Approximate XY slow error with fixed control points and interpolation."""
    n_points = len(slow_error)
    if n_points == 0:
        return np.zeros((0, 2), dtype=np.float64)

    control_indices = np.linspace(0, n_points - 1, n_control_points).round().astype(int)
    controls = np.zeros((n_control_points, 2), dtype=np.float64)

    for i, idx in enumerate(control_indices):
        start = max(0, idx - radius)
        end = min(n_points, idx + radius + 1)
        controls[i] = np.mean(slow_error[start:end, :2], axis=0)

    return interpolate_control_points(controls, n_points, interpolation)


def length_xy(positions):
    """Compute horizontal path length from positions."""
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)))


def metrics_for_positions(pos_pred, pos_true):
    """Compute per-recording XY metrics."""
    if len(pos_pred) == 0:
        return {
            "rms_xy_m": 0.0,
            "final_xy_m": 0.0,
            "length_diff_xy_m": 0.0,
            "true_length_xy_m": 0.0,
        }

    error = pos_pred - pos_true
    xy = np.linalg.norm(error[:, :2], axis=1)
    true_length = length_xy(pos_true)
    return {
        "rms_xy_m": float(np.sqrt(np.mean(xy**2))),
        "final_xy_m": float(xy[-1]),
        "length_diff_xy_m": abs(length_xy(pos_pred) - true_length),
        "true_length_xy_m": true_length,
    }


def aggregate_metrics(rows, method):
    """Aggregate per-recording metrics for one method."""
    subset = [row for row in rows if row["method"] == method]
    if not subset:
        return {}
    mean_true_length = float(np.mean([row["true_length_xy_m"] for row in subset]))
    rms_xy = float(np.sqrt(np.mean([row["rms_xy_m"] ** 2 for row in subset])))
    final_xy = float(np.mean([row["final_xy_m"] for row in subset]))
    length_diff = float(np.mean([row["length_diff_xy_m"] for row in subset]))
    return {
        "recordings": len(subset),
        "rms_xy_m": rms_xy,
        "mean_final_xy_m": final_xy,
        "mean_length_diff_xy_m": length_diff,
        "mean_true_length_xy_m": mean_true_length,
        "rms_xy_pct": rms_xy / mean_true_length * 100.0 if mean_true_length > 0 else 0.0,
        "length_diff_pct": length_diff / mean_true_length * 100.0 if mean_true_length > 0 else 0.0,
    }


def plot_recording(output_path, recording, pos_filtered, pos_clean, slow_error, param_curves):
    """Save diagnostic plots for one recording."""
    error_xy = np.linalg.norm(pos_filtered[:, :2] - pos_clean[:, :2], axis=1)
    slow_xy = np.linalg.norm(slow_error[:, :2], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(error_xy, label="XY error")
    axes[0].plot(slow_xy, label="XY moving-average oracle")
    for label, curve in param_curves.items():
        axes[0].plot(np.linalg.norm(curve, axis=1), label=label)
    axes[0].set_title(f"Parametric slow drift - {recording}")
    axes[0].set_ylabel("Meters")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(slow_error[:, 0], label="slow x")
    axes[1].plot(slow_error[:, 1], label="slow y")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Meters")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose parametric approximations of v3 slow drift",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--v3_model", default="models/model_best_v3.keras", help="Frozen v3 model path")
    parser.add_argument("--smooth_window", type=int, default=1800, help="Moving-average oracle window")
    parser.add_argument("--control_points", default="2,3,5,8", help="Comma-separated control point counts")
    parser.add_argument("--control_radius", type=int, default=90, help="Radius around each control point")
    parser.add_argument(
        "--interpolation",
        default="linear",
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

    control_points = [int(value.strip()) for value in args.control_points.split(",") if value.strip()]
    output_dir = Path(
        args.output_dir
        or f"results/diagnostics/parametric_slow_drift_v3_{args.split}_w{args.smooth_window}_{args.interpolation}"
    )
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=== PARAMETRIC SLOW DRIFT DIAGNOSTIC ===")
    print(f"Split: {args.split}")
    print(f"Smooth window: {args.smooth_window}")
    print(f"Control points: {control_points}")
    print(f"Control radius: {args.control_radius}")
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

    rows = []
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

        baseline = metrics_for_positions(pos_filtered, pos_clean)
        rows.append({"recording": recording, "method": "baseline", **baseline})

        moving_corrected = pos_filtered.copy()
        moving_corrected[:, :2] -= slow_error[:, :2]
        moving_metrics = metrics_for_positions(moving_corrected, pos_clean)
        rows.append({"recording": recording, "method": "moving_average_oracle", **moving_metrics})

        param_curves = {}
        for k in control_points:
            curve_xy = parametric_slow_xy(slow_error, k, args.control_radius, args.interpolation)
            corrected = pos_filtered.copy()
            corrected[:, :2] -= curve_xy
            method = f"control_{k}"
            rows.append({"recording": recording, "method": method, **metrics_for_positions(corrected, pos_clean)})
            param_curves[method] = curve_xy

        plot_candidates.append(
            (
                baseline["rms_xy_m"],
                recording,
                pos_filtered,
                pos_clean,
                slow_error,
                param_curves,
            )
        )

    methods = ["baseline", "moving_average_oracle"] + [f"control_{k}" for k in control_points]
    summary = {
        "split": args.split,
        "v3_model": args.v3_model,
        "smooth_window": args.smooth_window,
        "control_points": control_points,
        "control_radius": args.control_radius,
        "interpolation": args.interpolation,
        "recordings": int(manifest["grabacion"].nunique()),
        "methods": {method: aggregate_metrics(rows, method) for method in methods},
    }

    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "per_recording.csv"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(rows_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "recording",
                "method",
                "rms_xy_m",
                "final_xy_m",
                "length_diff_xy_m",
                "true_length_xy_m",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    plot_candidates.sort(reverse=True, key=lambda item: item[0])
    for _, recording, pos_filtered, pos_clean, slow_error, param_curves in plot_candidates[: args.plots]:
        safe_name = str(recording).replace("/", "_").replace("\\", "_")
        plot_recording(plots_dir / f"{safe_name}.png", recording, pos_filtered, pos_clean, slow_error, param_curves)

    print("\n=== SUMMARY ===")
    for method in methods:
        metrics = summary["methods"][method]
        print(
            f"{method:22s} "
            f"RMS XY {metrics['rms_xy_m']:8.2f} m | "
            f"Final XY {metrics['mean_final_xy_m']:8.2f} m | "
            f"Length diff {metrics['mean_length_diff_xy_m']:8.2f} m"
        )

    print(f"\nSummary saved to: {summary_path}")
    print(f"Per-recording CSV saved to: {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
