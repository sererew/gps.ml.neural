#!/usr/bin/env python3
"""
Diagnose low-frequency drift in residual GPS correction outputs.

The script loads a trained residual v3 model, predicts corrections on a split,
integrates filtered and clean deltas into positions, and decomposes the
position error into slow and fast components with a centered moving average.
"""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


FEATURES = ["dx", "dy", "dz"]


def create_v3_model(sequence_length, n_features):
    """Create the residual v3 architecture used by the training script."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, n_features), name="input_layer"),
            tf.keras.layers.Masking(mask_value=0.0, name="masking"),
            tf.keras.layers.LSTM(
                128,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.0,
                name="lstm",
            ),
            tf.keras.layers.Dense(64, activation="relu", name="dense"),
            tf.keras.layers.Dropout(0.2, name="dropout"),
            tf.keras.layers.Dense(n_features, activation="linear", name="dense_1"),
        ]
    )
    return model


def load_prediction_model(model_path, sequence_length, n_features):
    """Load a model, falling back to architecture rebuild plus weight loading."""
    model_path = Path(model_path)
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        print(f"Direct model load failed, trying v3 weight fallback: {exc.__class__.__name__}")

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
        if col in manifest_df.columns:
            manifest_df[col] = manifest_df[col].fillna("").astype(str).str.replace("\\", "/", regex=False)


def load_split(data_root, split):
    """Load one split from the precomputed input dataset."""
    data_root = Path(data_root)
    manifest_path = data_root / split / f"manifest_{split}.csv"
    norm_stats_path = data_root / "norm_stats_train.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {norm_stats_path}")

    manifest = pd.read_csv(manifest_path)
    normalize_manifest_paths(manifest)

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

    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(mask_list, dtype=np.float32),
        manifest,
        norm_stats,
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
    return (
        pd.DataFrame(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def squared_norm_xy(values):
    """Squared horizontal norm for an array with x,y,z columns."""
    return np.sum(values[:, :2] ** 2, axis=1)


def safe_ratio(numerator, denominator):
    """Return numerator / denominator, with zero for empty denominators."""
    return float(numerator / denominator) if denominator > 0 else 0.0


def rms_from_energy(energy, count):
    """Return RMS from summed squared energy and sample count."""
    return float(np.sqrt(energy / count)) if count > 0 else 0.0


def trajectory_length_xy(deltas):
    """Compute horizontal trajectory length from delta sequence."""
    return float(np.sum(np.linalg.norm(deltas[:, :2], axis=1)))


def compute_window_metrics(index, row, pred_meters, true_meters, mask, smooth_window):
    """Compute slow-drift metrics for one valid window."""
    valid = mask.astype(bool)
    pred_valid = pred_meters[valid]
    true_valid = true_meters[valid]
    n_valid = len(pred_valid)

    if n_valid == 0:
        return None, None

    pos_pred = np.cumsum(pred_valid, axis=0)
    pos_true = np.cumsum(true_valid, axis=0)
    error = pos_pred - pos_true
    slow = moving_average(error, smooth_window)
    fast = error - slow

    xy_total_energy = float(np.sum(squared_norm_xy(error)))
    xy_slow_energy = float(np.sum(squared_norm_xy(slow)))
    xy_fast_energy = float(np.sum(squared_norm_xy(fast)))

    z_total_energy = float(np.sum(error[:, 2] ** 2))
    z_slow_energy = float(np.sum(slow[:, 2] ** 2))
    z_fast_energy = float(np.sum(fast[:, 2] ** 2))

    final_xy_error = float(np.linalg.norm(error[-1, :2]))
    final_z_error = float(error[-1, 2])
    true_length = trajectory_length_xy(true_valid)
    pred_length = trajectory_length_xy(pred_valid)

    metrics = {
        "window_index": int(index),
        "pasada": row.get("pasada", ""),
        "modalidad": row.get("modalidad", ""),
        "grabacion": row.get("grabacion", ""),
        "pattern": row.get("pattern", ""),
        "window_id": row.get("window_id", ""),
        "n_valid": int(n_valid),
        "smooth_window": int(min(smooth_window, n_valid)),
        "true_length_xy_m": true_length,
        "pred_length_xy_m": pred_length,
        "length_diff_xy_m": abs(pred_length - true_length),
        "final_drift_xy_m": final_xy_error,
        "final_z_error_m": final_z_error,
        "xy_rms_error_m": rms_from_energy(xy_total_energy, n_valid),
        "xy_rms_slow_m": rms_from_energy(xy_slow_energy, n_valid),
        "xy_rms_fast_m": rms_from_energy(xy_fast_energy, n_valid),
        "xy_slow_energy_ratio_total": safe_ratio(xy_slow_energy, xy_total_energy),
        "xy_slow_energy_share": safe_ratio(xy_slow_energy, xy_slow_energy + xy_fast_energy),
        "z_rms_error_m": rms_from_energy(z_total_energy, n_valid),
        "z_rms_slow_m": rms_from_energy(z_slow_energy, n_valid),
        "z_rms_fast_m": rms_from_energy(z_fast_energy, n_valid),
        "z_slow_energy_ratio_total": safe_ratio(z_slow_energy, z_total_energy),
        "z_slow_energy_share": safe_ratio(z_slow_energy, z_slow_energy + z_fast_energy),
        "xy_total_energy": xy_total_energy,
        "xy_slow_energy": xy_slow_energy,
        "xy_fast_energy": xy_fast_energy,
        "z_total_energy": z_total_energy,
        "z_slow_energy": z_slow_energy,
        "z_fast_energy": z_fast_energy,
    }

    series = {
        "time": np.arange(n_valid),
        "xy_error_norm": np.linalg.norm(error[:, :2], axis=1),
        "xy_slow_norm": np.linalg.norm(slow[:, :2], axis=1),
        "xy_fast_norm": np.linalg.norm(fast[:, :2], axis=1),
        "z_error": error[:, 2],
        "z_slow": slow[:, 2],
        "z_fast": fast[:, 2],
    }

    return metrics, series


def aggregate_metrics(per_window):
    """Build global and grouped summaries from per-window metrics."""
    df = pd.DataFrame(per_window)
    sample_count = int(df["n_valid"].sum())

    xy_total = float(df["xy_total_energy"].sum())
    xy_slow = float(df["xy_slow_energy"].sum())
    xy_fast = float(df["xy_fast_energy"].sum())
    z_total = float(df["z_total_energy"].sum())
    z_slow = float(df["z_slow_energy"].sum())
    z_fast = float(df["z_fast_energy"].sum())

    summary = {
        "n_windows": int(len(df)),
        "n_samples": sample_count,
        "xy_rms_error_m": rms_from_energy(xy_total, sample_count),
        "xy_rms_slow_m": rms_from_energy(xy_slow, sample_count),
        "xy_rms_fast_m": rms_from_energy(xy_fast, sample_count),
        "xy_slow_energy_ratio_total": safe_ratio(xy_slow, xy_total),
        "xy_slow_energy_share": safe_ratio(xy_slow, xy_slow + xy_fast),
        "z_rms_error_m": rms_from_energy(z_total, sample_count),
        "z_rms_slow_m": rms_from_energy(z_slow, sample_count),
        "z_rms_fast_m": rms_from_energy(z_fast, sample_count),
        "z_slow_energy_ratio_total": safe_ratio(z_slow, z_total),
        "z_slow_energy_share": safe_ratio(z_slow, z_slow + z_fast),
        "mean_final_drift_xy_m": float(df["final_drift_xy_m"].mean()),
        "median_final_drift_xy_m": float(df["final_drift_xy_m"].median()),
        "mean_abs_final_z_error_m": float(df["final_z_error_m"].abs().mean()),
        "mean_length_diff_xy_m": float(df["length_diff_xy_m"].mean()),
        "mean_true_length_xy_m": float(df["true_length_xy_m"].mean()),
    }

    group_cols = ["pasada"]
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_windows=("window_index", "count"),
            n_samples=("n_valid", "sum"),
            mean_final_drift_xy_m=("final_drift_xy_m", "mean"),
            mean_length_diff_xy_m=("length_diff_xy_m", "mean"),
            mean_xy_slow_energy_share=("xy_slow_energy_share", "mean"),
            mean_z_slow_energy_share=("z_slow_energy_share", "mean"),
            mean_xy_rms_error_m=("xy_rms_error_m", "mean"),
            mean_xy_rms_slow_m=("xy_rms_slow_m", "mean"),
            mean_xy_rms_fast_m=("xy_rms_fast_m", "mean"),
        )
        .reset_index()
    )

    return summary, grouped


def plot_examples(example_series, output_dir):
    """Save diagnostic plots for selected windows."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for item in example_series:
        metrics = item["metrics"]
        series = item["series"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax1.plot(series["time"], series["xy_error_norm"], label="XY error")
        ax1.plot(series["time"], series["xy_slow_norm"], label="XY slow")
        ax1.plot(series["time"], series["xy_fast_norm"], label="XY fast")
        ax1.set_ylabel("Meters")
        ax1.set_title(f"Window {metrics['window_index']} - {metrics['grabacion']}")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(series["time"], series["z_error"], label="Z error")
        ax2.plot(series["time"], series["z_slow"], label="Z slow")
        ax2.plot(series["time"], series["z_fast"], label="Z fast")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Meters")
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        filename = f"window_{metrics['window_index']:03d}_{metrics['grabacion']}.png"
        safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        fig.savefig(plot_dir / safe_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose low-frequency drift in v3 residual model outputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--model", default="models/model_final_v3.keras", help="Keras model path")
    parser.add_argument("--output_dir", default="results/diagnostics/slow_drift_v3", help="Output directory")
    parser.add_argument("--smooth_window", type=int, default=600, help="Moving-average window in timesteps")
    parser.add_argument("--batch_size", type=int, default=32, help="Prediction batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of windows to process")
    parser.add_argument("--plots", type=int, default=5, help="Number of highest-drift windows to plot")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== SLOW DRIFT DIAGNOSTIC ===")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Smooth window: {args.smooth_window}")

    x_norm, y_norm, masks, manifest, norm_stats = load_split(args.data_root, args.split)

    if args.limit is not None:
        x_norm = x_norm[: args.limit]
        y_norm = y_norm[: args.limit]
        masks = masks[: args.limit]
        manifest = manifest.iloc[: args.limit].reset_index(drop=True)

    print(f"Loaded windows: {len(x_norm)}")
    model = load_prediction_model(args.model, x_norm.shape[1], x_norm.shape[2])
    residual_pred_norm = model.predict(x_norm, batch_size=args.batch_size, verbose=1)
    filtered_norm = x_norm + residual_pred_norm

    filtered_meters = denormalize_deltas(filtered_norm, norm_stats)
    clean_meters = denormalize_deltas(y_norm, norm_stats)

    per_window = []
    all_series = []

    for i, row in manifest.iterrows():
        metrics, series = compute_window_metrics(
            i,
            row,
            filtered_meters[i],
            clean_meters[i],
            masks[i],
            args.smooth_window,
        )
        if metrics is None:
            continue
        per_window.append(metrics)
        all_series.append({"metrics": metrics, "series": series})

    summary, grouped = aggregate_metrics(per_window)
    summary.update(
        {
            "split": args.split,
            "model": args.model,
            "smooth_window": args.smooth_window,
            "energy_note": (
                "Slow and fast components come from a moving-average decomposition. "
                "They are useful diagnostics but are not an orthogonal spectral split."
            ),
        }
    )

    per_window_df = pd.DataFrame(per_window)
    per_window_df.to_csv(output_dir / "slow_drift_per_window.csv", index=False)
    grouped.to_csv(output_dir / "slow_drift_by_pasada.csv", index=False)
    with open(output_dir / "slow_drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.plots > 0:
        selected = sorted(
            all_series,
            key=lambda item: item["metrics"]["final_drift_xy_m"],
            reverse=True,
        )[: args.plots]
        plot_examples(selected, output_dir)

    print("\n=== SUMMARY ===")
    print(f"Windows: {summary['n_windows']}")
    print(f"XY RMS error: {summary['xy_rms_error_m']:.3f} m")
    print(f"XY RMS slow: {summary['xy_rms_slow_m']:.3f} m")
    print(f"XY RMS fast: {summary['xy_rms_fast_m']:.3f} m")
    print(f"XY slow energy / total error energy: {summary['xy_slow_energy_ratio_total']:.3f}")
    print(f"XY slow energy share: {summary['xy_slow_energy_share']:.3f}")
    print(f"Z RMS error: {summary['z_rms_error_m']:.3f} m")
    print(f"Z RMS slow: {summary['z_rms_slow_m']:.3f} m")
    print(f"Z RMS fast: {summary['z_rms_fast_m']:.3f} m")
    print(f"Z slow energy / total error energy: {summary['z_slow_energy_ratio_total']:.3f}")
    print(f"Z slow energy share: {summary['z_slow_energy_share']:.3f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
