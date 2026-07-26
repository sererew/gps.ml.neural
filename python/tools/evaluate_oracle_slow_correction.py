#!/usr/bin/env python3
"""
Evaluate an oracle slow-drift correction upper bound.

The oracle uses the clean pattern to estimate the slow accumulated position
error, subtracts that slow error from the filtered positions, and reports the
remaining drift. This is not usable in production; it measures the best-case
benefit of a slow-drift corrector.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_slow_drift import (
    denormalize_deltas,
    load_prediction_model,
    load_split,
    moving_average,
)


def rms(values):
    """Return root mean square as float, or zero for empty sequences."""
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2))) if len(values) else 0.0


def trajectory_length_xy_from_positions(positions):
    """Compute horizontal trajectory length from positions."""
    if len(positions) < 2:
        return 0.0
    deltas = np.diff(positions[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=1)))


def compute_metrics_for_positions(pos_pred, pos_true):
    """Compute XY and Z drift metrics for position sequences."""
    error = pos_pred - pos_true
    xy_error = np.linalg.norm(error[:, :2], axis=1)
    z_error = error[:, 2]

    pred_length = trajectory_length_xy_from_positions(pos_pred)
    true_length = trajectory_length_xy_from_positions(pos_true)
    length_diff = abs(pred_length - true_length)

    return {
        "final_drift_xy_m": float(xy_error[-1]),
        "rms_drift_xy_m": rms(xy_error),
        "final_z_error_m": float(z_error[-1]),
        "abs_final_z_error_m": float(abs(z_error[-1])),
        "rms_z_drift_m": rms(z_error),
        "pred_length_xy_m": pred_length,
        "true_length_xy_m": true_length,
        "length_diff_xy_m": length_diff,
    }


def evaluate_oracle(filtered_meters, clean_meters, masks, manifest, smooth_window):
    """Evaluate baseline filtered tracks and oracle-corrected tracks."""
    rows = []
    plot_data = []

    for window_index, row in manifest.iterrows():
        valid = masks[window_index].astype(bool)
        pred_valid = filtered_meters[window_index, valid]
        true_valid = clean_meters[window_index, valid]
        n_valid = len(pred_valid)
        if n_valid == 0:
            continue

        pos_pred = np.cumsum(pred_valid, axis=0)
        pos_true = np.cumsum(true_valid, axis=0)
        error = pos_pred - pos_true
        slow_error = moving_average(error, min(smooth_window, n_valid))
        pos_oracle = pos_pred - slow_error

        baseline = compute_metrics_for_positions(pos_pred, pos_true)
        oracle = compute_metrics_for_positions(pos_oracle, pos_true)

        base = {
            "window_index": int(window_index),
            "pasada": row.get("pasada", ""),
            "modalidad": row.get("modalidad", ""),
            "grabacion": row.get("grabacion", ""),
            "pattern": row.get("pattern", ""),
            "window_id": row.get("window_id", ""),
            "n_valid": int(n_valid),
        }

        rows.append(
            {
                **base,
                "baseline_final_drift_xy_m": baseline["final_drift_xy_m"],
                "oracle_final_drift_xy_m": oracle["final_drift_xy_m"],
                "baseline_rms_drift_xy_m": baseline["rms_drift_xy_m"],
                "oracle_rms_drift_xy_m": oracle["rms_drift_xy_m"],
                "baseline_abs_final_z_error_m": baseline["abs_final_z_error_m"],
                "oracle_abs_final_z_error_m": oracle["abs_final_z_error_m"],
                "baseline_rms_z_drift_m": baseline["rms_z_drift_m"],
                "oracle_rms_z_drift_m": oracle["rms_z_drift_m"],
                "baseline_length_diff_xy_m": baseline["length_diff_xy_m"],
                "oracle_length_diff_xy_m": oracle["length_diff_xy_m"],
            }
        )

        plot_data.append(
            {
                "meta": base,
                "baseline_xy": np.linalg.norm(error[:, :2], axis=1),
                "oracle_xy": np.linalg.norm(pos_oracle[:, :2] - pos_true[:, :2], axis=1),
                "baseline_z": error[:, 2],
                "oracle_z": pos_oracle[:, 2] - pos_true[:, 2],
            }
        )

    return pd.DataFrame(rows), plot_data


def aggregate(rows):
    """Aggregate oracle metrics globally and by group."""
    def summarize(df):
        baseline_rms = float(np.sqrt(np.average(df["baseline_rms_drift_xy_m"] ** 2, weights=df["n_valid"])))
        oracle_rms = float(np.sqrt(np.average(df["oracle_rms_drift_xy_m"] ** 2, weights=df["n_valid"])))
        baseline_z_rms = float(np.sqrt(np.average(df["baseline_rms_z_drift_m"] ** 2, weights=df["n_valid"])))
        oracle_z_rms = float(np.sqrt(np.average(df["oracle_rms_z_drift_m"] ** 2, weights=df["n_valid"])))
        return {
            "n_windows": int(len(df)),
            "n_samples": int(df["n_valid"].sum()),
            "baseline_mean_final_drift_xy_m": float(df["baseline_final_drift_xy_m"].mean()),
            "oracle_mean_final_drift_xy_m": float(df["oracle_final_drift_xy_m"].mean()),
            "baseline_rms_drift_xy_m": baseline_rms,
            "oracle_rms_drift_xy_m": oracle_rms,
            "baseline_mean_abs_final_z_error_m": float(df["baseline_abs_final_z_error_m"].mean()),
            "oracle_mean_abs_final_z_error_m": float(df["oracle_abs_final_z_error_m"].mean()),
            "baseline_rms_z_drift_m": baseline_z_rms,
            "oracle_rms_z_drift_m": oracle_z_rms,
            "baseline_mean_length_diff_xy_m": float(df["baseline_length_diff_xy_m"].mean()),
            "oracle_mean_length_diff_xy_m": float(df["oracle_length_diff_xy_m"].mean()),
            "xy_rms_improvement_pct": (baseline_rms - oracle_rms) / baseline_rms * 100.0 if baseline_rms > 0 else 0.0,
            "z_rms_improvement_pct": (baseline_z_rms - oracle_z_rms) / baseline_z_rms * 100.0 if baseline_z_rms > 0 else 0.0,
        }

    global_summary = summarize(rows)
    by_pasada_rows = []
    for pasada, group in rows.groupby("pasada", dropna=False):
        by_pasada_rows.append({"pasada": pasada, **summarize(group)})
    by_pasada = pd.DataFrame(by_pasada_rows)
    return global_summary, by_pasada


def plot_examples(plot_data, rows, output_dir, count):
    """Plot windows with largest baseline XY drift."""
    if count <= 0:
        return
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    top_indices = rows.sort_values("baseline_final_drift_xy_m", ascending=False).head(count)["window_index"].tolist()
    selected = [item for item in plot_data if item["meta"]["window_index"] in top_indices]

    for item in selected:
        meta = item["meta"]
        time = np.arange(len(item["baseline_xy"]))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        ax1.plot(time, item["baseline_xy"], label="baseline XY drift")
        ax1.plot(time, item["oracle_xy"], label="oracle XY drift")
        ax1.set_ylabel("Meters")
        ax1.set_title(f"Oracle slow correction - {meta['window_index']} - {meta['grabacion']}")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(time, item["baseline_z"], label="baseline Z error")
        ax2.plot(time, item["oracle_z"], label="oracle Z error")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Meters")
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()
        filename = f"oracle_window_{meta['window_index']:03d}_{meta['grabacion']}.png"
        safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        fig.savefig(plot_dir / safe_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate oracle slow-drift correction upper bound",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--model", default="models/model_final_v3.keras", help="Keras model path")
    parser.add_argument("--output_dir", default="results/diagnostics/oracle_slow_correction_v3_w1800", help="Output directory")
    parser.add_argument("--smooth_window", type=int, default=1800, help="Moving-average window in timesteps")
    parser.add_argument("--batch_size", type=int, default=32, help="Prediction batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of windows to process")
    parser.add_argument("--plots", type=int, default=5, help="Number of largest-drift windows to plot")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== ORACLE SLOW CORRECTION ===")
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

    model = load_prediction_model(args.model, x_norm.shape[1], x_norm.shape[2])
    residual_pred_norm = model.predict(x_norm, batch_size=args.batch_size, verbose=1)
    filtered_norm = x_norm + residual_pred_norm
    filtered_meters = denormalize_deltas(filtered_norm, norm_stats)
    clean_meters = denormalize_deltas(y_norm, norm_stats)

    rows, plot_data = evaluate_oracle(filtered_meters, clean_meters, masks, manifest, args.smooth_window)
    global_summary, by_pasada = aggregate(rows)

    rows.to_csv(output_dir / "oracle_slow_correction_per_window.csv", index=False)
    by_pasada.to_csv(output_dir / "oracle_slow_correction_by_pasada.csv", index=False)
    with open(output_dir / "oracle_slow_correction_summary.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "split": args.split,
                "smooth_window": args.smooth_window,
                "summary": global_summary,
                "note": "Oracle correction uses clean labels and is only an upper-bound diagnostic.",
            },
            f,
            indent=2,
        )

    plot_examples(plot_data, rows, output_dir, args.plots)

    print("\n=== SUMMARY ===")
    print(f"Windows: {global_summary['n_windows']}")
    print(f"Baseline RMS XY drift: {global_summary['baseline_rms_drift_xy_m']:.3f} m")
    print(f"Oracle RMS XY drift: {global_summary['oracle_rms_drift_xy_m']:.3f} m")
    print(f"XY RMS improvement: {global_summary['xy_rms_improvement_pct']:.2f}%")
    print(f"Baseline mean final XY drift: {global_summary['baseline_mean_final_drift_xy_m']:.3f} m")
    print(f"Oracle mean final XY drift: {global_summary['oracle_mean_final_drift_xy_m']:.3f} m")
    print(f"Baseline RMS Z drift: {global_summary['baseline_rms_z_drift_m']:.3f} m")
    print(f"Oracle RMS Z drift: {global_summary['oracle_rms_z_drift_m']:.3f} m")
    print(f"Z RMS improvement: {global_summary['z_rms_improvement_pct']:.2f}%")
    print(f"Baseline length diff XY: {global_summary['baseline_mean_length_diff_xy_m']:.3f} m")
    print(f"Oracle length diff XY: {global_summary['oracle_mean_length_diff_xy_m']:.3f} m")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
