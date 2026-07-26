#!/usr/bin/env python3
"""
Diagnose whether slow drift follows repeatable signed patterns.

This tool complements diagnose_slow_drift.py. Instead of looking only at drift
magnitude, it studies signed slow error components (x, y, z) over normalized
time, grouped by pass and recording.
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


def safe_ratio(numerator, denominator):
    """Return numerator / denominator, with zero for empty denominators."""
    return float(numerator / denominator) if denominator > 0 else 0.0


def compute_slow_series(filtered_meters, clean_meters, masks, manifest, smooth_window, n_bins):
    """Return per-sample slow drift records and per-window summaries."""
    records = []
    window_rows = []

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
        slow = moving_average(error, min(smooth_window, n_valid))

        t_norm = np.linspace(0.0, 1.0, n_valid)
        bins = np.minimum((t_norm * n_bins).astype(int), n_bins - 1)
        xy_norm = np.linalg.norm(slow[:, :2], axis=1)

        base = {
            "window_index": int(window_index),
            "pasada": row.get("pasada", ""),
            "modalidad": row.get("modalidad", ""),
            "grabacion": row.get("grabacion", ""),
            "pattern": row.get("pattern", ""),
            "window_id": row.get("window_id", ""),
        }

        for i in range(n_valid):
            records.append(
                {
                    **base,
                    "t_norm": float(t_norm[i]),
                    "bin": int(bins[i]),
                    "slow_x_m": float(slow[i, 0]),
                    "slow_y_m": float(slow[i, 1]),
                    "slow_z_m": float(slow[i, 2]),
                    "slow_xy_norm_m": float(xy_norm[i]),
                }
            )

        window_rows.append(
            {
                **base,
                "n_valid": int(n_valid),
                "final_slow_x_m": float(slow[-1, 0]),
                "final_slow_y_m": float(slow[-1, 1]),
                "final_slow_z_m": float(slow[-1, 2]),
                "final_slow_xy_norm_m": float(xy_norm[-1]),
                "mean_slow_xy_norm_m": float(np.mean(xy_norm)),
                "mean_abs_slow_z_m": float(np.mean(np.abs(slow[:, 2]))),
            }
        )

    return pd.DataFrame(records), pd.DataFrame(window_rows)


def aggregate_time_bins(records, group_cols):
    """Aggregate signed slow drift by time bins and optional groups."""
    rows = []
    grouped = records.groupby(group_cols + ["bin"], dropna=False)

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_data = dict(zip(group_cols + ["bin"], keys))
        mean_x = float(group["slow_x_m"].mean())
        mean_y = float(group["slow_y_m"].mean())
        mean_z = float(group["slow_z_m"].mean())
        mean_xy_norm = float(group["slow_xy_norm_m"].mean())
        mean_vector_norm = float(np.linalg.norm([mean_x, mean_y]))

        rows.append(
            {
                **key_data,
                "n_samples": int(len(group)),
                "n_windows": int(group["window_index"].nunique()),
                "t_norm_mean": float(group["t_norm"].mean()),
                "mean_slow_x_m": mean_x,
                "mean_slow_y_m": mean_y,
                "mean_slow_z_m": mean_z,
                "std_slow_x_m": float(group["slow_x_m"].std(ddof=0)),
                "std_slow_y_m": float(group["slow_y_m"].std(ddof=0)),
                "std_slow_z_m": float(group["slow_z_m"].std(ddof=0)),
                "mean_slow_xy_norm_m": mean_xy_norm,
                "mean_vector_xy_norm_m": mean_vector_norm,
                "directional_coherence_xy": safe_ratio(mean_vector_norm, mean_xy_norm),
            }
        )

    return pd.DataFrame(rows)


def summarize_groups(window_summary, records, group_cols):
    """Summarize final slow drift and average directional coherence by group."""
    time_bins = aggregate_time_bins(records, group_cols)
    rows = []

    for keys, group in window_summary.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_data = dict(zip(group_cols, keys))
        mean_final_x = float(group["final_slow_x_m"].mean())
        mean_final_y = float(group["final_slow_y_m"].mean())
        mean_final_z = float(group["final_slow_z_m"].mean())
        mean_final_xy_norm = float(group["final_slow_xy_norm_m"].mean())
        mean_final_vector_norm = float(np.linalg.norm([mean_final_x, mean_final_y]))

        matching = time_bins
        for col, value in key_data.items():
            matching = matching[matching[col] == value]

        rows.append(
            {
                **key_data,
                "n_windows": int(len(group)),
                "mean_final_slow_x_m": mean_final_x,
                "mean_final_slow_y_m": mean_final_y,
                "mean_final_slow_z_m": mean_final_z,
                "mean_final_slow_xy_norm_m": mean_final_xy_norm,
                "final_directional_coherence_xy": safe_ratio(mean_final_vector_norm, mean_final_xy_norm),
                "mean_slow_xy_norm_m": float(group["mean_slow_xy_norm_m"].mean()),
                "mean_abs_slow_z_m": float(group["mean_abs_slow_z_m"].mean()),
                "mean_timebin_directional_coherence_xy": float(matching["directional_coherence_xy"].mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_final_slow_xy_norm_m", ascending=False)


def build_summary(global_bins, by_pasada_summary, window_summary, args):
    """Build compact JSON summary."""
    return {
        "model": args.model,
        "split": args.split,
        "smooth_window": args.smooth_window,
        "n_bins": args.n_bins,
        "n_windows": int(len(window_summary)),
        "global_mean_timebin_directional_coherence_xy": float(global_bins["directional_coherence_xy"].mean()),
        "global_min_timebin_directional_coherence_xy": float(global_bins["directional_coherence_xy"].min()),
        "global_max_timebin_directional_coherence_xy": float(global_bins["directional_coherence_xy"].max()),
        "mean_final_slow_xy_norm_m": float(window_summary["final_slow_xy_norm_m"].mean()),
        "mean_abs_final_slow_z_m": float(window_summary["final_slow_z_m"].abs().mean()),
        "top_pasadas_by_final_slow_xy": by_pasada_summary.head(10).to_dict(orient="records"),
        "coherence_note": (
            "directional_coherence_xy is norm(mean signed XY vector) / mean(norm XY). "
            "Values near 1 mean aligned drift direction; values near 0 mean cancellation."
        ),
    }


def plot_time_profile(time_bins, output_dir, title, filename):
    """Plot signed slow drift components over normalized time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(time_bins["t_norm_mean"], time_bins["mean_slow_x_m"], label="slow x")
    ax1.plot(time_bins["t_norm_mean"], time_bins["mean_slow_y_m"], label="slow y")
    ax1.plot(time_bins["t_norm_mean"], time_bins["mean_vector_xy_norm_m"], label="mean XY vector norm")
    ax1.set_ylabel("Meters")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_bins["t_norm_mean"], time_bins["mean_slow_z_m"], label="slow z")
    ax2.plot(time_bins["t_norm_mean"], time_bins["directional_coherence_xy"], label="XY coherence")
    ax2.set_xlabel("Normalized time")
    ax2.set_ylabel("Meters / coherence")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose whether slow drift follows signed repeatable patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--model", default="models/model_final_v3.keras", help="Keras model path")
    parser.add_argument("--output_dir", default="results/diagnostics/slow_drift_pattern_v3_w1800", help="Output directory")
    parser.add_argument("--smooth_window", type=int, default=1800, help="Moving-average window in timesteps")
    parser.add_argument("--n_bins", type=int, default=20, help="Number of normalized-time bins")
    parser.add_argument("--batch_size", type=int, default=32, help="Prediction batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of windows to process")
    parser.add_argument("--plots", action="store_true", help="Write global and per-pasada plots")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== SLOW DRIFT PATTERN DIAGNOSTIC ===")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")
    print(f"Smooth window: {args.smooth_window}")
    print(f"Time bins: {args.n_bins}")

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

    records, window_summary = compute_slow_series(
        filtered_meters,
        clean_meters,
        masks,
        manifest,
        args.smooth_window,
        args.n_bins,
    )

    global_bins = aggregate_time_bins(records, [])
    by_pasada_bins = aggregate_time_bins(records, ["pasada"])
    by_pasada_summary = summarize_groups(window_summary, records, ["pasada"])
    by_grabacion_summary = summarize_groups(window_summary, records, ["pasada", "grabacion"])

    records.to_csv(output_dir / "slow_drift_signed_samples.csv", index=False)
    window_summary.to_csv(output_dir / "slow_drift_window_summary.csv", index=False)
    global_bins.to_csv(output_dir / "slow_drift_global_time_bins.csv", index=False)
    by_pasada_bins.to_csv(output_dir / "slow_drift_by_pasada_time_bins.csv", index=False)
    by_pasada_summary.to_csv(output_dir / "slow_drift_by_pasada_summary.csv", index=False)
    by_grabacion_summary.to_csv(output_dir / "slow_drift_by_grabacion_summary.csv", index=False)

    summary = build_summary(global_bins, by_pasada_summary, window_summary, args)
    with open(output_dir / "slow_drift_pattern_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.plots:
        plot_time_profile(global_bins, output_dir, "Global signed slow drift", "global_signed_slow_drift.png")
        for pasada, group in by_pasada_bins.groupby("pasada", dropna=False):
            safe_pasada = "".join(c if str(c).isalnum() else "_" for c in str(pasada))
            plot_time_profile(
                group,
                output_dir,
                f"Signed slow drift - pasada {pasada}",
                f"pasada_{safe_pasada}_signed_slow_drift.png",
            )

    print("\n=== SUMMARY ===")
    print(f"Windows: {summary['n_windows']}")
    print(f"Global mean time-bin XY coherence: {summary['global_mean_timebin_directional_coherence_xy']:.3f}")
    print(f"Global min time-bin XY coherence: {summary['global_min_timebin_directional_coherence_xy']:.3f}")
    print(f"Global max time-bin XY coherence: {summary['global_max_timebin_directional_coherence_xy']:.3f}")
    print(f"Mean final slow XY norm: {summary['mean_final_slow_xy_norm_m']:.3f} m")
    print(f"Mean abs final slow Z: {summary['mean_abs_final_slow_z_m']:.3f} m")
    print("\nTop pasadas by final slow XY:")
    for row in summary["top_pasadas_by_final_slow_xy"]:
        print(
            f"  pasada {row['pasada']}: final XY {row['mean_final_slow_xy_norm_m']:.3f} m, "
            f"time-bin coherence {row['mean_timebin_directional_coherence_xy']:.3f}"
        )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
