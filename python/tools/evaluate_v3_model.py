#!/usr/bin/env python3
"""
Evaluate a trained residual v3 model with separate XY and Z metrics.

This is a v3.1 evaluation tool: it does not train the model. It loads an
existing model, runs predictions on a dataset split, and writes metrics that
keep horizontal and vertical errors separate.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_slow_drift import denormalize_deltas, load_prediction_model, load_split


def safe_mean(values):
    """Return mean as float, or zero for empty sequences."""
    return float(np.mean(values)) if len(values) else 0.0


def rms(values):
    """Return root mean square as float, or zero for empty sequences."""
    return float(np.sqrt(np.mean(np.asarray(values) ** 2))) if len(values) else 0.0


def trajectory_length_xy(deltas):
    """Compute horizontal trajectory length from delta sequence."""
    return float(np.sum(np.linalg.norm(deltas[:, :2], axis=1)))


def calculate_metrics(filtered_meters, clean_meters, masks):
    """Calculate local and accumulated metrics with XY/Z separation."""
    valid_positions = masks.astype(bool)
    local_error = filtered_meters[valid_positions] - clean_meters[valid_positions]
    local_xy = np.linalg.norm(local_error[:, :2], axis=1)
    local_z = local_error[:, 2]

    final_xy_drifts = []
    final_z_errors = []
    all_xy_drifts = []
    all_z_drifts = []
    pred_lengths = []
    true_lengths = []
    length_diffs = []

    for i in range(filtered_meters.shape[0]):
        valid = valid_positions[i]
        pred_valid = filtered_meters[i, valid]
        true_valid = clean_meters[i, valid]
        if len(pred_valid) == 0:
            continue

        pos_pred = np.cumsum(pred_valid, axis=0)
        pos_true = np.cumsum(true_valid, axis=0)
        pos_error = pos_pred - pos_true

        final_xy_drifts.append(np.linalg.norm(pos_error[-1, :2]))
        final_z_errors.append(pos_error[-1, 2])
        all_xy_drifts.extend(np.linalg.norm(pos_error[:, :2], axis=1))
        all_z_drifts.extend(pos_error[:, 2])

        pred_length = trajectory_length_xy(pred_valid)
        true_length = trajectory_length_xy(true_valid)
        pred_lengths.append(pred_length)
        true_lengths.append(true_length)
        length_diffs.append(abs(pred_length - true_length))

    mean_true_length = safe_mean(true_lengths)

    return {
        "mae_dx_m": float(np.mean(np.abs(local_error[:, 0]))),
        "mae_dy_m": float(np.mean(np.abs(local_error[:, 1]))),
        "mae_z_m": float(np.mean(np.abs(local_z))),
        "mae_xy_step_m": float(np.mean(local_xy)),
        "rmse_xy_step_m": rms(local_xy),
        "rmse_z_step_m": rms(local_z),
        "mean_final_drift_xy_m": safe_mean(final_xy_drifts),
        "median_final_drift_xy_m": float(np.median(final_xy_drifts)) if len(final_xy_drifts) else 0.0,
        "rms_drift_xy_m": rms(all_xy_drifts),
        "mean_final_z_error_m": safe_mean(final_z_errors),
        "mean_abs_final_z_error_m": safe_mean(np.abs(final_z_errors)),
        "rms_z_drift_m": rms(all_z_drifts),
        "mean_pred_length_xy_m": safe_mean(pred_lengths),
        "mean_true_length_xy_m": mean_true_length,
        "mean_length_diff_xy_m": safe_mean(length_diffs),
        "length_diff_xy_pct": safe_mean(length_diffs) / mean_true_length * 100.0 if mean_true_length > 0 else 0.0,
        "final_drift_xy_pct": safe_mean(final_xy_drifts) / mean_true_length * 100.0 if mean_true_length > 0 else 0.0,
        "rms_drift_xy_pct": rms(all_xy_drifts) / mean_true_length * 100.0 if mean_true_length > 0 else 0.0,
        "n_windows": int(filtered_meters.shape[0]),
        "n_valid_points": int(np.sum(valid_positions)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate v3 residual model with separate XY and Z metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--model", default="models/model_best_v3.keras", help="Keras model path")
    parser.add_argument("--output", default="results/evaluation/evaluation_v3_1_test.json", help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=32, help="Prediction batch size")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of windows to process")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== V3.1 MODEL EVALUATION ===")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model}")

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
    metrics = calculate_metrics(filtered_meters, clean_meters, masks)

    result = {
        "model": args.model,
        "split": args.split,
        "data_root": args.data_root,
        "metrics": metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== METRICS ===")
    print(f"MAE XY step: {metrics['mae_xy_step_m']:.4f} m")
    print(f"RMSE XY step: {metrics['rmse_xy_step_m']:.4f} m")
    print(f"MAE Z step: {metrics['mae_z_m']:.4f} m")
    print(f"RMSE Z step: {metrics['rmse_z_step_m']:.4f} m")
    print(f"Mean final XY drift: {metrics['mean_final_drift_xy_m']:.4f} m")
    print(f"RMS XY drift: {metrics['rms_drift_xy_m']:.4f} m")
    print(f"Mean abs final Z error: {metrics['mean_abs_final_z_error_m']:.4f} m")
    print(f"RMS Z drift: {metrics['rms_z_drift_m']:.4f} m")
    print(f"Mean XY length diff: {metrics['mean_length_diff_xy_m']:.4f} m")
    print(f"XY length diff: {metrics['length_diff_xy_pct']:.2f}%")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
