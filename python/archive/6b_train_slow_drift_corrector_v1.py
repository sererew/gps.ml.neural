#!/usr/bin/env python3
"""
Train a second-stage slow-drift corrector on top of the frozen v3 model.

The frozen v3 model predicts local residual delta corrections. This script
integrates those filtered deltas into positions, computes the slow accumulated
position error against the clean pattern, and trains a small sequence model to
predict that slow error from filtered-track features.

This is not an oracle: the corrector input only uses filtered-track signals and
context features. The oracle target is only used as supervision during training.
"""

import argparse
import json
import random
import tempfile
import time
import zipfile
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


FEATURES = ["dx", "dy", "dz"]


def create_v3_model(sequence_length, n_features):
    """Create the residual v3 architecture for loading frozen weights."""
    model = Sequential(
        [
            Input(shape=(sequence_length, n_features), name="input_layer"),
            Masking(mask_value=0.0, name="masking"),
            LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.0, name="lstm"),
            Dense(64, activation="relu", name="dense"),
            Dropout(0.2, name="dropout"),
            Dense(n_features, activation="linear", name="dense_1"),
        ]
    )
    return model


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
        if col in manifest_df.columns:
            manifest_df[col] = manifest_df[col].fillna("").astype(str).str.replace("\\", "/", regex=False)


def add_recording_ranges(*manifests):
    """Add recording-level time ranges for absolute_t_norm."""
    combined = pd.concat([df[["grabacion", "t_start", "t_end"]] for df in manifests], ignore_index=True)
    ranges = combined.groupby("grabacion").agg(recording_start=("t_start", "min"), recording_end=("t_end", "max"))
    for manifest in manifests:
        manifest["recording_start"] = manifest["grabacion"].map(ranges["recording_start"])
        manifest["recording_end"] = manifest["grabacion"].map(ranges["recording_end"])


def load_manifest(data_root, split):
    """Load and normalize one split manifest."""
    path = Path(data_root) / split / f"manifest_{split}.csv"
    manifest = pd.read_csv(path)
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


def build_slow_corrector_dataset(filtered_norm, filtered_meters, clean_meters, masks, manifest, smooth_window):
    """Build input features and slow-error targets for the second-stage model."""
    n_windows, sequence_length, _ = filtered_meters.shape
    x_slow = np.zeros((n_windows, sequence_length, 9), dtype=np.float32)
    y_slow = np.zeros((n_windows, sequence_length, 3), dtype=np.float32)
    scales = np.ones((n_windows, 2), dtype=np.float32)

    for i, row in manifest.iterrows():
        valid = masks[i].astype(bool)
        valid_indices = np.where(valid)[0]
        n_valid = len(valid_indices)
        if n_valid == 0:
            continue

        filtered_valid = filtered_meters[i, valid]
        clean_valid = clean_meters[i, valid]
        pos_filtered = np.cumsum(filtered_valid, axis=0)
        pos_clean = np.cumsum(clean_valid, axis=0)
        slow_error = moving_average(pos_filtered - pos_clean, smooth_window)

        step_lengths = np.linalg.norm(filtered_valid[:, :2], axis=1)
        cumulative_distance = np.cumsum(step_lengths)
        length_scale = max(float(cumulative_distance[-1]), 1.0)
        z_scale = max(float(np.ptp(pos_filtered[:, 2])), 10.0)
        scales[i] = [length_scale, z_scale]

        t_norm = np.zeros(n_valid, dtype=np.float32)
        if n_valid > 1:
            t_norm = np.arange(n_valid, dtype=np.float32) / float(n_valid - 1)
        distance_norm = cumulative_distance / length_scale

        recording_start = float(row.get("recording_start", row.get("t_start", 0.0)))
        recording_end = float(row.get("recording_end", row.get("t_end", recording_start)))
        recording_duration = max(recording_end - recording_start, 1.0)
        timestamps = float(row["t_start"]) + np.arange(n_valid, dtype=np.float32)
        absolute_t_norm = np.clip((timestamps - recording_start) / recording_duration, 0.0, 1.0)

        x_slow[i, valid_indices, 0:3] = filtered_norm[i, valid]
        x_slow[i, valid_indices, 3] = pos_filtered[:, 0] / length_scale
        x_slow[i, valid_indices, 4] = pos_filtered[:, 1] / length_scale
        x_slow[i, valid_indices, 5] = pos_filtered[:, 2] / z_scale
        x_slow[i, valid_indices, 6] = t_norm
        x_slow[i, valid_indices, 7] = distance_norm
        x_slow[i, valid_indices, 8] = absolute_t_norm

        y_slow[i, valid_indices, 0] = slow_error[:, 0] / length_scale
        y_slow[i, valid_indices, 1] = slow_error[:, 1] / length_scale
        y_slow[i, valid_indices, 2] = slow_error[:, 2] / z_scale

    return x_slow, y_slow, scales


def create_slow_corrector(sequence_length, n_features):
    """Create a small sequence model for slow accumulated drift."""
    return Sequential(
        [
            Input(shape=(sequence_length, n_features), name="slow_input"),
            Masking(mask_value=0.0, name="slow_masking"),
            LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.0, name="slow_lstm"),
            Dense(32, activation="relu", name="slow_dense"),
            Dropout(0.1, name="slow_dropout"),
            Dense(3, activation="linear", name="slow_output"),
        ]
    )


def slow_mae_loss(y_true, y_pred):
    """Per-timestep MAE for scaled slow accumulated error."""
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)


def positions_from_deltas(deltas, mask):
    """Return valid accumulated positions for one window."""
    return np.cumsum(deltas[mask.astype(bool)], axis=0)


def length_xy_from_positions(positions):
    """Compute horizontal path length from positions."""
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)))


def compute_position_metrics(pos_pred_list, pos_true_list):
    """Compute aggregate position metrics from lists of position arrays."""
    final_xy = []
    all_xy = []
    final_z_abs = []
    all_z = []
    length_diff = []
    true_lengths = []

    for pos_pred, pos_true in zip(pos_pred_list, pos_true_list):
        if len(pos_pred) == 0:
            continue
        error = pos_pred - pos_true
        xy = np.linalg.norm(error[:, :2], axis=1)
        z = error[:, 2]
        final_xy.append(float(xy[-1]))
        all_xy.extend(xy)
        final_z_abs.append(float(abs(z[-1])))
        all_z.extend(z)
        pred_length = length_xy_from_positions(pos_pred)
        true_length = length_xy_from_positions(pos_true)
        length_diff.append(abs(pred_length - true_length))
        true_lengths.append(true_length)

    rms_xy = float(np.sqrt(np.mean(np.asarray(all_xy) ** 2))) if all_xy else 0.0
    rms_z = float(np.sqrt(np.mean(np.asarray(all_z) ** 2))) if all_z else 0.0
    mean_true_length = float(np.mean(true_lengths)) if true_lengths else 0.0

    return {
        "mean_final_drift_xy_m": float(np.mean(final_xy)) if final_xy else 0.0,
        "rms_drift_xy_m": rms_xy,
        "mean_abs_final_z_error_m": float(np.mean(final_z_abs)) if final_z_abs else 0.0,
        "rms_z_drift_m": rms_z,
        "mean_length_diff_xy_m": float(np.mean(length_diff)) if length_diff else 0.0,
        "mean_true_length_xy_m": mean_true_length,
        "rms_drift_xy_pct": rms_xy / mean_true_length * 100.0 if mean_true_length > 0 else 0.0,
    }


def evaluate_corrector(model, x_slow, scales, filtered_meters, clean_meters, masks, smooth_window, batch_size):
    """Evaluate baseline, learned slow corrector, and oracle upper bound."""
    pred_scaled = model.predict(x_slow, batch_size=batch_size, verbose=1)

    baseline_positions = []
    corrected_positions = []
    oracle_positions = []
    true_positions = []

    for i in range(filtered_meters.shape[0]):
        valid = masks[i].astype(bool)
        if not np.any(valid):
            continue

        pos_filtered = positions_from_deltas(filtered_meters[i], valid)
        pos_true = positions_from_deltas(clean_meters[i], valid)
        slow_true = moving_average(pos_filtered - pos_true, smooth_window)

        pred_valid = pred_scaled[i, valid].copy()
        pred_valid[:, 0] *= scales[i, 0]
        pred_valid[:, 1] *= scales[i, 0]
        pred_valid[:, 2] *= scales[i, 1]
        pred_valid = moving_average(pred_valid, smooth_window)

        baseline_positions.append(pos_filtered)
        corrected_positions.append(pos_filtered - pred_valid)
        oracle_positions.append(pos_filtered - slow_true)
        true_positions.append(pos_true)

    return {
        "baseline": compute_position_metrics(baseline_positions, true_positions),
        "slow_corrected": compute_position_metrics(corrected_positions, true_positions),
        "oracle": compute_position_metrics(oracle_positions, true_positions),
    }


def plot_history(history, output_path):
    """Save training history plot."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Val Loss")
    ax.set_title("Slow Corrector Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_split(manifest, v3_model, norm_stats, smooth_window, batch_size):
    """Load one split, run v3, and build slow-corrector tensors."""
    x_norm, y_norm, masks = load_split_arrays(manifest)
    residual_pred = v3_model.predict(x_norm, batch_size=batch_size, verbose=1)
    filtered_norm = x_norm + residual_pred
    filtered_meters = denormalize_deltas(filtered_norm, norm_stats)
    clean_meters = denormalize_deltas(y_norm, norm_stats)
    x_slow, y_slow, scales = build_slow_corrector_dataset(
        filtered_norm,
        filtered_meters,
        clean_meters,
        masks,
        manifest,
        smooth_window,
    )
    return x_slow, y_slow, masks, scales, filtered_meters, clean_meters


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a slow-drift corrector on top of frozen v3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--v3_model", default="models/model_best_v3.keras", help="Frozen v3 model path")
    parser.add_argument("--smooth_window", type=int, default=1800, help="Oracle slow target smoothing window")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--fast", action="store_true", help="Fast mode for quick checks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    print("=== SLOW DRIFT CORRECTOR TRAINING V1 ===")
    print(f"Data root: {args.data_root}")
    print(f"Frozen v3 model: {args.v3_model}")
    print(f"Smooth window: {args.smooth_window}")
    print(f"Fast mode: {args.fast}")

    train_manifest = load_manifest(args.data_root, "train")
    val_manifest = load_manifest(args.data_root, "val")
    test_manifest = load_manifest(args.data_root, "test")
    add_recording_ranges(train_manifest, val_manifest, test_manifest)

    if args.fast:
        args.epochs = min(args.epochs, 10)
        args.batch_size = min(args.batch_size, 16)
        args.patience = min(args.patience, 5)
        train_manifest = train_manifest.sample(n=min(len(train_manifest), 100), random_state=args.seed).reset_index(drop=True)
        val_manifest = val_manifest.sample(n=min(len(val_manifest), 25), random_state=args.seed).reset_index(drop=True)
        test_manifest = test_manifest.sample(n=min(len(test_manifest), 25), random_state=args.seed).reset_index(drop=True)
        print(f"FAST MODE: Train {len(train_manifest)}, Val {len(val_manifest)}, Test {len(test_manifest)}")

    norm_stats = load_norm_stats(args.data_root)
    sequence_length = 3600
    v3_model = load_v3_model(args.v3_model, sequence_length, 3)

    print("\nPreparing train split")
    train_data = prepare_split(train_manifest, v3_model, norm_stats, args.smooth_window, args.batch_size)
    print("\nPreparing val split")
    val_data = prepare_split(val_manifest, v3_model, norm_stats, args.smooth_window, args.batch_size)
    print("\nPreparing test split")
    test_data = prepare_split(test_manifest, v3_model, norm_stats, args.smooth_window, args.batch_size)

    X_train, y_train, masks_train = train_data[0], train_data[1], train_data[2]
    X_val, y_val, masks_val = val_data[0], val_data[1], val_data[2]
    X_test, masks_test, scales_test, filtered_test, clean_test = (
        test_data[0],
        test_data[2],
        test_data[3],
        test_data[4],
        test_data[5],
    )

    model = create_slow_corrector(X_train.shape[1], X_train.shape[2])
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=slow_mae_loss, metrics=["mae"])
    print(f"Slow corrector input shape: {X_train.shape[1:]}")

    Path("models").mkdir(exist_ok=True)
    Path("results/training").mkdir(parents=True, exist_ok=True)

    suffix = "_fast" if args.fast else "_complete"
    best_model_path = f"models/model_slow_corrector_v1_best{suffix}.keras"
    final_model_path = f"models/model_slow_corrector_v1_final{suffix}.keras"
    results_path = Path("results/training") / f"slow_corrector_v1_results{suffix}.json"
    history_path = Path("results/training") / f"slow_corrector_v1_history{suffix}.png"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        sample_weight=masks_train,
        validation_data=(X_val, y_val, masks_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start
    plot_history(history, history_path)

    print("\nEvaluating slow corrector")
    metrics = evaluate_corrector(
        model,
        X_test,
        scales_test,
        filtered_test,
        clean_test,
        masks_test,
        args.smooth_window,
        args.batch_size,
    )

    model.save(final_model_path)

    result = {
        "model_type": "slow_drift_corrector_v1_on_frozen_v3",
        "v3_model": args.v3_model,
        "smooth_window": args.smooth_window,
        "fast": args.fast,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "seed": args.seed,
        },
        "training_time_minutes": training_time / 60.0,
        "epochs_trained": len(history.history["loss"]),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "metrics": metrics,
    }

    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    baseline = metrics["baseline"]
    corrected = metrics["slow_corrected"]
    oracle = metrics["oracle"]
    print("\n=== RESULTS ===")
    print(f"Baseline RMS XY drift: {baseline['rms_drift_xy_m']:.3f} m")
    print(f"Corrected RMS XY drift: {corrected['rms_drift_xy_m']:.3f} m")
    print(f"Oracle RMS XY drift: {oracle['rms_drift_xy_m']:.3f} m")
    print(f"Baseline mean final XY drift: {baseline['mean_final_drift_xy_m']:.3f} m")
    print(f"Corrected mean final XY drift: {corrected['mean_final_drift_xy_m']:.3f} m")
    print(f"Oracle mean final XY drift: {oracle['mean_final_drift_xy_m']:.3f} m")
    print(f"Baseline RMS Z drift: {baseline['rms_z_drift_m']:.3f} m")
    print(f"Corrected RMS Z drift: {corrected['rms_z_drift_m']:.3f} m")
    print(f"Oracle RMS Z drift: {oracle['rms_z_drift_m']:.3f} m")
    print(f"Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
