#!/usr/bin/env python3
"""
Train an XY-only recording-level slow-drift corrector on top of frozen v3.

This v2.1 corrector keeps the recording-level structure from v2, but it only
predicts horizontal slow drift. Z is left unchanged from the frozen v3 output.

The predicted control points are interpolated back to each window timeline and
subtracted from the v3 reconstructed XY positions.
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
from tensorflow.keras.layers import (
    AveragePooling1D,
    Conv1D,
    Dense,
    Dropout,
    GRU,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    Masking,
    Multiply,
    Reshape,
    TimeDistributed,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


FEATURES = ["dx", "dy", "dz"]
SLOW_FEATURES = 9
CONTROL_DIMS = 2


def create_v3_model(sequence_length, n_features):
    """Create the residual v3 architecture for loading frozen weights."""
    return Sequential(
        [
            Input(shape=(sequence_length, n_features), name="input_layer"),
            Masking(mask_value=0.0, name="masking"),
            tf.keras.layers.LSTM(
                128,
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.0,
                name="lstm",
            ),
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


def valid_positions_from_deltas(deltas, mask):
    """Return accumulated positions for valid timesteps only."""
    return np.cumsum(deltas[mask.astype(bool)], axis=0)


def length_xy_from_positions(positions):
    """Compute horizontal path length from positions."""
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)))


def build_window_features(filtered_norm, filtered_meters, clean_meters, masks, manifest, smooth_window):
    """Build per-window features from recording-level accumulated positions."""
    n_windows, sequence_length, _ = filtered_meters.shape
    x_slow = np.zeros((n_windows, sequence_length, SLOW_FEATURES), dtype=np.float32)
    slow_errors = np.zeros((n_windows, sequence_length, 3), dtype=np.float32)
    time_norms = np.zeros((n_windows, sequence_length), dtype=np.float32)
    pos_filtered_windows = np.zeros((n_windows, sequence_length, 3), dtype=np.float32)
    pos_clean_windows = np.zeros((n_windows, sequence_length, 3), dtype=np.float32)
    scales = np.ones((n_windows, 2), dtype=np.float32)

    manifest_with_index = manifest.reset_index().sort_values(["grabacion", "t_start"])

    for _, group in manifest_with_index.groupby("grabacion", sort=False):
        recording_start = float(group["recording_start"].min())
        recording_end = float(group["recording_end"].max())
        recording_duration = max(recording_end - recording_start, 1.0)

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
            continue

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

        pos_filtered = np.cumsum(filtered_unique, axis=0)
        pos_clean = np.cumsum(clean_unique, axis=0)
        slow_error = moving_average(pos_filtered - pos_clean, smooth_window)
        cumulative_distance = np.cumsum(np.linalg.norm(filtered_unique[:, :2], axis=1))
        length_scale = max(float(cumulative_distance[-1]), 1.0)
        z_scale = max(float(np.ptp(pos_filtered[:, 2])), 10.0)

        for _, row in group.iterrows():
            window_idx = int(row["index"])
            valid = masks[window_idx].astype(bool)
            valid_indices = np.where(valid)[0]
            n_valid = len(valid_indices)
            if n_valid == 0:
                continue

            timestamps = float(row["t_start"]) + np.arange(n_valid, dtype=np.float64)
            unique_indices = np.searchsorted(unique_times, timestamps)
            unique_indices = np.clip(unique_indices, 0, len(unique_times) - 1)

            local_t_norm = np.zeros(n_valid, dtype=np.float32)
            if n_valid > 1:
                local_t_norm = np.arange(n_valid, dtype=np.float32) / float(n_valid - 1)
            absolute_t_norm = np.clip((timestamps - recording_start) / recording_duration, 0.0, 1.0)
            distance_norm = cumulative_distance[unique_indices] / length_scale

            x_slow[window_idx, valid_indices, 0:3] = filtered_norm[window_idx, valid]
            x_slow[window_idx, valid_indices, 3] = pos_filtered[unique_indices, 0] / length_scale
            x_slow[window_idx, valid_indices, 4] = pos_filtered[unique_indices, 1] / length_scale
            x_slow[window_idx, valid_indices, 5] = pos_filtered[unique_indices, 2] / z_scale
            x_slow[window_idx, valid_indices, 6] = local_t_norm
            x_slow[window_idx, valid_indices, 7] = distance_norm
            x_slow[window_idx, valid_indices, 8] = absolute_t_norm

            slow_errors[window_idx, valid_indices] = slow_error[unique_indices]
            time_norms[window_idx, valid_indices] = absolute_t_norm.astype(np.float32)
            pos_filtered_windows[window_idx, valid_indices] = pos_filtered[unique_indices]
            pos_clean_windows[window_idx, valid_indices] = pos_clean[unique_indices]
            scales[window_idx] = [length_scale, z_scale]

    return x_slow, slow_errors, time_norms, pos_filtered_windows, pos_clean_windows, scales


def scaled_control_target(times, slow_values, length_scale, n_control_points):
    """Sample a scaled XY slow-error target at fixed control times."""
    valid = np.isfinite(times)
    if not np.any(valid):
        return np.zeros((n_control_points, CONTROL_DIMS), dtype=np.float32)

    times = np.asarray(times[valid], dtype=np.float64)
    slow_values = np.asarray(slow_values[valid, :CONTROL_DIMS], dtype=np.float64)
    order = np.argsort(times)
    times = times[order]
    slow_values = slow_values[order]

    unique_times, inverse = np.unique(times, return_inverse=True)
    averaged = np.zeros((len(unique_times), CONTROL_DIMS), dtype=np.float64)
    counts = np.zeros(len(unique_times), dtype=np.float64)
    for idx, inv in enumerate(inverse):
        averaged[inv] += slow_values[idx]
        counts[inv] += 1.0
    averaged /= counts[:, None]

    control_t = np.linspace(0.0, 1.0, n_control_points)
    target = np.zeros((n_control_points, CONTROL_DIMS), dtype=np.float32)
    if len(unique_times) == 1:
        sampled = np.repeat(averaged, n_control_points, axis=0)
    else:
        sampled = np.column_stack(
            [
                np.interp(control_t, unique_times, averaged[:, 0]),
                np.interp(control_t, unique_times, averaged[:, 1]),
            ]
        )
    target[:, 0] = sampled[:, 0] / length_scale
    target[:, 1] = sampled[:, 1] / length_scale
    return target


def build_recording_dataset(
    manifest,
    x_window,
    slow_errors,
    time_norms,
    window_scales,
    filtered_meters,
    clean_meters,
    masks,
    pos_filtered_windows,
    pos_clean_windows,
    max_windows,
    n_control_points,
):
    """Pack window tensors into recording-level examples."""
    grouped = list(manifest.reset_index().sort_values(["grabacion", "t_start"]).groupby("grabacion", sort=False))
    n_recordings = len(grouped)
    sequence_length = x_window.shape[1]

    x_rec = np.zeros((n_recordings, max_windows, sequence_length, SLOW_FEATURES), dtype=np.float32)
    window_mask = np.zeros((n_recordings, max_windows), dtype=np.float32)
    y_rec = np.zeros((n_recordings, n_control_points, CONTROL_DIMS), dtype=np.float32)
    rec_scales = np.ones((n_recordings, 2), dtype=np.float32)
    rec_meta = []

    for rec_idx, (recording, group) in enumerate(grouped):
        window_indices = group["index"].to_numpy(dtype=int)[:max_windows]
        n_windows = len(window_indices)
        if n_windows == 0:
            continue

        x_rec[rec_idx, :n_windows] = x_window[window_indices]
        window_mask[rec_idx, :n_windows] = 1.0

        valid_times = []
        valid_slow = []
        length_scale = max(float(np.mean(window_scales[window_indices, 0])), 1.0)
        z_scale = max(float(np.mean(window_scales[window_indices, 1])), 10.0)
        rec_scales[rec_idx] = [length_scale, z_scale]

        for window_idx in window_indices:
            valid = masks[window_idx].astype(bool)
            valid_times.append(time_norms[window_idx, valid])
            valid_slow.append(slow_errors[window_idx, valid])

        all_times = np.concatenate(valid_times) if valid_times else np.asarray([], dtype=np.float32)
        all_slow = np.concatenate(valid_slow) if valid_slow else np.zeros((0, 3), dtype=np.float32)
        y_rec[rec_idx] = scaled_control_target(all_times, all_slow, length_scale, n_control_points)

        rec_meta.append(
            {
                "recording": recording,
                "window_indices": window_indices,
                "filtered_meters": filtered_meters[window_indices],
                "clean_meters": clean_meters[window_indices],
                "pos_filtered": pos_filtered_windows[window_indices],
                "pos_clean": pos_clean_windows[window_indices],
                "masks": masks[window_indices],
                "slow_errors": slow_errors[window_indices],
                "time_norms": time_norms[window_indices],
                "scale": rec_scales[rec_idx].copy(),
            }
        )

    return x_rec, window_mask, y_rec, rec_scales, rec_meta


def prepare_split(manifest, v3_model, norm_stats, smooth_window, batch_size, max_windows, n_control_points):
    """Load one split, run v3, and build recording-level tensors."""
    x_norm, y_norm, masks = load_split_arrays(manifest)
    residual_pred = v3_model.predict(x_norm, batch_size=batch_size, verbose=1)
    filtered_norm = x_norm + residual_pred
    filtered_meters = denormalize_deltas(filtered_norm, norm_stats)
    clean_meters = denormalize_deltas(y_norm, norm_stats)
    x_window, slow_errors, time_norms, pos_filtered_windows, pos_clean_windows, window_scales = build_window_features(
        filtered_norm,
        filtered_meters,
        clean_meters,
        masks,
        manifest,
        smooth_window,
    )
    return build_recording_dataset(
        manifest,
        x_window,
        slow_errors,
        time_norms,
        window_scales,
        filtered_meters,
        clean_meters,
        masks,
        pos_filtered_windows,
        pos_clean_windows,
        max_windows,
        n_control_points,
    )


def create_recording_corrector(max_windows, sequence_length, n_features, n_control_points):
    """Create a compact Conv1D window encoder plus recording-level GRU model."""
    window_input = Input(shape=(max_windows, sequence_length, n_features), name="recording_windows")
    window_valid = Input(shape=(max_windows,), name="window_valid")

    encoder = Sequential(
        [
            Conv1D(16, kernel_size=31, strides=8, padding="same", activation="relu"),
            AveragePooling1D(pool_size=4),
            Conv1D(24, kernel_size=15, strides=4, padding="same", activation="relu"),
            GlobalAveragePooling1D(),
            Dense(32, activation="relu"),
        ],
        name="window_encoder",
    )

    summaries = TimeDistributed(encoder, name="encode_windows")(window_input)
    valid_expanded = Lambda(lambda value: tf.expand_dims(value, axis=-1), name="expand_window_valid")(window_valid)
    summaries = Multiply(name="mask_window_summaries")([summaries, valid_expanded])
    summaries = Masking(mask_value=0.0, name="mask_empty_windows")(summaries)
    encoded_recording = GRU(32, dropout=0.2, recurrent_dropout=0.0, name="recording_gru")(summaries)
    encoded_recording = Dense(32, activation="relu", name="control_dense")(encoded_recording)
    encoded_recording = Dropout(0.2, name="control_dropout")(encoded_recording)
    output = Dense(n_control_points * CONTROL_DIMS, activation="linear", name="control_output")(encoded_recording)
    output = Reshape((n_control_points, CONTROL_DIMS), name="slow_control_points_xy")(output)
    return Model(inputs=[window_input, window_valid], outputs=output, name="slow_corrector_v2_1_xy")


def control_mae_loss(y_true, y_pred):
    """MAE over scaled XY slow-drift control points."""
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)


def interpolate_control_points(control_points_m, times):
    """Interpolate XY control points to per-timestep slow error."""
    control_t = np.linspace(0.0, 1.0, len(control_points_m))
    return np.column_stack(
        [
            np.interp(times, control_t, control_points_m[:, 0]),
            np.interp(times, control_t, control_points_m[:, 1]),
        ]
    )


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


def evaluate_corrector(model, x_rec, window_mask, rec_meta, rec_scales, batch_size):
    """Evaluate baseline, XY-only slow corrector, and XY-only oracle."""
    pred_scaled = model.predict([x_rec, window_mask], batch_size=batch_size, verbose=1)

    baseline_positions = []
    corrected_positions = []
    oracle_positions = []
    true_positions = []

    for rec_idx, meta in enumerate(rec_meta):
        scale = rec_scales[rec_idx]
        control_m = pred_scaled[rec_idx].copy()
        control_m[:, 0] *= scale[0]
        control_m[:, 1] *= scale[0]

        for window_i in range(len(meta["window_indices"])):
            valid = meta["masks"][window_i].astype(bool)
            if not np.any(valid):
                continue
            pos_filtered = meta["pos_filtered"][window_i, valid]
            pos_true = meta["pos_clean"][window_i, valid]
            times = meta["time_norms"][window_i, valid]
            pred_slow_xy = interpolate_control_points(control_m, times)
            oracle_slow = meta["slow_errors"][window_i, valid]

            pos_corrected = pos_filtered.copy()
            pos_corrected[:, :CONTROL_DIMS] -= pred_slow_xy
            pos_oracle = pos_filtered.copy()
            pos_oracle[:, :CONTROL_DIMS] -= oracle_slow[:, :CONTROL_DIMS]

            baseline_positions.append(pos_filtered)
            corrected_positions.append(pos_corrected)
            oracle_positions.append(pos_oracle)
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
    ax.set_title("XY-Only Recording-Level Slow Corrector Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def limit_recordings(manifest, max_recordings, seed):
    """Limit a manifest to whole recordings for fast-mode checks."""
    recordings = sorted(manifest["grabacion"].unique())
    if len(recordings) <= max_recordings:
        return manifest.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    selected = set(rng.choice(recordings, size=max_recordings, replace=False))
    return manifest[manifest["grabacion"].isin(selected)].reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an XY-only recording-level slow-drift corrector on top of frozen v3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input", help="Input dataset root")
    parser.add_argument("--v3_model", default="models/model_best_v3.keras", help="Frozen v3 model path")
    parser.add_argument("--smooth_window", type=int, default=1800, help="Oracle slow target smoothing window")
    parser.add_argument("--control_points", type=int, default=5, help="Number of global XY slow-drift control points")
    parser.add_argument("--max_windows", type=int, default=20, help="Maximum windows kept per recording")
    parser.add_argument("--epochs", type=int, default=120, help="Maximum epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Recording batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--fast", action="store_true", help="Fast mode for quick checks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    print("=== XY-ONLY RECORDING-LEVEL SLOW DRIFT CORRECTOR TRAINING V2.1 ===")
    print(f"Data root: {args.data_root}")
    print(f"Frozen v3 model: {args.v3_model}")
    print(f"Smooth window: {args.smooth_window}")
    print(f"XY control points: {args.control_points}")
    print(f"Max windows per recording: {args.max_windows}")
    print(f"Fast mode: {args.fast}")

    train_manifest = load_manifest(args.data_root, "train")
    val_manifest = load_manifest(args.data_root, "val")
    test_manifest = load_manifest(args.data_root, "test")
    add_recording_ranges(train_manifest, val_manifest, test_manifest)

    if args.fast:
        args.epochs = min(args.epochs, 12)
        args.batch_size = min(args.batch_size, 4)
        args.patience = min(args.patience, 5)
        train_manifest = limit_recordings(train_manifest, 60, args.seed)
        val_manifest = limit_recordings(val_manifest, 20, args.seed)
        test_manifest = limit_recordings(test_manifest, 20, args.seed)
        print(
            "FAST MODE: "
            f"Train {train_manifest['grabacion'].nunique()} recordings, "
            f"Val {val_manifest['grabacion'].nunique()}, "
            f"Test {test_manifest['grabacion'].nunique()}"
        )

    norm_stats = load_norm_stats(args.data_root)
    sequence_length = 3600
    v3_model = load_v3_model(args.v3_model, sequence_length, 3)

    print("\nPreparing train split")
    train_data = prepare_split(
        train_manifest,
        v3_model,
        norm_stats,
        args.smooth_window,
        args.batch_size,
        args.max_windows,
        args.control_points,
    )
    print("\nPreparing val split")
    val_data = prepare_split(
        val_manifest,
        v3_model,
        norm_stats,
        args.smooth_window,
        args.batch_size,
        args.max_windows,
        args.control_points,
    )
    print("\nPreparing test split")
    test_data = prepare_split(
        test_manifest,
        v3_model,
        norm_stats,
        args.smooth_window,
        args.batch_size,
        args.max_windows,
        args.control_points,
    )

    x_train, window_mask_train, y_train = train_data[0], train_data[1], train_data[2]
    x_val, window_mask_val, y_val = val_data[0], val_data[1], val_data[2]
    x_test, window_mask_test, rec_scales_test, rec_meta_test = test_data[0], test_data[1], test_data[3], test_data[4]

    model = create_recording_corrector(
        args.max_windows,
        sequence_length,
        SLOW_FEATURES,
        args.control_points,
    )
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=control_mae_loss, metrics=["mae"])
    print(f"Recording input shape: {x_train.shape[1:]}")
    print(f"Training recordings: {x_train.shape[0]}")
    print(f"Validation recordings: {x_val.shape[0]}")
    model.summary()

    Path("models").mkdir(exist_ok=True)
    Path("results/training").mkdir(parents=True, exist_ok=True)

    suffix = "_fast" if args.fast else "_complete"
    best_model_path = f"models/model_slow_corrector_v2_1_xy_best{suffix}.keras"
    final_model_path = f"models/model_slow_corrector_v2_1_xy_final{suffix}.keras"
    results_path = Path("results/training") / f"slow_corrector_v2_1_xy_results{suffix}.json"
    history_path = Path("results/training") / f"slow_corrector_v2_1_xy_history{suffix}.png"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    start = time.time()
    history = model.fit(
        [x_train, window_mask_train],
        y_train,
        validation_data=([x_val, window_mask_val], y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    training_time = time.time() - start
    plot_history(history, history_path)

    print("\nEvaluating recording-level slow corrector")
    metrics = evaluate_corrector(
        model,
        x_test,
        window_mask_test,
        rec_meta_test,
        rec_scales_test,
        args.batch_size,
    )

    model.save(final_model_path)

    result = {
        "model_type": "xy_only_recording_level_slow_drift_corrector_v2_1_on_frozen_v3",
        "v3_model": args.v3_model,
        "smooth_window": args.smooth_window,
        "control_points": args.control_points,
        "control_dims": CONTROL_DIMS,
        "corrects_z": False,
        "max_windows": args.max_windows,
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
        "train_recordings": int(x_train.shape[0]),
        "val_recordings": int(x_val.shape[0]),
        "test_recordings": int(x_test.shape[0]),
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
    print(f"Corrected RMS Z drift (unchanged): {corrected['rms_z_drift_m']:.3f} m")
    print(f"Oracle RMS Z drift (unchanged): {oracle['rms_z_drift_m']:.3f} m")
    print(f"Baseline length diff XY: {baseline['mean_length_diff_xy_m']:.3f} m")
    print(f"Corrected length diff XY: {corrected['mean_length_diff_xy_m']:.3f} m")
    print(f"Oracle length diff XY: {oracle['mean_length_diff_xy_m']:.3f} m")
    print(f"Results saved to: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
