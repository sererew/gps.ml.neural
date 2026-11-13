#!/usr/bin/env python3
"""
Train a two-stage contextual residual cascade model.

Stage 1, fast:
    input_context -> full_residual_delta

Stage 2, slow:
    fast_filtered_context -> slow_residual_delta

Inference:
    fast_delta = noisy_delta + fast_residual_delta
    filtered_delta = fast_delta + slow_residual_delta
"""

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Add, Conv1D, Dense, Dropout, Input, LayerNormalization, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


MODEL_TAG = "context_cascade_v2"
FAST_MODEL_TAG = f"{MODEL_TAG}_fast"
SLOW_MODEL_TAG = f"{MODEL_TAG}_slow"
DELTA_FEATURES = ["dx", "dy", "dz"]
CONTEXT_DATASET_SCRIPT = Path(__file__).resolve().parent / "5_generate_input_dataset_context_v1.py"


def load_module(name, path):
    """Load a Python script as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


context_features = load_module("context_dataset_v1", CONTEXT_DATASET_SCRIPT)


physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU available: {physical_devices[0]}")
else:
    print("Training on CPU")


class ContextTrackDataset:
    """Dataset loader for contextual pre-split GPS track data."""

    def __init__(self, data_dir="data/input_context_v1"):
        self.data_dir = Path(data_dir)
        self.train_manifest_path = self.data_dir / "train" / "manifest_train.csv"
        self.val_manifest_path = self.data_dir / "val" / "manifest_val.csv"
        self.test_manifest_path = self.data_dir / "test" / "manifest_test.csv"
        self.norm_stats_path = self.data_dir / "norm_stats_train.json"

        required_files = [
            self.train_manifest_path,
            self.val_manifest_path,
            self.test_manifest_path,
            self.norm_stats_path,
        ]
        missing_files = [path for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")

        with open(self.norm_stats_path, "r", encoding="utf-8") as f:
            self.norm_stats = json.load(f)

        self.input_features = self.norm_stats.get("input_features")
        if not self.input_features:
            self.input_features = None
        self.label_features = self.norm_stats.get("label_features", DELTA_FEATURES)

        print(f"Context dataset loaded from {data_dir}")
        print(f"  - Stats: {self.norm_stats_path}")
        print(f"  - Input features: {self.input_features or 'from CSV columns'}")
        print(f"  - Label features: {self.label_features}")

    def _normalize_manifest_paths(self, manifest_df):
        """Normalize path separators for cross-platform compatibility."""
        for col in ["slice_path", "label_path", "mask_path"]:
            if col in manifest_df.columns:
                manifest_df[col] = manifest_df[col].fillna("").astype(str).apply(lambda p: p.replace("\\", "/"))

    def load_by_sets(self):
        """Load train, validation, and test splits."""
        train_manifest = pd.read_csv(self.train_manifest_path)
        val_manifest = pd.read_csv(self.val_manifest_path)
        test_manifest = pd.read_csv(self.test_manifest_path)

        self._normalize_manifest_paths(train_manifest)
        self._normalize_manifest_paths(val_manifest)
        self._normalize_manifest_paths(test_manifest)

        print("Loading contextual data by split:")
        print(f"  Train: {len(train_manifest)} windows")
        print(f"  Val: {len(val_manifest)} windows")
        print(f"  Test: {len(test_manifest)} windows")

        X_train, y_train, masks_train = self._load_data_batch(train_manifest)
        X_val, y_val, masks_val = self._load_data_batch(val_manifest)
        X_test, y_test, masks_test = self._load_data_batch(test_manifest)

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError("Could not load one or more dataset splits")

        print(f"Data loaded - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test)

    def load_window_data(self, row):
        """Load one contextual window."""
        def _to_path(value):
            return Path(str(value).replace("\\", "/")).expanduser()

        input_data = pd.read_csv(_to_path(row["slice_path"]))
        label_data = pd.read_csv(_to_path(row["label_path"]))
        mask_data = pd.read_csv(_to_path(row["mask_path"]))

        input_features = self.input_features
        if input_features is None:
            input_features = [col for col in input_data.columns if col != "time"]

        missing_input = [col for col in input_features if col not in input_data.columns]
        missing_label = [col for col in self.label_features if col not in label_data.columns]
        if missing_input:
            raise ValueError(f"Missing input columns {missing_input} in {row['slice_path']}")
        if missing_label:
            raise ValueError(f"Missing label columns {missing_label} in {row['label_path']}")

        X = input_data[input_features].to_numpy(dtype=np.float32)
        y = label_data[self.label_features].to_numpy(dtype=np.float32)
        mask = mask_data["mask"].to_numpy(dtype=np.float32)
        return X, y, mask

    def _load_data_batch(self, manifest_subset):
        """Load a complete split from a manifest."""
        X_list, y_list, masks_list = [], [], []
        for _, row in manifest_subset.iterrows():
            try:
                X, y, mask = self.load_window_data(row)
                X_list.append(X)
                y_list.append(y)
                masks_list.append(mask)
            except Exception as exc:
                print(f"Error loading window {row['slice_path']}: {exc}")
        return np.array(X_list), np.array(y_list), np.array(masks_list)


def residual_mae_loss(y_true, y_pred):
    """Per-timestep MAE over residual delta correction channels."""
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)


def normalize_feature_frame(feature_frame, norm_stats, input_features):
    """Normalize contextual features with training statistics."""
    missing = [col for col in input_features if col not in feature_frame.columns]
    if missing:
        raise ValueError(f"Missing computed input features: {missing}")

    normalized = np.zeros((len(feature_frame), len(input_features)), dtype=np.float32)
    for i, col in enumerate(input_features):
        mean = norm_stats["mean"].get(col, 0.0)
        std = norm_stats["std"].get(col, 1.0)
        values = feature_frame[col].to_numpy(dtype=np.float64)
        normalized[:, i] = 0.0 if std <= 1e-12 else (values - mean) / std
    return normalized


def denormalize_delta_channels(delta_norm, norm_stats):
    """Denormalize normalized dx, dy, dz channels."""
    delta = delta_norm.copy()
    for i, feature in enumerate(DELTA_FEATURES):
        delta[..., i] = delta[..., i] * norm_stats["std"][feature] + norm_stats["mean"][feature]
    return delta


def build_context_from_delta_norm(delta_norm, norm_stats, input_features, masks=None):
    """Build normalized context features from normalized delta channels."""
    delta_meters = denormalize_delta_channels(delta_norm, norm_stats)
    out = np.zeros((delta_norm.shape[0], delta_norm.shape[1], len(input_features)), dtype=np.float32)

    for i in range(delta_meters.shape[0]):
        if masks is None:
            first = 0
            last = delta_meters.shape[1]
        else:
            valid_idx = np.where(masks[i].astype(bool))[0]
            if len(valid_idx) == 0:
                continue
            first = int(valid_idx[0])
            last = int(valid_idx[-1]) + 1

        dx = delta_meters[i, first:last, 0]
        dy = delta_meters[i, first:last, 1]
        dz = delta_meters[i, first:last, 2]
        x = np.cumsum(dx)
        y = np.cumsum(dy)
        feature_frame = context_features.build_feature_frame(dx, dy, dz, x, y)
        out[i, first:last, :] = normalize_feature_frame(feature_frame, norm_stats, input_features)

    return out


def build_fast_target(X, y, masks):
    """Build the full residual target for the fast model."""
    total_residual = (y - X[:, :, :3]).astype(np.float32)
    total_residual[~masks.astype(bool)] = 0.0
    return total_residual


def build_slow_data(X, y, masks, fast_residual, norm_stats, input_features):
    """Build slow-model input and residual target from actual fast predictions."""
    fast_delta = (X[:, :, :3] + fast_residual).astype(np.float32)
    slow_input = build_context_from_delta_norm(fast_delta, norm_stats, input_features, masks=masks)
    slow_target = (y - fast_delta).astype(np.float32)
    slow_target[~masks.astype(bool)] = 0.0
    return slow_input, slow_target.astype(np.float32)


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    """Build one residual dilated temporal convolution block."""
    residual = x
    y = Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate, activation=None)(x)
    y = LayerNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(dropout_rate)(y)
    y = Conv1D(filters, 1, padding="same", activation=None)(y)
    y = LayerNormalization()(y)
    y = Add()([residual, y])
    return Activation("relu")(y)


def create_fast_model(sequence_length=3600, n_features=15, output_features=3):
    """Create a short-context fast residual model."""
    inputs = Input(shape=(sequence_length, n_features))
    x = Conv1D(64, 5, padding="same", activation=None)(inputs)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(64, 3, padding="same", activation=None)(x)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(output_features, activation="linear")(x)
    return Model(inputs=inputs, outputs=outputs, name=FAST_MODEL_TAG)


def create_slow_model(sequence_length=3600, n_features=15, output_features=3):
    """Create a dilated TCN slow residual model."""
    inputs = Input(shape=(sequence_length, n_features))
    x = Conv1D(64, 1, padding="same", activation=None)(inputs)
    x = LayerNormalization()(x)
    x = Activation("relu")(x)

    for dilation_rate in [1, 2, 4, 8, 16, 32, 64, 128]:
        x = tcn_block(x, filters=64, kernel_size=5, dilation_rate=dilation_rate, dropout_rate=0.1)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_features, activation="linear")(x)
    return Model(inputs=inputs, outputs=outputs, name=SLOW_MODEL_TAG)


def plot_training_histories(fast_history, slow_history, save_path):
    """Save a combined training history plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(fast_history.history["loss"], marker="o", markersize=3, label="Train Loss")
    axes[0].plot(fast_history.history["val_loss"], marker="o", markersize=3, label="Val Loss")
    axes[0].set_title("Fast Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(slow_history.history["loss"], marker="o", markersize=3, label="Train Loss")
    axes[1].plot(slow_history.history["val_loss"], marker="o", markersize=3, label="Val Loss")
    axes[1].set_title("Slow Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    lr_key = "lr" if "lr" in slow_history.history else "learning_rate" if "learning_rate" in slow_history.history else None
    if lr_key:
        axes[2].plot(slow_history.history[lr_key], marker="o", markersize=3, label="Slow LR")
        axes[2].set_yscale("log")
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No LR data", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {save_path}")


def denormalize_deltas(values, norm_stats):
    """Denormalize dx, dy, dz values."""
    out = values.copy()
    for i, feature in enumerate(DELTA_FEATURES):
        out[..., i] = out[..., i] * norm_stats["std"][feature] + norm_stats["mean"][feature]
    return out


def predict_cascade(fast_model, slow_model, X, norm_stats, input_features, masks=None, batch_size=32):
    """Predict final normalized deltas with the two-stage cascade."""
    fast_residual = fast_model.predict(X, batch_size=batch_size, verbose=1)
    fast_delta = X[:, :, :3] + fast_residual
    slow_input = build_context_from_delta_norm(fast_delta, norm_stats, input_features, masks=masks)
    slow_residual = slow_model.predict(slow_input, batch_size=batch_size, verbose=1)
    return fast_delta + slow_residual


def calculate_drift_metrics(y_pred_meters, y_test_meters, masks_test):
    """Compute spatial drift metrics by integrating deltas into positions."""
    valid_mask = masks_test.astype(bool)
    y_pred_masked = y_pred_meters.copy()
    y_test_masked = y_test_meters.copy()

    for i in range(y_pred_meters.shape[0]):
        y_pred_masked[i, ~valid_mask[i], :] = 0
        y_test_masked[i, ~valid_mask[i], :] = 0

    pos_pred = np.cumsum(y_pred_masked, axis=1)
    pos_true = np.cumsum(y_test_masked, axis=1)

    final_drifts = []
    final_z_errors = []
    all_drifts = []
    all_z_errors = []
    length_diffs = []
    pred_total_lengths = []
    true_total_lengths = []

    for i in range(pos_pred.shape[0]):
        valid_times = np.where(valid_mask[i])[0]
        if len(valid_times) == 0:
            continue

        last_valid_t = valid_times[-1]
        final_drifts.append(float(np.linalg.norm(pos_pred[i, last_valid_t, :2] - pos_true[i, last_valid_t, :2])))
        final_z_errors.append(float(pos_pred[i, last_valid_t, 2] - pos_true[i, last_valid_t, 2]))

        drift_xy = np.linalg.norm(pos_pred[i, valid_times, :2] - pos_true[i, valid_times, :2], axis=1)
        z_error = pos_pred[i, valid_times, 2] - pos_true[i, valid_times, 2]
        all_drifts.extend(drift_xy.tolist())
        all_z_errors.extend(z_error.tolist())

        pred_lengths = np.linalg.norm(y_pred_masked[i, valid_times, :2], axis=1)
        true_lengths = np.linalg.norm(y_test_masked[i, valid_times, :2], axis=1)
        pred_total = float(np.sum(pred_lengths))
        true_total = float(np.sum(true_lengths))
        length_diffs.append(abs(pred_total - true_total))
        pred_total_lengths.append(pred_total)
        true_total_lengths.append(true_total)

    drift_final_mean_m = float(np.mean(final_drifts)) if final_drifts else 0.0
    drift_rms_m = float(np.sqrt(np.mean(np.square(all_drifts)))) if all_drifts else 0.0
    z_final_mean_m = float(np.mean(final_z_errors)) if final_z_errors else 0.0
    z_final_abs_mean_m = float(np.mean(np.abs(final_z_errors))) if final_z_errors else 0.0
    z_rms_m = float(np.sqrt(np.mean(np.square(all_z_errors)))) if all_z_errors else 0.0
    length_diff_m = float(np.mean(length_diffs)) if length_diffs else 0.0
    pred_length_mean_m = float(np.mean(pred_total_lengths)) if pred_total_lengths else 0.0
    true_length_mean_m = float(np.mean(true_total_lengths)) if true_total_lengths else 0.0
    length_diff_pct = (length_diff_m / true_length_mean_m * 100.0) if true_length_mean_m > 0 else 0.0
    drift_final_pct = (drift_final_mean_m / true_length_mean_m * 100.0) if true_length_mean_m > 0 else 0.0
    drift_rms_pct = (drift_rms_m / true_length_mean_m * 100.0) if true_length_mean_m > 0 else 0.0

    return {
        "drift_final_mean_m": drift_final_mean_m,
        "drift_rms_m": drift_rms_m,
        "drift_final_xy_mean_m": drift_final_mean_m,
        "drift_rms_xy_m": drift_rms_m,
        "z_final_mean_m": z_final_mean_m,
        "z_final_abs_mean_m": z_final_abs_mean_m,
        "z_rms_m": z_rms_m,
        "length_diff_m": length_diff_m,
        "length_diff_xy_m": length_diff_m,
        "pred_length_mean_m": pred_length_mean_m,
        "true_length_mean_m": true_length_mean_m,
        "length_diff_pct": length_diff_pct,
        "length_diff_xy_pct": length_diff_pct,
        "drift_final_pct": drift_final_pct,
        "drift_rms_pct": drift_rms_pct,
        "drift_final_xy_pct": drift_final_pct,
        "drift_rms_xy_pct": drift_rms_pct,
    }


def evaluate_cascade(fast_model, slow_model, X_test, y_test, masks_test, norm_stats):
    """Evaluate the complete cascade."""
    print("\n=== CONTEXT CASCADE v2 MODEL EVALUATION ===")
    input_features = norm_stats.get("input_features")
    y_pred = predict_cascade(fast_model, slow_model, X_test, norm_stats, input_features, masks=masks_test, batch_size=32)

    valid_positions = masks_test.astype(bool)
    y_valid = y_test[valid_positions]
    pred_valid = y_pred[valid_positions]

    mae_dx_norm = float(np.mean(np.abs(y_valid[:, 0] - pred_valid[:, 0])))
    mae_dy_norm = float(np.mean(np.abs(y_valid[:, 1] - pred_valid[:, 1])))
    mae_dz_norm = float(np.mean(np.abs(y_valid[:, 2] - pred_valid[:, 2])))
    xy_step_error_norm = np.linalg.norm(y_valid[:, :2] - pred_valid[:, :2], axis=1)
    mae_xy_norm = float(np.mean(xy_step_error_norm))
    rmse_xy_norm = float(np.sqrt(np.mean(xy_step_error_norm**2)))
    rmse_z_norm = float(np.sqrt(np.mean((y_valid[:, 2] - pred_valid[:, 2]) ** 2)))
    mae_total_norm = float(np.mean([mae_dx_norm, mae_dy_norm, mae_dz_norm]))

    y_pred_meters = denormalize_deltas(y_pred, norm_stats)
    y_test_meters = denormalize_deltas(y_test, norm_stats)
    y_valid_m = y_test_meters[valid_positions]
    pred_valid_m = y_pred_meters[valid_positions]

    mae_dx_meters = float(np.mean(np.abs(y_valid_m[:, 0] - pred_valid_m[:, 0])))
    mae_dy_meters = float(np.mean(np.abs(y_valid_m[:, 1] - pred_valid_m[:, 1])))
    mae_dz_meters = float(np.mean(np.abs(y_valid_m[:, 2] - pred_valid_m[:, 2])))
    xy_step_error_meters = np.linalg.norm(y_valid_m[:, :2] - pred_valid_m[:, :2], axis=1)
    mae_xy_meters = float(np.mean(xy_step_error_meters))
    rmse_xy_meters = float(np.sqrt(np.mean(xy_step_error_meters**2)))
    rmse_z_meters = float(np.sqrt(np.mean((y_valid_m[:, 2] - pred_valid_m[:, 2]) ** 2)))
    mae_total_meters = float(np.mean([mae_dx_meters, mae_dy_meters, mae_dz_meters]))

    print(f"MAE dx (meters): {mae_dx_meters:.4f} m")
    print(f"MAE dy (meters): {mae_dy_meters:.4f} m")
    print(f"MAE XY step (meters): {mae_xy_meters:.4f} m")
    print(f"RMSE XY step (meters): {rmse_xy_meters:.4f} m")
    print(f"MAE dz (meters): {mae_dz_meters:.4f} m")
    print(f"RMSE z (meters): {rmse_z_meters:.4f} m")
    print(f"MAE total (meters): {mae_total_meters:.4f} m")

    drift_metrics = calculate_drift_metrics(y_pred_meters, y_test_meters, masks_test)
    drift_metrics = {key: float(value) for key, value in drift_metrics.items()}

    print("\n=== SPATIAL DRIFT METRICS ===")
    print(f"Mean final drift: {drift_metrics['drift_final_mean_m']:.4f} m")
    print(f"RMS drift: {drift_metrics['drift_rms_m']:.4f} m")
    print(f"Trajectory length difference: {drift_metrics['length_diff_m']:.4f} m")
    print(f"Relative final drift: {drift_metrics['drift_final_pct']:.2f}%")
    print(f"Relative RMS drift: {drift_metrics['drift_rms_pct']:.2f}%")

    results = {
        "mae_dx_norm": mae_dx_norm,
        "mae_dy_norm": mae_dy_norm,
        "mae_dz_norm": mae_dz_norm,
        "mae_xy_norm": mae_xy_norm,
        "rmse_xy_norm": rmse_xy_norm,
        "rmse_z_norm": rmse_z_norm,
        "mae_total_norm": mae_total_norm,
        "mae_dx_meters": mae_dx_meters,
        "mae_dy_meters": mae_dy_meters,
        "mae_dz_meters": mae_dz_meters,
        "mae_xy_meters": mae_xy_meters,
        "rmse_xy_meters": rmse_xy_meters,
        "rmse_z_meters": rmse_z_meters,
        "mae_total_meters": mae_total_meters,
        "residual_training": True,
    }
    results.update(drift_metrics)
    return results


def train_model(dataset, model_config, fast_mode=False):
    """Train the fast and slow cascade models."""
    print("\n=== CONTEXT CASCADE v2 TRAINING ===")
    (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test) = dataset.load_by_sets()
    input_features = dataset.input_features or dataset.norm_stats.get("input_features")
    if not input_features:
        raise ValueError("Context norm stats must include input_features")

    if fast_mode:
        print("FAST MODE: limiting contextual data")
        max_samples = 100
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_train, y_train, masks_train = X_train[indices], y_train[indices], masks_train[indices]
        if len(X_val) > max_samples // 4:
            indices = np.random.choice(len(X_val), max_samples // 4, replace=False)
            X_val, y_val, masks_val = X_val[indices], y_val[indices], masks_val[indices]
        if len(X_test) > max_samples // 4:
            indices = np.random.choice(len(X_test), max_samples // 4, replace=False)
            X_test, y_test, masks_test = X_test[indices], y_test[indices], masks_test[indices]
        print(f"Limited data - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    y_train_fast = build_fast_target(X_train, y_train, masks_train)
    y_val_fast = build_fast_target(X_val, y_val, masks_val)

    fast_model = create_fast_model(sequence_length=X_train.shape[1], n_features=X_train.shape[2], output_features=3)
    fast_model.compile(optimizer=Adam(learning_rate=model_config["learning_rate"]), loss=residual_mae_loss, metrics=["mae"])

    callbacks_fast = [
        EarlyStopping(monitor="val_loss", patience=model_config["patience"], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    print("Training fast full-residual model...")
    start_time = time.time()
    fast_history = fast_model.fit(
        X_train,
        y_train_fast,
        sample_weight=masks_train,
        batch_size=model_config["batch_size"],
        epochs=model_config["epochs"],
        validation_data=(X_val, y_val_fast, masks_val),
        callbacks=callbacks_fast,
        verbose=1,
    )
    fast_time = time.time() - start_time

    print("Predicting fast outputs for slow-model dataset...")
    fast_train_pred = fast_model.predict(X_train, batch_size=32, verbose=1)
    fast_val_pred = fast_model.predict(X_val, batch_size=32, verbose=1)
    X_train_slow, y_train_slow = build_slow_data(
        X_train, y_train, masks_train, fast_train_pred, dataset.norm_stats, input_features
    )
    X_val_slow, y_val_slow = build_slow_data(
        X_val, y_val, masks_val, fast_val_pred, dataset.norm_stats, input_features
    )
    slow_model = create_slow_model(sequence_length=X_train_slow.shape[1], n_features=X_train_slow.shape[2], output_features=3)
    slow_model.compile(optimizer=Adam(learning_rate=model_config["learning_rate"]), loss=residual_mae_loss, metrics=["mae"])

    callbacks_slow = [
        EarlyStopping(monitor="val_loss", patience=model_config["patience"], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]

    print("Training slow low-frequency model...")
    start_time = time.time()
    slow_history = slow_model.fit(
        X_train_slow,
        y_train_slow,
        sample_weight=masks_train,
        batch_size=model_config["batch_size"],
        epochs=model_config["epochs"],
        validation_data=(X_val_slow, y_val_slow, masks_val),
        callbacks=callbacks_slow,
        verbose=1,
    )
    slow_time = time.time() - start_time
    training_time = fast_time + slow_time

    print("\nTraining completed:")
    print(f"  - Fast epochs trained: {len(fast_history.history['loss'])}/{model_config['epochs']}")
    print(f"  - Slow epochs trained: {len(slow_history.history['loss'])}/{model_config['epochs']}")
    print(f"  - Total time: {training_time / 60:.2f} minutes")
    print(f"  - Final fast val loss: {fast_history.history['val_loss'][-1]:.6f}")
    print(f"  - Final slow val loss: {slow_history.history['val_loss'][-1]:.6f}")

    results_dir = Path("results") / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_training_histories(fast_history, slow_history, str(results_dir / f"training_history_{MODEL_TAG}.png"))

    print("\nEvaluating on TEST split...")
    test_metrics = evaluate_cascade(fast_model, slow_model, X_test, y_test, masks_test, dataset.norm_stats)

    os.makedirs("models", exist_ok=True)
    fast_model_path = f"models/model_final_{FAST_MODEL_TAG}.keras"
    slow_model_path = f"models/model_final_{SLOW_MODEL_TAG}.keras"
    fast_model.save(fast_model_path)
    slow_model.save(slow_model_path)
    print(f"Fast model saved to: {fast_model_path}")
    print(f"Slow model saved to: {slow_model_path}")

    fast_weights_path = f"models/model_final_{FAST_MODEL_TAG}.weights.h5"
    slow_weights_path = f"models/model_final_{SLOW_MODEL_TAG}.weights.h5"
    fast_model.save_weights(fast_weights_path)
    slow_model.save_weights(slow_weights_path)
    print(f"Fast model weights saved to: {fast_weights_path}")
    print(f"Slow model weights saved to: {slow_weights_path}")

    histories = {
        "fast": fast_history,
        "slow": slow_history,
    }
    extra = {
        "fast_epochs_trained": len(fast_history.history["loss"]),
        "slow_epochs_trained": len(slow_history.history["loss"]),
        "final_fast_train_loss": float(fast_history.history["loss"][-1]),
        "final_fast_val_loss": float(fast_history.history["val_loss"][-1]),
        "final_slow_train_loss": float(slow_history.history["loss"][-1]),
        "final_slow_val_loss": float(slow_history.history["val_loss"][-1]),
    }
    return fast_model, slow_model, histories, test_metrics, training_time, extra


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a contextual GPS correction cascade",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", default="data/input_context_v1", help="Context dataset root")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs per stage")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--fast", action="store_true", help="Fast mode for quick checks")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible comparisons")
    return parser.parse_args()


def main():
    print("=== GPS CONTEXT CASCADE v2 MODEL TRAINING ===\n")
    try:
        args = parse_args()
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)

        model_config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "fast_target": "clean_delta - noisy_delta",
            "slow_input_features": "context features recomputed from fast_delta",
            "slow_target": "clean_delta - fast_delta",
        }
        if args.fast:
            print("FAST MODE ENABLED")
            model_config["epochs"] = min(model_config["epochs"], 10)
            model_config["batch_size"] = min(model_config["batch_size"], 16)
            model_config["patience"] = min(model_config["patience"], 5)

        print("Configuration:")
        print(f"  - Data directory: {args.data_root}")
        print(f"  - Epochs per stage: {model_config['epochs']}")
        print(f"  - Batch size: {model_config['batch_size']}")
        print(f"  - Learning rate: {model_config['learning_rate']}")
        print(f"  - Patience: {model_config['patience']}")
        print(f"  - Seed: {args.seed}")

        dataset = ContextTrackDataset(data_dir=args.data_root)
        _, _, histories, test_metrics, training_time, extra = train_model(dataset, model_config, fast_mode=args.fast)

        mode_suffix = "_fast" if args.fast else "_complete"
        results_dir = Path("results") / "training"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"training_results_{MODEL_TAG}{mode_suffix}.json"

        results = {
            "config": model_config,
            "model_type": "contextual_fast_slow_residual_cascade",
            "model_tag": MODEL_TAG,
            "fast_model_tag": FAST_MODEL_TAG,
            "slow_model_tag": SLOW_MODEL_TAG,
            "input_features": dataset.input_features,
            "label_features": dataset.label_features,
            "test_metrics": test_metrics,
            "training_time_minutes": training_time / 60,
        }
        results.update(extra)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\n=== FINAL RESULTS ===")
        print(f"Mode: {'FAST' if args.fast else 'FULL'}")
        print(f"MAE total (meters): {test_metrics['mae_total_meters']:.4f} m")
        print(f"Mean final XY drift: {test_metrics['drift_final_mean_m']:.4f} m")
        print(f"RMS XY drift: {test_metrics['drift_rms_m']:.4f} m")
        print(f"Total time: {training_time / 60:.2f} minutes")
        print(f"Results saved to: {results_file}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
