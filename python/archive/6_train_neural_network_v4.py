#!/usr/bin/env python3
"""
Train a residual neural network for GPS track correction with augmented features.

The network receives sequences of noisy deltas plus context features and
learns to predict the residual correction against the clean pattern track:

    residual = clean_delta - noisy_delta
    filtered_delta = noisy_delta + predicted_residual

Architecture:
- LSTM(128) with return_sequences=True
- Dense(64) with ReLU activation
- Output(3) with linear activation for corrections (dx, dy, dz)

Input: [batch, time, features], where features = [dx, dy, dz, t_norm, distance_norm, absolute_t_norm]
Output: [batch, time, features], where features = [correction_dx, correction_dy, correction_dz]

Uses a pre-split dataset (train/val/test) to avoid data leakage.
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import MeanAbsoluteError, Huber
import tensorflow.keras.backend as K

# GPU configuration if available.
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"GPU available: {physical_devices[0]}")
else:
    print("Training on CPU")

class GPSTrackDataset:
    """Dataset loader for pre-split GPS track data."""
    
    def __init__(self, data_dir="data/input"):
        self.data_dir = Path(data_dir)
        
        # Manifest paths by split and normalization statistics.
        self.train_manifest_path = self.data_dir / "train" / "manifest_train.csv"
        self.val_manifest_path = self.data_dir / "val" / "manifest_val.csv"
        self.test_manifest_path = self.data_dir / "test" / "manifest_test.csv"
        self.norm_stats_path = self.data_dir / "norm_stats_train.json"
        
        # Check that all required files exist.
        required_files = [
            self.train_manifest_path,
            self.val_manifest_path, 
            self.test_manifest_path,
            self.norm_stats_path
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing files: {missing_files}")
        
        # Load normalization statistics.
        with open(self.norm_stats_path, 'r') as f:
            self.norm_stats = json.load(f)
        
        print(f"Pre-split dataset loaded from {data_dir}")
        print(f"  - Train: {self.train_manifest_path}")
        print(f"  - Val: {self.val_manifest_path}")
        print(f"  - Test: {self.test_manifest_path}")
        print(f"  - Stats: {self.norm_stats_path}")

    def _normalize_manifest_paths(self, manifest_df):
        """Normalize path separators for cross-platform compatibility."""
        path_cols = ['slice_path', 'label_path', 'mask_path']
        for col in path_cols:
            if col in manifest_df.columns:
                manifest_df[col] = manifest_df[col].fillna('').astype(str).apply(lambda p: p.replace('\\', '/'))

    def _add_recording_ranges(self, *manifests):
        """Add recording-level time ranges used by absolute_t_norm."""
        combined = pd.concat(
            [df[['grabacion', 't_start', 't_end']] for df in manifests],
            ignore_index=True
        )
        ranges = combined.groupby('grabacion').agg(
            recording_start=('t_start', 'min'),
            recording_end=('t_end', 'max')
        )

        for manifest in manifests:
            manifest['recording_start'] = manifest['grabacion'].map(ranges['recording_start'])
            manifest['recording_end'] = manifest['grabacion'].map(ranges['recording_end'])

    def load_by_sets(self):
        """
        Load the dataset using pre-split manifests.
        
        Returns:
            tuple: (train_data, val_data, test_data), each as (X, y, masks)
        """
        # Load manifests by split.
        train_manifest = pd.read_csv(self.train_manifest_path)
        val_manifest = pd.read_csv(self.val_manifest_path)
        test_manifest = pd.read_csv(self.test_manifest_path)
        
        # Normalize paths in all manifests.
        self._normalize_manifest_paths(train_manifest)
        self._normalize_manifest_paths(val_manifest)
        self._normalize_manifest_paths(test_manifest)
        self._add_recording_ranges(train_manifest, val_manifest, test_manifest)
        
        print(f"Loading data by split:")
        print(f"  Train: {len(train_manifest)} windows")
        print(f"  Val: {len(val_manifest)} windows")
        print(f"  Test: {len(test_manifest)} windows")
        
        # Load each split.
        X_train, y_train, masks_train = self._load_data_batch(train_manifest)
        X_val, y_val, masks_val = self._load_data_batch(val_manifest)
        X_test, y_test, masks_test = self._load_data_batch(test_manifest)
        
        # Check that each split has data.
        if len(X_train) == 0:
            raise ValueError("Could not load training data. Check paths in manifest_train.csv")
        if len(X_val) == 0:
            raise ValueError("Could not load validation data. Check paths in manifest_val.csv")
        if len(X_test) == 0:
            raise ValueError("Could not load test data. Check paths in manifest_test.csv")
        
        print(f"Data loaded - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test)

    def load_window_data(self, row):
        """Load data for one specific window."""
        def _to_path(p):
            if pd.isna(p) or str(p) == '':
                return Path(p)
            pstr = str(p).replace('\\', '/')
            return Path(pstr).expanduser()

        slice_path = _to_path(row['slice_path'])
        input_data = pd.read_csv(slice_path)
        
        # Load label data (clean pattern).
        label_path = _to_path(row['label_path'])
        label_data = pd.read_csv(label_path)
        
        # Load mask.
        mask_path = _to_path(row['mask_path'])
        mask_data = pd.read_csv(mask_path)
        
        # Extract features (dx, dy, dz) and add context features.
        input_features = input_data[['dx', 'dy', 'dz']].values
        label_features = label_data[['dx', 'dy', 'dz']].values
        mask_values = mask_data['mask'].values
        input_features = self._augment_input_features(input_features, mask_values, row)
        
        return input_features, label_features, mask_values

    def _augment_input_features(self, input_features, mask_values, row):
        """Append t_norm, distance_norm, and absolute_t_norm to input features."""
        valid_mask = mask_values.astype(bool)
        n_steps = input_features.shape[0]
        n_valid = int(np.sum(valid_mask))
        context = np.zeros((n_steps, 3), dtype=input_features.dtype)

        if n_valid == 0:
            return np.concatenate([input_features, context], axis=1)

        valid_indices = np.where(valid_mask)[0]
        t_norm = np.zeros(n_valid, dtype=input_features.dtype)
        if n_valid > 1:
            t_norm = np.arange(n_valid, dtype=input_features.dtype) / float(n_valid - 1)

        dx_m = input_features[valid_indices, 0] * self.norm_stats['std']['dx'] + self.norm_stats['mean']['dx']
        dy_m = input_features[valid_indices, 1] * self.norm_stats['std']['dy'] + self.norm_stats['mean']['dy']
        step_lengths = np.sqrt(dx_m**2 + dy_m**2)
        cumulative_distance = np.cumsum(step_lengths)
        total_distance = cumulative_distance[-1]
        if total_distance > 0:
            distance_norm = cumulative_distance / total_distance
        else:
            distance_norm = np.zeros(n_valid, dtype=input_features.dtype)

        recording_start = float(row.get('recording_start', row.get('t_start', 0.0)))
        recording_end = float(row.get('recording_end', row.get('t_end', recording_start)))
        recording_duration = max(recording_end - recording_start, 1.0)
        timestamps = float(row['t_start']) + np.arange(n_valid, dtype=input_features.dtype)
        absolute_t_norm = np.clip((timestamps - recording_start) / recording_duration, 0.0, 1.0)

        context[valid_indices, 0] = t_norm
        context[valid_indices, 1] = distance_norm
        context[valid_indices, 2] = absolute_t_norm

        return np.concatenate([input_features, context], axis=1)
    
    def _load_data_batch(self, manifest_subset):
        """Load one batch of data from a manifest."""
        X_list, y_list, masks_list = [], [], []
        
        for idx, row in manifest_subset.iterrows():
            try:
                input_features, label_features, mask_values = self.load_window_data(row)
                X_list.append(input_features)
                y_list.append(label_features)
                masks_list.append(mask_values)
            except Exception as e:
                print(f"Error loading window {row['slice_path']}: {e}")
                continue
        
        # Convert to numpy arrays.
        X = np.array(X_list)
        y = np.array(y_list) 
        masks = np.array(masks_list)
        
        return X, y, masks

def residual_mae_loss(y_true, y_pred):
    """
    Per-timestep MAE over the normalized residual correction.

    Padding is masked with sample_weight=masks in model.fit().
    This avoids inferring the mask from y_true, because in residual mode a
    real correction can be exactly zero.
    """
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)

def create_model(sequence_length=3600, n_input_features=6, n_output_features=3):
    """Create the neural network model."""
    model = Sequential([
        # Masking layer to ignore padding.
        Masking(mask_value=0.0, input_shape=(sequence_length, n_input_features)),
        
        # LSTM with 128 units, returning full sequences.
        LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.0),
        
        # Dense layer with 64 units and ReLU.
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        # Output layer with residual correction units (dx, dy, dz).
        Dense(n_output_features, activation='linear')
    ])
    
    return model

def plot_training_history(history, save_path="training_history.png"):
    """Plot the training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Learning rate, if available.
    if 'lr' in history.history:
        ax2.plot(history.history['lr'], label='Learning Rate')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('LR')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {save_path}")

def evaluate_model(model, X_test, y_test, masks_test, norm_stats=None):
    """Evaluate the residual model and compute metrics on filtered deltas."""
    print("\n=== MODEL EVALUATION ===")
    
    # The network predicts residual corrections. Metrics are computed over:
    # filtered_delta = noisy_delta + predicted_correction.
    residual_pred = model.predict(X_test, batch_size=32, verbose=1)
    noisy_deltas = X_test[..., :3]
    y_pred = noisy_deltas + residual_pred
    
    # Apply masks to compute metrics only on valid positions.
    valid_positions = masks_test.astype(bool)
    
    # MAE in normalized values.
    mae_dx_norm = np.mean(np.abs(y_test[valid_positions][:, 0] - y_pred[valid_positions][:, 0]))
    mae_dy_norm = np.mean(np.abs(y_test[valid_positions][:, 1] - y_pred[valid_positions][:, 1]))  
    mae_dz_norm = np.mean(np.abs(y_test[valid_positions][:, 2] - y_pred[valid_positions][:, 2]))
    xy_step_error_norm = np.linalg.norm(y_test[valid_positions][:, :2] - y_pred[valid_positions][:, :2], axis=1)
    mae_xy_norm = np.mean(xy_step_error_norm)
    rmse_xy_norm = np.sqrt(np.mean(xy_step_error_norm**2))
    rmse_z_norm = np.sqrt(np.mean((y_test[valid_positions][:, 2] - y_pred[valid_positions][:, 2])**2))
    mae_total_norm = np.mean([mae_dx_norm, mae_dy_norm, mae_dz_norm])
    
    print(f"MAE dx (normalized): {mae_dx_norm:.6f}")
    print(f"MAE dy (normalized): {mae_dy_norm:.6f}")
    print(f"MAE XY step (normalized): {mae_xy_norm:.6f}")
    print(f"RMSE XY step (normalized): {rmse_xy_norm:.6f}")
    print(f"MAE dz (normalized): {mae_dz_norm:.6f}")
    print(f"RMSE z (normalized): {rmse_z_norm:.6f}")
    print(f"MAE total (normalized): {mae_total_norm:.6f}")
    
    # Initialize results dictionary.
    results = {
        'mae_dx_norm': mae_dx_norm,
        'mae_dy_norm': mae_dy_norm, 
        'mae_dz_norm': mae_dz_norm,
        'mae_xy_norm': mae_xy_norm,
        'rmse_xy_norm': rmse_xy_norm,
        'rmse_z_norm': rmse_z_norm,
        'mae_total_norm': mae_total_norm,
        'residual_training': True
    }
    
    # If normalization statistics are available, compute metrics in real meters.
    if norm_stats:
        # Denormalize predictions and ground truth.
        y_pred_meters = y_pred.copy()
        y_test_meters = y_test.copy()
        
        for i, feature in enumerate(['dx', 'dy', 'dz']):
            std = norm_stats['std'][feature]
            mean = norm_stats['mean'][feature]
            
            y_pred_meters[..., i] = y_pred[..., i] * std + mean
            y_test_meters[..., i] = y_test[..., i] * std + mean
        
        # MAE in real meters.
        mae_dx_meters = np.mean(np.abs(y_test_meters[valid_positions][:, 0] - y_pred_meters[valid_positions][:, 0]))
        mae_dy_meters = np.mean(np.abs(y_test_meters[valid_positions][:, 1] - y_pred_meters[valid_positions][:, 1]))  
        mae_dz_meters = np.mean(np.abs(y_test_meters[valid_positions][:, 2] - y_pred_meters[valid_positions][:, 2]))
        xy_step_error_meters = np.linalg.norm(
            y_test_meters[valid_positions][:, :2] - y_pred_meters[valid_positions][:, :2],
            axis=1
        )
        mae_xy_meters = np.mean(xy_step_error_meters)
        rmse_xy_meters = np.sqrt(np.mean(xy_step_error_meters**2))
        rmse_z_meters = np.sqrt(np.mean((y_test_meters[valid_positions][:, 2] - y_pred_meters[valid_positions][:, 2])**2))
        mae_total_meters = np.mean([mae_dx_meters, mae_dy_meters, mae_dz_meters])
        
        print(f"\nMAE dx (meters): {mae_dx_meters:.4f} m")
        print(f"MAE dy (meters): {mae_dy_meters:.4f} m")
        print(f"MAE XY step (meters): {mae_xy_meters:.4f} m")
        print(f"RMSE XY step (meters): {rmse_xy_meters:.4f} m")
        print(f"MAE dz (meters): {mae_dz_meters:.4f} m")
        print(f"RMSE z (meters): {rmse_z_meters:.4f} m")
        print(f"MAE total (meters): {mae_total_meters:.4f} m")
        
        # Update results.
        results.update({
            'mae_dx_meters': mae_dx_meters,
            'mae_dy_meters': mae_dy_meters,
            'mae_dz_meters': mae_dz_meters,
            'mae_xy_meters': mae_xy_meters,
            'rmse_xy_meters': rmse_xy_meters,
            'rmse_z_meters': rmse_z_meters,
            'mae_total_meters': mae_total_meters
        })
        
        # ===== SPATIAL DRIFT METRICS =====
        print(f"\n=== SPATIAL DRIFT METRICS ===")
        
        # Apply masks so integration uses only valid timesteps.
        drift_metrics = calculate_drift_metrics(y_pred_meters, y_test_meters, masks_test)
        
        print(f"Mean final drift: {drift_metrics['drift_final_mean_m']:.4f} m")
        print(f"RMS drift: {drift_metrics['drift_rms_m']:.4f} m") 
        print(f"Mean abs final Z error: {drift_metrics['z_final_abs_mean_m']:.4f} m")
        print(f"RMS Z drift: {drift_metrics['z_rms_m']:.4f} m")
        print(f"Trajectory length difference: {drift_metrics['length_diff_m']:.4f} m")
        print(f"Mean pattern length: {drift_metrics['true_length_mean_m']:.4f} m")
        print(f"Relative length difference: {drift_metrics['length_diff_pct']:.2f}%")
        print(f"Relative final drift: {drift_metrics['drift_final_pct']:.2f}%")
        print(f"Relative RMS drift: {drift_metrics['drift_rms_pct']:.2f}%")
        
        # Add drift metrics to results.
        results.update(drift_metrics)
        
    return results

def calculate_drift_metrics(y_pred_meters, y_test_meters, masks_test):
    """
    Compute spatial drift metrics by integrating deltas into positions.
    
    Args:
        y_pred_meters: Denormalized predictions [batch, time, 3]
        y_test_meters: Denormalized ground truth [batch, time, 3]
        masks_test: Binary masks [batch, time]
        
    Returns:
        dict: Spatial drift metrics
    """
    # Convert masks to boolean.
    valid_mask = masks_test.astype(bool)
    
    # Apply masks by zeroing padding timesteps.
    y_pred_masked = y_pred_meters.copy()
    y_test_masked = y_test_meters.copy()
    
    for i in range(y_pred_meters.shape[0]):  # For each sequence in the batch.
        for t in range(y_pred_meters.shape[1]):  # For each timestep.
            if not valid_mask[i, t]:
                y_pred_masked[i, t, :] = 0
                y_test_masked[i, t, :] = 0
    
    # Integrate deltas into positions (cumsum along the time axis).
    pos_pred = np.cumsum(y_pred_masked, axis=1)  # [batch, time, 3]
    pos_true = np.cumsum(y_test_masked, axis=1)  # [batch, time, 3]
    
    # ===== 1. MEAN FINAL DRIFT =====
    # Euclidean distance between final positions (last valid timestep per sequence).
    final_drifts = []
    final_z_errors = []
    
    for i in range(pos_pred.shape[0]):
        # Find the last valid timestep for this sequence.
        valid_times = np.where(valid_mask[i])[0]
        if len(valid_times) > 0:
            last_valid_t = valid_times[-1]
            # Compute Euclidean distance in x,y only (ignore z).
            drift_xy = np.linalg.norm(pos_pred[i, last_valid_t, :2] - pos_true[i, last_valid_t, :2])
            final_drifts.append(drift_xy)
            final_z_errors.append(pos_pred[i, last_valid_t, 2] - pos_true[i, last_valid_t, 2])
    
    drift_final_mean_m = np.mean(final_drifts) if final_drifts else 0.0
    z_final_mean_m = np.mean(final_z_errors) if final_z_errors else 0.0
    z_final_abs_mean_m = np.mean(np.abs(final_z_errors)) if final_z_errors else 0.0
    
    # ===== 2. RMS DRIFT =====
    # RMS of Euclidean distances along the whole trajectory.
    all_drifts = []
    all_z_errors = []
    
    for i in range(pos_pred.shape[0]):
        for t in range(pos_pred.shape[1]):
            if valid_mask[i, t]:
                # Euclidean distance at each valid timestep (x,y only).
                drift_xy = np.linalg.norm(pos_pred[i, t, :2] - pos_true[i, t, :2])
                all_drifts.append(drift_xy)
                all_z_errors.append(pos_pred[i, t, 2] - pos_true[i, t, 2])
    
    drift_rms_m = np.sqrt(np.mean(np.array(all_drifts)**2)) if all_drifts else 0.0
    z_rms_m = np.sqrt(np.mean(np.array(all_z_errors)**2)) if all_z_errors else 0.0
    
    # ===== 3. TRAJECTORY LENGTH DIFFERENCE =====
    # Difference between total predicted and true trajectory lengths.
    length_diffs = []
    pred_total_lengths = []
    true_total_lengths = []
    
    for i in range(y_pred_meters.shape[0]):
        # Compute total predicted and true trajectory length.
        pred_lengths = []
        true_lengths = []
        
        for t in range(y_pred_meters.shape[1]):
            if valid_mask[i, t]:
                # Euclidean norm of the delta at this timestep (x,y only).
                pred_step_length = np.linalg.norm(y_pred_masked[i, t, :2])
                true_step_length = np.linalg.norm(y_test_masked[i, t, :2])
                pred_lengths.append(pred_step_length)
                true_lengths.append(true_step_length)
        
        if pred_lengths and true_lengths:
            total_pred_length = np.sum(pred_lengths)
            total_true_length = np.sum(true_lengths)
            length_diff = abs(total_pred_length - total_true_length)
            length_diffs.append(length_diff)
            pred_total_lengths.append(total_pred_length)
            true_total_lengths.append(total_true_length)
    
    length_diff_m = np.mean(length_diffs) if length_diffs else 0.0
    pred_length_mean_m = np.mean(pred_total_lengths) if pred_total_lengths else 0.0
    true_length_mean_m = np.mean(true_total_lengths) if true_total_lengths else 0.0
    length_diff_pct = (length_diff_m / true_length_mean_m * 100.0) if true_length_mean_m > 0 else 0.0
    drift_final_pct = (drift_final_mean_m / true_length_mean_m * 100.0) if true_length_mean_m > 0 else 0.0
    drift_rms_pct = (drift_rms_m / true_length_mean_m * 100.0) if true_length_mean_m > 0 else 0.0
    
    return {
        'drift_final_mean_m': drift_final_mean_m,
        'drift_rms_m': drift_rms_m,
        'drift_final_xy_mean_m': drift_final_mean_m,
        'drift_rms_xy_m': drift_rms_m,
        'z_final_mean_m': z_final_mean_m,
        'z_final_abs_mean_m': z_final_abs_mean_m,
        'z_rms_m': z_rms_m,
        'length_diff_m': length_diff_m,
        'length_diff_xy_m': length_diff_m,
        'pred_length_mean_m': pred_length_mean_m,
        'true_length_mean_m': true_length_mean_m,
        'length_diff_pct': length_diff_pct,
        'length_diff_xy_pct': length_diff_pct,
        'drift_final_pct': drift_final_pct,
        'drift_rms_pct': drift_rms_pct,
        'drift_final_xy_pct': drift_final_pct,
        'drift_rms_xy_pct': drift_rms_pct
    }

def train_model(dataset, model_config, fast_mode=False):
    """
    Train using a pre-split dataset (train/val/test).
    
    Args:
        dataset: GPSTrackDataset
        model_config: Training configuration dictionary
        fast_mode: If True, reduce data and epochs for quick checks
        
    Returns:
        tuple: (model, history, metrics, training_time)
    """
    print("\n=== NEURAL NETWORK TRAINING ===")
    
    # Load data by split.
    (X_train, y_train, masks_train), (X_val, y_val, masks_val), (X_test, y_test, masks_test) = dataset.load_by_sets()
    
    # Apply fast-mode limits.
    if fast_mode:
        print("FAST MODE: Limiting data for quick checks")
        max_samples = 100
        
        if len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
            masks_train = masks_train[indices]
            
        if len(X_val) > max_samples // 4:
            indices = np.random.choice(len(X_val), max_samples // 4, replace=False)
            X_val = X_val[indices]
            y_val = y_val[indices]
            masks_val = masks_val[indices]
            
        if len(X_test) > max_samples // 4:
            indices = np.random.choice(len(X_test), max_samples // 4, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
            masks_test = masks_test[indices]
        
        print(f"Limited data - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # In v4 the network learns residual corrections from augmented inputs:
    # residual = clean_delta - noisy_delta.
    y_train_residual = y_train - X_train[..., :3]
    y_val_residual = y_val - X_val[..., :3]
    y_test_clean = y_test
    
    # Create model.
    model = create_model(
        sequence_length=X_train.shape[1],
        n_input_features=X_train.shape[2],
        n_output_features=y_train_residual.shape[2]
    )
    model.compile(
        optimizer=Adam(learning_rate=model_config['learning_rate']),
        loss=residual_mae_loss,
        metrics=['mae']
    )
    
    print(f"Model created:")
    print(f"  - Sequence: {X_train.shape[1]} timesteps")
    print(f"  - Input features: {X_train.shape[2]} (dx, dy, dz, t_norm, distance_norm, absolute_t_norm)")
    print(f"  - Output features: {y_train_residual.shape[2]} (residual dx, dy, dz)")
    print(f"  - Mode: residual (output = clean_delta - noisy_delta)")
    
    # Create the models directory if needed.
    os.makedirs('models', exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=model_config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'models/model_best_v4.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Training.
    start_time = time.time()
    
    print(f"\nStarting training:")
    print(f"  - Epochs: {model_config['epochs']}")
    print(f"  - Batch size: {model_config['batch_size']}")
    print(f"  - Learning rate: {model_config['learning_rate']}")
    print(f"  - Patience: {model_config['patience']}")
    
    history = model.fit(
        X_train, y_train_residual,
        sample_weight=masks_train,
        batch_size=model_config['batch_size'],
        epochs=model_config['epochs'],
        validation_data=(X_val, y_val_residual, masks_val),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Show training summary.
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    epochs_trained = len(history.history['loss'])
    
    print(f"\nTraining completed:")
    print(f"  - Epochs trained: {epochs_trained}/{model_config['epochs']}")
    print(f"  - Total time: {training_time/60:.2f} minutes")
    print(f"  - Final train loss: {final_train_loss:.6f}")
    print(f"  - Final val loss: {final_val_loss:.6f}")
    
    # Plot history.
    Path("results/training").mkdir(parents=True, exist_ok=True)
    plot_training_history(history, "results/training/training_history_v4.png")
    
    # Test evaluation.
    print(f"\nEvaluating on TEST split...")
    test_metrics = evaluate_model(model, X_test, y_test_clean, masks_test, norm_stats=dataset.norm_stats)
    
    # Save final model.
    model.save('models/model_final_v4.keras')
    print(f"Final model saved to: models/model_final_v4.keras")
    
    return model, history, test_metrics, training_time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a GPS correction neural network with augmented context features',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main arguments.
    parser.add_argument('--data_root', type=str, default='data/input',
                        help='Data root directory')
    
    # Training hyperparameters.
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    # Additional options.
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: reduce epochs and data for quick checks')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducible comparisons')
    
    return parser.parse_args()

def main():
    """Main training entry point."""
    print("=== GPS TRACK CORRECTION NEURAL NETWORK TRAINING V4 ===\n")
    
    try:
        # Parse arguments.
        args = parse_args()
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.keras.utils.set_random_seed(args.seed)
        
        print(f"Configuration:")
        print(f"  - Data directory: {args.data_root}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Learning rate: {args.lr}")
        print(f"  - Patience: {args.patience}")
        print(f"  - Seed: {args.seed}")
        print(f"  - Fast mode: {args.fast}")
        
        # Model configuration.
        model_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'patience': args.patience
        }
        
        # Apply fast-mode limits.
        if args.fast:
            print(f"\nFAST MODE ENABLED")
            model_config['epochs'] = min(model_config['epochs'], 10)
            model_config['batch_size'] = min(model_config['batch_size'], 16)
            model_config['patience'] = min(model_config['patience'], 5)
            print(f"  - Epochs limited to: {model_config['epochs']}")
            print(f"  - Batch size limited to: {model_config['batch_size']}")
            print(f"  - Patience limited to: {model_config['patience']}")
        
        # Load dataset and train.
        print(f"\nLoading dataset...")
        dataset = GPSTrackDataset(data_dir=args.data_root)
        
        print(f"\nRunning training")
        model, history, test_metrics, training_time = train_model(
            dataset, model_config, fast_mode=args.fast
        )
        
        # Save results.
        mode_suffix = "_fast" if args.fast else "_complete"
        results_dir = Path('results') / 'training'
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f'training_results_v4{mode_suffix}.json'
        
        results = {
            'config': model_config,
            'model_type': 'residual_augmented_features',
            'input_features': ['dx', 'dy', 'dz', 't_norm', 'distance_norm', 'absolute_t_norm'],
            'output_features': ['residual_dx', 'residual_dy', 'residual_dz'],
            'test_metrics': test_metrics,
            'training_time_minutes': training_time / 60,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Mode: {'FAST' if args.fast else 'FULL'}")
        if 'mae_total_meters' in test_metrics:
            print(f"MAE total (meters): {test_metrics['mae_total_meters']:.4f} m")
            if 'mae_xy_meters' in test_metrics:
                print(f"MAE XY step (meters): {test_metrics['mae_xy_meters']:.4f} m")
                print(f"MAE Z step (meters): {test_metrics['mae_dz_meters']:.4f} m")
            if 'drift_final_mean_m' in test_metrics:
                print(f"Mean final XY drift: {test_metrics['drift_final_mean_m']:.4f} m")
                print(f"RMS XY drift: {test_metrics['drift_rms_m']:.4f} m")
                if 'z_final_abs_mean_m' in test_metrics:
                    print(f"Mean abs final Z error: {test_metrics['z_final_abs_mean_m']:.4f} m")
                    print(f"RMS Z drift: {test_metrics['z_rms_m']:.4f} m")
        else:
            print(f"MAE total (normalized): {test_metrics['mae_total_norm']:.6f}")
        print(f"Total time: {training_time/60:.2f} minutes")
        print(f"Results saved to: {results_file}")
        
        print(f"\nTraining completed successfully")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
