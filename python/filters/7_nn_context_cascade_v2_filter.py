#!/usr/bin/env python3
"""Filter a GPX track with the contextual fast+slow residual cascade model."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model as keras_load_model


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONTEXT_DATASET_SCRIPT = PROJECT_ROOT / "python" / "pipeline" / "5_generate_input_dataset_context_v1.py"
BASE_FILTER_SCRIPT = SCRIPT_DIR / "7_nn_filter.py"
DELTA_FEATURES = ["dx", "dy", "dz"]


def load_module(name, path):
    """Load a Python script as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


context_features = load_module("context_dataset_v1", CONTEXT_DATASET_SCRIPT)
base_filter = load_module("nn_filter_base", BASE_FILTER_SCRIPT)


def normalize_feature_frame(feature_frame, norm_stats, input_features):
    """Normalize contextual input features with train statistics."""
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


def build_context_from_delta_norm(delta_norm, norm_stats, input_features):
    """Build normalized context features from normalized delta channels."""
    delta_meters = denormalize_delta_channels(delta_norm, norm_stats)
    dx = delta_meters[:, 0]
    dy = delta_meters[:, 1]
    dz = delta_meters[:, 2]
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    feature_frame = context_features.build_feature_frame(dx, dy, dz, x, y)
    return normalize_feature_frame(feature_frame, norm_stats, input_features)


def load_inference_model(model_path):
    """Load a Keras model for inference."""
    return keras_load_model(model_path, compile=False)


def apply_context_cascade_filter(
    track_df,
    fast_model_path="models/model_final_context_cascade_v2_fast.keras",
    slow_model_path="models/model_final_context_cascade_v2_slow.keras",
    norm_stats_path="data/input_context_v1/norm_stats_train.json",
    max_sequence=3600,
):
    """Apply the contextual fast+slow cascade filter to a parsed GPX DataFrame."""
    print("Loading contextual normalization statistics...")
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)

    input_features = norm_stats.get("input_features")
    if not input_features:
        raise ValueError("Context norm stats must include input_features")

    print(f"Input features: {input_features}")
    print("Loading fast model...")
    fast_model = load_inference_model(fast_model_path)
    print("Loading slow model...")
    slow_model = load_inference_model(slow_model_path)

    print("Setting up geodesic projection...")
    lat_center = track_df["lat"].mean()
    lon_center = track_df["lon"].mean()
    transformer = base_filter.setup_projection(lat_center, lon_center)

    lat_ref = track_df["lat"].iloc[0]
    lon_ref = track_df["lon"].iloc[0]
    x, y = base_filter.latlon_to_meters(track_df["lat"], track_df["lon"], transformer, lat_ref, lon_ref)
    z = track_df["ele"].to_numpy(dtype=np.float64)

    print("Calculating base deltas...")
    dx, dy, dz = base_filter.calculate_deltas(x, y, z)

    print("Building contextual features...")
    feature_frame = context_features.build_feature_frame(dx, dy, dz, x, y)
    input_matrix = normalize_feature_frame(feature_frame, norm_stats, input_features)

    sequence_length = len(input_matrix)
    print(f"Processing {sequence_length} points in chunks of {max_sequence}")

    fast_delta_chunks = []
    for i in range(0, sequence_length, max_sequence):
        end_idx = min(i + max_sequence, sequence_length)
        chunk_len = end_idx - i

        input_data = np.zeros((1, max_sequence, len(input_features)), dtype=np.float32)
        input_data[0, :chunk_len, :] = input_matrix[i:end_idx, :]

        fast_residual = fast_model.predict(input_data, verbose=0)
        fast_delta = input_data[:, :, :3] + fast_residual
        fast_delta_chunks.append(fast_delta[0, :chunk_len, :])

    fast_delta_norm = np.vstack(fast_delta_chunks)
    print("Rebuilding context from fast deltas...")
    slow_input_matrix = build_context_from_delta_norm(fast_delta_norm, norm_stats, input_features)

    filtered_chunks = []
    for i in range(0, sequence_length, max_sequence):
        end_idx = min(i + max_sequence, sequence_length)
        chunk_len = end_idx - i

        slow_input = np.zeros((1, max_sequence, len(input_features)), dtype=np.float32)
        slow_input[0, :chunk_len, :] = slow_input_matrix[i:end_idx, :]
        slow_residual = slow_model.predict(slow_input, verbose=0)
        filtered_chunks.append(fast_delta_norm[i:end_idx, :] + slow_residual[0, :chunk_len, :])

    filtered_norm = np.vstack(filtered_chunks)
    filtered_delta = denormalize_delta_channels(filtered_norm, norm_stats)

    print("Integrating filtered deltas...")
    x_filt, y_filt, z_filt = base_filter.integrate_deltas(
        filtered_delta[:, 0],
        filtered_delta[:, 1],
        filtered_delta[:, 2],
        x[0],
        y[0],
        z[0],
    )

    print("Converting back to lat/lon...")
    lat_filt, lon_filt = base_filter.meters_to_latlon(x_filt, y_filt, transformer, lat_ref, lon_ref)

    filtered_df = track_df.copy()
    filtered_df["lat"] = lat_filt
    filtered_df["lon"] = lon_filt
    filtered_df["ele"] = z_filt

    print("Position preservation check:")
    print(f"  Original: ({track_df['lat'].iloc[0]:.8f}, {track_df['lon'].iloc[0]:.8f})")
    print(f"  Filtered: ({filtered_df['lat'].iloc[0]:.8f}, {filtered_df['lon'].iloc[0]:.8f})")
    return filtered_df


def parse_args():
    parser = argparse.ArgumentParser(description="Filter GPS track using contextual fast+slow residual cascade")
    parser.add_argument("input_gpx", help="Input GPX file")
    parser.add_argument("output_gpx", nargs="?", help="Output filtered GPX file")
    parser.add_argument("--fast-model", default="models/model_final_context_cascade_v2_fast.keras")
    parser.add_argument("--slow-model", default="models/model_final_context_cascade_v2_slow.keras")
    parser.add_argument("--norm-stats", default="data/input_context_v1/norm_stats_train.json")
    parser.add_argument("--max-sequence", type=int, default=3600, help="Chunk length used by the model")
    parser.add_argument("--suffix", default="nn_context_cascade_v2_filtered")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.output_gpx is None:
            input_path = Path(args.input_gpx)
            args.output_gpx = input_path.parent / f"{input_path.stem}_{args.suffix}{input_path.suffix}"
            print(f"Output file auto-generated: {args.output_gpx}")
        else:
            Path(args.output_gpx).parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing {args.input_gpx}...")
        track_df = base_filter.parse_gpx(args.input_gpx)
        filtered_df = apply_context_cascade_filter(
            track_df,
            fast_model_path=args.fast_model,
            slow_model_path=args.slow_model,
            norm_stats_path=args.norm_stats,
            max_sequence=args.max_sequence,
        )

        base_filter.create_gpx_with_gpxpy(
            filtered_df["lat"],
            filtered_df["lon"],
            filtered_df["ele"],
            filtered_df["time"] if "time" in filtered_df.columns else None,
            str(args.output_gpx),
        )

        print("SUCCESS: Context cascade filtering completed")
        print(f"   Input: {len(track_df)} points")
        print(f"   Output: {len(filtered_df)} points")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
