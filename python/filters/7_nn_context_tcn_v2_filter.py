#!/usr/bin/env python3
"""Wrapper for the contextual TCN v2 neural GPX filter."""

import importlib.util
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, Conv1D, Dense, Dropout, Input, LayerNormalization, SpatialDropout1D
from tensorflow.keras.models import Model


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_FILTER_SCRIPT = SCRIPT_DIR / "7_nn_context_v1_filter.py"
MODEL_TAG = "context_tcn_v2"


def load_base_filter():
    """Load the contextual filter implementation."""
    spec = importlib.util.spec_from_file_location("nn_context_v1_filter", BASE_FILTER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_filter = load_base_filter()


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    """Build one residual dilated temporal convolution block."""
    residual = x
    y = Conv1D(
        filters,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        activation=None,
    )(x)
    y = LayerNormalization()(y)
    y = Activation("relu")(y)
    y = SpatialDropout1D(dropout_rate)(y)
    y = Conv1D(filters, 1, padding="same", activation=None)(y)
    y = LayerNormalization()(y)
    y = Add()([residual, y])
    return Activation("relu")(y)


def local_branch(inputs, output_features):
    """Create the short-context branch for fast local residual corrections."""
    x = Conv1D(64, 5, padding="same", activation=None, name="fast_conv1")(inputs)
    x = LayerNormalization(name="fast_norm1")(x)
    x = Activation("relu", name="fast_relu1")(x)
    x = Conv1D(64, 3, padding="same", activation=None, name="fast_conv2")(x)
    x = LayerNormalization(name="fast_norm2")(x)
    x = Activation("relu", name="fast_relu2")(x)
    x = Dense(32, activation="relu", name="fast_dense")(x)
    return Dense(output_features, activation="linear", name="fast_residual")(x)


def slow_branch(inputs, output_features):
    """Create the dilated TCN branch for slow residual corrections."""
    x = Conv1D(64, 1, padding="same", activation=None, name="slow_projection")(inputs)
    x = LayerNormalization(name="slow_projection_norm")(x)
    x = Activation("relu", name="slow_projection_relu")(x)

    for dilation_rate in [1, 2, 4, 8, 16, 32, 64, 128]:
        x = tcn_block(
            x,
            filters=64,
            kernel_size=5,
            dilation_rate=dilation_rate,
            dropout_rate=0.1,
        )

    x = Dense(64, activation="relu", name="slow_dense")(x)
    x = Dropout(0.2, name="slow_dropout")(x)
    return Dense(output_features, activation="linear", name="slow_residual")(x)


def create_tcn_model(sequence_length=3600, n_features=15, output_features=3):
    """Create the contextual TCN v2 inference model used by filters."""
    inputs = Input(shape=(sequence_length, n_features))
    fast_residual = local_branch(inputs, output_features)
    slow_residual = slow_branch(inputs, output_features)
    total_residual = Add(name="total_residual")([fast_residual, slow_residual])
    return Model(inputs=inputs, outputs=total_residual, name=MODEL_TAG)


def default_weights_path(model_path):
    """Return the final weights path associated with a Keras model path."""
    path = Path(model_path)
    if path.name.endswith(".keras"):
        return path.with_name(path.name[:-6] + ".weights.h5")
    return path


def load_tcn_model(model_path, n_features, max_sequence):
    """Load the TCN model, falling back to architecture plus weights."""
    try:
        return base_filter.base_filter.load_model_robust(model_path)
    except Exception as exc:
        weights_path = default_weights_path(model_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Could not load {model_path} and weights file was not found: {weights_path}"
            ) from exc

        print(f"Falling back to TCN architecture plus weights: {weights_path}")
        model = create_tcn_model(sequence_length=max_sequence, n_features=n_features)
        model.load_weights(str(weights_path))
        return model


def main():
    """Run the contextual filter with TCN v2 defaults."""
    return base_filter.main(
        default_model="models/model_final_context_tcn_v2.keras",
        default_suffix="nn_context_tcn_v2_filtered",
        description="Filter GPS track using contextual TCN v2 neural network",
        model_loader=load_tcn_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
