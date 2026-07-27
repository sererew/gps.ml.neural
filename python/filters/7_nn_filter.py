#!/usr/bin/env python3
"""
Script para filtrar un track GPS usando la red neuronal entrenada.

El proceso es:
1. Cargar track GPX usando gpxpy (manejo robusto)
2. Convertir a coordenadas métricas usando pyproj (geodésicamente correcto)
3. Calcular deltas (dx[1:] = diff(x), dx[0] = 0)
4. Normalizar usando estadísticas del entrenamiento
5. Aplicar filtro de red neuronal
6. Desnormalizar deltas filtrados
7. Integrar deltas preservando posición inicial
8. Guardar como GPX filtrado usando gpxpy

Uso:
    python 7_nn_filter.py input_track.gpx [output_track.gpx]
    python 7_nn_filter.py input_track.gpx [output_track.gpx] --model custom_model.keras
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
import os
import tempfile
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

# Librerías especializadas
import gpxpy
import gpxpy.gpx
from pyproj import Transformer
from scipy.interpolate import CubicSpline

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking


CHANNEL_TO_INDEX = {"x": 0, "y": 1, "z": 2}
ALL_CHANNELS = ("x", "y", "z")

def parse_gpx(gpx_path):
    """
    Parsea un archivo GPX usando gpxpy.
    
    Returns:
        DataFrame con columnas: lat, lon, ele, time
    """
    print(f"Loading GPX from {gpx_path}...")
    with open(gpx_path, 'r', encoding='utf-8') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    points = []
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation if point.elevation is not None else 0.0,
                    'time': point.time
                })
    
    if not points:
        raise ValueError(f"No trackpoints found in {gpx_path}")
    
    df = pd.DataFrame(points)
    print(f"Loaded {len(df)} points from {gpx_path}")
    return df

def setup_projection(lat_center, lon_center):
    """
    Configura proyección geodésica precisa usando UTM automático.
    
    Returns:
        Transformer para conversión lat/lon <-> x/y
    """
    # Determinar zona UTM automáticamente
    utm_zone = int((lon_center + 180) / 6) + 1
    hemisphere = 'north' if lat_center >= 0 else 'south'
    
    # Crear transformador geodésico preciso
    # WGS84 (EPSG:4326) -> UTM
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    
    print(f"Using UTM Zone {utm_zone}{hemisphere[0].upper()} projection")
    return transformer

def latlon_to_meters(lat, lon, transformer, lat_ref=None, lon_ref=None):
    """
    Convierte lat/lon a coordenadas métricas usando proyección geodésica.
    """
    # Convertir a UTM
    x_utm, y_utm = transformer.transform(lon, lat)
    
    # Si se proporciona referencia, hacer coordenadas relativas
    if lat_ref is not None and lon_ref is not None:
        x_ref, y_ref = transformer.transform(lon_ref, lat_ref)
        x_utm = x_utm - x_ref
        y_utm = y_utm - y_ref
    
    return x_utm, y_utm

def meters_to_latlon(x, y, transformer, lat_ref=None, lon_ref=None):
    """
    Convierte coordenadas métricas de vuelta a lat/lon.
    """
    # Si hay referencia, convertir de relativas a absolutas
    if lat_ref is not None and lon_ref is not None:
        x_ref, y_ref = transformer.transform(lon_ref, lat_ref)
        x = x + x_ref
        y = y + y_ref
    
    # Convertir de UTM a lat/lon
    lon, lat = transformer.transform(x, y, direction='INVERSE')
    
    return lat, lon

def parse_channels(value):
    """Parse a comma-separated channel list."""
    channels = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(channels) - set(ALL_CHANNELS))
    if unknown:
        raise ValueError(f"Unknown anchor channels: {unknown}. Use any of: {', '.join(ALL_CHANNELS)}")
    return channels

def infer_pattern_path(input_gpx):
    """Infer the aligned pattern path from a preprocessed recording path."""
    input_path = Path(input_gpx)
    pasada = input_path.parent.name
    pattern_path = input_path.parent / f"{pasada}_aligned_pattern_resampled.gpx"
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found for {input_gpx}: {pattern_path}")
    return pattern_path

def choose_anchor_indices(n_points, duration_s, anchors_per_hour, min_anchors, max_anchors, edge_skip_points=0):
    """Choose anchor indices uniformly over the common track-pattern timeline."""
    if n_points <= 0:
        return np.asarray([], dtype=np.int64)

    first_index = int(max(0, edge_skip_points))
    last_index = int(min(n_points - 1, n_points - 1 - edge_skip_points))
    if last_index <= first_index:
        first_index = 0
        last_index = n_points - 1

    duration_hours = max(float(duration_s) / 3600.0, 0.0)
    anchor_count = int(round(duration_hours * anchors_per_hour))
    anchor_count = max(anchor_count, min_anchors)
    if max_anchors and max_anchors > 0:
        anchor_count = min(anchor_count, max_anchors)
    anchor_count = max(2, min(anchor_count, last_index - first_index + 1))

    return np.unique(np.linspace(first_index, last_index, anchor_count).round().astype(np.int64))

def anchor_values(error, anchor_indices, channel_idx, radius):
    """Return exact or local-mean error values at anchor indices."""
    values = np.zeros(len(anchor_indices), dtype=np.float64)
    for i, idx in enumerate(anchor_indices):
        if radius <= 0:
            values[i] = error[idx, channel_idx]
            continue
        start = max(0, int(idx) - radius)
        end = min(len(error), int(idx) + radius + 1)
        values[i] = float(np.mean(error[start:end, channel_idx]))
    return values

def interpolate_values(indices, values, n_points, interpolation):
    """Interpolate sparse anchor values over all timesteps with clamped edges."""
    if n_points <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(indices) == 0:
        return np.zeros(n_points, dtype=np.float64)
    if len(indices) == 1:
        return np.full(n_points, values[0], dtype=np.float64)

    target = np.arange(n_points, dtype=np.float64)
    indices_float = indices.astype(np.float64)
    clipped_target = np.clip(target, indices_float[0], indices_float[-1])

    if interpolation == "cubic" and len(indices) >= 3:
        spline = CubicSpline(indices_float, values, bc_type="natural")
        curve = spline(clipped_target)
    else:
        curve = np.interp(clipped_target, indices_float, values)

    curve[target < indices_float[0]] = values[0]
    curve[target > indices_float[-1]] = values[-1]
    return curve

def apply_pattern_anchor_correction(
    filtered_df,
    pattern_df,
    anchors_per_hour=8.0,
    min_anchors=8,
    max_anchors=0,
    anchor_error_radius=0,
    anchor_channels="x,y,z",
    anchor_interpolation="cubic",
    anchor_edge_blend_points=30,
    anchor_trim_to_pattern=False,
    anchor_edge_skip_points=0,
):
    """Apply an experimental oracle slow correction using sparse pattern anchors."""
    print("Applying experimental pattern-anchor slow correction...")
    channels = parse_channels(anchor_channels)

    track_indices = None
    pattern_indices = None
    if (
        "time" in filtered_df.columns
        and "time" in pattern_df.columns
        and filtered_df["time"].notna().any()
        and pattern_df["time"].notna().any()
    ):
        track_times = pd.to_datetime(filtered_df["time"], utc=True, errors="coerce")
        pattern_times = pd.to_datetime(pattern_df["time"], utc=True, errors="coerce")
        track_time_df = pd.DataFrame({"time": track_times, "track_idx": np.arange(len(filtered_df))}).dropna()
        pattern_time_df = pd.DataFrame({"time": pattern_times, "pattern_idx": np.arange(len(pattern_df))}).dropna()
        common = pd.merge(track_time_df, pattern_time_df, on="time", how="inner").sort_values("track_idx")
        if len(common) >= 2:
            track_indices = common["track_idx"].to_numpy(dtype=np.int64)
            pattern_indices = common["pattern_idx"].to_numpy(dtype=np.int64)

    if track_indices is None or pattern_indices is None:
        n_common = min(len(filtered_df), len(pattern_df))
        if n_common < 2:
            raise ValueError("Need at least two common points for pattern-anchor correction")
        track_indices = np.arange(n_common, dtype=np.int64)
        pattern_indices = np.arange(n_common, dtype=np.int64)
        print(f"Pattern-anchor alignment: positional fallback with {n_common} common points")
    else:
        print(f"Pattern-anchor alignment: matched {len(track_indices)} points by timestamp")

    combined_lat = pd.concat([filtered_df["lat"], pattern_df["lat"]])
    combined_lon = pd.concat([filtered_df["lon"], pattern_df["lon"]])
    transformer = setup_projection(combined_lat.mean(), combined_lon.mean())
    lat_ref = filtered_df["lat"].iloc[0]
    lon_ref = filtered_df["lon"].iloc[0]

    x_filt, y_filt = latlon_to_meters(filtered_df["lat"], filtered_df["lon"], transformer, lat_ref, lon_ref)
    z_filt = filtered_df["ele"].to_numpy(dtype=np.float64)

    x_pattern, y_pattern = latlon_to_meters(pattern_df["lat"], pattern_df["lon"], transformer, lat_ref, lon_ref)
    z_pattern = pattern_df["ele"].to_numpy(dtype=np.float64)

    filtered_pos = np.column_stack([x_filt, y_filt, z_filt])
    pattern_pos = np.column_stack([x_pattern, y_pattern, z_pattern])
    matched_error = filtered_pos[track_indices] - pattern_pos[pattern_indices]

    if "time" in filtered_df.columns and filtered_df["time"].notna().any():
        t0 = filtered_df["time"].iloc[int(track_indices[0])]
        t1 = filtered_df["time"].iloc[int(track_indices[-1])]
        try:
            duration_s = max((t1 - t0).total_seconds(), 0.0)
        except Exception:
            duration_s = float(len(track_indices) - 1)
    else:
        duration_s = float(len(track_indices) - 1)

    anchor_pair_indices = choose_anchor_indices(
        len(track_indices),
        duration_s,
        anchors_per_hour,
        min_anchors,
        max_anchors,
        anchor_edge_skip_points,
    )
    anchor_track_indices = track_indices[anchor_pair_indices]
    print(
        f"Pattern anchors: {len(anchor_track_indices)} anchors over {duration_s / 3600.0:.2f} h "
        f"({anchors_per_hour:g}/h, min {min_anchors})"
    )

    correction = np.zeros((len(filtered_df), 3), dtype=np.float64)
    for channel in channels:
        channel_idx = CHANNEL_TO_INDEX[channel]
        values = anchor_values(matched_error, anchor_pair_indices, channel_idx, anchor_error_radius)
        curve = interpolate_values(
            anchor_track_indices,
            values,
            len(filtered_df),
            anchor_interpolation,
        )
        first_anchor = int(anchor_track_indices[0])
        if first_anchor > 0:
            curve[:first_anchor] = 0.0

        if anchor_edge_blend_points > 0:
            blend_end = min(len(curve), first_anchor + int(anchor_edge_blend_points) + 1)
            if blend_end > first_anchor:
                weights = np.linspace(0.0, 1.0, blend_end - first_anchor)
                curve[first_anchor:blend_end] *= weights
        correction[:, channel_idx] = curve

    corrected_pos = filtered_pos - correction
    lat_corr, lon_corr = meters_to_latlon(corrected_pos[:, 0], corrected_pos[:, 1], transformer, lat_ref, lon_ref)

    corrected_df = filtered_df.copy()
    corrected_df["lat"] = lat_corr
    corrected_df["lon"] = lon_corr
    corrected_df["ele"] = corrected_pos[:, 2]
    if anchor_trim_to_pattern:
        corrected_df = corrected_df.iloc[track_indices].copy().reset_index(drop=True)
    return corrected_df

def calculate_deltas(x, y, z):
    """
    Calcula deltas entre puntos consecutivos
    
    El primer punto no tiene delta (dx[0] = 0), 
    los siguientes son diferencias entre consecutivos.
    """
    # Inicializar arrays de deltas
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dz = np.zeros_like(z)
    
    # El primer punto no tiene delta (permanece en 0)
    # Los siguientes puntos: delta[i] = pos[i] - pos[i-1]
    dx[1:] = np.diff(x)  # x[1] - x[0], x[2] - x[1], etc.
    dy[1:] = np.diff(y)
    dz[1:] = np.diff(z)
    
    print(f"Delta calculation check:")
    print(f"  First 5 deltas dx: {dx[:5]}")
    print(f"  Delta stats: dx={np.mean(dx):.3f}±{np.std(dx):.3f}, "
          f"dy={np.mean(dy):.3f}±{np.std(dy):.3f}, dz={np.mean(dz):.3f}±{np.std(dz):.3f}")
    
    return dx, dy, dz

def normalize_deltas(dx, dy, dz, norm_stats):
    """
    Normaliza deltas usando estadísticas de entrenamiento.
    VERIFICA que las estadísticas sean consistentes.
    """
    print(f"Normalization stats:")
    print(f"  dx: mean={norm_stats['mean']['dx']:.6f}, std={norm_stats['std']['dx']:.6f}")
    print(f"  dy: mean={norm_stats['mean']['dy']:.6f}, std={norm_stats['std']['dy']:.6f}")  
    print(f"  dz: mean={norm_stats['mean']['dz']:.6f}, std={norm_stats['std']['dz']:.6f}")
    
    # Verificar que std no sea cero
    for component in ['dx', 'dy', 'dz']:
        if norm_stats['std'][component] == 0:
            print(f"WARNING: std for {component} is zero! Setting to 1.0")
            norm_stats['std'][component] = 1.0
    
    dx_norm = (dx - norm_stats['mean']['dx']) / norm_stats['std']['dx']
    dy_norm = (dy - norm_stats['mean']['dy']) / norm_stats['std']['dy']
    dz_norm = (dz - norm_stats['mean']['dz']) / norm_stats['std']['dz']
    
    print(f"Input deltas stats: dx={np.mean(dx):.3f}±{np.std(dx):.3f}, "
          f"dy={np.mean(dy):.3f}±{np.std(dy):.3f}, dz={np.mean(dz):.3f}±{np.std(dz):.3f}")
    print(f"Normalized deltas stats: dx={np.mean(dx_norm):.3f}±{np.std(dx_norm):.3f}, "
          f"dy={np.mean(dy_norm):.3f}±{np.std(dy_norm):.3f}, dz={np.mean(dz_norm):.3f}±{np.std(dz_norm):.3f}")
    
    return dx_norm, dy_norm, dz_norm

def denormalize_deltas(dx_norm, dy_norm, dz_norm, norm_stats):
    """
    Desnormaliza deltas filtrados.
    """
    dx = dx_norm * norm_stats['std']['dx'] + norm_stats['mean']['dx']
    dy = dy_norm * norm_stats['std']['dy'] + norm_stats['mean']['dy']
    dz = dz_norm * norm_stats['std']['dz'] + norm_stats['mean']['dz']
    
    print(f"Filtered deltas stats: dx={np.mean(dx):.3f}±{np.std(dx):.3f}, "
          f"dy={np.mean(dy):.3f}±{np.std(dy):.3f}, dz={np.mean(dz):.3f}±{np.std(dz):.3f}")
    
    return dx, dy, dz

def integrate_deltas(dx, dy, dz, x0, y0, z0):
    """
    Integra deltas para obtener coordenadas absolutas.
    
    El primer punto debe mantenerse igual, los siguientes
    son la posición anterior + delta.
    
    CRÍTICO: 
    - Posición[0] = (x0, y0, z0) [EXACTA]
    - Posición[i] = Posición[i-1] + Delta[i] para i > 0
    """
    x = np.zeros_like(dx)
    y = np.zeros_like(dy)  
    z = np.zeros_like(dz)
    
    # El primer punto es la posición inicial exacta
    x[0] = x0
    y[0] = y0
    z[0] = z0
    
    # Integración secuencial: posición[i] = posición[i-1] + delta[i]
    for i in range(1, len(dx)):
        x[i] = x[i-1] + dx[i]
        y[i] = y[i-1] + dy[i]
        z[i] = z[i-1] + dz[i]
    
    print(f"Integration verification:")
    print(f"  Original first point: ({x0:.3f}, {y0:.3f}, {z0:.3f})")
    print(f"  Integrated first point: ({x[0]:.3f}, {y[0]:.3f}, {z[0]:.3f})")
    print(f"  Position preservation error: {np.sqrt((x[0]-x0)**2 + (y[0]-y0)**2):.6f}m")
    
    # Verificar que la integración es correcta comparando con cumsum
    x_cumsum = np.cumsum(dx) + x0
    print(f"  Integration vs cumsum difference: {np.mean(np.abs(x - x_cumsum)):.6f}m")
    
    return x, y, z

def create_gpx_with_gpxpy(lat, lon, ele, time=None, output_path="filtered_track.gpx"):
    """
    Crea un archivo GPX usando gpxpy.
    """
    print(f"Creating GPX with {len(lat)} points...")
    
    # Crear objeto GPX
    gpx = gpxpy.gpx.GPX()
    
    # Crear track y segmento
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    
    # Añadir puntos
    for i in range(len(lat)):
        point_time = time[i] if time is not None and i < len(time) and pd.notna(time[i]) else None
        
        point = gpxpy.gpx.GPXTrackPoint(
            latitude=float(lat[i]),
            longitude=float(lon[i]),
            elevation=float(ele[i]),
            time=point_time
        )
        gpx_segment.points.append(point)
    
    # Escribir archivo
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(gpx.to_xml())
    
    print(f"Filtered track saved to {output_path}")

def masked_mae_loss(y_true, y_pred):
    """Custom loss function for loading legacy models."""
    mask = tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True) > 1e-7
    mask = tf.cast(mask, tf.float32)
    
    mae = tf.abs(y_true - y_pred)
    masked_mae = mae * mask
    
    sum_mae = tf.reduce_sum(masked_mae)
    sum_mask = tf.reduce_sum(mask)
    
    return sum_mae / (sum_mask + 1e-7)

def residual_mae_loss(y_true, y_pred):
    """Per-timestep MAE for residual correction models."""
    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1)

def create_v3_model(sequence_length=3600, n_features=3):
    """Create the residual v3 architecture for weight-only fallback loading."""
    model = Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, n_features), name="input_layer"),
            Masking(mask_value=0.0, name="masking"),
            LSTM(
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
    model(np.zeros((1, sequence_length, n_features), dtype=np.float32))
    return model

def load_v3_weights_fallback(model_path):
    """Load a v3 .keras archive by rebuilding the architecture and loading weights."""
    model = create_v3_model()
    with tempfile.TemporaryDirectory() as tmp_dir:
        with zipfile.ZipFile(model_path, "r") as archive:
            archive.extract("model.weights.h5", tmp_dir)
        weights_path = Path(tmp_dir) / "model.weights.h5"
        model.load_weights(weights_path)
    print("Model loaded with v3 weight fallback.")
    return model

def load_model_robust(model_path):
    """Load the model robustly."""
    print("Loading neural network model...")
    
    try:
        model = load_model(
            model_path,
            custom_objects={
                'masked_mae_loss': masked_mae_loss,
                'residual_mae_loss': residual_mae_loss
            }
        )
        print("Model loaded successfully with custom objects.")
        return model
    except Exception as e:
        print(f"Direct model load failed ({e.__class__.__name__}); trying compile=False.")
    
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully without compilation.")
        return model
    except Exception as e:
        print(f"Model load with compile=False failed ({e.__class__.__name__}); trying v3 weight fallback.")

    try:
        return load_v3_weights_fallback(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def apply_neural_network_filter(
    track_df,
    model_path="models/model_final_v3.keras",
    norm_stats_path="data/input/norm_stats_train.json",
    model_output="residual",
    pattern_gpx=None,
    anchors_per_hour=8.0,
    min_anchors=8,
    max_anchors=0,
    anchor_error_radius=0,
    anchor_channels="x,y,z",
    anchor_interpolation="cubic",
    anchor_edge_blend_points=30,
    anchor_trim_to_pattern=False,
    anchor_edge_skip_points=0,
):
    """
    Aplica el filtro de red neuronal a un track con correcciones geodésicas.
    """
    print("Loading normalization statistics...")
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    model = load_model_robust(model_path)
    
    print("Setting up geodesic projection...")
    # Usar centro del track para proyección
    lat_center = track_df['lat'].mean()
    lon_center = track_df['lon'].mean()
    transformer = setup_projection(lat_center, lon_center)
    
    print("Converting to metric coordinates...")
    # Usar primer punto como referencia para coordenadas relativas
    lat_ref = track_df['lat'].iloc[0]
    lon_ref = track_df['lon'].iloc[0]
    
    x, y = latlon_to_meters(track_df['lat'], track_df['lon'], transformer, lat_ref, lon_ref)
    z = track_df['ele'].values
    
    print(f"Coordinate range: x=[{x.min():.1f}, {x.max():.1f}], y=[{y.min():.1f}, {y.max():.1f}], z=[{z.min():.1f}, {z.max():.1f}]")
    
    print("Calculating deltas correctly...")
    dx, dy, dz = calculate_deltas(x, y, z)
    
    print("Normalizing deltas...")
    dx_norm, dy_norm, dz_norm = normalize_deltas(dx, dy, dz, norm_stats)
    
    print("Applying neural network filter...")
    sequence_length = len(dx_norm)
    max_sequence = 3600
    
    # Process in chunks so both long and short tracks work.
    print(f"Processing {sequence_length} points in chunks of {max_sequence}")
    
    filtered_dx, filtered_dy, filtered_dz = [], [], []
    
    for i in range(0, sequence_length, max_sequence):
        end_idx = min(i + max_sequence, sequence_length)
        chunk_len = end_idx - i
        
        # Create a padded tensor up to max_sequence.
        input_data = np.zeros((1, max_sequence, 3))
        input_data[0, :chunk_len, 0] = dx_norm[i:end_idx]
        input_data[0, :chunk_len, 1] = dy_norm[i:end_idx]
        input_data[0, :chunk_len, 2] = dz_norm[i:end_idx]
        
        # Aplicar modelo y extraer solo la porción válida
        model_chunk = model.predict(input_data, verbose=0)
        if model_output == "residual":
            filtered_chunk = input_data + model_chunk
        elif model_output == "direct":
            filtered_chunk = model_chunk
        else:
            raise ValueError(f"Unsupported model output mode: {model_output}")
        #filtered_chunk = input_data.copy() # puenteado para pruebas
        
        filtered_dx.extend(filtered_chunk[0, :chunk_len, 0])
        filtered_dy.extend(filtered_chunk[0, :chunk_len, 1])
        filtered_dz.extend(filtered_chunk[0, :chunk_len, 2])
    
    # Convert to numpy arrays.
    filtered_dx = np.array(filtered_dx)
    filtered_dy = np.array(filtered_dy)
    filtered_dz = np.array(filtered_dz)
    
    print("Denormalizing filtered deltas...")
    dx_filt, dy_filt, dz_filt = denormalize_deltas(filtered_dx, filtered_dy, filtered_dz, norm_stats)
    
    print("Integrating deltas correctly to get absolute coordinates...")
    x_filt, y_filt, z_filt = integrate_deltas(dx_filt, dy_filt, dz_filt, x[0], y[0], z[0])
    
    print("Converting back to lat/lon...")
    lat_filt, lon_filt = meters_to_latlon(x_filt, y_filt, transformer, lat_ref, lon_ref)
    
    # Verificar preservación de posición inicial
    print(f"Position preservation check:")
    print(f"  Original: ({track_df['lat'].iloc[0]:.8f}, {track_df['lon'].iloc[0]:.8f})")
    print(f"  Filtered: ({lat_filt[0]:.8f}, {lon_filt[0]:.8f})")
    
    # Create the filtered track DataFrame.
    filtered_df = track_df.copy()
    filtered_df['lat'] = lat_filt
    filtered_df['lon'] = lon_filt
    filtered_df['ele'] = z_filt

    if pattern_gpx is not None:
        pattern_df = parse_gpx(pattern_gpx)
        filtered_df = apply_pattern_anchor_correction(
            filtered_df,
            pattern_df,
            anchors_per_hour=anchors_per_hour,
            min_anchors=min_anchors,
            max_anchors=max_anchors,
            anchor_error_radius=anchor_error_radius,
            anchor_channels=anchor_channels,
            anchor_interpolation=anchor_interpolation,
            anchor_edge_blend_points=anchor_edge_blend_points,
            anchor_trim_to_pattern=anchor_trim_to_pattern,
            anchor_edge_skip_points=anchor_edge_skip_points,
        )
    
    return filtered_df

def main(default_slow_correction="none", default_anchor_trim_to_pattern=False, default_anchor_edge_skip_points=0):
    """Función principal."""
    parser = argparse.ArgumentParser(description='Filter GPS track using neural network')
    parser.add_argument('input_gpx', help='Input GPX file')
    parser.add_argument('output_gpx', nargs='?', help='Output filtered GPX file')
    parser.add_argument('--model', default='models/model_final_v3.keras', help='Path to trained model')
    parser.add_argument('--norm-stats', default='data/input/norm_stats_train.json', help='Path to normalization statistics')
    parser.add_argument('--model-output', choices=['residual', 'direct'], default='residual', help='Model output interpretation')
    parser.add_argument('--slow-correction', choices=['none', 'pattern-anchor'], default=default_slow_correction, help='Experimental slow correction mode')
    parser.add_argument('--pattern-gpx', default=None, help='Pattern GPX path for pattern-anchor correction')
    parser.add_argument('--anchors-per-hour', type=float, default=8.0, help='Pattern-anchor density per hour')
    parser.add_argument('--min-anchors', type=int, default=8, help='Minimum pattern anchors per track')
    parser.add_argument('--max-anchors', type=int, default=0, help='Maximum pattern anchors per track; 0 means no cap')
    parser.add_argument('--anchor-error-radius', type=int, default=0, help='Local radius for anchor error averaging')
    parser.add_argument('--anchor-channels', default='x,y,z', help='Comma-separated channels for pattern-anchor correction')
    parser.add_argument('--anchor-interpolation', choices=['linear', 'cubic'], default='cubic', help='Pattern-anchor interpolation mode')
    parser.add_argument('--anchor-edge-blend-points', type=int, default=30, help='Points used to blend in the first pattern-anchor correction')
    parser.add_argument('--anchor-edge-skip-points', type=int, default=default_anchor_edge_skip_points, help='Common-timeline edge points excluded from anchor selection')
    parser.add_argument('--anchor-trim-to-pattern', action='store_true', default=default_anchor_trim_to_pattern, help='Trim output to the common pattern-track timeline')
    parser.add_argument('--no-anchor-trim-to-pattern', dest='anchor_trim_to_pattern', action='store_false', help='Keep points outside the common pattern-track timeline')
    parser.add_argument('--suffix', default='nn_filtered', help='Suffix for auto-generated output filename')
    
    args = parser.parse_args()
    
    # Verificar dependencias
    try:
        import gpxpy
        import pyproj
    except ImportError as e:
        print(f"ERROR: Missing required library: {e}")
        print("Install with: pip install gpxpy pyproj")
        sys.exit(1)
    
    try:
        if args.output_gpx is None:
            input_path = Path(args.input_gpx)
            output_dir = input_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_filename = f"{input_path.stem}_{args.suffix}{input_path.suffix}"
            args.output_gpx = output_dir / output_filename
            print(f"Output file auto-generated: {args.output_gpx}")
        else:
            Path(args.output_gpx).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {args.input_gpx}...")
        
        track_df = parse_gpx(args.input_gpx)

        pattern_gpx = None
        if args.slow_correction == 'pattern-anchor':
            pattern_gpx = args.pattern_gpx or infer_pattern_path(args.input_gpx)
            print(f"Pattern-anchor correction enabled with pattern: {pattern_gpx}")
        
        filtered_df = apply_neural_network_filter(
            track_df,
            model_path=args.model,
            norm_stats_path=args.norm_stats,
            model_output=args.model_output,
            pattern_gpx=pattern_gpx,
            anchors_per_hour=args.anchors_per_hour,
            min_anchors=args.min_anchors,
            max_anchors=args.max_anchors,
            anchor_error_radius=args.anchor_error_radius,
            anchor_channels=args.anchor_channels,
            anchor_interpolation=args.anchor_interpolation,
            anchor_edge_blend_points=args.anchor_edge_blend_points,
            anchor_trim_to_pattern=args.anchor_trim_to_pattern,
            anchor_edge_skip_points=args.anchor_edge_skip_points,
        )
        
        create_gpx_with_gpxpy(
            filtered_df['lat'],
            filtered_df['lon'],
            filtered_df['ele'],
            filtered_df['time'] if 'time' in filtered_df.columns else None,
            str(args.output_gpx)
        )
        
        print(f"SUCCESS: Filtering completed!")
        print(f"   Input: {len(track_df)} points")
        print(f"   Output: {len(filtered_df)} points")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
