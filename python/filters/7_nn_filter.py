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
from pathlib import Path
from datetime import datetime, timedelta

# Librerías especializadas
import gpxpy
import gpxpy.gpx
from pyproj import Transformer

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from accelerate.commands.menu import input

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
        print(f"Warning: {e}")
    
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully without compilation.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def apply_neural_network_filter(
    track_df,
    model_path="models/model_final_v3.keras",
    norm_stats_path="data/input/norm_stats_train.json",
    model_output="residual"
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
    
    return filtered_df

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Filter GPS track using neural network')
    parser.add_argument('input_gpx', help='Input GPX file')
    parser.add_argument('output_gpx', nargs='?', help='Output filtered GPX file')
    parser.add_argument('--model', default='models/model_final_v3.keras', help='Path to trained model')
    parser.add_argument('--norm-stats', default='data/input/norm_stats_train.json', help='Path to normalization statistics')
    parser.add_argument('--model-output', choices=['residual', 'direct'], default='residual', help='Model output interpretation')
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
        
        filtered_df = apply_neural_network_filter(
            track_df,
            model_path=args.model,
            norm_stats_path=args.norm_stats,
            model_output=args.model_output
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
