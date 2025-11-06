#!/usr/bin/env python3
"""
Script para filtrar un track GPS usando la red neuronal entrenada.

El proceso es:
1. Cargar track GPX de entrada (ya sampleado a 1Hz)
2. Convertir a coordenadas métricas locales
3. Calcular deltas (dx, dy, dz)
4. Normalizar usando estadísticas del entrenamiento
5. Aplicar filtro de red neuronal
6. Desnormalizar deltas filtrados
7. Regenerar coordenadas absolutas
8. Guardar como GPX filtrado

Uso:
    python 7_nn_filter.py input_track.gpx output_track.gpx
    python 7_nn_filter.py input_track.gpx output_track.gpx --model custom_model.h5
"""

import numpy as np
import pandas as pd
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking

def parse_gpx(gpx_path):
    """
    Parsea un archivo GPX y extrae lat, lon, ele, time.
    
    Returns:
        DataFrame con columnas: lat, lon, ele, time
    """
    tree = ET.parse(gpx_path)
    root = tree.getroot()
    
    # Manejar namespace de GPX
    ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
    if root.tag.startswith('{'):
        # Ya tiene namespace
        ns = {'gpx': root.tag.split('}')[0][1:]}
    
    points = []
    
    # Buscar todos los trackpoints
    for trkpt in root.findall('.//gpx:trkpt', ns):
        try:
            lat = float(trkpt.get('lat'))
            lon = float(trkpt.get('lon'))
            
            # Elevación
            ele_elem = trkpt.find('gpx:ele', ns)
            ele = float(ele_elem.text) if ele_elem is not None else 0.0
            
            # Tiempo
            time_elem = trkpt.find('gpx:time', ns)
            time_str = time_elem.text if time_elem is not None else None
            
            points.append({
                'lat': lat,
                'lon': lon, 
                'ele': ele,
                'time': time_str
            })
            
        except (ValueError, TypeError) as e:
            print(f"Warning: Error parsing point {trkpt}: {e}")
            continue
    
    if not points:
        raise ValueError(f"No valid trackpoints found in {gpx_path}")
    
    df = pd.DataFrame(points)
    
    # Convertir tiempo si está disponible
    if df['time'].notna().any():
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
    
    print(f"Loaded {len(df)} points from {gpx_path}")
    return df

def latlon_to_meters(lat, lon, lat_ref, lon_ref):
    """
    Convierte lat/lon a coordenadas métricas locales usando proyección simple.
    
    Args:
        lat, lon: Arrays de latitud y longitud
        lat_ref, lon_ref: Punto de referencia (primer punto del track)
    
    Returns:
        x, y: Coordenadas en metros
    """
    # Constantes aproximadas para conversión
    lat_to_m = 111320.0  # metros por grado de latitud
    
    # Corrección de longitud por latitud
    lon_to_m = 111320.0 * np.cos(np.radians(lat_ref))
    
    # Calcular desplazamientos en metros
    x = (lon - lon_ref) * lon_to_m
    y = (lat - lat_ref) * lat_to_m
    
    return x, y

def meters_to_latlon(x, y, lat_ref, lon_ref):
    """
    Convierte coordenadas métricas locales de vuelta a lat/lon.
    
    Args:
        x, y: Coordenadas en metros
        lat_ref, lon_ref: Punto de referencia original
    
    Returns:
        lat, lon: Latitud y longitud
    """
    # Constantes aproximadas para conversión
    lat_to_m = 111320.0
    lon_to_m = 111320.0 * np.cos(np.radians(lat_ref))
    
    # Convertir de vuelta
    lat = lat_ref + (y / lat_to_m)
    lon = lon_ref + (x / lon_to_m)
    
    return lat, lon

def calculate_deltas(x, y, z):
    """
    Calcula deltas entre puntos consecutivos.
    
    Args:
        x, y, z: Arrays de coordenadas
        
    Returns:
        dx, dy, dz: Arrays de deltas
    """
    dx = np.diff(x, prepend=x[0])  # Primer delta = 0
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])
    
    return dx, dy, dz

def normalize_deltas(dx, dy, dz, norm_stats):
    """
    Normaliza deltas usando estadísticas de entrenamiento.
    """
    dx_norm = (dx - norm_stats['mean']['dx']) / norm_stats['std']['dx']
    dy_norm = (dy - norm_stats['mean']['dy']) / norm_stats['std']['dy'] 
    dz_norm = (dz - norm_stats['mean']['dz']) / norm_stats['std']['dz']
    
    return dx_norm, dy_norm, dz_norm

def denormalize_deltas(dx_norm, dy_norm, dz_norm, norm_stats):
    """
    Desnormaliza deltas filtrados.
    """
    dx = dx_norm * norm_stats['std']['dx'] + norm_stats['mean']['dx']
    dy = dy_norm * norm_stats['std']['dy'] + norm_stats['mean']['dy']
    dz = dz_norm * norm_stats['std']['dz'] + norm_stats['mean']['dz']
    
    return dx, dy, dz

def integrate_deltas(dx, dy, dz, x0, y0, z0):
    """
    Integra deltas para obtener coordenadas absolutas.
    
    Args:
        dx, dy, dz: Deltas filtrados
        x0, y0, z0: Punto inicial
        
    Returns:
        x, y, z: Coordenadas absolutas
    """
    x = np.cumsum(dx) + x0
    y = np.cumsum(dy) + y0  
    z = np.cumsum(dz) + z0
    
    return x, y, z

def create_gpx(lat, lon, ele, time=None, output_path="filtered_track.gpx"):
    """
    Crea un archivo GPX con los puntos filtrados.
    """
    # Crear estructura GPX
    gpx = ET.Element("gpx")
    gpx.set("version", "1.1")
    gpx.set("creator", "nn_filter")
    gpx.set("xmlns", "http://www.topografix.com/GPX/1/1")
    
    trk = ET.SubElement(gpx, "trk")
    name = ET.SubElement(trk, "name")
    name.text = "Filtered Track"
    
    trkseg = ET.SubElement(trk, "trkseg")
    
    for i in range(len(lat)):
        trkpt = ET.SubElement(trkseg, "trkpt")
        trkpt.set("lat", f"{lat[i]:.8f}")
        trkpt.set("lon", f"{lon[i]:.8f}")
        
        # Elevación
        ele_elem = ET.SubElement(trkpt, "ele")
        ele_elem.text = f"{ele[i]:.2f}"
        
        # Tiempo si está disponible
        if time is not None and i < len(time):
            time_elem = ET.SubElement(trkpt, "time")
            if pd.notna(time[i]):
                time_elem.text = time[i].strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Escribir archivo
    tree = ET.ElementTree(gpx)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Filtered track saved to {output_path}")

def masked_mae_loss(y_true, y_pred):
    """
    Custom loss function that may be needed when loading the model.
    Mean Absolute Error que ignora valores enmascarados.
    """
    # Crear máscara: verdadero donde NO hay padding (no todos los valores son 0)
    mask = tf.reduce_sum(tf.abs(y_true), axis=-1, keepdims=True) > 1e-7
    mask = tf.cast(mask, tf.float32)
    
    # Calcular MAE solo en posiciones válidas
    mae = tf.abs(y_true - y_pred)
    masked_mae = mae * mask
    
    # Promedio sobre posiciones válidas
    sum_mae = tf.reduce_sum(masked_mae)
    sum_mask = tf.reduce_sum(mask) 
    
    return sum_mae / (sum_mask + 1e-7)  # Evitar división por 0

def create_compatible_model(sequence_length=3600, n_features=3):
    """Crea un modelo compatible con la arquitectura original."""
    model = Sequential([
        # Capa de enmascaramiento para ignorar padding
        Masking(mask_value=0.0, input_shape=(sequence_length, n_features)),
        
        # LSTM con 128 unidades, devuelve secuencias completas
        LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
        
        # Capa densa con 64 neuronas y ReLU
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        # Capa de salida con 3 neuronas (dx, dy, dz) y activación lineal
        Dense(n_features, activation='linear')
    ])
    
    return model

def load_model_robust(model_path):
    """
    Carga el modelo de forma robusta manejando diferentes versiones de TensorFlow/Keras.
    """
    print("Loading neural network model...")
    
    # Primero intentar con custom objects
    try:
        model = load_model(model_path, custom_objects={'masked_mae_loss': masked_mae_loss})
        print(f"Model loaded successfully with custom objects.")
        return model
    except Exception as e:
        print(f"Warning: Error loading with custom objects: {e}")
    
    # Intentar carga estándar
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully with standard loading.")
        return model
    except Exception as e:
        print(f"Warning: Error with standard loading: {e}")
    
    # Intentar con compile=False para evitar problemas de optimizador
    try:
        model = load_model(model_path, compile=False)
        print(f"Model loaded successfully without compilation.")
        print("Note: Model loaded without optimizer state - inference only.")
        return model
    except Exception as e:
        print(f"Warning: Error with compile=False: {e}")
    
    # Como último recurso, crear modelo compatible y cargar pesos
    try:
        print("Attempting to load weights into compatible model architecture...")
        
        # Crear modelo compatible
        compatible_model = create_compatible_model()
        
        # Intentar cargar solo los pesos
        # Primero necesitamos compilar el modelo para que tenga la estructura correcta
        compatible_model.compile(optimizer='adam', loss='mae')
        
        # Cargar el modelo original para extraer pesos
        import h5py
        with h5py.File(model_path, 'r') as f:
            # Verificar si tiene pesos guardados
            if 'model_weights' in f.keys():
                compatible_model.load_weights(model_path)
                print("Model weights loaded successfully into compatible architecture.")
                return compatible_model
            else:
                raise ValueError("No model weights found in file")
                
    except Exception as e:
        print(f"Error loading weights into compatible model: {e}")
        
    raise RuntimeError(f"Failed to load model from {model_path} with any method")

def apply_neural_network_filter(track_df, model_path="final_model.h5", norm_stats_path="data/input/norm_stats.json"):
    """
    Aplica el filtro de red neuronal a un track.
    
    Args:
        track_df: DataFrame con columnas lat, lon, ele, time
        model_path: Ruta al modelo entrenado
        norm_stats_path: Ruta a las estadísticas de normalización
        
    Returns:
        DataFrame con track filtrado
    """
    print("Loading normalization statistics...")
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    
    model = load_model_robust(model_path)
    
    print("Converting to metric coordinates...")
    # Usar primer punto como referencia
    lat_ref = track_df['lat'].iloc[0]
    lon_ref = track_df['lon'].iloc[0] 
    
    x, y = latlon_to_meters(track_df['lat'], track_df['lon'], lat_ref, lon_ref)
    z = track_df['ele'].values
    
    print("Calculating deltas...")
    dx, dy, dz = calculate_deltas(x, y, z)
    
    print("Normalizing deltas...")
    dx_norm, dy_norm, dz_norm = normalize_deltas(dx, dy, dz, norm_stats)
    
    # Preparar entrada para la red neuronal
    print("Preparing input for neural network...")
    sequence_length = len(dx_norm)
    max_sequence = 3600  # Longitud máxima que maneja la red
    
    if sequence_length > max_sequence:
        print(f"Warning: Track has {sequence_length} points, processing in chunks of {max_sequence}")
        
        # Procesar en chunks
        filtered_dx, filtered_dy, filtered_dz = [], [], []
        
        for i in range(0, sequence_length, max_sequence):
            end_idx = min(i + max_sequence, sequence_length)
            chunk_len = end_idx - i
            
            # Crear entrada con padding si es necesario
            input_data = np.zeros((1, max_sequence, 3))
            input_data[0, :chunk_len, 0] = dx_norm[i:end_idx]
            input_data[0, :chunk_len, 1] = dy_norm[i:end_idx] 
            input_data[0, :chunk_len, 2] = dz_norm[i:end_idx]
            
            # Aplicar filtro
            filtered_chunk = model.predict(input_data, verbose=0)
            
            # Extraer solo la parte válida
            filtered_dx.extend(filtered_chunk[0, :chunk_len, 0])
            filtered_dy.extend(filtered_chunk[0, :chunk_len, 1])
            filtered_dz.extend(filtered_chunk[0, :chunk_len, 2])
        
        filtered_dx = np.array(filtered_dx)
        filtered_dy = np.array(filtered_dy)  
        filtered_dz = np.array(filtered_dz)
        
    else:
        # Procesar track completo
        input_data = np.zeros((1, max_sequence, 3))
        input_data[0, :sequence_length, 0] = dx_norm
        input_data[0, :sequence_length, 1] = dy_norm
        input_data[0, :sequence_length, 2] = dz_norm
        
        print("Applying neural network filter...")
        filtered_output = model.predict(input_data, verbose=0)
        
        # Extraer solo la parte válida
        filtered_dx = filtered_output[0, :sequence_length, 0]
        filtered_dy = filtered_output[0, :sequence_length, 1]
        filtered_dz = filtered_output[0, :sequence_length, 2]
    
    print("Denormalizing filtered deltas...")
    dx_filt, dy_filt, dz_filt = denormalize_deltas(filtered_dx, filtered_dy, filtered_dz, norm_stats)
    
    print("Integrating deltas to get absolute coordinates...")
    x_filt, y_filt, z_filt = integrate_deltas(dx_filt, dy_filt, dz_filt, x[0], y[0], z[0])
    
    print("Converting back to lat/lon...")
    lat_filt, lon_filt = meters_to_latlon(x_filt, y_filt, lat_ref, lon_ref)
    
    # Crear DataFrame con track filtrado
    filtered_df = track_df.copy()
    filtered_df['lat'] = lat_filt
    filtered_df['lon'] = lon_filt
    filtered_df['ele'] = z_filt
    
    return filtered_df

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Filter GPS track using neural network')
    parser.add_argument('input_gpx', help='Input GPX file (sampled at 1Hz)')
    parser.add_argument('output_gpx', help='Output filtered GPX file')
    parser.add_argument('--model', default='final_model.h5', help='Path to trained model')
    parser.add_argument('--norm-stats', default='data/input/norm_stats.json', help='Path to normalization statistics')
    
    args = parser.parse_args()
    
    try:
        print(f"Processing {args.input_gpx}...")
        
        # Cargar track de entrada
        track_df = parse_gpx(args.input_gpx)
        
        # Aplicar filtro de red neuronal
        filtered_df = apply_neural_network_filter(
            track_df, 
            model_path=args.model,
            norm_stats_path=args.norm_stats
        )
        
        # Guardar track filtrado
        create_gpx(
            filtered_df['lat'], 
            filtered_df['lon'], 
            filtered_df['ele'],
            filtered_df['time'] if 'time' in filtered_df.columns else None,
            args.output_gpx
        )
        
        print(f"✅ Filtering completed successfully!")
        print(f"   Input: {len(track_df)} points")
        print(f"   Output: {len(filtered_df)} points")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()