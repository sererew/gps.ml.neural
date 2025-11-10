#!/usr/bin/env python3
"""
Script para comparar todos los tracks filtrados con sus patrones de referencia.

Este script:
1. Busca todos los tracks filtrados en data/filtered/<filtro>/<pasada>/
2. Los compara con sus patrones de referencia en data/preprocessed/<pasada>/
3. Calcula métricas de desviación (distancia, desnivel, desviación 3D)
4. Genera un Excel con resultados comparativos de todos los filtros

IMPORTANTE: Solo se comparan puntos dentro del rango temporal del patrón,
excluyendo puntos que estén antes del inicio o después del final del patrón.

Métricas calculadas:
- Distancia total, desnivel positivo/negativo (con y sin umbral)
- Desviación punto a punto 3D (media y desviación estándar)
- Todas las métricas como desviación respecto al patrón

Umbrales:
- Desnivel con umbral: 5m de altura en tramos de 50m de distancia
- Velocidad mínima de movimiento: 1 km/h  
- Distancia para pendientes: 50 m

Uso:
    python 9_compare_tracks.py
    python 9_compare_tracks.py --pasadas 1,2,3
    python 9_compare_tracks.py --output results.xlsx
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import glob
from geopy.distance import geodesic
import xlsxwriter
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Configuración de umbrales
ELEVATION_THRESHOLD = 5.0  # metros
MIN_SPEED_KMH = 1.0  # km/h
SLOPE_DISTANCE = 50.0  # metros

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
    
    return df

def trim_track_to_pattern_timerange(track_df, pattern_df):
    """
    Recorta tanto el track como el patrón al rango temporal común entre ambos.
    
    Args:
        track_df: DataFrame del track filtrado
        pattern_df: DataFrame del patrón de referencia
        
    Returns:
        tuple: (track_recortado, patrón_recortado, dict con información del recorte)
    """
    trim_info = {
        'is_trimmed': False,
        'pattern_coverage_percent': 100.0,
        'track_time_range': None,
        'pattern_time_range': None,
        'points_lost_start': 0,
        'points_lost_end': 0
    }
    
    if track_df['time'].isna().any() or pattern_df['time'].isna().any():
        return track_df, pattern_df, trim_info
    
    # Obtener rangos temporales originales
    pattern_start_original = pattern_df['time'].min()
    pattern_end_original = pattern_df['time'].max()
    track_start = track_df['time'].min()
    track_end = track_df['time'].max()
    
    trim_info['pattern_time_range'] = f"{pattern_start_original} to {pattern_end_original}"
    trim_info['track_time_range'] = f"{track_start} to {track_end}"
    
    # Calcular rango temporal común (intersección)
    trim_start = max(pattern_start_original, track_start)
    trim_end = min(pattern_end_original, track_end)
    
    # Verificar si hay recorte del patrón original
    if track_start > pattern_start_original or track_end < pattern_end_original:
        trim_info['is_trimmed'] = True
        
        # Calcular cobertura del patrón original
        pattern_duration_original = (pattern_end_original - pattern_start_original).total_seconds()
        covered_duration = (trim_end - trim_start).total_seconds()
        
        if pattern_duration_original > 0:
            trim_info['pattern_coverage_percent'] = (covered_duration / pattern_duration_original) * 100
        
        # Contar puntos perdidos del patrón original
        if track_start > pattern_start_original:
            trim_info['points_lost_start'] = len(pattern_df[pattern_df['time'] < track_start])
        if track_end < pattern_end_original:
            trim_info['points_lost_end'] = len(pattern_df[pattern_df['time'] > track_end])
    
    # Aplicar recorte temporal a AMBOS (track y patrón)
    track_mask = (track_df['time'] >= trim_start) & (track_df['time'] <= trim_end)
    pattern_mask = (pattern_df['time'] >= trim_start) & (pattern_df['time'] <= trim_end)
    
    trimmed_track = track_df[track_mask].copy()
    trimmed_pattern = pattern_df[pattern_mask].copy()
    
    return trimmed_track, trimmed_pattern, trim_info

def calculate_distance_cumulative(df):
    """
    Calcula la distancia acumulada total del track.
    
    Args:
        df: DataFrame con columnas lat, lon
        
    Returns:
        float: Distancia total en metros
    """
    if len(df) < 2:
        return 0.0
    
    total_distance = 0.0
    for i in range(1, len(df)):
        coord1 = (df.iloc[i-1]['lat'], df.iloc[i-1]['lon'])
        coord2 = (df.iloc[i]['lat'], df.iloc[i]['lon'])
        distance = geodesic(coord1, coord2).meters
        total_distance += distance
    
    return total_distance

def calculate_elevation_gain_loss_by_distance(df, distance_threshold=50.0, elevation_threshold=5.0):
    """
    Calcula desnivel positivo y negativo acumulado por tramos de distancia fija.
    
    Args:
        df: DataFrame con columnas lat, lon, ele
        distance_threshold: Distancia mínima en metros para evaluar desnivel (default: 50m)
        elevation_threshold: Umbral mínimo de desnivel para considerar cambio (default: 5m)
        
    Returns:
        tuple: (gain, loss) en metros
    """
    if len(df) < 2:
        return 0.0, 0.0
    
    gain = 0.0
    loss = 0.0
    
    # Punto de referencia inicial
    ref_index = 0
    ref_elevation = df.iloc[0]['ele']
    accumulated_distance = 0.0
    
    for i in range(1, len(df)):
        # Calcular distancia desde el punto anterior
        coord1 = (df.iloc[i-1]['lat'], df.iloc[i-1]['lon'])
        coord2 = (df.iloc[i]['lat'], df.iloc[i]['lon'])
        segment_distance = geodesic(coord1, coord2).meters
        accumulated_distance += segment_distance
        
        # Si hemos acumulado la distancia mínima, evaluar desnivel
        if accumulated_distance >= distance_threshold:
            current_elevation = df.iloc[i]['ele']
            elevation_change = current_elevation - ref_elevation
            
            # Aplicar umbral de elevación
            if abs(elevation_change) >= elevation_threshold:
                if elevation_change > 0:
                    gain += elevation_change
                else:
                    loss += abs(elevation_change)
            
            # Resetear punto de referencia
            ref_index = i
            ref_elevation = current_elevation
            accumulated_distance = 0.0
    
    return gain, loss

def calculate_elevation_gain_loss(df, threshold=None):
    """
    Calcula desnivel positivo y negativo acumulado.
    
    Args:
        df: DataFrame con columna ele
        threshold: Si es None, calcula sin umbral punto a punto.
                  Si es un número, usa umbral de distancia (SLOPE_DISTANCE metros)
        
    Returns:
        tuple: (gain, loss) en metros
    """
    if len(df) < 2:
        return 0.0, 0.0
    
    if threshold is None:
        # Cálculo sin umbral - punto a punto temporal
        gain = 0.0
        loss = 0.0
        
        for i in range(1, len(df)):
            elevation_change = df.iloc[i]['ele'] - df.iloc[i-1]['ele']
            if elevation_change > 0:
                gain += elevation_change
            else:
                loss += abs(elevation_change)
        
        return gain, loss
    else:
        # Cálculo con umbral - por tramos de distancia
        return calculate_elevation_gain_loss_by_distance(
            df, 
            distance_threshold=SLOPE_DISTANCE,  # 50 metros
            elevation_threshold=threshold       # 5 metros
        )

def interpolate_track_to_pattern_times(track_df, pattern_df):
    """
    Interpola un track para que tenga exactamente los mismos tiempos que el patrón.
    
    Args:
        track_df: DataFrame del track filtrado
        pattern_df: DataFrame del patrón de referencia
        
    Returns:
        DataFrame del track interpolado a los tiempos del patrón
    """
    if track_df['time'].isna().any() or pattern_df['time'].isna().any():
        print("Warning: Missing time data, cannot interpolate by time")
        return track_df
    
    # Convertir tiempos a segundos desde el primer punto del patrón
    pattern_start = pattern_df['time'].min()
    pattern_times_sec = (pattern_df['time'] - pattern_start).dt.total_seconds()
    track_times_sec = (track_df['time'] - pattern_start).dt.total_seconds()
    
    # Interpolar cada coordenada del track a los tiempos del patrón
    interpolated_track = pattern_df[['time']].copy()  # Mantener tiempos del patrón
    
    # Interpolar lat, lon, ele
    interpolated_track['lat'] = np.interp(pattern_times_sec, track_times_sec, track_df['lat'])
    interpolated_track['lon'] = np.interp(pattern_times_sec, track_times_sec, track_df['lon'])
    interpolated_track['ele'] = np.interp(pattern_times_sec, track_times_sec, track_df['ele'])
    
    print(f"  Track interpolated from {len(track_df)} to {len(interpolated_track)} points")
    
    return interpolated_track

def calculate_point_deviation_3d(track_df, pattern_df):
    """
    Calcula desviación 3D punto a punto entre track y patrón.
    
    Args:
        track_df: DataFrame del track filtrado
        pattern_df: DataFrame del patrón de referencia
        
    Returns:
        tuple: (mean_deviation, std_deviation) en metros
    """
    if len(track_df) != len(pattern_df):
        print(f"Length mismatch - track: {len(track_df)}, pattern: {len(pattern_df)}")
        # Interpolar track a los tiempos exactos del patrón
        track_df = interpolate_track_to_pattern_times(track_df, pattern_df)
    
    deviations = []
    
    for i in range(len(track_df)):
        # Distancia horizontal
        coord_track = (track_df.iloc[i]['lat'], track_df.iloc[i]['lon'])
        coord_pattern = (pattern_df.iloc[i]['lat'], pattern_df.iloc[i]['lon'])
        horizontal_dist = geodesic(coord_track, coord_pattern).meters
        
        # Distancia vertical
        vertical_dist = abs(track_df.iloc[i]['ele'] - pattern_df.iloc[i]['ele'])
        
        # Distancia 3D
        deviation_3d = np.sqrt(horizontal_dist**2 + vertical_dist**2)
        deviations.append(deviation_3d)
    
    deviations = np.array(deviations)
    return np.mean(deviations), np.std(deviations)

def find_pattern_file(pasada, preprocessed_dir):
    """
    Encuentra el archivo de patrón para una pasada dada.
    
    Args:
        pasada: Nombre de la pasada
        preprocessed_dir: Directorio de archivos preprocessados
        
    Returns:
        str: Ruta al archivo de patrón
    """
    pattern_file = os.path.join(preprocessed_dir, pasada, f"{pasada}_aligned_pattern_resampled.gpx")
    
    if not os.path.exists(pattern_file):
        raise FileNotFoundError(f"Pattern file not found: {pattern_file}")
    
    return pattern_file

def find_all_filtered_tracks(filtered_dir):
    """
    Encuentra todos los tracks filtrados organizados por filtro y pasada.
    
    Args:
        filtered_dir: Directorio raíz de tracks filtrados
        
    Returns:
        dict: {filter_name: {pasada: [track_files]}}
    """
    filtered_tracks = defaultdict(lambda: defaultdict(list))
    
    # Buscar estructura: data/filtered/<filtro>/<pasada>/*.gpx
    for filter_dir in glob.glob(os.path.join(filtered_dir, "*")):
        if not os.path.isdir(filter_dir):
            continue
            
        filter_name = os.path.basename(filter_dir)
        
        for pasada_dir in glob.glob(os.path.join(filter_dir, "*")):
            if not os.path.isdir(pasada_dir):
                continue
                
            pasada = os.path.basename(pasada_dir)
            
            # Buscar archivos GPX filtrados
            gpx_files = glob.glob(os.path.join(pasada_dir, "*_filtered.gpx"))
            if gpx_files:
                filtered_tracks[filter_name][pasada].extend(gpx_files)
    
    return filtered_tracks

def calculate_track_metrics(track_df):
    """
    Calcula todas las métricas de un track.
    
    Args:
        track_df: DataFrame del track
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    metrics = {}
    
    # Métricas básicas
    metrics['total_points'] = len(track_df)
    metrics['total_length'] = calculate_distance_cumulative(track_df)
    
    # Desniveles sin umbral
    gain, loss = calculate_elevation_gain_loss(track_df)
    metrics['total_elevation_gain'] = gain
    metrics['total_elevation_loss'] = loss
    
    # Desniveles con umbral
    gain_th, loss_th = calculate_elevation_gain_loss(track_df, threshold=ELEVATION_THRESHOLD)
    metrics['total_elevation_gain_threshold'] = gain_th
    metrics['total_elevation_loss_threshold'] = loss_th
    
    return metrics

def process_single_track(track_file, pattern_file, filter_name):
    """
    Procesa un track individual comparándolo con su patrón.
    
    Args:
        track_file: Ruta al track filtrado
        pattern_file: Ruta al patrón de referencia
        filter_name: Nombre del filtro
        
    Returns:
        dict: Resultados de la comparación
    """
    try:
        print(f"    Processing: {os.path.basename(track_file)}")
        
        # Cargar track y patrón
        track_df = parse_gpx(track_file)
        pattern_df = parse_gpx(pattern_file)
        
        # Recortar track al rango temporal del patrón
        track_df, pattern_df, trim_info = trim_track_to_pattern_timerange(track_df, pattern_df)
        
        if len(track_df) == 0:
            print(f"Warning: No overlapping time range found")
            return None
        
        # Calcular métricas del patrón
        pattern_metrics = calculate_track_metrics(pattern_df)
        
        # Calcular métricas del track filtrado
        track_metrics = calculate_track_metrics(track_df)
        
        # Calcular desviación punto a punto 3D
        mean_dev_3d, std_dev_3d = calculate_point_deviation_3d(track_df, pattern_df)
        
        # Crear resultado
        result = {
            'track_name': os.path.basename(track_file).replace(f'_{filter_name}_filtered.gpx', ''),
            'filter_name': filter_name,
            'total_points': track_metrics['total_points'],
            
            # Métricas del patrón
            'total_pattern_length': pattern_metrics['total_length'],
            'total_pattern_elevation_gain': pattern_metrics['total_elevation_gain'],
            'total_pattern_elevation_loss': pattern_metrics['total_elevation_loss'],
            'total_pattern_elevation_gain_threshold': pattern_metrics['total_elevation_gain_threshold'],
            'total_pattern_elevation_loss_threshold': pattern_metrics['total_elevation_loss_threshold'],
            
            # Desviaciones respecto al patrón
            'total_length_deviation': track_metrics['total_length'] - pattern_metrics['total_length'],
            'total_elevation_gain_deviation': track_metrics['total_elevation_gain'] - pattern_metrics['total_elevation_gain'],
            'total_elevation_loss_deviation': track_metrics['total_elevation_loss'] - pattern_metrics['total_elevation_loss'],
            'total_elevation_gain_deviation_threshold': track_metrics['total_elevation_gain_threshold'] - pattern_metrics['total_elevation_gain_threshold'],
            'total_elevation_loss_deviation_threshold': track_metrics['total_elevation_loss_threshold'] - pattern_metrics['total_elevation_loss_threshold'],
            'mean_point_deviation': mean_dev_3d,
            'std_point_deviation': std_dev_3d,
            
            # Información de recorte
            'is_trimmed': trim_info['is_trimmed'],
            'pattern_coverage_percent': trim_info['pattern_coverage_percent'],
            'track_time_range': trim_info['track_time_range'],
            'pattern_time_range': trim_info['pattern_time_range'],
            'points_lost_start': trim_info['points_lost_start'],
            'points_lost_end': trim_info['points_lost_end']
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {track_file}: {e}")
        return None

def process_single_track_parallel(args):
    """
    Función auxiliar para procesamiento paralelo.
    
    Args:
        args: Tupla con (track_file, pattern_file, filter_name, pasada)
        
    Returns:
        dict: Resultados de la comparación con pasada incluida
    """
    track_file, pattern_file, filter_name, pasada = args
    
    try:
        # Cargar track y patrón
        track_df = parse_gpx(track_file)
        pattern_df = parse_gpx(pattern_file)
        
        # Recortar track al rango temporal del patrón
        track_df, pattern_df, trim_info = trim_track_to_pattern_timerange(track_df, pattern_df)
        
        if len(track_df) == 0:
            print(f"Warning: No overlapping time range found for {os.path.basename(track_file)}")
            return None
        
        # Calcular métricas del patrón
        pattern_metrics = calculate_track_metrics(pattern_df)
        
        # Calcular métricas del track filtrado
        track_metrics = calculate_track_metrics(track_df)
        
        # Calcular desviación punto a punto 3D
        mean_dev_3d, std_dev_3d = calculate_point_deviation_3d(track_df, pattern_df)
        
        # Crear resultado
        result = {
            'pasada': pasada,
            'track_name': os.path.basename(track_file).replace(f'_{filter_name}_filtered.gpx', ''),
            'filter_name': filter_name,
            'total_points': track_metrics['total_points'],
            
            # Métricas del patrón
            'total_pattern_length': pattern_metrics['total_length'],
            'total_pattern_elevation_gain': pattern_metrics['total_elevation_gain'],
            'total_pattern_elevation_loss': pattern_metrics['total_elevation_loss'],
            'total_pattern_elevation_gain_threshold': pattern_metrics['total_elevation_gain_threshold'],
            'total_pattern_elevation_loss_threshold': pattern_metrics['total_elevation_loss_threshold'],
            
            # Desviaciones respecto al patrón
            'total_length_deviation': track_metrics['total_length'] - pattern_metrics['total_length'],
            'total_elevation_gain_deviation': track_metrics['total_elevation_gain'] - pattern_metrics['total_elevation_gain'],
            'total_elevation_loss_deviation': track_metrics['total_elevation_loss'] - pattern_metrics['total_elevation_loss'],
            'total_elevation_gain_deviation_threshold': track_metrics['total_elevation_gain_threshold'] - pattern_metrics['total_elevation_gain_threshold'],
            'total_elevation_loss_deviation_threshold': track_metrics['total_elevation_loss_threshold'] - pattern_metrics['total_elevation_loss_threshold'],
            'mean_point_deviation': mean_dev_3d,
            'std_point_deviation': std_dev_3d,
            
            # Información de recorte
            'is_trimmed': trim_info['is_trimmed'],
         
            'pattern_coverage_percent': trim_info['pattern_coverage_percent'],
            'track_time_range': trim_info['track_time_range'],
            'pattern_time_range': trim_info['pattern_time_range'],
            'points_lost_start': trim_info['points_lost_start'],
            'points_lost_end': trim_info['points_lost_end']
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {os.path.basename(track_file)}: {e}")
        return None

def compare_all_tracks(filtered_tracks, preprocessed_dir, selected_pasadas=None):
    """
    Compara todos los tracks filtrados con sus patrones.
    
    Args:
        filtered_tracks: Dict con tracks filtrados organizados por filtro y pasada
        preprocessed_dir: Directorio de archivos preprocessados
        selected_pasadas: Lista de pasadas a procesar (None = todas)
        
    Returns:
        list: Lista de resultados de comparación
    """
    results = []
    
    for filter_name, pasadas_dict in filtered_tracks.items():
        print(f"\nProcessing filter: {filter_name}")
        
        for pasada, track_files in pasadas_dict.items():
            if selected_pasadas is not None and pasada not in selected_pasadas:
                continue
                
            print(f"  Processing pasada: {pasada}")
            
            try:
                # Encontrar archivo de patrón
                pattern_file = find_pattern_file(pasada, preprocessed_dir)
                
                # Procesar cada track de esta pasada
                for track_file in track_files:
                    result = process_single_track(track_file, pattern_file, filter_name)
                    if result is not None:
                        result['pasada'] = pasada
                        results.append(result)
                        
            except Exception as e:
                print(f"Error processing pasada {pasada}: {e}")
                continue
    
    return results

def compare_all_tracks_parallel(filtered_tracks, preprocessed_dir, selected_pasadas=None):
    """
    Compara todos los tracks filtrados con sus patrones usando procesamiento paralelo.
    
    Args:
        filtered_tracks: Dict con tracks filtrados organizados por filtro y pasada
        preprocessed_dir: Directorio de archivos preprocessados
        selected_pasadas: Lista de pasadas a procesar (None = todas)
        
    Returns:
        list: Lista de resultados de comparación
    """
    results = []
    total_tasks = 0
    completed_tasks = 0
    
    # Contar total de tareas para mostrar progreso
    for filter_name, pasadas_dict in filtered_tracks.items():
        for pasada, track_files in pasadas_dict.items():
            if selected_pasadas is not None and pasada not in selected_pasadas:
                continue
            total_tasks += len(track_files)
    
    print(f"📊 Total tasks to process: {total_tasks}")
    print("💡 Press Ctrl+C to gracefully stop and save partial results")
    print()
    
    with ProcessPoolExecutor() as executor:
        future_to_track = {}
        
        # Enviar todas las tareas al pool
        for filter_name, pasadas_dict in filtered_tracks.items():
            print(f"\n🔄 Processing filter: {filter_name}")
            
            for pasada, track_files in pasadas_dict.items():
                if selected_pasadas is not None and pasada not in selected_pasadas:
                    continue
                    
                print(f"  📁 Processing pasada: {pasada} ({len(track_files)} tracks)")
                
                try:
                    # Encontrar archivo de patrón
                    pattern_file = find_pattern_file(pasada, preprocessed_dir)
                    
                    # Asignar tarea para cada track de esta pasada
                    for track_file in track_files:
                        future = executor.submit(process_single_track_parallel, (track_file, pattern_file, filter_name, pasada))
                        future_to_track[future] = {
                            'track_file': track_file,
                            'filter_name': filter_name,
                            'pasada': pasada
                        }
                
                except Exception as e:
                    print(f"❌ Error processing pasada {pasada}: {e}")
                    continue
        
        print(f"\n⏳ Processing {len(future_to_track)} tasks in parallel...")
        start_time = time.time()
        last_progress_time = start_time
        
        # Recolectar resultados conforme van completándose
        for future in as_completed(future_to_track):
            track_info = future_to_track[future]
            track_file = track_info['track_file']
            
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                
                completed_tasks += 1
                
                # Mostrar progreso cada 50 tareas o cada 30 segundos
                current_time = time.time()
                elapsed = current_time - start_time
                time_since_last_update = current_time - last_progress_time
                
                if completed_tasks % 50 == 0 or time_since_last_update > 30 or completed_tasks == total_tasks:
                    percent_done = (completed_tasks / total_tasks) * 100
                    rate = completed_tasks / elapsed if elapsed > 0 else 0
                    eta_seconds = (total_tasks - completed_tasks) / rate if rate > 0 else 0
                    
                    # Formatear ETA de manera legible
                    if eta_seconds < 60:
                        eta_str = f"{int(eta_seconds)}s"
                    elif eta_seconds < 3600:
                        eta_str = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
                    else:
                        eta_str = f"{eta_seconds/3600:.1f}h"
                    
                    # Barra de progreso simple
                    bar_width = 30
                    filled = int(bar_width * percent_done / 100)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    
                    print(f"📈 Progress: [{bar}] {completed_tasks}/{total_tasks} ({percent_done:.1f}%) | "
                          f"Rate: {rate:.1f} tracks/s | ETA: {eta_str}")
                    
                    last_progress_time = current_time
                    
            except Exception as e:
                print(f"❌ Error processing {os.path.basename(track_file)}: {e}")
                completed_tasks += 1
                continue
    
    final_count = len(results)
    total_elapsed = time.time() - start_time
    
    print(f"\n✅ Processing completed!")
    print(f"📊 Results: {final_count}/{total_tasks} tracks processed successfully")
    print(f"⏱️  Total time: {total_elapsed/60:.1f} minutes")
    print(f"🚀 Average rate: {final_count/total_elapsed:.1f} tracks/second")
    
    return results

def create_excel_report(results, output_file):
    """
    Crea un reporte en Excel con todos los resultados de comparación.
    
    Args:
        results: Lista de resultados de comparación
        output_file: Ruta del archivo Excel de salida
    """
    print(f"\nCreating Excel report: {output_file}")
    
    # Crear DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("Warning: No results to save")
        return
    
    # Reorganizar columnas incluyendo información de recorte
    column_order = [
        'pasada', 'track_name', 'filter_name', 'total_points',
        # Información de recorte temporal
        'is_trimmed', 'pattern_coverage_percent', 'points_lost_start', 'points_lost_end',
        # Métricas del patrón (pueden estar afectadas por recorte)
        'total_pattern_length', 'total_pattern_elevation_gain', 'total_pattern_elevation_loss',
        'total_pattern_elevation_gain_threshold', 'total_pattern_elevation_loss_threshold',
        # Desviaciones respecto al patrón
        'total_length_deviation', 'total_elevation_gain_deviation', 'total_elevation_loss_deviation',
        'total_elevation_gain_deviation_threshold', 'total_elevation_loss_deviation_threshold',
        'mean_point_deviation', 'std_point_deviation',
        # Rangos temporales para referencia
        'track_time_range', 'pattern_time_range'
    ]
    
    # Reordenar columnas
    df = df[column_order]
    
    # Crear workbook
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Escribir datos principales
        df.to_excel(writer, sheet_name='Track_Comparison', index=False)
        
        # Crear hoja de resumen por filtro
        summary_data = []
        for filter_name in df['filter_name'].unique():
            filter_df = df[df['filter_name'] == filter_name]
            
            summary_row = {
                'filter_name': filter_name,
                'total_tracks': len(filter_df),
                'tracks_with_trimming': filter_df['is_trimmed'].sum(),
                'avg_pattern_coverage': filter_df['pattern_coverage_percent'].mean(),
                'min_pattern_coverage': filter_df['pattern_coverage_percent'].min(),
                'mean_length_deviation': filter_df['total_length_deviation'].mean(),
                'std_length_deviation': filter_df['total_length_deviation'].std(),
                'mean_elevation_gain_deviation': filter_df['total_elevation_gain_deviation'].mean(),
                'std_elevation_gain_deviation': filter_df['total_elevation_gain_deviation'].std(),
                'mean_elevation_loss_deviation': filter_df['total_elevation_loss_deviation'].mean(),
                'std_elevation_loss_deviation': filter_df['total_elevation_loss_deviation'].std(),
                'mean_point_deviation_avg': filter_df['mean_point_deviation'].mean(),
                'std_point_deviation_avg': filter_df['std_point_deviation'].mean()
            }
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Filter_Summary', index=False)
        
        # Crear hoja de análisis de recortes
        trim_analysis = df[df['is_trimmed'] == True].copy() if 'is_trimmed' in df.columns else pd.DataFrame()
        if not trim_analysis.empty:
            trim_summary = trim_analysis.groupby(['pasada', 'filter_name']).agg({
                'pattern_coverage_percent': ['mean', 'min', 'max'],
                'points_lost_start': 'mean',
                'points_lost_end': 'mean',
                'track_name': 'count'
            }).round(2)
            
            # Aplanar nombres de columnas
            trim_summary.columns = ['_'.join(col).strip() for col in trim_summary.columns.values]
            trim_summary = trim_summary.reset_index()
            trim_summary.rename(columns={'track_name_count': 'affected_tracks'}, inplace=True)
            
            trim_summary.to_excel(writer, sheet_name='Trimming_Analysis', index=False)
        
        # Obtener workbook y worksheets para formateo
        workbook = writer.book
        worksheet1 = writer.sheets['Track_Comparison']
        worksheet2 = writer.sheets['Filter_Summary']
        
        # Formato para números
        number_format = workbook.add_format({'num_format': '0.00'})
        percent_format = workbook.add_format({'num_format': '0.0%'})
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC'})
        warning_format = workbook.add_format({'bg_color': '#FFE6E6'})  # Fondo rosa claro para recortes
        
        # Aplicar formato a headers
        for col_num, value in enumerate(df.columns.values):
            worksheet1.write(0, col_num, value, header_format)
        
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet2.write(0, col_num, value, header_format)
        
        # Formateo condicional para indicar recortes
        # Resaltar filas donde is_trimmed = True
        if 'is_trimmed' in df.columns:
            trimmed_rows = df.index[df['is_trimmed'] == True].tolist()
            for row_num in trimmed_rows:
                # row_num + 1 porque Excel es 1-indexed y tenemos header
                worksheet1.set_row(row_num + 1, None, warning_format)
        
        # Formatear columna de porcentaje de cobertura
        if 'pattern_coverage_percent' in df.columns:
            coverage_col = df.columns.get_loc('pattern_coverage_percent')
            for row in range(len(df)):
                coverage_value = df.iloc[row]['pattern_coverage_percent'] / 100.0  # Convertir a decimal
                worksheet1.write(row + 1, coverage_col, coverage_value, percent_format)
        
        # Ajustar ancho de columnas
        worksheet1.set_column('A:V', 18)  # Más columnas ahora
        worksheet2.set_column('A:M', 20)
        
        # Añadir hoja de explicación
        if 'Trimming_Analysis' in writer.sheets:
            worksheet3 = writer.sheets['Trimming_Analysis']
            worksheet3.set_column('A:H', 18)
    
    print(f"Excel report saved: {output_file}")
    print(f"  Total comparisons: {len(results)}")
    print(f"  Filters analyzed: {df['filter_name'].nunique()}")
    print(f"  Pasadas processed: {df['pasada'].nunique()}")
    
    # Estadísticas de recortes
    if 'is_trimmed' in df.columns:
        trimmed_count = df['is_trimmed'].sum()
        total_count = len(df)
        print(f"  Tracks with temporal trimming: {trimmed_count}/{total_count} ({trimmed_count/total_count*100:.1f}%)")
        if trimmed_count > 0:
            min_coverage = df[df['is_trimmed']]['pattern_coverage_percent'].min()
            avg_coverage = df[df['is_trimmed']]['pattern_coverage_percent'].mean()
            print(f"  Pattern coverage range: {min_coverage:.1f}% - 100.0% (avg: {avg_coverage:.1f}%)")
    
    print(f"\nExcel sheets created:")
    print(f"  - Track_Comparison: Detailed results with trimming indicators")
    print(f"  - Filter_Summary: Aggregated statistics by filter")
    if 'is_trimmed' in df.columns and df['is_trimmed'].any():
        print(f"  - Trimming_Analysis: Analysis of temporal trimming effects")

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Compare filtered tracks with reference patterns')
    parser.add_argument('--pasadas', type=str, help='Comma-separated list of passes to process (e.g., "1,2,3")')
    parser.add_argument('--output', default='track_comparison_results.xlsx', help='Output Excel file (default: track_comparison_results.xlsx)')
    parser.add_argument('--filtered-dir', default='data/filtered', help='Directory with filtered tracks')
    parser.add_argument('--preprocessed-dir', default='data/preprocessed', help='Directory with preprocessed tracks and patterns')
    
    args = parser.parse_args()
    
    try:
        print("=== Track Comparison Analysis ===")
        print(f"Filtered tracks directory: {args.filtered_dir}")
        print(f"Preprocessed directory: {args.preprocessed_dir}")
        print(f"Output file: {args.output}")
        print()
        
        # Verificar directorios
        if not os.path.exists(args.filtered_dir):
            print(f"ERROR: Filtered tracks directory not found: {args.filtered_dir}")
            sys.exit(1)
            
        if not os.path.exists(args.preprocessed_dir):
            print(f"ERROR: Preprocessed directory not found: {args.preprocessed_dir}")
            sys.exit(1)
        
        # Parsear pasadas seleccionadas
        selected_pasadas = None
        if args.pasadas:
            selected_pasadas = [p.strip() for p in args.pasadas.split(',')]
            print(f"Processing only pasadas: {selected_pasadas}")
        
        # Encontrar todos los tracks filtrados
        print("Finding filtered tracks...")
        filtered_tracks = find_all_filtered_tracks(args.filtered_dir)
        
        if not filtered_tracks:
            print("ERROR: No filtered tracks found")
            sys.exit(1)
        
        total_tracks = sum(len(tracks) for pasadas in filtered_tracks.values() for tracks in pasadas.values())
        print(f"Found {len(filtered_tracks)} filters with {total_tracks} total tracks")
        
        for filter_name, pasadas in filtered_tracks.items():
            filter_total = sum(len(tracks) for tracks in pasadas.values())
            print(f"  {filter_name}: {filter_total} tracks in {len(pasadas)} pasadas")
        
        # Comparar todos los tracks
        print("\nStarting track comparison...")
        results = compare_all_tracks_parallel(filtered_tracks, args.preprocessed_dir, selected_pasadas)
        
        if not results:
            print("ERROR: No comparison results generated")
            sys.exit(1)
        
        # Crear reporte Excel
        create_excel_report(results, args.output)
        
        print(f"\nSUCCESS: Track comparison completed!")
        print(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        print(f"\n\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()