#!/usr/bin/env python3
"""
Filtro Kalman simple - aplica filtrado óptimo basado en modelo de movimiento.
Especialmente efectivo para tracks GPS ya que modela la dinámica del movimiento.

Uso:
    python 7_kalman_filter.py input_track.gpx [output_track.gpx]
    python 7_kalman_filter.py input_track.gpx --process_noise 0.1 --measurement_noise 1.0
    
Si no se especifica output_track.gpx, se genera automáticamente como:
    <directorio_entrada>/<nombre_original>_kalman_filtered.gpx
"""

import numpy as np
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

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

def create_gpx(lat, lon, ele, time=None, output_path="filtered_track.gpx"):
    """
    Crea un archivo GPX con los puntos filtrados.
    """
    # Crear estructura GPX
    gpx = ET.Element("gpx")
    gpx.set("version", "1.1")
    gpx.set("creator", "kalman_filter")
    gpx.set("xmlns", "http://www.topografix.com/GPX/1/1")
    
    trk = ET.SubElement(gpx, "trk")
    name = ET.SubElement(trk, "name")
    name.text = "Kalman Filtered Track"
    
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

class SimpleKalmanFilter1D:
    """
    Filtro Kalman simple para una dimensión.
    Modelo de movimiento con velocidad constante.
    """
    
    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Estado: [posición, velocidad]
        self.x = np.array([0.0, 0.0])
        
        # Matriz de covarianza del estado
        self.P = np.eye(2) * 1000.0
        
        # Matriz de transición (modelo de velocidad constante)
        dt = 1.0  # 1 segundo (1Hz sampling)
        self.F = np.array([[1.0, dt],
                          [0.0, 1.0]])
        
        # Matriz de observación (solo observamos posición)
        self.H = np.array([[1.0, 0.0]])
        
        # Ruido del proceso
        self.Q = np.array([[dt**4/4, dt**3/2],
                          [dt**3/2, dt**2]]) * process_noise
        
        # Ruido de medición
        self.R = np.array([[measurement_noise]])
        
        self.initialized = False
    
    def update(self, measurement):
        """Actualiza el filtro con una nueva medición."""
        if not self.initialized:
            # Inicializar con la primera medición
            self.x[0] = measurement
            self.x[1] = 0.0
            self.initialized = True
            return measurement
        
        # Predicción
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Actualización
        y = measurement - self.H @ x_pred  # Residuo
        S = self.H @ P_pred @ self.H.T + self.R  # Covarianza del residuo
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Ganancia de Kalman
        
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred
        
        return self.x[0]  # Retornar posición filtrada

def apply_kalman_filter(track_df, process_noise=0.1, measurement_noise=1.0):
    """
    Aplica filtro Kalman a las coordenadas.
    
    Args:
        track_df: DataFrame con columnas lat, lon, ele, time
        process_noise: Ruido del modelo de proceso
        measurement_noise: Ruido de las mediciones
        
    Returns:
        DataFrame con track filtrado
    """
    print(f"Applying Kalman filter with process_noise={process_noise}, measurement_noise={measurement_noise}...")
    
    filtered_df = track_df.copy()
    
    # Crear filtros Kalman independientes para cada coordenada
    kf_lat = SimpleKalmanFilter1D(process_noise, measurement_noise)
    kf_lon = SimpleKalmanFilter1D(process_noise, measurement_noise)
    kf_ele = SimpleKalmanFilter1D(process_noise, measurement_noise)
    
    # Aplicar filtro a cada coordenada
    filtered_lat = []
    filtered_lon = []
    filtered_ele = []
    
    for i, row in track_df.iterrows():
        filtered_lat.append(kf_lat.update(row['lat']))
        filtered_lon.append(kf_lon.update(row['lon']))
        filtered_ele.append(kf_ele.update(row['ele']))
    
    filtered_df['lat'] = filtered_lat
    filtered_df['lon'] = filtered_lon
    filtered_df['ele'] = filtered_ele
    
    return filtered_df

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Apply Kalman filter to GPS track')
    parser.add_argument('input_gpx', help='Input GPX file (sampled at 1Hz)')
    parser.add_argument('output_gpx', nargs='?', help='Output filtered GPX file (optional)')
    parser.add_argument('--process_noise', type=float, default=0.1, help='Process noise parameter (default: 0.1)')
    parser.add_argument('--measurement_noise', type=float, default=1.0, help='Measurement noise parameter (default: 1.0)')
    parser.add_argument('--suffix', default='kalman_filtered', help='Suffix for auto-generated output filename')
    
    args = parser.parse_args()
    
    try:
        # Validar parámetros
        if args.process_noise <= 0 or args.measurement_noise <= 0:
            print("Error: Noise parameters must be positive")
            sys.exit(1)
        
        # Generar nombre de archivo de salida automáticamente si no se especifica
        if args.output_gpx is None:
            input_path = Path(args.input_gpx)
            output_dir = input_path.parent
            
            # Crear directorio de salida si no existe
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generar nombre con sufijo
            output_filename = f"{input_path.stem}_{args.suffix}{input_path.suffix}"
            args.output_gpx = output_dir / output_filename
            
            print(f"Output file auto-generated: {args.output_gpx}")
        else:
            # Crear directorio de salida si no existe
            output_path = Path(args.output_gpx)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {args.input_gpx}...")
        
        # Cargar track de entrada
        track_df = parse_gpx(args.input_gpx)
        
        # Aplicar filtro Kalman
        filtered_df = apply_kalman_filter(track_df, 
                                         process_noise=args.process_noise, 
                                         measurement_noise=args.measurement_noise)
        
        # Guardar track filtrado
        create_gpx(
            filtered_df['lat'], 
            filtered_df['lon'], 
            filtered_df['ele'],
            filtered_df['time'] if 'time' in filtered_df.columns else None,
            str(args.output_gpx)
        )
        
        print(f"SUCCESS: Kalman filtering completed successfully!")
        print(f"   Input: {len(track_df)} points")
        print(f"   Output: {len(filtered_df)} points")
        print(f"   Process noise: {args.process_noise}")
        print(f"   Measurement noise: {args.measurement_noise}")
        print(f"   Filtered track saved to: {args.output_gpx}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()