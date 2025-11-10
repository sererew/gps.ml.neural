#!/usr/bin/env python3
"""
Filtro de mediana móvil - aplica suavizado usando la mediana en ventana deslizante.
Especialmente efectivo para eliminar outliers y ruido impulsivo.

Uso:
    python 7_median_filter.py input_track.gpx [output_track.gpx]
    python 7_median_filter.py input_track.gpx --window 5
    
Si no se especifica output_track.gpx, se genera automáticamente como:
    <directorio_entrada>/<nombre_original>_median_filtered.gpx
"""

import numpy as np
import pandas as pd
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from scipy.ndimage import median_filter

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
    gpx.set("creator", "median_filter")
    gpx.set("xmlns", "http://www.topografix.com/GPX/1/1")
    
    trk = ET.SubElement(gpx, "trk")
    name = ET.SubElement(trk, "name")
    name.text = "Median Filtered Track"
    
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

def apply_median_filter(track_df, window_size=5):
    """
    Aplica filtro de mediana móvil a las coordenadas.
    
    Args:
        track_df: DataFrame con columnas lat, lon, ele, time
        window_size: Tamaño de la ventana para el filtro de mediana
        
    Returns:
        DataFrame con track filtrado
    """
    print(f"Applying median filter with window size {window_size}...")
    
    filtered_df = track_df.copy()
    
    # Aplicar filtro de mediana a cada coordenada
    filtered_df['lat'] = median_filter(track_df['lat'].values, size=window_size)
    filtered_df['lon'] = median_filter(track_df['lon'].values, size=window_size)
    filtered_df['ele'] = median_filter(track_df['ele'].values, size=window_size)
    
    return filtered_df

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Apply median filter to GPS track')
    parser.add_argument('input_gpx', help='Input GPX file (sampled at 1Hz)')
    parser.add_argument('output_gpx', nargs='?', help='Output filtered GPX file (optional)')
    parser.add_argument('--window', type=int, default=5, help='Window size for median filter (default: 5)')
    parser.add_argument('--suffix', default='median_filtered', help='Suffix for auto-generated output filename')
    
    args = parser.parse_args()
    
    try:
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
        
        # Aplicar filtro de mediana
        filtered_df = apply_median_filter(track_df, window_size=args.window)
        
        # Guardar track filtrado
        create_gpx(
            filtered_df['lat'], 
            filtered_df['lon'], 
            filtered_df['ele'],
            filtered_df['time'] if 'time' in filtered_df.columns else None,
            str(args.output_gpx)
        )
        
        print(f"SUCCESS: Median filtering completed successfully!")
        print(f"   Input: {len(track_df)} points")
        print(f"   Output: {len(filtered_df)} points")
        print(f"   Window size: {args.window}")
        print(f"   Filtered track saved to: {args.output_gpx}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()