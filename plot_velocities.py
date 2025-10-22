#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grafica la velocidad a lo largo del tiempo de todos los tracks de cada pasada.
Estructura esperada:
data/
    preprocessed/
        <pasada>/
            <grabacion1>_resampled.gpx
            <grabacion2>_resampled.gpx
            ...
            <n>_pattern_aligned.gpx
"""

import os
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime, timezone
import gpxpy
import numpy as np

PRE_DIR = os.path.join("data", "preprocessed")
PLOTS_DIR = os.path.join("data", "reports", "plots", "velocities")

R_EARTH = 6371000.0

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia haversine entre dos puntos en metros.
    """
    # Convertir a radianes
    lat1_r = math.radians(lat1)
    lon1_r = math.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    
    # Diferencias
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    
    # Fórmula de haversine
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R_EARTH * c

def read_gpx_points(path):
    """Lee puntos GPX con tiempo."""
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    
    pts = []
    if not gpx.tracks:
        return pts
    
    trk = gpx.tracks[0]
    if not trk.segments:
        return pts
    
    seg = trk.segments[0]
    for p in seg.points:
        t = p.time
        if t is not None:
            # Convertir a datetime nativo de Python para evitar conflictos con matplotlib
            if hasattr(t, 'replace') and t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            # Asegurar que sea un datetime nativo
            if hasattr(t, 'timestamp'):
                t = datetime.fromtimestamp(t.timestamp(), tz=timezone.utc)
            pts.append({
                "lat": p.latitude, 
                "lon": p.longitude, 
                "ele": p.elevation, 
                "time": t
            })
    
    return pts

def calculate_velocities(points):
    """
    Calcula velocidades instantáneas entre puntos consecutivos.
    Retorna listas de tiempos y velocidades en m/s.
    """
    if len(points) < 2:
        return [], []
    
    times = []
    velocities = []
    
    for i in range(1, len(points)):
        p1 = points[i-1]
        p2 = points[i]
        
        # Verificar que ambos puntos tengan tiempo válido
        if p1["time"] is None or p2["time"] is None:
            continue
            
        # Distancia entre puntos
        dist = haversine_distance(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        
        # Tiempo transcurrido
        dt = (p2["time"] - p1["time"]).total_seconds()
        
        if dt > 0:
            velocity = dist / dt  # m/s
            # Usar el tiempo del punto final para esta velocidad
            times.append(p2["time"])
            velocities.append(velocity)
    
    return times, velocities

def plot_pasada_velocities(pasada_dir, pasada_name):
    """
    Grafica las velocidades de todos los tracks de una pasada.
    """
    # Buscar todos los archivos GPX procesados
    gpx_files = sorted(glob.glob(os.path.join(pasada_dir, "*.gpx")))
    
    if not gpx_files:
        print(f"[{pasada_name}] ⚠️ No se encontraron archivos GPX")
        return
    
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(gpx_files)))
    
    track_count = 0
    all_times = []  # Para determinar el rango temporal global
    
    for gpx_file, color in zip(gpx_files, colors):
        filename = os.path.basename(gpx_file)
        
        # Leer puntos
        points = read_gpx_points(gpx_file)
        if len(points) < 2:
            print(f"[{pasada_name}] ⚠️ {filename}: insuficientes puntos con tiempo")
            continue
        
        # Calcular velocidades
        times, velocities = calculate_velocities(points)
        if not times:
            continue
        
        # Convertir a km/h para mejor visualización
        velocities_kmh = [v * 3.6 for v in velocities]
        
        # Determinar estilo de línea
        if "_pattern_aligned" in filename:
            label = f"Patrón ({filename})"
            linestyle = '-'
            linewidth = 2.5
            alpha = 0.9
        elif "_resampled" in filename:
            label = f"Grabación ({filename.replace('_resampled.gpx', '')})"
            linestyle = '-'
            linewidth = 1.0
            alpha = 0.7
        else:
            label = filename
            linestyle = '-'
            linewidth = 1.0
            alpha = 0.7
        
        plt.plot(times, velocities_kmh, 
                color=color, 
                label=label,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha)
        
        all_times.extend(times)
        track_count += 1
    
    if track_count == 0:
        print(f"[{pasada_name}] ❌ No hay tracks válidos para graficar")
        plt.close()
        return
    
    # Configurar el gráfico
    plt.title(f'Velocidades - Pasada {pasada_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Velocidad (km/h)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Formatear eje de tiempo de manera más robusta
    ax = plt.gca()
    
    if all_times:
        # Usar MaxNLocator para controlar estrictamente el número de ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, prune='both'))
        
        # Determinar formato basado en la duración
        time_range = (max(all_times) - min(all_times)).total_seconds()
        
        if time_range > 3600:  # Más de 1 hora
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif time_range > 600:  # Más de 10 minutos
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%M:%S'))
    
    plt.xticks(rotation=45)
    
    # Leyenda
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, f"velocities_pasada_{pasada_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[{pasada_name}] 📊 Gráfico de velocidades guardado: {output_path}")
    if all_times:
        time_range = (max(all_times) - min(all_times)).total_seconds()
        duration_mins = time_range / 60
        print(f"[{pasada_name}] ⏱ Duración del recorrido: {duration_mins:.1f} minutos")

def main():
    """Procesa todas las pasadas encontradas."""
    if not os.path.isdir(PRE_DIR):
        print(f"❌ No existe {PRE_DIR}")
        print("   Ejecuta primero resample_recordings.py y align_patterns_times.py")
        return
    
    # Encontrar todas las pasadas
    pasadas = [d for d in sorted(os.listdir(PRE_DIR)) 
               if os.path.isdir(os.path.join(PRE_DIR, d))]
    
    if not pasadas:
        print(f"❌ No se encontraron pasadas en {PRE_DIR}")
        return
    
    print(f"📂 Encontradas {len(pasadas)} pasadas en {PRE_DIR}")
    
    for pasada in pasadas:
        pasada_dir = os.path.join(PRE_DIR, pasada)
        plot_pasada_velocities(pasada_dir, pasada)
    
    print(f"\n✅ Gráficos de velocidades generados en {PLOTS_DIR}")

if __name__ == "__main__":
    main()