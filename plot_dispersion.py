#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grafica la dispersión en las posiciones de las grabaciones proyectadas sobre el track patrón
a lo largo de la curvilínea de cada track patrón.

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
import gpxpy
import numpy as np
from collections import defaultdict

PRE_DIR = os.path.join("data", "preprocessed")
RAW_DIR = os.path.join("data", "raw")
PLOTS_DIR = os.path.join("data", "reports", "plots", "dispersion")

R_EARTH = 6371000.0

def deg2rad(d): 
    return d * math.pi / 180.0

def to_xy(lat0, lon0, lat, lon):
    """Convierte (lat, lon) a coordenadas planas (x, y) en metros."""
    lat0r = deg2rad(lat0)
    x = deg2rad(lon - lon0) * math.cos(lat0r) * R_EARTH
    y = deg2rad(lat - lat0) * R_EARTH
    return x, y

def proj_on_segment(px, py, ax, ay, bx, by, clamp=True):
    """Proyecta punto P sobre segmento AB."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx*vx + vy*vy
    if vv == 0.0:
        u = 0.0
        qx, qy = ax, ay
    else:
        u = (wx*vx + wy*vy) / vv
        if clamp:
            u = max(0.0, min(1.0, u))
        qx, qy = ax + u*vx, ay + u*vy
    dist = math.hypot(px - qx, py - qy)
    return u, qx, qy, dist

def cumdist(xs, ys):
    """Calcula distancias acumuladas en una polilínea."""
    seglen = []
    for i in range(len(xs) - 1):
        seglen.append(math.hypot(xs[i+1] - xs[i], ys[i+1] - ys[i]))
    s = [0.0]
    for L in seglen:
        s.append(s[-1] + L)
    return seglen, s

def read_gpx_points(path):
    """Lee puntos GPX con tiempo opcional."""
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
            # Convertir a datetime nativo de Python
            if hasattr(t, 'replace') and t.tzinfo is None:
                from datetime import timezone
                t = t.replace(tzinfo=timezone.utc)
            if hasattr(t, 'timestamp'):
                from datetime import datetime, timezone
                t = datetime.fromtimestamp(t.timestamp(), tz=timezone.utc)
        
        pts.append({
            "lat": p.latitude, 
            "lon": p.longitude, 
            "ele": p.elevation,
            "time": t
        })
    
    return pts

def build_pattern_geometry(pattern_pts):
    """Construye geometría del patrón en coordenadas planas."""
    if len(pattern_pts) < 2:
        return None, None, [], [], [], []
    
    lat0, lon0 = pattern_pts[0]["lat"], pattern_pts[0]["lon"]
    px, py = [], []
    
    for p in pattern_pts:
        x, y = to_xy(lat0, lon0, p["lat"], p["lon"])
        px.append(x)
        py.append(y)
    
    seglen, s_vertices = cumdist(px, py)
    return lat0, lon0, px, py, seglen, s_vertices

def project_track_to_pattern(track_pts, lat0, lon0, px, py, seglen, s_vertices):
    """
    Proyecta todos los puntos de un track sobre el patrón usando proyección progresiva.
    Mantiene una ventana deslizante para evitar que puntos de ida y vuelta se mezclen.
    Retorna listas de (distancia_curvilinea, distancia_proyeccion, tiempo).
    """
    if len(px) < 2:
        return [], [], []
    
    s_list = []
    d_list = []
    time_list = []
    nseg = len(seglen)
    
    # Parámetros para la ventana progresiva
    WINDOW_SIZE = 10  # Número de segmentos hacia atrás y adelante
    MAX_PROJ_DIST = 50.0  # Distancia máxima de proyección válida (metros)
    
    # Inicializar posición en el patrón
    current_seg = 0
    
    for idx, pt in enumerate(track_pts):
        # Convertir punto a coordenadas planas
        ptx, pty = to_xy(lat0, lon0, pt["lat"], pt["lon"])
        
        # Definir ventana de búsqueda alrededor de current_seg
        seg_start = max(0, current_seg - WINDOW_SIZE)
        seg_end = min(nseg - 1, current_seg + WINDOW_SIZE)
        
        # Buscar la mejor proyección en la ventana
        best_dist = float("inf")
        best_s = 0.0
        best_seg = current_seg
        
        for i in range(seg_start, seg_end + 1):
            ax, ay = px[i], py[i]
            bx, by = px[i+1], py[i+1]
            
            u, qx, qy, dist = proj_on_segment(ptx, pty, ax, ay, bx, by, clamp=True)
            
            if dist < best_dist:
                best_dist = dist
                best_s = s_vertices[i] + u * seglen[i]
                best_seg = i
        
        # Solo aceptar proyecciones que están dentro de la distancia máxima
        if best_dist <= MAX_PROJ_DIST:
            s_list.append(best_s)
            d_list.append(best_dist)
            time_list.append(pt["time"])
            
            # Actualizar posición actual para el siguiente punto
            # Permitir pequeños retrocesos pero favorecer avance
            if best_seg > current_seg - 2:  # No retroceder más de 2 segmentos
                current_seg = best_seg
        else:
            # Si la proyección está muy lejos, es probable que sea un punto atípico
            # o que hayamos perdido el track. Intentar avanzar la ventana.
            current_seg = min(current_seg + 1, nseg - 1)
    
    return s_list, d_list, time_list

def bin_dispersions(s_list, d_list, s_vertices, bin_size=10.0):
    """
    Agrupa las distancias de proyección en bins a lo largo de la curvilínea.
    
    Args:
        s_list: distancias curvilíneas de proyecciones
        d_list: distancias de proyección correspondientes
        s_vertices: distancias curvilíneas de vértices del patrón
        bin_size: tamaño de bin en metros
    
    Returns:
        bin_centers: centros de bins
        bin_means: medias de distancias en cada bin
        bin_stds: desviaciones estándar en cada bin
        bin_counts: número de puntos en cada bin
    """
    if not s_list:
        return [], [], [], []
    
    s_max = max(max(s_list), max(s_vertices))
    n_bins = int(math.ceil(s_max / bin_size))
    
    # Inicializar bins
    bins = defaultdict(list)
    
    # Asignar puntos a bins
    for s, d in zip(s_list, d_list):
        bin_idx = int(s / bin_size)
        bins[bin_idx].append(d)
    
    # Calcular estadísticas por bin
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(n_bins):
        if i in bins and bins[i]:
            distances = bins[i]
            bin_centers.append((i + 0.5) * bin_size)
            bin_means.append(np.mean(distances))
            bin_stds.append(np.std(distances))
            bin_counts.append(len(distances))
    
    return bin_centers, bin_means, bin_stds, bin_counts

def bin_dispersions_with_time(s_list, d_list, time_list, pattern_length, bin_size=20.0):
    """
    Agrupa las distancias de proyección y tiempos en bins a lo largo de la curvilínea.
    Los bins se definen basándose en la longitud total del patrón para garantizar
    consistencia entre todas las grabaciones de una pasada.
    
    Args:
        s_list: distancias curvilíneas de proyecciones
        d_list: distancias de proyección correspondientes
        time_list: tiempos correspondientes
        pattern_length: longitud total del track patrón (metros)
        bin_size: tamaño de bin en metros
    
    Returns:
        bin_centers: centros de bins (basados en el patrón completo)
        bin_means: medias de distancias en cada bin
        bin_stds: desviaciones estándar espaciales en cada bin
        bin_time_means: medias de tiempos en cada bin
        bin_time_stds: desviaciones estándar temporales en cada bin (segundos)
        bin_counts: número de puntos en cada bin
    """
    if not s_list or not time_list:
        return [], [], [], [], [], []
    
    # Calcular bins basándose en la longitud TOTAL del patrón
    n_bins = int(math.ceil(pattern_length / bin_size))
    
    # Inicializar bins
    bins_dist = defaultdict(list)
    bins_time = defaultdict(list)
    
    # Asignar puntos a bins
    for s, d, t in zip(s_list, d_list, time_list):
        if t is not None:  # Solo procesar puntos con tiempo válido
            bin_idx = int(s / bin_size)
            # Asegurar que el bin está dentro del rango válido
            if 0 <= bin_idx < n_bins:
                bins_dist[bin_idx].append(d)
                bins_time[bin_idx].append(t.timestamp())  # Convertir a timestamp para cálculos
    
    # Calcular estadísticas para TODOS los bins del patrón (incluso vacíos)
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_time_means = []
    bin_time_stds = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_center = (i + 0.5) * bin_size
        
        if i in bins_dist and bins_dist[i] and i in bins_time and bins_time[i]:
            # Bin con datos
            distances = bins_dist[i]
            timestamps = bins_time[i]
            
            bin_centers.append(bin_center)
            bin_means.append(np.mean(distances))
            bin_stds.append(np.std(distances))
            bin_time_means.append(np.mean(timestamps))
            bin_time_stds.append(np.std(timestamps))  # Desviación en segundos
            bin_counts.append(len(distances))
        else:
            # Bin sin datos - agregar valores NaN para mantener consistencia
            bin_centers.append(bin_center)
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
            bin_time_means.append(np.nan)
            bin_time_stds.append(np.nan)
            bin_counts.append(0)
    
    return bin_centers, bin_means, bin_stds, bin_time_means, bin_time_stds, bin_counts

def find_closest_points_to_pattern_point(track_pts, pattern_s, lat0, lon0, px, py, seglen, s_vertices, max_distance=30.0):
    """
    Encuentra los puntos de una grabación más cercanos a un punto específico del patrón.
    Usa una ventana deslizante temporal para evitar mezclar puntos de ida y vuelta.
    
    Args:
        track_pts: puntos de la grabación
        pattern_s: distancia curvilínea del punto patrón objetivo (metros)
        lat0, lon0: origen de coordenadas
        px, py: coordenadas del patrón
        seglen, s_vertices: geometría del patrón
        max_distance: distancia máxima para considerar un punto válido (metros)
    
    Returns:
        closest_points: lista de puntos cercanos con distancias y tiempos
    """
    if len(px) < 2 or len(track_pts) < 2:
        return []
    
    # Encontrar el punto del patrón en la distancia curvilínea especificada
    pattern_point = interpolate_pattern_point(pattern_s, px, py, s_vertices, seglen)
    if pattern_point is None:
        return []
    
    pattern_x, pattern_y = pattern_point
    
    # Primero, proyectar todos los puntos de la grabación sobre el patrón
    # para encontrar la ventana temporal apropiada
    nseg = len(seglen)
    WINDOW_SIZE = 15  # Número de segmentos hacia atrás y adelante
    MAX_PROJ_DIST = 50.0  # Distancia máxima de proyección válida (metros)
    
    # Encontrar puntos de la grabación que se proyectan cerca del punto patrón objetivo
    candidate_indices = []
    current_seg = 0
    
    for idx, pt in enumerate(track_pts):
        if pt["time"] is None:
            continue
            
        # Convertir punto a coordenadas planas
        ptx, pty = to_xy(lat0, lon0, pt["lat"], pt["lon"])
        
        # Definir ventana de búsqueda alrededor de current_seg
        seg_start = max(0, current_seg - WINDOW_SIZE)
        seg_end = min(nseg - 1, current_seg + WINDOW_SIZE)
        
        # Buscar la mejor proyección en la ventana
        best_dist = float("inf")
        best_s = 0.0
        best_seg = current_seg
        
        for i in range(seg_start, seg_end + 1):
            ax, ay = px[i], py[i]
            bx, by = px[i+1], py[i+1]
            
            u, qx, qy, dist = proj_on_segment(ptx, pty, ax, ay, bx, by, clamp=True)
            
            if dist < best_dist:
                best_dist = dist
                best_s = s_vertices[i] + u * seglen[i]
                best_seg = i
        
        # Solo considerar puntos que se proyectan cerca del punto patrón objetivo
        if best_dist <= MAX_PROJ_DIST and abs(best_s - pattern_s) <= 50.0:  # 50m de tolerancia en curvilínea
            candidate_indices.append(idx)
            
            # Actualizar posición actual para el siguiente punto
            if best_seg > current_seg - 2:
                current_seg = best_seg
        else:
            # Avanzar la ventana si perdemos el track
            if best_dist > MAX_PROJ_DIST:
                current_seg = min(current_seg + 1, nseg - 1)
    
    # Ahora, de los candidatos, seleccionar los que están cerca espacialmente del punto patrón
    closest_points = []
    
    for idx in candidate_indices:
        pt = track_pts[idx]
        ptx, pty = to_xy(lat0, lon0, pt["lat"], pt["lon"])
        
        # Calcular distancia euclidiana al punto patrón
        distance = math.hypot(ptx - pattern_x, pty - pattern_y)
        
        if distance <= max_distance:
            closest_points.append({
                "distance": distance,
                "time": pt["time"],
                "lat": pt["lat"],
                "lon": pt["lon"]
            })
    
    return closest_points

def interpolate_pattern_point(target_s, px, py, s_vertices, seglen):
    """
    Interpola un punto en el patrón en la distancia curvilínea especificada.
    
    Args:
        target_s: distancia curvilínea objetivo
        px, py: coordenadas del patrón
        s_vertices: distancias curvilíneas de vértices
        seglen: longitudes de segmentos
    
    Returns:
        (x, y) del punto interpolado o None si está fuera del rango
    """
    if target_s < 0 or target_s > s_vertices[-1]:
        return None
    
    # Encontrar el segmento que contiene target_s
    for i in range(len(seglen)):
        if s_vertices[i] <= target_s <= s_vertices[i+1]:
            # Interpolar linealmente en el segmento
            seg_start_s = s_vertices[i]
            seg_length = seglen[i]
            
            if seg_length == 0:
                return (px[i], py[i])
            
            # Parámetro de interpolación (0 a 1)
            t = (target_s - seg_start_s) / seg_length
            
            # Interpolación lineal
            x = px[i] + t * (px[i+1] - px[i])
            y = py[i] + t * (py[i+1] - py[i])
            
            return (x, y)
    
    return None

def remove_outliers(distances, times, method="iqr", factor=1.5):
    """
    Elimina outliers de las distancias usando el método IQR.
    
    Args:
        distances: lista de distancias
        times: lista de tiempos correspondientes
        method: método para detectar outliers ("iqr" o "zscore")
        factor: factor para el rango IQR
    
    Returns:
        distances_clean, times_clean: listas sin outliers
    """
    if len(distances) < 4:  # Necesitamos al menos 4 puntos para estadísticas robustas
        return distances, times
    
    distances_array = np.array(distances)
    
    if method == "iqr":
        q1 = np.percentile(distances_array, 25)
        q3 = np.percentile(distances_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        valid_indices = (distances_array >= lower_bound) & (distances_array <= upper_bound)
    else:  # zscore
        mean_dist = np.mean(distances_array)
        std_dist = np.std(distances_array)
        z_scores = np.abs((distances_array - mean_dist) / std_dist)
        valid_indices = z_scores < factor
    
    distances_clean = [distances[i] for i in range(len(distances)) if valid_indices[i]]
    times_clean = [times[i] for i in range(len(times)) if valid_indices[i]]
    
    return distances_clean, times_clean

def calculate_pattern_point_statistics(pattern_s, recording_files, lat0, lon0, px, py, seglen, s_vertices):
    """
    Calcula estadísticas para un punto específico del patrón basándose en todas las grabaciones.
    
    Args:
        pattern_s: distancia curvilínea del punto patrón
        recording_files: lista de archivos de grabaciones
        lat0, lon0, px, py, seglen, s_vertices: geometría del patrón
    
    Returns:
        dict con estadísticas o None si no hay datos suficientes
    """
    all_distances = []
    all_times = []
    recording_stats = []
    
    for rec_file in recording_files:
        # Cargar grabación
        rec_pts = read_gpx_points(rec_file)
        if len(rec_pts) < 2:
            continue
        
        # Encontrar puntos cercanos al punto patrón
        closest_points = find_closest_points_to_pattern_point(
            rec_pts, pattern_s, lat0, lon0, px, py, seglen, s_vertices
        )
        
        if not closest_points:
            continue
        
        # Extraer distancias y tiempos
        distances = [cp["distance"] for cp in closest_points]
        times = [cp["time"].timestamp() for cp in closest_points]
        
        # Eliminar outliers
        distances_clean, times_clean = remove_outliers(distances, times)
        
        if len(distances_clean) >= 2:
            all_distances.extend(distances_clean)
            all_times.extend(times_clean)
            
            # Estadísticas por grabación
            recording_stats.append({
                "file": os.path.basename(rec_file),
                "n_points": len(distances_clean),
                "mean_distance": np.mean(distances_clean),
                "std_distance": np.std(distances_clean),
                "mean_time": np.mean(times_clean),
                "std_time": np.std(times_clean)
            })
    
    if len(all_distances) < 4:  # Necesitamos datos suficientes
        return None
    
    return {
        "pattern_s": pattern_s,
        "n_recordings": len(recording_stats),
        "total_points": len(all_distances),
        "mean_distance": np.mean(all_distances),
        "std_distance": np.std(all_distances),
        "median_distance": np.median(all_distances),
        "mean_time": np.mean(all_times),
        "std_time": np.std(all_times),
        "recording_stats": recording_stats
    }

def plot_pasada_dispersion(pasada_dir, pasada_name):
    """
    Grafica la dispersión de posiciones y temporal para una pasada usando puntos específicos del patrón cada 100m.
    """
    print(f"\n🔹 Iniciando análisis de dispersión para pasada: {pasada_name}")
    
    # Buscar grabaciones resampleadas en preprocessed
    gpx_files = sorted(glob.glob(os.path.join(pasada_dir, "*.gpx")))
    recording_files = [f for f in gpx_files if "_resampled" in os.path.basename(f)]
    if not recording_files:
        print(f"[{pasada_name}] ⚠️ No se encontraron grabaciones resampleadas")
        return
    
    # Buscar patrón original en data/raw/
    raw_pasada_dir = os.path.join(RAW_DIR, pasada_name)
    if not os.path.isdir(raw_pasada_dir):
        print(f"[{pasada_name}] ⚠️ No se encontró directorio raw: {raw_pasada_dir}")
        return
        
    raw_gpx_files = sorted(glob.glob(os.path.join(raw_pasada_dir, "*.gpx")))
    pattern_files = [f for f in raw_gpx_files if "_pattern" in os.path.basename(f).lower()]
    if not pattern_files:
        print(f"[{pasada_name}] ⚠️ No se encontró patrón original (*_pattern.gpx) en {raw_pasada_dir}")
        return
    
    pattern_file = pattern_files[0]
    print(f"[{pasada_name}] 📍 Procesando {len(recording_files)} grabaciones vs patrón original")
    
    # Cargar patrón original (sin tiempos)
    print(f"[{pasada_name}] 🔄 Cargando patrón: {os.path.basename(pattern_file)}")
    pattern_pts = read_gpx_points(pattern_file)
    if len(pattern_pts) < 2:
        print(f"[{pasada_name}] ❌ Patrón insuficiente")
        return
    
    # Construir geometría del patrón
    print(f"[{pasada_name}] 🔄 Construyendo geometría del patrón ({len(pattern_pts)} puntos)")
    lat0, lon0, px, py, seglen, s_vertices = build_pattern_geometry(pattern_pts)
    pattern_length = s_vertices[-1]
    print(f"[{pasada_name}] 📏 Longitud del patrón: {pattern_length:.1f} metros")
    
    # Generar puntos del patrón cada 100 metros
    point_interval = 100.0  # metros
    pattern_points = []
    current_s = 0.0
    
    while current_s <= pattern_length:
        pattern_points.append(current_s)
        current_s += point_interval
    
    print(f"[{pasada_name}] 📊 Analizando {len(pattern_points)} puntos del patrón (cada {point_interval}m)")
    
    # Calcular estadísticas para cada punto del patrón
    pattern_distances = []
    mean_distances = []
    std_distances = []
    std_times = []
    point_counts = []
    
    for i, pattern_s in enumerate(pattern_points):
        print(f"[{pasada_name}] 🔄 Punto {i+1}/{len(pattern_points)}: {pattern_s:.0f}m", end="")
        
        stats = calculate_pattern_point_statistics(
            pattern_s, recording_files, lat0, lon0, px, py, seglen, s_vertices
        )
        
        if stats is not None:
            pattern_distances.append(pattern_s)
            mean_distances.append(stats["mean_distance"])
            std_distances.append(stats["std_distance"])
            std_times.append(stats["std_time"])
            point_counts.append(stats["total_points"])
            print(f" → {stats['n_recordings']} grab., {stats['total_points']} pts ✅")
        else:
            print(" ❌ sin datos")
    
    if not pattern_distances:
        print(f"[{pasada_name}] ❌ No hay datos suficientes para graficar")
        return
    
    print(f"[{pasada_name}] 🎨 Generando gráfico con {len(pattern_distances)} puntos válidos...")
    
    # Crear figura con tres subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15))
    
    # Subplot 1: Distancia media de proyección
    ax1.plot(pattern_distances, mean_distances, 'o-', color='blue', 
             linewidth=2, markersize=6, alpha=0.8, label='Distancia media')
    ax1.set_title(f'Distancia Media de Proyección - Pasada {pasada_name}', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Distancia a lo largo del patrón (m)', fontsize=10)
    ax1.set_ylabel('Distancia media de proyección (m)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, pattern_length)
    
    # Subplot 2: Desviación estándar espacial
    ax2.plot(pattern_distances, std_distances, 's-', color='red', 
             linewidth=2, markersize=6, alpha=0.8, label='Desviación estándar espacial')
    ax2.set_title(f'Desviación Estándar Espacial - Pasada {pasada_name}', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distancia a lo largo del patrón (m)', fontsize=10)
    ax2.set_ylabel('Desviación estándar espacial (m)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, pattern_length)
    
    # Subplot 3: Desviación estándar temporal
    ax3.plot(pattern_distances, std_times, '^-', color='green', 
             linewidth=2, markersize=6, alpha=0.8, label='Desviación estándar temporal')
    ax3.set_title(f'Desviación Estándar Temporal - Pasada {pasada_name}', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Distancia a lo largo del patrón (m)', fontsize=10)
    ax3.set_ylabel('Desviación estándar temporal (s)', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(0, pattern_length)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    os.makedirs(PLOTS_DIR, exist_ok=True)
    output_path = os.path.join(PLOTS_DIR, f"dispersion_pasada_{pasada_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Estadísticas resumen
    if mean_distances:
        overall_mean_distance = np.mean(mean_distances)
        overall_std_spatial = np.mean(std_distances)
        overall_std_temporal = np.mean(std_times)
        total_points_analyzed = sum(point_counts)
        
        print(f"[{pasada_name}] 📊 Puntos analizados: {len(pattern_distances)}/{len(pattern_points)}")
        print(f"[{pasada_name}] 📊 Total puntos de datos: {total_points_analyzed}")
        print(f"[{pasada_name}] 📊 Dispersión espacial media: {overall_mean_distance:.2f}m ± {overall_std_spatial:.2f}m")
        print(f"[{pasada_name}] ⏱ Dispersión temporal media: {overall_std_temporal:.2f}s")
        print(f"[{pasada_name}] 📊 Gráfico guardado: {output_path}")
    else:
        print(f"[{pasada_name}] ❌ No hay datos suficientes para graficar")

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
    print(f"🚀 Iniciando procesamiento de dispersión...")
    
    procesadas_ok = 0
    procesadas_error = 0
    
    for idx, pasada in enumerate(pasadas, 1):
        print(f"\n{'='*60}")
        print(f"🔄 PROCESANDO PASADA {idx}/{len(pasadas)}: {pasada}")
        print(f"{'='*60}")
        
        pasada_dir = os.path.join(PRE_DIR, pasada)
        try:
            plot_pasada_dispersion(pasada_dir, pasada)
            procesadas_ok += 1
            print(f"✅ Pasada {pasada} completada ({procesadas_ok}/{len(pasadas)})")
        except Exception as e:
            procesadas_error += 1
            print(f"❌ Error en pasada {pasada}: {str(e)}")
            print(f"⚠️ Continuando con la siguiente pasada...")
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN FINAL:")
    print(f"✅ Pasadas procesadas correctamente: {procesadas_ok}")
    print(f"❌ Pasadas con errores: {procesadas_error}")
    print(f"📁 Total procesadas: {procesadas_ok + procesadas_error}/{len(pasadas)}")
    print(f"🎯 Gráficos generados en: {PLOTS_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()