#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detecta tramos con pendiente anómala en los tracks patrón *_pattern.gpx

Analiza ventanas deslizantes de 10 metros y detecta pendientes superiores al 35%
tanto en subida como en bajada.

Genera:
  - results/reports/slopes/pattern_slopes.csv  (tramos anómalos)
  - results/reports/slopes/gpx/<pattern>_slope_anomalies.gpx  (waypoints anómalos)
"""
import os
import glob
import math
import csv
import gpxpy
import gpxpy.gpx
from tqdm import tqdm

# ====================================================
# Configuración
# ====================================================
RAW_DIR = os.path.join("data", "raw")
REPORT_DIR = os.path.join("results", "reports", "slopes")
GPX_DIR = os.path.join(REPORT_DIR, "gpx")
REPORT_PATH = os.path.join(REPORT_DIR, "pattern_slopes.csv")

# Pendiente máxima tolerada (35% = 0.35)
SLOPE_THRESHOLD = 0.35
# Longitud de ventana para análisis de tramos (metros)
WINDOW_LENGTH = 10.0

R_EARTH = 6371000.0  # radio terrestre [m]

# ====================================================
# Utilidades
# ====================================================
def deg2rad(d): return d * math.pi / 180.0

def to_local_xy(lat0, lon0, lat, lon):
    """Proyección equirectangular local (m)."""
    lat0r = deg2rad(lat0)
    x = deg2rad(lon - lon0) * math.cos(lat0r) * R_EARTH
    y = deg2rad(lat - lat0) * R_EARTH
    return x, y

def read_gpx_points(path):
    """Devuelve lista de puntos con lat,lon,ele,time."""
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
        pts.append({
            "lat": p.latitude,
            "lon": p.longitude,
            "ele": p.elevation or 0.0,
            "time": p.time
        })
    return pts

def compute_distances_and_slopes(points):
    """Calcula distancias acumuladas y pendientes entre puntos consecutivos."""
    distances = [0.0] * len(points)
    slopes = [0.0] * len(points)
    
    if len(points) < 2:
        return distances, slopes
    
    lat0, lon0 = points[0]["lat"], points[0]["lon"]
    
    for i in range(1, len(points)):
        p0, p1 = points[i-1], points[i]
        x0, y0 = to_local_xy(lat0, lon0, p0["lat"], p0["lon"])
        x1, y1 = to_local_xy(lat0, lon0, p1["lat"], p1["lon"])
        
        dz = p1["ele"] - p0["ele"]
        dh = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        
        distances[i] = distances[i-1] + dh
        
        if dh < 1e-6:
            slopes[i] = 0.0
        else:
            slopes[i] = dz / dh
    
    return distances, slopes

def detect_anomalous_segments(points):
    """Detecta tramos con pendiente anómala en ventanas deslizantes de 10 metros."""
    anomalous_segments = []
    if len(points) < 2:
        return anomalous_segments

    distances, slopes = compute_distances_and_slopes(points)
    
    for i in range(len(points)):
        # Encontrar todos los puntos dentro de una ventana de 10 metros desde el punto i
        for j in range(i + 1, len(points)):
            if distances[j] - distances[i] > WINDOW_LENGTH:
                break
            
            # Verificar si algún tramo en esta ventana tiene pendiente anómala
            for k in range(i + 1, j + 1):
                if abs(slopes[k]) > SLOPE_THRESHOLD:
                    slope_percent = slopes[k] * 100.0
                    window_length = distances[j] - distances[i]
                    
                    anomalous_segments.append({
                        "start_index": i,
                        "end_index": j,
                        "anomaly_index": k,
                        "slope": slopes[k],
                        "slope_percent": slope_percent,
                        "window_length": window_length,
                        "start_point": points[i],
                        "end_point": points[j],
                        "anomaly_point": points[k]
                    })
                    break
    
    # Eliminar duplicados manteniendo solo uno por cada punto anómalo
    unique_segments = {}
    for seg in anomalous_segments:
        key = seg["anomaly_index"]
        if key not in unique_segments or abs(seg["slope"]) > abs(unique_segments[key]["slope"]):
            unique_segments[key] = seg
    
    return list(unique_segments.values())

# ====================================================
# Proceso principal
# ====================================================
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(GPX_DIR, exist_ok=True)

    rows = []
    pasadas = [d for d in sorted(os.listdir(RAW_DIR)) if os.path.isdir(os.path.join(RAW_DIR, d))]

    for pasada in tqdm(pasadas, desc="Analizando tramos de pendiente anómala"):
        pdir = os.path.join(RAW_DIR, pasada)
        pattern_files = glob.glob(os.path.join(pdir, "*_pattern.gpx"))
        if not pattern_files:
            continue

        for path in pattern_files:
            pts = read_gpx_points(path)
            if len(pts) < 2:
                continue

            pattern_name = os.path.splitext(os.path.basename(path))[0]
            gpx_out = gpxpy.gpx.GPX()
            wpt_added = 0

            # Detectar tramos anómalos
            anomalous_segments = detect_anomalous_segments(pts)
            
            for segment in anomalous_segments:
                p = segment["anomaly_point"]
                slope_percent = segment["slope_percent"]
                window_length = segment["window_length"]
                
                rows.append([
                    pasada,
                    pattern_name,
                    segment["anomaly_index"],
                    f"{slope_percent:.2f}",
                    p["lat"],
                    p["lon"],
                    p["ele"],
                    f"{window_length:.1f}"
                ])

                # Añadir waypoint al GPX
                wpt = gpxpy.gpx.GPXWaypoint(
                    latitude=p["lat"],
                    longitude=p["lon"],
                    elevation=p["ele"]
                )
                wpt.name = f"Tramo {segment['anomaly_index']}: {slope_percent:.1f}%"
                wpt.description = f"Pendiente {slope_percent:.1f}% en ventana de {window_length:.1f}m (umbral {SLOPE_THRESHOLD*100:.0f}%)"
                gpx_out.waypoints.append(wpt)
                wpt_added += 1

            if wpt_added > 0:
                gpx_path = os.path.join(GPX_DIR, f"{pattern_name}_slope_anomalies.gpx")
                with open(gpx_path, "w", encoding="utf-8") as f:
                    f.write(gpx_out.to_xml())

    # Guardar CSV
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pasada", "pattern_file", "point_index", "slope_percent", "lat", "lon", "ele", "window_length_m"])
        w.writerows(rows)

    print(f"\n✅ Informe generado: {REPORT_PATH}")
    print(f"✅ GPX de anomalías guardados en: {GPX_DIR}")
    print(f"Total de tramos anómalos detectados: {len(rows)}")
    print(f"Criterio: Pendientes > {SLOPE_THRESHOLD*100:.0f}% en ventanas de {WINDOW_LENGTH}m")

if __name__ == "__main__":
    main()
