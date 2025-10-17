#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detecta puntos con pendiente anómala en los tracks patrón *_pattern.gpx

Genera:
  - data/reports/slopes/pattern_slopes.csv  (pendientes anómalas)
  - data/reports/slopes/gpx/<pattern>_slope_anomalies.gpx  (waypoints anómalos)
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
REPORT_DIR = os.path.join("data", "reports", "slopes")
GPX_DIR = os.path.join(REPORT_DIR, "gpx")
REPORT_PATH = os.path.join(REPORT_DIR, "pattern_slopes.csv")

# Pendiente máxima tolerada (20% = 0.2)
SLOPE_THRESHOLD = 0.2

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

def compute_slopes(points):
    """Calcula pendientes entre puntos consecutivos (en fracción)."""
    slopes = [0.0] * len(points)
    if len(points) < 2:
        return slopes
    lat0, lon0 = points[0]["lat"], points[0]["lon"]
    for i in range(1, len(points)):
        p0, p1 = points[i-1], points[i]
        x0, y0 = to_local_xy(lat0, lon0, p0["lat"], p0["lon"])
        x1, y1 = to_local_xy(lat0, lon0, p1["lat"], p1["lon"])
        dz = p1["ele"] - p0["ele"]
        dh = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        if dh < 1e-6:
            slope = 0.0
        else:
            slope = dz / dh
        slopes[i] = slope
    return slopes

# ====================================================
# Proceso principal
# ====================================================
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(GPX_DIR, exist_ok=True)

    rows = []
    pasadas = [d for d in sorted(os.listdir(RAW_DIR)) if os.path.isdir(os.path.join(RAW_DIR, d))]

    for pasada in tqdm(pasadas, desc="Analizando pendientes"):
        pdir = os.path.join(RAW_DIR, pasada)
        pattern_files = glob.glob(os.path.join(pdir, "*_pattern.gpx"))
        if not pattern_files:
            continue

        for path in pattern_files:
            pts = read_gpx_points(path)
            if len(pts) < 2:
                continue

            slopes = compute_slopes(pts)
            pattern_name = os.path.splitext(os.path.basename(path))[0]
            gpx_out = gpxpy.gpx.GPX()
            wpt_added = 0

            for i, slope in enumerate(slopes):
                if abs(slope) > SLOPE_THRESHOLD:
                    slope_percent = slope * 100.0
                    p = pts[i]
                    rows.append([
                        pasada,
                        pattern_name,
                        i,
                        f"{slope_percent:.2f}",
                        p["lat"],
                        p["lon"],
                        p["ele"]
                    ])

                    # Añadir waypoint al GPX
                    wpt = gpxpy.gpx.GPXWaypoint(
                        latitude=p["lat"],
                        longitude=p["lon"],
                        elevation=p["ele"]
                    )
                    wpt.name = f"{i}: {slope_percent:.1f}%"
                    wpt.description = f"Pendiente {slope_percent:.1f}% (umbral {SLOPE_THRESHOLD*100:.0f}%)"
                    gpx_out.waypoints.append(wpt)
                    wpt_added += 1

            if wpt_added > 0:
                gpx_path = os.path.join(GPX_DIR, f"{pattern_name}_slope_anomalies.gpx")
                with open(gpx_path, "w", encoding="utf-8") as f:
                    f.write(gpx_out.to_xml())

    # Guardar CSV
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pasada", "pattern_file", "point_index", "slope_percent", "lat", "lon", "ele"])
        w.writerows(rows)

    print(f"\n✅ Informe generado: {REPORT_PATH}")
    print(f"✅ GPX de anomalías guardados en: {GPX_DIR}")
    print(f"Total de puntos anómalos detectados: {len(rows)}")

if __name__ == "__main__":
    main()
