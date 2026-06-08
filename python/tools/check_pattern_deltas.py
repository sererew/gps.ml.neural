#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analiza tracks patrón *_pattern_aligned_resampled.gpx y detecta zonas anómalas.

Salida:
  - data/reports/pattern_delta_anomalies.csv  (incluye causas y métricas interpretables)
  - data/reports/plots/<pattern>.png          (dx,dy,dz, |v| con tramos en rojo)
  - data/reports/gpx/<pattern>_anomalies.gpx  (segmentos + waypoints con causas/metrics)
"""

import os
import glob
import math
import csv
import numpy as np
from datetime import timezone
import gpxpy
import gpxpy.gpx
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====================================================
# Configuración
# ====================================================
PRE_DIR = os.path.join("data", "preprocessed")
REPORT_DIR = os.path.join("data", "reports")
PLOTS_DIR  = os.path.join(REPORT_DIR, "plots")
GPX_DIR    = os.path.join(REPORT_DIR, "gpx")
REPORT_PATH = os.path.join(REPORT_DIR, "pattern_delta_anomalies.csv")

R_EARTH = 6371000.0

# Límites físicos / robustos (ajústalos a tu caso)
V_H_MAX   = 12.0   # m/s   (43.2 km/h)  velocidad horizontal máxima
V_TOT_MAX = 14.0   # m/s   (50.4 km/h)  velocidad 3D máxima
GRADE_UP_MAX   = 0.35   # pendiente máx. subiendo (~35%)
GRADE_DOWN_MAX = 0.60   # pendiente máx. bajando  (~60%)
JERK_MAX = 6.0          # m/s por segundo (1 Hz)
MIN_ZONE_LEN = 2        # nº mínimo de puntos consecutivos para zona anómala

# Filtro Hampel (para picos verticales)
HAMPEL_WIN = 15 	# ventana (puntos, aprox. 15 s)
HAMPEL_K   = 3.0	# umbral (3.0 es típico)

# ====================================================
# Utilidades
# ====================================================
def deg2rad(d): return d * math.pi / 180.0

def to_local_xy(lat0, lon0, lat, lon):
    """Convierte lat/lon → coords locales (m) con proyección equirectangular."""
    lat0r = deg2rad(lat0)
    x = deg2rad(lon - lon0) * math.cos(lat0r) * R_EARTH
    y = deg2rad(lat - lat0) * R_EARTH
    return x, y

def read_gpx_points(path):
    """Lee puntos del GPX."""
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    pts = []
    if not gpx.tracks:
        return pts
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                t = p.time
                if t and t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                pts.append({
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "ele": p.elevation or 0.0,
                    "time": t
                })
        break
    return pts

def compute_deltas(points):
    """Calcula dx,dy,dz (m/s) entre puntos consecutivos (1 Hz)."""
    if len(points) < 2:
        return [], [], []
    lat0, lon0 = points[0]["lat"], points[0]["lon"]
    xs, ys, zs = [], [], []
    for p in points:
        x, y = to_local_xy(lat0, lon0, p["lat"], p["lon"])
        z = p["ele"]
        xs.append(x); ys.append(y); zs.append(z)
    dx, dy, dz = [], [], []
    for i in range(1, len(xs)):
        dx.append(xs[i] - xs[i-1])
        dy.append(ys[i] - ys[i-1])
        dz.append(zs[i] - zs[i-1])
    # muestreo 1 Hz → deltas en m/s
    return dx, dy, dz

# ====================================================
# Detección robusta "slope-aware"
# ====================================================
def rolling_hampel(x, win=15, k=3.0):
    """Máscara booleana de outliers por Hampel (mediana + MAD)."""
    n = len(x)
    out = [False]*n
    half = win//2
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        window = [v for v in x[i0:i1] if not math.isnan(v)]
        if len(window) < 3:
            continue
        med = np.median(window)
        mad = np.median(np.abs(np.array(window) - med)) + 1e-9
        if abs(x[i] - med) > k * 1.4826 * mad:
            out[i] = True
    return out

def compute_metrics(dx, dy, dz):
    """Devuelve métricas por índice: vh (m/s), vt (m/s), grade (tanθ), vvert (m/s)."""
    n = len(dx)
    vh = [math.sqrt(dx[i]**2 + dy[i]**2) for i in range(n)]
    vt = [math.sqrt(dx[i]**2 + dy[i]**2 + dz[i]**2) for i in range(n)]
    grade = [(dz[i] / max(1e-6, vh[i])) for i in range(n)]  # tan(θ) ~ pendiente
    vvert = dz[:]  # m/s (ya es delta vertical)
    return vh, vt, grade, vvert

def reasons_at(i, vh, vt, grade, dz_out, jerk):
    """Devuelve lista de causas (strings) que disparan la alarma en i."""
    reasons = []
    if vh[i] > V_H_MAX:  reasons.append("VH")
    if vt[i] > V_TOT_MAX: reasons.append("VT")
    if grade[i] > GRADE_UP_MAX: reasons.append("GRADE_UP")
    if grade[i] < -GRADE_DOWN_MAX: reasons.append("GRADE_DOWN")
    if dz_out[i]: reasons.append("DZ_HAMPEL")
    if jerk[i] > JERK_MAX: reasons.append("JERK")
    return reasons

def find_anomalous_zones_explained(dx, dy, dz):
    """Detecta zonas y devuelve zonas con causas y métricas agregadas."""
    n = len(dx)
    vh, vt, grade, vvert = compute_metrics(dx, dy, dz)
    dz_out = rolling_hampel(dz, win=HAMPEL_WIN, k=HAMPEL_K)
    jerk = [0.0]*n
    for i in range(1, n):
        jerk[i] = abs(vt[i] - vt[i-1])

    zones = []
    i = 0
    while i < n:
        r = reasons_at(i, vh, vt, grade, dz_out, jerk)
        if r:
            j = i
            # métricas extremas dentro de la zona
            max_abs_dx = abs(dx[i]); max_abs_dy = abs(dy[i]); max_abs_dz = abs(dz[i])
            reasons_set = set(r)

            # extremos interpretables:
            # - vh_kmh máximo
            best_vh_kmh = vh[i]*3.6
            # - grade extrema por |grade|
            best_grade = grade[i]
            # - vvert extrema por |vvert|
            best_vvert = vvert[i]  # m/s

            while j + 1 < n:
                r2 = reasons_at(j+1, vh, vt, grade, dz_out, jerk)
                if not r2:
                    break
                j += 1
                reasons_set.update(r2)
                max_abs_dx = max(max_abs_dx, abs(dx[j]))
                max_abs_dy = max(max_abs_dy, abs(dy[j]))
                max_abs_dz = max(max_abs_dz, abs(dz[j]))
                # actualizar extremos interpretables
                if vh[j]*3.6 > best_vh_kmh:
                    best_vh_kmh = vh[j]*3.6
                if abs(grade[j]) > abs(best_grade):
                    best_grade = grade[j]
                if abs(vvert[j]) > abs(best_vvert):
                    best_vvert = vvert[j]

            if j - i + 1 >= MIN_ZONE_LEN:
                zones.append({
                    "idx_ini": i,
                    "idx_fin": j + 1,
                    "dur": j - i + 1,
                    "max_dx": max_abs_dx,
                    "max_dy": max_abs_dy,
                    "max_dz": max_abs_dz,
                    "reasons": sorted(list(reasons_set)),
                    # métricas interpretables (representativas del tramo)
                    "vh_kmh_max": best_vh_kmh,
                    "grade_percent_extreme": best_grade * 100.0,
                    "vz_m_per_min_extreme": best_vvert * 60.0
                })
            i = j + 1
        else:
            i += 1
    return zones, (vh, vt, grade, vvert)

# ====================================================
# Visualización
# ====================================================
def plot_deltas_with_anomalies(dx, dy, dz, zones, pattern_name):
    """Genera gráfica con dx,dy,dz y |v| total; sombrea zonas anómalas."""
    plt.figure(figsize=(12,6))
    t = range(len(dx))
    v_total = [math.sqrt(dx[i]**2 + dy[i]**2 + dz[i]**2) for i in range(len(dx))]

    plt.plot(t, dx, label='dx (m/s)', color='tab:blue', linewidth=0.8)
    plt.plot(t, dy, label='dy (m/s)', color='tab:orange', linewidth=0.8)
    plt.plot(t, dz, label='dz (m/s)', color='tab:green', linewidth=0.8)
    plt.plot(t, v_total, label='|v| total (m/s)', color='black', linewidth=1.0, alpha=0.8)

    for z in zones:
        plt.axvspan(z["idx_ini"], z["idx_fin"], color='red', alpha=0.2)

    plt.title(f"Análisis de deltas - {pattern_name}")
    plt.xlabel("Índice de punto (1 Hz)")
    plt.ylabel("Delta / Velocidad (m/s)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, f"{pattern_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

# ====================================================
# GPX de anomalías (segmentos + waypoint por tramo)
# ====================================================
def save_anomalies_gpx(points, zones, pattern_name):
    """Guarda segmentos anómalos + waypoint con causas y métricas."""
    gpx = gpxpy.gpx.GPX()
    trk = gpxpy.gpx.GPXTrack(name=f"{pattern_name}_anomalies")
    gpx.tracks.append(trk)

    for idx, z in enumerate(zones, start=1):
        seg = gpxpy.gpx.GPXTrackSegment()
        i0 = z["idx_ini"]
        i1 = min(z["idx_fin"], len(points))
        for i in range(i0, i1):
            p = points[i]
            seg.points.append(gpxpy.gpx.GPXTrackPoint(
                p["lat"], p["lon"], elevation=p["ele"], time=p["time"]
            ))
        trk.segments.append(seg)

        # waypoint en el inicio del tramo
        p0 = points[i0]
        wpt = gpxpy.gpx.GPXWaypoint(latitude=p0["lat"], longitude=p0["lon"],
                                     elevation=p0["ele"], time=p0["time"])
        cause_str = ",".join(z["reasons"])
        wpt.name = f"ANOM:{cause_str}"
        wpt.description = (
            f"Tramo #{idx} ({i0}-{i1-1}), dur={z['dur']}s\n"
            f"Causas: {cause_str}\n"
            f"vh_max={z['vh_kmh_max']:.2f} km/h | "
            f"grade_ext={z['grade_percent_extreme']:.1f}% | "
            f"vz_ext={z['vz_m_per_min_extreme']:.1f} m/min\n"
            f"max|dx|={z['max_dx']:.2f} m/s, max|dy|={z['max_dy']:.2f} m/s, max|dz|={z['max_dz']:.2f} m/s"
        )
        gpx.waypoints.append(wpt)

    os.makedirs(GPX_DIR, exist_ok=True)
    out_path = os.path.join(GPX_DIR, f"{pattern_name}_anomalies.gpx")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())
    return out_path

# ====================================================
# Proceso principal
# ====================================================
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(GPX_DIR, exist_ok=True)

    rows = []
    pasadas = [d for d in sorted(os.listdir(PRE_DIR)) if os.path.isdir(os.path.join(PRE_DIR, d))]

    for pasada in tqdm(pasadas, desc="Analizando patrones"):
        pdir = os.path.join(PRE_DIR, pasada)
        pattern_files = glob.glob(os.path.join(pdir, "*_pattern_aligned_resampled.gpx"))
        if not pattern_files:
            pattern_files = glob.glob(os.path.join(pdir, "*pattern*resampled.gpx"))
        if not pattern_files:
            continue

        for path in pattern_files:
            pts = read_gpx_points(path)
            if len(pts) < 3:
                continue
            dx, dy, dz = compute_deltas(pts)
            zones, metrics = find_anomalous_zones_explained(dx, dy, dz)
            pattern_name = os.path.splitext(os.path.basename(path))[0]
            plot_path = plot_deltas_with_anomalies(dx, dy, dz, zones, pattern_name)
            gpx_path = save_anomalies_gpx(pts, zones, pattern_name)

            for z in zones:
                rows.append([
                    pasada,
                    os.path.basename(path),
                    z["idx_ini"],
                    z["idx_fin"],
                    z["dur"],
                    "|".join(z["reasons"]),
                    f"{z['vh_kmh_max']:.3f}",
                    f"{z['grade_percent_extreme']:.3f}",
                    f"{z['vz_m_per_min_extreme']:.3f}",
                    f"{z['max_dx']:.3f}",
                    f"{z['max_dy']:.3f}",
                    f"{z['max_dz']:.3f}",
                    plot_path,
                    gpx_path
                ])
                print(f"[{pasada}] {pattern_name}: {z['idx_ini']}–{z['idx_fin']} "
                      f"({z['dur']} s) reasons={','.join(z['reasons'])} "
                      f"vh_max={z['vh_kmh_max']:.2f} km/h "
                      f"grade_ext={z['grade_percent_extreme']:.1f}% "
                      f"vz_ext={z['vz_m_per_min_extreme']:.1f} m/min")

    # guardar CSV
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pasada","pattern_file","idx_ini","idx_fin","dur_points",
            "reasons","vh_kmh_max","grade_percent_extreme","vz_m_per_min_extreme",
            "max_dx_m_s","max_dy_m_s","max_dz_m_s","plot_path","gpx_path"
        ])
        w.writerows(rows)

    print(f"\n✅ Informe generado: {REPORT_PATH}")
    print(f"✅ Gráficas guardadas en: {PLOTS_DIR}")
    print(f"✅ GPX de anomalías guardados en: {GPX_DIR}")
    print(f"Total de zonas anómalas detectadas: {len(rows)}")

if __name__ == "__main__":
    main()
