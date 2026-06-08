#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corrección LOCAL de alturas en tracks patrón con control simétrico de pendiente.

Corrige tanto subidas como bajadas anómalas (positivas y negativas)
sin generar nuevas rampas excesivas.

Entradas: data/raw/<pasada>/*_pattern.gpx
Salidas:
  - data/preprocessed/<pasada>/<pattern>_elev_fixed.gpx
  - data/reports/elev_fixes/pattern_elev_fixes.csv
"""

import os, glob, math, csv
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import gpxpy, gpxpy.gpx

# ==========================
# CONFIGURACIÓN
# ==========================
RAW_DIR  = os.path.join("data", "raw")
OUT_BASE = os.path.join("data", "preprocessed")
REPORT_DIR = os.path.join("data", "reports", "elev_fixes")
REPORT_CSV = os.path.join(REPORT_DIR, "pattern_elev_fixes.csv")

SLOPE_THRESHOLD   = 0.20   # pendiente anómala (>20%)
MIN_RUN_LEN       = 2
STEP_MEDIAN_DIFF  = 1.5    # m
SPIKE_MAX_LEN     = 3
WINDOW_SPIKE      = 15
MARGIN_STEP       = 10
MAX_SLOPE_FIX     = 0.20   # ±20% límite
MIN_HDIST         = 1.0    # m
GAUSS_SIGMA       = 5.0    # m
R_EARTH = 6371000.0

# ==========================
# UTILIDADES
# ==========================
def deg2rad(d): return d * math.pi / 180.0
def to_local_xy(lat0, lon0, lat, lon):
    lat0r = deg2rad(lat0)
    x = deg2rad(lon - lon0) * math.cos(lat0r) * R_EARTH
    y = deg2rad(lat - lat0) * R_EARTH
    return x, y

def cumulative_s(points_xy):
    s = [0.0]
    for i in range(1, len(points_xy)):
        dx = points_xy[i][0] - points_xy[i-1][0]
        dy = points_xy[i][1] - points_xy[i-1][1]
        s.append(s[-1] + math.hypot(dx, dy))
    return s

def read_pattern_points(path):
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    pts = []
    if not gpx.tracks: return pts
    seg = gpx.tracks[0].segments[0]
    for p in seg.points:
        pts.append([p.latitude, p.longitude, (p.elevation or 0.0), p.time])
    return pts

def write_pattern_points(path_out, pts, name="pattern_elev_fixed"):
    gpx = gpxpy.gpx.GPX()
    trk = gpxpy.gpx.GPXTrack(name=name)
    gpx.tracks.append(trk)
    seg = gpxpy.gpx.GPXTrackSegment()
    trk.segments.append(seg)
    for lat, lon, ele, t in pts:
        seg.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=ele, time=t))
    with open(path_out, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())

def compute_slopes(points):
    n = len(points)
    if n < 2: return [0.0]*n
    lat0, lon0 = points[0][0], points[0][1]
    xy = [to_local_xy(lat0, lon0, p[0], p[1]) for p in points]
    slopes = [0.0]*n
    for i in range(1, n):
        dz = points[i][2] - points[i-1][2]
        dh = math.hypot(xy[i][0] - xy[i-1][0], xy[i][1] - xy[i-1][1])
        slopes[i] = 0.0 if dh < 1e-6 else dz / dh
    return slopes

def group_runs(mask):
    runs=[]; i=0; n=len(mask)
    while i<n:
        if mask[i]:
            j=i
            while j+1<n and mask[j+1]:
                j+=1
            if j-i+1 >= MIN_RUN_LEN:
                runs.append((i,j))
            i=j+1
        else:
            i+=1
    return runs

def median_around(z, i0, i1, left_win, right_win):
    Ls = z[max(0, i0-left_win):i0]
    Rs = z[i1+1:min(len(z), i1+1+right_win)]
    lmed = float(np.median(Ls)) if len(Ls)>0 else z[i0]
    rmed = float(np.median(Rs)) if len(Rs)>0 else z[i1]
    return lmed, rmed

# ==========================
# CORRECCIÓN LOCAL
# ==========================
def weighted_linear_fit(s, z, weights):
    A = np.vstack([s, np.ones_like(s)]).T
    W = np.diag(weights)
    Aw = W @ A
    zw = W @ z
    coeff, *_ = np.linalg.lstsq(Aw, zw, rcond=None)
    return float(coeff[0]), float(coeff[1])

def limit_slope(z_prev, s_prev, z_est, s_curr):
    """Limita pendiente local (simétrica ±MAX_SLOPE_FIX)."""
    ds = s_curr - s_prev
    if ds <= 0:
        return z_est
    dz = z_est - z_prev
    slope = dz / ds
    if abs(slope) > MAX_SLOPE_FIX:
        dz = math.copysign(MAX_SLOPE_FIX * ds, dz)
        return z_prev + dz
    return z_est

def apply_spike_fix(z, s, i0, i1, win=15):
    """Regresión ponderada simétrica + control de pendiente local."""
    n = len(z)
    L = max(0, i0 - win)
    R = min(n - 1, i1 + win)
    center = (i0 + i1) / 2
    w = np.exp(-0.5 * ((s[L:R+1] - s[int(center)]) / GAUSS_SIGMA)**2)
    mask = np.ones(R - L + 1, dtype=bool)
    mask[(i0 - L):(i1 - L + 1)] = False
    if mask.sum() < 2:
        return z
    a, b = weighted_linear_fit(s[L:R+1][mask], z[L:R+1][mask], w[mask])
    z_new = z.copy()
    for i in range(i0, i1+1):
        z_est = a * s[i] + b
        if i > 0:
            z_est = limit_slope(z_new[i-1], s[i-1], z_est, s[i])
        z_new[i] = z_est
    return z_new

def apply_step_fix(points, z, s, i0, i1, margin=10):
    """Rampa simétrica con control local de pendiente."""
    n = len(z)
    L = max(0, i0 - margin)
    R = min(n - 1, i1 + margin)
    sL, sR = s[L], s[R]
    if sR - sL < MIN_HDIST:
        return z
    zL, zR = z[L], z[R]
    dz = zR - zL
    slope = dz / (sR - sL)
    # limitar pendiente global simétricamente
    if abs(slope) > MAX_SLOPE_FIX:
        dz = math.copysign(MAX_SLOPE_FIX * (sR - sL), dz)
        zR = zL + dz
        slope = dz / (sR - sL)
    z_new = z.copy()
    for i in range(L, R+1):
        alpha = (s[i] - sL) / (sR - sL)
        alpha = min(1.0, max(0.0, alpha))
        z_interp = (1 - alpha) * zL + alpha * zR
        if i > 0:
            z_interp = limit_slope(z_new[i-1], s[i-1], z_interp, s[i])
        z_new[i] = z_interp
    return z_new

# ==========================
# PROCESO DE CORRECCIÓN
# ==========================
@dataclass
class FixRecord:
    pasada: str
    pattern: str
    fix_type: str
    i0: int
    i1: int
    dz_median_diff: float
    max_abs_slope_before: float

def process_pattern(pasada, pattern_path):
    pts = read_pattern_points(pattern_path)
    if len(pts) < 3: return []
    lat0, lon0 = pts[0][0], pts[0][1]
    xy = [to_local_xy(lat0, lon0, p[0], p[1]) for p in pts]
    s = np.array(cumulative_s(xy))
    z = np.array([p[2] for p in pts])
    slopes = compute_slopes(pts)
    mask = [abs(slopes[i]) > SLOPE_THRESHOLD for i in range(len(slopes))]
    runs = group_runs(mask)
    fixes = []
    z_corr = z.copy()

    for i0, i1 in runs:
        run_len = i1 - i0 + 1
        max_slope = max(abs(slopes[i]) for i in range(i0, i1+1))
        lmed, rmed = median_around(z_corr, i0, i1, WINDOW_SPIKE, WINDOW_SPIKE)
        dz_med = rmed - lmed
        is_step = (abs(dz_med) >= STEP_MEDIAN_DIFF) and (run_len >= SPIKE_MAX_LEN)
        if is_step:
            z_corr = apply_step_fix(pts, z_corr, s, i0, i1, margin=MARGIN_STEP)
            fixes.append(FixRecord(pasada, os.path.basename(pattern_path), "STEP", i0, i1, dz_med, max_slope))
        else:
            z_corr = apply_spike_fix(z_corr, s, i0, i1, win=WINDOW_SPIKE)
            fixes.append(FixRecord(pasada, os.path.basename(pattern_path), "SPIKE", i0, i1, dz_med, max_slope))

    out_dir = os.path.join(OUT_BASE, pasada)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(pattern_path))[0] + "_elev_fixed.gpx")
    pts_out = [[pts[i][0], pts[i][1], float(z_corr[i]), pts[i][3]] for i in range(len(pts))]
    write_pattern_points(out_path, pts_out, name=os.path.basename(out_path))
    return fixes

# ==========================
# MAIN
# ==========================
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    rows = []
    pasadas = [d for d in sorted(os.listdir(RAW_DIR)) if os.path.isdir(os.path.join(RAW_DIR, d))]
    for pasada in tqdm(pasadas, desc="Corrigiendo patrones"):
        pdir = os.path.join(RAW_DIR, pasada)
        patterns = glob.glob(os.path.join(pdir, "*_pattern.gpx"))
        if not patterns:
            continue
        for p in patterns:
            fixes = process_pattern(pasada, p)
            for f in fixes:
                rows.append([f.pasada, f.pattern, f.fix_type, f.i0, f.i1, f.dz_median_diff, f.max_abs_slope_before])
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pasada","pattern_file","fix_type","idx_ini","idx_fin","dz_median_diff_m","max_abs_slope_before"])
        w.writerows(rows)
    print(f"\n✅ GPX corregidos guardados en data/preprocessed/<pasada>/")
    print(f"📝 Informe: {REPORT_CSV} | Total tramos corregidos: {len(rows)}")

if __name__ == "__main__":
    main()
