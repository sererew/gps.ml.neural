#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alinea los tiempos del track patrón con las grabaciones resampleadas a 1 Hz.

Estructura esperada:
data/
  raw/
    <pasada>/
      <grabacion1>.gpx
      <grabacion2>.gpx
      ...
      <n>_pattern.gpx

Salida:
data/
  preprocessed/
    <pasada>/
      <grabacion1>_resampled.gpx
      <grabacion2>_resampled.gpx
      ...
      <n>_pattern_aligned.gpx
"""

import os
import math
import glob
import time
import gpxpy
import gpxpy.gpx
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

RAW_DIR = os.path.join("data", "raw")
PRE_DIR = os.path.join("data", "preprocessed")

R_EARTH = 6371000.0  # Radio terrestre medio [m]

# -------------------------------------------------------
# Utilidades geométricas y de proyección
# -------------------------------------------------------
def deg2rad(d): return d * math.pi / 180.0

def local_xy(lat0, lon0, lat, lon):
    """Proyección equirectangular local centrada en (lat0, lon0)."""
    lat0r = deg2rad(lat0)
    x = deg2rad(lon - lon0) * math.cos(lat0r) * R_EARTH
    y = deg2rad(lat - lat0) * R_EARTH
    return x, y

def point_segment_projection(px, py, ax, ay, bx, by):
    """Proyecta P sobre el segmento AB. Devuelve (u, qx, qy, dist)."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx*vx + vy*vy
    if vv == 0.0:
        return 0.0, ax, ay, math.hypot(px - ax, py - ay)
    u = (wx*vx + wy*vy) / vv
    qx = ax + u*vx
    qy = ay + u*vy
    dist = math.hypot(px - qx, py - qy)
    return u, qx, qy, dist

# -------------------------------------------------------
# GPX helpers
# -------------------------------------------------------
def read_gpx_points(path):
    """Lee puntos de un GPX (primer track y segmento)."""
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    points = []
    if not gpx.tracks:
        return points
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                t = p.time
                if t is not None and t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                points.append({
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "ele": p.elevation,
                    "time": t
                })
        break
    return points

def write_gpx_points(path, name, points):
    """Escribe puntos en un GPX."""
    gpx = gpxpy.gpx.GPX()
    trk = gpxpy.gpx.GPXTrack(name=name)
    gpx.tracks.append(trk)
    seg = gpxpy.gpx.GPXTrackSegment()
    trk.segments.append(seg)
    for p in points:
        seg.points.append(gpxpy.gpx.GPXTrackPoint(
            p["lat"], p["lon"], elevation=p.get("ele"), time=p.get("time")
        ))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())

# -------------------------------------------------------
# Resample a 1 Hz
# -------------------------------------------------------
def interpolate_point(pA, pB, t_target):
    """Interpola posición linealmente para el tiempo t_target."""
    tA, tB = pA["time"], pB["time"]
    if tA is None or tB is None:
        return {"lat": pA["lat"], "lon": pA["lon"], "ele": pA["ele"], "time": t_target}
    total = (tB - tA).total_seconds()
    if total <= 0:
        return {"lat": pA["lat"], "lon": pA["lon"], "ele": pA["ele"], "time": t_target}
    alpha = (t_target - tA).total_seconds() / total
    alpha = max(0.0, min(1.0, alpha))
    lat = pA["lat"] + alpha * (pB["lat"] - pA["lat"])
    lon = pA["lon"] + alpha * (pB["lon"] - pA["lon"])
    ele = None
    if pA["ele"] is not None and pB["ele"] is not None:
        ele = pA["ele"] + alpha * (pB["ele"] - pA["ele"])
    return {"lat": lat, "lon": lon, "ele": ele, "time": t_target}

def resample_1hz(points):
    """Devuelve puntos a 1 Hz (interpolando linealmente)."""
    pts = [p for p in points if p["time"] is not None]
    if len(pts) < 2:
        return pts
    pts = sorted(pts, key=lambda p: p["time"])
    t0, t1 = pts[0]["time"], pts[-1]["time"]
    start = t0.replace(microsecond=0)
    end = t1.replace(microsecond=0)
    res = []
    i = 0
    t = start
    while t <= end:
        while i + 1 < len(pts) and pts[i+1]["time"] < t:
            i += 1
        if i + 1 < len(pts) and pts[i]["time"] <= t <= pts[i+1]["time"]:
            res.append(interpolate_point(pts[i], pts[i+1], t))
        else:
            nearest = pts[0] if t < pts[0]["time"] else pts[-1]
            res.append({"lat": nearest["lat"], "lon": nearest["lon"], "ele": nearest["ele"], "time": t})
        t += timedelta(seconds=1)
    return res

# -------------------------------------------------------
# Ajuste del instante más cercano (interpolación real)
# -------------------------------------------------------
def closest_time_via_segment_interp(pp, tr_points):
    """Busca el segmento más cercano e interpola el tiempo según u."""
    lat0, lon0 = pp["lat"], pp["lon"]
    XY = [local_xy(lat0, lon0, p["lat"], p["lon"]) for p in tr_points]

    best = None
    for i in range(len(tr_points) - 1):
        A, B = tr_points[i], tr_points[i+1]
        (ax, ay), (bx, by) = XY[i], XY[i+1]
        u, _, _, dist = point_segment_projection(0.0, 0.0, ax, ay, bx, by)
        if 0.0 <= u <= 1.0:
            tA, tB = A["time"], B["time"]
            if tA is None or tB is None:
                continue
            t_interp = tA + (tB - tA) * u
            if best is None or dist < best[0]:
                best = (dist, t_interp)

    if best is None:
        for i, P in enumerate(tr_points):
            x, y = XY[i]
            dist = math.hypot(x, y)
            if P["time"] is None:
                continue
            if best is None or dist < best[0]:
                best = (dist, P["time"])

    return best[1] if best else None

# -------------------------------------------------------
# Pipeline por "pasada"
# -------------------------------------------------------
def process_pasada(pasada_dir):
    base = os.path.basename(pasada_dir.rstrip(os.sep))
    print(f"\n🔹 Procesando pasada: {base}")
    start_time = time.time()

    out_dir = os.path.join(PRE_DIR, base)
    os.makedirs(out_dir, exist_ok=True)

    gpx_files = sorted(glob.glob(os.path.join(pasada_dir, "*.gpx")))
    if not gpx_files:
        print(f"[{base}] ❌ Sin GPX en {pasada_dir}")
        return

    pattern_files = [p for p in gpx_files if "_pattern" in os.path.basename(p).lower()]
    if not pattern_files:
        print(f"[{base}] ⚠️ No se encontró track patrón (*_pattern.gpx)")
        return

    trp_path = pattern_files[0]
    rec_paths = [p for p in gpx_files if p != trp_path]

    # 1️⃣ Resamplear grabaciones
    resampled_tracks = []
    for rp in rec_paths:
        pts = read_gpx_points(rp)
        pts = [p for p in pts if p["time"] is not None]
        if len(pts) < 2:
            print(f"[{base}] ⚠️ {os.path.basename(rp)}: menos de 2 puntos válidos.")
            continue
        pts_res = resample_1hz(pts)
        resampled_tracks.append(pts_res)
        name = os.path.splitext(os.path.basename(rp))[0] + "_resampled"
        out_path = os.path.join(out_dir, f"{name}.gpx")
        write_gpx_points(out_path, name, pts_res)
        print(f"[{base}] ✅ Resampleado: {name}.gpx ({len(pts_res)} pts)")

    if not resampled_tracks:
        print(f"[{base}] ❌ Sin grabaciones resampleadas. Saltando.")
        return

    # 2️⃣ Cargar patrón
    trp_pts = read_gpx_points(trp_path)
    if not trp_pts:
        print(f"[{base}] ❌ Patrón vacío: {os.path.basename(trp_path)}")
        return

    # 3️⃣ Ajustar tiempos
    aligned = []
    print(f"[{base}] ⏱ Ajustando {len(trp_pts)} puntos del patrón...")
    for pp in tqdm(trp_pts, desc=f"{base}", ncols=80):
        times = []
        for tr in resampled_tracks:
            t_near = closest_time_via_segment_interp(pp, tr)
            if t_near:
                times.append(t_near)
        if times:
            avg_epoch = sum(t.timestamp() for t in times) / len(times)
            tpp = datetime.fromtimestamp(avg_epoch, tz=timezone.utc)
        else:
            tpp = pp.get("time")
        aligned.append({**pp, "time": tpp})

    # 4️⃣ Guardar resultado
    pattern_name = os.path.splitext(os.path.basename(trp_path))[0]
    out_pattern = os.path.join(out_dir, f"{pattern_name}_aligned.gpx")
    write_gpx_points(out_pattern, f"{pattern_name}_aligned", aligned)

    elapsed = time.time() - start_time
    print(f"[{base}] 🎯 Patrón alineado guardado ({len(aligned)} puntos, {elapsed:.2f}s)")

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    if not os.path.isdir(RAW_DIR):
        print(f"❌ No existe {RAW_DIR}")
        return

    pasadas = [d for d in sorted(os.listdir(RAW_DIR))
               if os.path.isdir(os.path.join(RAW_DIR, d))]

    print(f"📂 Encontradas {len(pasadas)} pasadas en {RAW_DIR}")
    for pasada in pasadas:
        process_pasada(os.path.join(RAW_DIR, pasada))

if __name__ == "__main__":
    main()
