#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resamplea las grabaciones a 1 Hz.
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

"""

import os, glob, time
from datetime import datetime, timedelta, timezone
import gpxpy, gpxpy.gpx

RAW_DIR = os.path.join("data", "raw")
PRE_DIR = os.path.join("data", "preprocessed")

# ---------------- GPX helpers ----------------
def read_gpx_points(path):
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
        if t is not None and t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        pts.append({"lat": p.latitude, "lon": p.longitude, "ele": p.elevation, "time": t})
    return pts

def write_gpx_points(path, name, points):
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

# ---------------- Resample 1 Hz ----------------
def interpolate_point(pA, pB, t_target):
    """
       Interpola linealmente entre pA y pB para obtener punto en t_target.
    """
    tA, tB = pA["time"], pB["time"]
    if tA is None or tB is None:
        return {"lat": pA["lat"], "lon": pA["lon"], "ele": pA["ele"], "time": t_target}
    total = (tB - tA).total_seconds()
    if total <= 0:
        return {"lat": pA["lat"], "lon": pA["lon"], "ele": pA["ele"], "time": t_target}
    a = (t_target - tA).total_seconds() / total
    a = max(0.0, min(1.0, a))
    lat = pA["lat"] + a*(pB["lat"] - pA["lat"])
    lon = pA["lon"] + a*(pB["lon"] - pA["lon"])
    ele = None
    if pA["ele"] is not None and pB["ele"] is not None:
        ele = pA["ele"] + a*(pB["ele"] - pA["ele"])
    return {"lat": lat, "lon": lon, "ele": ele, "time": t_target}

def resample_1hz(points):
    pts = [p for p in points if p["time"] is not None]
    if len(pts) < 2:
        return pts
    pts = sorted(pts, key=lambda p: p["time"])
    t = pts[0]["time"].replace(microsecond=0)
    end = pts[-1]["time"].replace(microsecond=0)
    res, i = [], 0
    while t <= end:
        # Avanza i hasta que pts[i]["time"] <= t < pts[i+1]["time"]
        while i+1 < len(pts) and pts[i+1]["time"] < t:
            i += 1
        if i+1 < len(pts) and pts[i]["time"] <= t <= pts[i+1]["time"]:
            res.append(interpolate_point(pts[i], pts[i+1], t))
        else:
            nearest = pts[0] if t < pts[0]["time"] else pts[-1]
            res.append({"lat": nearest["lat"], "lon": nearest["lon"], "ele": nearest["ele"], "time": t})
        t += timedelta(seconds=1)
    return res

# ---------------- Pipeline por pasada ----------------
def process_pasada(pasada_dir):
    base = os.path.basename(pasada_dir.rstrip(os.sep))
    print(f"\n🔹 Procesando pasada: {base}")
    t_start = time.time()

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

    # ---- Resample y guardar grabaciones ----
    for rp in rec_paths:
        pts = read_gpx_points(rp)
        pts = [p for p in pts if p["time"] is not None]
        if len(pts) < 2:
            print(f"[{base}] ⚠️ {os.path.basename(rp)}: menos de 2 puntos con tiempo.")
            continue
        
        pts_res = resample_1hz(pts)
        name = os.path.splitext(os.path.basename(rp))[0] + "_resampled"
        out_path = os.path.join(out_dir, f"{name}.gpx")
        write_gpx_points(out_path, name, pts_res)
        print(f"[{base}] ✅ Resampleado: {name}.gpx ({len(pts_res)} pts)")

    elapsed = time.time() - t_start
    print(f"[{base}] ⏱ Tiempo de procesamiento: {elapsed:.2f}s")

def main():
    if not os.path.isdir(RAW_DIR):
        print(f"❌ No existe {RAW_DIR}")
        return
    pasadas = [d for d in sorted(os.listdir(RAW_DIR)) if os.path.isdir(os.path.join(RAW_DIR, d))]
    print(f"📂 Encontradas {len(pasadas)} pasadas en {RAW_DIR}")
    for pasada in pasadas:
        process_pasada(os.path.join(RAW_DIR, pasada))

if __name__ == "__main__":
    main()