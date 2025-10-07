#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Resamplea a 1 Hz los tracks patrón alineados de cada pasada.

Estructura esperada:
data/
  raw/
    <pasada>/
      <grabacion1>.gpx
      <grabacion2>.gpx
      ...
      <n>_pattern_aligned.gpx

Salida:
data/
  preprocessed/
    <pasada>/
      <n>_pattern_aligned_resampled.gpx
"""

import os
import glob
import gpxpy
import gpxpy.gpx
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

RAW_DIR = os.path.join("data", "preprocessed")
PRE_DIR = os.path.join("data", "preprocessed")


def read_gpx_points(path):
    """Lee puntos de un GPX (primer track y segmento)."""
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    pts = []
    if not gpx.tracks:
        return pts
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                t = p.time
                if t is not None and t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                pts.append({
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "ele": p.elevation,
                    "time": t
                })
        break
    return pts


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


def interpolate_point(pA, pB, t_target):
    """Interpola posición linealmente entre pA y pB para el tiempo t_target."""
    tA, tB = pA["time"], pB["time"]
    total = (tB - tA).total_seconds()
    if total <= 0:
        return {
            "lat": pA["lat"], "lon": pA["lon"],
            "ele": pA["ele"], "time": t_target
        }
    alpha = (t_target - tA).total_seconds() / total
    alpha = max(0.0, min(1.0, alpha))
    lat = pA["lat"] + alpha * (pB["lat"] - pA["lat"])
    lon = pA["lon"] + alpha * (pB["lon"] - pA["lon"])
    ele = None
    if pA["ele"] is not None and pB["ele"] is not None:
        ele = pA["ele"] + alpha * (pB["ele"] - pA["ele"])
    return {"lat": lat, "lon": lon, "ele": ele, "time": t_target}


def resample_1hz(points):
    """Interpola el track a 1 Hz."""
    pts = [p for p in points if p["time"] is not None]
    if len(pts) < 2:
        return pts
    pts = sorted(pts, key=lambda p: p["time"])
    t0 = pts[0]["time"].replace(microsecond=0)
    t1 = pts[-1]["time"].replace(microsecond=0)
    resampled = []
    i = 0
    t = t0
    while t <= t1:
        while i + 1 < len(pts) and pts[i+1]["time"] < t:
            i += 1
        if i + 1 < len(pts) and pts[i]["time"] <= t <= pts[i+1]["time"]:
            resampled.append(interpolate_point(pts[i], pts[i+1], t))
        else:
            # fuera de rango: usar el más cercano
            nearest = pts[0] if t < pts[0]["time"] else pts[-1]
            resampled.append({
                "lat": nearest["lat"],
                "lon": nearest["lon"],
                "ele": nearest["ele"],
                "time": t
            })
        t += timedelta(seconds=1)
    return resampled


def process_pasada(pasada_dir):
    base = os.path.basename(pasada_dir.rstrip(os.sep))
    print(f"\n🔹 Procesando pasada: {base}")
    os.makedirs(os.path.join(PRE_DIR, base), exist_ok=True)

    pattern_files = glob.glob(os.path.join(pasada_dir, "*_pattern_aligned.gpx"))
    if not pattern_files:
        print(f"[{base}] ⚠️ No se encontró *_pattern_aligned.gpx")
        return
    trp_path = pattern_files[0]

    pts = read_gpx_points(trp_path)
    if not pts:
        print(f"[{base}] ⚠️ Patrón vacío")
        return

    print(f"[{base}] ⏱ Resampleando {len(pts)} puntos → 1 Hz...")
    pts_resampled = resample_1hz(pts)
    out_path = os.path.join(PRE_DIR, base,
                            os.path.basename(trp_path).replace(".gpx", "_resampled.gpx"))
    write_gpx_points(out_path, os.path.basename(out_path), pts_resampled)
    print(f"[{base}] ✅ Guardado: {out_path} ({len(pts_resampled)} puntos)")


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"❌ No existe {RAW_DIR}")
        return

    pasadas = [d for d in sorted(os.listdir(RAW_DIR))
               if os.path.isdir(os.path.join(RAW_DIR, d))]

    print(f"📂 Encontradas {len(pasadas)} pasadas")
    for pasada in tqdm(pasadas, desc="Procesando pasadas"):
        process_pasada(os.path.join(RAW_DIR, pasada))


if __name__ == "__main__":
    main()
