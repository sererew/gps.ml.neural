#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analiza las grabaciones en cada pasada (data/raw/<pasada>/)
y detecta incoherencias en las fechas/horas de inicio y fin.

Genera:
  - data/reports/time_ranges.csv
"""

import os
import glob
import csv
import statistics
from datetime import datetime, timezone
import gpxpy
from tqdm import tqdm

# ====================================================
# Configuración
# ====================================================
RAW_DIR = os.path.join("data", "raw")
REPORT_DIR = os.path.join("data", "reports")
REPORT_PATH = os.path.join(REPORT_DIR, "time_ranges.csv")

# Si las horas difieren más de este margen del resto (segundos) → marcar como error
THRESHOLD_SECONDS = 60 * 15 # minutos

# ====================================================
# Utilidades
# ====================================================
def read_gpx_times(path):
    """Devuelve lista de timestamps UTC de un archivo GPX."""
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    times = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.time is None:
                    continue
                t = p.time
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                times.append(t)
    return sorted(times)

def format_dt(t):
    return t.strftime("%Y-%m-%d %H:%M:%S %Z")

def analyze_passada(pasada_dir):
    """Analiza todas las grabaciones de una pasada."""
    pattern_files = glob.glob(os.path.join(pasada_dir, "*_pattern.gpx"))
    pattern = pattern_files[0] if pattern_files else None

    rec_files = [f for f in glob.glob(os.path.join(pasada_dir, "*.gpx"))
                 if f != pattern]

    results = []
    for f in rec_files:
        times = read_gpx_times(f)
        if not times:
            continue
        t0 = times[0]
        t1 = times[-1]
        dur_s = (t1 - t0).total_seconds()
        dur_m = dur_s / 60.0
        results.append({
            "file": os.path.basename(f),
            "t_start": t0,
            "t_end": t1,
            "duration_min": dur_m,
            "duration_s": dur_s,
            "n_points": len(times)
        })
    return results

def detect_anomalies(results):
    """Marca posibles grabaciones con hora incoherente."""
    if not results:
        return
    # medianas de inicio y fin
    start_ts = [r["t_start"].timestamp() for r in results]
    end_ts = [r["t_end"].timestamp() for r in results]
    med_start = statistics.median(start_ts)
    med_end = statistics.median(end_ts)

    for r in results:
        delta_start = abs(r["t_start"].timestamp() - med_start)
        delta_end = abs(r["t_end"].timestamp() - med_end)
        if delta_start > THRESHOLD_SECONDS or delta_end > THRESHOLD_SECONDS:
            r["anomaly"] = "⚠️ fuera de rango"
        else:
            r["anomaly"] = "✅ ok"

# ====================================================
# Proceso principal
# ====================================================
def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    rows = []

    pasadas = [d for d in sorted(os.listdir(RAW_DIR))
               if os.path.isdir(os.path.join(RAW_DIR, d))]

    for pasada in tqdm(pasadas, desc="Analizando pasadas"):
        pdir = os.path.join(RAW_DIR, pasada)
        results = analyze_passada(pdir)
        detect_anomalies(results)

        for r in results:
            rows.append([
                pasada,
                r["file"],
                format_dt(r["t_start"]),
                format_dt(r["t_end"]),
                f"{r['duration_min']:.1f}",
                int(r["duration_s"]),
                r["n_points"],
                r["anomaly"]
            ])

    # Guardar CSV
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pasada","grabacion","t_start","t_end",
            "duration_min","duration_s","n_points","anomaly"
        ])
        w.writerows(rows)

    print(f"\n✅ Informe generado: {REPORT_PATH}")
    print("Revisa las filas con '⚠️ fuera de rango' para detectar relojes mal configurados.")

if __name__ == "__main__":
    main()
