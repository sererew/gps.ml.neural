#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera dataset para entrenamiento de la red con splits train/val/test sin fuga de información:
- Lee metadatos CSV con splits por pasada
- Lee data/preprocessed/<pasada> con:
    - <n>_pattern_aligned_resampled.gpx  (patrón limpio 1 Hz)
    - <grabacion>_resampled.gpx          (grabaciones 1 Hz)
- Sincroniza grabaciones con el patrón por rango de tiempos común
- Convierte lat/lon/(ele) -> (x,y,z) locales (m)
- Calcula deltas (dx,dy,dz)
- Ventanas configurables con solape
- Padding con ceros y máscara binaria
- Normalización basada SOLO en TRAIN (sin fuga de información)
- Guarda CSVs en estructura por set:
      {OUT_ROOT}/{SET}/slices/
      {OUT_ROOT}/{SET}/labels/
      {OUT_ROOT}/{SET}/masks/
- Crea manifests por set y estadísticas de normalización
"""

import os, glob, math, json, csv, argparse
from datetime import timezone
import gpxpy
from tqdm import tqdm
import pandas as pd

# ==========================================================
# Configuración por defecto
# ==========================================================
PRE_DIR = os.path.join("data", "preprocessed")
R_EARTH = 6371000.0  # radio terrestre [m]

# ==========================================================
# Utilidades geométricas y GPX
# ==========================================================
def deg2rad(d): 
    return d * math.pi / 180.0

def to_local_xy(lat0, lon0, lat, lon):
    """Proyección equirectangular local (m)."""
    lat0r = deg2rad(lat0)
    x = deg2rad(lon - lon0) * math.cos(lat0r) * R_EARTH
    y = deg2rad(lat - lat0) * R_EARTH
    return x, y

def read_gpx_points(path):
    """Devuelve lista de puntos con lat, lon, ele, time."""
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
        if t is None: 
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        ele = p.elevation if p.elevation is not None else 0.0
        pts.append({"lat": p.latitude, "lon": p.longitude, "ele": ele, "time": t})
    return pts

def build_time_index(points):
    """Mapea segundo UNIX -> punto."""
    return {int(p["time"].timestamp()): p for p in points}

def common_time_range(a, b):
    """Devuelve rango [t0, t1] de solape común (en epoch segundos)."""
    if not a or not b: 
        return (None, None)
    ta0, ta1 = min(a), max(a)
    tb0, tb1 = min(b), max(b)
    t0, t1 = max(ta0, tb0), min(ta1, tb1)
    return (t0, t1) if (t1 - t0) >= 1 else (None, None)

def to_seq(idx, lat0, lon0, t0, t1, use_z=True):
    """Convierte índice GPX -> secuencias x,y,z,t (1 Hz)."""
    xs, ys, zs, ts = [], [], [], []
    last = None
    for t in range(t0, t1 + 1):
        p = idx.get(t, last)
        if p is None:
            xs.append(math.nan)
            ys.append(math.nan)
            zs.append(0.0)
            ts.append(t)
            continue
        x, y = to_local_xy(lat0, lon0, p["lat"], p["lon"])
        z = p["ele"] if use_z and p["ele"] is not None else 0.0
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ts.append(t)
        last = p
    return xs, ys, zs, ts

def deltas(xs, ys, zs):
    """Calcula deltas consecutivos."""
    n = len(xs)
    dx = [0] * n
    dy = [0] * n
    dz = [0] * n
    for i in range(1, n):
        dx[i] = xs[i] - xs[i-1]
        dy[i] = ys[i] - ys[i-1]
        dz[i] = zs[i] - zs[i-1]
    return dx, dy, dz

# ==========================================================
# Utilidades varias
# ==========================================================
def window_indices(n, win, step):
    """Devuelve (i0, i1, sufijo) para ventanas solapadas."""
    out = []
    k = 1
    start = 0
    while start < n:
        end = min(n - 1, start + win - 1)
        suffix = '' if (k % 2) == 1 else 'a'
        out.append((start, end, suffix))
        if end == n - 1:
            break
        start += step
        k += 1
    return out

def pad(rows, win):
    """Rellena con ceros hasta win puntos."""
    m = len(rows)
    if m >= win:
        return rows[:win], [1] * win
    # Padding con ceros, manteniendo el último timestamp + incrementos
    last_time = rows[-1][0] if rows else 0
    padrows = rows + [[last_time + i + 1, 0, 0, 0] for i in range(win - m)]
    mask = [1] * m + [0] * (win - m)
    return padrows, mask

def norm(v, m, s): 
    return 0.0 if s <= 1e-12 else (v - m) / s

def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def load_metadata(meta_path):
    """Carga CSV de metadatos y devuelve dict por pasada."""
    df = pd.read_csv(meta_path)
    required_cols = {'pasada', 'modalidad', 'set'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"CSV de metadatos debe tener columnas {required_cols}. Faltan: {missing}")
    
    meta_dict = {}
    for _, row in df.iterrows():
        pasada = str(row['pasada'])
        meta_dict[pasada] = {
            'modalidad': row['modalidad'],
            'set': row['set']
        }
    return meta_dict

def get_eligible_pasadas(meta_dict, target_set):
    """Devuelve lista de pasadas elegibles para el set dado."""
    eligible = []
    for pasada, info in meta_dict.items():
        if info['set'] == target_set:
            eligible.append(pasada)
    return sorted(eligible)

# ==========================================================
# PASO 1: calcular estadísticas solo de TRAIN
# ==========================================================
def calculate_train_stats(eligible_pasadas, use_z=True):
    """Calcula estadísticas de normalización solo con pasadas de TRAIN."""
    sumx = sumy = sumz = sum2x = sum2y = sum2z = 0.0
    count = 0
    
    print(f"📊 Calculando estadísticas con {len(eligible_pasadas)} pasadas de TRAIN...")
    
    for pasada in tqdm(eligible_pasadas, desc="Calculando stats TRAIN"):
        pdir = os.path.join(PRE_DIR, pasada)
        if not os.path.isdir(pdir):
            print(f"⚠️  Pasada {pasada} no encontrada en {pdir}")
            continue
            
        # Buscar patrón
        pattern_files = glob.glob(os.path.join(pdir, "*_pattern_aligned_resampled.gpx"))
        if not pattern_files:
            pattern_files = glob.glob(os.path.join(pdir, "*pattern*resampled.gpx"))
        
        if not pattern_files:
            print(f"⚠️  No se encontró patrón en {pdir}")
            continue
        
        trp_path = pattern_files[0]
        trp_pts = read_gpx_points(trp_path)
        if len(trp_pts) < 2:
            continue
        
        lat0, lon0 = trp_pts[0]["lat"], trp_pts[0]["lon"]
        trp_idx = build_time_index(trp_pts)
        
        # Buscar grabaciones ruidosas
        recs = [p for p in glob.glob(os.path.join(pdir, "*_resampled.gpx"))
                if os.path.basename(p) != os.path.basename(trp_path)]
        
        for rp in recs:
            rec_pts = read_gpx_points(rp)
            if len(rec_pts) < 2:
                continue
            
            rec_idx = build_time_index(rec_pts)
            t0, t1 = common_time_range(trp_idx, rec_idx)
            if t0 is None:
                continue
            
            # Convertir grabación ruidosa a secuencia local
            xg, yg, zg, tg = to_seq(rec_idx, lat0, lon0, t0, t1, use_z)
            valid = [i for i in range(len(xg)) 
                     if not (math.isnan(xg[i]) or math.isnan(yg[i]))]
            if len(valid) < 2:
                continue
            
            xg = [xg[i] for i in valid]
            yg = [yg[i] for i in valid] 
            zg = [zg[i] for i in valid]
            dx, dy, dz = deltas(xg, yg, zg)
            
            # Acumular estadísticas
            for a, b, c in zip(dx, dy, dz):
                sumx += a
                sumy += b
                sumz += c
                sum2x += a * a
                sum2y += b * b
                sum2z += c * c
                count += 1
    
    if count == 0:
        print("⚠️  No se encontraron datos válidos para calcular estadísticas")
        return {
            "mean": {"dx": 0, "dy": 0, "dz": 0},
            "std": {"dx": 1, "dy": 1, "dz": 1},
            "count": 0
        }
    
    # Calcular estadísticas finales
    meanx = sumx / count
    meany = sumy / count
    meanz = sumz / count
    
    stdx = math.sqrt(max(1e-12, (sum2x / count) - meanx**2))
    stdy = math.sqrt(max(1e-12, (sum2y / count) - meany**2))
    stdz = math.sqrt(max(1e-12, (sum2z / count) - meanz**2))
    
    return {
        "mean": {"dx": meanx, "dy": meany, "dz": meanz},
        "std": {"dx": stdx, "dy": stdy, "dz": stdz},
        "count": count
    }

# ==========================================================
# PASO 2: generar CSVs para un set específico
# ==========================================================
def generate_csvs_for_set(eligible_pasadas, meta_dict, stats, args):
    """Genera CSVs para las pasadas de un set específico."""
    mean, std = stats["mean"], stats["std"]
    manifest = []
    
    # Configurar directorios de salida para este set
    set_dir = os.path.join(args.out, args.set)
    slices_dir = os.path.join(set_dir, "slices")
    labels_dir = os.path.join(set_dir, "labels")
    masks_dir = os.path.join(set_dir, "masks")
    
    # Crear directorios
    for d in [slices_dir, labels_dir, masks_dir]:
        os.makedirs(d, exist_ok=True)
    
    print(f"📁 Generando {args.set.upper()} con {len(eligible_pasadas)} pasadas...")
    
    for pasada in tqdm(eligible_pasadas, desc=f"Generando {args.set}"):
        pdir = os.path.join(PRE_DIR, pasada)
        if not os.path.isdir(pdir):
            continue
        
        # Buscar patrón
        pattern_files = glob.glob(os.path.join(pdir, "*_pattern_aligned_resampled.gpx"))
        if not pattern_files:
            pattern_files = glob.glob(os.path.join(pdir, "*pattern*resampled.gpx"))
        
        if not pattern_files:
            continue
        
        trp_path = pattern_files[0]
        trp_pts = read_gpx_points(trp_path)
        if len(trp_pts) < 2:
            continue
        
        lat0, lon0 = trp_pts[0]["lat"], trp_pts[0]["lon"]
        trp_idx = build_time_index(trp_pts)
        pattern_name = os.path.splitext(os.path.basename(trp_path))[0]
        
        # Buscar grabaciones
        recs = [p for p in glob.glob(os.path.join(pdir, "*_resampled.gpx"))
                if os.path.basename(p) != os.path.basename(trp_path)]
        
        for rp in recs:
            rec_name = os.path.splitext(os.path.basename(rp))[0]
            rec_pts = read_gpx_points(rp)
            if len(rec_pts) < 2:
                continue
            
            rec_idx = build_time_index(rec_pts)
            t0, t1 = common_time_range(trp_idx, rec_idx)
            if t0 is None:
                continue
            
            # Convertir ambas secuencias
            xp, yp, zp, tp = to_seq(trp_idx, lat0, lon0, t0, t1, args.use_z)
            xg, yg, zg, tg = to_seq(rec_idx, lat0, lon0, t0, t1, args.use_z)
            
            # Filtrar puntos válidos
            valid = [i for i in range(len(xg)) 
                     if not (math.isnan(xg[i]) or math.isnan(yg[i]) or 
                            math.isnan(xp[i]) or math.isnan(yp[i]))]
            if len(valid) < 2:
                continue
            
            # Extraer puntos válidos
            xp = [xp[i] for i in valid]
            yp = [yp[i] for i in valid] 
            zp = [zp[i] for i in valid]
            tp = [tp[i] for i in valid]
            xg = [xg[i] for i in valid]
            yg = [yg[i] for i in valid]
            zg = [zg[i] for i in valid]
            tg = [tg[i] for i in valid]
            
            # Calcular deltas
            dxp, dyp, dzp = deltas(xp, yp, zp)
            dxg, dyg, dzg = deltas(xg, yg, zg)
            
            # Normalizar usando estadísticas de TRAIN
            dxp = [norm(v, mean["dx"], std["dx"]) for v in dxp]
            dyp = [norm(v, mean["dy"], std["dy"]) for v in dyp]
            dzp = [norm(v, mean["dz"], std["dz"]) for v in dzp]
            dxg = [norm(v, mean["dx"], std["dx"]) for v in dxg]
            dyg = [norm(v, mean["dy"], std["dy"]) for v in dyg]
            dzg = [norm(v, mean["dz"], std["dz"]) for v in dzg]
            
            n = len(tp)
            
            # Generar ventanas
            for k, (i0, i1, suf) in enumerate(window_indices(n, args.win, args.step), start=1):
                rows_lab = [[i - i0, dxp[i], dyp[i], dzp[i]] for i in range(i0, i1 + 1)]
                rows_slc = [[i - i0, dxg[i], dyg[i], dzg[i]] for i in range(i0, i1 + 1)]
                
                rows_lab, mask_lab = pad(rows_lab, args.win)
                rows_slc, mask_slc = pad(rows_slc, args.win)
                
                # Ajustar índices de tiempo
                for i in range(args.win):
                    rows_lab[i][0] = i
                    rows_slc[i][0] = i
                
                tag = f"{k}{suf}"
                label_fn = f"{pattern_name}_{tag}.csv"
                slice_fn = f"{rec_name}_{tag}.csv"
                
                label_path = os.path.join(labels_dir, label_fn)
                slice_path = os.path.join(slices_dir, slice_fn)
                mask_path = os.path.join(masks_dir, slice_fn)
                
                # Guardar archivos
                save_csv(label_path, ["time", "dx", "dy", "dz"], rows_lab)
                save_csv(slice_path, ["time", "dx", "dy", "dz"], rows_slc)
                save_csv(mask_path, ["mask"], [[m] for m in mask_slc])
                
                # Añadir a manifest
                modalidad = meta_dict.get(pasada, {}).get('modalidad', 'unknown')
                manifest.append([
                    pasada, modalidad, args.set, rec_name, pattern_name, tag,
                    tp[i0], tp[min(i1, len(tp) - 1)],
                    slice_path, label_path, mask_path,
                    len(rows_slc)
                ])
    
    # Escribir manifest
    manifest_path = os.path.join(set_dir, f"manifest_{args.set}.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pasada", "modalidad", "set", "grabacion", "pattern", "window_id",
            "t_start", "t_end", "slice_path", "label_path", "mask_path", "n_points"
        ])
        w.writerows(manifest)
    
    return len(manifest)

# ==========================================================
# MAIN
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Genera dataset por splits sin fuga de información")
    parser.add_argument("--meta", required=True, help="CSV con metadatos de pasadas")
    parser.add_argument("--set", choices=["train", "val", "test"], required=True, help="Set a generar")
    parser.add_argument("--out", default="data/input", help="Directorio raíz de salida")
    parser.add_argument("--win", type=int, default=3600, help="Tamaño de ventana en segundos")
    parser.add_argument("--step", type=int, default=1800, help="Paso de ventana en segundos")
    parser.add_argument("--use_z", type=bool, default=True, help="Usar elevación")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"🚀 Generando dataset {args.set.upper()}")
    print(f"   Metadatos: {args.meta}")
    print(f"   Salida: {args.out}")
    print(f"   Ventana: {args.win}s, Paso: {args.step}s")
    
    # Cargar metadatos
    try:
        meta_dict = load_metadata(args.meta)
        print(f"📋 Cargados metadatos para {len(meta_dict)} pasadas")
    except Exception as e:
        print(f"❌ Error cargando metadatos: {e}")
        return 1
    
    # Obtener pasadas elegibles
    eligible_pasadas = get_eligible_pasadas(meta_dict, args.set)
    if not eligible_pasadas:
        print(f"❌ No se encontraron pasadas para set '{args.set}'")
        return 1
    
    print(f"🎯 Pasadas elegibles para {args.set}: {eligible_pasadas}")
    
    # Manejar estadísticas de normalización
    stats_path = os.path.join(args.out, "norm_stats_train.json")
    
    if args.set == "train":
        # Calcular estadísticas solo con TRAIN
        stats = calculate_train_stats(eligible_pasadas, args.use_z)
        os.makedirs(args.out, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"💾 Estadísticas TRAIN guardadas: {stats['count']} puntos")
        print(f"   Mean: {stats['mean']}")
        print(f"   Std: {stats['std']}")
    else:
        # Cargar estadísticas de TRAIN
        if not os.path.exists(stats_path):
            print(f"❌ Faltan estadísticas de TRAIN: {stats_path}")
            print("   Ejecuta primero con --set train")
            return 1
        
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(f"📖 Usando estadísticas TRAIN: {stats['count']} puntos")
    
    # Generar CSVs
    n_windows = generate_csvs_for_set(eligible_pasadas, meta_dict, stats, args)
    
    print(f"✅ Dataset {args.set.upper()} generado")
    print(f"   Pasadas procesadas: {len(eligible_pasadas)}")
    print(f"   Ventanas generadas: {n_windows}")
    print(f"   Directorio: {os.path.join(args.out, args.set)}")
    
    return 0

if __name__ == "__main__":
    exit(main())