#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Genera dataset para entrenamiento de la red:
- Lee data/preprocessed/<pasada> con:
    - <n>_pattern_aligned_resampled.gpx  (patrón limpio 1 Hz)
    - <grabacion>_resampled.gpx          (grabaciones 1 Hz)
- Sincroniza grabaciones con el patrón por rango de tiempos común.
- Convierte lat/lon/(ele) -> (x,y,z) locales (m).
- Calcula deltas (dx,dy,dz).
- Ventanas de 3600 s (1h) con solape de 1800 s (media hora).
- Padding con ceros y máscara binaria.
- Normalización global (media/STD sobre todas las grabaciones).
- Guarda CSVs en:
      data/input/slices/
      data/input/labels/
      data/input/masks/
- Crea data/input/norm_stats.json y data/input/manifest.csv
"""

import os, glob, math, json, csv
from datetime import timezone
import gpxpy
from tqdm import tqdm

# ==========================================================
# Configuración
# ==========================================================
PRE_DIR = os.path.join("data", "preprocessed")
OUT_DIR = os.path.join("data", "input")
SLICES_DIR = os.path.join(OUT_DIR, "slices")
LABELS_DIR = os.path.join(OUT_DIR, "labels")
MASKS_DIR  = os.path.join(OUT_DIR, "masks")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.csv")

WINDOW_SIZE = 3600   # segundos por fragmento
STEP_SIZE   = 1800   # desplazamiento (solape)
PAD_VALUE   = 0.0
USE_Z       = True   # usar elevación si existe
R_EARTH     = 6371000.0  # radio terrestre [m]

# ==========================================================
# Utilidades geométricas y GPX
# ==========================================================
def deg2rad(d): return d * math.pi / 180.0

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
    if not gpx.tracks: return pts
    trk = gpx.tracks[0]
    if not trk.segments: return pts
    seg = trk.segments[0]
    for p in seg.points:
        t = p.time
        if t is None: 
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        ele = p.elevation if p.elevation is not None else (0.0 if USE_Z else None)
        pts.append({"lat": p.latitude, "lon": p.longitude, "ele": ele, "time": t})
    return pts

def build_time_index(points):
    """Mapea segundo UNIX -> punto."""
    return {int(p["time"].timestamp()): p for p in points}

def common_time_range(a, b):
    """Devuelve rango [t0, t1] de solape común (en epoch segundos)."""
    if not a or not b: return (None, None)
    ta0, ta1 = min(a), max(a)
    tb0, tb1 = min(b), max(b)
    t0, t1 = max(ta0, tb0), min(ta1, tb1)
    return (t0, t1) if (t1 - t0) >= 1 else (None, None)

def to_seq(idx, lat0, lon0, t0, t1):
    """Convierte índice GPX -> secuencias x,y,z,t (1 Hz)."""
    xs, ys, zs, ts = [], [], [], []
    last = None
    for t in range(t0, t1 + 1):
        p = idx.get(t, last)
        if p is None:
            xs.append(math.nan); ys.append(math.nan); zs.append(0.0); ts.append(t)
            continue
        x, y = to_local_xy(lat0, lon0, p["lat"], p["lon"])
        z = p["ele"] if USE_Z and p["ele"] is not None else 0.0
        xs.append(x); ys.append(y); zs.append(z); ts.append(t)
        last = p
    return xs, ys, zs, ts

def deltas(xs, ys, zs):
    """Calcula deltas consecutivos."""
    n = len(xs)
    dx = [0]*n; dy = [0]*n; dz = [0]*n
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
    out = []; k = 1; start = 0
    while start < n:
        end = min(n - 1, start + win - 1)
        suffix = '' if (k % 2) == 1 else 'a'
        out.append((start, end, suffix))
        if end == n - 1: break
        start += step; k += 1
    return out

def pad(rows, win):
    """Rellena con ceros hasta win puntos."""
    m = len(rows)
    if m >= win:
        return rows[:win], [1]*win
    padrows = rows + [[rows[-1][0]+i+1, 0, 0, 0] for i in range(win - m)]
    mask = [1]*m + [0]*(win - m)
    return padrows, mask

def norm(v, m, s): 
    return 0.0 if s <= 1e-12 else (v - m) / s

def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def ensure_dirs():
    for d in [SLICES_DIR, LABELS_DIR, MASKS_DIR]:
        os.makedirs(d, exist_ok=True)

# ==========================================================
# PASO 1: calcular estadísticas globales
# ==========================================================
def pass1_stats():
    sumx=sumy=sumz=sum2x=sum2y=sum2z=0.0
    count=0
    pasadas=[d for d in sorted(os.listdir(PRE_DIR)) if os.path.isdir(os.path.join(PRE_DIR,d))]
    for pasada in tqdm(pasadas, desc="Pass1: global stats"):
        pdir=os.path.join(PRE_DIR,pasada)
        pattern=glob.glob(os.path.join(pdir,"*_pattern_aligned_resampled.gpx"))
        if not pattern:
            pattern=glob.glob(os.path.join(pdir,"*pattern*resampled.gpx"))
            
        if not pattern: continue
        
        trp_path=pattern[0]
        trp_pts=read_gpx_points(trp_path)
        if len(trp_pts)<2: continue
        
        lat0,lon0=trp_pts[0]["lat"],trp_pts[0]["lon"]
        trp_idx=build_time_index(trp_pts)
        recs=[p for p in glob.glob(os.path.join(pdir,"*_resampled.gpx"))
              if os.path.basename(p)!=os.path.basename(trp_path)]
        for rp in recs:
            rec_pts=read_gpx_points(rp)
            if len(rec_pts)<2: continue
            rec_idx=build_time_index(rec_pts)
            t0,t1=common_time_range(trp_idx,rec_idx)
            if t0 is None: continue
            
            xg,yg,zg,tg=to_seq(rec_idx,lat0,lon0,t0,t1)
            valid=[i for i in range(len(xg)) if not math.isnan(xg[i]) and not math.isnan(yg[i])]
            if len(valid)<2: continue
            
            xg=[xg[i] for i in valid]; yg=[yg[i] for i in valid]; zg=[zg[i] for i in valid]
            dx,dy,dz=deltas(xg,yg,zg)
            for a,b,c in zip(dx,dy,dz):
                sumx+=a; sumy+=b; sumz+=c
                sum2x+=a*a; sum2y+=b*b; sum2z+=c*c
                count+=1
    if count==0:
        return {"mean":{"dx":0,"dy":0,"dz":0},"std":{"dx":1,"dy":1,"dz":1},"count":0}
        
    meanx=sumx/count; meany=sumy/count; meanz=sumz/count
    stdx=math.sqrt(max(1e-12,(sum2x/count)-meanx**2))
    stdy=math.sqrt(max(1e-12,(sum2y/count)-meany**2))
    stdz=math.sqrt(max(1e-12,(sum2z/count)-meanz**2))
    return {"mean":{"dx":meanx,"dy":meany,"dz":meanz},
            "std":{"dx":stdx,"dy":stdy,"dz":stdz},
            "count":count}

# ==========================================================
# PASO 2: generar CSVs y manifest
# ==========================================================
def pass2_generate_csvs(stats):
    mean,std=stats["mean"],stats["std"]
    manifest=[]
    pasadas=[d for d in sorted(os.listdir(PRE_DIR)) if os.path.isdir(os.path.join(PRE_DIR,d))]
    for pasada in tqdm(pasadas, desc="Pass2: generar CSVs"):
        pdir=os.path.join(PRE_DIR,pasada)
        pattern_files=glob.glob(os.path.join(pdir,"*_pattern_aligned_resampled.gpx"))
        if not pattern_files:
            pattern_files=glob.glob(os.path.join(pdir,"*pattern*resampled.gpx"))
            
        if not pattern_files: continue
        
        trp_path=pattern_files[0]
        trp_pts=read_gpx_points(trp_path)
        if len(trp_pts)<2: continue
        
        lat0,lon0=trp_pts[0]["lat"],trp_pts[0]["lon"]
        trp_idx=build_time_index(trp_pts)
        pattern_name=os.path.splitext(os.path.basename(trp_path))[0]
        recs=[p for p in glob.glob(os.path.join(pdir,"*_resampled.gpx"))
              if os.path.basename(p)!=os.path.basename(trp_path)]
        for rp in recs:
            rec_name=os.path.splitext(os.path.basename(rp))[0]
            rec_pts=read_gpx_points(rp)
            if len(rec_pts)<2: continue
            
            rec_idx=build_time_index(rec_pts)
            t0,t1=common_time_range(trp_idx,rec_idx)
            if t0 is None: continue
            
            xp,yp,zp,tp=to_seq(trp_idx,lat0,lon0,t0,t1)
            xg,yg,zg,tg=to_seq(rec_idx,lat0,lon0,t0,t1)
            valid=[i for i in range(len(xg)) if not (math.isnan(xg[i]) or math.isnan(yg[i]) or math.isnan(xp[i]) or math.isnan(yp[i]))]
            if len(valid)<2: continue
            
            xp=[xp[i] for i in valid]; yp=[yp[i] for i in valid]; zp=[zp[i] for i in valid]; tp=[tp[i] for i in valid]
            xg=[xg[i] for i in valid]; yg=[yg[i] for i in valid]; zg=[zg[i] for i in valid]; tg=[tg[i] for i in valid]
            dxp,dyp,dzp=deltas(xp,yp,zp)
            dxg,dyg,dzg=deltas(xg,yg,zg)
            dxp=[norm(v,mean["dx"],std["dx"]) for v in dxp]
            dyp=[norm(v,mean["dy"],std["dy"]) for v in dyp]
            dzp=[norm(v,mean["dz"],std["dz"]) for v in dzp]
            dxg=[norm(v,mean["dx"],std["dx"]) for v in dxg]
            dyg=[norm(v,mean["dy"],std["dy"]) for v in dyg]
            dzg=[norm(v,mean["dz"],std["dz"]) for v in dzg]
            n=len(tp)
            for k,(i0,i1,suf) in enumerate(window_indices(n,WINDOW_SIZE,STEP_SIZE),start=1):
                rows_lab=[[i-i0,dxp[i],dyp[i],dzp[i]] for i in range(i0,i1+1)]
                rows_slc=[[i-i0,dxg[i],dyg[i],dzg[i]] for i in range(i0,i1+1)]
                rows_lab,mask_lab=pad(rows_lab,WINDOW_SIZE)
                rows_slc,mask_slc=pad(rows_slc,WINDOW_SIZE)
                for i in range(WINDOW_SIZE):
                    rows_lab[i][0]=i; rows_slc[i][0]=i
                tag=f"{k}{suf}"
                label_fn=f"{pattern_name}_{tag}.csv"
                slice_fn=f"{rec_name}_{tag}.csv"
                label_path=os.path.join(LABELS_DIR,label_fn)
                slice_path=os.path.join(SLICES_DIR,slice_fn)
                mask_path =os.path.join(MASKS_DIR,slice_fn)
                save_csv(label_path,["time","dx","dy","dz"],rows_lab)
                save_csv(slice_path,["time","dx","dy","dz"],rows_slc)
                save_csv(mask_path,["mask"],[[m] for m in mask_slc])
                manifest.append([
                    pasada, rec_name, pattern_name, tag,
                    tp[i0], tp[min(i1,len(tp)-1)],
                    slice_path, label_path, mask_path,
                    len(rows_slc)
                ])
                
    # escribir manifest
    with open(MANIFEST_PATH,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["pasada","grabacion","pattern","window_id","t_start","t_end",
                    "slice_path","label_path","mask_path","n_points"])
        w.writerows(manifest)

# ==========================================================
# MAIN
# ==========================================================
def main():
    ensure_dirs()
    
    # Paso 1: estadísticas globales
    stats = pass1_stats()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR,"norm_stats.json"),"w",encoding="utf-8") as f:
        json.dump(stats,f,indent=2)
    print("Norm stats:", stats)
    
    # Paso 2: generación
    pass2_generate_csvs(stats)
    print(f"✅ Dataset generado en {OUT_DIR}")

if __name__ == "__main__":
    main()
