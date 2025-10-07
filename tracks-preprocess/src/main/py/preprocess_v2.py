#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, glob, time, bisect
from datetime import datetime, timedelta, timezone
import gpxpy, gpxpy.gpx
from tqdm import tqdm

RAW_DIR = os.path.join("data", "raw")
PRE_DIR = os.path.join("data", "preprocessed")

# Parámetros del mapeo y robustez
MAX_PROJ_DIST = 30.0     # m: descartar segundos cuya proyección al patrón esté demasiado lejos
SEARCH_BACK = 20         # nº de segmentos hacia atrás a inspeccionar en la búsqueda progresiva
SEARCH_AHEAD = 80        # nº de segmentos hacia delante
R_EARTH = 6371000.0

# ---------------- Geometría básica ----------------
def deg2rad(d): return d * math.pi / 180.0

def to_xy(lat0, lon0, lat, lon):
    lat0r = deg2rad(lat0)
    x = deg2rad(lon - lon0) * math.cos(lat0r) * R_EARTH
    y = deg2rad(lat - lat0) * R_EARTH
    return x, y

def proj_on_segment(px, py, ax, ay, bx, by, clamp=True):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx*vx + vy*vy
    if vv == 0.0:
        u = 0.0
        qx, qy = ax, ay
    else:
        u = (wx*vx + wy*vy) / vv
        if clamp:
            u = max(0.0, min(1.0, u))
        qx, qy = ax + u*vx, ay + u*vy
    dist = math.hypot(px - qx, py - qy)
    return u, qx, qy, dist

def cumdist(xs, ys):
    seglen = []
    for i in range(len(xs) - 1):
        seglen.append(math.hypot(xs[i+1] - xs[i], ys[i+1] - ys[i]))
    s = [0.0]
    for L in seglen:
        s.append(s[-1] + L)
    return seglen, s  # len(seglen)=n-1, len(s)=n

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
        while i+1 < len(pts) and pts[i+1]["time"] < t:
            i += 1
        if i+1 < len(pts) and pts[i]["time"] <= t <= pts[i+1]["time"]:
            res.append(interpolate_point(pts[i], pts[i+1], t))
        else:
            nearest = pts[0] if t < pts[0]["time"] else pts[-1]
            res.append({"lat": nearest["lat"], "lon": nearest["lon"], "ele": nearest["ele"], "time": t})
        t += timedelta(seconds=1)
    return res

# ---------------- Isotonic regression (PAV) ----------------
def isotonic_increasing(y, w=None):
    """
    Ajusta y_hat no-decreciente que minimiza sum w*(y_hat - y)^2.
    y: lista de floats (p.ej., tiempos epoch)
    w: lista de pesos >0 (opcional)
    """
    n = len(y)
    if n == 0:
        return []
    if w is None:
        w = [1.0]*n
    # PAV en pila de bloques: cada bloque = (w_sum, y_sum, count)
    blocks = []
    for i in range(n):
        blocks.append([w[i], w[i]*y[i], 1])
        # fusiona mientras viole isotonicidad
        while len(blocks) >= 2:
            w2, s2, c2 = blocks[-1]
            w1, s1, c1 = blocks[-2]
            if (s1 / w1) <= (s2 / w2):
                break
            # viola: fusionar
            blocks.pop()
            blocks.pop()
            blocks.append([w1+w2, s1+s2, c1+c2])
    # expandir
    y_hat = []
    for w_sum, s_sum, count in blocks:
        v = s_sum / w_sum
        y_hat.extend([v]*count)
    return y_hat

# ---------------- Proyección progresiva al patrón ----------------
def build_pattern_geometry(trp_pts):
    lat0, lon0 = trp_pts[0]["lat"], trp_pts[0]["lon"]
    px = []; py = []
    for p in trp_pts:
        x,y = to_xy(lat0, lon0, p["lat"], p["lon"])
        px.append(x); py.append(y)
    seglen, S = cumdist(px, py)  # S: curvilínea de vértices del patrón
    return lat0, lon0, px, py, seglen, S

def closest_vertex_index(px, py, X, Y):
    best_i, best_d = 0, float("inf")
    for i in range(len(X)):
        d = (X[i]-px)**2 + (Y[i]-py)**2
        if d < best_d:
            best_d, best_i = d, i
    return best_i

def project_points_to_pattern(resampled_pts, lat0, lon0, X, Y, seglen, S):
    """
    Para cada punto (1 Hz) de una grabación, devuelve listas:
    s_list (m a lo largo del patrón), t_list (epoch), d_list (m).
    Usa búsqueda progresiva de mejor segmento con ventana local.
    """
    if len(X) < 2:
        return [], [], []
    # Precalcular por eficiencia
    nseg = len(seglen)
    # Primer punto: arrancar cerca del vértice más próximo
    p0 = resampled_pts[0]
    p0x, p0y = to_xy(lat0, lon0, p0["lat"], p0["lon"])
    j = max(0, min(nseg-1, closest_vertex_index(p0x, p0y, X, Y)-1))
    s_list, t_list, d_list = [], [], []

    for p in resampled_pts:
        t = p["time"].timestamp()
        px_, py_ = to_xy(lat0, lon0, p["lat"], p["lon"])
        best = (float("inf"), j, 0.0)  # (dist, seg_idx, u)
        i0 = max(0, j - SEARCH_BACK)
        i1 = min(nseg-1, j + SEARCH_AHEAD)
        for i in range(i0, i1+1):
            ax, ay = X[i], Y[i]
            bx, by = X[i+1], Y[i+1]
            u, qx, qy, dist = proj_on_segment(px_, py_, ax, ay, bx, by, clamp=True)
            if dist < best[0]:
                best = (dist, i, u)
        dist, j, u = best
        s = S[j] + u * seglen[j]
        s_list.append(s)
        t_list.append(t)
        d_list.append(dist)
        # Avanzar la ventana favoreciendo movimiento hacia delante
        # (el índice j ya se queda en el mejor segmento)
    return s_list, t_list, d_list

def weighted_mask(s_list, t_list, d_list, max_dist=MAX_PROJ_DIST):
    s2, t2, w2 = [], [], []
    for s, t, d in zip(s_list, t_list, d_list):
        if d <= max_dist:
            # peso suave, mayor si más cerca (escala ~ 10 m)
            w = 1.0 / (1.0 + (d/10.0)**2)
            s2.append(s); t2.append(t); w2.append(w)
    return s2, t2, w2

def fit_monotone_t_of_s(s_list, t_list, w_list):
    """
    Ajusta t_hat(s) monótona.
    Ordena por s, aplica isotónica sobre t(s).
    Devuelve s_sorted, t_hat_sorted.
    """
    if not s_list:
        return [], []
    order = sorted(range(len(s_list)), key=lambda i: s_list[i])
    s_sorted = [s_list[i] for i in order]
    t_sorted = [t_list[i] for i in order]
    w_sorted = [w_list[i] for i in order]
    t_hat = isotonic_increasing(t_sorted, w_sorted)
    return s_sorted, t_hat

def eval_t_of_s(s_sorted, t_hat, s_query):
    if not s_sorted:
        return None
    i = bisect.bisect_left(s_sorted, s_query)
    if i == 0:
        return None  # fuera por la izquierda: sin extrapolación
    if i == len(s_sorted):
        return None  # fuera por la derecha
    s0, s1 = s_sorted[i-1], s_sorted[i]
    t0, t1 = t_hat[i-1], t_hat[i]
    if s1 == s0:
        return t0
    a = (s_query - s0) / (s1 - s0)
    return t0 + a*(t1 - t0)

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

    # ---- Cargar geometría del patrón (sin tiempos) ----
    trp_pts = read_gpx_points(trp_path)
    if len(trp_pts) < 2:
        print(f"[{base}] ❌ Patrón insuficiente.")
        return
    lat0, lon0, PX, PY, seglen, S_vertices = build_pattern_geometry(trp_pts)

    # Curvilínea de cada punto del patrón (en vértices)
    S_pp = []
    # Si el patrón tiene muchos puntos, usamos la distancia acumulada exacta por vértices
    # (ya calculada en S_vertices); si quieres la curvilínea por cada punto (que ya coincide
    # con los vértices), es S_vertices:
    S_pp = S_vertices[:]  # len == len(trp_pts)

    # ---- Resample y guardar grabaciones ----
    resampled_tracks = []
    for rp in rec_paths:
        pts = read_gpx_points(rp)
        pts = [p for p in pts if p["time"] is not None]
        if len(pts) < 2:
            print(f"[{base}] ⚠️ {os.path.basename(rp)}: menos de 2 puntos con tiempo.")
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

    # ---- Para cada grabación: t(s) monótona ----
    mappings = []   # lista de (s_sorted, t_hat)
    coverages = []  # cobertura en % para diagnosticar
    for idx, tr in enumerate(resampled_tracks, 1):
        s_list, t_list, d_list = project_points_to_pattern(tr, lat0, lon0, PX, PY, seglen, S_vertices)
        s_w, t_w, w_w = weighted_mask(s_list, t_list, d_list, MAX_PROJ_DIST)
        s_sorted, t_hat = fit_monotone_t_of_s(s_w, t_w, w_w)
        mappings.append((s_sorted, t_hat))
        cov = 100.0 * (len(s_sorted) / max(1, len(s_list)))
        coverages.append(cov)
        print(f"[{base}] 📈 Mapeo {idx}: {len(s_sorted)} muestras útiles ({cov:.1f}% cobertura)")

    # ---- Evaluar tiempos para cada punto del patrón (mediana entre grabaciones) ----
    def median(vals):
        vals = sorted(vals)
        n = len(vals)
        if n == 0: return None
        if n % 2 == 1: return vals[n//2]
        return 0.5*(vals[n//2 - 1] + vals[n//2])

    t_epoch_pp = []
    print(f"[{base}] ⏱ Calculando tiempos del patrón (mediana entre {len(mappings)} mapas)...")
    for s in tqdm(S_pp, ncols=80, desc=f"{base} t(s)"):
        cand = []
        for (s_sorted, t_hat) in mappings:
            t = eval_t_of_s(s_sorted, t_hat, s)
            if t is not None:
                cand.append(t)
        t_epoch_pp.append(median(cand))

    # ---- Relleno/interpolación interna y última pasada isotónica ----
    # Interpolar huecos internos por s (si hay None entre valores válidos)
    # Construimos arrays de s y t con válidos
    valid = [(s, t) for s, t in zip(S_pp, t_epoch_pp) if t is not None]
    if len(valid) >= 2:
        s_valid = [s for s,_ in valid]
        t_valid = [t for _,t in valid]
        # Para cada None, interpola entre vecinos válidos
        for i, (s, t) in enumerate(zip(S_pp, t_epoch_pp)):
            if t is None:
                j = bisect.bisect_left(s_valid, s)
                if 0 < j < len(s_valid):
                    s0, s1 = s_valid[j-1], s_valid[j]
                    t0, t1 = t_valid[j-1], t_valid[j]
                    a = 0.0 if s1==s0 else (s - s0)/(s1 - s0)
                    t_epoch_pp[i] = t0 + a*(t1 - t0)
        # Para extremos sin cobertura, extrapola linealmente con la pendiente más cercana
        # (opcional) — si siguen quedando None:
        first_defined = next((i for i,t in enumerate(t_epoch_pp) if t is not None), None)
        last_defined  = next((i for i in range(len(t_epoch_pp)-1, -1, -1) if t_epoch_pp[i] is not None), None)
        if first_defined is not None and last_defined is not None:
            # Extrapola a la izquierda
            if first_defined+1 <= last_defined:
                s0, t0 = S_pp[first_defined], t_epoch_pp[first_defined]
                s1, t1 = S_pp[first_defined+1], t_epoch_pp[first_defined+1]
                slope = 0.0 if s1==s0 else (t1 - t0)/(s1 - s0)
                for i in range(first_defined-1, -1, -1):
                    t_epoch_pp[i] = t0 - slope*(s0 - S_pp[i])
            # Extrapola a la derecha
            if last_defined-1 >= 0:
                s0, t0 = S_pp[last_defined-1], t_epoch_pp[last_defined-1]
                s1, t1 = S_pp[last_defined], t_epoch_pp[last_defined]
                slope = 0.0 if s1==s0 else (t1 - t0)/(s1 - s0)
                for i in range(last_defined+1, len(S_pp)):
                    t_epoch_pp[i] = t1 + slope*(S_pp[i] - s1)

    # Última pasada isotónica global (garantiza no-decreciente)
    t_vals = [t if t is not None else float("nan") for t in t_epoch_pp]
    # Rellena NaN (si quedara alguno) con interpolación simple por índice
    # para poder aplicar isotónica; en la práctica con la extrapolación previa no deberían quedar.
    if any(math.isnan(v) for v in t_vals):
        # fallback simple: copia vecino más cercano definido
        for i in range(len(t_vals)):
            if math.isnan(t_vals[i]):
                # busca anterior
                j = i-1
                while j >=0 and math.isnan(t_vals[j]): j -= 1
                k = i+1
                while k < len(t_vals) and math.isnan(t_vals[k]): k += 1
                if (j >= 0 and k < len(t_vals)):
                    t_vals[i] = 0.5*(t_vals[j]+t_vals[k])
                elif j >= 0:
                    t_vals[i] = t_vals[j]
                elif k < len(t_vals):
                    t_vals[i] = t_vals[k]
                else:
                    t_vals[i] = 0.0
    # Ajuste isotónico final sobre t(s)
    t_hat_final = isotonic_increasing(t_vals, [1.0]*len(t_vals))
    # Construir puntos alineados
    aligned = []
    for p, te in zip(trp_pts, t_hat_final):
        aligned.append({
            "lat": p["lat"], "lon": p["lon"], "ele": p.get("ele"),
            "time": datetime.fromtimestamp(te, tz=timezone.utc)
        })

    # Guardar patrón alineado
    pattern_name = os.path.splitext(os.path.basename(trp_path))[0]
    out_pattern = os.path.join(out_dir, f"{pattern_name}_aligned.gpx")
    write_gpx_points(out_pattern, f"{pattern_name}_aligned", aligned)

    elapsed = time.time() - t_start
    print(f"[{base}] 🎯 Patrón alineado guardado: {os.path.relpath(out_pattern)} "
          f"({len(aligned)} pts). Tiempo: {elapsed:.2f}s")

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
