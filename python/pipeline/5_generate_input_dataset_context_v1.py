#!/usr/bin/env python3
"""
Generate a contextual ML dataset for residual GPS correction.

This script keeps the same train/val/test split logic as the v2 dataset
generator, but expands each input timestep with features derived only from the
noisy recording. Labels remain clean pattern deltas: dx, dy, dz.
"""

import argparse
import csv
import glob
import json
import math
import os
from datetime import timezone
from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PRE_DIR = os.path.join("data", "preprocessed")
R_EARTH = 6371000.0

BASE_FEATURES = ["dx", "dy", "dz"]
CONTEXT_FEATURES = [
    "xy_speed",
    "heading_cos",
    "heading_sin",
    "slope_clipped",
    "rolling_xy_speed_mean_30",
    "rolling_xy_speed_std_30",
    "rolling_pause_ratio_30",
    "rolling_turn_abs_mean_30",
    "rolling_tortuosity_30",
    "rolling_xy_speed_mean_180",
    "rolling_pause_ratio_180",
    "rolling_tortuosity_180",
]
INPUT_FEATURES = BASE_FEATURES + CONTEXT_FEATURES
LABEL_FEATURES = BASE_FEATURES
EPS = 1e-6


def deg2rad(value):
    """Convert degrees to radians."""
    return value * math.pi / 180.0


def to_local_xy(lat0, lon0, lat, lon):
    """Project lat/lon to local equirectangular x/y meters."""
    lat0r = deg2rad(lat0)
    x = deg2rad(lon - lon0) * math.cos(lat0r) * R_EARTH
    y = deg2rad(lat - lat0) * R_EARTH
    return x, y


def read_gpx_points(path):
    """Read GPX trackpoints as dictionaries with lat, lon, ele, and time."""
    with open(path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    points = []
    if not gpx.tracks or not gpx.tracks[0].segments:
        return points

    for point in gpx.tracks[0].segments[0].points:
        if point.time is None:
            continue
        point_time = point.time
        if point_time.tzinfo is None:
            point_time = point_time.replace(tzinfo=timezone.utc)
        elevation = point.elevation if point.elevation is not None else 0.0
        points.append(
            {
                "lat": point.latitude,
                "lon": point.longitude,
                "ele": elevation,
                "time": point_time,
            }
        )
    return points


def build_time_index(points):
    """Map Unix second to GPX point."""
    return {int(point["time"].timestamp()): point for point in points}


def common_time_range(a, b):
    """Return the common inclusive Unix-second time range."""
    if not a or not b:
        return None, None
    t0 = max(min(a), min(b))
    t1 = min(max(a), max(b))
    return (t0, t1) if (t1 - t0) >= 1 else (None, None)


def to_seq(index, lat0, lon0, t0, t1, use_z=True):
    """Convert a time-indexed GPX track to x/y/z/t sequences at 1 Hz."""
    xs, ys, zs, ts = [], [], [], []
    last = None
    for t in range(t0, t1 + 1):
        point = index.get(t, last)
        if point is None:
            xs.append(math.nan)
            ys.append(math.nan)
            zs.append(0.0)
            ts.append(t)
            continue
        x, y = to_local_xy(lat0, lon0, point["lat"], point["lon"])
        z = point["ele"] if use_z and point["ele"] is not None else 0.0
        xs.append(x)
        ys.append(y)
        zs.append(z)
        ts.append(t)
        last = point
    return xs, ys, zs, ts


def deltas(xs, ys, zs):
    """Compute consecutive x/y/z deltas."""
    n = len(xs)
    dx = [0.0] * n
    dy = [0.0] * n
    dz = [0.0] * n
    for i in range(1, n):
        dx[i] = xs[i] - xs[i - 1]
        dy[i] = ys[i] - ys[i - 1]
        dz[i] = zs[i] - zs[i - 1]
    return dx, dy, dz


def window_indices(n, win, step):
    """Return overlapping window index ranges."""
    out = []
    k = 1
    start = 0
    while start < n:
        end = min(n - 1, start + win - 1)
        suffix = "" if (k % 2) == 1 else "a"
        out.append((start, end, suffix))
        if end == n - 1:
            break
        start += step
        k += 1
    return out


def load_metadata(meta_path):
    """Load pass metadata indexed by pass id."""
    df = pd.read_csv(meta_path)
    required_cols = {"pasada", "modalidad", "set"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing required columns: {sorted(missing)}")

    meta_dict = {}
    for _, row in df.iterrows():
        pasada = str(row["pasada"])
        meta_dict[pasada] = {"modalidad": row["modalidad"], "set": row["set"]}
    return meta_dict


def create_family_groups(meta_dict):
    """Group related pass variants by their numeric family id."""
    family_groups = {}
    for pasada in meta_dict:
        base_num = "".join(filter(str.isdigit, str(pasada)))
        family_groups.setdefault(base_num, []).append(pasada)
    for family_base in family_groups:
        family_groups[family_base] = sorted(family_groups[family_base])
    return family_groups


def validate_family_consistency(meta_dict, family_groups):
    """Validate that pass variants from the same family stay in one split."""
    inconsistent = []
    for family_base, pasadas in family_groups.items():
        if len(pasadas) <= 1:
            continue
        family_sets = {meta_dict[pasada]["set"] for pasada in pasadas}
        if len(family_sets) <= 1:
            continue
        set_details = {}
        for pasada in pasadas:
            set_name = meta_dict[pasada]["set"]
            set_details.setdefault(set_name, []).append(pasada)
        inconsistent.append(
            {
                "family": family_base,
                "pasadas": pasadas,
                "sets_found": family_sets,
                "set_details": set_details,
            }
        )
    return inconsistent


def get_eligible_pasadas(meta_dict, target_set):
    """Return passes assigned to a target split."""
    return sorted([pasada for pasada, info in meta_dict.items() if info["set"] == target_set])


def find_base_family_pattern(pasada, family_groups):
    """Return the base pass that stores the shared family pattern."""
    pasada_str = str(pasada)
    base_num = "".join(filter(str.isdigit, pasada_str))
    if base_num not in family_groups:
        return pasada
    for member in family_groups[base_num]:
        if member == base_num:
            return member
    return family_groups[base_num][0]


def rolling_mean(values, window):
    """Return centered rolling mean with edge min periods."""
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def rolling_std(values, window):
    """Return centered rolling standard deviation with edge min periods."""
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .std(ddof=0)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def rolling_sum(values, window):
    """Return centered rolling sum with edge min periods."""
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .sum()
        .to_numpy(dtype=np.float64)
    )


def rolling_tortuosity(x, y, xy_speed, window):
    """Compute local path length divided by net displacement."""
    n = len(x)
    half = max(1, window // 2)
    path_len = rolling_sum(xy_speed, window)
    out = np.zeros(n, dtype=np.float64)

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n - 1, i + half)
        net = math.hypot(float(x[i1] - x[i0]), float(y[i1] - y[i0]))
        out[i] = path_len[i] / max(net, 1.0)

    return np.clip(out, 1.0, 20.0)


def build_feature_frame(dx, dy, dz, x=None, y=None, pause_threshold=0.5):
    """
    Build normalized-ready input features from raw meter deltas.

    All features are computed only from the noisy recording. If absolute x/y are
    not supplied, they are reconstructed from the deltas.
    """
    dx = np.asarray(dx, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)

    if x is None:
        x = np.cumsum(dx)
    else:
        x = np.asarray(x, dtype=np.float64)
    if y is None:
        y = np.cumsum(dy)
    else:
        y = np.asarray(y, dtype=np.float64)

    xy_speed = np.hypot(dx, dy)
    heading_cos = np.divide(dx, xy_speed + EPS)
    heading_sin = np.divide(dy, xy_speed + EPS)
    slope_clipped = np.clip(np.divide(dz, xy_speed + EPS), -2.0, 2.0)

    prev_cos = np.roll(heading_cos, 1)
    prev_sin = np.roll(heading_sin, 1)
    prev_cos[0] = heading_cos[0]
    prev_sin[0] = heading_sin[0]
    cross = prev_cos * heading_sin - prev_sin * heading_cos
    dot = prev_cos * heading_cos + prev_sin * heading_sin
    turn_abs = np.abs(np.arctan2(cross, dot))
    turn_abs[xy_speed < EPS] = 0.0

    pause = (xy_speed < pause_threshold).astype(np.float64)

    data = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "xy_speed": xy_speed,
        "heading_cos": heading_cos,
        "heading_sin": heading_sin,
        "slope_clipped": slope_clipped,
        "rolling_xy_speed_mean_30": rolling_mean(xy_speed, 30),
        "rolling_xy_speed_std_30": rolling_std(xy_speed, 30),
        "rolling_pause_ratio_30": rolling_mean(pause, 30),
        "rolling_turn_abs_mean_30": rolling_mean(turn_abs, 30),
        "rolling_tortuosity_30": rolling_tortuosity(x, y, xy_speed, 30),
        "rolling_xy_speed_mean_180": rolling_mean(xy_speed, 180),
        "rolling_pause_ratio_180": rolling_mean(pause, 180),
        "rolling_tortuosity_180": rolling_tortuosity(x, y, xy_speed, 180),
    }
    return pd.DataFrame(data, columns=INPUT_FEATURES)


def normalize_frame(frame, stats, columns):
    """Normalize selected frame columns with train statistics."""
    out = frame.copy()
    for col in columns:
        std = stats["std"].get(col, 1.0)
        mean = stats["mean"].get(col, 0.0)
        out[col] = 0.0 if std <= 1e-12 else (out[col] - mean) / std
    return out


def norm_value(value, stats, col):
    """Normalize one scalar value."""
    std = stats["std"].get(col, 1.0)
    mean = stats["mean"].get(col, 0.0)
    return 0.0 if std <= 1e-12 else (value - mean) / std


def pad_rows(rows, win):
    """Pad rows with zeros while preserving the time column shape."""
    if len(rows) >= win:
        return rows[:win], [1] * win

    width = len(rows[0]) if rows else len(INPUT_FEATURES) + 1
    last_time = rows[-1][0] if rows else 0
    padded = list(rows)
    for i in range(win - len(rows)):
        padded.append([last_time + i + 1] + [0.0] * (width - 1))
    mask = [1] * len(rows) + [0] * (win - len(rows))
    return padded, mask


def save_csv(path, header, rows):
    """Write a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def calculate_train_stats(eligible_pasadas, use_z=True):
    """Calculate normalization statistics from train noisy recordings only."""
    sums = {col: 0.0 for col in INPUT_FEATURES}
    sums2 = {col: 0.0 for col in INPUT_FEATURES}
    count = 0

    print(f"Calculating contextual train stats from {len(eligible_pasadas)} passes...")

    for pasada in tqdm(eligible_pasadas, desc="Context stats"):
        pdir = os.path.join(PRE_DIR, pasada)
        if not os.path.isdir(pdir):
            print(f"Warning: pass {pasada} not found in {pdir}")
            continue

        pattern_files = glob.glob(os.path.join(pdir, "*_pattern_aligned_resampled.gpx"))
        if not pattern_files:
            pattern_files = glob.glob(os.path.join(pdir, "*pattern*resampled.gpx"))
        if not pattern_files:
            print(f"Warning: no pattern found in {pdir}")
            continue

        trp_pts = read_gpx_points(pattern_files[0])
        if len(trp_pts) < 2:
            continue

        lat0, lon0 = trp_pts[0]["lat"], trp_pts[0]["lon"]
        trp_idx = build_time_index(trp_pts)
        recs = [
            p
            for p in glob.glob(os.path.join(pdir, "*_resampled.gpx"))
            if os.path.basename(p) != os.path.basename(pattern_files[0])
        ]

        for rp in recs:
            rec_pts = read_gpx_points(rp)
            if len(rec_pts) < 2:
                continue
            rec_idx = build_time_index(rec_pts)
            t0, t1 = common_time_range(trp_idx, rec_idx)
            if t0 is None:
                continue

            xg, yg, zg, _ = to_seq(rec_idx, lat0, lon0, t0, t1, use_z)
            valid = [i for i in range(len(xg)) if not (math.isnan(xg[i]) or math.isnan(yg[i]))]
            if len(valid) < 2:
                continue

            xg = [xg[i] for i in valid]
            yg = [yg[i] for i in valid]
            zg = [zg[i] for i in valid]
            dxg, dyg, dzg = deltas(xg, yg, zg)
            features = build_feature_frame(dxg, dyg, dzg, xg, yg)

            for col in INPUT_FEATURES:
                values = features[col].to_numpy(dtype=np.float64)
                sums[col] += float(values.sum())
                sums2[col] += float(np.square(values).sum())
            count += len(features)

    if count == 0:
        print("Warning: no valid data found for train statistics")
        return {
            "input_features": INPUT_FEATURES,
            "label_features": LABEL_FEATURES,
            "mean": {col: 0.0 for col in INPUT_FEATURES},
            "std": {col: 1.0 for col in INPUT_FEATURES},
            "count": 0,
        }

    means = {col: sums[col] / count for col in INPUT_FEATURES}
    stds = {
        col: math.sqrt(max(1e-12, (sums2[col] / count) - means[col] ** 2))
        for col in INPUT_FEATURES
    }
    return {
        "input_features": INPUT_FEATURES,
        "label_features": LABEL_FEATURES,
        "mean": means,
        "std": stds,
        "count": count,
    }


def generate_csvs_for_set(eligible_pasadas, meta_dict, stats, args, family_groups):
    """Generate contextual window CSV files for one split."""
    manifest = []
    set_dir = os.path.join(args.out, args.set)
    slices_dir = os.path.join(set_dir, "slices")
    labels_dir = os.path.join(set_dir, "labels")
    masks_dir = os.path.join(set_dir, "masks")

    for d in [slices_dir, labels_dir, masks_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Generating contextual {args.set.upper()} with {len(eligible_pasadas)} passes...")

    for pasada in tqdm(eligible_pasadas, desc=f"Context {args.set}"):
        pdir = os.path.join(PRE_DIR, pasada)
        if not os.path.isdir(pdir):
            continue

        pattern_pasada = find_base_family_pattern(pasada, family_groups)
        pattern_dir = os.path.join(PRE_DIR, pattern_pasada)
        pattern_files = glob.glob(os.path.join(pattern_dir, "*_pattern_aligned_resampled.gpx"))
        if not pattern_files:
            pattern_files = glob.glob(os.path.join(pattern_dir, "*pattern*resampled.gpx"))
        if not pattern_files:
            print(f"Warning: no pattern for base family {pattern_pasada} (pass {pasada})")
            continue
        if pattern_pasada != pasada:
            print(f"Pass {pasada} uses base-family pattern {pattern_pasada}")

        trp_path = pattern_files[0]
        trp_pts = read_gpx_points(trp_path)
        if len(trp_pts) < 2:
            continue

        lat0, lon0 = trp_pts[0]["lat"], trp_pts[0]["lon"]
        trp_idx = build_time_index(trp_pts)
        pattern_name = os.path.splitext(os.path.basename(trp_path))[0]

        recs = [
            p
            for p in glob.glob(os.path.join(pdir, "*_resampled.gpx"))
            if os.path.basename(p) != os.path.basename(trp_path)
        ]

        for rp in recs:
            rec_name = os.path.splitext(os.path.basename(rp))[0]
            rec_pts = read_gpx_points(rp)
            if len(rec_pts) < 2:
                continue

            rec_idx = build_time_index(rec_pts)
            t0, t1 = common_time_range(trp_idx, rec_idx)
            if t0 is None:
                continue

            xp, yp, zp, tp = to_seq(trp_idx, lat0, lon0, t0, t1, args.use_z)
            xg, yg, zg, _ = to_seq(rec_idx, lat0, lon0, t0, t1, args.use_z)

            valid = [
                i
                for i in range(len(xg))
                if not (math.isnan(xg[i]) or math.isnan(yg[i]) or math.isnan(xp[i]) or math.isnan(yp[i]))
            ]
            if len(valid) < 2:
                continue

            xp = [xp[i] for i in valid]
            yp = [yp[i] for i in valid]
            zp = [zp[i] for i in valid]
            tp = [tp[i] for i in valid]
            xg = [xg[i] for i in valid]
            yg = [yg[i] for i in valid]
            zg = [zg[i] for i in valid]

            dxp, dyp, dzp = deltas(xp, yp, zp)
            dxg, dyg, dzg = deltas(xg, yg, zg)
            input_features = normalize_frame(build_feature_frame(dxg, dyg, dzg, xg, yg), stats, INPUT_FEATURES)

            label_frame = pd.DataFrame(
                {
                    "dx": [norm_value(v, stats, "dx") for v in dxp],
                    "dy": [norm_value(v, stats, "dy") for v in dyp],
                    "dz": [norm_value(v, stats, "dz") for v in dzp],
                },
                columns=LABEL_FEATURES,
            )

            n = len(tp)
            for k, (i0, i1, suf) in enumerate(window_indices(n, args.win, args.step), start=1):
                rows_lab = [
                    [i - i0] + [label_frame.iloc[i][col] for col in LABEL_FEATURES]
                    for i in range(i0, i1 + 1)
                ]
                rows_slc = [
                    [i - i0] + [input_features.iloc[i][col] for col in INPUT_FEATURES]
                    for i in range(i0, i1 + 1)
                ]

                rows_lab, _ = pad_rows(rows_lab, args.win)
                rows_slc, mask_slc = pad_rows(rows_slc, args.win)

                for i in range(args.win):
                    rows_lab[i][0] = i
                    rows_slc[i][0] = i

                tag = f"{k}{suf}"
                label_fn = f"{pattern_name}_{tag}.csv"
                slice_fn = f"{rec_name}_{tag}.csv"
                label_path = os.path.join(labels_dir, label_fn)
                slice_path = os.path.join(slices_dir, slice_fn)
                mask_path = os.path.join(masks_dir, slice_fn)

                save_csv(label_path, ["time"] + LABEL_FEATURES, rows_lab)
                save_csv(slice_path, ["time"] + INPUT_FEATURES, rows_slc)
                save_csv(mask_path, ["mask"], [[m] for m in mask_slc])

                modalidad = meta_dict.get(pasada, {}).get("modalidad", "unknown")
                manifest.append(
                    [
                        pasada,
                        modalidad,
                        args.set,
                        rec_name,
                        pattern_name,
                        tag,
                        tp[i0],
                        tp[min(i1, len(tp) - 1)],
                        slice_path,
                        label_path,
                        mask_path,
                        len(rows_slc),
                    ]
                )

    manifest_path = os.path.join(set_dir, f"manifest_{args.set}.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pasada",
                "modalidad",
                "set",
                "grabacion",
                "pattern",
                "window_id",
                "t_start",
                "t_end",
                "slice_path",
                "label_path",
                "mask_path",
                "n_points",
            ]
        )
        writer.writerows(manifest)

    return len(manifest)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate contextual ML dataset")
    parser.add_argument("--meta", required=True, help="Pass metadata CSV")
    parser.add_argument("--set", choices=["train", "val", "test"], required=True, help="Split to generate")
    parser.add_argument("--out", default="data/input_context_v1", help="Output dataset root")
    parser.add_argument("--win", type=int, default=3600, help="Window size in seconds")
    parser.add_argument("--step", type=int, default=1800, help="Window step in seconds")
    parser.add_argument("--use_z", type=bool, default=True, help="Use elevation")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== GENERATE CONTEXT DATASET V1 ===")
    print(f"Metadata: {args.meta}")
    print(f"Output: {args.out}")
    print(f"Split: {args.set}")
    print(f"Window: {args.win}s, step: {args.step}s")
    print(f"Input features: {INPUT_FEATURES}")

    try:
        meta_dict = load_metadata(args.meta)
    except Exception as exc:
        print(f"ERROR: could not load metadata: {exc}")
        return 1

    family_groups = create_family_groups(meta_dict)
    inconsistencies = validate_family_consistency(meta_dict, family_groups)
    if inconsistencies:
        print("ERROR: family split inconsistencies found")
        for item in inconsistencies:
            print(f"  - Family {item['family']}: {item['set_details']}")
        return 1

    eligible_pasadas = get_eligible_pasadas(meta_dict, args.set)
    if not eligible_pasadas:
        print(f"ERROR: no passes found for split {args.set}")
        return 1

    stats_path = os.path.join(args.out, "norm_stats_train.json")
    if args.set == "train":
        stats = calculate_train_stats(eligible_pasadas, args.use_z)
        os.makedirs(args.out, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved train stats to {stats_path}")
    else:
        if not os.path.exists(stats_path):
            print(f"ERROR: missing train stats at {stats_path}. Generate train first.")
            return 1
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        missing = [col for col in INPUT_FEATURES if col not in stats.get("mean", {})]
        if missing:
            print(f"ERROR: stats file is missing contextual features: {missing}")
            return 1

    n_windows = generate_csvs_for_set(eligible_pasadas, meta_dict, stats, args, family_groups)
    print(f"Generated {n_windows} {args.set} windows in {os.path.join(args.out, args.set)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
