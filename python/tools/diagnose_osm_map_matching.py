#!/usr/bin/env python3
"""
Run an isolated OpenStreetMap map-matching diagnostic for one GPS recording.

The clean pattern is used only after matching to evaluate the result. Matching
itself receives the raw GPX and an OpenStreetMap walking network.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import gpxpy
import gpxpy.gpx
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
from pyproj import Geod, Transformer

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from gps_aberration_detector import DetectorConfig, detect_events
from diagnose_error_spectrum import (
    CHANNELS,
    add_raw_ratios,
    align_track_to_pattern,
    compute_spectrogram,
    parse_bands,
    parse_gpx,
    plot_band_ratio_summary,
    plot_recording,
    summarize_bands,
)


DEFAULT_BANDS = (
    "0:0.0005,0.0005:0.001,0.001:0.003,0.003:0.01,"
    "0.01:0.05,0.05:0.15,0.15:0.5"
)

MATCHER_SETTINGS = {
    "sample_step": 1,
    "max_dist_m": 150.0,
    "obs_noise_m": 50.0,
    "dist_noise_m": 20.0,
    "max_lattice_width": 10,
    "non_emitting_length_factor": 0.5,
    "match_enter_dist_m": 15.0,
    "match_exit_dist_m": 25.0,
    "match_enter_hold_s": 10.0,
    "match_exit_hold_s": 3.0,
    "min_match_segment_s": 15.0,
    "transition_blend_s": 10.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match one GPS recording to an OSM walking network and evaluate it",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pasada", default="7", help="Pass/family identifier")
    parser.add_argument(
        "--recording",
        default="7_2_1_Q_W_resampled",
        help="Recording stem inside data/preprocessed/<pass>",
    )
    parser.add_argument("--preprocessed_dir", default="data/preprocessed")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--network_type", default="walk")
    parser.add_argument("--map_margin_m", type=float, default=300.0)
    parser.add_argument("--sample_step", type=int, default=None)
    parser.add_argument("--max_dist_m", type=float, default=None)
    parser.add_argument("--obs_noise_m", type=float, default=None)
    parser.add_argument("--dist_noise_m", type=float, default=None)
    parser.add_argument("--max_lattice_width", type=int, default=None)
    parser.add_argument("--non_emitting_length_factor", type=float, default=None)
    parser.add_argument("--match_enter_dist_m", type=float, default=None)
    parser.add_argument("--match_exit_dist_m", type=float, default=None)
    parser.add_argument("--match_enter_hold_s", type=float, default=None)
    parser.add_argument("--match_exit_hold_s", type=float, default=None)
    parser.add_argument("--min_match_segment_s", type=float, default=None)
    parser.add_argument("--transition_blend_s", type=float, default=None)
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--bands", default=DEFAULT_BANDS)
    parser.add_argument("--refresh_map", action="store_true")
    parser.add_argument(
        "--reuse_hybrid_outputs",
        action="store_true",
        help="Reuse existing hybrid GPX and trace files, then rebuild diagnostics",
    )
    parser.add_argument(
        "--no_evaluation",
        action="store_true",
        help="Run without a clean pattern or evaluation metrics",
    )
    return parser.parse_args()


def resolve_settings(args):
    override_names = (
        "sample_step",
        "max_dist_m",
        "obs_noise_m",
        "dist_noise_m",
        "max_lattice_width",
        "non_emitting_length_factor",
        "match_enter_dist_m",
        "match_exit_dist_m",
        "match_enter_hold_s",
        "match_exit_hold_s",
        "min_match_segment_s",
        "transition_blend_s",
    )
    settings = MATCHER_SETTINGS.copy()
    for option in override_names:
        value = getattr(args, option)
        if value is not None:
            settings[option] = value
    if settings["sample_step"] < 1:
        raise ValueError("--sample_step must be at least 1")
    if not 0.0 < settings["non_emitting_length_factor"] <= 1.0:
        raise ValueError("--non_emitting_length_factor must be in (0, 1]")
    if settings["match_enter_dist_m"] > settings["match_exit_dist_m"]:
        raise ValueError(
            "--match_enter_dist_m must not exceed --match_exit_dist_m"
        )
    if settings["match_exit_dist_m"] > settings["max_dist_m"]:
        raise ValueError("--match_exit_dist_m must not exceed --max_dist_m")
    if settings["match_enter_hold_s"] < 0.0:
        raise ValueError("--match_enter_hold_s must not be negative")
    if settings["match_exit_hold_s"] < 0.0:
        raise ValueError("--match_exit_hold_s must not be negative")
    if settings["min_match_segment_s"] < 0.0:
        raise ValueError("--min_match_segment_s must not be negative")
    if settings["transition_blend_s"] < 0.0:
        raise ValueError("--transition_blend_s must not be negative")
    return settings


def read_recording(path):
    df = parse_gpx(path)
    if len(df) < 2:
        raise ValueError(f"Not enough points in {path}")
    return df


def calculate_bbox(track_df, margin_m):
    lat_mid = float(track_df["lat"].mean())
    lat_pad = margin_m / 111_320.0
    lon_scale = max(math.cos(math.radians(lat_mid)), 0.1)
    lon_pad = margin_m / (111_320.0 * lon_scale)
    return (
        float(track_df["lon"].min() - lon_pad),
        float(track_df["lat"].min() - lat_pad),
        float(track_df["lon"].max() + lon_pad),
        float(track_df["lat"].max() + lat_pad),
    )


def load_or_download_graph(track_df, cache_path, network_type, margin_m, refresh):
    bbox = calculate_bbox(track_df, margin_m)
    if cache_path.exists() and not refresh:
        print(f"Loading cached OSM graph: {cache_path}")
        graph = ox.io.load_graphml(cache_path)
    else:
        print(f"Downloading OSM {network_type} network for bbox {bbox}")
        graph = ox.graph.graph_from_bbox(
            bbox,
            network_type=network_type,
            simplify=False,
            retain_all=True,
            truncate_by_edge=True,
        )
        ox.io.save_graphml(graph, filepath=cache_path)
        print(f"Saved OSM graph cache: {cache_path}")
    return graph, bbox


def build_metric_graph(graph):
    graph_metric = ox.projection.project_graph(graph)
    if graph_metric.graph.get("crs") is None:
        raise ValueError("Projected OSM graph has no CRS")
    return graph_metric


def project_track(track_df, target_crs):
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    x_values, y_values = transformer.transform(
        track_df["lon"].to_numpy(dtype=np.float64),
        track_df["lat"].to_numpy(dtype=np.float64),
    )
    return np.asarray(x_values), np.asarray(y_values)


def graph_to_matcher_map(graph_metric):
    map_con = InMemMap(
        "osm_walk",
        use_latlon=False,
        use_rtree=True,
        index_edges=True,
    )
    for node_id, data in graph_metric.nodes(data=True):
        map_con.add_node(int(node_id), (float(data["y"]), float(data["x"])))

    seen_edges = set()
    for node_a, node_b in graph_metric.edges():
        node_a = int(node_a)
        node_b = int(node_b)
        for edge in ((node_a, node_b), (node_b, node_a)):
            if edge not in seen_edges:
                map_con.add_edge(*edge)
                seen_edges.add(edge)
    map_con.purge()
    map_con.setup_index(force=True)
    return map_con


def nearest_edge_distances(graph_metric, x_values, y_values):
    _, distances = ox.distance.nearest_edges(
        graph_metric,
        X=x_values,
        Y=y_values,
        return_dist=True,
    )
    return np.asarray(distances, dtype=np.float64)


def map_support_mask(
    distances,
    enter_dist,
    exit_dist,
    enter_hold_s=0.0,
    exit_hold_s=0.0,
    sample_step=1,
):
    mask = np.zeros(len(distances), dtype=bool)
    active = False
    enter_streak = 0
    exit_streak = 0
    enter_required = max(1, int(math.ceil(enter_hold_s / sample_step)))
    exit_required = max(1, int(math.ceil(exit_hold_s / sample_step)))
    for index, distance in enumerate(distances):
        if active:
            if distance > exit_dist:
                exit_streak += 1
                if exit_streak >= exit_required:
                    active = False
                    exit_streak = 0
            else:
                exit_streak = 0
        else:
            if distance <= enter_dist:
                enter_streak += 1
                if enter_streak >= enter_required:
                    active = True
                    enter_streak = 0
            else:
                enter_streak = 0
        mask[index] = active
    return mask


def contiguous_runs(mask):
    runs = []
    start = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def run_matcher_segment(map_con, x_values, y_values, raw_indices, settings):
    observations = [
        (float(y_values[index]), float(x_values[index]))
        for index in raw_indices
    ]
    matcher = DistanceMatcher(
        map_con,
        max_dist=settings["max_dist_m"],
        max_dist_init=settings["max_dist_m"],
        obs_noise=settings["obs_noise_m"],
        obs_noise_ne=settings["obs_noise_m"] * 1.5,
        dist_noise=settings["dist_noise_m"],
        min_prob_norm=1e-5,
        non_emitting_states=True,
        non_emitting_length_factor=settings["non_emitting_length_factor"],
        only_edges=True,
        max_lattice_width=settings["max_lattice_width"],
        avoid_goingback=True,
    )
    _, last_index = matcher.match(observations)
    if last_index is None:
        del matcher
        gc.collect()
        return None

    final_matching = matcher.get_matching(last_index)
    if final_matching is None:
        del matcher
        gc.collect()
        return None

    path = matcher.get_matching_path(final_matching)
    emitting = {}
    for matching in path:
        if not matching.is_emitting():
            continue
        projected = matching.edge_m.pi
        if projected is None:
            projected = matching.edge_m.p1
        emitting[int(matching.obs)] = {
            "y": float(projected[0]),
            "x": float(projected[1]),
            "distance_m": float(matching.dist_obs),
            "edge": [int(matching.edge_m.l1), int(matching.edge_m.l2)],
        }

    matched_observations = sorted(emitting)
    if len(matched_observations) < 2:
        return None

    observation_indices = np.asarray(matched_observations, dtype=int)
    matched_raw_indices = raw_indices[observation_indices]
    matched_edges = [emitting[index]["edge"] for index in matched_observations]
    route_edges = [
        edge
        for index, edge in enumerate(matched_edges)
        if index == 0 or edge != matched_edges[index - 1]
    ]
    undirected_edges = [tuple(sorted(edge)) for edge in route_edges]
    immediate_reversals = sum(
        previous == current[::-1]
        for previous, current in zip(route_edges, route_edges[1:])
    )
    revisited_edges = len(undirected_edges) - len(set(undirected_edges))
    result = {
        "raw_indices": matched_raw_indices,
        "x": np.asarray(
            [emitting[index]["x"] for index in matched_observations],
            dtype=np.float64,
        ),
        "y": np.asarray(
            [emitting[index]["y"] for index in matched_observations],
            dtype=np.float64,
        ),
        "distances_m": np.asarray(
            [emitting[index]["distance_m"] for index in matched_observations],
            dtype=np.float64,
        ),
        "last_observation": int(last_index),
        "observation_count": len(observations),
        "route_edge_count": len(route_edges),
        "route_unique_edge_count": len(set(undirected_edges)),
        "route_revisited_edge_count": int(revisited_edges),
        "route_immediate_reversals": int(immediate_reversals),
    }
    del matcher, path, final_matching
    gc.collect()
    return result


def segment_transition_weights(length, blend_points):
    weights = np.ones(length, dtype=np.float64)
    if blend_points <= 0 or length == 0:
        return weights
    ramp_length = min(blend_points, int(math.ceil(length / 2.0)))
    ramp = np.arange(1, ramp_length + 1, dtype=np.float64) / (
        ramp_length + 1
    )
    weights[:ramp_length] = ramp
    weights[-ramp_length:] = ramp[::-1]
    return weights


def run_matcher(
    raw_df,
    graph_metric,
    map_con,
    x_values,
    y_values,
    nearest_distances,
    settings,
):
    sample_step = settings["sample_step"]
    sample_indices = np.arange(0, len(x_values), sample_step, dtype=int)
    if sample_indices[-1] != len(x_values) - 1:
        sample_indices = np.append(sample_indices, len(x_values) - 1)

    initial_support_mask = map_support_mask(
        nearest_distances,
        settings["match_enter_dist_m"],
        settings["match_exit_dist_m"],
        settings["match_enter_hold_s"],
        settings["match_exit_hold_s"],
        1,
    )
    elapsed_s = (
        raw_df["time"] - raw_df["time"].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=np.float64)
    detector_config = DetectorConfig(
        min_event_s=settings["min_match_segment_s"],
    )
    detector_events = detect_events(
        raw_df,
        graph_metric,
        x_values,
        y_values,
        nearest_distances,
        initial_support_mask,
        elapsed_s,
        detector_config,
    )
    effective_support_mask = initial_support_mask.copy()
    detector_labels = np.full(len(x_values), "unclassified", dtype=object)
    detector_labels[initial_support_mask] = "map_support"
    for event in detector_events.itertuples(index=False):
        event_slice = slice(event.start_index, event.end_index + 1)
        detector_labels[event_slice] = event.predicted_label
        if event.predicted_label == "gps_aberration":
            effective_support_mask[event_slice] = True
    promoted_mask = effective_support_mask & ~initial_support_mask
    support_mask = effective_support_mask[sample_indices]

    min_segment_observations = max(
        2,
        int(math.ceil(settings["min_match_segment_s"] / sample_step)),
    )

    correction_x = np.zeros(len(x_values), dtype=np.float64)
    correction_y = np.zeros(len(y_values), dtype=np.float64)
    matched_mask = np.zeros(len(x_values), dtype=bool)
    blend_weights = np.zeros(len(x_values), dtype=np.float64)
    blend_points = int(round(settings["transition_blend_s"]))
    segment_rows = []
    matched_distances = []
    matcher_attempts = 0
    edge_count = 0
    unique_edge_count = 0
    revisited_edge_count = 0
    immediate_reversals = 0

    for support_start, support_end in contiguous_runs(support_mask):
        cursor = support_start
        while cursor <= support_end:
            remaining_indices = sample_indices[cursor : support_end + 1]
            if len(remaining_indices) < min_segment_observations:
                break

            matcher_attempts += 1
            result = run_matcher_segment(
                map_con,
                x_values,
                y_values,
                remaining_indices,
                settings,
            )
            if result is None:
                cursor += min_segment_observations
                continue

            consumed = max(1, result["last_observation"] + 1)
            matched_indices = result["raw_indices"]
            matched_duration = matched_indices[-1] - matched_indices[0]
            if matched_duration >= settings["min_match_segment_s"]:
                dense_indices = np.arange(
                    matched_indices[0],
                    matched_indices[-1] + 1,
                    dtype=int,
                )
                dense_correction_x = np.interp(
                    dense_indices,
                    matched_indices,
                    result["x"] - x_values[matched_indices],
                )
                dense_correction_y = np.interp(
                    dense_indices,
                    matched_indices,
                    result["y"] - y_values[matched_indices],
                )
                correction_x[dense_indices] = dense_correction_x
                correction_y[dense_indices] = dense_correction_y
                matched_mask[dense_indices] = True
                blend_weights[dense_indices] = segment_transition_weights(
                    len(dense_indices),
                    blend_points,
                )
                matched_distances.extend(result["distances_m"].tolist())
                edge_count += result["route_edge_count"]
                unique_edge_count += result["route_unique_edge_count"]
                revisited_edge_count += result["route_revisited_edge_count"]
                immediate_reversals += result["route_immediate_reversals"]
                segment_rows.append(
                    {
                        "start_index": int(dense_indices[0]),
                        "end_index": int(dense_indices[-1]),
                        "points": int(len(dense_indices)),
                        "matcher_observations": int(
                            len(result["raw_indices"])
                        ),
                    }
                )
            cursor += consumed

    output_x = x_values + correction_x * blend_weights
    output_y = y_values + correction_y * blend_weights

    matched_distances_array = np.asarray(matched_distances, dtype=np.float64)
    if len(matched_distances_array):
        distance_summary = {
            "mean": float(np.mean(matched_distances_array)),
            "median": float(np.median(matched_distances_array)),
            "p95": float(np.percentile(matched_distances_array, 95)),
            "max": float(np.max(matched_distances_array)),
        }
    else:
        distance_summary = {
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    details = {
        "observations": int(len(sample_indices)),
        "matched_observations": int(np.sum(matched_mask[sample_indices])),
        "coverage": float(np.mean(matched_mask)),
        "map_support_coverage": float(np.mean(initial_support_mask)),
        "detector_support_coverage": float(np.mean(effective_support_mask)),
        "detector_promoted_coverage": float(np.mean(promoted_mask)),
        "gps_fallback_coverage": float(1.0 - np.mean(matched_mask)),
        "sample_step": int(sample_step),
        "matcher_attempts": int(matcher_attempts),
        "detector_events": json.loads(
            detector_events.to_json(orient="records")
        ),
        "matched_segments": segment_rows,
        "observation_distance_m": distance_summary,
        "route_edge_count": int(edge_count),
        "route_unique_edge_count": int(unique_edge_count),
        "route_revisited_edge_count": int(revisited_edge_count),
        "route_immediate_reversals": int(immediate_reversals),
    }
    trace = pd.DataFrame(
        {
            "point_index": np.arange(len(x_values), dtype=int),
            "nearest_osm_edge_m": nearest_distances,
            "initial_map_support": initial_support_mask,
            "detector_label": detector_labels,
            "detector_promoted": promoted_mask,
            "source": np.where(matched_mask, "map_matched", "gps_fallback"),
            "blend_weight": blend_weights,
            "applied_correction_m": np.hypot(
                correction_x * blend_weights,
                correction_y * blend_weights,
            ),
        }
    )
    return output_x, output_y, details, trace


def metric_to_latlon(x_values, y_values, source_crs):
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon_values, lat_values = transformer.transform(x_values, y_values)
    return np.asarray(lat_values), np.asarray(lon_values)


def write_gpx(track_df, lat_values, lon_values, output_path):
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=output_path.stem)
    segment = gpxpy.gpx.GPXTrackSegment()
    gpx.tracks.append(track)
    track.segments.append(segment)
    for index, row in track_df.iterrows():
        timestamp = row["time"]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()
        segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                latitude=float(lat_values[index]),
                longitude=float(lon_values[index]),
                elevation=float(row["ele"]),
                time=timestamp,
            )
        )
    output_path.write_text(gpx.to_xml(), encoding="utf-8")


def plot_map(
    graph_metric,
    raw_xy,
    matched_variants,
    pattern_df,
    output_path,
):
    fig, ax = ox.plot.plot_graph(
        graph_metric,
        show=False,
        close=False,
        node_size=0,
        edge_color="#b8b8b8",
        edge_linewidth=0.8,
        bgcolor="white",
        figsize=(11, 9),
    )
    ax.plot(raw_xy[0], raw_xy[1], color="#2774b8", linewidth=1.2, alpha=0.75, label="raw")
    for matched_xy in matched_variants.values():
        ax.plot(
            matched_xy[0],
            matched_xy[1],
            color="#d62728",
            linewidth=1.8,
            alpha=0.9,
            label="OSM hybrid",
        )
    if pattern_df is not None:
        pattern_x, pattern_y = project_track(
            pattern_df,
            graph_metric.graph["crs"],
        )
        ax.plot(
            pattern_x,
            pattern_y,
            color="#ff7f0e",
            linewidth=1.6,
            alpha=0.9,
            label="pattern (evaluation only)",
        )
    ax.legend(loc="best")
    ax.set_title("OSM map-matching diagnostic")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def max_contiguous_duration(mask, time_seconds):
    if not np.any(mask):
        return 0.0
    if len(time_seconds) > 1:
        sample_period = float(np.median(np.diff(time_seconds)))
    else:
        sample_period = 0.0
    longest = 0
    current = 0
    for active in mask:
        if active:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest * sample_period)


def horizontal_length_m(track_df):
    if len(track_df) < 2:
        return 0.0
    geod = Geod(ellps="WGS84")
    _, _, distances = geod.inv(
        track_df["lon"].to_numpy(dtype=np.float64)[:-1],
        track_df["lat"].to_numpy(dtype=np.float64)[:-1],
        track_df["lon"].to_numpy(dtype=np.float64)[1:],
        track_df["lat"].to_numpy(dtype=np.float64)[1:],
    )
    return float(np.sum(distances))


def calculate_track_metrics(pattern_df, variants):
    rows = []
    aligned_variants = {}
    for variant, track_df in variants.items():
        aligned = align_track_to_pattern(pattern_df, track_df)
        aligned_variants[variant] = aligned
        error_x = aligned["errors"]["x"]
        error_y = aligned["errors"]["y"]
        radial_error = np.hypot(error_x, error_y)
        common_start = max(pattern_df["time"].min(), track_df["time"].min())
        common_end = min(pattern_df["time"].max(), track_df["time"].max())
        pattern_common = pattern_df[
            (pattern_df["time"] >= common_start)
            & (pattern_df["time"] <= common_end)
        ]
        track_common = track_df[
            (track_df["time"] >= common_start)
            & (track_df["time"] <= common_end)
        ]
        pattern_length = horizontal_length_m(pattern_common)
        track_length = horizontal_length_m(track_common)
        rows.append(
            {
                "variant": variant,
                "n_points": aligned["n_points"],
                "xy_mae_m": float(np.mean(radial_error)),
                "xy_rms_m": float(np.sqrt(np.mean(error_x**2 + error_y**2))),
                "xy_p95_m": float(np.percentile(radial_error, 95)),
                "xy_max_m": float(np.max(radial_error)),
                "xy_over_25_pct": float(np.mean(radial_error > 25.0) * 100.0),
                "xy_over_50_pct": float(np.mean(radial_error > 50.0) * 100.0),
                "max_run_over_25_s": max_contiguous_duration(
                    radial_error > 25.0,
                    aligned["time_seconds"],
                ),
                "max_run_over_50_s": max_contiguous_duration(
                    radial_error > 50.0,
                    aligned["time_seconds"],
                ),
                "track_length_m": track_length,
                "length_delta_to_pattern_m": track_length - pattern_length,
            }
        )
    return pd.DataFrame(rows), aligned_variants


def plot_error_timeline(aligned_variants, output_path):
    colors = {
        "raw": "#2774b8",
        "osm_hybrid": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(13, 5))
    for variant, aligned in aligned_variants.items():
        radial_error = np.hypot(
            aligned["errors"]["x"],
            aligned["errors"]["y"],
        )
        ax.plot(
            aligned["time_seconds"] / 60.0,
            radial_error,
            color=colors.get(variant),
            linewidth=1.2,
            label=variant,
        )
    ax.axhline(25.0, color="#777777", linestyle="--", linewidth=0.9)
    ax.axhline(50.0, color="#333333", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("XY error (m)")
    ax.set_title("Map-matching XY error over time")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def calculate_spectral_outputs(recording, aligned_variants, bands, args, output_dir):
    summary_rows = []
    analyses = {}
    for variant, aligned in aligned_variants.items():
        spectra = {}
        for channel in ("x", "y", "z"):
            freqs, times, power = compute_spectrogram(
                aligned["errors"][channel],
                1.0,
                args.nperseg,
                args.overlap,
            )
            spectra[channel] = {"freqs": freqs, "times": times, "power": power}
        spectra["xy"] = {
            "freqs": spectra["x"]["freqs"],
            "times": spectra["x"]["times"],
            "power": spectra["x"]["power"] + spectra["y"]["power"],
        }
        signal_powers = {
            channel: float(np.mean(aligned["errors"][channel] ** 2))
            for channel in ("x", "y", "z")
        }
        signal_powers["xy"] = signal_powers["x"] + signal_powers["y"]

        for channel in CHANNELS:
            spectrum = spectra[channel]
            for band_row in summarize_bands(
                spectrum["freqs"],
                spectrum["power"],
                bands,
            ):
                summary_rows.append(
                    {
                        "recording": recording,
                        "variant": variant,
                        "channel": channel,
                        "band_label": (
                            f"{band_row['f_low_hz']:.6g}-"
                            f"{band_row['f_high_hz']:.6g}"
                        ),
                        **band_row,
                        "signal_power": signal_powers[channel],
                        "n_points": aligned["n_points"],
                        "common_start": aligned["common_start"],
                        "common_end": aligned["common_end"],
                    }
                )
        analyses[variant] = {
            "spectra": spectra,
            "time_seconds": aligned["time_seconds"],
            "distance_m": aligned["distance_m"],
        }

    summary_df = add_raw_ratios(pd.DataFrame(summary_rows))
    summary_df.to_csv(output_dir / "band_power.csv", index=False)
    aggregate = (
        summary_df.groupby(
            ["variant", "channel", "band_label", "f_low_hz", "f_high_hz"],
            dropna=False,
        )
        .agg(
            recordings=("recording", "nunique"),
            mean_band_power=("band_power", "mean"),
            mean_total_power=("total_power", "mean"),
            mean_signal_power=("signal_power", "mean"),
            mean_band_pct=("band_pct", "mean"),
            mean_ratio_to_raw=("ratio_to_raw", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "band_power_summary.csv", index=False)
    plot_band_ratio_summary(aggregate, output_dir / "band_ratio_to_raw.png")
    plot_recording(
        recording,
        analyses,
        "time",
        output_dir / "spectrogram.png",
        clip_percentiles=(2, 98),
        title_suffix=" (detail)",
    )
    return aggregate


def main():
    args = parse_args()
    settings = resolve_settings(args)

    repo_root = Path.cwd()
    pass_dir = repo_root / args.preprocessed_dir / args.pasada
    recording_path = pass_dir / f"{args.recording}.gpx"
    pattern_path = pass_dir / f"{args.pasada}_aligned_pattern_resampled.gpx"
    if not recording_path.exists():
        raise FileNotFoundError(recording_path)
    if not args.no_evaluation and not pattern_path.exists():
        raise FileNotFoundError(pattern_path)

    output_dir = Path(
        args.output_dir
        or repo_root
        / "results"
        / "diagnostics"
        / "osm_map_matching"
        / f"family_{args.pasada}"
        / args.recording
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_cache = output_dir / f"osm_{args.network_type}.graphml"

    raw_df = read_recording(recording_path)
    pattern_df = None if args.no_evaluation else parse_gpx(pattern_path)
    graph, bbox = load_or_download_graph(
        raw_df,
        graph_cache,
        args.network_type,
        args.map_margin_m,
        args.refresh_map,
    )
    graph_metric = build_metric_graph(graph)
    raw_x, raw_y = project_track(raw_df, graph_metric.graph["crs"])
    nearest_distances = nearest_edge_distances(graph_metric, raw_x, raw_y)
    print(
        "Nearest OSM edge distances: "
        f"median={np.median(nearest_distances):.1f} m, "
        f"p95={np.percentile(nearest_distances, 95):.1f} m, "
        f"max={np.max(nearest_distances):.1f} m"
    )

    variants = {"raw": raw_df}
    matched_variants = {}
    print("\nRunning detector-guided map matching")
    matched_path = output_dir / f"{args.recording}_osm_hybrid.gpx"
    trace_path = output_dir / f"{args.recording}_osm_hybrid_trace.csv"
    if args.reuse_hybrid_outputs:
        if not matched_path.exists() or not trace_path.exists():
            raise FileNotFoundError("Missing reusable detector-guided outputs")
        matched_df = parse_gpx(matched_path)
        matched_x, matched_y = project_track(
            matched_df,
            graph_metric.graph["crs"],
        )
        trace = pd.read_csv(trace_path)
        required_columns = {
            "source",
            "initial_map_support",
            "detector_promoted",
        }
        missing_columns = required_columns - set(trace.columns)
        if missing_columns:
            raise ValueError(
                "Reusable trace predates detector integration; missing: "
                + ", ".join(sorted(missing_columns))
            )
        matched_mask = trace["source"].eq("map_matched").to_numpy()
        initial_support_mask = trace["initial_map_support"].to_numpy(
            dtype=bool
        )
        promoted_mask = trace["detector_promoted"].to_numpy(dtype=bool)
        matching = {
            "observations": int(len(trace)),
            "matched_observations": int(np.sum(matched_mask)),
            "coverage": float(np.mean(matched_mask)),
            "map_support_coverage": float(np.mean(initial_support_mask)),
            "detector_support_coverage": float(
                np.mean(initial_support_mask | promoted_mask)
            ),
            "detector_promoted_coverage": float(np.mean(promoted_mask)),
            "gps_fallback_coverage": float(1.0 - np.mean(matched_mask)),
            "sample_step": int(settings["sample_step"]),
            "matcher_attempts": None,
            "detector_events": None,
            "matched_segments": [
                {
                    "start_index": int(start),
                    "end_index": int(end),
                    "points": int(end - start + 1),
                }
                for start, end in contiguous_runs(matched_mask)
            ],
            "observation_distance_m": None,
            "route_edge_count": None,
            "route_unique_edge_count": None,
            "route_revisited_edge_count": None,
            "route_immediate_reversals": None,
        }
    else:
        map_con = graph_to_matcher_map(graph_metric)
        matched_x, matched_y, matching, trace = run_matcher(
            raw_df,
            graph_metric,
            map_con,
            raw_x,
            raw_y,
            nearest_distances,
            settings,
        )
        del map_con
        gc.collect()
        matched_lat, matched_lon = metric_to_latlon(
            matched_x,
            matched_y,
            graph_metric.graph["crs"],
        )
        write_gpx(raw_df, matched_lat, matched_lon, matched_path)
        trace.insert(1, "time", raw_df["time"].astype(str).to_numpy())
        trace.to_csv(trace_path, index=False)
        matched_df = parse_gpx(matched_path)
    variants["osm_hybrid"] = matched_df
    matched_variants["osm_hybrid"] = (matched_x, matched_y)
    matching_result = {
        "settings": settings,
        **matching,
    }

    if pattern_df is not None:
        metrics_df, aligned_variants = calculate_track_metrics(
            pattern_df,
            variants,
        )
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)
        plot_error_timeline(
            aligned_variants,
            output_dir / "error_timeline.png",
        )
        bands = parse_bands(args.bands)
        aggregate = calculate_spectral_outputs(
            args.recording,
            aligned_variants,
            bands,
            args,
            output_dir,
        )
    else:
        metrics_df = None
        aggregate = None
    plot_map(
        graph_metric,
        (raw_x, raw_y),
        matched_variants,
        pattern_df,
        output_dir / "map.png",
    )

    metadata = {
        "pasada": args.pasada,
        "recording": args.recording,
        "recording_path": str(recording_path),
        "pattern_path": str(pattern_path) if pattern_df is not None else None,
        "pattern_role": (
            "evaluation_only" if pattern_df is not None else "not_used"
        ),
        "network_type": args.network_type,
        "bbox": list(bbox),
        "graph_nodes": int(len(graph.nodes)),
        "graph_edges": int(len(graph.edges)),
        "graph_crs": str(graph_metric.graph["crs"]),
        "nearest_osm_edge_m": {
            "median": float(np.median(nearest_distances)),
            "p95": float(np.percentile(nearest_distances, 95)),
            "max": float(np.max(nearest_distances)),
        },
        "matching": matching_result,
        "arguments": vars(args),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n=== OSM MAP-MATCHING RESULT ===")
    if metrics_df is not None:
        print(metrics_df.to_string(index=False))
        xy_summary = aggregate[aggregate["channel"] == "xy"]
        total_energy = (
            xy_summary.groupby("variant")["mean_signal_power"].mean().to_dict()
        )
        raw_power = total_energy.get("raw", float("nan"))
    else:
        print("Evaluation skipped: no clean pattern was used")
        total_energy = {}
        raw_power = float("nan")

    if metrics_df is not None:
        matched_power = total_energy.get("osm_hybrid", float("nan"))
        energy_text = (
            f", residual XY energy="
            f"{100.0 * matched_power / raw_power:.1f}%"
        )
    else:
        energy_text = ""
    print(
        f"coverage={matching_result['coverage'] * 100.0:.1f}%"
        f"{energy_text}, "
        f"revisited edges="
        f"{matching_result.get('route_revisited_edge_count')}, "
        f"immediate reversals="
        f"{matching_result.get('route_immediate_reversals')}"
    )
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
