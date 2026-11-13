#!/usr/bin/env python3
"""
Diagnose whether off-map GPS excursions look like receiver aberrations.

Classification uses only the raw recording and the OSM graph. A clean pattern
is optional and is used only to evaluate the predicted event labels.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from gps_aberration_detector import DetectorConfig, detect_events
from diagnose_error_spectrum import align_track_to_pattern, parse_gpx
from diagnose_osm_map_matching import (
    build_metric_graph,
    horizontal_length_m,
    map_support_mask,
    nearest_edge_distances,
    project_track,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose GPS aberrations around unmapped excursions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pasada", default="7")
    parser.add_argument("--recording", default="7_2_1_Q_W_resampled")
    parser.add_argument("--preprocessed_dir", default="data/preprocessed")
    parser.add_argument(
        "--map_matching_dir",
        default="results/diagnostics/osm_map_matching",
    )
    parser.add_argument(
        "--output_dir",
        default="results/diagnostics/gps_aberrations",
    )
    parser.add_argument("--enter_dist_m", type=float, default=15.0)
    parser.add_argument("--exit_dist_m", type=float, default=25.0)
    parser.add_argument("--enter_hold_s", type=float, default=10.0)
    parser.add_argument("--exit_hold_s", type=float, default=3.0)
    parser.add_argument("--min_event_s", type=float, default=15.0)
    parser.add_argument("--short_event_s", type=float, default=180.0)
    parser.add_argument("--route_ratio_low", type=float, default=0.7)
    parser.add_argument("--route_ratio_high", type=float, default=1.3)
    parser.add_argument("--strong_departure_m", type=float, default=50.0)
    parser.add_argument("--without_pattern", action="store_true")
    return parser.parse_args()


def pattern_error_on_raw_timeline(raw_df, pattern_df):
    aligned = align_track_to_pattern(raw_df, pattern_df)
    radial_error = np.hypot(
        aligned["errors"]["x"],
        aligned["errors"]["y"],
    )
    raw_start = raw_df["time"].iloc[0]
    common_start = pd.Timestamp(aligned["common_start"])
    offset_s = float((common_start - raw_start).total_seconds())
    aligned_times = aligned["time_seconds"] + offset_s
    raw_times = (
        raw_df["time"] - raw_start
    ).dt.total_seconds().to_numpy(dtype=np.float64)
    output = np.full(len(raw_df), np.nan, dtype=np.float64)
    valid = (raw_times >= aligned_times[0]) & (raw_times <= aligned_times[-1])
    output[valid] = np.interp(
        raw_times[valid],
        aligned_times,
        radial_error,
    )
    return output


def boundary_error(error_values, start, end, window_points=30):
    before = error_values[max(0, start - window_points) : start]
    after = error_values[end + 1 : min(len(error_values), end + 1 + window_points)]
    values = np.concatenate([before, after])
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def oracle_label(error_values, start, end):
    if error_values is None:
        return None, None, None
    event_error = error_values[start : end + 1]
    event_error = event_error[np.isfinite(event_error)]
    if not len(event_error):
        return "unavailable", None, None

    event_median = float(np.median(event_error))
    boundary_median = boundary_error(error_values, start, end)
    if boundary_median is None:
        return "unavailable", event_median, None

    excess = event_median - boundary_median
    if excess >= 10.0:
        label = "gps_aberration"
    elif excess <= 5.0:
        label = "off_map"
    else:
        label = "uncertain"
    return label, event_median, boundary_median


def add_pattern_evaluation(events, pattern_error):
    columns = [
        "oracle_label",
        "event_pattern_error_median_m",
        "boundary_pattern_error_median_m",
    ]
    values = [
        oracle_label(
            pattern_error,
            int(row.start_index),
            int(row.end_index),
        )
        for row in events.itertuples(index=False)
    ]
    if values:
        events[columns] = pd.DataFrame(values, index=events.index)
    else:
        for column in columns:
            events[column] = pd.Series(dtype=object)
    return events


def plot_timeline(
    elapsed_s,
    nearest_distances,
    pattern_error,
    events,
    output_path,
    enter_dist_m,
    exit_dist_m,
):
    fig, axes = plt.subplots(
        2 if pattern_error is not None else 1,
        1,
        figsize=(15, 8 if pattern_error is not None else 5),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    time_minutes = elapsed_s / 60.0
    axes[0].plot(time_minutes, nearest_distances, linewidth=1.0, color="#2774b8")
    axes[0].axhline(
        enter_dist_m,
        color="#2ca02c",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0].axhline(
        exit_dist_m,
        color="#d62728",
        linestyle="--",
        linewidth=0.9,
    )
    axes[0].set_ylabel("Nearest OSM edge (m)")
    axes[0].set_title("GPS off-map excursion diagnostic")

    if pattern_error is not None:
        axes[1].plot(time_minutes, pattern_error, linewidth=1.0, color="#444444")
        axes[1].set_ylabel("Pattern error (m)")

    colors = {
        "gps_aberration": "#d62728",
        "off_map": "#2ca02c",
        "uncertain": "#ffbf00",
    }
    for row in events.itertuples(index=False):
        color = colors.get(row.predicted_label, "#999999")
        for axis in axes:
            axis.axvspan(
                row.start_minute,
                row.end_minute,
                color=color,
                alpha=0.18,
            )
    axes[-1].set_xlabel("Time (minutes)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize(events):
    summary = {
        "events": int(len(events)),
        "predicted_counts": (
            events["predicted_label"].value_counts().to_dict()
            if len(events)
            else {}
        ),
    }
    evaluable = events[
        events["oracle_label"].isin(["gps_aberration", "off_map"])
    ]
    decided = evaluable[
        evaluable["predicted_label"].isin(["gps_aberration", "off_map"])
    ]
    if len(evaluable):
        summary["evaluated_events"] = int(len(evaluable))
        summary["decided_events"] = int(len(decided))
        summary["decision_rate"] = float(len(decided) / len(evaluable))
    if len(decided):
        summary["correct_decisions"] = int(
            np.sum(decided["predicted_label"] == decided["oracle_label"])
        )
        summary["decided_accuracy"] = (
            summary["correct_decisions"] / summary["decided_events"]
        )
    predicted_aberrations = evaluable[
        evaluable["predicted_label"] == "gps_aberration"
    ]
    oracle_aberrations = evaluable[
        evaluable["oracle_label"] == "gps_aberration"
    ]
    if len(predicted_aberrations):
        summary["aberration_precision"] = float(
            np.mean(
                predicted_aberrations["oracle_label"] == "gps_aberration"
            )
        )
    if len(oracle_aberrations):
        summary["aberration_recall"] = float(
            np.mean(
                oracle_aberrations["predicted_label"] == "gps_aberration"
            )
        )
    return summary


def main():
    args = parse_args()
    repo_root = Path.cwd()
    pass_dir = repo_root / args.preprocessed_dir / args.pasada
    recording_path = pass_dir / f"{args.recording}.gpx"
    pattern_path = pass_dir / f"{args.pasada}_aligned_pattern_resampled.gpx"
    map_dir = (
        repo_root
        / args.map_matching_dir
        / f"family_{args.pasada}"
        / args.recording
    )
    graph_path = map_dir / "osm_walk.graphml"
    if not recording_path.exists():
        raise FileNotFoundError(recording_path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Cached OSM graph not found; run map matching first: {graph_path}"
        )
    if not args.without_pattern and not pattern_path.exists():
        raise FileNotFoundError(pattern_path)

    output_dir = (
        repo_root
        / args.output_dir
        / f"family_{args.pasada}"
        / args.recording
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = parse_gpx(recording_path)
    graph_metric = build_metric_graph(ox.io.load_graphml(graph_path))
    x_values, y_values = project_track(raw_df, graph_metric.graph["crs"])
    nearest_distances = nearest_edge_distances(
        graph_metric,
        x_values,
        y_values,
    )
    support_mask = map_support_mask(
        nearest_distances,
        args.enter_dist_m,
        args.exit_dist_m,
        args.enter_hold_s,
        args.exit_hold_s,
        1,
    )
    elapsed_s = (
        raw_df["time"] - raw_df["time"].iloc[0]
    ).dt.total_seconds().to_numpy(dtype=np.float64)
    detector_config = DetectorConfig(
        min_event_s=args.min_event_s,
        short_event_s=args.short_event_s,
        route_ratio_low=args.route_ratio_low,
        route_ratio_high=args.route_ratio_high,
        strong_departure_m=args.strong_departure_m,
    )

    if args.without_pattern:
        pattern_error = None
    else:
        pattern_df = parse_gpx(pattern_path)
        pattern_error = pattern_error_on_raw_timeline(raw_df, pattern_df)

    events = detect_events(
        raw_df,
        graph_metric,
        x_values,
        y_values,
        nearest_distances,
        support_mask,
        elapsed_s,
        detector_config,
    )
    events = add_pattern_evaluation(events, pattern_error)
    events.to_csv(output_dir / "events.csv", index=False)
    plot_timeline(
        elapsed_s,
        nearest_distances,
        pattern_error,
        events,
        output_dir / "timeline.png",
        args.enter_dist_m,
        args.exit_dist_m,
    )

    summary = {
        "pasada": args.pasada,
        "recording": args.recording,
        "pattern_role": (
            "not_used" if args.without_pattern else "evaluation_only"
        ),
        "track_points": int(len(raw_df)),
        "track_length_m": horizontal_length_m(raw_df),
        "map_support_pct": float(np.mean(support_mask) * 100.0),
        "parameters": vars(args),
        **summarize(events),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== GPS ABERRATION DIAGNOSTIC ===")
    if len(events):
        columns = [
            "event_id",
            "start_minute",
            "duration_s",
            "max_distance_to_osm_m",
            "osm_to_raw_length_ratio",
            "predicted_label",
            "oracle_label",
        ]
        print(events[columns].to_string(index=False))
    else:
        print("No off-map events found")
    print(json.dumps(summary, indent=2))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
