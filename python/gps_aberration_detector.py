"""Reusable event detector for GPS excursions around mapped routes."""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds used by event extraction and classification."""

    min_event_s: float = 15.0
    short_event_s: float = 180.0
    route_ratio_low: float = 0.7
    route_ratio_high: float = 1.3
    strong_departure_m: float = 50.0


def _contiguous_runs(mask):
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


def _xy_length(x_values, y_values):
    if len(x_values) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(x_values), np.diff(y_values))))


def _shortest_map_distance(graph, start_node, end_node):
    try:
        return float(
            nx.shortest_path_length(
                graph,
                source=int(start_node),
                target=int(end_node),
                weight="length",
            )
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _classify_event(
    bounded,
    duration_s,
    osm_length_m,
    raw_length_m,
    max_distance_m,
    config,
):
    reasons = []
    if bounded:
        reasons.append("bounded_by_map_support")
    if duration_s <= config.short_event_s:
        reasons.append("short_excursion")
    if osm_length_m is not None:
        reasons.append("connected_osm_route")

    route_ratio = None
    if osm_length_m is not None and raw_length_m > 1.0:
        route_ratio = osm_length_m / raw_length_m
        if config.route_ratio_low <= route_ratio <= config.route_ratio_high:
            reasons.append("map_and_gps_lengths_agree")

    strong_departure = max_distance_m >= config.strong_departure_m
    if strong_departure:
        reasons.append("strong_map_departure")

    coherent_route = (
        route_ratio is not None
        and config.route_ratio_low <= route_ratio <= config.route_ratio_high
    )
    short_bounded = bounded and duration_s <= config.short_event_s
    if short_bounded and coherent_route and strong_departure:
        label = "gps_aberration"
    elif (
        not bounded
        or duration_s > 300.0
        or (
            route_ratio is not None
            and (route_ratio < 0.5 or route_ratio > 1.5)
        )
    ):
        label = "off_map"
    else:
        label = "uncertain"
    return label, route_ratio, reasons


def detect_events(
    raw_df,
    graph_metric,
    x_values,
    y_values,
    nearest_distances,
    support_mask,
    elapsed_s,
    config,
):
    """Extract and classify events using only GPS and OSM information."""

    graph_undirected = graph_metric.to_undirected()
    rows = []
    for event_id, (start, end) in enumerate(
        _contiguous_runs(~support_mask),
        1,
    ):
        duration_s = float(elapsed_s[end] - elapsed_s[start] + 1.0)
        if duration_s < config.min_event_s:
            continue

        bounded = start > 0 and end < len(raw_df) - 1
        anchor_start = start - 1 if start > 0 else start
        anchor_end = end + 1 if end < len(raw_df) - 1 else end
        raw_length_m = _xy_length(
            x_values[anchor_start : anchor_end + 1],
            y_values[anchor_start : anchor_end + 1],
        )

        osm_length_m = None
        if bounded:
            nearest_nodes = ox.distance.nearest_nodes(
                graph_metric,
                X=[x_values[anchor_start], x_values[anchor_end]],
                Y=[y_values[anchor_start], y_values[anchor_end]],
            )
            osm_length_m = _shortest_map_distance(
                graph_undirected,
                nearest_nodes[0],
                nearest_nodes[1],
            )

        max_distance_m = float(
            np.max(nearest_distances[start : end + 1])
        )
        predicted, route_ratio, reasons = _classify_event(
            bounded,
            duration_s,
            osm_length_m,
            raw_length_m,
            max_distance_m,
            config,
        )
        rows.append(
            {
                "event_id": event_id,
                "start_index": int(start),
                "end_index": int(end),
                "start_time": str(raw_df["time"].iloc[start]),
                "end_time": str(raw_df["time"].iloc[end]),
                "start_minute": float(elapsed_s[start] / 60.0),
                "end_minute": float(elapsed_s[end] / 60.0),
                "duration_s": duration_s,
                "bounded_by_map_support": bounded,
                "max_distance_to_osm_m": max_distance_m,
                "median_distance_to_osm_m": float(
                    np.median(nearest_distances[start : end + 1])
                ),
                "raw_length_m": raw_length_m,
                "osm_route_length_m": osm_length_m,
                "osm_to_raw_length_ratio": route_ratio,
                "predicted_label": predicted,
                "prediction_reasons": "|".join(reasons),
            }
        )
    return pd.DataFrame(rows)
