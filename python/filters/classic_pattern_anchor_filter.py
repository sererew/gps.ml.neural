#!/usr/bin/env python3
"""Shared implementation for classic-filter plus pattern-anchor pipelines."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pattern_anchor_common as anchor_common


def load_filter_module(script_name: str, module_name: str):
    """Load a filter module whose filename is not a valid Python import name."""
    module_path = Path(__file__).with_name(script_name)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def output_path_for(input_gpx: str, output_gpx: str | None, suffix: str) -> Path:
    """Return the explicit or auto-generated output path."""
    if output_gpx:
        output_path = Path(output_gpx)
    else:
        input_path = Path(input_gpx)
        output_path = input_path.parent / f"{input_path.stem}_{suffix}{input_path.suffix}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def run_classic_pattern_anchor(
    filter_name: str,
    apply_filter,
    default_suffix: str,
    filter_args: dict,
) -> None:
    """Run a classic local filter followed by the oracle pattern-anchor correction."""
    parser = argparse.ArgumentParser(
        description=f"Apply {filter_name} filter followed by pattern-anchor correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_gpx", help="Input GPX file sampled at 1Hz")
    parser.add_argument("output_gpx", nargs="?", help="Output filtered GPX file")
    parser.add_argument("--pattern-gpx", default=None, help="Pattern GPX path for pattern-anchor correction")
    parser.add_argument("--suffix", default=default_suffix, help="Suffix for auto-generated output filename")
    parser.add_argument("--anchors-per-hour", type=float, default=8.0, help="Pattern-anchor density per hour")
    parser.add_argument("--min-anchors", type=int, default=8, help="Minimum pattern anchors per track")
    parser.add_argument("--max-anchors", type=int, default=0, help="Maximum pattern anchors per track; 0 means no cap")
    parser.add_argument("--anchor-error-radius", type=int, default=0, help="Local radius for anchor error averaging")
    parser.add_argument("--anchor-channels", default="x,y,z", help="Comma-separated channels for anchor correction")
    parser.add_argument(
        "--anchor-interpolation",
        choices=["linear", "cubic"],
        default="cubic",
        help="Pattern-anchor interpolation mode",
    )
    parser.add_argument("--anchor-edge-blend-points", type=int, default=30, help="Points used to blend in correction")
    parser.add_argument("--anchor-edge-skip-points", type=int, default=180, help="Common-timeline edge points excluded")
    parser.add_argument("--anchor-trim-to-pattern", action="store_true", default=True, help="Trim to common timeline")
    parser.add_argument("--no-anchor-trim-to-pattern", dest="anchor_trim_to_pattern", action="store_false")
    for name, kwargs in filter_args.items():
        parser.add_argument(name, **kwargs)
    args = parser.parse_args()

    output_path = output_path_for(args.input_gpx, args.output_gpx, args.suffix)
    pattern_gpx = Path(args.pattern_gpx) if args.pattern_gpx else anchor_common.infer_pattern_path(args.input_gpx)

    try:
        print(f"Processing {args.input_gpx}...")
        print(f"Base filter: {filter_name}")
        print(f"Pattern-anchor correction enabled with pattern: {pattern_gpx}")
        track_df = anchor_common.parse_gpx(args.input_gpx)
        filtered_df = apply_filter(track_df, args)
        pattern_df = anchor_common.parse_gpx(pattern_gpx)
        corrected_df = anchor_common.apply_pattern_anchor_correction(
            filtered_df,
            pattern_df,
            anchors_per_hour=args.anchors_per_hour,
            min_anchors=args.min_anchors,
            max_anchors=args.max_anchors,
            anchor_error_radius=args.anchor_error_radius,
            anchor_channels=args.anchor_channels,
            anchor_interpolation=args.anchor_interpolation,
            anchor_edge_blend_points=args.anchor_edge_blend_points,
            anchor_trim_to_pattern=args.anchor_trim_to_pattern,
            anchor_edge_skip_points=args.anchor_edge_skip_points,
        )
        anchor_common.create_gpx(
            corrected_df["lat"].to_numpy(),
            corrected_df["lon"].to_numpy(),
            corrected_df["ele"].to_numpy(),
            corrected_df["time"].to_numpy() if "time" in corrected_df.columns else None,
            str(output_path),
        )
        print("SUCCESS: Classic pattern-anchor filtering completed successfully")
        print(f"   Input: {len(track_df)} points")
        print(f"   Output: {len(corrected_df)} points")
        print(f"   Filtered track saved to: {output_path}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
