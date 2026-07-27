#!/usr/bin/env python3
"""Run the error-spectrum diagnostic on one recording per family."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from diagnose_error_spectrum import plot_band_ratio_summary


DEFAULT_FILTERS = (
    "identity",
    "gaussian",
    "kalman",
    "median",
    "moving_average",
    "exponential",
    "savgol",
    "triangular_weighted",
    "gaussian_pattern_anchor",
    "moving_average_pattern_anchor",
    "nn",
    "nn_pattern_anchor",
)

DEFAULT_BANDS = (
    "0:0.0005,"
    "0.0005:0.001,"
    "0.001:0.003,"
    "0.003:0.01,"
    "0.01:0.05,"
    "0.05:0.15,"
    "0.15:0.5"
)


def natural_family_key(value: str) -> tuple[int, str]:
    digits = ""
    for char in value:
        if char.isdigit():
            digits += char
        else:
            break
    return (int(digits) if digits else 10**9, value)


def natural_text_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def find_one_recording_per_family(preprocessed_dir: Path) -> list[dict[str, str]]:
    rows = []
    for family_dir in sorted(
        [path for path in preprocessed_dir.iterdir() if path.is_dir()],
        key=lambda path: natural_family_key(path.name),
    ):
        candidates = sorted(
            (
                path
                for path in family_dir.glob("*_resampled.gpx")
                if "pattern" not in path.name and "aligned_pattern" not in path.name
            ),
            key=lambda path: natural_text_key(path.name),
        )
        if not candidates:
            continue
        recording_path = candidates[0]
        rows.append(
            {
                "family": family_dir.name,
                "recording": recording_path.stem,
                "input_path": str(recording_path),
            }
        )
    return rows


def read_selection(selection_csv: Path) -> list[dict[str, str]]:
    with open(selection_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "family": row["family"],
                "recording": row["recording"],
                "input_path": row["input_path"],
            }
            for row in reader
        ]


def filter_script_path(filters_dir: Path, filter_name: str) -> Path:
    return filters_dir / f"7_{filter_name}_filter.py"


def filtered_output_path(filtered_dir: Path, filter_name: str, family: str, recording: str) -> Path:
    return filtered_dir / filter_name / family / f"{recording}_{filter_name}_filtered.gpx"


def run_command(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def apply_missing_filters(
    selection: Iterable[dict[str, str]],
    filters: list[str],
    filters_dir: Path,
    filtered_dir: Path,
    overwrite: bool,
    timeout: int,
) -> None:
    for row in selection:
        family = row["family"]
        recording = row["recording"]
        input_path = Path(row["input_path"])
        for filter_name in filters:
            script_path = filter_script_path(filters_dir, filter_name)
            output_path = filtered_output_path(filtered_dir, filter_name, family, recording)
            if output_path.exists() and not overwrite:
                print(f"[skip] {family}/{recording} {filter_name}", flush=True)
                continue
            if not script_path.exists():
                raise FileNotFoundError(f"Filter script not found: {script_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[filter] {family}/{recording} {filter_name}", flush=True)
            result = run_command(
                [sys.executable, str(script_path), str(input_path), str(output_path)],
                timeout=timeout,
            )
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr, file=sys.stderr)
                raise RuntimeError(f"Filter failed: {family}/{recording} {filter_name}")


def run_diagnostics(
    selection: Iterable[dict[str, str]],
    filters: list[str],
    bands: str,
    output_dir: Path,
    sample_rate: float,
    nperseg: int,
    overlap: float,
) -> list[Path]:
    diagnostic_script = Path(__file__).with_name("diagnose_error_spectrum.py")
    csv_paths = []
    for row in selection:
        family = row["family"]
        recording = row["recording"]
        family_output_dir = output_dir / "families" / family
        print(f"[spectrum] {family}/{recording}", flush=True)
        result = run_command(
            [
                sys.executable,
                str(diagnostic_script),
                "--pasada",
                family,
                "--recordings",
                recording,
                "--filters",
                ",".join(filters),
                "--bands",
                bands,
                "--output_dir",
                str(family_output_dir),
                "--sample_rate",
                str(sample_rate),
                "--nperseg",
                str(nperseg),
                "--overlap",
                str(overlap),
            ],
            timeout=None,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Spectrum diagnostic failed: {family}/{recording}")
        csv_path = family_output_dir / "band_power.csv"
        if csv_path.exists():
            csv_paths.append(csv_path)
    return csv_paths


def write_selection(selection: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "selection.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "recording", "input_path"])
        writer.writeheader()
        writer.writerows(selection)


def combine_band_power(csv_paths: list[Path], output_dir: Path) -> None:
    if not csv_paths:
        raise RuntimeError("No band_power.csv files were generated")
    frames = [pd.read_csv(path) for path in csv_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "band_power.csv", index=False)

    aggregate = (
        combined.groupby(["variant", "channel", "band_label", "f_low_hz", "f_high_hz"], dropna=False)
        .agg(
            recordings=("recording", "nunique"),
            mean_band_power=("band_power", "mean"),
            mean_total_power=("total_power", "mean"),
            mean_band_pct=("band_pct", "mean"),
            mean_ratio_to_raw=("ratio_to_raw", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "band_power_summary.csv", index=False)
    plot_band_ratio_summary(aggregate, output_dir / "band_ratio_to_raw.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply all filters and run spectral diagnostics on one recording per family",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preprocessed_dir", default="data/preprocessed")
    parser.add_argument("--filters_dir", default="python/filters")
    parser.add_argument("--filtered_dir", default="results/filtered")
    parser.add_argument("--output_dir", default="results/diagnostics/error_spectrum_one_per_family")
    parser.add_argument("--selection_csv", default="", help="Optional fixed selection CSV from a previous run")
    parser.add_argument("--filters", default=",".join(DEFAULT_FILTERS))
    parser.add_argument("--families", default="", help="Optional comma-separated family subset")
    parser.add_argument("--bands", default=DEFAULT_BANDS)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--nperseg", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.75)
    parser.add_argument("--overwrite_filtered", action="store_true")
    parser.add_argument("--skip_filtering", action="store_true")
    parser.add_argument("--filter_timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    preprocessed_dir = Path(args.preprocessed_dir)
    filters_dir = Path(args.filters_dir)
    filtered_dir = Path(args.filtered_dir)
    output_dir = Path(args.output_dir)
    filters = split_csv(args.filters)

    if args.selection_csv:
        selection = read_selection(Path(args.selection_csv))
    else:
        selection = find_one_recording_per_family(preprocessed_dir)
    selected_families = set(split_csv(args.families))
    if selected_families:
        selection = [row for row in selection if row["family"] in selected_families]
    if not selection:
        raise RuntimeError("No recordings selected")

    write_selection(selection, output_dir)
    print(f"Selected recordings: {len(selection)}", flush=True)
    for row in selection:
        print(f"  {row['family']}: {row['recording']}", flush=True)

    if not args.skip_filtering:
        apply_missing_filters(
            selection,
            filters,
            filters_dir,
            filtered_dir,
            overwrite=args.overwrite_filtered,
            timeout=args.filter_timeout,
        )

    csv_paths = run_diagnostics(
        selection,
        filters,
        args.bands,
        output_dir,
        sample_rate=args.sample_rate,
        nperseg=args.nperseg,
        overlap=args.overlap,
    )
    combine_band_power(csv_paths, output_dir)

    print("\n=== DONE ===")
    print(f"Selection: {output_dir / 'selection.csv'}")
    print(f"Band power CSV: {output_dir / 'band_power.csv'}")
    print(f"Band summary CSV: {output_dir / 'band_power_summary.csv'}")
    print(f"Band ratio plot: {output_dir / 'band_ratio_to_raw.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
