#!/usr/bin/env python3
"""
Run fast training sweeps for different lambda_traj values.

Each run executes:
    python -X utf8 python/pipeline/6_train_neural_network_v2.py --fast ...

Results are copied to:
    results/training/sweeps/lambda_traj_<value>/

The script also writes:
    results/training/sweeps/summary.csv
    results/training/sweeps/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_LAMBDAS = ["0.01", "0.03", "0.1", "0.3", "1.0"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_name(value: str) -> str:
    return value.replace(".", "p").replace("-", "neg")


def run_training(
    root: Path,
    lambda_traj: str,
    lambda_bias: str,
    seed: int,
    log_path: Path,
) -> subprocess.CompletedProcess:
    script = root / "python" / "pipeline" / "6_train_neural_network_v2.py"
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        str(script),
        "--fast",
        "--lambda_traj",
        lambda_traj,
        "--lambda_bias",
        lambda_bias,
        "--seed",
        str(seed),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.run(
            cmd,
            cwd=root,
            text=True,
            encoding="utf-8",
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def run_dir_for(sweep_dir: Path, lambda_traj: str, lambda_bias: str) -> Path:
    return sweep_dir / f"lambda_traj_{safe_name(lambda_traj)}__lambda_bias_{safe_name(lambda_bias)}"


def collect_run_outputs(root: Path, sweep_dir: Path, lambda_traj: str, lambda_bias: str, seed: int) -> dict:
    training_dir = root / "results" / "training"
    result_json = training_dir / "training_results_fast.json"
    history_png = training_dir / "training_history.png"

    run_dir = run_dir_for(sweep_dir, lambda_traj, lambda_bias)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not result_json.exists():
        raise FileNotFoundError(f"Missing expected result file: {result_json}")

    target_json = run_dir / "training_results_fast.json"
    shutil.copy2(result_json, target_json)

    if history_png.exists():
        shutil.copy2(history_png, run_dir / "training_history.png")

    with target_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["test_metrics"]
    return {
        "lambda_traj": float(lambda_traj),
        "lambda_bias": data["config"]["lambda_bias"],
        "seed": seed,
        "mae_total_meters": metrics["mae_total_meters"],
        "mae_dx_meters": metrics["mae_dx_meters"],
        "mae_dy_meters": metrics["mae_dy_meters"],
        "mae_dz_meters": metrics["mae_dz_meters"],
        "drift_final_mean_m": metrics["drift_final_mean_m"],
        "drift_rms_m": metrics["drift_rms_m"],
        "length_diff_m": metrics["length_diff_m"],
        "training_time_minutes": data["training_time_minutes"],
        "epochs_trained": data["epochs_trained"],
        "final_train_loss": data["final_train_loss"],
        "final_val_loss": data["final_val_loss"],
        "run_dir": str(run_dir.relative_to(root)),
    }


def write_summary(sweep_dir: Path, rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda row: row["drift_final_mean_m"])

    csv_path = sweep_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    json_path = sweep_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows_sorted, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run --fast lambda_traj sweep")
    parser.add_argument(
        "--lambda-traj-values",
        nargs="+",
        default=DEFAULT_LAMBDAS,
        help="lambda_traj values to test",
    )
    parser.add_argument("--lambda-bias", default="0.1", help="lambda_bias value")
    parser.add_argument(
        "--lambda-bias-values",
        nargs="+",
        help="lambda_bias values to test. Overrides --lambda-bias when provided",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to each training run")
    parser.add_argument(
        "--sweep-dir",
        default="results/training/sweeps",
        help="Directory for copied run results and summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    sweep_dir = root / args.sweep_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    lambda_bias_values = args.lambda_bias_values or [args.lambda_bias]
    total = len(args.lambda_traj_values) * len(lambda_bias_values)
    run_index = 0

    for lambda_traj in args.lambda_traj_values:
        for lambda_bias in lambda_bias_values:
            run_index += 1
            print(f"\n=== Sweep {run_index}/{total}: lambda_traj={lambda_traj}, lambda_bias={lambda_bias} ===")
            run_dir = run_dir_for(sweep_dir, lambda_traj, lambda_bias)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "training.log"
            result = run_training(
                root,
                lambda_traj=lambda_traj,
                lambda_bias=lambda_bias,
                seed=args.seed,
                log_path=log_path,
            )
            if result.returncode != 0:
                print(
                    f"ERROR: training failed for lambda_traj={lambda_traj}, lambda_bias={lambda_bias}",
                    file=sys.stderr,
                )
                print(f"Log: {log_path}", file=sys.stderr)
                return result.returncode

            row = collect_run_outputs(root, sweep_dir, lambda_traj, lambda_bias, args.seed)
            rows.append(row)
            print(
                "Result: "
                f"MAE={row['mae_total_meters']:.4f} m, "
                f"drift={row['drift_final_mean_m']:.2f} m, "
                f"RMS={row['drift_rms_m']:.2f} m, "
                f"length_diff={row['length_diff_m']:.2f} m"
            )

    write_summary(sweep_dir, rows)

    best = min(rows, key=lambda row: row["drift_final_mean_m"])
    print("\n=== Best by drift_final_mean_m ===")
    print(
        f"lambda_traj={best['lambda_traj']} | "
        f"MAE={best['mae_total_meters']:.4f} m | "
        f"drift={best['drift_final_mean_m']:.2f} m | "
        f"RMS={best['drift_rms_m']:.2f} m | "
        f"length_diff={best['length_diff_m']:.2f} m"
    )
    print(f"Summary: {sweep_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
