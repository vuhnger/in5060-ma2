#!/usr/bin/env python3
"""
Convenience launcher that runs the full passive analysis pipeline:
1. analyze_passive_quality.py
2. analyze_results_and_report.py
3. plot_passive_quality_results.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


def run_step(command: List[str], description: str) -> None:
    print(f"\n==> {description}")
    print("    " + " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"Error: {description} failed with exit code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the passive quality analysis, reporting, and plotting scripts sequentially."
    )
    parser.add_argument("--four-g-dir", default="./4G", help="Path to the passive 4G dataset.")
    parser.add_argument("--five-g-dir", default="./5G", help="Path to the passive 5G dataset.")
    parser.add_argument("--include-ow", action="store_true", help="Pass through to include OW files.")
    parser.add_argument(
        "--pclip",
        default="0.05,0.95",
        help="Pass through percentile clip, e.g., '0.05,0.95'.",
    )
    parser.add_argument(
        "--speed-thresholds",
        default="0.5,5",
        help="Pass through OD speed thresholds, e.g., '0.5,5'.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold forwarded to analyze_results_and_report.py.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=2000,
        help="Bootstrap iterations for effect size CIs in analyze_results_and_report.py.",
    )
    parser.add_argument(
        "--skip-ecdf",
        action="store_true",
        help="Skip ECDF plots in plot_passive_quality_results.py.",
    )
    args = parser.parse_args()

    python_executable = sys.executable

    analyze_cmd = [
        python_executable,
        "analyze_passive_quality.py",
        "--four_g_dir",
        args.four_g_dir,
        "--five_g_dir",
        args.five_g_dir,
        "--pclip",
        args.pclip,
        "--speed_thresholds",
        args.speed_thresholds,
    ]
    if args.include_ow:
        analyze_cmd.append("--include_ow")

    report_cmd = [
        python_executable,
        "analyze_results_and_report.py",
        "--results_dir",
        "./results",
        "--alpha",
        str(args.alpha),
        "--bootstrap_iters",
        str(args.bootstrap_iters),
    ]

    plot_cmd = [
        python_executable,
        "plot_passive_quality_results.py",
        "--results_dir",
        "./results",
    ]
    if args.skip_ecdf:
        plot_cmd.append("--skip_ecdf")

    run_step(analyze_cmd, "Running analyze_passive_quality.py")
    run_step(report_cmd, "Running analyze_results_and_report.py")
    run_step(plot_cmd, "Running plot_passive_quality_results.py")

    print("\nPipeline completed successfully. Results are stored under ./results/")


if __name__ == "__main__":
    main()
