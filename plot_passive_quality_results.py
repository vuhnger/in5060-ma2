#!/usr/bin/env python3
"""
Generate matplotlib figures from the CSV outputs produced by analyze_passive_quality.py.
This script reads summaries under results/ and creates diagnostic plots without
recomputing upstream statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


SUMMARY_FILES = {
    "overall": "summary_overall.csv",
    "by_rat": "summary_by_rat.csv",
    "by_speed": "summary_by_speed.csv",
}

OPTIONAL_FILES = {
    "per_file": "per_file_stats.csv",
    "kpi_presence": "kpi_presence.csv",
}


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "mean_quality": "mean",
        "median_quality": "median",
        "std_quality": "std",
        "file_count": "count",
    }
    return df.rename(columns=rename_map)


def _normalize_per_file(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "mean_quality": "mean",
        "median_quality": "median",
        "std_quality": "std",
    }
    return df.rename(columns=rename_map)


def load_data(results_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    data: Dict[str, Optional[pd.DataFrame]] = {}
    for key, filename in SUMMARY_FILES.items():
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required summary file missing: {path}")
        df = pd.read_csv(path)
        data[key] = _normalize_summary(df)

    per_file_path = results_dir / OPTIONAL_FILES["per_file"]
    data["per_file"] = (
        _normalize_per_file(pd.read_csv(per_file_path)) if per_file_path.exists() else None
    )
    kpi_path = results_dir / OPTIONAL_FILES["kpi_presence"]
    data["kpi_presence"] = pd.read_csv(kpi_path) if kpi_path.exists() else None
    return data


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_overall_mean_bar(summary_overall: pd.DataFrame, output_path: Path) -> bool:
    subset = summary_overall[summary_overall["env"].isin(["inside", "outdoor_driving"])]
    if subset.empty:
        return False

    plt.figure(figsize=(6, 4))
    positions = np.arange(len(subset))
    plt.bar(positions, subset["mean"], yerr=subset["std"], capsize=6)
    plt.xticks(positions, subset["env"])
    plt.ylabel("NetworkQuality")
    plt.title("Mean NetworkQuality by Environment")
    _save_figure(output_path)
    return True


def plot_by_rat_mean_bar(summary_by_rat: pd.DataFrame, output_path: Path) -> bool:
    if summary_by_rat.empty:
        return False

    rats = sorted(summary_by_rat["rat"].dropna().unique())
    envs = ["inside", "outdoor_driving"]
    positions = np.arange(len(rats))
    width = 0.35

    plt.figure(figsize=(7, 4))
    for idx, env in enumerate(envs):
        env_data = summary_by_rat[summary_by_rat["env"] == env]
        means = []
        stds = []
        for rat in rats:
            row = env_data[env_data["rat"] == rat]
            if row.empty:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(row.iloc[0]["mean"])
                stds.append(row.iloc[0]["std"])
        bar_positions = positions + (idx - 0.5) * width
        plt.bar(bar_positions, means, width=width, yerr=stds, capsize=5, label=env)

    plt.xticks(positions, rats)
    plt.ylabel("NetworkQuality")
    plt.title("Mean NetworkQuality by RAT and Environment")
    plt.legend()
    _save_figure(output_path)
    return True


def plot_speed_buckets_bar(summary_by_speed: pd.DataFrame, output_path: Path) -> bool:
    if summary_by_speed.empty:
        return False

    order = ["inside", "od_quasi_static", "od_slow", "od_fast"]
    df = summary_by_speed.copy()
    df["order"] = df["speed_bucket"].apply(lambda x: order.index(x) if x in order else len(order))
    df.sort_values("order", inplace=True)

    plt.figure(figsize=(7, 4))
    positions = np.arange(len(df))
    plt.bar(positions, df["mean"], yerr=df["std"], capsize=5)
    plt.xticks(positions, df["speed_bucket"], rotation=20)
    plt.ylabel("NetworkQuality")
    plt.title("Mean NetworkQuality by Speed Bucket")
    _save_figure(output_path)
    return True


def plot_per_file_box_env(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or "env" not in per_file.columns:
        return False
    subset = per_file[per_file["env"].isin(["inside", "outdoor_driving"])]
    if subset.empty:
        return False

    plt.figure(figsize=(6, 4))
    data = [subset[subset["env"] == env]["mean"].dropna() for env in ["inside", "outdoor_driving"]]
    if all(series.empty for series in data):
        return False
    plt.boxplot(data, labels=["inside", "outdoor_driving"], showmeans=True)
    plt.ylabel("Per-file Mean NetworkQuality")
    plt.title("Per-file Means by Environment")
    _save_figure(output_path)
    return True


def plot_per_file_box_by_rat(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"env", "rat"}.issubset(per_file.columns):
        return False

    rats = sorted(per_file["rat"].dropna().unique())
    if not rats:
        return False

    plt.figure(figsize=(8, 4))
    data = []
    labels = []
    for rat in rats:
        for env in ["inside", "outdoor_driving"]:
            subset = per_file[(per_file["rat"] == rat) & (per_file["env"] == env)]["mean"].dropna()
            if subset.empty:
                continue
            data.append(subset)
            labels.append(f"{rat}-{env}")

    if not data:
        return False

    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Per-file Mean NetworkQuality")
    plt.title("Per-file Means by RAT and Environment")
    plt.xticks(rotation=30)
    _save_figure(output_path)
    return True


def plot_ecdf_env_by_rat(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"env", "rat"}.issubset(per_file.columns):
        return False

    rats = sorted(per_file["rat"].dropna().unique())
    if not rats:
        return False

    plt.figure(figsize=(7, 4))
    for rat in rats:
        for env in ["inside", "outdoor_driving"]:
            subset = per_file[(per_file["rat"] == rat) & (per_file["env"] == env)]["mean"].dropna()
            if subset.empty:
                continue
            sorted_vals = np.sort(subset)
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            plt.step(sorted_vals, ecdf, where="post", label=f"{rat}-{env}")

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc="lower right")
    plt.xlabel("Per-file Mean NetworkQuality")
    plt.ylabel("ECDF")
    plt.title("Per-file ECDF by RAT and Environment")
    _save_figure(output_path)
    return True


def plot_scatter_file_mean_vs_std(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"mean", "std"}.issubset(per_file.columns):
        return False

    plt.figure(figsize=(6, 4))
    jitter = np.random.default_rng(42).normal(scale=0.002, size=len(per_file))
    plt.scatter(per_file["mean"], per_file["std"] + jitter, alpha=0.7)
    plt.xlabel("Per-file Mean NetworkQuality")
    plt.ylabel("Per-file Std NetworkQuality")
    plt.title("Per-file Mean vs Std")
    _save_figure(output_path)
    return True


def plot_kpi_presence(kpi_presence: pd.DataFrame, output_path: Path) -> bool:
    if kpi_presence is None or "kpi" not in kpi_presence.columns:
        return False

    sorted_df = kpi_presence.sort_values("files_with_kpi", ascending=True)
    plt.figure(figsize=(8, 4))
    y_positions = np.arange(len(sorted_df))
    plt.barh(y_positions, sorted_df["files_with_kpi"])
    plt.yticks(y_positions, sorted_df["kpi"])
    plt.xlabel("Files with KPI")
    plt.title("KPI Availability (File Counts)")
    _save_figure(output_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matplotlib plots from passive quality summaries."
    )
    parser.add_argument(
        "--results_dir",
        default="./results",
        help="Directory containing CSV outputs from the upstream analysis.",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        default=True,
        help="If set, save all plots defined; default behaviour is to save all.",
    )
    parser.add_argument(
        "--skip_ecdf",
        action="store_true",
        default=False,
        help="If set, skip ECDF plots (useful if per_file_stats.csv is large or missing).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    try:
        data = load_data(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    created: List[str] = []
    skipped: List[str] = []

    output_map = {
        "plot_overall_mean_bar.png": lambda: plot_overall_mean_bar(
            data["overall"], results_dir / "plot_overall_mean_bar.png"
        ),
        "plot_by_rat_mean_bar.png": lambda: plot_by_rat_mean_bar(
            data["by_rat"], results_dir / "plot_by_rat_mean_bar.png"
        ),
        "plot_speed_buckets_bar.png": lambda: plot_speed_buckets_bar(
            data["by_speed"], results_dir / "plot_speed_buckets_bar.png"
        ),
        "plot_per_file_box_env.png": lambda: plot_per_file_box_env(
            data["per_file"], results_dir / "plot_per_file_box_env.png"
        ),
        "plot_per_file_box_by_rat.png": lambda: plot_per_file_box_by_rat(
            data["per_file"], results_dir / "plot_per_file_box_by_rat.png"
        ),
        "plot_ecdf_env_by_rat.png": lambda: plot_ecdf_env_by_rat(
            data["per_file"], results_dir / "plot_ecdf_env_by_rat.png"
        ),
        "plot_scatter_file_mean_vs_std.png": lambda: plot_scatter_file_mean_vs_std(
            data["per_file"], results_dir / "plot_scatter_file_mean_vs_std.png"
        ),
        "plot_kpi_presence.png": lambda: plot_kpi_presence(
            data["kpi_presence"], results_dir / "plot_kpi_presence.png"
        ),
    }

    for filename, plot_func in output_map.items():
        if filename == "plot_ecdf_env_by_rat.png" and args.skip_ecdf:
            skipped.append(filename)
            continue
        success = plot_func()
        if success:
            created.append(filename)
        else:
            skipped.append(filename)

    if created:
        print("Plots created:")
        for name in created:
            print(f"  - {name}")
    else:
        print("No plots created.")

    if skipped:
        print("Plots skipped or unavailable inputs:")
        for name in skipped:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
