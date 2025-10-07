#!/usr/bin/env python3
"""
Generate matplotlib figures from the CSV outputs produced by analyze_passive_quality.py.
This script reads summaries under results/ and creates diagnostic plots (bar charts,
boxplots, violin plots, ECDF curves, scatter diagnostics, KPI availability) without
recomputing upstream statistics. Axes and labels adapt automatically depending on
whether the inputs are normalized (0–1) or raw KPI averages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None


SUMMARY_FILES = {
    "overall": "summary_overall.csv",
    "by_rat": "summary_by_rat.csv",
    "by_speed": "summary_by_speed.csv",
}

OPTIONAL_FILES = {
    "per_file": "per_file_stats.csv",
    "per_file_speed": "per_file_stats_by_speed.csv",
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
    per_file_speed_path = results_dir / OPTIONAL_FILES["per_file_speed"]
    data["per_file_speed"] = (
        _normalize_per_file(pd.read_csv(per_file_speed_path)) if per_file_speed_path.exists() else None
    )
    kpi_path = results_dir / OPTIONAL_FILES["kpi_presence"]
    data["kpi_presence"] = pd.read_csv(kpi_path) if kpi_path.exists() else None
    return data


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _value_limits(values: Iterable[float], errors: Optional[Iterable[float]] = None) -> Optional[Tuple[float, float]]:
    values_series = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float).dropna()
    if values_series.empty:
        return None
    if errors is not None:
        errors_series = pd.Series(pd.to_numeric(errors, errors="coerce"), dtype=float).reindex(values_series.index, fill_value=0.0)
        lower = (values_series - errors_series).dropna()
        upper = (values_series + errors_series).dropna()
        if lower.empty or upper.empty:
            finite = values_series
        else:
            finite = pd.concat([lower, upper], ignore_index=True)
    else:
        finite = values_series
    finite = finite.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return None
    return float(finite.min()), float(finite.max())


def _is_normalized(values: Iterable[float]) -> bool:
    finite = pd.Series(pd.to_numeric(values, errors="coerce"), dtype=float).dropna()
    if finite.empty:
        return True
    return finite.min() >= -0.05 and finite.max() <= 1.05


def _range_text(limits: Optional[Tuple[float, float]]) -> str:
    if not limits:
        return ""
    return f"{limits[0]:.1f}..{limits[1]:.1f}"


def _axis_label(base: str, normalized: bool, limits: Optional[Tuple[float, float]]) -> str:
    if normalized:
        return f"{base} (0-1)"
    if limits:
        return f"{base} (raw range {_range_text(limits)})"
    return f"{base} (raw units)"


def _expanded_limits(limits: Optional[Tuple[float, float]], normalized: bool) -> Optional[Tuple[float, float]]:
    if limits is None:
        return None
    low, high = limits
    if normalized:
        low_lim = min(0.0, low - 0.05)
        high_lim = max(1.0, high + 0.05)
        return (low_lim, high_lim)
    span = high - low
    if span == 0:
        padding = max(1.0, max(abs(high), abs(low), 1.0) * 0.1)
    else:
        padding = max(1.0, abs(span) * 0.1)
    return (low - padding, high + padding)


def _apply_axis_meta(
    ax: plt.Axes,
    values: Iterable[float],
    errors: Optional[Iterable[float]] = None,
    base_label: str = "NetworkQuality",
    axis: str = "y",
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    limits = _value_limits(values, errors)
    normalized = _is_normalized(values)
    label = _axis_label(base_label, normalized, limits)
    expanded = _expanded_limits(limits, normalized)

    if axis == "y":
        ax.set_ylabel(label)
        if expanded:
            ax.set_ylim(expanded)
    else:
        ax.set_xlabel(label)
        if expanded:
            ax.set_xlim(expanded)

    return normalized, limits


def _welch_pvalue(samples_a: pd.Series, samples_b: pd.Series) -> Optional[float]:
    if stats is None:
        return None
    a = pd.to_numeric(samples_a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(samples_b, errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return None
    _, p_value = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return float(p_value)


def plot_overall_mean_bar(
    summary_overall: pd.DataFrame,
    per_file: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    subset = summary_overall[summary_overall["env"].isin(["inside", "outdoor_driving"])]
    if subset.empty:
        return False

    inside_mean = subset.loc[subset["env"] == "inside", "mean"].iloc[0]
    outdoor_mean = subset.loc[subset["env"] == "outdoor_driving", "mean"].iloc[0]
    delta = inside_mean - outdoor_mean
    pct = np.nan
    if np.isfinite(outdoor_mean) and outdoor_mean != 0:
        pct = delta / outdoor_mean

    p_value = None
    if per_file is not None:
        inside_samples = per_file.loc[per_file["env"] == "inside", "mean"]
        outdoor_samples = per_file.loc[per_file["env"] == "outdoor_driving", "mean"]
        p_value = _welch_pvalue(inside_samples, outdoor_samples)

    plt.figure(figsize=(6, 4))
    positions = np.arange(len(subset))
    plt.bar(positions, subset["mean"], yerr=subset["std"], capsize=6)
    plt.xticks(positions, subset["env"])
    plt.xlabel("Environment")

    ax = plt.gca()
    normalized, limits = _apply_axis_meta(ax, subset["mean"], subset["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Mean NetworkQuality by Environment {scale_suffix}")

    ymin, ymax = ax.get_ylim()
    ax.text(
        0.5,
        ymax - (ymax - ymin) * 0.05,
        f"Δ={delta:.3f} ({_format_percent(pct)}) • p={_format_pvalue(p_value)}",
        ha="center",
        va="top",
    )

    _save_figure(output_path)
    return True


def _format_percent(delta_pct: float) -> str:
    if not np.isfinite(delta_pct):
        return "n/a"
    return f"{delta_pct*100:.1f}%"


def _format_pvalue(p_value: Optional[float]) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-4:
        return "<1e-4"
    return f"{p_value:.4f}"


def plot_by_rat_mean_bar(
    summary_by_rat: pd.DataFrame,
    per_file: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    if summary_by_rat.empty:
        return False

    rats = sorted(summary_by_rat["rat"].dropna().unique())
    envs = ["inside", "outdoor_driving"]
    positions = np.arange(len(rats))
    width = 0.35

    plt.figure(figsize=(max(7, len(rats) * 1.5), 4))
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
    plt.xlabel("RAT")

    ax = plt.gca()
    normalized, limits = _apply_axis_meta(ax, summary_by_rat["mean"], summary_by_rat["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Mean NetworkQuality by RAT and Environment {scale_suffix}")
    ax.legend()

    annotation_lines: List[str] = []
    if per_file is not None:
        for rat in rats:
            inside_mean = summary_by_rat.loc[
                (summary_by_rat["rat"] == rat) & (summary_by_rat["env"] == "inside"),
                "mean",
            ]
            outdoor_mean = summary_by_rat.loc[
                (summary_by_rat["rat"] == rat) & (summary_by_rat["env"] == "outdoor_driving"),
                "mean",
            ]
            if inside_mean.empty or outdoor_mean.empty:
                continue
            delta = inside_mean.iloc[0] - outdoor_mean.iloc[0]
            pct = np.nan
            if np.isfinite(outdoor_mean.iloc[0]) and outdoor_mean.iloc[0] != 0:
                pct = delta / outdoor_mean.iloc[0]
            inside_samples = per_file.loc[
                (per_file["rat"] == rat) & (per_file["env"] == "inside"), "mean"
            ]
            outdoor_samples = per_file.loc[
                (per_file["rat"] == rat) & (per_file["env"] == "outdoor_driving"), "mean"
            ]
            p_value = _welch_pvalue(inside_samples, outdoor_samples)
            annotation_lines.append(
                f"{rat}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )

    if annotation_lines:
        ax.text(
            1.02,
            0.95,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )

    _save_figure(output_path)
    return True


def plot_speed_buckets_bar(
    summary_by_speed: pd.DataFrame,
    per_file_speed: Optional[pd.DataFrame],
    output_path: Path,
) -> bool:
    if summary_by_speed.empty:
        return False

    order = ["inside", "od_quasi_static", "od_slow", "od_fast"]
    df = summary_by_speed.copy()
    df["order"] = df["speed_bucket"].apply(lambda x: order.index(x) if x in order else len(order))
    df.sort_values("order", inplace=True)

    plt.figure(figsize=(max(7, len(df) * 1.3), 4))
    positions = np.arange(len(df))
    plt.bar(positions, df["mean"], yerr=df["std"], capsize=5)
    plt.xticks(positions, df["speed_bucket"], rotation=20)
    plt.xlabel("Speed Bucket")

    ax = plt.gca()
    normalized, limits = _apply_axis_meta(ax, df["mean"], df["std"])
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(
        f"Mean NetworkQuality by Speed Bucket {scale_suffix}\n"
        "inside (indoor); od_quasi_static ≤0.5 m/s; od_slow 0.5–5 m/s; od_fast >5 m/s"
    )

    if per_file_speed is not None:
        inside_samples = per_file_speed.loc[per_file_speed["speed_bucket"] == "inside", "mean"]
        annotation_lines: List[str] = []
        for bucket in df["speed_bucket"]:
            if bucket == "inside":
                continue
            bucket_mean = df.loc[df["speed_bucket"] == bucket, "mean"].iloc[0]
            inside_mean = df.loc[df["speed_bucket"] == "inside", "mean"].iloc[0]
            delta = inside_mean - bucket_mean
            pct = np.nan
            if np.isfinite(bucket_mean) and bucket_mean != 0:
                pct = delta / bucket_mean
            bucket_samples = per_file_speed.loc[per_file_speed["speed_bucket"] == bucket, "mean"]
            p_value = _welch_pvalue(inside_samples, bucket_samples)
            annotation_lines.append(
                f"inside vs {bucket}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )

        if annotation_lines:
            ax.text(
                1.02,
                0.95,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )

    _save_figure(output_path)
    return True


def plot_per_file_box_env(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or "env" not in per_file.columns:
        return False
    subset = per_file[per_file["env"].isin(["inside", "outdoor_driving"])]
    if subset.empty:
        return False

    data = [subset[subset["env"] == env]["mean"].dropna() for env in ["inside", "outdoor_driving"]]
    if any(series.empty for series in data):
        return False

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=["inside", "outdoor_driving"], showmeans=True)
    plt.xlabel("Environment")

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Distribution by Environment {scale_suffix}")

    inside_samples, outdoor_samples = data
    p_value = _welch_pvalue(inside_samples, outdoor_samples)
    delta = inside_samples.mean() - outdoor_samples.mean()
    pct = np.nan
    if np.isfinite(outdoor_samples.mean()) and outdoor_samples.mean() != 0:
        pct = delta / outdoor_samples.mean()
    ymin, ymax = ax.get_ylim()
    ax.text(
        0.5,
        ymax - (ymax - ymin) * 0.05,
        f"Δ={delta:.3f} ({_format_percent(pct)}) • p={_format_pvalue(p_value)}",
        ha="center",
        va="top",
    )

    _save_figure(output_path)
    return True


def plot_per_file_box_by_rat(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"env", "rat"}.issubset(per_file.columns):
        return False

    rats = sorted(per_file["rat"].dropna().unique())
    if not rats:
        return False

    plt.figure(figsize=(max(8, len(rats) * 2.0), 4))
    data: List[pd.Series] = []
    labels: List[str] = []
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
    plt.xlabel("RAT and Environment")

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Distribution by RAT and Environment {scale_suffix}")

    annotation_lines = [
        f"{label}: μ={series.mean():.3f}, σ={series.std(ddof=1):.3f}, n={len(series)}"
        for label, series in zip(labels, data)
    ]
    ax.text(
        1.02,
        0.95,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    _save_figure(output_path)
    return True


def plot_ecdf_env_by_rat(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"env", "rat"}.issubset(per_file.columns):
        return False

    rats = sorted(per_file["rat"].dropna().unique())
    if not rats:
        return False

    plt.figure(figsize=(7, 4))
    series_bundle: List[pd.Series] = []
    for rat in rats:
        for env in ["inside", "outdoor_driving"]:
            subset = per_file[(per_file["rat"] == rat) & (per_file["env"] == env)]["mean"].dropna()
            if subset.empty:
                continue
            sorted_vals = np.sort(subset)
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            plt.step(sorted_vals, ecdf, where="post", label=f"{rat}-{env}")
            series_bundle.append(pd.Series(sorted_vals))

    if not series_bundle:
        return False

    ax = plt.gca()
    normalized, limits = _apply_axis_meta(
        ax,
        pd.concat(series_bundle, ignore_index=True),
        base_label="Per-file Mean NetworkQuality",
        axis="x",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_ylabel("ECDF (0-1)")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right")
    ax.set_title(f"Per-file ECDF by RAT and Environment {scale_suffix}")

    _save_figure(output_path)
    return True


def plot_scatter_file_mean_vs_std(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"mean", "std"}.issubset(per_file.columns):
        return False
    if per_file.empty:
        return False

    plt.figure(figsize=(6, 4))
    jitter = np.random.default_rng(42).normal(scale=0.002, size=len(per_file))
    plt.scatter(per_file["mean"], per_file["std"] + jitter, alpha=0.7)

    ax = plt.gca()
    mean_normalized, mean_limits = _apply_axis_meta(
        ax,
        per_file["mean"],
        base_label="Per-file Mean NetworkQuality",
        axis="x",
    )
    std_normalized, std_limits = _apply_axis_meta(
        ax,
        per_file["std"],
        base_label="Per-file Std of NetworkQuality",
        axis="y",
    )
    scale_suffix = "(normalized)" if mean_normalized else "(raw)"
    ax.set_title(f"Per-file Mean vs Std {scale_suffix}")

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
    plt.xlabel("Files with KPI (count)")
    plt.title("KPI Availability (File Counts)")
    _save_figure(output_path)
    return True


def plot_per_file_violin_env(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or "env" not in per_file.columns:
        return False

    groups = ["inside", "outdoor_driving"]
    data = [per_file.loc[per_file["env"] == env, "mean"].dropna() for env in groups]
    if any(series.empty for series in data):
        return False

    plt.figure(figsize=(6, 4))
    parts = plt.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    plt.xticks(np.arange(1, len(groups) + 1), groups)
    plt.xlabel("Environment")

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Violin by Environment {scale_suffix}")

    inside_samples, outdoor_samples = data
    p_value = _welch_pvalue(inside_samples, outdoor_samples)
    delta = inside_samples.mean() - outdoor_samples.mean()
    pct = np.nan
    if np.isfinite(outdoor_samples.mean()) and outdoor_samples.mean() != 0:
        pct = delta / outdoor_samples.mean()
    ymin, ymax = ax.get_ylim()
    ax.text(
        0.5,
        ymax - (ymax - ymin) * 0.05,
        f"Δ={delta:.3f} ({_format_percent(pct)}) • p={_format_pvalue(p_value)}",
        ha="center",
        va="top",
    )

    _save_figure(output_path)
    return True


def plot_per_file_violin_by_rat(per_file: pd.DataFrame, output_path: Path) -> bool:
    if per_file is None or not {"env", "rat"}.issubset(per_file.columns):
        return False

    combinations: List[str] = []
    data: List[pd.Series] = []
    for rat in sorted(per_file["rat"].dropna().unique()):
        for env in ["inside", "outdoor_driving"]:
            subset = per_file[(per_file["rat"] == rat) & (per_file["env"] == env)]["mean"].dropna()
            if subset.empty:
                continue
            combinations.append(f"{rat}-{env}")
            data.append(subset)

    if not data:
        return False

    plt.figure(figsize=(max(8, len(data) * 1.5), 4))
    parts = plt.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    plt.xticks(np.arange(1, len(combinations) + 1), combinations, rotation=30)
    plt.xlabel("RAT and Environment")

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(f"Per-file NetworkQuality Violin by RAT and Environment {scale_suffix}")

    annotation_lines = [
        f"{combo}: μ={series.mean():.3f}, σ={series.std(ddof=1):.3f}, n={len(series)}"
        for combo, series in zip(combinations, data)
    ]
    ax.text(
        1.02,
        0.95,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )

    _save_figure(output_path)
    return True


def plot_speed_bucket_violin(per_file_speed: pd.DataFrame, output_path: Path) -> bool:
    if per_file_speed is None or "speed_bucket" not in per_file_speed.columns:
        return False

    order = ["inside", "od_quasi_static", "od_slow", "od_fast"]
    data = []
    labels = []
    for bucket in order:
        subset = per_file_speed[per_file_speed["speed_bucket"] == bucket]["mean"].dropna()
        if subset.empty:
            continue
        labels.append(bucket)
        data.append(subset)

    if not data:
        return False

    plt.figure(figsize=(7, 4))
    parts = plt.violinplot(data, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    plt.xticks(np.arange(1, len(labels) + 1), labels)
    plt.xlabel("Speed Bucket")

    ax = plt.gca()
    combined = pd.concat(data, ignore_index=True)
    normalized, limits = _apply_axis_meta(
        ax,
        combined,
        base_label="Per-file Mean NetworkQuality",
    )
    scale_suffix = "(normalized)" if normalized else "(raw)"
    ax.set_title(
        f"Per-file NetworkQuality Violin by Speed Bucket {scale_suffix}\n"
        "inside (indoor); od_quasi_static ≤0.5 m/s; od_slow 0.5–5 m/s; od_fast >5 m/s"
    )

    if "inside" in labels:
        inside_index = labels.index("inside")
        inside_samples = data[inside_index]
        annotation_lines: List[str] = []
        for label, sample in zip(labels, data):
            if label == "inside":
                continue
            delta = inside_samples.mean() - sample.mean()
            pct = np.nan
            if np.isfinite(sample.mean()) and sample.mean() != 0:
                pct = delta / sample.mean()
            p_value = _welch_pvalue(inside_samples, sample)
            annotation_lines.append(
                f"inside vs {label}: Δ={delta:.3f} ({_format_percent(pct)}), p={_format_pvalue(p_value)}"
            )
        if annotation_lines:
            ax.text(
                1.02,
                0.95,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )

    _save_figure(output_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate matplotlib plots from passive quality summaries."
    )
    parser.add_argument(
        "--results_dir",
        default="./results_normalized",
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
            data["overall"], data["per_file"], results_dir / "plot_overall_mean_bar.png"
        ),
        "plot_by_rat_mean_bar.png": lambda: plot_by_rat_mean_bar(
            data["by_rat"], data["per_file"], results_dir / "plot_by_rat_mean_bar.png"
        ),
        "plot_speed_buckets_bar.png": lambda: plot_speed_buckets_bar(
            data["by_speed"], data["per_file_speed"], results_dir / "plot_speed_buckets_bar.png"
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
        "plot_per_file_violin_env.png": lambda: plot_per_file_violin_env(
            data["per_file"], results_dir / "plot_per_file_violin_env.png"
        ),
        "plot_per_file_violin_by_rat.png": lambda: plot_per_file_violin_by_rat(
            data["per_file"], results_dir / "plot_per_file_violin_by_rat.png"
        ),
        "plot_speed_bucket_violin.png": lambda: plot_speed_bucket_violin(
            data["per_file_speed"], results_dir / "plot_speed_bucket_violin.png"
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
