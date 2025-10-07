#!/usr/bin/env python3
"""
This script analyzes passive cellular measurement logs to quantify network quality
differences between Indoor Static (IS) and Outdoor Driving (OD) environments across
4G and 5G. It discovers files from the 4G/ and 5G/ folders, excludes Active datasets
and archives, selects consistent RF KPIs, performs robust per-RAT normalization
(5th–95th percentile), computes a 0–1 NetworkQuality score, assigns OD speed buckets,
aggregates at the file level to avoid file-size bias, and outputs summary statistics
and optional plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency
    stats = None


KPI_PRIORITY: List[str] = ["DM_RS-SINR", "DM_RS-RSRP", "SSS-RSRP", "SSS-RSRQ"]
KPI_FALLBACK: List[str] = [
    "SINR",
    "RSRP",
    "RSRQ",
    "PBCH-SINR",
    "PBCH-RSRP",
    "PBCH-RSRQ",
    "PSS-SINR",
    "PSS-RSRP",
    "PSS-RSRQ",
]
OPTIONAL_COLUMNS: List[str] = ["Speed", "Band", "Latitude", "Longitude", "scenario"]

IGNORE_DIR_NAMES = {"Active", "__MACOSX"}
IGNORE_FILES = {
    "4G_2023_passive.zip",
    "5G_2023_passive.zip",
    "Active Performance Dataset - 1.zip",
}

ENV_PATTERNS = {
    "IS": re.compile(r"(?i)(?:^|[^a-z0-9])is(?:[^a-z0-9]|$)"),
    "OD": re.compile(r"(?i)(?:^|[^a-z0-9])od(?:[^a-z0-9]|$)"),
    "OW": re.compile(r"(?i)(?:^|[^a-z0-9])ow(?:[^a-z0-9]|$)"),
}

ENV_MAPPING = {"IS": "inside", "OD": "outdoor_driving", "OW": "outside_walking"}
INDOOR_LABEL = "inside"
OD_BUCKET_LABELS = ["od_quasi_static", "od_slow", "od_fast"]


@dataclass
class LoadResult:
    dataframe: pd.DataFrame
    kpi_columns: List[str]
    kpi_presence: pd.DataFrame


def select_kpis(columns: Sequence[str]) -> List[str]:
    """Select up to five KPI columns prioritising the primary list, then fallbacks."""
    available = set(columns)
    selected: List[str] = [col for col in KPI_PRIORITY if col in available]

    if len(selected) < 2:
        for fallback in KPI_FALLBACK:
            if fallback in available and fallback not in selected:
                selected.append(fallback)
            if len(selected) >= 5:
                break
    else:
        for fallback in KPI_FALLBACK:
            if fallback in available and fallback not in selected:
                selected.append(fallback)
            if len(selected) >= 5:
                break

    return selected


def infer_environment(file_path: Path) -> str | None:
    """Infer environment class (IS/OD/OW) from the filename using configured patterns."""
    stem = file_path.stem
    for tag, pattern in ENV_PATTERNS.items():
        if pattern.search(stem):
            return tag
    return None


def is_ignored(path: Path) -> bool:
    """Return True if the path should be ignored due to folder or filename rules."""
    if any(part in IGNORE_DIR_NAMES for part in path.parts):
        return True
    if path.name in IGNORE_FILES:
        return True
    if path.suffix.lower() != ".csv":
        return True
    return False


def load_passive_files(
    four_g_dir: Path,
    five_g_dir: Path,
    include_ow: bool,
) -> LoadResult:
    """
    Load CSVs from 4G/ and 5G/ only. Infer env from filename (is/od/optional ow),
    rat from folder. Keep KPI columns, Speed, Band, scenario if present. Return a
    single DataFrame with columns: chosen KPIs + env + rat + file + Speed (if exists).
    """
    frames: List[pd.DataFrame] = []
    all_kpis: List[str] = []
    kpi_presence_files: Dict[str, set[str]] = {}
    kpi_presence_rows: Dict[str, int] = {}

    for rat_dir, rat_label in ((four_g_dir, "4G"), (five_g_dir, "5G")):
        if not rat_dir.exists():
            continue

        for csv_path in sorted(rat_dir.rglob("*.csv")):
            if is_ignored(csv_path):
                continue

            env_tag = infer_environment(csv_path)
            if env_tag is None:
                continue
            if env_tag == "OW" and not include_ow:
                continue

            env_label = ENV_MAPPING[env_tag]

            try:
                df_raw = pd.read_csv(csv_path)
            except Exception as exc:  # pragma: no cover - IO guard
                print(f"Warning: failed to read {csv_path}: {exc}")
                continue

            kpis = select_kpis(df_raw.columns)
            if not kpis:
                continue

            keep_cols = [col for col in kpis if col in df_raw.columns]
            keep_cols.extend(col for col in OPTIONAL_COLUMNS if col in df_raw.columns)
            df_subset = df_raw.loc[:, keep_cols].copy()
            df_subset[kpis] = df_subset[kpis].apply(pd.to_numeric, errors="coerce")
            df_subset.dropna(subset=kpis, how="all", inplace=True)
            if df_subset.empty:
                continue

            for kpi in kpis:
                kpi_presence_files.setdefault(kpi, set()).add(csv_path.name)
                kpi_presence_rows[kpi] = kpi_presence_rows.get(kpi, 0) + int(
                    df_subset[kpi].notna().sum()
                )

            df_subset["env"] = env_label
            df_subset["rat"] = rat_label
            try:
                file_label = str(csv_path.relative_to(four_g_dir.parent))
            except ValueError:
                file_label = str(csv_path)
            df_subset["file"] = file_label

            frames.append(df_subset)
            for kpi in kpis:
                if kpi not in all_kpis:
                    all_kpis.append(kpi)

    if not frames:
        empty_df = pd.DataFrame(
            columns=["env", "rat", "file"] + KPI_PRIORITY + KPI_FALLBACK + OPTIONAL_COLUMNS
        )
        presence_df = pd.DataFrame(columns=["kpi", "files_with_kpi", "rows_with_kpi"])
        return LoadResult(empty_df, [], presence_df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    ordered_kpis: List[str] = [
        col for col in KPI_PRIORITY if col in combined.columns
    ] + [col for col in KPI_FALLBACK if col in combined.columns]
    ordered_kpis = list(dict.fromkeys(ordered_kpis))

    for col in ordered_kpis:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    optional_present = [col for col in OPTIONAL_COLUMNS if col in combined.columns]
    ordered_columns = ordered_kpis + optional_present + ["env", "rat", "file"]
    combined = combined.loc[:, ordered_columns]

    presence_records = []
    for kpi in ordered_kpis:
        presence_records.append(
            {
                "kpi": kpi,
                "files_with_kpi": len(kpi_presence_files.get(kpi, set())),
                "rows_with_kpi": kpi_presence_rows.get(kpi, 0),
            }
        )
    presence_df = pd.DataFrame(presence_records)

    return LoadResult(combined, ordered_kpis, presence_df)


def robust_minmax(series: pd.Series, pclip: Tuple[float, float]) -> pd.Series:
    """Normalize a numeric Series using robust min-max scaling (5th–95th percentiles)."""
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    result = pd.Series(0.5, index=series.index, dtype=float)

    if valid.empty:
        result.loc[values.isna()] = np.nan
        return result

    lower_q = valid.quantile(pclip[0])
    upper_q = valid.quantile(pclip[1])

    if not np.isfinite(lower_q) or not np.isfinite(upper_q) or upper_q <= lower_q:
        result.loc[values.isna()] = np.nan
        return result

    normalized = (values - lower_q) / (upper_q - lower_q)
    normalized = normalized.clip(0.0, 1.0).astype(float)
    normalized[values.isna()] = np.nan
    return normalized


def compute_network_quality(
    df: pd.DataFrame, kpi_columns: Sequence[str], pclip: Tuple[float, float]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Per-RAT normalization for each KPI via robust_minmax, then compute row-level
    NetworkQuality as the mean of normalized KPIs available.
    """
    df = df.copy()
    norm_columns: List[str] = []

    for kpi in kpi_columns:
        if kpi not in df.columns:
            continue
        norm_col = f"{kpi}_norm"
        df[norm_col] = (
            df.groupby("rat", group_keys=False)[kpi]
            .transform(lambda s: robust_minmax(s, pclip))
            .astype(float)
        )
        norm_columns.append(norm_col)

    if norm_columns:
        df["NetworkQuality"] = df[norm_columns].mean(axis=1, skipna=True)
        df.loc[df[norm_columns].isna().all(axis=1), "NetworkQuality"] = np.nan
    else:
        df["NetworkQuality"] = np.nan

    return df, norm_columns


def assign_speed_buckets(
    df: pd.DataFrame, thresholds: Tuple[float, float], indoor_label: str = INDOOR_LABEL
) -> pd.DataFrame:
    """Create `speed_bucket` for OD rows using thresholds; label IS rows as `inside`."""
    df = df.copy()
    low, high = thresholds

    if "Speed" in df.columns:
        speed = pd.to_numeric(df["Speed"], errors="coerce")
    else:
        speed = pd.Series(np.nan, index=df.index)

    speed_bucket = pd.Series(indoor_label, index=df.index, dtype=object)
    od_mask = df["env"] == "outdoor_driving"

    speed_bucket.loc[od_mask & (speed.isna() | (speed <= low))] = OD_BUCKET_LABELS[0]
    speed_bucket.loc[od_mask & (speed > low) & (speed <= high)] = OD_BUCKET_LABELS[1]
    speed_bucket.loc[od_mask & (speed > high)] = OD_BUCKET_LABELS[2]

    ow_mask = df["env"] == "outside_walking"
    speed_bucket.loc[ow_mask] = "outside_walking"

    df["speed_bucket"] = speed_bucket
    return df


def aggregate_per_file(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (rat, env/speed_bucket, file), compute mean, median, std of NetworkQuality.
    Returns (per_file_env, per_file_speed) DataFrames.
    """
    if df.empty or "NetworkQuality" not in df.columns:
        empty_env = pd.DataFrame(columns=["rat", "env", "file", "mean_quality", "median_quality", "std_quality", "sample_count"])
        empty_speed = pd.DataFrame(columns=["rat", "speed_bucket", "file", "mean_quality", "median_quality", "std_quality", "sample_count"])
        return empty_env, empty_speed

    agg_spec = ["mean", "median", "std", "count"]

    env_stats = (
        df.groupby(["rat", "env", "file"])["NetworkQuality"]
        .agg(agg_spec)
        .reset_index()
        .rename(
            columns={
                "mean": "mean_quality",
                "median": "median_quality",
                "std": "std_quality",
                "count": "sample_count",
            }
        )
    )
    env_stats = env_stats[env_stats["sample_count"] > 0]

    speed_stats = (
        df.groupby(["rat", "speed_bucket", "file"])["NetworkQuality"]
        .agg(agg_spec)
        .reset_index()
        .rename(
            columns={
                "mean": "mean_quality",
                "median": "median_quality",
                "std": "std_quality",
                "count": "sample_count",
            }
        )
    )
    speed_stats = speed_stats[speed_stats["sample_count"] > 0]

    return env_stats, speed_stats


def _summarise(
    df: pd.DataFrame, group_cols: Sequence[str]
) -> pd.DataFrame:
    """Helper to summarise per-file statistics across groups."""
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "mean_quality", "median_quality", "std_quality", "file_count"])

    summary = (
        df.groupby(list(group_cols))["mean_quality"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_quality",
                "median": "median_quality",
                "std": "std_quality",
                "count": "file_count",
            }
        )
    )
    return summary


def summarize_groups(
    per_file_env: pd.DataFrame, per_file_speed: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aggregate per-file stats into group-level summaries for overall, by RAT, and by
    speed buckets; return DataFrames (overall, by_rat, by_speed_bucket).
    """
    overall = _summarise(per_file_env, ["env"])
    by_rat = _summarise(per_file_env, ["rat", "env"])
    by_speed = _summarise(per_file_speed, ["speed_bucket"])

    if not overall.empty:
        overall["env"] = pd.Categorical(
            overall["env"], categories=["inside", "outdoor_driving", "outside_walking"], ordered=True
        )
        overall = overall.sort_values("env").reset_index(drop=True)

    if not by_rat.empty:
        by_rat["env"] = pd.Categorical(
            by_rat["env"], categories=["inside", "outdoor_driving", "outside_walking"], ordered=True
        )
        by_rat = by_rat.sort_values(["rat", "env"]).reset_index(drop=True)

    if not by_speed.empty:
        desired_order = ["inside", *OD_BUCKET_LABELS, "outside_walking"]
        by_speed["speed_bucket"] = pd.Categorical(
            by_speed["speed_bucket"], categories=desired_order, ordered=True
        )
        by_speed = by_speed.sort_values("speed_bucket").reset_index(drop=True)

    return overall, by_rat, by_speed


def save_results(
    results_dir: Path,
    summary_overall: pd.DataFrame,
    summary_by_rat: pd.DataFrame,
    summary_by_speed: pd.DataFrame,
) -> None:
    """Create results/ if missing; write summary CSVs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_overall.to_csv(results_dir / "summary_overall.csv", index=False)
    summary_by_rat.to_csv(results_dir / "summary_by_rat.csv", index=False)
    summary_by_speed.to_csv(results_dir / "summary_by_speed.csv", index=False)


def plot_optionals(
    df: pd.DataFrame,
    summary_by_rat: pd.DataFrame,
    results_dir: Path,
    save_plots: bool,
) -> None:
    """
    If `save_plots` is set, produce matplotlib plots (no seaborn): boxplot of
    NetworkQuality by env, and bar chart of mean±std by RAT.
    """
    if not save_plots:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"Warning: matplotlib not available ({exc}); skipping plots.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    env_order = ["inside", "outdoor_driving", "outside_walking", "od_unknown"]
    env_data = [df.loc[df["env"] == env, "NetworkQuality"].dropna() for env in env_order]
    env_labels = [label for label, data in zip(env_order, env_data) if not data.empty]
    env_data = [data for data in env_data if not data.empty]

    if env_data:
        plt.figure(figsize=(8, 5))
        plt.boxplot(env_data, labels=env_labels, showmeans=True)
        plt.ylabel("NetworkQuality (0-1)")
        plt.title("Network Quality Distribution by Environment")
        plt.tight_layout()
        plt.savefig(results_dir / "boxplot_quality_by_env.png", dpi=200)
        plt.close()

    if not summary_by_rat.empty:
        plt.figure(figsize=(6, 4))
        rats = summary_by_rat["rat"].unique()
        means = []
        stds = []
        for rat in rats:
            rat_rows = df[df["rat"] == rat]["NetworkQuality"].dropna()
            if rat_rows.empty:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(rat_rows.mean())
                stds.append(rat_rows.std(ddof=1))

        positions = np.arange(len(rats))
        plt.bar(positions, means, yerr=stds, capsize=6, color=["#1f77b4", "#ff7f0e"])
        plt.xticks(positions, rats)
        plt.ylabel("NetworkQuality (0-1)")
        plt.title("Mean Network Quality by RAT")
        plt.tight_layout()
        plt.savefig(results_dir / "bar_quality_mean_std_by_rat.png", dpi=200)
        plt.close()


def print_table(title: str, df: pd.DataFrame) -> None:
    """Pretty-print a DataFrame with a heading."""
    print(f"\n{title}")
    if df.empty:
        print("  No data available.")
        return

    display_df = df.copy()
    for col in display_df.select_dtypes(include=["float", "float64"]).columns:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    print(display_df.to_string(index=False))


def maybe_print_ttests(per_file_env: pd.DataFrame) -> None:
    """Optionally print Welch t-tests comparing IS vs OD per RAT if scipy is available."""
    if stats is None or per_file_env.empty:
        return

    print("\nWelch t-test (per RAT, IS vs OD):")

    rats = sorted(per_file_env["rat"].unique())
    for rat in rats:
        inside = per_file_env[
            (per_file_env["rat"] == rat) & (per_file_env["env"] == "inside")
        ]["mean_quality"]
        outdoor = per_file_env[
            (per_file_env["rat"] == rat) & (per_file_env["env"] == "outdoor_driving")
        ]["mean_quality"]

        if len(inside) < 2 or len(outdoor) < 2:
            print(f"  {rat}: insufficient data for t-test.")
            continue

        t_stat, p_value = stats.ttest_ind(inside, outdoor, equal_var=False, nan_policy="omit")
        print(f"  {rat}: t={t_stat:.3f}, p={p_value:.4f}")


def parse_percentile_clip(pclip_arg: str) -> Tuple[float, float]:
    """Parse percentile clip argument."""
    parts = [p.strip() for p in pclip_arg.split(",")]
    if len(parts) != 2:
        raise ValueError("Percentile clip must be provided as 'low,high'.")
    low, high = float(parts[0]), float(parts[1])
    if not 0.0 <= low < high <= 1.0:
        raise ValueError("Percentile clip values must satisfy 0 <= low < high <= 1.")
    return low, high


def parse_speed_thresholds(arg: str) -> Tuple[float, float]:
    """Parse OD speed thresholds argument."""
    parts = [p.strip() for p in arg.split(",")]
    if len(parts) != 2:
        raise ValueError("Speed thresholds must be provided as 'low,high'.")
    low, high = float(parts[0]), float(parts[1])
    if not 0.0 <= low <= high:
        raise ValueError("Speed thresholds must satisfy 0 <= low <= high.")
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze passive 4G/5G indoor vs outdoor network quality."
    )
    parser.add_argument("--four_g_dir", default="./4G", help="Path to 4G passive CSV folder.")
    parser.add_argument("--five_g_dir", default="./5G", help="Path to 5G passive CSV folder.")
    parser.add_argument(
        "--include_ow",
        action="store_true",
        default=False,
        help="Include outside walking (OW) files if present.",
    )
    parser.add_argument(
        "--pclip",
        default="0.05,0.95",
        help="Robust percentile clip for normalization, e.g., '0.05,0.95'.",
    )
    parser.add_argument(
        "--speed_thresholds",
        default="0.5,5",
        help="OD speed thresholds in m/s as 'low,high' (inclusive bounds at low, high).",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Save optional matplotlib figures to results/.",
    )

    args = parser.parse_args()

    four_g_dir = Path(args.four_g_dir)
    five_g_dir = Path(args.five_g_dir)

    print("Starting passive quality analysis...")
    print(f"  4G directory: {four_g_dir}")
    print(f"  5G directory: {five_g_dir}")
    print(f"  Include OW: {'yes' if args.include_ow else 'no'}")

    pclip = parse_percentile_clip(args.pclip)
    speed_thresholds = parse_speed_thresholds(args.speed_thresholds)
    print(f"  Percentile clip: {pclip[0]:.2f}–{pclip[1]:.2f}")
    print(f"  OD speed thresholds (m/s): {speed_thresholds[0]:.2f}, {speed_thresholds[1]:.2f}")

    print("Loading passive measurement files...")
    load_result = load_passive_files(four_g_dir, five_g_dir, include_ow=args.include_ow)
    df = load_result.dataframe

    if df.empty:
        print("No passive data found under the provided directories.")
        return

    total_rows = len(df)
    unique_files = df["file"].nunique()
    print(f"  Loaded {unique_files} files with {total_rows} usable rows.")
    if load_result.kpi_columns:
        print(f"  KPI columns selected: {', '.join(load_result.kpi_columns)}")

    df = assign_speed_buckets(df, speed_thresholds)
    print("  Assigned speed buckets.")

    print("Computing NetworkQuality scores...")
    df, norm_columns = compute_network_quality(df, load_result.kpi_columns, pclip)

    print("Aggregating per-file statistics...")
    per_file_env, per_file_speed = aggregate_per_file(df)
    summary_overall, summary_by_rat, summary_by_speed = summarize_groups(per_file_env, per_file_speed)

    print_table("Overall IS vs OD (all RATs combined)", summary_overall)
    print_table("By RAT (4G / 5G): IS vs OD", summary_by_rat)
    print_table("By speed bucket (4G+5G)", summary_by_speed)

    if not load_result.kpi_presence.empty:
        print_table("KPI availability across files", load_result.kpi_presence)

    maybe_print_ttests(per_file_env)

    results_dir = Path("results")
    print(f"Writing summaries to {results_dir}/ ...")
    save_results(results_dir, summary_overall, summary_by_rat, summary_by_speed)
    plot_optionals(df, summary_by_rat, results_dir, save_plots=args.save_plots)

    if norm_columns:
        print(f"\nComputed NetworkQuality using normalized KPIs: {', '.join(norm_columns)}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
