#!/usr/bin/env python3
"""
Analyze downstream results produced by analyze_passive_quality.py. The script
ingests summary CSVs under results/, optionally leverages per-file statistics,
computes comparative metrics (mean deltas, effect sizes, Welch tests), ranks
speed buckets, and writes a concise Markdown summary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

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

EXPECTED_PLOTS = [
    "plot_overall_mean_bar.png",
    "plot_by_rat_mean_bar.png",
    "plot_speed_buckets_bar.png",
    "plot_per_file_box_env.png",
    "plot_per_file_box_by_rat.png",
    "plot_ecdf_env_by_rat.png",
    "plot_scatter_file_mean_vs_std.png",
    "plot_kpi_presence.png",
    "plot_per_file_violin_env.png",
    "plot_per_file_violin_by_rat.png",
    "plot_speed_bucket_violin.png",
]


@dataclass
class LoadedData:
    summary_overall: pd.DataFrame
    summary_by_rat: pd.DataFrame
    summary_by_speed: pd.DataFrame
    per_file_stats: Optional[pd.DataFrame]
    per_file_speed_stats: Optional[pd.DataFrame]
    kpi_presence: Optional[pd.DataFrame]


def _normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rename summary columns to canonical names."""
    rename_map = {
        "mean_quality": "mean",
        "median_quality": "median",
        "std_quality": "std",
        "file_count": "count",
    }
    df = df.rename(columns=rename_map)
    return df


def _normalize_per_file(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise per-file stats column names."""
    rename_map = {
        "mean_quality": "mean",
        "median_quality": "median",
        "std_quality": "std",
    }
    df = df.rename(columns=rename_map)
    return df


def load_summaries(results_dir: Path) -> LoadedData:
    """
    Load summary_overall, summary_by_rat, summary_by_speed; optionally per_file_stats
    and kpi_presence.
    """
    data: Dict[str, pd.DataFrame] = {}
    for key, filename in SUMMARY_FILES.items():
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required summary file missing: {path}")
        df = pd.read_csv(path)
        data[key] = _normalize_summary(df)

    per_file_df = None
    per_file_path = results_dir / OPTIONAL_FILES["per_file"]
    if per_file_path.exists():
        per_file_df = _normalize_per_file(pd.read_csv(per_file_path))

    per_file_speed_df = None
    per_file_speed_path = results_dir / OPTIONAL_FILES["per_file_speed"]
    if per_file_speed_path.exists():
        per_file_speed_df = _normalize_per_file(pd.read_csv(per_file_speed_path))

    kpi_presence_df = None
    kpi_presence_path = results_dir / OPTIONAL_FILES["kpi_presence"]
    if kpi_presence_path.exists():
        kpi_presence_df = pd.read_csv(kpi_presence_path)

    return LoadedData(
        summary_overall=data["overall"],
        summary_by_rat=data["by_rat"],
        summary_by_speed=data["by_speed"],
        per_file_stats=per_file_df,
        per_file_speed_stats=per_file_speed_df,
        kpi_presence=kpi_presence_df,
    )


def compute_effect_size(
    sample_inside: pd.Series,
    sample_outdoor: pd.Series,
    bootstrap_iters: int,
    rng: np.random.Generator,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Compute Hedges' g (bias-corrected Cohen's d) for inside vs outdoor_driving
    using per-file mean samples. Optionally bootstrap confidence intervals.
    """
    inside = sample_inside.dropna().to_numpy()
    outdoor = sample_outdoor.dropna().to_numpy()
    if inside.size < 2 or outdoor.size < 2:
        return None, None

    mean_inside = inside.mean()
    mean_outdoor = outdoor.mean()
    var_inside = inside.var(ddof=1)
    var_outdoor = outdoor.var(ddof=1)
    pooled_denom = ((inside.size - 1) * var_inside + (outdoor.size - 1) * var_outdoor)
    pooled_denom /= (inside.size + outdoor.size - 2)
    if pooled_denom <= 0:
        return None, None

    pooled_std = np.sqrt(pooled_denom)
    cohen_d = (mean_inside - mean_outdoor) / pooled_std
    correction = 1.0
    total = inside.size + outdoor.size
    if total > 2:
        correction = 1 - (3 / (4 * total - 9))
    hedges_g = cohen_d * correction

    if bootstrap_iters <= 0:
        return hedges_g, None

    bootstrap_values: List[float] = []
    for _ in range(bootstrap_iters):
        resample_inside = rng.choice(inside, size=inside.size, replace=True)
        resample_outdoor = rng.choice(outdoor, size=outdoor.size, replace=True)
        r_var_inside = resample_inside.var(ddof=1)
        r_var_outdoor = resample_outdoor.var(ddof=1)
        pooled = ((inside.size - 1) * r_var_inside + (outdoor.size - 1) * r_var_outdoor)
        pooled /= (inside.size + outdoor.size - 2)
        if pooled <= 0:
            continue
        r_cohen = (resample_inside.mean() - resample_outdoor.mean()) / np.sqrt(pooled)
        bootstrap_values.append(r_cohen * correction)

    if not bootstrap_values:
        return hedges_g, None

    ci_lower, ci_upper = np.percentile(bootstrap_values, [2.5, 97.5])
    return hedges_g, (ci_lower, ci_upper)


def welch_ttest(
    sample_inside: pd.Series,
    sample_outdoor: pd.Series,
) -> Optional[Tuple[float, float]]:
    """Welch’s t-test on per-file mean distributions (inside vs OD)."""
    if stats is None:
        return None
    inside = sample_inside.dropna()
    outdoor = sample_outdoor.dropna()
    if len(inside) < 2 or len(outdoor) < 2:
        return None

    t_stat, p_value = stats.ttest_ind(inside, outdoor, equal_var=False, nan_policy="omit")
    return float(t_stat), float(p_value)


def _build_result_row(
    inside_row: pd.Series,
    outdoor_row: pd.Series,
) -> Dict[str, float]:
    """Compute delta metrics between inside and outdoor rows."""
    mean_inside = inside_row.get("mean", np.nan)
    mean_outdoor = outdoor_row.get("mean", np.nan)
    delta = mean_inside - mean_outdoor
    pct = np.nan
    if np.isfinite(mean_outdoor) and mean_outdoor != 0:
        pct = 100.0 * delta / mean_outdoor

    return {
        "mean_inside": mean_inside,
        "mean_outdoor": mean_outdoor,
        "mean_delta": delta,
        "mean_delta_pct": pct,
        "median_inside": inside_row.get("median", np.nan),
        "median_outdoor": outdoor_row.get("median", np.nan),
        "std_inside": inside_row.get("std", np.nan),
        "std_outdoor": outdoor_row.get("std", np.nan),
        "count_inside": inside_row.get("count", np.nan),
        "count_outdoor": outdoor_row.get("count", np.nan),
    }


def compare_overall(
    summary_overall: pd.DataFrame,
    per_file_stats: Optional[pd.DataFrame],
    bootstrap_iters: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Produce dict of overall mean/median/std delta and effect size + p-value."""
    inside = summary_overall.loc[summary_overall["env"] == "inside"]
    outdoor = summary_overall.loc[summary_overall["env"] == "outdoor_driving"]
    if inside.empty or outdoor.empty:
        raise ValueError("Summary overall must contain both inside and outdoor_driving rows.")

    row = _build_result_row(inside.iloc[0], outdoor.iloc[0])

    effect = None
    effect_ci = None
    t_result = None
    if per_file_stats is not None and "env" in per_file_stats.columns:
        inside_samples = per_file_stats.loc[per_file_stats["env"] == "inside", "mean"]
        outdoor_samples = per_file_stats.loc[per_file_stats["env"] == "outdoor_driving", "mean"]
        effect, effect_ci = compute_effect_size(inside_samples, outdoor_samples, bootstrap_iters, rng)
        t_result = welch_ttest(inside_samples, outdoor_samples)

    row["effect_size"] = effect
    row["effect_ci"] = effect_ci
    row["t_test"] = t_result
    return row


def compare_by_rat(
    summary_by_rat: pd.DataFrame,
    per_file_stats: Optional[pd.DataFrame],
    bootstrap_iters: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    """Produce metrics per RAT."""
    results: List[Dict[str, object]] = []
    rats = sorted(summary_by_rat["rat"].dropna().unique())
    for rat in rats:
        subset = summary_by_rat[summary_by_rat["rat"] == rat]
        inside = subset[subset["env"] == "inside"]
        outdoor = subset[subset["env"] == "outdoor_driving"]
        if inside.empty or outdoor.empty:
            continue
        result = {"rat": rat}
        result.update(_build_result_row(inside.iloc[0], outdoor.iloc[0]))

        effect = None
        effect_ci = None
        t_result = None
        if per_file_stats is not None and {"env", "rat"}.issubset(per_file_stats.columns):
            inside_samples = per_file_stats.loc[
                (per_file_stats["env"] == "inside") & (per_file_stats["rat"] == rat),
                "mean",
            ]
            outdoor_samples = per_file_stats.loc[
                (per_file_stats["env"] == "outdoor_driving") & (per_file_stats["rat"] == rat),
                "mean",
            ]
            if not inside_samples.empty and not outdoor_samples.empty:
                effect, effect_ci = compute_effect_size(
                    inside_samples, outdoor_samples, bootstrap_iters, rng
                )
                t_result = welch_ttest(inside_samples, outdoor_samples)

        result["effect_size"] = effect
        result["effect_ci"] = effect_ci
        result["t_test"] = t_result
        results.append(result)
    return results


def analyze_speed_buckets(
    summary_by_speed: pd.DataFrame,
    per_file_speed_stats: Optional[pd.DataFrame],
    bootstrap_iters: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Rank speed buckets by mean; compute diffs vs inside; check ordering."""
    if summary_by_speed.empty:
        return {"ranking": [], "violations": [], "deltas": {}}

    df = summary_by_speed.copy()
    ordering = ["inside", "od_quasi_static", "od_slow", "od_fast"]
    df["rank_key"] = df["speed_bucket"].apply(lambda x: ordering.index(x) if x in ordering else len(ordering))
    df.sort_values("rank_key", inplace=True)

    ranking = df["speed_bucket"].tolist()
    deltas = {}
    inside_mean = df.loc[df["speed_bucket"] == "inside", "mean"]
    inside_median = df.loc[df["speed_bucket"] == "inside", "median"]
    inside_mean_val = inside_mean.iloc[0] if not inside_mean.empty else np.nan
    inside_median_val = inside_median.iloc[0] if not inside_median.empty else np.nan

    for _, row in df.iterrows():
        bucket = row["speed_bucket"]
        deltas[bucket] = {
            "mean_delta": inside_mean_val - row["mean"] if np.isfinite(inside_mean_val) else np.nan,
            "median_delta": inside_median_val - row["median"] if np.isfinite(inside_median_val) else np.nan,
        }

    violations: List[str] = []
    for metric in ["mean", "median"]:
        values = [row[metric] for _, row in df.iterrows()]
        filtered_values = [v for v in values if isinstance(v, (int, float, np.floating))]
        if len(filtered_values) < len(values):
            continue
        is_non_increasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
        if not is_non_increasing:
            violations.append(f"{metric} ordering deviates from expected inside ≥ quasi-static ≥ slow ≥ fast")

    significance: Dict[str, Dict[str, Optional[object]]] = {}
    if per_file_speed_stats is not None and "speed_bucket" in per_file_speed_stats.columns:
        inside_samples = per_file_speed_stats.loc[
            per_file_speed_stats["speed_bucket"] == "inside", "mean"
        ].dropna()
        if not inside_samples.empty:
            for bucket in df["speed_bucket"]:
                if bucket == "inside":
                    continue
                bucket_samples = per_file_speed_stats.loc[
                    per_file_speed_stats["speed_bucket"] == bucket, "mean"
                ].dropna()
                if bucket_samples.empty:
                    significance[bucket] = {"effect_size": None, "effect_ci": None, "t_test": None}
                    continue
                effect, ci = compute_effect_size(
                    inside_samples, bucket_samples, bootstrap_iters, rng
                )
                t_result = welch_ttest(inside_samples, bucket_samples)
                significance[bucket] = {
                    "effect_size": effect,
                    "effect_ci": ci,
                    "t_test": t_result,
                }

    return {
        "ranking": ranking,
        "deltas": deltas,
        "violations": violations,
        "significance": significance,
    }


def _format_effect(effect: Optional[float], ci: Optional[Tuple[float, float]]) -> str:
    if effect is None:
        return "n/a"
    if ci is None:
        return f"{effect:.3f}"
    return f"{effect:.3f} (95% CI {ci[0]:.3f}–{ci[1]:.3f})"


def _format_ttest(t_result: Optional[Tuple[float, float]]) -> str:
    if t_result is None:
        return "n/a"
    t_stat, p_value = t_result
    return f"t={t_stat:.2f}, p={p_value:.4f}"


def _significance_label(t_result: Optional[Tuple[float, float]], alpha: float) -> str:
    if t_result is None:
        return "n/a"
    _, p_value = t_result
    if not np.isfinite(p_value):
        return "n/a"
    return "yes" if p_value < alpha else "no"


def _markdown_table(headers: Iterable[str], rows: List[Iterable[object]]) -> str:
    head_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = []
    for row in rows:
        formatted = []
        for value in row:
            if isinstance(value, float):
                if np.isnan(value):
                    formatted.append("n/a")
                else:
                    formatted.append(f"{value:.3f}")
            else:
                formatted.append(str(value))
        row_lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join([head_line, sep_line, *row_lines])


def write_markdown_summary(
    results_dir: Path,
    overall_metrics: Dict[str, object],
    rat_metrics: List[Dict[str, object]],
    speed_analysis: Dict[str, object],
    summary_overall: pd.DataFrame,
    summary_by_rat: pd.DataFrame,
    summary_by_speed: pd.DataFrame,
    available_plots: List[str],
    alpha: float,
) -> None:
    """Write results/summary.md with required sections and compact tables."""
    lines: List[str] = []

    # Title
    lines.append("# Indoor vs Outdoor Network Quality — Results Summary")

    # Key Findings
    lines.append("\n## Key Findings")
    key_points: List[str] = []
    overall_mean = overall_metrics.get("mean_inside")
    overall_out = overall_metrics.get("mean_outdoor")
    overall_delta = overall_metrics.get("mean_delta")
    overall_std_in = overall_metrics.get("std_inside")
    overall_std_out = overall_metrics.get("std_outdoor")
    key_points.append(
        f"- Overall mean NetworkQuality: inside {overall_mean:.3f} ± {overall_std_in:.3f}, "
        f"outdoor {overall_out:.3f} ± {overall_std_out:.3f} (Δ={overall_delta:.3f})."
    )
    for rat_entry in rat_metrics:
        key_points.append(
            f"- {rat_entry['rat']} mean difference: {rat_entry['mean_delta']:.3f} "
            f"(inside {rat_entry['mean_inside']:.3f} vs outdoor {rat_entry['mean_outdoor']:.3f})."
        )
    ranking = speed_analysis.get("ranking", [])
    if ranking:
        key_points.append(
            "- Speed buckets ranked (mean): " + " > ".join(ranking)
        )
    violations = speed_analysis.get("violations", [])
    if violations:
        for violation in violations:
            key_points.append(f"- Warning: {violation}.")
    effect_text = _format_effect(
        overall_metrics.get("effect_size"),
        overall_metrics.get("effect_ci"),
    )
    t_text = _format_ttest(overall_metrics.get("t_test"))
    key_points.append(f"- Effect size (Hedges’ g): {effect_text}; Welch t-test: {t_text}.")
    lines.extend(key_points)

    # Method (Brief)
    lines.append("\n## Method (Brief)")
    method_text = textwrap.dedent(
        """
        - Upstream pipeline: per-RAT robust normalization (5th–95th percentile) to map KPIs into 0–1 NetworkQuality, followed by per-file aggregation to avoid file-size bias.
        - This script: compares inside vs outdoor means, medians, and stds; computes Hedges’ g effect sizes with optional bootstrap CIs; runs Welch’s t-tests when per-file means are available; checks speed-bucket monotonicity.
        """
    ).strip()
    lines.append(method_text)

    # Tables
    lines.append("\n## Tables")
    overall_table = _markdown_table(
        ["env", "count", "mean", "median", "std"],
        summary_overall[["env", "count", "mean", "median", "std"]].values.tolist(),
    )
    lines.append("\n### Overall IS vs OD")
    lines.append(overall_table)

    if not summary_by_rat.empty:
        rat_table = _markdown_table(
            ["rat", "env", "count", "mean", "median", "std"],
            summary_by_rat[["rat", "env", "count", "mean", "median", "std"]].values.tolist(),
        )
        lines.append("\n### Per-RAT IS vs OD")
        lines.append(rat_table)

    if not summary_by_speed.empty:
        speed_table = _markdown_table(
            ["speed_bucket", "count", "mean", "median", "std"],
            summary_by_speed[["speed_bucket", "count", "mean", "median", "std"]].values.tolist(),
        )
        lines.append("\n### Speed Buckets")
        lines.append(speed_table)

    # Significance tests
    lines.append("\n## Significance Tests")
    significance_rows: List[Iterable[object]] = []
    significance_rows.append(
        [
            "Overall (inside vs outdoor)",
            _format_effect(overall_metrics.get("effect_size"), overall_metrics.get("effect_ci")),
            _format_ttest(overall_metrics.get("t_test")),
            _significance_label(overall_metrics.get("t_test"), alpha),
        ]
    )
    for entry in rat_metrics:
        significance_rows.append(
            [
                f"{entry['rat']} (inside vs outdoor)",
                _format_effect(entry.get("effect_size"), entry.get("effect_ci")),
                _format_ttest(entry.get("t_test")),
                _significance_label(entry.get("t_test"), alpha),
            ]
        )
    if significance_rows:
        significance_table = _markdown_table(
            ["Comparison", "Effect (Hedges’ g)", "Welch t-test", f"Significant @ α={alpha}"],
            significance_rows,
        )
        lines.append(significance_table)

    speed_significance = speed_analysis.get("significance") or {}
    if speed_significance:
        bucket_rows: List[Iterable[object]] = []
        for bucket, stats_dict in speed_significance.items():
            bucket_rows.append(
                [
                    f"inside vs {bucket}",
                    _format_effect(stats_dict.get("effect_size"), stats_dict.get("effect_ci")),
                    _format_ttest(stats_dict.get("t_test")),
                    _significance_label(stats_dict.get("t_test"), alpha),
                ]
            )
        if bucket_rows:
            lines.append("\n### Speed Bucket Contrasts")
            bucket_table = _markdown_table(
                ["Comparison", "Effect (Hedges’ g)", "Welch t-test", f"Significant @ α={alpha}"],
                bucket_rows,
            )
            lines.append(bucket_table)

    # Plots (Optional)
    lines.append("\n## Plots (Optional)")
    if available_plots:
        for plot in available_plots:
            lines.append(f"- {plot}")
    else:
        lines.append("- No plots detected.")

    # Caveats
    lines.append("\n## Caveats")
    caveats = textwrap.dedent(
        """
        - KPI coverage may vary; missing KPIs reduce confidence in normalized scores.
        - File counts per environment/RAT can be unbalanced, affecting variance estimates.
        - Outdoor mobility mixes quasi-static, slow, and fast segments; residual confounds may remain.
        - Results depend on upstream percentile clipping choices.
        """
    ).strip()
    lines.append(caveats)

    # Next Steps
    lines.append("\n## Next Steps")
    next_steps = textwrap.dedent(
        """
        - Stratify by band, operator, or scenario tags to isolate key drivers.
        - Test sensitivity to different percentile clips (`--pclip`) in the upstream script.
        - Include outside-walking (OW) data as a sanity check when available.
        """
    ).strip()
    lines.append(next_steps)

    output_path = results_dir / "summary.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze passive quality summaries and write Markdown report."
    )
    parser.add_argument(
        "--results_dir",
        default="./results_normalized",
        help="Directory with summary CSV outputs.",
    )
    parser.add_argument(
        "--alpha",
        default=0.05,
        type=float,
        help="Significance threshold.",
    )
    parser.add_argument(
        "--bootstrap_iters",
        default=2000,
        type=int,
        help="Bootstrap iterations for effect size CI when per-file stats exist.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rng = np.random.default_rng(42)

    try:
        data = load_summaries(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return

    overall_metrics = compare_overall(
        data.summary_overall,
        data.per_file_stats,
        args.bootstrap_iters,
        rng,
    )
    rat_metrics = compare_by_rat(
        data.summary_by_rat,
        data.per_file_stats,
        args.bootstrap_iters,
        rng,
    )
    speed_analysis = analyze_speed_buckets(
        data.summary_by_speed, data.per_file_speed_stats, args.bootstrap_iters, rng
    )

    available_plots = [
        str((results_dir / plot).as_posix())
        for plot in EXPECTED_PLOTS
        if (results_dir / plot).exists()
    ]

    write_markdown_summary(
        results_dir,
        overall_metrics,
        rat_metrics,
        speed_analysis,
        data.summary_overall,
        data.summary_by_rat,
        data.summary_by_speed,
        available_plots,
        args.alpha,
    )

    print("Report written to results/summary.md")

    effect = overall_metrics.get("effect_size")
    if effect is None:
        if data.per_file_stats is None:
            print("Effect sizes unavailable: per_file_stats.csv not found.")
        else:
            print("Effect sizes unavailable: insufficient per-file samples.")
    else:
        ci = overall_metrics.get("effect_ci")
        ci_text = f" (95% CI {ci[0]:.3f}–{ci[1]:.3f})" if ci else ""
        print(f"Overall Hedges’ g: {effect:.3f}{ci_text}")

    for entry in rat_metrics:
        effect = entry.get("effect_size")
        if effect is None:
            continue
        ci = entry.get("effect_ci")
        ci_text = f" (95% CI {ci[0]:.3f}–{ci[1]:.3f})" if ci else ""
        print(f"{entry['rat']} Hedges’ g: {effect:.3f}{ci_text}")

    t_test = overall_metrics.get("t_test")
    if t_test is not None:
        t_stat, p_value = t_test
        significance = "significant" if p_value < args.alpha else "not significant"
        print(f"Welch t-test: t={t_stat:.2f}, p={p_value:.4f} ({significance} at alpha={args.alpha})")
    else:
        if stats is None:
            print("Welch t-test skipped: install scipy to enable statistical tests.")
        elif data.per_file_stats is None:
            print("Welch t-test skipped: per_file_stats.csv missing.")
        else:
            print("Welch t-test skipped: insufficient per-file samples.")

    for entry in rat_metrics:
        t_result = entry.get("t_test")
        if t_result is None:
            continue
        t_stat, p_value = t_result
        significance = "significant" if p_value < args.alpha else "not significant"
        print(
            f"{entry['rat']} Welch t-test: t={t_stat:.2f}, p={p_value:.4f} "
            f"({significance} at alpha={args.alpha})"
        )

    violations = speed_analysis.get("violations") or []
    if violations:
        print("Speed bucket monotonicity warnings:")
        for note in violations:
            print(f"  - {note}")

    bucket_significance = speed_analysis.get("significance") or {}
    for bucket, stats_dict in bucket_significance.items():
        t_result = stats_dict.get("t_test")
        if t_result is None:
            continue
        t_stat, p_value = t_result
        significance = "significant" if p_value < args.alpha else "not significant"
        print(
            f"inside vs {bucket}: t={t_stat:.2f}, p={p_value:.4f} "
            f"({significance} at alpha={args.alpha})"
        )


if __name__ == "__main__":
    main()
