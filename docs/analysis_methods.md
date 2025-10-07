# Passive Results Analysis Overview

The downstream analysis is handled by two scripts that consume only the CSV artefacts in `results/`:

- `analyze_results_and_report.py`: computes statistics, significance tests, and writes a Markdown summary.
- `plot_passive_quality_results.py`: generates matplotlib figures for dashboards and diagnostics.

## Statistical analysis workflow

1. **Data ingestion**
   - Required inputs: `summary_overall.csv`, `summary_by_rat.csv`, `summary_by_speed.csv`.
   - Optional inputs:
     - `per_file_stats.csv`: per-file mean/median/std by environment (enables inference on independent samples).
     - `per_file_stats_by_speed.csv`: per-file metrics by speed bucket.
     - `kpi_presence.csv`: KPI coverage diagnostics.

2. **Comparisons**
   - Overall: contrast `inside` vs `outdoor_driving` means, medians, and standard deviations; compute absolute and percentage deltas.
   - Per RAT: repeat the comparison for each RAT present (e.g., 4G, 5G).
   - Speed buckets: rank `inside`, `od_quasi_static`, `od_slow`, `od_fast`; highlight violations of the expected inside ≥ quasi-static ≥ slow ≥ fast ordering.

3. **Effect sizes and significance tests**
   - **Effect size**: Hedges’ g (bias-corrected Cohen’s d) computed on per-file mean distributions.
   - **Bootstrap confidence interval**: optional resampling (default 2000 iterations) around the effect size.
   - **Welch’s t-test**: independent two-sample t-test with unequal variances (requires SciPy and per-file stats).
   - **Speed-bucket tests**: compare `inside` vs each OD bucket using per-file speed statistics when available.
   - Missing prerequisites (SciPy or per-file stats) are reported clearly; analyses continue with descriptive statistics.

4. **Markdown summary**
   - Written to `results/summary.md`.
   - Sections: Title, Key Findings, Method (Brief), Tables (overall, by RAT, by speed), Significance Tests, Plots (optional), Caveats, Next Steps.
   - Significance tables list t-statistics, p-values, and alpha comparisons for overall, per-RAT, and speed-bucket contrasts.

## Plotting workflow

`plot_passive_quality_results.py` reuses the same CSV inputs to produce PNG figures under `results/`:

- Bar charts of overall means, RAT-by-environment means, and speed buckets.
- Optional boxplots, ECDFs, and mean-vs-std scatterplots using per-file stats.
- KPI availability bar chart if `kpi_presence.csv` exists.
- Each figure is rendered with matplotlib, one per file, using `tight_layout()` and default colour schemes.

## Reproducible execution

The `run_full_analysis.py` script orchestrates the entire pipeline:

1. Run `analyze_passive_quality.py` to regenerate CSV summaries and diagnostics.
2. Run `analyze_results_and_report.py` to compute statistics and write the Markdown report.
3. Run `plot_passive_quality_results.py` to refresh plots.

Command-line flags allow overriding data directories, percentile clips, bootstrap iterations, and ECDF plotting. Refer to the README for usage examples.
