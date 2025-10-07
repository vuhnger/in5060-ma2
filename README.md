# in5060-ma2
Mandatory assignment 2 for IN5060 H25.

## Participants
- victou
- jonasbny
- kribb

## Passive Quality Analysis Script

The repository contains `analyze_passive_quality.py`, which compares Indoor Static (IS) and Outdoor Driving (OD) network quality from the passive measurement CSVs in `4G/` and `5G/`.

### Prerequisites
- Python 3.9+
- Installed Python packages: `pandas`, `numpy`, `matplotlib` (optional for plots), and `scipy` (optional for Welch t-tests).

Install requirements with:
```bash
python3 -m pip install pandas numpy matplotlib scipy
```

### Running the script
Run from the repository root so the default folder layout is detected:
```bash
python3 analyze_passive_quality.py
```

Key options:
- `--four_g_dir PATH` / `--five_g_dir PATH`: override locations of the passive 4G/5G CSV folders.
- `--include_ow`: include outside-walking (`OW`) files if present.
- `--pclip LOW,HIGH`: adjust the robust normalization percentiles (default `0.05,0.95`).
- `--speed_thresholds LOW,HIGH`: tweak OD speed buckets in m/s (default `0.5,5`).
- `--save_plots`: toggle saving matplotlib figures into `results/` (enabled by default).

### Outputs
- Console tables comparing IS vs OD overall, per RAT, and by speed bucket.
- CSV summaries written to `results/summary_overall.csv`, `results/summary_by_rat.csv`, and `results/summary_by_speed.csv`.
- Optional plots saved under `results/` when matplotlib is available (`boxplot_quality_by_env.png`, `bar_quality_mean_std_by_rat.png`).
- Additional diagnostics: per-file statistics (`results/per_file_stats.csv`, `results/per_file_stats_by_speed.csv`) and KPI coverage (`results/kpi_presence.csv`).

## Post-processing Scripts

Once `analyze_passive_quality.py` has produced the summary CSVs under `results/`, two additional scripts help document and visualise findings without re-reading the raw datasets.

### Statistical report
```bash
python3 analyze_results_and_report.py --results_dir ./results
```
- Computes overall and per-RAT deltas, Hedges’ g effect sizes (with bootstrap CIs if per-file stats exist), and Welch t-tests.
- Ranks OD speed buckets and flags monotonicity issues.
- Writes a concise Markdown report to `results/summary.md` including compact tables and references to any generated plots.

Optional flags:
- `--alpha FLOAT` (default `0.05`): significance threshold for reporting p-values.
- `--bootstrap_iters INT` (default `2000`): resamples for effect-size confidence intervals.

> Effect sizes and Welch tests require both `results/per_file_stats.csv` and SciPy (`python3 -m pip install scipy`).

### Plotting utility
```bash
python3 plot_passive_quality_results.py --results_dir ./results
```
- Generates PNG figures from the summary CSVs (`plot_*` files under `results/`).
- Requires only matplotlib; gracefully skips plots whose inputs are missing.

Optional flags:
- `--save_all`: default behaviour is to save every available plot; flag retained for parity.
- `--skip_ecdf`: omit ECDF plots when per-file statistics are large or absent.

### Full pipeline runner
```bash
python3 run_full_analysis.py --four-g-dir ./4G --five-g-dir ./5G
```
- Executes the full pipeline (analysis → report → plots) with one command.
- Pass through key options: `--include-ow`, `--pclip`, `--speed-thresholds`, `--alpha`, `--bootstrap-iters`, and `--skip-ecdf`.

## Additional documentation
- `docs/data_processing_pipeline.md`: detailed description of how passive CSVs are discovered, normalized, and bucketed.
- `docs/analysis_methods.md`: explains downstream statistical methods, significance testing, and plotting outputs.

### Terminology
- **RAT**: Radio Access Technology (e.g., 4G, 5G).
- **IS**: Indoor Static measurements.
- **OD**: Outdoor Driving measurements.
- **OW**: Outside Walking measurements (included only with `--include_ow`).
- **KPI**: Key Performance Indicator (RF metrics like SINR, RSRP, RSRQ).
- **ECDF**: Empirical cumulative distribution function; shows the proportion of per-file samples below each NetworkQuality value.
- **Hedges’ g**: Effect-size metric (bias-corrected Cohen’s d) used to quantify the standardized mean difference between environments.
- **Welch’s t-test**: Two-sample statistical test that compares group means without assuming equal variances.
