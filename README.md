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
- `--save_plots`: toggle saving matplotlib figures (enabled by default).
- `--disable_normalization`: skip per-RAT robust scaling and compute raw KPI averages instead (outputs land under `results_non_normalized/`).

### Outputs
- Console tables comparing IS vs OD overall, per RAT, and by speed bucket.
- CSV summaries written to `results_normalized/summary_overall.csv`, `results_normalized/summary_by_rat.csv`, and `results_normalized/summary_by_speed.csv` by default.
- Optional plots and diagnostics saved under `results_normalized/` (`per_file_stats.csv`, `per_file_stats_by_speed.csv`, `kpi_presence.csv`, and the bar/box/violin/ECDF/scatter figures).
- Running with `--disable_normalization` produces a parallel set of artefacts under `results_non_normalized/`.

### How the NetworkQuality metrics are derived
- **File scope**: only passive CSVs directly under `4G/` or `5G/` are loaded; archives and `Active/` content are ignored.
- **Environment tagging**: filenames containing `is`, `od`, or `ow` (case-insensitive) determine the scenario (`inside`, `outdoor_driving`, `outside_walking`).
- **KPI selection**: up to five RF KPIs are chosen per file, prioritising `DM_RS-*` and `SSS-*` metrics before falling back to SINR/RSRP/RSRQ variants. Rows with all selected KPIs missing are dropped.
- **Per-RAT robust scaling**: for each KPI within each RAT, the 5th percentile maps to 0 and the 95th percentile to 1 (configurable via `--pclip`). Degenerate distributions default to 0.5.
- **Row-level score**: `NetworkQuality` is the average of available normalized KPIs for that row, yielding a 0–1 range.
- **Speed buckets**: Outdoor Driving samples are bucketed using `--speed_thresholds` (default ≤0.5 m/s quasi-static, 0.5–5 m/s slow, >5 m/s fast); indoor samples map to `inside`.
- **Per-file aggregation**: statistics are first computed per file to avoid dataset-size bias, then averaged across files for the summary CSVs.

## Post-processing Scripts

Once `analyze_passive_quality.py` has produced the summary CSVs under `results_normalized/` (or `results_non_normalized/`), two additional scripts help document and visualise findings without re-reading the raw datasets.

### Statistical report
```bash
python3 analyze_results_and_report.py --results_dir ./results_normalized
```
- Computes overall and per-RAT deltas, Hedges’ g effect sizes (with bootstrap CIs if per-file stats exist), and Welch t-tests.
- Ranks OD speed buckets and flags monotonicity issues.
- Writes a concise Markdown report to `<results_dir>/summary.md` including compact tables and references to any generated plots.

Optional flags:
- `--alpha FLOAT` (default `0.05`): significance threshold for reporting p-values.
- `--bootstrap_iters INT` (default `2000`): resamples for effect-size confidence intervals.

> Effect sizes and Welch tests require both `results_normalized/per_file_stats.csv` (or the non-normalized counterpart) and SciPy (`python3 -m pip install scipy`).

### Plotting utility
```bash
python3 plot_passive_quality_results.py --results_dir ./results_normalized
```
- Generates PNG figures from the summary CSVs (`plot_*` files under the selected `results_dir`).
- Produces bar charts, boxplots, violin plots, ECDF curves, scatter diagnostics, and KPI availability summaries; every axis is annotated with the metric and units (e.g., `NetworkQuality (0-1)`).
- Bar and violin charts include annotated mean deltas, percentage differences, and Welch p-values whenever per-file statistics and SciPy are available (otherwise they report `p=n/a`).
- Speed-bucket plots embed the threshold definitions (≤0.5 m/s, 0.5–5 m/s, >5 m/s) directly in their titles for quick reference.
- Axes automatically adapt to normalized (0–1) versus raw KPI outputs, with range information reflected in the labels.
- Uses matplotlib for rendering; SciPy is optional but enables p-value annotations. Plots are skipped gracefully if required inputs are missing.
- Re-run with `--results_dir ./results_non_normalized` to visualise the raw (non-normalized) pipeline output.

Optional flags:
- `--save_all`: default behaviour is to save every available plot; flag retained for parity.
- `--skip_ecdf`: omit ECDF plots when per-file statistics are large or absent.

### Full pipeline runner
```bash
python3 run_full_analysis.py --four-g-dir ./4G --five-g-dir ./5G
```
- Executes the full pipeline (analysis → report → plots) with one command.
- Pass through key options: `--include-ow`, `--pclip`, `--speed-thresholds`, `--alpha`, `--bootstrap-iters`, and `--skip-ecdf`.
- Use `--disable-normalization` to run only the raw KPI pipeline, or `--run-both` to emit both normalized (`results_normalized/`) and raw (`results_non_normalized/`) artefacts sequentially.

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
