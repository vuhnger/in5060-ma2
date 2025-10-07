# Passive Dataset Processing Overview

This repository’s passive workflow starts with `analyze_passive_quality.py`. The script
discovers passive CSV logs under the `4G/` and `5G/` folders and produces network-quality
summaries in `results_normalized/` by default (raw KPI variants appear under
`results_non_normalized/`). The key processing stages are:

1. **File discovery**
   - Scan only the top-level `4G/` and `5G/` directories (no recursion outside them).
   - Ignore archives (`*.zip`), the `Active/` folder, and `__MACOSX/`.
   - Infer Radio Access Technology (RAT) from the parent directory name (`4G/` → 4G, `5G/` → 5G).
   - Determine environment from file names using case-insensitive tokens:
     - `is` → Indoor Static (`inside`)
     - `od` → Outdoor Driving (`outdoor_driving`)
     - `ow` → Outside Walking (`outside_walking`, only included when `--include_ow` is set).

2. **KPI selection**
   - Load each CSV and coerce KPI columns to numeric values.
   - Prioritise `DM_RS-SINR`, `DM_RS-RSRP`, `SSS-RSRP`, `SSS-RSRQ`; fall back to SINR/RSRP/RSRQ variants until up to five KPIs are selected.
   - Drop rows where all chosen KPI values are `NaN`.

3. **Per-RAT robust normalization**
   - For every KPI and RAT separately, compute the 5th and 95th percentiles (configurable via `--pclip`).
   - Map the 5th percentile to 0 and the 95th percentile to 1; clip outside this range.
   - Handle degenerate distributions by assigning 0.5 to valid samples.
   - Pass `--disable_normalization` to skip this step and work with raw KPI averages instead (results saved under `results_non_normalized/`).

4. **NetworkQuality score**
   - For each row, average the available normalized KPIs to obtain `NetworkQuality` (0–1 scale).
   - Rows with all KPIs missing remain `NaN`.

5. **Speed bucketing**
   - Convert the optional `Speed` column to m/s.
   - Assign Outdoor Driving rows to one of three buckets using configurable thresholds (`--speed_thresholds`, default `0.5,5.0` m/s):
     - `od_quasi_static`: ≤ 0.5 m/s
     - `od_slow`: 0.5–5 m/s
     - `od_fast`: > 5 m/s
   - Indoor rows receive the `inside` bucket; OW rows retain `outside_walking`.

6. **Aggregation**
   - Aggregate per file to avoid size bias:
     - (RAT, environment, file) → mean/median/std/count of `NetworkQuality`
     - (RAT, speed bucket, file) → mean/median/std/count
   - Aggregate across files for summary CSVs:
     - `summary_overall.csv`
     - `summary_by_rat.csv`
     - `summary_by_speed.csv`

7. **Auxiliary outputs**
   - `per_file_stats.csv`: per-file metrics by environment.
   - `per_file_stats_by_speed.csv`: per-file metrics by speed bucket.
   - `kpi_presence.csv`: diagnostic counts of KPI availability.
   - Optional plots (`boxplot_quality_by_env.png`, `bar_quality_mean_std_by_rat.png`) when `--save_plots` is enabled.

These outputs form the sole inputs for downstream analysis and plotting scripts.
