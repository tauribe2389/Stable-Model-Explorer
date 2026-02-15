# Stable Model Explorer (SME)

Local-first Flask web app for ANOVA/ANCOVA-style engineering workflows:

1. CSV ingestion + profiling
2. Data health checks + group outlier review (MAD robust z-score)
3. Candidate model registry generation
4. Model fit + diagnostics + validity filtering
5. Stability analysis (Stable / Conditional / Non-effect / Red-flag)
6. HTML + PDF report export

All data and derived artifacts are stored locally (SQLite + local files), with persisted configuration for reproducibility and auditability.

## Tech stack

- Python 3.x
- Flask
- SQLite
- pandas, numpy, scipy, statsmodels
- matplotlib
- reportlab

## Quick start

```bash
py -3.9 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Workflow pages

1. **Home / Datasets**
- Upload CSV.
- Tune profiling rules (`categorical_max_cardinality`, datetime cardinality threshold, text uniqueness exclusion threshold).

2. **Dataset Detail**
- View inferred schema and column roles (continuous/categorical/excluded).
- Re-profile with updated rules.

3. **Analysis Config Wizard**
- Choose response, primary factors, group variables, covariates, forced terms, interactions.
- Group variables act as population split controls: models are generated and run per selected observed group combination.
- Advanced group options:
  - include unobserved combinations for sparsity diagnostics (auto-skipped as non-viable),
  - missing group policy (`AS_LEVEL` as `__NA__` or `DROP_ROWS`),
  - minimum primary-factor level size per group,
  - auto-limit to top N groups by raw row count (with coverage reporting).
- Define model universe caps and assumption/diagnostic thresholds.
- Save configuration presets.
- On submit, the app computes:
  - univariate screens (categorical ANOVA effect size, continuous correlation/regression),
  - confounding map,
  - covariate collinearity clusters,
  - health checks.

4. **Outlier Review**
- Group-wise robust z-score using MAD (`|z| > 3.5` by default).
- Explicit keep/remove decision per flagged row (`row_index` persisted).

5. **Model Registry**
- Generated candidate formula universe with forced terms + all-subsets under caps, materialized per selected `group_key`.
- If combinations exceed `max_total_models`, deterministic random sampling with saved seed.

6. **Model Runs Summary**
- For each candidate model:
  - OLS fit (ANOVA/ANCOVA)
  - VIF check
  - Shapiro (warning)
  - Breusch-Pagan
  - Cook’s distance (stored per row per model; warning/fail configurable)
  - residual/QQ plots
- Validity classes:
  - `valid`
  - `valid_with_robust_se` (if BP fails and robust SE enabled)
  - `invalid` (with stored reasons)

7. **Stability Dashboard**
- Effect objects:
  - `PAIRWISE_DIFF` (primary categorical level differences via adjusted means)
  - `RANKING` (level ranking consistency)
  - `SLOPE` (continuous covariate effects)
  - `INTERACTION_FLAG`
- Buckets:
  - `STABLE`
  - `CONDITIONAL`
  - `NON_EFFECT`
  - `REDFLAG`
- Per-group buckets plus combined weighted summaries.
- Combined weighting uses normalized cross-weighting from:
  - row share by group (`rows_after_outlier / total_rows`),
  - model quality share by group (`valid_models_in_group / total_models_in_group`).

8. **Report Export**
- HTML + PDF snapshot with config, checks, registry/runs, and stability summaries.

## Auditability and reproducibility

The app persists:

- dataset schema + profile rules
- analysis config JSON
- screening + health check outputs
- outlier policy and row decisions
- model registry definition
- per-model diagnostics, pass/fail reasons, and warnings
- Cook’s distance details per model row
- stability thresholds and bucket outcomes
- report snapshots

SQLite DB: `instance/sme.db`

Artifacts:
- plots: `artifacts/plots/...`
- reports: `artifacts/reports/...`

## Default thresholds (engineering-sane starts)

- VIF fail: `10` (warn: `5`)
- Breusch-Pagan alpha: `0.05`
- Shapiro fail alpha: `0.01` (warn alpha: `0.05`)
- MAD outlier threshold: `3.5`
- Group missing policy: `AS_LEVEL`
- Group max analyzed (top N): `25`
- Group min primary-factor level count per group: `5`
- Stable sign consistency: `0.9`
- Stable practical significance rate: `0.8`
- Non-effect rate threshold: `0.8`
- Engineering delta: `0.1`
- Equivalence bound: `0.05`

## Notes

- App is local-first by design: no cloud dependency.
- Text columns with high uniqueness are marked not fit for modeling and excluded.
- Use Power Query (or equivalent) for upstream cleaning/feature engineering when needed.
# Stable-Model-explorer
