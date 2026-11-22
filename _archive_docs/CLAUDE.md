# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Challenge Data "Enedis" competition repo: reconstruct missing values in 1,000
synthetic electricity-consumption curves (`holed_*` columns), using tens of
thousands of near-complete donor curves for context. This is not a software
product — it's a research/experimentation repo of benchmark scripts that
iterate toward a single best-scoring submission CSV. There is no app, server,
build, lint, or test suite; everything runs as standalone Python scripts.

Not a git repository (no `.git`) — do not assume version control is available.

## Environment & running scripts

Use the project venv explicitly (no ambient activation assumed):

```bash
.venv/bin/python <script>.py
```

`requirements.txt` only lists `pandas`/`numpy`/`scikit-learn`/`tabpfn-client`;
the venv actually also has `lightgbm`, `catboost`, `xgboost` installed (used
by the `*_bench.py` regressor comparisons) — check `.venv/bin/pip list`
rather than trusting `requirements.txt` alone.

TabPFN access is a cloud API call (`tabpfn_client`), not a local model — set
`TABPFN_API_KEY` in the environment before running any `tabpfn_*` or
`generate_submission_tabpfn_*` script. Some older scripts fall back to a
hardcoded token default when the env var is unset; prefer always exporting
`TABPFN_API_KEY` and treat any embedded token in old scripts as compromised/
to be removed, not reused.

Long-running benchmark/submission scripts are checkpointed and resumable —
safe to kill and re-run:
- `*_bench.py` scripts write `*_progress.json` (per-config results as they
  finish) and a final `*_results.json`.
- `generate_submission_*.py` scripts write `*_partial.csv` after every batch
  of columns and skip columns already present on restart.

Some scripts are driven by a `run_*.sh` wrapper (e.g. `run_round6.sh`,
`run_gemini.sh`) that retries a script per-config up to 3x and skips configs
already recorded in the results JSON — check for one before re-running a
`*_bench_roundN.py` file directly.

## Core architecture

Everything downstream builds on two files:

- **`evaluate.py`** — the harness. `load_data()` loads the three challenge
  CSVs from `data/`; `official_mae()` implements the competition metric
  (MAE computed *only* at positions that are truly missing in `X_train`/
  `X_test`, excluding the rare positions where `y_train` itself is NaN);
  `repeated_holdout_eval()` runs an N-repeat `ShuffleSplit` over the 1,000
  holed columns (columns, not rows, are split into dev/val) so models can be
  cross-validated without synthetic masking, since the real gap mask and
  ground truth are already known for every `holed_*` column in train.
- **`feature_engineering_bench.py`** — the shared feature/model library,
  imported by nearly every other script (`from evaluate import ...` +
  `import feature_engineering_bench as feb`). Contains the reusable building
  blocks: neighbor selection (`adaptive_top_k_corr_euclid`,
  `weighted_knn_pred`), shape clustering (`fit_shape_clusters`, `cluster_of`),
  robust neighbor scaling, calendar/seasonality features
  (`time_features_full_seasonality`, `same_curve_seasonality_feature`),
  periodic-copy features at `t±48/96/144/336` with availability flags
  (`periodic_copy_features`, `periodic_agreement`), gap-shape features
  (`run_lengths_for`, `edge_distance_for`), and the adaptive
  interpolation/model blend weight (`gap_alpha`, `continuous_alpha`,
  `volatility_alpha_mult`). It also contains numbered `make_vN_model()`
  factories representing the historical progression (V1 → V11) of the local
  (non-TabPFN) pipeline.

Every other script is one experiment built on top of these two files:
- `*_bench.py` / `*_screen.py` / `*_ablationN.py` — compare a specific idea
  or hyperparameter against a fixed baseline, using the *same* random seeds
  as prior runs so results are paired/comparable (see "Evaluation protocols"
  below). Model-specific families: `xgboost_bench*.py`, `catboost_*bench*.py`,
  `advanced_regressors_bench.py` (LightGBM/CatBoost/ExtraTrees + ensembles),
  `tabpfn_*.py` (TabPFN via cloud API, the best-performing family).
- `generate_submission_*.py` — takes one validated pipeline and runs it over
  the real `X_test` holed columns against the full donor pool
  (`X_test` donors + `X_train` donors + `y_train`) to produce the actual
  submission CSV, in parallel across columns (`ThreadPoolExecutor`) with a
  deterministic KNN fallback if the primary model (e.g. TabPFN) errors on a
  column, so a run never yields an invalid/missing column.
- `eda_plots.py` — generates the figures under `figures/`.

## Evaluation protocols (must match to compare scores)

Two incompatible protocols coexist — never compare their means directly, and
when adding a new comparison, match the *exact* protocol (repeats, test_size,
random_state, and ideally the same scored columns) of whatever you're
comparing against, otherwise the delta is meaningless noise:

1. **Robust `5×100`**: `n_repeats=5, test_size=0.10, random_state=42` via
   `evaluate.repeated_holdout_eval` (100 val columns per repeat). Used for
   fast local regressors (LightGBM/CatBoost/XGBoost/etc.) where cost isn't a
   concern.
2. **Paired `10×10`**: 10 repeats × 10 columns, used for TabPFN comparisons
   because each column costs a cloud API call. Recent TabPFN scripts pin the
   same seeds/columns across configs specifically so deltas are paired
   (compared per-repeat, not just on the mean) — check `benchmark.md` for
   examples of this paired-delta table format before adding a new TabPFN
   comparison.

`README.md` documents the current best validated model and how to reproduce
its submission; `benchmark.md` is the full experiment log (what was tried,
what won/lost, and why) — read it before proposing an idea, since many
plausible-sounding features (extra lags, higher `k`, calendar harmonics,
PCHIP interpolation, classic stacking, etc.) have already been tried and
rejected there with numbers. When you finish a new experiment, append to
`benchmark.md` in the same style (paired repeat-by-repeat table + a
one-line verdict) rather than only writing a fresh `*_results.json`.

## Data

`data/` holds the challenge-provided files (30-minute-interval curves,
9–31 Jan 2023): `X_train_78VdSWL.csv` (20,000 donor columns + 1,000
`holed_*` columns with real gaps), `y_train_u0UkKEh.csv` (ground truth for
the holed columns), `X_test_XKVc4no.csv` (donors + `holed_*` columns to
predict for the real submission), and `new_output_sample.csv` (the expected
submission shape/format to validate against — see the asserts at the end of
any `generate_submission_*.py`).
