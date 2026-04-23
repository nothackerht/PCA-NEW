# PCA Suite — Module Reference

## Overview

The suite is split into 8 Python modules plus the two inherited data-layer files (`data_loader.py`, `preprocessing.py`). Running `python ultimate_pca_suite.py` is the only entry point needed.

```
ultimate_pca_suite.py          ← orchestrator; calls everything below
├── pca_config.py              ← all constants / paths
├── pca_utils.py               ← stateless helpers
├── pca_preprocessing.py       ← preprocessing wrapper
├── pca_metrics.py             ← all scientific metrics
├── pca_plotting.py            ← all figures
├── pca_runner_core.py         ← PCA fitting + sweep engine
└── pca_outputs.py             ← leaderboards, export, audit
    ├── data_loader.py         ← reads raw .txt spectral files
    └── preprocessing.py       ← SNV / EMSC / etc. implementations
```

---

## Data flow from disk to results

```
.txt spectral files (Cohort 1 and Cohort 2)
        │
        ▼
  data_loader.load_data()
        │  returns:
        │    wavenumbers         (1731,)         float64
        │    averaged_spectra    (1731, N)        one column per patient
        │    all_spectra         (1731, 9N)       9 spectra per patient
        │    aligned_meta        DataFrame        one row per patient
        │
        ▼
  ultimate_pca_suite.main()
        │  splits into train (Cohort 1) and test (Cohort 2)
        │  detects SI column and DX column from metadata
        │  builds dm1_mask_train, ctrl_mask_train (boolean arrays, length N)
        │
        ├──▶ run_question1_raw_geometry()     no preprocessing
        │
        └──▶ run_master_sweep()
                │
                │  for each spectral window × concept × preprocessing combo:
                │
                ├── apply_pipeline_train_test()   fit on train, transform both
                │
                ├── build_fit_matrix()            selects which rows define PCA axes
                │
                ├── fit_pca_and_project()         sklearn PCA → scores (N×2)
                │
                ├── _collect_metrics_for_mode()   all scientific metrics
                │
                └── plot_*()                      saves PNGs to question dirs
```

---

## Module-by-module reference

---

### `pca_config.py`

**Purpose:** Single source of truth for every constant and path. No logic, no functions.

**What to edit before running:**
```
OUT_ROOT          where all outputs are written
BOX12_DATA_DIR    folder of Cohort 1 .txt spectral files
BOX12_META_CSV    metadata CSV for Cohort 1
BOX3_DATA_DIR     folder of Cohort 2 .txt spectral files
BOX3_META_CSV     metadata CSV for Cohort 2
```

**Key constants used by every other module:**

| Constant | Value | Used for |
|---|---|---|
| `CONCEPTS` | `["WEIGHTS__BOX12_DM1_ONLY", "WEIGHTS__BOX12_ALL"]` | Which rows define PCA axes |
| `SPECTRUM_MODES` | full / 500–3100 / 700–2800 | Spectral window variants |
| `SUPPORTED_METHODS` | Normalization, SNV, Second Derivative, EMSC | Preprocessing combos |
| `TOP_K_WAVENUMBERS` | 25 | How many loading peaks to highlight |
| `CHI2_95` | ≈ 5.991 | 95% confidence ellipse threshold (chi² df=2) |
| `FIGURE_DPI / JOURNAL_DPI` | 200 / 300 | Save DPI for per-run vs journal figures |

**Data source:** None. Constants only.

---

### `pca_utils.py`

**Purpose:** Stateless helper functions that every other module imports. No scientific computation here.

**Functions:**

| Function | What it does |
|---|---|
| `sanitize_filename(s)` | Strips illegal path characters for safe filenames |
| `safe_name(methods, do_baseline)` | Converts a method list to a run-name string, e.g. `"SNV+SECOND_DERIVATIVE"` |
| `detect_si_column(meta)` | Finds the Splicing Index column in the metadata DataFrame |
| `detect_dx_column(meta)` | Finds the diagnosis/type column |
| `get_dm1_control_masks(meta, dx_col, require_controls)` | Returns two boolean arrays: which rows are DM1, which are controls |
| `build_all_ordered_combos(methods)` | Generates all 9 preprocessing combinations (RAW + singles + fixed multi-step combos) |
| `_run_name_to_methods(run_name)` | Reverse-parses `"SNV+SECOND_DERIVATIVE"` back to `["SNV", "Second Derivative"]` |
| `trim_region(wavenumbers, X, lo, hi)` | Slices both the wavenumber array and the spectral matrix to a cm⁻¹ window |
| `compute_edge_fraction(wavenumbers, top_wn)` | Fraction of top loading wavenumbers in the outer 5% of spectral range (artefact flag) |

**Data source:** Metadata DataFrames passed in by the caller. No disk I/O.

---

### `pca_preprocessing.py`

**Purpose:** A single wrapper function that enforces the train/test split during preprocessing, preventing data leakage.

**Function:**
```python
apply_pipeline_train_test(X_train, X_test, methods, do_baseline=False)
```

**How it works:**
1. Creates a fresh `Preprocessing()` instance (from `preprocessing.py`)
2. Calls `.fit(X_train, methods)` — the EMSC reference spectrum is computed from training data only
3. Calls `.transform()` on both `X_train` and `X_test` using the fitted state
4. Returns `(X_train_preprocessed, X_test_preprocessed)`

**Why train-only fitting matters:** If EMSC were fitted on the combined train+test pool, the reference spectrum would contain information from Cohort 2, which is the held-out validation cohort. All other methods (SNV, Normalization, Second Derivative) are per-spectrum and stateless, so they are unaffected — but the same code path is used for consistency.

**Data source:** `X_train` and `X_test` are `(features × N)` float arrays passed in by `pca_runner_core`. No disk I/O.

---

### `pca_metrics.py`

**Purpose:** All scientific metric computations. Pure functions — no plotting, no file I/O, no PCA fitting.

#### Geometry helpers (Section 3)

These are low-level building blocks used by the metric bundles:

| Function | Returns |
|---|---|
| `_safe_distribution(points)` | `(mean, cov, cov_inv)` or `None` if fewer than 3 points or singular covariance |
| `_mahal_sq_array(query, ref_mean, ref_cov_inv)` | Per-point squared Mahalanobis distances via einsum |
| `mahalanobis_centroid_dist(query, ref)` | Mahalanobis distance from the query centroid to the reference distribution |
| `individual_mahalanobis_stats(query, ref)` | `(mean, median, max)` of per-point Mahalanobis distances |
| `fraction_inside_ellipse(query, ref)` | Fraction of query points inside the ref 95% confidence ellipse |
| `fraction_inside_convex_hull(query, ref)` | Fraction of query points inside the convex hull of ref (non-parametric) |
| `mean_nn_dist(query, ref)` | Mean nearest-neighbour distance from each query point to ref |
| `bhattacharyya_distance(a, b)` | Bhattacharyya distance between two 2-D Gaussian fits |
| `fisher_separation(a, b)` | Centroid-distance² / pooled within-class variance |

#### Metric bundles (Section 4)

These assemble the geometry helpers into the per-question metric dictionaries:

**`compute_alignment_metrics(scores_box3_dm1, scores_box12_dm1)`**
- Used for Q2 and Q4
- Input: 2-D PCA score arrays for Cohort 2 DM1 and Cohort 1 DM1
- Returns dict with 9 keys: `centroid_dist_*`, `mahalanobis_centroid_*`, `mahalanobis_mean/median/max_*`, `frac_box3_inside_dm1_95ellipse`, `frac_box3_inside_dm1_hull`, `mean_nn_dist_*`, `frac_box12dm1_inside_own_95ellipse`

**`compute_separation_metrics(scores_box12_dm1, scores_box3_dm1, scores_box12_ctrl, include_box3_in_dm1)`**
- Used for Q3A (`include_box3_in_dm1=False`) and Q3B (`include_box3_in_dm1=True`)
- Bhattacharyya and Fisher always use Cohort 1 training data only
- When `include_box3_in_dm1=True`, silhouette and centroid distance use the combined DM1 cloud (Cohort 1 + Cohort 2)
- Returns dict with 5 keys: `bhattacharyya_*`, `fisher_separation_*`, `centroid_dist_*`, `silhouette_*`, `frac_controls_outside_*`
- In the master table, results are stored with `hol_` prefix (holistic) or `cln_` prefix (clean)

**`compute_gradient_metrics(scores_dm1, si_dm1)`**
- Used for Q5
- Input: 2-D scores for Cohort 1 DM1 only + their SI values (Cohort 2 excluded)
- Returns: `rho_pc1`, `rho_pc2`, `grad_score` (= max(|ρ₁|, |ρ₂|)), `cv_r2_pc12_to_si`

**`add_composite_score(df)`**
- Applied post-sweep to the full results DataFrame
- Rank-normalises 6 metrics to (0,1] then computes weighted sum: 50% alignment + 50% separation
- Weights: frac_box3_inside_ellipse 0.20, mahal_mean 0.15, mean_nn_dist 0.15, bhattacharyya 0.20, silhouette 0.15, frac_ctrl_outside 0.15

**Data source:** 2-D numpy arrays of PCA scores, sliced by group masks. All scores are computed by `pca_runner_core` before being passed here.

---

### `pca_plotting.py`

**Purpose:** All figure generation. Every function that saves a file returns its `Path`.

#### Low-level helpers

| Function | Purpose |
|---|---|
| `break_by_gap(x, gap_thresh)` | Splits a wavenumber array into continuous segments so line plots don't connect across gaps (e.g. when a spectral window is trimmed) |
| `_draw_confidence_ellipse(ax, points_2d, confidence)` | Draws a 95% confidence ellipse on a matplotlib axes from eigenvalue decomposition of the point cloud's covariance |
| `_common_si_range(si_12, si_3, dm1_12, dm1_3)` | Computes a shared viridis colormap range across both DM1 cohorts so SI colours are directly comparable |

#### Per-run scatter plots

All three scatter functions take the same core inputs: PCA score arrays for Cohort 1 and Cohort 2, metadata DataFrames, and the `(pc1_var, pc2_var)` explained variance tuple.

| Function | Saved as | Description |
|---|---|---|
| `plot_group_scatter(...)` | `*__group.png` | Fixed colours: orange = Controls, steelblue = Cohort 1 DM1, crimson = Cohort 2 DM1 |
| `plot_si_scatter(...)` | `*__si_gradient.png` | DM1 points coloured by SI severity on a shared viridis scale |
| `plot_enhanced_pca_scatter(...)` | `*__<suffix>.png` | SI gradient + 95% confidence ellipses + centroids + metrics text box |

#### Loadings and overlay

| Function | Saved as | Description |
|---|---|---|
| `plot_loadings(...)` | `*__loadings.png` | PC1 and PC2 loading vectors vs wavenumber |
| `plot_overlay(...)` | `*__overlay.png` | All spectra (low alpha) with blue/red vertical lines at top PC1/PC2 wavenumbers |
| `save_loading_and_overlay(...)` | both + `*__top_wavenumbers.csv` | Convenience wrapper; returns `(wn_pc1, wn_pc2, loadings_path, overlay_path)` |
| `compute_top_wavenumbers(wavenumbers, pc_vec, top_k)` | — | Returns the `top_k` wavenumber values with largest absolute loading |

#### Journal figure

`make_journal_figure(...)` — two-panel figure: enhanced scatter (left) + spectra overlay (right). Saved at `JOURNAL_DPI`. Used only for question winners.

**Data source:** PCA score arrays and metadata from `pca_runner_core`. Spectral matrices `X_train / X_test` for overlay backgrounds. No disk reads.

---

### `pca_runner_core.py`

**Purpose:** The scientific engine. Fits PCA, runs the full sweep, and re-runs winner configurations.

#### PCA fitting (Section 6)

**`build_fit_matrix(concept, X_train, dm1_spec_mask)`**
- `WEIGHTS__BOX12_DM1_ONLY` → returns `X_train[:, dm1_mask]` (DM1 columns only)
- `WEIGHTS__BOX12_ALL` → returns the full `X_train` matrix
- This is the only place where concept affects the analysis

**`fit_pca_and_project(X_fit_matrix, X_train, X_test)`**
- Fits sklearn `PCA(n_components=2)` on `X_fit_matrix.T`
- Projects both `X_train.T` and `X_test.T` into that space
- Returns `(pca_model, scores_train, scores_test, (pc1_var, pc2_var))`

#### Metric collector (Section 7)

**`_collect_metrics_for_mode(pca, scores_12, scores_3, meta_train, meta_test, ...)`**
- Called once per (concept, preprocessing, plot_mode) combination
- Extracts group-specific score sub-arrays using masks
- Calls all four metric bundles from `pca_metrics`
- Returns a flat dict of ~30 metric values that becomes one row in the master table
- Works identically for `PATIENT_AVG` (N rows) and `ALL_SPECTRA` (9N rows) — the caller supplies the right masks

#### Q1 runner (Section 8)

**`run_question1_raw_geometry(...)`**
- No preprocessing applied (passes empty method list)
- Fits PCA with `WEIGHTS__BOX12_DM1_ONLY` concept only
- Generates plots and loadings for all 3 spectral windows × 2 plot modes
- Returns a list of metric dicts that feeds into the master table

#### Sweep engine (Section 9)

**`_get_question_dirs(out_root)`**
Creates the six question output folders:
```
02_question2_dm1_alignment/
03_question3A_clean_control_separation/
04_question3B_holistic_control_separation/
05_question4_geometry_shift/
06_question5_severity_gradient/
07_question6_exploratory_cohort_separation/
```

**`_run_one_sweep_config(...)`**
Runs one `(spectral_window, concept, preprocessing)` combination for both plot modes. Routing logic:
- `WEIGHTS__BOX12_DM1_ONLY` → plots go to Q2 (primary), Q5, Q6
  - Loadings + overlay saved in Q2 dir (PATIENT_AVG mode only)
  - Per-run `metrics.json` saved in Q2 dir
- `WEIGHTS__BOX12_ALL` → plots go to Q3A (primary), Q3B, Q4
  - Loadings + overlay saved in Q3A dir (PATIENT_AVG mode only)
  - Per-run `metrics.json` saved in Q3A dir
- Edge fractions computed and stored in the row dict

**`run_master_sweep(...)`**
Outer loop: `3 spectral_windows × 2 concepts × 9 preprocessing_combos × 2 plot_modes = 108 PCA runs`. Returns a DataFrame with one row per run.

**`_rerun_winner(winner, ...)`**
Given a winner row from the master table, reconstructs the exact preprocessing + PCA that produced it. Used by `pca_outputs.generate_winner_outputs` to regenerate assets at journal DPI.

**Data source:** Raw spectral matrices from `data_loader`, preprocessed inside the sweep loop by `pca_preprocessing.apply_pipeline_train_test`. Metadata DataFrames for group/SI labels.

---

### `pca_outputs.py`

**Purpose:** Everything after the science — leaderboards, winner packaging, export, and audit.

#### Leaderboards

**`build_leaderboards(df, summaries_dir)`**
Splits the sweep DataFrame by concept and sorts by the primary metric for each question. Saves 7 CSVs to `00_master_summaries/`:

| File | Primary sort |
|---|---|
| `severity_gradient_leaderboard.csv` | `grad_score` ↓ |
| `alignment_leaderboard.csv` | `frac_box3_inside_dm1_95ellipse` ↓ |
| `control_separation_clean_leaderboard.csv` | `cln_bhattacharyya_dm1_vs_ctrl` ↓ |
| `control_separation_holistic_leaderboard.csv` | `hol_bhattacharyya_dm1_vs_ctrl` ↓ |
| `geometry_shift_leaderboard.csv` | `frac_box3_inside_dm1_95ellipse` ↓ |
| `exploratory_cohort_separation_leaderboard.csv` | `frac_box3_inside_dm1_95ellipse` ↑ |
| `combined_alignment_separation_leaderboard.csv` | `composite_score` ↓ |

#### Winner selection

**`select_winners(boards)`**
Takes the top row of each leaderboard. Prefers `PATIENT_AVG` plot mode when available (single-point-per-patient is the primary paper representation).

**`_build_winner_reason(q_key, winner)`**
Generates a plain-English sentence explaining the winner, e.g.:
> `"Highest frac_box3_inside_dm1_95ellipse=87% (tiebreak: mahal_median=1.23); preprocessing=SNV, window=FULL_SPECTRUM, mode=PATIENT_AVG"`

#### Winner asset generation

**`generate_winner_outputs(...)`**
Re-runs each winner via `_rerun_winner` and saves assets into four folders:

| Folder | Contents |
|---|---|
| `08_winner_overlays/` | Enhanced scatter at journal DPI |
| `09_winner_loadings/` | Loadings plot + top_wavenumbers CSV |
| `10_winner_scatterplots/` | Group + SI scatter at journal DPI |
| `11_journal_ready_figures/` | Two-panel journal figure |

Returns an `asset_index` dict `{q_key: {asset_type: path_string}}`.

#### Master table export

**`export_master_summaries(...)`**
Writes to `13_master_tables_for_paper/`:
- `PCA_MASTER_ALL_RUNS.csv/.xlsx` — every run, every metric, edge flags, path columns
- `PCA_MASTER_WINNERS.csv/.xlsx` — one row per winner with `winner_reason` and asset paths

**`export_question_workbooks(df_all, winners, out_root)`**
Writes 6 Excel workbooks (one per Q2–Q6) to `13_master_tables_for_paper/`. Each workbook has 5 sheets:
- `ALL_RUNS` — full data for that concept
- `RANKED` — sorted by primary metric
- `WINNER` — the single winning row + reason text
- `TOP_10` — top 10 ranked rows
- `PAPER_CANDIDATES` — top 10 filtered to `non_edge_candidate=True`

#### Supplement bundles

**`build_supplement_bundles(boards, out_root)`**
Copies the top-10 enhanced scatter PNGs per question into `12_supplement_figure_sets/top10_{q_key}/` using `shutil.copy2`. Reads the `primary_enhanced_scatter` path column that was stored in the sweep rows.

#### Audit and README

**`write_audit_files(out_root, run_log, error_log)`**
Writes four files to `14_logs_and_audit/`:
- `PCA_RUN_LOG.txt` — timestamped run entries
- `PCA_ERRORS_AND_SKIPS.txt` — any preprocessing failures
- `PCA_CONFIG_USED.json` — all config constants serialized to JSON
- `PCA_QUESTION_DEFINITIONS.txt` — human-readable question descriptions

**`write_readme(out_root)`**
Writes `README__output_guide.txt` to `OUT_ROOT` describing every output folder.

---

### `ultimate_pca_suite.py`

**Purpose:** Orchestrator only. No scientific logic. Loads data once and calls the modules in order.

**Execution sequence:**
1. Load Cohort 1 and Cohort 2 data via `data_loader.load_data()`
2. Detect SI and DX columns, build patient-level group masks
3. `run_question1_raw_geometry()` — Q1 baseline
4. `run_master_sweep()` — Q2–Q6 full sweep, returns DataFrame
5. `build_leaderboards()` — sorts and saves CSVs
6. `select_winners()` — picks best row per question
7. `generate_winner_outputs()` — saves assets to 08/09/10/11
8. `export_master_summaries()` — master CSV/XLSX + winner table
9. `export_question_workbooks()` — per-question Excel files
10. `build_supplement_bundles()` — copies top-10 PNGs to 12
11. `export_journal_figure_index()` — writes index CSV to 11
12. `write_audit_files()` + `write_readme()` — 14 and root

---

## Inherited files (not modified)

### `data_loader.py`
Reads the raw `.txt` spectral files. Called only in `ultimate_pca_suite.main()`.
- Expects exactly 3 `.txt` files per patient, each containing a 4-row × 1732-column block (row 0 = header with wavenumbers, rows 1–3 = spectra)
- Returns `averaged_spectra (1731, N)` and `all_spectra (1731, 9N)` along with the aligned metadata DataFrame
- The canonical patient ID is `{Type}_{###}` (first two underscore-separated tokens)

### `preprocessing.py`
Implements the signal processing methods. Called only via `pca_preprocessing.apply_pipeline_train_test`.
- API: `Preprocessing().fit(X_fxN, methods)` then `.transform(X_fxN, methods)`
- Supported methods: `"Normalization"`, `"SNV"`, `"Second Derivative"`, `"EMSC"`
- EMSC stores the reference spectrum in `self.emsc_reference_` during `.fit()`

---

## Dependency graph

```
pca_config
    └── pca_utils
            └── pca_preprocessing  (also uses preprocessing.py)
                    └── pca_metrics  (also uses pca_utils)
                            └── pca_plotting  (also uses pca_utils, pca_config)
                                    └── pca_runner_core  (uses all of the above)
                                            └── pca_outputs  (uses all of the above)
                                                    └── ultimate_pca_suite
                                                            └── data_loader
```

No circular imports. Each module only imports from modules above it in this chain.
