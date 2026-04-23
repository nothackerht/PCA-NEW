# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to run

```bash
# Full suite (Q1–Q6, all preprocessing combos, all outputs)
python ultimate_pca_suite.py

# Verify all modules import cleanly after any edit
python -c "import pca_config, pca_utils, pca_preprocessing, pca_metrics, pca_plotting, pca_runner_core, pca_outputs, ultimate_pca_suite; print('All imports OK')"
```

There are no tests, no build step, and no linter configured.

## Before running: set paths in pca_config.py

Six path constants at the top of `pca_config.py` must point to the correct machine locations:

```python
OUT_ROOT        # where all numbered output folders are written
BOX12_DATA_DIR  # Cohort 1 .txt spectra folder
BOX12_META_CSV  # Cohort 1 metadata CSV
BOX3_DATA_DIR   # Cohort 2 .txt spectra folder
BOX3_META_CSV   # Cohort 2 metadata CSV
```

`BASELINE_FLAGS` must stay `[False]` — baseline correction is not implemented in `preprocessing.py`.

## Architecture

The suite is a linear pipeline orchestrated by `ultimate_pca_suite.py`:

```
data_loader.py           → load raw spectra + metadata
pca_preprocessing.py     → fit-on-train-only wrapper around preprocessing.py
pca_runner_core.py       → PCA fitting, Q1 runner, master sweep (Q2–Q6)
pca_metrics.py           → all metric computations (alignment, separation, gradient)
pca_plotting.py          → all figure generation; every save function returns a Path
pca_outputs.py           → leaderboards, winner selection, export, audit, README
pca_config.py            → all constants (paths, sweep params, plot settings)
pca_utils.py             → stateless helpers (masks, combos, string/trim utils)
```

### Data format

- Each patient: exactly 3 `.txt` files × 3 spectra each = 9 spectra per patient.
- File format: 4 rows × 1732 columns (row 0 = wavenumber header, rows 1–3 = spectra).
- All spectral matrices are shape `(features, N)` — wavenumbers on axis 0, samples on axis 1. This is the transpose of the sklearn convention; `fit_pca_and_project` handles `.T` internally.
- Metadata CSV: needs `Sample_ID`, `Type`, and a Splicing Index column (auto-detected by `detect_si_column` as `target_SI`, `SI`, `SplicingIndex`, or `Splicing_Index`). `Type` values: `"DM1"` and `"Control"` / `"AdCo"`.

### Two concepts drive PCA fitting

| Concept | Fitting matrix | Questions |
|---|---|---|
| `WEIGHTS__BOX12_DM1_ONLY` | Cohort 1 DM1 spectra only | Q2, Q5, Q6 |
| `WEIGHTS__BOX12_ALL` | All Cohort 1 spectra (DM1 + Control) | Q3A, Q3B, Q4 |

`build_fit_matrix()` selects the matrix; `fit_pca_and_project()` fits on the concept matrix but projects both cohorts.

### Two plot modes per run

Each `(spectrum_mode, concept, preprocessing)` combo runs twice:
- `PATIENT_AVG` — one averaged spectrum per patient
- `ALL_SPECTRA` — all 9 raw spectra per patient (masks/metadata are `np.repeat`-ed × 9)

### Sweep dimensions

3 spectral windows × 2 concepts × 9 preprocessing combos × 2 plot modes = 108 rows per concept per window. Preprocessing combos: RAW, Normalization, SNV, Second Derivative, EMSC, Normalization+EMSC, SNV+Second Derivative, Normalization+SNV, Normalization+SNV+Second Derivative.

### Output folder numbering

```
00_master_summaries/         leaderboard CSVs
01_question1_raw_geometry/   Q1 no-preprocessing baseline
02–07                        Q2 (DM1 alignment), Q3A, Q3B, Q4, Q5, Q6
08_winner_overlays/          journal-DPI enhanced scatter per question winner
09_winner_loadings/          loadings + top_wavenumbers.csv per winner
10_winner_scatterplots/      group + SI scatter per winner
11_journal_ready_figures/    two-panel figure + JOURNAL_READY_FIGURE_INDEX.csv
12_supplement_figure_sets/   top-10 asset bundles per question
13_master_tables_for_paper/  PCA_MASTER_ALL_RUNS.csv/.xlsx, per-Q workbooks
14_logs_and_audit/           run log, error log, config JSON
```

Within each run folder, files use **short fixed names**: `group.png`, `si_gradient.png`, `overlay.png`, `loadings.png`, `top_wavenumbers.csv`, `metrics.json`, and `{suffix_clean}.png` for enhanced scatters (e.g. `Q2alignment.png`).

## Windows path-length rules

**Never build filenames from `sanitize_filename(title)` or `sanitize_filename(run_name)`.** The folder hierarchy encodes all experiment identity. All plot functions accept an optional `fname_stem` parameter — pass it only when two files of the same type land in the same folder (e.g. Q1 writes both `PATIENT_AVG` and `ALL_SPECTRA` into the same dir). Use `safe_save_path(out_dir, name)` (in `pca_plotting.py`) for all new save sites — it calls `mkdir(parents=True, exist_ok=True)` before returning the path.

## Key metric columns

| Column | Meaning |
|---|---|
| `frac_box3_inside_dm1_95ellipse` | Cohort 2 DM1 overlap with Cohort 1 DM1 — Q2 primary sort key |
| `hol_bhattacharyya_dm1_vs_ctrl` | DM1 vs Control separation, holistic (Cohort 1 + 2 DM1) |
| `cln_bhattacharyya_dm1_vs_ctrl` | DM1 vs Control separation, clean Cohort 1 DM1 only |
| `grad_score` | max(\|Spearman ρ(PC1,SI)\|, \|Spearman ρ(PC2,SI)\|) — Q5 primary |
| `composite_score` | 50% alignment + 50% separation, rank-normalised across full sweep |
| `edge_flag` | True if >50% of top-k loading wavenumbers fall in outer 5% of spectral range |

`hol_` prefix = holistic (includes Cohort 2 DM1 in the disease cloud for silhouette/centroid); `cln_` prefix = Cohort 1 DM1 only. Bhattacharyya and Fisher always use Cohort 1 training distributions regardless of prefix.
