PCA RESULTS — OUTPUT GUIDE
============================================================

00_master_summaries/
  Per-question leaderboard CSVs (alignment, separation, gradient, composite).

01_question1_raw_geometry/
  Q1: Native raw spectra, no preprocessing, DM1-only PCA axes.

02_question2_dm1_alignment/
  Q2: How well does Cohort 2 DM1 overlap Cohort 1 DM1 (DM1-only PCA)?

03_question3A_clean_control_separation/
  Q3A: How cleanly are Controls separated from Cohort 1 DM1 only?

04_question3B_holistic_control_separation/
  Q3B: Controls separation with Cohort 2 DM1 included in the disease cloud.

05_question4_geometry_shift/
  Q4: Cohort 2 DM1 alignment in the full disease/control geometry (ALL concept).

06_question5_severity_gradient/
  Q5: Preprocessing that maximises SI-severity gradient (Spearman rho).

07_question6_exploratory_cohort_separation/
  Q6: Supplementary — preprocessing-induced cohort spread.

08_winner_overlays/
  Enhanced scatter (journal DPI) for each question winner.
  Standardized name: {Q}__BEST_{LABEL}__{WINDOW}__{CONCEPT}__{MODE}__{PP}__enhanced.png

09_winner_loadings/
  Loadings plots + top_wavenumbers CSVs for each question winner.

10_winner_scatterplots/
  Group-colored and SI-colored scatter for each question winner.

11_journal_ready_figures/
  Two-panel journal figures (scatter + overlay) per winner.
  JOURNAL_READY_FIGURE_INDEX.csv — full asset index with captions.

12_supplement_figure_sets/
  question{N}_top10_{label}/ — full asset sets for top-10 runs per question.
  Assets: group scatter, SI scatter, enhanced scatter, loadings, overlay, top_wn CSV.

13_master_tables_for_paper/
  PCA_MASTER_ALL_RUNS.csv/.xlsx  — all runs, all metrics.
  PCA_MASTER_WINNERS.csv/.xlsx   — one row per winner with reason + asset paths.
  question{N}_{label}_winners.xlsx — per-question workbooks.
    Sheets: ALL_RUNS | RANKED | WINNER | TOP_10 | PAPER_CANDIDATES
  top_non_edge_candidates.xlsx/.csv — runs with non-edge-dominated loadings.
  edge_dominated_runs.xlsx/.csv     — runs flagged as edge-artefact dominated.

14_logs_and_audit/
  PCA_RUN_LOG.txt             — timestamped entry per completed run.
  PCA_ERRORS_AND_SKIPS.txt    — any preprocessing failures.
  PCA_CONFIG_USED.json        — full config at run time.
  PCA_QUESTION_DEFINITIONS.txt — plain-text question descriptions.

KEY METRICS
  frac_box3_inside_dm1_95ellipse  fraction of Cohort 2 DM1 patients inside Cohort 1 DM1 95% ellipse
  bhattacharyya_dm1_vs_ctrl       higher = better Controls vs DM1 separation (Bhattacharyya distance)
  grad_score                      max(|Spearman rho(PC1,SI)|, |Spearman rho(PC2,SI)|)
  composite_score                 50% alignment + 50% separation (sweep tool only)
  edge_flag                       True if >50% of top-k loading wavenumbers in outer 5% of range
  non_edge_candidate              not edge_flag — recommended filter for paper candidates

HOW TO RE-RUN
  python ultimate_pca_suite.py
