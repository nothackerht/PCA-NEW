#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ultimate_pca_suite.py
=====================
Lightweight orchestrator for the SERS DM1 PCA suite.

Scientific logic lives in the supporting modules:
  pca_config.py         — paths, constants, question definitions
  pca_utils.py          — string / mask / combo helpers
  pca_preprocessing.py  — train/test preprocessing wrapper
  pca_metrics.py        — alignment, separation, gradient, composite metrics
  pca_plotting.py       — all visualization functions
  pca_runner_core.py    — PCA fitting, Q1 runner, master sweep
  pca_outputs.py        — leaderboards, winners, export, audit, README

Run:
  python ultimate_pca_suite.py
"""

from __future__ import annotations

import numpy as np

from data_loader import load_data

from pca_config import (
    BOX12_DATA_DIR, BOX12_META_CSV,
    BOX3_DATA_DIR,  BOX3_META_CSV,
    OUT_ROOT,
)
from pca_outputs import (
    build_leaderboards,
    build_supplement_bundles,
    export_edge_tables,
    export_journal_figure_index,
    export_master_summaries,
    export_question_workbooks,
    generate_winner_outputs,
    select_winners,
    write_audit_files,
    write_readme,
)
from pca_runner_core import run_master_sweep, run_question1_raw_geometry
from pca_utils import detect_dx_column, detect_si_column, get_dm1_control_masks


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ULTIMATE PCA SUITE — SERS DM1 PAPER")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading Cohort 1 (train) data …")
    wn_train, X_avg_train, X_all_train, meta_train = load_data(
        data_dir=BOX12_DATA_DIR,
        metadata_path=BOX12_META_CSV,
        include_types=["DM1", "Control"],
        strict=False,
    )
    meta_train["Box"] = "Cohort1"

    print("Loading Cohort 2 (test) data …")
    wn_test, X_avg_test, X_all_test, meta_test = load_data(
        data_dir=BOX3_DATA_DIR,
        metadata_path=BOX3_META_CSV,
        include_types=["DM1", "Control"],
        strict=False,
    )
    meta_test["Box"] = "Cohort2"

    if not np.allclose(wn_train, wn_test):
        raise ValueError("Wavenumber axes differ between Cohort 1 and Cohort 2.")

    wavenumbers_full = wn_train
    si_col  = detect_si_column(meta_train)
    dx_col  = detect_dx_column(meta_train)
    dm1_mask_train, ctrl_mask_train = get_dm1_control_masks(
        meta_train, dx_col, require_controls=True
    )

    print(f"Cohort 1: {dm1_mask_train.sum()} DM1, {ctrl_mask_train.sum()} controls")
    print(f"Cohort 2: {len(meta_test)} patients")
    print(f"SI col: {si_col}   |   DX col: {dx_col}")

    # ------------------------------------------------------------------
    # Q1 — Raw geometry (no preprocessing)
    # ------------------------------------------------------------------
    print("\n── Q1: Raw geometry ──")
    q1_dir  = OUT_ROOT / "01_question1_raw_geometry"
    q1_rows = run_question1_raw_geometry(
        q1_dir, wavenumbers_full,
        X_avg_train, X_avg_test, X_all_train, X_all_test,
        meta_train, meta_test, si_col, dx_col,
        dm1_mask_train, ctrl_mask_train,
    )

    # ------------------------------------------------------------------
    # Master sweep — Q2 through Q6
    # ------------------------------------------------------------------
    print("\n── Master sweep: Q2–Q6 ──")
    df_sweep, run_log, error_log = run_master_sweep(
        OUT_ROOT, wavenumbers_full,
        X_avg_train, X_avg_test, X_all_train, X_all_test,
        meta_train, meta_test, si_col, dx_col,
        dm1_mask_train, ctrl_mask_train,
    )

    # ------------------------------------------------------------------
    # Leaderboards
    # ------------------------------------------------------------------
    print("\n── Building leaderboards ──")
    summaries_dir = OUT_ROOT / "00_master_summaries"
    boards = build_leaderboards(df_sweep, summaries_dir)

    # ------------------------------------------------------------------
    # Winner selection
    # ------------------------------------------------------------------
    winners = select_winners(boards)
    print(f"\nSelected {len(winners)} winners:")
    for k, w in winners.items():
        print(f"  {k}: {w.get('preprocessing')} | {w.get('spectrum_mode')} | {w.get('plot_type')}")

    # ------------------------------------------------------------------
    # Winner asset generation (08 / 09 / 10 / 11)
    # ------------------------------------------------------------------
    print("\n── Generating winner assets ──")
    winner_assets = generate_winner_outputs(
        winners, boards, OUT_ROOT,
        wavenumbers_full,
        X_avg_train, X_avg_test, X_all_train, X_all_test,
        meta_train, meta_test, si_col, dx_col,
        dm1_mask_train, ctrl_mask_train,
    )

    # ------------------------------------------------------------------
    # Export master tables (13)
    # ------------------------------------------------------------------
    print("\n── Exporting master tables ──")
    df_all = export_master_summaries(
        df_sweep, q1_rows, boards, winners, winner_assets, OUT_ROOT,
    )

    # Per-question workbooks
    export_question_workbooks(df_all, winners, OUT_ROOT)

    # Edge artifact tables
    export_edge_tables(df_all, OUT_ROOT)

    # ------------------------------------------------------------------
    # Supplement bundles (12)
    # ------------------------------------------------------------------
    print("\n── Building supplement bundles ──")
    build_supplement_bundles(boards, df_all, OUT_ROOT)

    # ------------------------------------------------------------------
    # Journal figure index (11)
    # ------------------------------------------------------------------
    export_journal_figure_index(winner_assets, winners, OUT_ROOT)

    # ------------------------------------------------------------------
    # Audit + README (14)
    # ------------------------------------------------------------------
    print("\n── Writing audit files and README ──")
    write_audit_files(OUT_ROOT, run_log=run_log, error_log=error_log)
    write_readme(OUT_ROOT)

    print(f"\n{'='*60}")
    print(f"DONE.  All outputs saved to:\n  {OUT_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
