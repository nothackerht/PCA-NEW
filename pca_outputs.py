#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_outputs.py
==============
All output/reporting infrastructure:
  - Leaderboards (per-question CSVs)
  - Winner selection + winner-reason text
  - Master table enrichment (ranks, winner flags, column aliases)
  - Centralized winner assets (08 / 09 / 10) with standardized naming
  - Journal-ready figures + richer index CSV (11)
  - Per-question Excel workbooks with 5 sheets each (13)
  - Supplement figure bundles with full asset sets (12)
  - Edge/artifact summary tables (in 13)
  - Audit logs + config dump (14)
  - README__output_guide.txt
"""

from __future__ import annotations

import json
import shutil
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pca_config import (
    BASELINE_FLAGS, BOX12_DATA_DIR, BOX12_META_CSV,
    BOX3_DATA_DIR, BOX3_META_CSV, CONCEPTS, FIGURE_DPI,
    GAP_THRESH, JOURNAL_DPI, OUT_ROOT, QUESTION_DEFINITIONS,
    SPECTRUM_MODES, SUPPORTED_METHODS, TOP_K_WAVENUMBERS,
)
from pca_metrics import add_composite_score
from pca_plotting import (
    make_journal_figure,
    plot_enhanced_pca_scatter,
    plot_grad_axis,
    plot_grad_bins,
    plot_grad_panels,
    plot_grad_proj,
    plot_gradient_direction,
    plot_group_enh,
    plot_group_scatter,
    plot_si_scatter,
    save_loading_and_overlay,
)
from pca_utils import get_dm1_control_masks, sanitize_filename


# =============================================================================
# SECTION 11 — LEADERBOARDS
# =============================================================================

def build_leaderboards(df: pd.DataFrame, summaries_dir: Path) -> Dict[str, pd.DataFrame]:
    """Build and save all question-specific leaderboard CSVs."""
    summaries_dir.mkdir(parents=True, exist_ok=True)

    dm1_only = df[df["concept"] == "WEIGHTS__BOX12_DM1_ONLY"].copy()
    all_cpt  = df[df["concept"] == "WEIGHTS__BOX12_ALL"].copy()

    boards: Dict[str, pd.DataFrame] = {}

    lb_q5 = dm1_only.sort_values("grad_score", ascending=False)
    lb_q5.to_csv(summaries_dir / "severity_gradient_leaderboard.csv", index=False)
    boards["Q5_gradient"] = lb_q5

    lb_q2 = dm1_only.sort_values(
        ["frac_box3_inside_dm1_95ellipse", "mahalanobis_median_box3_to_box12dm1"],
        ascending=[False, True],
    )
    lb_q2.to_csv(summaries_dir / "alignment_leaderboard.csv", index=False)
    boards["Q2_alignment"] = lb_q2

    lb_q3a = all_cpt.sort_values(
        ["cln_bhattacharyya_dm1_vs_ctrl", "cln_frac_controls_outside_dm1_95ellipse"],
        ascending=[False, False],
    )
    lb_q3a.to_csv(summaries_dir / "control_separation_clean_leaderboard.csv", index=False)
    boards["Q3A_clean"] = lb_q3a

    lb_q3b = all_cpt.sort_values(
        ["hol_bhattacharyya_dm1_vs_ctrl", "hol_silhouette_dm1_vs_ctrl"],
        ascending=[False, False],
    )
    lb_q3b.to_csv(summaries_dir / "control_separation_holistic_leaderboard.csv", index=False)
    boards["Q3B_holistic"] = lb_q3b

    lb_q4 = all_cpt.sort_values(
        ["frac_box3_inside_dm1_95ellipse", "mahalanobis_mean_box3_to_box12dm1"],
        ascending=[False, True],
    )
    lb_q4.to_csv(summaries_dir / "geometry_shift_leaderboard.csv", index=False)
    boards["Q4_shift"] = lb_q4

    lb_q6 = dm1_only.sort_values("frac_box3_inside_dm1_95ellipse", ascending=True)
    lb_q6.to_csv(summaries_dir / "exploratory_cohort_separation_leaderboard.csv", index=False)
    boards["Q6_exploratory"] = lb_q6

    df_combined = add_composite_score(df.copy())
    df_combined = df_combined.sort_values("composite_score", ascending=False)
    df_combined.to_csv(summaries_dir / "combined_alignment_separation_leaderboard.csv", index=False)
    boards["combined"] = df_combined

    return boards


# =============================================================================
# SECTION 12 — WINNER SELECTION + REASON TEXT
# =============================================================================

def select_winners(boards: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """Select the single best row per question (prefers PATIENT_AVG mode)."""
    def top1(lb: pd.DataFrame) -> Optional[pd.Series]:
        if lb.empty:
            return None
        if "plot_type" in lb.columns:
            sub = lb[lb["plot_type"] == "PATIENT_AVG"]
            return sub.iloc[0] if not sub.empty else lb.iloc[0]
        return lb.iloc[0]

    return {key: w for key, lb in boards.items() if (w := top1(lb)) is not None}


def _fmt(val, fmt: str = ".2f") -> str:
    try:
        return f"{float(val):{fmt}}" if np.isfinite(float(val)) else "N/A"
    except Exception:
        return "N/A"


def _pct(val) -> str:
    try:
        return f"{float(val)*100:.0f}%" if np.isfinite(float(val)) else "N/A"
    except Exception:
        return "N/A"


def _build_winner_reason(q_key: str, winner: pd.Series) -> str:
    """Human-readable explanation of why this configuration won."""
    pp = winner.get("preprocessing", "RAW")
    sm = winner.get("spectrum_mode", "")
    pt = winner.get("plot_type", "")

    if q_key == "Q2_alignment":
        v   = winner.get("frac_box3_inside_dm1_95ellipse", np.nan)
        med = winner.get("mahalanobis_median_box3_to_box12dm1", np.nan)
        return (f"Highest frac_box3_inside_dm1_95ellipse={_pct(v)} "
                f"(tiebreak: mahal_median={_fmt(med)}); "
                f"preprocessing={pp}, window={sm}, mode={pt}")
    elif q_key == "Q3A_clean":
        bd  = winner.get("cln_bhattacharyya_dm1_vs_ctrl", np.nan)
        fco = winner.get("cln_frac_controls_outside_dm1_95ellipse", np.nan)
        return (f"Highest cln_bhattacharyya={_fmt(bd)} "
                f"(tiebreak: ctrl_outside_ellipse={_pct(fco)}); "
                f"preprocessing={pp}, window={sm}, mode={pt}")
    elif q_key == "Q3B_holistic":
        bd  = winner.get("hol_bhattacharyya_dm1_vs_ctrl", np.nan)
        sil = winner.get("hol_silhouette_dm1_vs_ctrl", np.nan)
        return (f"Highest hol_bhattacharyya={_fmt(bd)} "
                f"(tiebreak: silhouette={_fmt(sil, '.3f')}); "
                f"preprocessing={pp}, window={sm}, mode={pt}")
    elif q_key == "Q4_shift":
        v     = winner.get("frac_box3_inside_dm1_95ellipse", np.nan)
        mahal = winner.get("mahalanobis_mean_box3_to_box12dm1", np.nan)
        return (f"Highest frac_box3_inside_dm1_95ellipse={_pct(v)} in BOX12_ALL space; "
                f"mahal_mean={_fmt(mahal)}; "
                f"preprocessing={pp}, window={sm}, mode={pt}")
    elif q_key == "Q5_gradient":
        gs = winner.get("grad_score", np.nan)
        r1 = winner.get("rho_pc1", np.nan)
        r2 = winner.get("rho_pc2", np.nan)
        return (f"Highest grad_score={_fmt(gs)} "
                f"(rho_PC1={_fmt(r1)}, rho_PC2={_fmt(r2)}); "
                f"preprocessing={pp}, window={sm}, mode={pt}")
    elif q_key == "Q6_exploratory":
        v = winner.get("frac_box3_inside_dm1_95ellipse", np.nan)
        return (f"Lowest frac_box3_inside_dm1_95ellipse={_pct(v)} "
                f"(most cohort separation); "
                f"preprocessing={pp}, window={sm}, mode={pt}")
    elif q_key == "combined":
        cs = winner.get("composite_score", np.nan)
        return (f"Highest composite_score={_fmt(cs)} "
                f"(50% alignment + 50% separation); "
                f"preprocessing={pp}, window={sm}, mode={pt}")
    else:
        return f"Winner for {q_key}: preprocessing={pp}, window={sm}, mode={pt}"


# =============================================================================
# Master table enrichment
# =============================================================================

def _enrich_master_table(
    df: pd.DataFrame,
    winners: Dict[str, pd.Series],
    boards: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Add derived columns to the master run table:
      - control_dm1_centroid_dist alias
      - per-question rank columns
      - is_question_winner, is_question_runner_up
      - is_journal_candidate, is_supplement_candidate
      - winner_reason for winning rows
      - column aliases for paper-friendly names
    """
    df = df.copy()
    n  = len(df)
    if n == 0:
        return df

    # Alias control centroid dist
    if "hol_centroid_dist_dm1_vs_ctrl" in df.columns and "control_dm1_centroid_dist" not in df.columns:
        df["control_dm1_centroid_dist"] = df["hol_centroid_dist_dm1_vs_ctrl"]

    # Build a match key for winner identification
    def _match_key(row: pd.Series) -> str:
        return f"{row.get('spectrum_mode','')}|{row.get('concept','')}|{row.get('preprocessing','')}|{row.get('plot_type','')}"

    winner_keys = {_match_key(w): q for q, w in winners.items()}

    # Rank per question
    _Q_RANK_SPECS = [
        ("rank_q2",  "WEIGHTS__BOX12_DM1_ONLY", "frac_box3_inside_dm1_95ellipse", False),
        ("rank_q3a", "WEIGHTS__BOX12_ALL",       "cln_bhattacharyya_dm1_vs_ctrl",  False),
        ("rank_q3b", "WEIGHTS__BOX12_ALL",       "hol_bhattacharyya_dm1_vs_ctrl",  False),
        ("rank_q4",  "WEIGHTS__BOX12_ALL",       "frac_box3_inside_dm1_95ellipse", False),
        ("rank_q5",  "WEIGHTS__BOX12_DM1_ONLY",  "grad_score",                     False),
        ("rank_q6",  "WEIGHTS__BOX12_DM1_ONLY",  "frac_box3_inside_dm1_95ellipse", True),
    ]
    for rank_col, concept_filter, sort_col, sort_asc in _Q_RANK_SPECS:
        df[rank_col] = np.nan
        if sort_col not in df.columns:
            continue
        mask = df["concept"] == concept_filter
        df.loc[mask, rank_col] = (
            df.loc[mask, sort_col]
              .rank(ascending=sort_asc, na_option="bottom")
              .astype(float)
        )

    # Winner / runner-up / candidate flags
    df["is_question_winner"]   = False
    df["is_question_runner_up"] = False
    df["winner_reason"]         = ""

    for idx, row in df.iterrows():
        mk = _match_key(row)
        if mk in winner_keys:
            df.at[idx, "is_question_winner"] = True
            df.at[idx, "winner_reason"] = _build_winner_reason(winner_keys[mk], row)

    # Runner-up: rank 2 in any relevant leaderboard
    for rank_col in ["rank_q2", "rank_q3a", "rank_q3b", "rank_q4", "rank_q5", "rank_q6"]:
        if rank_col in df.columns:
            df.loc[df[rank_col] == 2, "is_question_runner_up"] = True

    df["is_journal_candidate"]   = df["is_question_winner"]
    df["is_supplement_candidate"] = False
    for rank_col in ["rank_q2", "rank_q3a", "rank_q3b", "rank_q4", "rank_q5", "rank_q6"]:
        if rank_col in df.columns:
            df.loc[df[rank_col] <= 10, "is_supplement_candidate"] = True

    # Rename path columns to paper-friendly names if not already present
    _path_renames = {
        "group_scatter_path":    "scatter_plot_path",
        "si_scatter_path":       "si_scatter_plot_path",
        "primary_enhanced_scatter": "enhanced_plot_path",
        "loadings_plot_path":    "loadings_plot_path",
        "overlay_plot_path":     "overlay_plot_path",
        "top_wavenumbers_csv_path": "top_wavenumbers_csv_path",
    }
    for old, new in _path_renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
        elif old in df.columns and new in df.columns and old != new:
            df[new] = df[old].where(df[new] == "", df[new])

    return df


# =============================================================================
# SECTION 13 — WINNER ASSET GENERATION
# =============================================================================

def _winner_asset_stem(q_key: str, winner: pd.Series) -> str:
    """Short stem — folder already encodes question identity."""
    return "winner"


def generate_winner_outputs(
    winners: Dict[str, pd.Series],
    boards: Dict[str, pd.DataFrame],
    out_root: Path,
    wavenumbers_full: np.ndarray,
    X_avg_train, X_avg_test,
    X_all_train, X_all_test,
    meta_train, meta_test,
    si_col: str, dx_col: str,
    dm1_mask_train, ctrl_mask_train,
) -> Dict[str, Dict[str, str]]:
    """
    For each question winner generate and save assets into:
      08_winner_overlays/      — enhanced scatter  (JOURNAL_DPI)
      09_winner_loadings/      — loadings + top_wn CSV
      10_winner_scatterplots/  — group + SI scatter (JOURNAL_DPI)
      11_journal_ready_figures/ — two-panel figure

    Returns asset_index: {q_key: {asset_type: path_str}}
    """
    from pca_runner_core import _rerun_winner
    from pca_metrics import compute_alignment_metrics, compute_separation_metrics

    dir_enh  = out_root / "08_winner_overlays"
    dir_load = out_root / "09_winner_loadings"
    dir_scat = out_root / "10_winner_scatterplots"
    dir_jour = out_root / "11_journal_ready_figures"
    for d in [dir_enh, dir_load, dir_scat, dir_jour]:
        d.mkdir(parents=True, exist_ok=True)

    asset_index: Dict[str, Dict[str, str]] = {}

    for q_key, winner in winners.items():
        print(f"[WINNER] {q_key} …")
        try:
            pca, sc12, sc3, wn_use, X_ov_tr, X_ov_te, evr = _rerun_winner(
                winner, wavenumbers_full,
                X_avg_train, X_avg_test, X_all_train, X_all_test,
                meta_train, meta_test, si_col, dx_col,
                dm1_mask_train, ctrl_mask_train,
            )
        except Exception as exc:
            print(f"  [SKIP] {q_key}: {exc}")
            continue

        stem     = _winner_asset_stem(q_key, winner)
        assets: Dict[str, str] = {}

        dm1_3, _ = get_dm1_control_masks(meta_test, dx_col, require_controls=False)

        if winner.get("plot_type") == "PATIENT_AVG":
            dm1_msk    = dm1_mask_train
            ctrl_msk   = ctrl_mask_train
            meta_sc_tr = meta_train
            meta_sc_te = meta_test
        else:
            dm1_msk    = np.repeat(dm1_mask_train, 9)
            ctrl_msk   = np.repeat(ctrl_mask_train, 9)
            meta_sc_tr = meta_train.loc[np.repeat(np.arange(len(meta_train)), 9)].reset_index(drop=True)
            meta_sc_te = meta_test.loc[np.repeat(np.arange(len(meta_test)),  9)].reset_index(drop=True)

        s_dm1_12  = sc12[dm1_msk]
        s_ctrl_12 = sc12[ctrl_msk]
        s_dm1_3   = (sc3[dm1_3] if winner.get("plot_type") == "PATIENT_AVG"
                     else sc3[np.repeat(dm1_3, 9)])

        align   = compute_alignment_metrics(s_dm1_3, s_dm1_12)
        sep_hol = compute_separation_metrics(s_dm1_12, s_dm1_3, s_ctrl_12, include_box3_in_dm1=True)

        # 08 — enhanced scatter
        w_enh = dir_enh / sanitize_filename(q_key)
        w_enh.mkdir(parents=True, exist_ok=True)
        enh_p = plot_enhanced_pca_scatter(
            w_enh, stem, sc12, sc3,
            meta_sc_tr, meta_sc_te, si_col, dx_col, evr,
            align, sep_hol, suffix="__enhanced", dpi=JOURNAL_DPI,
            fname_stem=stem,
        )
        assets["enhanced_scatter"] = str(enh_p)

        # 09 — loadings + top_wn CSV (group-colored overlay)
        w_load = dir_load / sanitize_filename(q_key)
        w_load.mkdir(parents=True, exist_ok=True)
        wn1, wn2, lp, op, csv_p = save_loading_and_overlay(
            w_load, stem, wn_use, pca,
            X_ov_tr, X_ov_te,
            top_k=TOP_K_WAVENUMBERS, gap_thresh=GAP_THRESH, dpi=JOURNAL_DPI,
            fname_stem=stem,
            meta_train=meta_sc_tr, meta_test=meta_sc_te, dx_col=dx_col,
        )
        assets["loadings"]       = str(lp)
        assets["overlay"]        = str(op)
        assets["top_wn_csv"]     = str(csv_p)

        # 10 — group + SI + group_enh scatters
        w_scat = dir_scat / sanitize_filename(q_key)
        w_scat.mkdir(parents=True, exist_ok=True)
        gp = plot_group_scatter(w_scat, stem,
                                sc12, sc3, meta_sc_tr, meta_sc_te, dx_col, evr,
                                fname_stem=stem)
        sp = plot_si_scatter(w_scat, stem,
                             sc12, sc3, meta_sc_tr, meta_sc_te, si_col, dx_col, evr,
                             fname_stem=stem)
        gep = plot_group_enh(w_scat, stem,
                             sc12, sc3, meta_sc_tr, meta_sc_te, dx_col, evr,
                             align, sep_hol, dpi=JOURNAL_DPI, fname_stem=stem)
        assets["group_scatter"] = str(gp)
        assets["si_scatter"]    = str(sp)
        assets["group_enh"]     = str(gep)

        # gradient figures (DM1-only concepts only)
        if winner.get("concept", "") == "WEIGHTS__BOX12_DM1_ONLY":
            grad_metrics = {
                "rho_pc1":    float(winner.get("rho_pc1",    np.nan)),
                "rho_pc2":    float(winner.get("rho_pc2",    np.nan)),
                "grad_score": float(winner.get("grad_score", np.nan)),
            }
            wtitle = f"Winner {q_key} gradient"
            gd_p  = plot_gradient_direction(
                w_scat, wtitle, sc12, meta_sc_tr, si_col, dx_col, evr,
                grad_metrics, dpi=JOURNAL_DPI, fname_stem=stem)
            ax_p  = plot_grad_axis(
                w_scat, wtitle, sc12, meta_sc_tr, si_col, dx_col, evr,
                grad_metrics, dpi=JOURNAL_DPI, fname_stem=stem)
            pr_p  = plot_grad_proj(
                w_scat, wtitle, sc12, meta_sc_tr, si_col, dx_col, evr,
                grad_metrics, dpi=JOURNAL_DPI, fname_stem=stem)
            bn_p  = plot_grad_bins(
                w_scat, wtitle, sc12, meta_sc_tr, si_col, dx_col, evr,
                grad_metrics, dpi=JOURNAL_DPI, fname_stem=stem)
            pn_p  = plot_grad_panels(
                w_scat, wtitle, sc12, meta_sc_tr, si_col, dx_col, evr,
                grad_metrics, dpi=JOURNAL_DPI, fname_stem=stem)
            assets["grad_dir"]    = str(gd_p)
            assets["grad_axis"]   = str(ax_p)
            assets["grad_proj"]   = str(pr_p)
            assets["grad_bins"]   = str(bn_p)
            assets["grad_panels"] = str(pn_p)

        # 11 — two-panel journal figure
        jp = make_journal_figure(
            dir_jour, q_key, winner, wn_use, pca,
            X_ov_tr, X_ov_te, sc12, sc3,
            meta_sc_tr, meta_sc_te, si_col, dx_col, evr,
            align, sep_hol, wn1, wn2,
        )
        assets["journal_figure"] = str(jp)

        asset_index[q_key] = assets

    print("[WINNER] All winner outputs generated.")
    return asset_index


# =============================================================================
# Master table export + enrichment
# =============================================================================

def export_master_summaries(
    df: pd.DataFrame,
    q1_rows: List[Dict],
    boards: Dict[str, pd.DataFrame],
    winners: Dict[str, pd.Series],
    winner_assets: Dict[str, Dict[str, str]],
    out_root: Path,
) -> pd.DataFrame:
    """Save master CSV/XLSX and winner table to 13_master_tables_for_paper/."""
    tables_dir = out_root / "13_master_tables_for_paper"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df_q1  = pd.DataFrame(q1_rows) if q1_rows else pd.DataFrame()
    df_all = pd.concat([df_q1, df], ignore_index=True) if not df_q1.empty else df.copy()

    df_all = add_composite_score(df_all)
    df_all = _enrich_master_table(df_all, winners, boards)

    df_all.to_csv(tables_dir / "PCA_MASTER_ALL_RUNS.csv", index=False)
    _try_excel(df_all, tables_dir / "PCA_MASTER_ALL_RUNS.xlsx")

    # Winners table
    if winners:
        winner_rows = []
        for key, row in winners.items():
            d = row.to_dict()
            d["question_key"]  = key
            d["winner_reason"] = _build_winner_reason(key, row)
            for atype, apath in winner_assets.get(key, {}).items():
                d[f"asset_{atype}"] = apath
            winner_rows.append(d)
        df_w = pd.DataFrame(winner_rows).set_index("question_key")
        df_w.to_csv(tables_dir / "PCA_MASTER_WINNERS.csv")
        _try_excel(df_w.reset_index(), tables_dir / "PCA_MASTER_WINNERS.xlsx")

    _print_leaderboard_preview("Q5 Severity Gradient",
                               boards.get("Q5_gradient"), "grad_score")
    _print_leaderboard_preview("Q2 DM1 Alignment",
                               boards.get("Q2_alignment"), "frac_box3_inside_dm1_95ellipse")
    _print_leaderboard_preview("Q3A Clean Separation",
                               boards.get("Q3A_clean"), "cln_bhattacharyya_dm1_vs_ctrl")
    _print_leaderboard_preview("Combined Composite",
                               boards.get("combined"), "composite_score")

    return df_all


def _print_leaderboard_preview(
    title: str, df: Optional[pd.DataFrame], sort_col: str, n: int = 5
) -> None:
    if df is None or df.empty:
        return
    cols = ["spectrum_mode", "concept", "preprocessing", "plot_type", sort_col]
    cols = [c for c in cols if c in df.columns]
    print(f"\n── {title} (top {n}) ──")
    print(df[cols].head(n).to_string(index=False))


# =============================================================================
# Per-question Excel workbooks
# =============================================================================

# (concept_filter, sort_col, sort_asc, xlsx_filename)
_WORKBOOK_SPECS = [
    ("Q2_alignment",   "WEIGHTS__BOX12_DM1_ONLY", "frac_box3_inside_dm1_95ellipse", False,
     "question2_dm1_alignment_winners.xlsx"),
    ("Q3A_clean",      "WEIGHTS__BOX12_ALL",       "cln_bhattacharyya_dm1_vs_ctrl",  False,
     "question3A_clean_control_separation_winners.xlsx"),
    ("Q3B_holistic",   "WEIGHTS__BOX12_ALL",       "hol_bhattacharyya_dm1_vs_ctrl",  False,
     "question3B_holistic_control_separation_winners.xlsx"),
    ("Q4_shift",       "WEIGHTS__BOX12_ALL",       "frac_box3_inside_dm1_95ellipse", False,
     "question4_geometry_shift_winners.xlsx"),
    ("Q5_gradient",    "WEIGHTS__BOX12_DM1_ONLY",  "grad_score",                     False,
     "question5_severity_gradient_winners.xlsx"),
    ("Q6_exploratory", "WEIGHTS__BOX12_DM1_ONLY",  "frac_box3_inside_dm1_95ellipse", True,
     "question6_exploratory_cohort_separation_winners.xlsx"),
]


def export_question_workbooks(
    df_all: pd.DataFrame,
    winners: Dict[str, pd.Series],
    out_root: Path,
) -> None:
    """Write per-question Excel workbooks to 13_master_tables_for_paper/."""
    tables_dir = out_root / "13_master_tables_for_paper"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Q1 workbook
    if "question" in df_all.columns:
        df_q1 = df_all[df_all["question"] == "Q1_RAW"].copy()
    elif "question_id" in df_all.columns:
        df_q1 = df_all[df_all["question_id"] == "Q1"].copy()
    else:
        df_q1 = pd.DataFrame()
    _write_question_workbook(
        df_q1, None, "Q1_raw_geometry",
        tables_dir, "pc1_var", False,
        "question1_raw_geometry_summary.xlsx",
    )

    for q_key, concept_filter, sort_col, sort_asc, xlsx_name in _WORKBOOK_SPECS:
        sub = df_all[df_all["concept"] == concept_filter].copy() if "concept" in df_all.columns else pd.DataFrame()
        _write_question_workbook(sub, winners.get(q_key), q_key, tables_dir,
                                 sort_col, sort_asc, xlsx_name)


def _write_question_workbook(
    df: pd.DataFrame,
    winner_row,
    q_key: str,
    tables_dir: Path,
    sort_col: str,
    sort_asc: bool,
    xlsx_name: str,
) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return

    if df.empty and winner_row is None:
        return

    ranked = (df.sort_values(sort_col, ascending=sort_asc)
              if (not df.empty and sort_col in df.columns)
              else df)
    top10 = ranked.head(10)

    nc_col = "non_edge_candidate"
    paper_cands = (
        ranked[ranked[nc_col]].head(10)
        if nc_col in ranked.columns
        else top10.copy()
    )

    winner_df = pd.DataFrame()
    if winner_row is not None:
        wd = winner_row.to_dict()
        wd["question_key"]  = q_key
        wd["winner_reason"] = _build_winner_reason(q_key, winner_row)
        winner_df = pd.DataFrame([wd])

    fname = tables_dir / xlsx_name
    with pd.ExcelWriter(fname, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="ALL_RUNS", index=False)
        ranked.to_excel(writer, sheet_name="RANKED", index=False)
        winner_df.to_excel(writer, sheet_name="WINNER", index=False)
        top10.to_excel(writer, sheet_name="TOP_10", index=False)
        paper_cands.to_excel(writer, sheet_name="PAPER_CANDIDATES", index=False)


# =============================================================================
# Edge / artifact summary tables
# =============================================================================

def export_edge_tables(df_all: pd.DataFrame, out_root: Path) -> None:
    """
    Write two supplementary tables to 13_master_tables_for_paper/:
      top_non_edge_candidates.xlsx — all non-edge rows, sorted by composite_score
      edge_dominated_runs.xlsx     — all edge_flag=True rows
    """
    tables_dir = out_root / "13_master_tables_for_paper"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if "edge_flag" not in df_all.columns:
        return

    non_edge = df_all[~df_all["edge_flag"].fillna(False)].copy()
    if "composite_score" in non_edge.columns:
        non_edge = non_edge.sort_values("composite_score", ascending=False)
    _try_excel(non_edge, tables_dir / "top_non_edge_candidates.xlsx")
    non_edge.to_csv(tables_dir / "top_non_edge_candidates.csv", index=False)

    edge = df_all[df_all["edge_flag"].fillna(False)].copy()
    _try_excel(edge, tables_dir / "edge_dominated_runs.xlsx")
    edge.to_csv(tables_dir / "edge_dominated_runs.csv", index=False)


# =============================================================================
# Supplement figure bundles
# =============================================================================

_SUPPLEMENT_SPEC = [
    # (q_key, folder_name, sort_col, sort_asc)
    ("Q2_alignment",   "question2_top10_alignment_runs",             "frac_box3_inside_dm1_95ellipse", False),
    ("Q3A_clean",      "question3A_top10_clean_separation_runs",     "cln_bhattacharyya_dm1_vs_ctrl",  False),
    ("Q3B_holistic",   "question3B_top10_holistic_separation_runs",  "hol_bhattacharyya_dm1_vs_ctrl",  False),
    ("Q4_shift",       "question4_top10_geometry_shift_runs",        "frac_box3_inside_dm1_95ellipse", False),
    ("Q5_gradient",    "question5_top10_gradient_runs",              "grad_score",                     False),
    ("Q6_exploratory", "question6_top_runs",                         "frac_box3_inside_dm1_95ellipse", True),
]

_BUNDLE_PATH_COLS = [
    "scatter_plot_path",
    "si_scatter_plot_path",
    "enhanced_plot_path",
    "group_enh_path",
    "loadings_plot_path",
    "overlay_plot_path",
    "top_wavenumbers_csv_path",
    "grad_axis_path",
    "grad_proj_path",
    "grad_bins_path",
    "grad_panels_path",
    # legacy names as fallback
    "group_scatter_path",
    "si_scatter_path",
    "primary_enhanced_scatter",
]


def build_supplement_bundles(
    boards: Dict[str, pd.DataFrame],
    df_all: pd.DataFrame,
    out_root: Path,
    top_n: int = 10,
) -> None:
    """
    Copy the full asset set for top-N runs per question into
    12_supplement_figure_sets/{folder_name}/.
    """
    supp_root = out_root / "12_supplement_figure_sets"
    supp_root.mkdir(parents=True, exist_ok=True)

    for q_key, folder_name, sort_col, sort_asc in _SUPPLEMENT_SPEC:
        lb = boards.get(q_key)
        if lb is None or lb.empty:
            continue
        dest = supp_root / folder_name
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0

        # Join board rows back with df_all to get path columns
        merge_cols = ["spectrum_mode", "concept", "preprocessing", "plot_type"]
        avail_merge = [c for c in merge_cols if c in lb.columns and c in df_all.columns]
        if avail_merge and len(avail_merge) == len(merge_cols):
            lb_with_paths = lb.merge(
                df_all[avail_merge + [c for c in _BUNDLE_PATH_COLS if c in df_all.columns]],
                on=avail_merge, how="left", suffixes=("", "_df"),
            )
        else:
            lb_with_paths = lb

        for _, row in lb_with_paths.head(top_n).iterrows():
            for col in _BUNDLE_PATH_COLS:
                path_str = str(row.get(col, "")).strip()
                if not path_str:
                    continue
                src = Path(path_str)
                if src.exists():
                    shutil.copy2(src, dest / src.name)
                    copied += 1

        if copied:
            print(f"[SUPPLEMENT] {q_key}: copied {copied} files → {dest}")


# =============================================================================
# Journal figure index
# =============================================================================

_Q_PAPER_SECTION = {
    "Q1_raw_geometry": "Supplementary Methods",
    "Q2_alignment":    "Results — DM1 Cohort Alignment",
    "Q3A_clean":       "Results — Control Separation",
    "Q3B_holistic":    "Results — Control Separation",
    "Q4_shift":        "Results — Geometry Shift",
    "Q5_gradient":     "Results — Severity Gradient",
    "Q6_exploratory":  "Supplementary Results",
    "combined":        "Methods — Composite Score",
}

_Q_SHORT_CAPTION = {
    "Q2_alignment":  "Cohort 2 DM1 overlap with Cohort 1 DM1 in DM1-only PCA space.",
    "Q3A_clean":     "Cohort 1 DM1 vs Controls separation (clean).",
    "Q3B_holistic":  "All DM1 vs Controls separation (holistic).",
    "Q4_shift":      "Cohort 2 DM1 alignment within disease/control geometry.",
    "Q5_gradient":   "DM1 severity (SI) gradient in PCA space.",
    "Q6_exploratory":"Preprocessing-induced cohort spread (exploratory).",
    "combined":      "Best composite alignment + separation configuration.",
}


def export_journal_figure_index(
    winner_assets: Dict[str, Dict[str, str]],
    winners: Dict[str, pd.Series],
    out_root: Path,
) -> None:
    """Write JOURNAL_READY_FIGURE_INDEX.csv to 11_journal_ready_figures/."""
    jour_dir = out_root / "11_journal_ready_figures"
    jour_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    fig_id = 1
    for q_key, assets in winner_assets.items():
        winner = winners.get(q_key)
        pp   = winner.get("preprocessing", "RAW") if winner is not None else ""
        sm   = winner.get("spectrum_mode", "")    if winner is not None else ""
        pm   = winner.get("plot_type", "")        if winner is not None else ""
        cpt  = winner.get("concept", "")          if winner is not None else ""

        base = dict(
            figure_id          = f"Fig{fig_id:02d}",
            paper_section      = _Q_PAPER_SECTION.get(q_key, ""),
            question_id        = q_key,
            short_caption      = _Q_SHORT_CAPTION.get(q_key, ""),
            run_description    = _build_winner_reason(q_key, winner) if winner is not None else "",
            spectral_window    = sm,
            concept            = cpt,
            plot_mode          = pm,
            preprocessing      = pp,
            paired_overlay_path  = assets.get("overlay", ""),
            paired_loadings_path = assets.get("loadings", ""),
        )
        # One row per asset type
        for atype, apath in assets.items():
            row = base.copy()
            row["asset_type"]   = atype
            row["figure_path"]  = apath
            rows.append(row)
        fig_id += 1

    if rows:
        pd.DataFrame(rows).to_csv(jour_dir / "JOURNAL_READY_FIGURE_INDEX.csv", index=False)


# =============================================================================
# Audit logs + config
# =============================================================================

def write_audit_files(
    out_root: Path,
    run_log: List[str],
    error_log: List[str],
) -> None:
    """Write audit files to 14_logs_and_audit/."""
    audit_dir = out_root / "14_logs_and_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.datetime.now().isoformat(timespec="seconds")

    with open(audit_dir / "PCA_RUN_LOG.txt", "w", encoding="utf-8") as fh:
        fh.write(f"PCA Suite run log — generated {stamp}\n")
        fh.write(f"Total logged entries: {len(run_log)}\n")
        fh.write("=" * 70 + "\n")
        fh.write("\n".join(run_log) if run_log else "(no entries)\n")

    with open(audit_dir / "PCA_ERRORS_AND_SKIPS.txt", "w", encoding="utf-8") as fh:
        fh.write(f"PCA Suite errors/skips — generated {stamp}\n")
        fh.write(f"Total errors: {len(error_log)}\n")
        fh.write("=" * 70 + "\n")
        if error_log:
            fh.write("\n".join(error_log) + "\n")
        else:
            fh.write("(no errors or skips)\n")

    config_dict = {
        "timestamp":         stamp,
        "OUT_ROOT":          str(OUT_ROOT),
        "BOX12_DATA_DIR":    str(BOX12_DATA_DIR),
        "BOX12_META_CSV":    str(BOX12_META_CSV),
        "BOX3_DATA_DIR":     str(BOX3_DATA_DIR),
        "BOX3_META_CSV":     str(BOX3_META_CSV),
        "CONCEPTS":          CONCEPTS,
        "SPECTRUM_MODES":    [(m, list(b) if b else None) for m, b in SPECTRUM_MODES],
        "SUPPORTED_METHODS": SUPPORTED_METHODS,
        "BASELINE_FLAGS":    BASELINE_FLAGS,
        "TOP_K_WAVENUMBERS": TOP_K_WAVENUMBERS,
        "GAP_THRESH":        GAP_THRESH,
        "FIGURE_DPI":        FIGURE_DPI,
        "JOURNAL_DPI":       JOURNAL_DPI,
    }
    with open(audit_dir / "PCA_CONFIG_USED.json", "w", encoding="utf-8") as fh:
        json.dump(config_dict, fh, indent=2)

    with open(audit_dir / "PCA_QUESTION_DEFINITIONS.txt", "w", encoding="utf-8") as fh:
        fh.write("PCA Suite — Question Definitions\n")
        fh.write("=" * 70 + "\n\n")
        for q, desc in QUESTION_DEFINITIONS.items():
            fh.write(f"{q}: {desc}\n\n")
        fh.write("\nFit-space concepts\n")
        fh.write("-" * 40 + "\n")
        fh.write("WEIGHTS__BOX12_DM1_ONLY: PCA axes derived from Cohort 1 DM1 spectra only.\n"
                 "  Use for Q2 (alignment), Q5 (gradient), Q6 (exploratory).\n\n")
        fh.write("WEIGHTS__BOX12_ALL: PCA axes derived from all Cohort 1 spectra (DM1 + Controls).\n"
                 "  Use for Q3A/Q3B (separation), Q4 (geometry shift).\n\n")
        fh.write("Preprocessing combos (9 total including RAW)\n")
        fh.write("-" * 40 + "\n")
        fh.write("RAW, Normalization, SNV, Second Derivative, EMSC,\n"
                 "Normalization+EMSC, SNV+Second Derivative,\n"
                 "Normalization+SNV, Normalization+SNV+Second Derivative\n")


# =============================================================================
# README
# =============================================================================

def write_readme(out_root: Path) -> None:
    lines = [
        "PCA RESULTS — OUTPUT GUIDE",
        "=" * 60,
        "",
        "00_master_summaries/",
        "  Per-question leaderboard CSVs (alignment, separation, gradient, composite).",
        "",
        "01_question1_raw_geometry/",
        "  Q1: Native raw spectra, no preprocessing, DM1-only PCA axes.",
        "",
        "02_question2_dm1_alignment/",
        "  Q2: How well does Cohort 2 DM1 overlap Cohort 1 DM1 (DM1-only PCA)?",
        "",
        "03_question3A_clean_control_separation/",
        "  Q3A: How cleanly are Controls separated from Cohort 1 DM1 only?",
        "",
        "04_question3B_holistic_control_separation/",
        "  Q3B: Controls separation with Cohort 2 DM1 included in the disease cloud.",
        "",
        "05_question4_geometry_shift/",
        "  Q4: Cohort 2 DM1 alignment in the full disease/control geometry (ALL concept).",
        "",
        "06_question5_severity_gradient/",
        "  Q5: Preprocessing that maximises SI-severity gradient (Spearman rho).",
        "",
        "07_question6_exploratory_cohort_separation/",
        "  Q6: Supplementary — preprocessing-induced cohort spread.",
        "",
        "08_winner_overlays/",
        "  Enhanced scatter (journal DPI) for each question winner.",
        "  Standardized name: {Q}__BEST_{LABEL}__{WINDOW}__{CONCEPT}__{MODE}__{PP}__enhanced.png",
        "",
        "09_winner_loadings/",
        "  Loadings plots + top_wavenumbers CSVs for each question winner.",
        "",
        "10_winner_scatterplots/",
        "  Group-colored and SI-colored scatter for each question winner.",
        "",
        "11_journal_ready_figures/",
        "  Two-panel journal figures (scatter + overlay) per winner.",
        "  JOURNAL_READY_FIGURE_INDEX.csv — full asset index with captions.",
        "",
        "12_supplement_figure_sets/",
        "  question{N}_top10_{label}/ — full asset sets for top-10 runs per question.",
        "  Assets: group scatter, SI scatter, enhanced scatter, loadings, overlay, top_wn CSV.",
        "",
        "13_master_tables_for_paper/",
        "  PCA_MASTER_ALL_RUNS.csv/.xlsx  — all runs, all metrics.",
        "  PCA_MASTER_WINNERS.csv/.xlsx   — one row per winner with reason + asset paths.",
        "  question{N}_{label}_winners.xlsx — per-question workbooks.",
        "    Sheets: ALL_RUNS | RANKED | WINNER | TOP_10 | PAPER_CANDIDATES",
        "  top_non_edge_candidates.xlsx/.csv — runs with non-edge-dominated loadings.",
        "  edge_dominated_runs.xlsx/.csv     — runs flagged as edge-artefact dominated.",
        "",
        "14_logs_and_audit/",
        "  PCA_RUN_LOG.txt             — timestamped entry per completed run.",
        "  PCA_ERRORS_AND_SKIPS.txt    — any preprocessing failures.",
        "  PCA_CONFIG_USED.json        — full config at run time.",
        "  PCA_QUESTION_DEFINITIONS.txt — plain-text question descriptions.",
        "",
        "KEY METRICS",
        "  frac_box3_inside_dm1_95ellipse  fraction of Cohort 2 DM1 patients inside Cohort 1 DM1 95% ellipse",
        "  bhattacharyya_dm1_vs_ctrl       higher = better Controls vs DM1 separation (Bhattacharyya distance)",
        "  grad_score                      max(|Spearman rho(PC1,SI)|, |Spearman rho(PC2,SI)|)",
        "  composite_score                 50% alignment + 50% separation (sweep tool only)",
        "  edge_flag                       True if >50% of top-k loading wavenumbers in outer 5% of range",
        "  non_edge_candidate              not edge_flag — recommended filter for paper candidates",
        "",
        "HOW TO RE-RUN",
        "  python ultimate_pca_suite.py",
    ]
    with open(out_root / "README__output_guide.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# =============================================================================
# Helpers
# =============================================================================

def _try_excel(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_excel(path, index=False)
    except Exception:
        pass
