#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_runner_core.py
==================
PCA fitting engine + sweep orchestration.
Contains:
  - fit_pca_and_project   — low-level PCA fit/project
  - build_fit_matrix      — concept-dependent matrix selection
  - _collect_metrics_for_mode — all metrics from a single PCA result
  - run_question1_raw_geometry — Q1 baseline (no preprocessing)
  - _get_question_dirs    — output folder layout
  - _run_one_sweep_config — one (mode, concept, preprocessing) combo
  - run_master_sweep      — full outer loop; returns (DataFrame, run_log, error_log)
  - _rerun_winner         — re-run preprocessing+PCA for a winner row
"""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pca_config import (
    BASELINE_FLAGS, CONCEPTS, FIGURE_DPI, GAP_THRESH,
    JOURNAL_DPI, SPECTRUM_MODES, SUPPORTED_METHODS, TOP_K_WAVENUMBERS,
)
from pca_metrics import (
    compute_alignment_metrics,
    compute_gradient_metrics,
    compute_separation_metrics,
)
from pca_plotting import (
    plot_enhanced_pca_scatter,
    plot_gradient_direction,
    plot_group_scatter,
    plot_si_scatter,
    save_loading_and_overlay,
)
from pca_preprocessing import apply_pipeline_train_test
from pca_utils import (
    _run_name_to_methods,
    build_all_ordered_combos,
    compute_edge_fraction,
    get_dm1_control_masks,
    safe_name,
    sanitize_filename,
    trim_region,
)

# Concept → human labels used in master table
_CONCEPT_LABEL = {
    "WEIGHTS__BOX12_DM1_ONLY": "DM1-only PCA",
    "WEIGHTS__BOX12_ALL":       "Full-cohort PCA",
}
_CONCEPT_PRIMARY_Q = {
    "WEIGHTS__BOX12_DM1_ONLY": "Q2",
    "WEIGHTS__BOX12_ALL":      "Q3A",
}
_CONCEPT_RELATED_QS = {
    "WEIGHTS__BOX12_DM1_ONLY": "Q2, Q5, Q6",
    "WEIGHTS__BOX12_ALL":      "Q3A, Q3B, Q4",
}

# Short aliases for folder path components
_WINDOW_SHORT = {
    "FULL_SPECTRUM": "FULL",
    "TRIM_500_3100": "T500",
    "TRIM_700_2800": "T700",
}
_CONCEPT_SHORT = {
    "WEIGHTS__BOX12_DM1_ONLY": "DM1_ONLY",
    "WEIGHTS__BOX12_ALL":      "ALL",
}


def _short_window(mode_name: str) -> str:
    return _WINDOW_SHORT.get(mode_name, mode_name)


def _short_concept(concept: str) -> str:
    return _CONCEPT_SHORT.get(concept, concept)


# =============================================================================
# SECTION 6 — PCA FIT + PROJECT
# =============================================================================

def fit_pca_and_project(
    X_fit_matrix: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[PCA, np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Fit PCA on X_fit_matrix.T, project X_train and X_test.
    All matrices have shape (features, N).
    Returns (pca, scores_train, scores_test, (pc1_var, pc2_var)).
    """
    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_fit_matrix.T)
    scores_train = pca.transform(X_train.T)
    scores_test  = pca.transform(X_test.T)
    evr = pca.explained_variance_ratio_
    return pca, scores_train, scores_test, (float(evr[0]), float(evr[1]))


def build_fit_matrix(
    concept: str,
    X_avg_or_all_train: np.ndarray,
    dm1_spec_mask: np.ndarray,
) -> np.ndarray:
    """
    Select PCA fitting matrix by concept:
      WEIGHTS__BOX12_DM1_ONLY → DM1 spectra only
      WEIGHTS__BOX12_ALL      → all training spectra
    """
    if concept == "WEIGHTS__BOX12_DM1_ONLY":
        return X_avg_or_all_train[:, dm1_spec_mask]
    elif concept == "WEIGHTS__BOX12_ALL":
        return X_avg_or_all_train
    else:
        raise ValueError(f"Unknown concept: {concept}")


# =============================================================================
# SECTION 7 — PER-RUN METRIC COLLECTOR
# =============================================================================

def _collect_metrics_for_mode(
    pca: PCA,
    scores_12: np.ndarray,
    scores_3: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    si_col: str,
    dx_col: str,
    dm1_mask_train: np.ndarray,
    ctrl_mask_train: np.ndarray,
) -> Dict:
    """
    Compute ALL metrics for one run. Works for both PATIENT_AVG and ALL_SPECTRA —
    caller supplies the correctly-shaped masks and repeated metadata.
    """
    dm1_3, _ = get_dm1_control_masks(meta_test, dx_col, require_controls=False)

    s_dm1_12  = scores_12[dm1_mask_train]
    s_ctrl_12 = scores_12[ctrl_mask_train]
    s_dm1_3   = scores_3[dm1_3]

    si_12 = pd.to_numeric(meta_train[si_col], errors="coerce").to_numpy(float)

    grad    = compute_gradient_metrics(s_dm1_12, si_12[dm1_mask_train])
    align   = compute_alignment_metrics(s_dm1_3, s_dm1_12)
    sep_hol = compute_separation_metrics(s_dm1_12, s_dm1_3, s_ctrl_12, include_box3_in_dm1=True)
    sep_cln = compute_separation_metrics(s_dm1_12, s_dm1_3, s_ctrl_12, include_box3_in_dm1=False)

    evr = pca.explained_variance_ratio_
    return dict(
        pc1_var     = float(evr[0]),
        pc2_var     = float(evr[1]),
        n_box12_dm1 = int(dm1_mask_train.sum()),
        n_box3_dm1  = int(dm1_3.sum()),
        n_controls  = int(ctrl_mask_train.sum()),
        **grad,
        **align,
        **{f"hol_{k}": v for k, v in sep_hol.items()},
        **{f"cln_{k}": v for k, v in sep_cln.items()},
    )


# =============================================================================
# SECTION 7b — COMPLETE ARTIFACT SAVER
# =============================================================================

def _save_artifacts(
    out_dir: Path,
    base_title: str,
    pmode: str,
    sc12: np.ndarray,
    sc3: np.ndarray,
    meta_tr: pd.DataFrame,
    meta_te: pd.DataFrame,
    si_col: str,
    dx_col: str,
    evr: Tuple[float, float],
    align: Dict,
    sep_hol: Dict,
    sep_cln: Dict,
    m: Dict,
    wn_use: np.ndarray,
    pca: PCA,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    concept: str,
    run_name: str,
    mode_name: str,
    q_id: str,
) -> Tuple[Path, Path, Path, Path, Path, Path, Path]:
    """
    Save ALL 7 artifact files for one (question, run, mode) into out_dir.
    Returns (group_path, si_path, enh_path, load_path, overlay_path, csv_path, grad_dir_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gp = plot_group_scatter(
        out_dir, base_title, sc12, sc3, meta_tr, meta_te, dx_col, evr,
        fname_stem=pmode)
    sp = plot_si_scatter(
        out_dir, base_title, sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
        fname_stem=pmode)
    ep = plot_enhanced_pca_scatter(
        out_dir, base_title, sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
        align, sep_hol, fname_stem=pmode)
    _, _, lp, op, cp = save_loading_and_overlay(
        out_dir, base_title, wn_use, pca, X_tr, X_te,
        top_k=TOP_K_WAVENUMBERS, gap_thresh=GAP_THRESH, fname_stem=pmode)
    grad_keys = ("rho_pc1", "rho_pc2", "grad_score")
    grad_m = {k: m[k] for k in grad_keys if k in m}
    gdp = plot_gradient_direction(
        out_dir, base_title, sc12, meta_tr, si_col, dx_col, evr,
        grad_m, fname_stem=pmode)
    _save_metrics_json(
        out_dir, base_title, m, align, sep_hol, sep_cln,
        extra_meta=dict(
            concept=concept, preprocessing=run_name,
            spectrum_mode=mode_name, spectral_window=mode_name,
            plot_type=pmode, question_id=q_id,
            primary_question_id=_CONCEPT_PRIMARY_Q.get(concept, ""),
            related_questions=_CONCEPT_RELATED_QS.get(concept, ""),
        ),
        fname_stem=pmode,
    )
    return gp, sp, ep, lp, op, cp, gdp


# =============================================================================
# SECTION 8 — Q1: RAW GEOMETRY
# =============================================================================

def run_question1_raw_geometry(
    q1_dir: Path,
    wavenumbers: np.ndarray,
    X_avg_train: np.ndarray, X_avg_test: np.ndarray,
    X_all_train: np.ndarray, X_all_test: np.ndarray,
    meta_train: pd.DataFrame, meta_test: pd.DataFrame,
    si_col: str, dx_col: str,
    dm1_mask_train: np.ndarray, ctrl_mask_train: np.ndarray,
) -> List[Dict]:
    """
    Q1: Native raw spectral geometry — no preprocessing.
    Fits PCA on Cohort 1 DM1 only (most conservative baseline).
    """
    rows = []
    concept = "WEIGHTS__BOX12_DM1_ONLY"

    for mode_name, trim_bounds in SPECTRUM_MODES:
        sw = _short_window(mode_name)
        if trim_bounds is None:
            wn, Xat, Xae, Xlt, Xle = (wavenumbers,
                X_avg_train, X_avg_test, X_all_train, X_all_test)
        else:
            lo, hi = trim_bounds
            wn,  Xat = trim_region(wavenumbers, X_avg_train, lo, hi)
            _,   Xae = trim_region(wavenumbers, X_avg_test,  lo, hi)
            _,   Xlt = trim_region(wavenumbers, X_all_train, lo, hi)
            _,   Xle = trim_region(wavenumbers, X_all_test,  lo, hi)

        for pmode, X_tr, X_te, dm1_mask, ctrl_mask, meta_tr, meta_te in [
            ("PATIENT_AVG",  Xat, Xae, dm1_mask_train, ctrl_mask_train,
             meta_train, meta_test),
            ("ALL_SPECTRA",  Xlt, Xle,
             np.repeat(dm1_mask_train, 9), np.repeat(ctrl_mask_train, 9),
             meta_train.loc[np.repeat(np.arange(len(meta_train)), 9)].reset_index(drop=True),
             meta_test.loc[np.repeat(np.arange(len(meta_test)),   9)].reset_index(drop=True)),
        ]:
            out_dir = q1_dir / sw / pmode
            out_dir.mkdir(parents=True, exist_ok=True)

            X_fit = build_fit_matrix(concept, X_tr, dm1_mask)
            pca, sc12, sc3, evr = fit_pca_and_project(X_fit, X_tr, X_te)

            base_title = f"Q1 RAW {mode_name} {pmode}"

            m = _collect_metrics_for_mode(pca, sc12, sc3, meta_tr, meta_te,
                                          si_col, dx_col, dm1_mask, ctrl_mask)

            align   = {k: v for k, v in m.items() if k.startswith("frac_") or k.startswith("mahal") or k.startswith("centroid") or k.startswith("mean_nn")}
            sep_hol = {k[4:]: v for k, v in m.items() if k.startswith("hol_")}
            sep_cln = {k[4:]: v for k, v in m.items() if k.startswith("cln_")}

            gp, sp, ep, lp, op, cp, gdp = _save_artifacts(
                out_dir, base_title, pmode,
                sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
                align, sep_hol, sep_cln, m,
                wn, pca, X_tr, X_te,
                concept, "RAW", mode_name, "Q1",
            )

            pc1_top = _top_wn_from_pca(wn, pca, 0, TOP_K_WAVENUMBERS)
            pc2_top = _top_wn_from_pca(wn, pca, 1, TOP_K_WAVENUMBERS)
            pc1_ef  = compute_edge_fraction(wn, pc1_top)
            pc2_ef  = compute_edge_fraction(wn, pc2_top)

            rows.append(dict(
                question_id="Q1", question_name="Raw Geometry",
                spectrum_mode=mode_name, spectral_window=mode_name,
                concept=concept, pca_concept=concept,
                concept_label=_CONCEPT_LABEL.get(concept, concept),
                primary_question_id=_CONCEPT_PRIMARY_Q.get(concept, ""),
                related_questions=_CONCEPT_RELATED_QS.get(concept, ""),
                preprocessing="RAW", baseline=False, plot_type=pmode,
                winner_eligible=True,
                **m,
                pc1_edge_fraction=pc1_ef,
                pc2_edge_fraction=pc2_ef,
                edge_flag=bool(pc1_ef > 0.5 or pc2_ef > 0.5),
                non_edge_candidate=not bool(pc1_ef > 0.5 or pc2_ef > 0.5),
                group_scatter_path=str(gp),
                si_scatter_path=str(sp),
                primary_enhanced_scatter=str(ep),
                loadings_plot_path=str(lp),
                overlay_plot_path=str(op),
                top_wavenumbers_csv_path=str(cp),
                grad_dir_path=str(gdp),
            ))
            print(f"[Q1] {mode_name} | {pmode} done")

    return rows


# =============================================================================
# SECTION 9 — MASTER SWEEP  (Q2–Q6)
# =============================================================================

def _get_question_dirs(out_root: Path) -> Dict[str, Path]:
    dirs = {
        "q2":  out_root / "02_q2_align",
        "q3a": out_root / "03_q3a_sep",
        "q3b": out_root / "04_q3b_sep",
        "q4":  out_root / "05_q4_geom",
        "q5":  out_root / "06_q5_grad",
        "q6":  out_root / "07_q6_explore",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _run_one_sweep_config(
    qdirs: Dict[str, Path],
    mode_name: str,
    concept: str,
    run_name: str,
    wn_use: np.ndarray,
    Xat_pp: np.ndarray,
    Xae_pp: np.ndarray,
    Xlt_pp: np.ndarray,
    Xle_pp: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    si_col: str,
    dx_col: str,
    dm1_mask_train: np.ndarray,
    ctrl_mask_train: np.ndarray,
) -> List[Dict]:
    """
    Run one (mode, concept, preprocessing) combo for both PATIENT_AVG and
    ALL_SPECTRA plot modes. Returns two metric rows with full path fields.
    Every question folder receives the full artifact set.
    """
    sw  = _short_window(mode_name)
    sc  = _short_concept(concept)

    n_pat_tr = len(meta_train)
    n_pat_te = len(meta_test)
    dm1_spec_tr  = np.repeat(dm1_mask_train,  9)
    ctrl_spec_tr = np.repeat(ctrl_mask_train, 9)
    meta_tr_spec = meta_train.loc[np.repeat(np.arange(n_pat_tr), 9)].reset_index(drop=True)
    meta_te_spec = meta_test.loc[np.repeat(np.arange(n_pat_te),  9)].reset_index(drop=True)

    rows = []

    for pmode, X_tr, X_te, dm1_msk, ctrl_msk, meta_tr, meta_te in [
        ("PATIENT_AVG",
         Xat_pp, Xae_pp, dm1_mask_train, ctrl_mask_train, meta_train, meta_test),
        ("ALL_SPECTRA",
         Xlt_pp, Xle_pp, dm1_spec_tr,   ctrl_spec_tr,    meta_tr_spec, meta_te_spec),
    ]:
        X_fit = build_fit_matrix(concept, X_tr, dm1_msk)
        pca, sc12, sc3, evr = fit_pca_and_project(X_fit, X_tr, X_te)

        m = _collect_metrics_for_mode(pca, sc12, sc3, meta_tr, meta_te,
                                      si_col, dx_col, dm1_msk, ctrl_msk)

        dm1_3, _ = get_dm1_control_masks(meta_te, dx_col, require_controls=False)
        s_dm1_12  = sc12[dm1_msk]
        s_ctrl_12 = sc12[ctrl_msk]
        s_dm1_3   = sc3[dm1_3]

        align   = compute_alignment_metrics(s_dm1_3, s_dm1_12)
        sep_hol = compute_separation_metrics(s_dm1_12, s_dm1_3, s_ctrl_12, include_box3_in_dm1=True)
        sep_cln = compute_separation_metrics(s_dm1_12, s_dm1_3, s_ctrl_12, include_box3_in_dm1=False)

        base_title = f"{run_name} {mode_name} {pmode}"

        pc1_top   = _top_wn_from_pca(wn_use, pca, 0, TOP_K_WAVENUMBERS)
        pc2_top   = _top_wn_from_pca(wn_use, pca, 1, TOP_K_WAVENUMBERS)
        pc1_ef    = compute_edge_fraction(wn_use, pc1_top)
        pc2_ef    = compute_edge_fraction(wn_use, pc2_top)
        edge_flag = bool(pc1_ef > 0.5 or pc2_ef > 0.5)

        # --- Save ALL artifacts per question ---
        if concept == "WEIGHTS__BOX12_DM1_ONLY":
            q2_out = qdirs["q2"] / sw / sc / pmode / run_name
            gp, sp, ep, lp, op, cp, gdp = _save_artifacts(
                q2_out, base_title, pmode,
                sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
                align, sep_hol, sep_cln, m, wn_use, pca, X_tr, X_te,
                concept, run_name, mode_name, "Q2")

            q5_out = qdirs["q5"] / sw / sc / pmode / run_name
            _save_artifacts(
                q5_out, base_title, pmode,
                sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
                align, sep_hol, sep_cln, m, wn_use, pca, X_tr, X_te,
                concept, run_name, mode_name, "Q5")

            q6_out = qdirs["q6"] / sw / sc / pmode / run_name
            _save_artifacts(
                q6_out, base_title, pmode,
                sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
                align, sep_hol, sep_cln, m, wn_use, pca, X_tr, X_te,
                concept, run_name, mode_name, "Q6")

        elif concept == "WEIGHTS__BOX12_ALL":
            q3a_out = qdirs["q3a"] / sw / sc / pmode / run_name
            gp, sp, ep, lp, op, cp, gdp = _save_artifacts(
                q3a_out, base_title, pmode,
                sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
                align, sep_hol, sep_cln, m, wn_use, pca, X_tr, X_te,
                concept, run_name, mode_name, "Q3A")

            q3b_out = qdirs["q3b"] / sw / sc / pmode / run_name
            _save_artifacts(
                q3b_out, base_title, pmode,
                sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
                align, sep_hol, sep_cln, m, wn_use, pca, X_tr, X_te,
                concept, run_name, mode_name, "Q3B")

            q4_out = qdirs["q4"] / sw / sc / pmode / run_name
            _save_artifacts(
                q4_out, base_title, pmode,
                sc12, sc3, meta_tr, meta_te, si_col, dx_col, evr,
                align, sep_hol, sep_cln, m, wn_use, pca, X_tr, X_te,
                concept, run_name, mode_name, "Q4")

        row = dict(
            question_id          = _CONCEPT_PRIMARY_Q.get(concept, ""),
            question_name        = {
                "WEIGHTS__BOX12_DM1_ONLY": "DM1 Alignment / Gradient",
                "WEIGHTS__BOX12_ALL":      "Control Separation / Geometry Shift",
            }.get(concept, ""),
            spectrum_mode        = mode_name,
            spectral_window      = mode_name,
            concept              = concept,
            pca_concept          = concept,
            concept_label        = _CONCEPT_LABEL.get(concept, concept),
            primary_question_id  = _CONCEPT_PRIMARY_Q.get(concept, ""),
            related_questions    = _CONCEPT_RELATED_QS.get(concept, ""),
            preprocessing        = run_name,
            baseline             = False,
            plot_type            = pmode,
            winner_eligible      = True,
            **m,
            pc1_edge_fraction    = pc1_ef,
            pc2_edge_fraction    = pc2_ef,
            edge_flag            = edge_flag,
            non_edge_candidate   = not edge_flag,
            group_scatter_path   = str(gp),
            si_scatter_path      = str(sp),
            primary_enhanced_scatter = str(ep),
            loadings_plot_path   = str(lp),
            overlay_plot_path    = str(op),
            top_wavenumbers_csv_path = str(cp),
            grad_dir_path        = str(gdp),
        )
        rows.append(row)

    return rows


def _top_wn_from_pca(
    wavenumbers: np.ndarray, pca: PCA, pc_idx: int, top_k: int
) -> np.ndarray:
    vec = pca.components_[pc_idx]
    idx = np.argsort(np.abs(vec))[::-1][:top_k]
    return wavenumbers[idx]


def _save_metrics_json(
    out_dir: Path,
    run_name: str,
    metrics: Dict,
    align: Dict,
    sep_hol: Dict,
    sep_cln: Dict,
    extra_meta: Optional[Dict] = None,
    fname_stem: Optional[str] = None,
) -> Path:
    data = {
        "run_name": run_name,
        **(extra_meta or {}),
        **metrics,
        **{f"align_{k}": v for k, v in align.items()},
        **{f"hol_{k}": v   for k, v in sep_hol.items()},
        **{f"cln_{k}": v   for k, v in sep_cln.items()},
    }
    clean = {}
    for k, v in data.items():
        if isinstance(v, (np.floating, float)):
            clean[k] = None if not np.isfinite(float(v)) else float(v)
        elif isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            clean[k] = bool(v)
        else:
            clean[k] = v
    _stem = sanitize_filename(fname_stem) if fname_stem else sanitize_filename(run_name)
    fpath = out_dir / f"{_stem}_metrics.json"
    with open(fpath, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2)
    return fpath


def run_master_sweep(
    out_root: Path,
    wavenumbers_full: np.ndarray,
    X_avg_train: np.ndarray, X_avg_test: np.ndarray,
    X_all_train: np.ndarray, X_all_test: np.ndarray,
    meta_train: pd.DataFrame, meta_test: pd.DataFrame,
    si_col: str, dx_col: str,
    dm1_mask_train: np.ndarray, ctrl_mask_train: np.ndarray,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Outer sweep: spectral windows × concepts × preprocessing combos.
    Returns (DataFrame, run_log_lines, error_log_lines).
    """
    qdirs      = _get_question_dirs(out_root)
    combos     = build_all_ordered_combos(SUPPORTED_METHODS)
    all_rows:  List[Dict]  = []
    run_log:   List[str]   = []
    error_log: List[str]   = []

    total = len(SPECTRUM_MODES) * len(CONCEPTS) * len(BASELINE_FLAGS) * len(combos)
    done  = 0

    for mode_name, trim_bounds in SPECTRUM_MODES:
        if trim_bounds is None:
            wn_use = wavenumbers_full
            Xat, Xae = X_avg_train, X_avg_test
            Xlt, Xle = X_all_train, X_all_test
        else:
            lo, hi = trim_bounds
            wn_use, Xat = trim_region(wavenumbers_full, X_avg_train, lo, hi)
            _,      Xae = trim_region(wavenumbers_full, X_avg_test,  lo, hi)
            _,      Xlt = trim_region(wavenumbers_full, X_all_train, lo, hi)
            _,      Xle = trim_region(wavenumbers_full, X_all_test,  lo, hi)

        for concept in CONCEPTS:
            for do_baseline in BASELINE_FLAGS:
                for methods in combos:
                    run_name = safe_name(methods, do_baseline)
                    stamp    = datetime.datetime.now().isoformat(timespec="seconds")
                    try:
                        Xat_pp, Xae_pp = apply_pipeline_train_test(Xat, Xae, methods, do_baseline)
                        Xlt_pp, Xle_pp = apply_pipeline_train_test(Xlt, Xle, methods, do_baseline)
                    except Exception as exc:
                        msg = (f"[{stamp}] SKIP {mode_name}|{concept}|{run_name}: {exc}")
                        print(msg)
                        error_log.append(msg)
                        done += 2
                        continue

                    rows = _run_one_sweep_config(
                        qdirs, mode_name, concept, run_name, wn_use,
                        Xat_pp, Xae_pp, Xlt_pp, Xle_pp,
                        meta_train, meta_test, si_col, dx_col,
                        dm1_mask_train, ctrl_mask_train,
                    )
                    for row in rows:
                        row["baseline"] = do_baseline
                    all_rows.extend(rows)
                    done += 2
                    msg = (f"[{stamp}] OK  [{done}/{total*2}] "
                           f"{mode_name} | {concept} | {run_name}")
                    print(msg)
                    run_log.append(msg)

    return pd.DataFrame(all_rows), run_log, error_log


# =============================================================================
# Re-run winner  (called from pca_outputs.generate_winner_outputs)
# =============================================================================

def _rerun_winner(
    winner: pd.Series,
    wavenumbers_full: np.ndarray,
    X_avg_train: np.ndarray, X_avg_test: np.ndarray,
    X_all_train: np.ndarray, X_all_test: np.ndarray,
    meta_train: pd.DataFrame, meta_test: pd.DataFrame,
    si_col: str, dx_col: str,
    dm1_mask_train: np.ndarray, ctrl_mask_train: np.ndarray,
) -> Tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Re-run preprocessing + PCA for a winner row.
    Returns (pca, sc12, sc3, wn_use, X_overlay_train, X_overlay_test, evr).
    ALL_SPECTRA uses its own preprocessed matrices for the overlay.
    """
    mode_name   = winner["spectrum_mode"]
    concept     = winner["concept"]
    run_name    = winner["preprocessing"]
    plot_type   = winner["plot_type"]

    methods     = _run_name_to_methods(run_name)
    do_baseline = run_name.startswith("BASELINE__")

    trim_bounds = {m: b for m, b in SPECTRUM_MODES}.get(mode_name)
    if trim_bounds is not None:
        lo, hi = trim_bounds
        wn_use, Xat = trim_region(wavenumbers_full, X_avg_train, lo, hi)
        _,      Xae = trim_region(wavenumbers_full, X_avg_test,  lo, hi)
        _,      Xlt = trim_region(wavenumbers_full, X_all_train, lo, hi)
        _,      Xle = trim_region(wavenumbers_full, X_all_test,  lo, hi)
    else:
        wn_use = wavenumbers_full
        Xat, Xae = X_avg_train, X_avg_test
        Xlt, Xle = X_all_train, X_all_test

    if plot_type == "PATIENT_AVG":
        X_tr, X_te = apply_pipeline_train_test(Xat, Xae, methods, do_baseline)
        dm1_msk, ctrl_msk = dm1_mask_train, ctrl_mask_train
    else:
        X_tr, X_te = apply_pipeline_train_test(Xlt, Xle, methods, do_baseline)
        dm1_msk  = np.repeat(dm1_mask_train, 9)
        ctrl_msk = np.repeat(ctrl_mask_train, 9)

    X_fit = build_fit_matrix(concept, X_tr, dm1_msk)
    pca, sc12, sc3, evr = fit_pca_and_project(X_fit, X_tr, X_te)
    return pca, sc12, sc3, wn_use, X_tr, X_te, evr
