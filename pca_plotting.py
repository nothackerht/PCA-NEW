#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_plotting.py
===============
All visualization functions for the SERS DM1 PCA suite.
Every function that saves a figure returns the saved Path object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA

from pca_config import (
    CHI2_95, FIGURE_DPI, GAP_THRESH, JOURNAL_DPI,
    GCOLOR_BOX12_DM1, GCOLOR_BOX3_DM1, GCOLOR_CTRL,
    MARKER_BOX12, MARKER_BOX3, SIZE_BOX12, SIZE_BOX3,
    TOP_K_WAVENUMBERS,
)
from pca_utils import get_dm1_control_masks, sanitize_filename


# =============================================================================
# Low-level plot helpers
# =============================================================================

def break_by_gap(x: np.ndarray, gap_thresh: float) -> List[np.ndarray]:
    """Split a wavenumber array into continuous segments (gap-aware line plots)."""
    dx = np.diff(x)
    breaks = np.where(dx > gap_thresh)[0]
    if breaks.size == 0:
        return [np.arange(len(x))]
    segments = []
    start = 0
    for b in breaks:
        segments.append(np.arange(start, b + 1))
        start = b + 1
    segments.append(np.arange(start, len(x)))
    return segments


def _draw_confidence_ellipse(
    ax, points_2d: np.ndarray, confidence: float = 0.95, **kwargs
) -> bool:
    if len(points_2d) < 3:
        return False
    mean = points_2d.mean(axis=0)
    cov  = np.cov(points_2d.T)
    if not np.all(np.isfinite(cov)) or cov.ndim < 2:
        return False
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return False
    eigenvalues = np.maximum(eigenvalues, 0.0)
    chi2_val    = chi2.ppf(confidence, df=2)
    v_major = eigenvectors[:, -1]
    angle   = np.degrees(np.arctan2(v_major[1], v_major[0]))
    width   = 2.0 * np.sqrt(chi2_val * eigenvalues[-1])
    height  = 2.0 * np.sqrt(chi2_val * eigenvalues[0])
    ax.add_patch(Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs))
    return True


def _fmt(val: float, fmt: str = ".2f") -> str:
    return f"{val:{fmt}}" if (val is not None and np.isfinite(val)) else "N/A"


def _pct(val: float) -> str:
    return f"{val * 100:.0f}%" if (val is not None and np.isfinite(val)) else "N/A"


def _common_si_range(
    si_12: np.ndarray, si_3: np.ndarray,
    dm1_12: np.ndarray, dm1_3: np.ndarray,
) -> Tuple[float, float]:
    all_si = np.concatenate([si_12[dm1_12], si_3[dm1_3]])
    all_si = all_si[np.isfinite(all_si)]
    vmin = all_si.min() if len(all_si) > 0 else 0.0
    vmax = all_si.max() if len(all_si) > 0 else 1.0
    return vmin, vmax


# =============================================================================
# Per-run scatter plots
# =============================================================================

def plot_group_scatter(
    out_dir: Path,
    title: str,
    scores_box12: np.ndarray,
    scores_box3: np.ndarray,
    meta_box12: pd.DataFrame,
    meta_box3: pd.DataFrame,
    dx_col: str,
    evr: Tuple[float, float],
    fname_stem: Optional[str] = None,
) -> Path:
    """Group-colored PCA scatter. Returns saved file path."""
    pc1_var, pc2_var = evr
    dm1_12, ctrl_12 = get_dm1_control_masks(meta_box12, dx_col, require_controls=True)
    dm1_3,  _       = get_dm1_control_masks(meta_box3,  dx_col, require_controls=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scores_box12[ctrl_12, 0], scores_box12[ctrl_12, 1],
               c=GCOLOR_CTRL, marker=MARKER_BOX12, s=SIZE_BOX12, alpha=0.85,
               label="Controls", edgecolors="none", zorder=3)
    ax.scatter(scores_box12[dm1_12, 0], scores_box12[dm1_12, 1],
               c=GCOLOR_BOX12_DM1, marker=MARKER_BOX12, s=SIZE_BOX12, alpha=0.85,
               label="Cohort 1 DM1", edgecolors="none", zorder=3)
    ax.scatter(scores_box3[dm1_3, 0], scores_box3[dm1_3, 1],
               c=GCOLOR_BOX3_DM1, marker=MARKER_BOX3, s=SIZE_BOX3, alpha=0.95,
               label="Cohort 2 DM1", edgecolors="k", linewidths=0.3, zorder=4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"PC1 ({pc1_var*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var*100:.1f}%)")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    _stem = sanitize_filename(fname_stem) if fname_stem else sanitize_filename(title)
    fpath = out_dir / (_stem + "_group.png")
    fig.savefig(fpath, dpi=FIGURE_DPI)
    plt.close(fig)
    return fpath


def plot_si_scatter(
    out_dir: Path,
    title: str,
    scores_box12: np.ndarray,
    scores_box3: np.ndarray,
    meta_box12: pd.DataFrame,
    meta_box3: pd.DataFrame,
    si_col: str,
    dx_col: str,
    evr: Tuple[float, float],
    fname_stem: Optional[str] = None,
) -> Path:
    """SI-gradient colored PCA scatter. Returns saved file path."""
    pc1_var, pc2_var = evr
    dm1_12, ctrl_12 = get_dm1_control_masks(meta_box12, dx_col, require_controls=True)
    dm1_3,  _       = get_dm1_control_masks(meta_box3,  dx_col, require_controls=False)
    si_12 = pd.to_numeric(meta_box12[si_col], errors="coerce").to_numpy(float)
    si_3  = pd.to_numeric(meta_box3[si_col],  errors="coerce").to_numpy(float)
    vmin, vmax = _common_si_range(si_12, si_3, dm1_12, dm1_3)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scores_box12[ctrl_12, 0], scores_box12[ctrl_12, 1],
               c="orange", marker=MARKER_BOX12, s=SIZE_BOX12, alpha=0.85,
               label="Controls", edgecolors="none", zorder=3)
    sc = ax.scatter(scores_box12[dm1_12, 0], scores_box12[dm1_12, 1],
                    c=si_12[dm1_12], cmap="viridis", vmin=vmin, vmax=vmax,
                    marker=MARKER_BOX12, s=SIZE_BOX12, alpha=0.85,
                    label="Cohort 1 DM1", edgecolors="none", zorder=3)
    ax.scatter(scores_box3[dm1_3, 0], scores_box3[dm1_3, 1],
               c=si_3[dm1_3], cmap="viridis", vmin=vmin, vmax=vmax,
               marker=MARKER_BOX3, s=SIZE_BOX3, alpha=0.95,
               label="Cohort 2 DM1", edgecolors="k", linewidths=0.3, zorder=4)
    plt.colorbar(sc, ax=ax, label="SI (DM1 severity)")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"PC1 ({pc1_var*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var*100:.1f}%)")
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    _stem = sanitize_filename(fname_stem) if fname_stem else sanitize_filename(title)
    fpath = out_dir / (_stem + "_si.png")
    fig.savefig(fpath, dpi=FIGURE_DPI)
    plt.close(fig)
    return fpath


def plot_gradient_direction(
    out_dir: Path,
    title: str,
    scores_box12: np.ndarray,
    meta_box12: pd.DataFrame,
    si_col: str,
    dx_col: str,
    evr: Tuple[float, float],
    grad_metrics: Dict[str, float],
    dpi: int = FIGURE_DPI,
    fname_stem: Optional[str] = None,
) -> Path:
    """
    SI-colored Cohort 1 DM1 scatter with dominant severity-gradient direction arrow.
    Returns saved file path.
    """
    pc1_var, pc2_var = evr
    dm1_12, ctrl_12 = get_dm1_control_masks(meta_box12, dx_col, require_controls=True)
    si_12  = pd.to_numeric(meta_box12[si_col], errors="coerce").to_numpy(float)
    s_dm1  = scores_box12[dm1_12]
    si_dm1 = si_12[dm1_12]
    ok     = np.isfinite(si_dm1)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scores_box12[ctrl_12, 0], scores_box12[ctrl_12, 1],
               c="lightgrey", marker="o", s=SIZE_BOX12, alpha=0.5,
               label="Controls", edgecolors="none", zorder=2)
    vmin = float(si_dm1[ok].min()) if ok.any() else 0.0
    vmax = float(si_dm1[ok].max()) if ok.any() else 1.0
    sc = ax.scatter(s_dm1[:, 0], s_dm1[:, 1],
                    c=si_dm1, cmap="plasma", vmin=vmin, vmax=vmax,
                    marker="o", s=SIZE_BOX12, alpha=0.9,
                    label="Cohort 1 DM1 (SI)", edgecolors="none", zorder=3)
    plt.colorbar(sc, ax=ax, label="SI (DM1 severity)")

    rho1       = grad_metrics.get("rho_pc1",    np.nan)
    rho2       = grad_metrics.get("rho_pc2",    np.nan)
    grad_score = grad_metrics.get("grad_score", np.nan)

    # Arrow along the dominant axis (the same axis that defines grad_score).
    # PC1-dominant → horizontal arrow; PC2-dominant → vertical arrow.
    # Sign matches the sign of the dominant rho.
    if np.isfinite(rho1) or np.isfinite(rho2):
        pc1_abs = abs(rho1) if np.isfinite(rho1) else 0.0
        pc2_abs = abs(rho2) if np.isfinite(rho2) else 0.0
        if pc1_abs >= pc2_abs:
            sign = 1.0 if (np.isfinite(rho1) and rho1 >= 0) else -1.0
            vec = np.array([sign, 0.0])
        else:
            sign = 1.0 if (np.isfinite(rho2) and rho2 >= 0) else -1.0
            vec = np.array([0.0, sign])
        spread = max(np.ptp(s_dm1[:, 0]), np.ptp(s_dm1[:, 1]), 1e-6) * 0.40
        cx, cy = float(s_dm1[:, 0].mean()), float(s_dm1[:, 1].mean())
        ax.annotate(
            "",
            xy=(cx + vec[0] * spread, cy + vec[1] * spread),
            xytext=(cx - vec[0] * spread, cy - vec[1] * spread),
            arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.0),
            zorder=6,
        )

    dom = "PC1"
    if np.isfinite(rho1) and np.isfinite(rho2):
        dom = "PC1" if abs(rho1) >= abs(rho2) else "PC2"
    elif not np.isfinite(rho1) and np.isfinite(rho2):
        dom = "PC2"

    txt = (
        f"Dominant: {dom}\n"
        f"rho_PC1 = {_fmt(rho1, '.3f')}\n"
        f"rho_PC2 = {_fmt(rho2, '.3f')}\n"
        f"grad_score = {_fmt(grad_score, '.3f')}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"PC1 ({pc1_var*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var*100:.1f}%)")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    _stem = sanitize_filename(fname_stem) if fname_stem else sanitize_filename(title)
    fpath = out_dir / (_stem + "_grad_dir.png")
    fig.savefig(fpath, dpi=dpi)
    plt.close(fig)
    return fpath


def plot_enhanced_pca_scatter(
    out_dir: Path,
    title: str,
    scores_box12: np.ndarray,
    scores_box3: np.ndarray,
    meta_box12: pd.DataFrame,
    meta_box3: pd.DataFrame,
    si_col: str,
    dx_col: str,
    evr: Tuple[float, float],
    alignment_metrics: Dict[str, float],
    separation_metrics: Dict[str, float],
    suffix: str = "__alignment",
    dpi: int = FIGURE_DPI,
    fname_stem: Optional[str] = None,
) -> Path:
    """
    Enhanced scatter with SI gradient, 95% confidence ellipses,
    centroids, and metrics text box. Returns saved file path.
    """
    pc1_var, pc2_var = evr
    dm1_12, ctrl_12 = get_dm1_control_masks(meta_box12, dx_col, require_controls=True)
    dm1_3,  _       = get_dm1_control_masks(meta_box3,  dx_col, require_controls=False)
    si_12 = pd.to_numeric(meta_box12[si_col], errors="coerce").to_numpy(float)
    si_3  = pd.to_numeric(meta_box3[si_col],  errors="coerce").to_numpy(float)
    vmin, vmax = _common_si_range(si_12, si_3, dm1_12, dm1_3)

    s_dm1_12  = scores_box12[dm1_12]
    s_ctrl_12 = scores_box12[ctrl_12]
    s_dm1_3   = scores_box3[dm1_3]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(s_ctrl_12[:, 0], s_ctrl_12[:, 1], c="orange", marker="o", s=18,
               alpha=0.85, label="Controls", edgecolors="none", zorder=3)
    sc = ax.scatter(s_dm1_12[:, 0], s_dm1_12[:, 1],
                    c=si_12[dm1_12], cmap="viridis", vmin=vmin, vmax=vmax,
                    marker="o", s=18, alpha=0.85, label="Cohort 1 DM1",
                    edgecolors="none", zorder=3)
    ax.scatter(s_dm1_3[:, 0], s_dm1_3[:, 1],
               c=si_3[dm1_3], cmap="viridis", vmin=vmin, vmax=vmax,
               marker="s", s=50, alpha=0.95, label="Cohort 2 DM1",
               edgecolors="k", linewidths=0.5, zorder=4)
    plt.colorbar(sc, ax=ax, label="SI (DM1 severity)")

    _draw_confidence_ellipse(ax, s_dm1_12, 0.95,
                             fill=False, edgecolor="steelblue", linewidth=1.5,
                             linestyle="--", label="Cohort 1 DM1 95% ellipse", zorder=2)
    _draw_confidence_ellipse(ax, s_ctrl_12, 0.95,
                             fill=False, edgecolor="darkorange", linewidth=1.5,
                             linestyle="--", label="Controls 95% ellipse", zorder=2)

    if len(s_dm1_12) > 0 and len(s_dm1_3) > 0:
        c12 = s_dm1_12.mean(axis=0)
        c3  = s_dm1_3.mean(axis=0)
        ax.plot([c12[0], c3[0]], [c12[1], c3[1]], "k--", lw=1.0, alpha=0.5, zorder=2)
        ax.scatter(*c12, marker="D", s=90, c="steelblue", edgecolors="k",
                   linewidths=0.8, zorder=5, label="Cohort 1 DM1 centroid")
        ax.scatter(*c3,  marker="*", s=160, c="darkred", edgecolors="k",
                   linewidths=0.5, zorder=5, label="Cohort 2 DM1 centroid")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"PC1 ({pc1_var*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var*100:.1f}%)")
    ax.legend(loc="best", frameon=True, fontsize=6.5, ncol=2)

    frac_el   = alignment_metrics.get("frac_box3_inside_dm1_95ellipse",      np.nan)
    frac_hull = alignment_metrics.get("frac_box3_inside_dm1_hull",           np.nan)
    mahal_m   = alignment_metrics.get("mahalanobis_mean_box3_to_box12dm1",   np.nan)
    bd        = separation_metrics.get("bhattacharyya_dm1_vs_ctrl",          np.nan)
    sil       = separation_metrics.get("silhouette_dm1_vs_ctrl",             np.nan)
    fco       = separation_metrics.get("frac_controls_outside_dm1_95ellipse",np.nan)

    txt = (
        f"── Alignment (Cohort 2 → Cohort 1 DM1) ──\n"
        f"Cohort 2 in 95% ellipse : {_pct(frac_el)}\n"
        f"Cohort 2 in hull        : {_pct(frac_hull)}\n"
        f"Mean Mahal dist         : {_fmt(mahal_m)}\n"
        f"── Separation (DM1 vs Controls) ──────────\n"
        f"Bhattacharyya dist      : {_fmt(bd)}\n"
        f"Silhouette              : {_fmt(sil, '.3f')}\n"
        f"Controls outside ellipse: {_pct(fco)}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=7.0,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88))

    fig.tight_layout()
    _stem = sanitize_filename(fname_stem) if fname_stem else sanitize_filename(title)
    fpath = out_dir / (_stem + "_enh.png")
    fig.savefig(fpath, dpi=dpi)
    plt.close(fig)
    return fpath


# =============================================================================
# Loadings and spectral overlay
# =============================================================================

def compute_top_wavenumbers(
    wavenumbers: np.ndarray, pc_vec: np.ndarray, top_k: int
) -> np.ndarray:
    idx = np.argsort(np.abs(pc_vec))[::-1][:top_k]
    return wavenumbers[idx]


def plot_loadings(
    out_dir: Path, title: str,
    wavenumbers: np.ndarray, pc1: np.ndarray, pc2: np.ndarray,
    gap_thresh: float = GAP_THRESH,
    fname_stem: Optional[str] = None,
) -> Path:
    """Plot PC1 and PC2 loadings. Returns saved file path."""
    segs = break_by_gap(wavenumbers, gap_thresh)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for seg in segs:
        ax.plot(wavenumbers[seg], pc1[seg], label="PC1", linewidth=1.2)
        ax.plot(wavenumbers[seg], pc2[seg], label="PC2", linewidth=1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Wavenumber (cm⁻\xb9)")
    ax.set_ylabel("Loading")
    ax.legend()
    fig.tight_layout()
    _stem = sanitize_filename(fname_stem) if fname_stem else sanitize_filename(title)
    fpath = out_dir / (_stem + "_load.png")
    fig.savefig(fpath, dpi=FIGURE_DPI)
    plt.close(fig)
    return fpath


def plot_overlay(
    out_dir: Path,
    title: str,
    wavenumbers: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    wn_pc1: np.ndarray,
    wn_pc2: np.ndarray,
    gap_thresh: float = GAP_THRESH,
    dpi: int = FIGURE_DPI,
    fname_stem: Optional[str] = None,
) -> Path:
    """
    Spectra overlay with highlighted important wavenumbers.
    Blue lines = top PC1, red lines = top PC2.
    Returns saved file path.
    """
    segs = break_by_gap(wavenumbers, gap_thresh)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    for j in range(X_train.shape[1]):
        for seg in segs:
            ax.plot(wavenumbers[seg], X_train[seg, j], alpha=0.07, linewidth=0.7, color="steelblue")
    for j in range(X_test.shape[1]):
        for seg in segs:
            ax.plot(wavenumbers[seg], X_test[seg, j], alpha=0.07, linewidth=0.7, color="crimson")
    for w in wn_pc1:
        ax.axvline(float(w), color="blue",  alpha=0.18, linewidth=1.0)
    for w in wn_pc2:
        ax.axvline(float(w), color="red",   alpha=0.18, linewidth=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Wavenumber (cm⁻\xb9)")
    ax.set_ylabel("Intensity")
    fig.tight_layout()
    _stem = sanitize_filename(fname_stem) if fname_stem else sanitize_filename(title)
    fpath = out_dir / (_stem + "_overlay.png")
    fig.savefig(fpath, dpi=dpi)
    plt.close(fig)
    return fpath


def save_loading_and_overlay(
    out_dir: Path,
    run_name: str,
    wavenumbers: np.ndarray,
    pca_model: PCA,
    X_overlay_train: np.ndarray,
    X_overlay_test: np.ndarray,
    top_k: int = TOP_K_WAVENUMBERS,
    gap_thresh: float = GAP_THRESH,
    dpi: int = FIGURE_DPI,
    fname_stem: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Path, Path, Path]:
    """
    Generate loadings plot + spectra overlay + top_wavenumbers CSV.
    Returns (wn_pc1, wn_pc2, loadings_path, overlay_path, top_wn_csv_path).
    """
    pc1_vec = pca_model.components_[0]
    pc2_vec = pca_model.components_[1]
    wn1 = compute_top_wavenumbers(wavenumbers, pc1_vec, top_k)
    wn2 = compute_top_wavenumbers(wavenumbers, pc2_vec, top_k)

    _stem = sanitize_filename(fname_stem) if fname_stem else sanitize_filename(run_name)
    csv_path = out_dir / f"{_stem}_top_wn.csv"
    pd.DataFrame({"top_PC1_wavenumbers": wn1, "top_PC2_wavenumbers": wn2}).to_csv(
        csv_path, index=False
    )
    lp = plot_loadings(out_dir, f"{run_name}__LOADINGS", wavenumbers, pc1_vec, pc2_vec,
                       gap_thresh, fname_stem=fname_stem)
    op = plot_overlay(
        out_dir,
        f"{run_name}__Overlay_top-{top_k}_PC1_blue_PC2_red",
        wavenumbers, X_overlay_train, X_overlay_test, wn1, wn2, gap_thresh, dpi=dpi,
        fname_stem=fname_stem,
    )
    return wn1, wn2, lp, op, csv_path


# =============================================================================
# Journal-ready two-panel figure
# =============================================================================

def make_journal_figure(
    journal_dir: Path,
    q_key: str,
    winner: "pd.Series",
    wn_use: np.ndarray,
    pca: PCA,
    X_ov_tr: np.ndarray,
    X_ov_te: np.ndarray,
    sc12: np.ndarray,
    sc3: np.ndarray,
    meta_tr: pd.DataFrame,
    meta_te: pd.DataFrame,
    si_col: str,
    dx_col: str,
    evr: Tuple[float, float],
    align: Dict,
    sep: Dict,
    wn1: np.ndarray,
    wn2: np.ndarray,
) -> Path:
    """
    Two-panel figure: enhanced scatter (left) + spectra overlay (right).
    Returns saved file path.
    """
    pc1_var, pc2_var = evr
    dm1_12, ctrl_12 = get_dm1_control_masks(meta_tr, dx_col, require_controls=True)
    dm1_3,  _       = get_dm1_control_masks(meta_te, dx_col, require_controls=False)
    si_12 = pd.to_numeric(meta_tr[si_col], errors="coerce").to_numpy(float)
    si_3  = pd.to_numeric(meta_te[si_col], errors="coerce").to_numpy(float)
    vmin, vmax = _common_si_range(si_12, si_3, dm1_12, dm1_3)

    s_dm1_12  = sc12[dm1_12]
    s_ctrl_12 = sc12[ctrl_12]
    s_dm1_3   = sc3[dm1_3]
    segs      = break_by_gap(wn_use, GAP_THRESH)

    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Left — enhanced scatter
    ax1.scatter(s_ctrl_12[:, 0], s_ctrl_12[:, 1], c="orange", marker="o", s=18,
                alpha=0.85, label="Controls", edgecolors="none", zorder=3)
    sc_plot = ax1.scatter(s_dm1_12[:, 0], s_dm1_12[:, 1],
                          c=si_12[dm1_12], cmap="viridis", vmin=vmin, vmax=vmax,
                          marker="o", s=18, alpha=0.85, label="Cohort 1 DM1",
                          edgecolors="none", zorder=3)
    ax1.scatter(s_dm1_3[:, 0], s_dm1_3[:, 1],
                c=si_3[dm1_3], cmap="viridis", vmin=vmin, vmax=vmax,
                marker="s", s=50, alpha=0.95, label="Cohort 2 DM1",
                edgecolors="k", linewidths=0.5, zorder=4)
    plt.colorbar(sc_plot, ax=ax1, label="SI")

    _draw_confidence_ellipse(ax1, s_dm1_12, 0.95,
                             fill=False, edgecolor="steelblue", lw=1.5, ls="--", zorder=2)
    _draw_confidence_ellipse(ax1, s_ctrl_12, 0.95,
                             fill=False, edgecolor="darkorange", lw=1.5, ls="--", zorder=2)

    if len(s_dm1_12) > 0 and len(s_dm1_3) > 0:
        c12 = s_dm1_12.mean(axis=0)
        c3  = s_dm1_3.mean(axis=0)
        ax1.plot([c12[0], c3[0]], [c12[1], c3[1]], "k--", lw=1.0, alpha=0.4)
        ax1.scatter(*c12, marker="D", s=80, c="steelblue", edgecolors="k", lw=0.8, zorder=5)
        ax1.scatter(*c3,  marker="*", s=140, c="darkred",  edgecolors="k", lw=0.5, zorder=5)

    frac_el = align.get("frac_box3_inside_dm1_95ellipse", np.nan)
    bd      = sep.get("hol_bhattacharyya_dm1_vs_ctrl", sep.get("bhattacharyya_dm1_vs_ctrl", np.nan))
    sil     = sep.get("hol_silhouette_dm1_vs_ctrl",    sep.get("silhouette_dm1_vs_ctrl",    np.nan))
    txt = (f"Cohort 2 in 95% ellipse: {_pct(frac_el)}\n"
           f"Bhattacharyya: {_fmt(bd)}\nSilhouette: {_fmt(sil, '.3f')}")
    ax1.text(0.02, 0.98, txt, transform=ax1.transAxes, fontsize=7, va="top",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88))
    ax1.set_xlabel(f"PC1 ({pc1_var*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pc2_var*100:.1f}%)")
    ax1.legend(fontsize=7, loc="lower right", frameon=True, ncol=2)

    # Right — spectra overlay
    for j in range(X_ov_tr.shape[1]):
        for seg in segs:
            ax2.plot(wn_use[seg], X_ov_tr[seg, j], alpha=0.07, lw=0.7, color="steelblue")
    for j in range(X_ov_te.shape[1]):
        for seg in segs:
            ax2.plot(wn_use[seg], X_ov_te[seg, j], alpha=0.07, lw=0.7, color="crimson")
    for w in wn1:
        ax2.axvline(float(w), color="blue", alpha=0.2, lw=1.0)
    for w in wn2:
        ax2.axvline(float(w), color="red",  alpha=0.2, lw=1.0)
    ax2.set_xlabel("Wavenumber (cm⁻\xb9)")
    ax2.set_ylabel("Intensity")
    ax2.set_title(f"Overlay — top-{TOP_K_WAVENUMBERS} PC1 (blue) / PC2 (red)", fontsize=9)

    pp_label  = winner.get("preprocessing", "RAW")
    spec_lab  = winner.get("spectrum_mode", "")
    mode_lab  = winner.get("plot_type", "")
    cpt_lab   = winner.get("concept", "")
    sup_title = (f"{q_key}  |  {pp_label}  |  {spec_lab}  |  {mode_lab}\n"
                 f"concept: {cpt_lab}")
    fig.suptitle(sup_title, fontsize=10, y=1.01)

    fig.tight_layout()
    fpath = journal_dir / f"journal_{sanitize_filename(q_key)}.png"
    fig.savefig(fpath, dpi=JOURNAL_DPI, bbox_inches="tight")
    plt.close(fig)
    return fpath
