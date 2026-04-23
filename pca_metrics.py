#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_metrics.py
==============
All scientific metric functions:
  - Geometry helpers (Mahalanobis, ellipse, hull, NN)
  - Alignment metric bundle  (Q2 / Q4)
  - Separation metric bundle  (Q3A / Q3B)
  - Gradient metric bundle    (Q5)
  - Composite score           (combined leaderboard)
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score as sk_silhouette

from pca_config import CHI2_95
from pca_utils import get_dm1_control_masks


# =============================================================================
# SECTION 3 — GEOMETRY HELPERS
# =============================================================================

def _safe_distribution(
    points: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (mean, cov, cov_inv) or None for degenerate inputs."""
    if points.shape[0] < 3:
        return None
    mean = points.mean(axis=0)
    cov  = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        return None
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(cov_inv)):
        return None
    return mean, cov, cov_inv


def _mahal_sq_array(
    query_points: np.ndarray,
    ref_mean: np.ndarray,
    ref_cov_inv: np.ndarray,
) -> np.ndarray:
    diffs = query_points - ref_mean
    return np.einsum("ij,jk,ik->i", diffs, ref_cov_inv, diffs)


def mahalanobis_centroid_dist(query_points: np.ndarray, ref_points: np.ndarray) -> float:
    dist = _safe_distribution(ref_points)
    if dist is None or len(query_points) < 1:
        return np.nan
    ref_mean, _, ref_cov_inv = dist
    diff = query_points.mean(axis=0) - ref_mean
    return float(np.sqrt(max(0.0, diff @ ref_cov_inv @ diff)))


def individual_mahalanobis_stats(
    query_points: np.ndarray, ref_points: np.ndarray
) -> Tuple[float, float, float]:
    """Return (mean, median, max) of per-point Mahalanobis distances."""
    dist = _safe_distribution(ref_points)
    if dist is None or len(query_points) < 1:
        return np.nan, np.nan, np.nan
    ref_mean, _, ref_cov_inv = dist
    mahal = np.sqrt(np.maximum(_mahal_sq_array(query_points, ref_mean, ref_cov_inv), 0.0))
    return float(np.mean(mahal)), float(np.median(mahal)), float(np.max(mahal))


def fraction_inside_ellipse(
    query_points: np.ndarray,
    ref_points: np.ndarray,
    chi2_threshold: float = CHI2_95,
) -> float:
    dist = _safe_distribution(ref_points)
    if dist is None or len(query_points) < 1:
        return np.nan
    ref_mean, _, ref_cov_inv = dist
    mahal_sq = _mahal_sq_array(query_points, ref_mean, ref_cov_inv)
    return float(np.mean(mahal_sq <= chi2_threshold))


def fraction_inside_convex_hull(
    query_points: np.ndarray, ref_points: np.ndarray
) -> float:
    if len(ref_points) < 4 or len(query_points) < 1:
        return np.nan
    try:
        tri = Delaunay(ref_points)
        return float(np.mean(tri.find_simplex(query_points) >= 0))
    except Exception:
        return np.nan


def mean_nn_dist(query_points: np.ndarray, ref_points: np.ndarray) -> float:
    if len(query_points) < 1 or len(ref_points) < 1:
        return np.nan
    return float(cdist(query_points, ref_points).min(axis=1).mean())


def bhattacharyya_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Bhattacharyya distance between two 2-D Gaussian distributions.
    BD = 0 means complete overlap; higher = better separation.
    Uses Cohort 1 training distributions; Cohort 2 does not contaminate this.
    """
    if len(points_a) < 3 or len(points_b) < 3:
        return np.nan
    mu_a, cov_a = points_a.mean(axis=0), np.cov(points_a.T)
    mu_b, cov_b = points_b.mean(axis=0), np.cov(points_b.T)
    if not (np.all(np.isfinite(cov_a)) and np.all(np.isfinite(cov_b))):
        return np.nan
    cov_mid = (cov_a + cov_b) / 2.0
    try:
        cov_mid_inv = np.linalg.inv(cov_mid)
        det_mid = np.linalg.det(cov_mid)
        det_a   = np.linalg.det(cov_a)
        det_b   = np.linalg.det(cov_b)
    except np.linalg.LinAlgError:
        return np.nan
    if not all(np.isfinite(x) and x > 0 for x in [det_mid, det_a, det_b]):
        return np.nan
    diff  = mu_a - mu_b
    term1 = (1.0 / 8.0) * (diff @ cov_mid_inv @ diff)
    term2 = 0.5 * np.log(det_mid / np.sqrt(det_a * det_b))
    bd    = term1 + term2
    return float(bd) if np.isfinite(bd) else np.nan


def fisher_separation(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if len(points_a) < 2 or len(points_b) < 2:
        return np.nan
    mu_a = points_a.mean(axis=0)
    mu_b = points_b.mean(axis=0)
    var_a = float(np.mean(np.sum((points_a - mu_a) ** 2, axis=1)))
    var_b = float(np.mean(np.sum((points_b - mu_b) ** 2, axis=1)))
    n_a, n_b = len(points_a), len(points_b)
    pooled = (n_a * var_a + n_b * var_b) / (n_a + n_b)
    if pooled <= 0.0:
        return np.nan
    result = float(np.sum((mu_a - mu_b) ** 2) / pooled)
    return result if np.isfinite(result) else np.nan


# =============================================================================
# SECTION 4 — METRIC BUNDLES
# =============================================================================

def compute_alignment_metrics(
    scores_box3_dm1: np.ndarray,
    scores_box12_dm1: np.ndarray,
) -> Dict[str, float]:
    """
    All alignment metrics for Cohort 2 DM1 vs Cohort 1 DM1 in PC1-PC2 space.
    Used for Q2 (DM1_ONLY concept) and Q4 (BOX12_ALL concept).
    """
    nan_row = dict(
        centroid_dist_box3_to_box12dm1        = np.nan,
        mahalanobis_centroid_box3_to_box12dm1 = np.nan,
        mahalanobis_mean_box3_to_box12dm1     = np.nan,
        mahalanobis_median_box3_to_box12dm1   = np.nan,
        mahalanobis_max_box3_to_box12dm1      = np.nan,
        frac_box3_inside_dm1_95ellipse        = np.nan,
        frac_box3_inside_dm1_hull             = np.nan,
        mean_nn_dist_box3_to_box12dm1         = np.nan,
        frac_box12dm1_inside_own_95ellipse    = np.nan,
    )
    if len(scores_box3_dm1) == 0 or len(scores_box12_dm1) == 0:
        return nan_row
    c12 = scores_box12_dm1.mean(axis=0)
    c3  = scores_box3_dm1.mean(axis=0)
    mahal_mean, mahal_med, mahal_max = individual_mahalanobis_stats(
        scores_box3_dm1, scores_box12_dm1
    )
    return dict(
        centroid_dist_box3_to_box12dm1        = float(np.linalg.norm(c3 - c12)),
        mahalanobis_centroid_box3_to_box12dm1 = mahalanobis_centroid_dist(scores_box3_dm1, scores_box12_dm1),
        mahalanobis_mean_box3_to_box12dm1     = mahal_mean,
        mahalanobis_median_box3_to_box12dm1   = mahal_med,
        mahalanobis_max_box3_to_box12dm1      = mahal_max,
        frac_box3_inside_dm1_95ellipse        = fraction_inside_ellipse(scores_box3_dm1, scores_box12_dm1),
        frac_box3_inside_dm1_hull             = fraction_inside_convex_hull(scores_box3_dm1, scores_box12_dm1),
        mean_nn_dist_box3_to_box12dm1         = mean_nn_dist(scores_box3_dm1, scores_box12_dm1),
        frac_box12dm1_inside_own_95ellipse    = fraction_inside_ellipse(scores_box12_dm1, scores_box12_dm1),
    )


def compute_separation_metrics(
    scores_box12_dm1:  np.ndarray,
    scores_box3_dm1:   np.ndarray,
    scores_box12_ctrl: np.ndarray,
    include_box3_in_dm1: bool = True,
) -> Dict[str, float]:
    """
    Control-vs-DM1 separation metrics.

    include_box3_in_dm1=True  → Q3B / holistic: silhouette and centroid use
                                 combined (Cohort 1 + Cohort 2) DM1 cloud.
    include_box3_in_dm1=False → Q3A / clean: Cohort 1 DM1 only.

    Bhattacharyya and Fisher always use Cohort 1 training distributions.
    """
    nan_row = dict(
        bhattacharyya_dm1_vs_ctrl           = np.nan,
        fisher_separation_dm1_vs_ctrl       = np.nan,
        centroid_dist_dm1_vs_ctrl           = np.nan,
        silhouette_dm1_vs_ctrl              = np.nan,
        frac_controls_outside_dm1_95ellipse = np.nan,
    )
    if len(scores_box12_ctrl) == 0 or len(scores_box12_dm1) == 0:
        return nan_row

    if include_box3_in_dm1 and len(scores_box3_dm1) > 0:
        all_dm1 = np.vstack([scores_box12_dm1, scores_box3_dm1])
    else:
        all_dm1 = scores_box12_dm1

    dm1_centroid  = all_dm1.mean(axis=0)
    ctrl_centroid = scores_box12_ctrl.mean(axis=0)
    centroid_dist = float(np.linalg.norm(dm1_centroid - ctrl_centroid))

    bd     = bhattacharyya_distance(scores_box12_dm1, scores_box12_ctrl)
    fisher = fisher_separation(scores_box12_dm1, scores_box12_ctrl)

    all_pts = np.vstack([all_dm1, scores_box12_ctrl])
    labels  = np.array([0] * len(all_dm1) + [1] * len(scores_box12_ctrl))
    sil = np.nan
    if len(np.unique(labels)) >= 2 and len(all_pts) >= 4:
        try:
            sil = float(sk_silhouette(all_pts, labels))
        except Exception:
            pass

    frac_ctrl_inside  = fraction_inside_ellipse(scores_box12_ctrl, scores_box12_dm1)
    frac_ctrl_outside = (1.0 - frac_ctrl_inside) if np.isfinite(frac_ctrl_inside) else np.nan

    return dict(
        bhattacharyya_dm1_vs_ctrl           = bd,
        fisher_separation_dm1_vs_ctrl       = fisher if np.isfinite(fisher) else np.nan,
        centroid_dist_dm1_vs_ctrl           = centroid_dist,
        silhouette_dm1_vs_ctrl              = sil,
        frac_controls_outside_dm1_95ellipse = frac_ctrl_outside,
    )


def compute_gradient_metrics(
    scores_dm1: np.ndarray, si_dm1: np.ndarray
) -> Dict[str, float]:
    """
    Severity-gradient metrics for Q5.
    Uses Cohort 1 DM1 only (Cohort 2 excluded to avoid circular evaluation).
    """
    ok = np.isfinite(si_dm1)
    if ok.sum() < 3:
        return dict(rho_pc1=np.nan, rho_pc2=np.nan, grad_score=np.nan, cv_r2_pc12_to_si=np.nan)
    s  = scores_dm1[ok]
    si = si_dm1[ok]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho1 = float(spearmanr(s[:, 0], si).correlation)
        rho2 = float(spearmanr(s[:, 1], si).correlation)
    rho1 = rho1 if np.isfinite(rho1) else np.nan
    rho2 = rho2 if np.isfinite(rho2) else np.nan
    grad = float(np.nanmax([abs(rho1), abs(rho2)]))
    r2   = np.nan
    if len(si) >= 5:
        try:
            lr = LinearRegression().fit(s[:, :2], si)
            r2 = float(lr.score(s[:, :2], si))
        except Exception:
            pass
    return dict(rho_pc1=rho1, rho_pc2=rho2, grad_score=grad, cv_r2_pc12_to_si=r2)


# =============================================================================
# SECTION 10 — COMPOSITE SCORE
# =============================================================================

def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-normalise each contributing metric then compute weighted composite.
    50 % alignment / 50 % separation.
    """
    n = len(df)
    if n == 0:
        df["composite_score"] = np.nan
        return df

    def rank_hi(col: str) -> pd.Series:
        return df[col].rank(ascending=True,  na_option="bottom") / n if col in df else pd.Series(np.nan, index=df.index)

    def rank_lo(col: str) -> pd.Series:
        return df[col].rank(ascending=False, na_option="bottom") / n if col in df else pd.Series(np.nan, index=df.index)

    df["composite_score"] = (
        0.20 * rank_hi("frac_box3_inside_dm1_95ellipse")
        + 0.15 * rank_lo("mahalanobis_mean_box3_to_box12dm1")
        + 0.15 * rank_lo("mean_nn_dist_box3_to_box12dm1")
        + 0.20 * rank_hi("hol_bhattacharyya_dm1_vs_ctrl")
        + 0.15 * rank_hi("hol_silhouette_dm1_vs_ctrl")
        + 0.15 * rank_hi("hol_frac_controls_outside_dm1_95ellipse")
    )
    return df
