# -*- coding: utf-8 -*-
"""
pca_box3_alignment_scoring.py

A companion evaluation to 'pca tells which is best si gradient.py'.
Runs the SAME sweep (spectral windows × PCA concepts × preprocessing combos ×
baseline flags × PATIENT_AVG / ALL_SPECTRA) but scores each configuration on
TWO additional criteria:

  1. Box3-DM1 ALIGNMENT with Box1-2 DM1: do the unseen patients project into
     the same PCA region as the training cohort?

  2. CONTROL SEPARATION from DM1: does the PCA geometry preserve the
     diagnostic boundary between healthy and diseased?

=============================================================================
METHODS NOTE
=============================================================================

WHY BOX3 IS EXCLUDED FROM THE SEVERITY-GRADIENT RANKING
---------------------------------------------------------
The severity-gradient score (Spearman rho of PC scores vs Splicing Index) is
computed exclusively on Box1-2 DM1 patients in the companion script.  Box3 is
an independent validation cohort; including it in the metric used to SELECT
the best preprocessing would cause circular evaluation and inflate apparent
generalization.  Box3 is therefore a held-out cohort evaluated here for
domain alignment only.

WHY BOX3 IS STILL SHOWN IN THE PCA SCATTER PLOTS
--------------------------------------------------
Projecting Box3 into a PCA space trained on Box1-2 is the standard way to
visualise domain generalization.  The plots show qualitatively whether unseen
patients fall inside or outside the Box1-2 DM1 cloud, providing evidence that
the learned axes are physically meaningful and not Box1-2-specific artefacts.

=============================================================================
ALIGNMENT METRICS  (Box3 DM1 vs Box1-2 DM1)
=============================================================================

centroid_dist_box3_to_box12dm1
    Euclidean distance in PC1-PC2 space between the two cohort centroids.
    Contextual but does not account for cloud spread; use alongside Mahal.

mahalanobis_centroid_box3_to_box12dm1
    Mahalanobis distance from the BOX3 DM1 centroid to the Box1-2 DM1
    distribution (mean + covariance).  A value <= sqrt(chi2_95) ~= 2.45
    means the Box3 centroid falls inside the Box1-2 95% confidence ellipse.
    Gives a single summary of whether the cohort centroids are compatible.

mahalanobis_mean_box3_to_box12dm1
mahalanobis_median_box3_to_box12dm1
mahalanobis_max_box3_to_box12dm1
    Per-PATIENT Mahalanobis distances from each Box3 DM1 point to the Box1-2
    DM1 distribution, then summarised as mean / median / max.  These reveal
    whether all Box3 patients overlap (low mean AND low max), or whether the
    cohort aligns well on average but with outliers (low mean, high max).
    Suitable for paper statements like "Box3 showed a mean Mahalanobis
    distance of Y relative to the Box1-2 DM1 distribution."

frac_box3_inside_dm1_95ellipse
    Fraction of individual Box3 DM1 points whose squared Mahalanobis distance
    is <= chi2(0.95, df=2) ~= 5.991 — inside the 95% confidence ellipse of
    the Box1-2 DM1 distribution.  Value near 1 = Box3 is statistically
    indistinguishable from Box1-2 in this PCA space.
    Supports paper statement: "X% of Box3 DM1 samples fell within the 95%
    PCA confidence ellipse of Box1-2 DM1."

frac_box3_inside_dm1_hull
    Fraction of Box3 DM1 points inside the convex hull of Box1-2 DM1 scores.
    Non-parametric complement to the ellipse fraction; robust when the Box1-2
    cloud is asymmetric or multimodal.

mean_nn_dist_box3_to_box12dm1
    Mean Euclidean distance from each Box3 DM1 point to its nearest Box1-2
    DM1 neighbour.  Preprocessing that creates a tight, overlapping cloud
    minimises this.

frac_box12dm1_inside_own_95ellipse
    Self-consistency check: fraction of Box1-2 DM1 points inside their own
    95% ellipse.  Should be ~0.95 for well-behaved data; values far from 0.95
    indicate non-Gaussian or heavy-tailed distributions where the ellipse
    metric should be interpreted with caution.

=============================================================================
SEPARATION METRICS  (Controls vs DM1)
=============================================================================

bhattacharyya_dm1_vs_ctrl
    Bhattacharyya distance between the Box1-2 DM1 and control distributions,
    modelled as 2D Gaussians in PC space.

        BD = (1/8)(μ_DM1 - μ_ctrl)ᵀ Σ_mid⁻¹ (μ_DM1 - μ_ctrl)
             + (1/2) ln( |Σ_mid| / √(|Σ_DM1| |Σ_ctrl|) )
    where Σ_mid = (Σ_DM1 + Σ_ctrl) / 2.

    This is the primary spread-aware separation metric: it accounts for both
    centroid displacement AND the spread/overlap of each group's covariance
    ellipse.  BD = 0 means complete overlap; larger values mean less overlap.

fisher_separation_dm1_vs_ctrl
    Fisher-criterion separation: centroid_dist² / pooled_within_group_variance.
    Pooled variance = (n_DM1 · var_DM1 + n_ctrl · var_ctrl) / (n_DM1 + n_ctrl),
    where var_x = mean squared distance from group centroid.
    Rewards configurations where groups are far apart AND internally compact.

control_dm1_centroid_dist
    Raw Euclidean centroid distance (retained for direct comparison with older
    results and as an interpretable baseline for the above two metrics).

silhouette_dm1_vs_ctrl
    Scikit-learn silhouette score with 2 labels: DM1 (label 0, Box1-2 + Box3
    combined) vs Controls (label 1, Box1-2 only).  Simultaneously rewards
    configurations where all DM1 patients cluster together AND away from
    controls.  Range [-1, +1]; higher is better.

frac_controls_outside_dm1_95ellipse
    Fraction of Box1-2 control points whose Mahalanobis distance to the
    Box1-2 DM1 distribution EXCEEDS the 95% threshold — i.e., controls that
    fall outside the DM1 confidence region.  Higher = cleaner separation.
    Complements bhattacharyya_dm1_vs_ctrl with a directly interpretable %.

=============================================================================
COMPOSITE SCORE
=============================================================================
All metrics are rank-normalised across the full configuration sweep.
Rank 1 = worst config, rank N = best.  Normalised rank = rank / N ∈ (0, 1].

Weights are EXACTLY 50 % alignment / 50 % separation:

  ALIGNMENT (50 %):
    0.20 × rank_hi(frac_box3_inside_dm1_95ellipse)
    0.15 × rank_lo(mahalanobis_mean_box3_to_box12dm1)     ← lower is better
    0.15 × rank_lo(mean_nn_dist_box3_to_box12dm1)         ← lower is better

  SEPARATION (50 %):
    0.20 × rank_hi(bhattacharyya_dm1_vs_ctrl)
    0.15 × rank_hi(silhouette_dm1_vs_ctrl)
    0.15 × rank_hi(frac_controls_outside_dm1_95ellipse)

  Total = 1.00

rank_hi: higher metric value → higher normalised rank (better).
rank_lo: lower metric value → higher normalised rank (better).

IMPORTANT — HOW TO USE THE COMPOSITE SCORE
-------------------------------------------
The composite score is intended as a sweep-ranking tool: use it to identify
which preprocessing + spectral window combinations rise consistently to the
top across BOTH alignment and separation objectives.  It is NOT a biological
measure and should not appear in the paper as a primary result.

Paper-facing biological statements should be built from the individual
sub-metrics reported in the CSV, for example:

  "X% of Box3 DM1 samples fell within the 95% PCA confidence ellipse of
   Box1-2 DM1 (frac_box3_inside_dm1_95ellipse)."

  "Box3 showed a mean Mahalanobis distance of Y relative to the Box1-2 DM1
   distribution (mahalanobis_mean_box3_to_box12dm1); the median was Z,
   indicating that [most / all / some] patients aligned well
   (mahalanobis_median_box3_to_box12dm1)."

  "The Bhattacharyya distance between DM1 and control distributions was W
   (bhattacharyya_dm1_vs_ctrl), corresponding to a Fisher separation ratio
   of V (fisher_separation_dm1_vs_ctrl)."

All three Mahalanobis outputs are retained in every saved CSV:
  mahalanobis_centroid_box3_to_box12dm1  — cohort-level centroid shift
  mahalanobis_mean_box3_to_box12dm1      — patient-level mean overlap
  mahalanobis_median_box3_to_box12dm1    — patient-level median (robust to outliers)
  mahalanobis_max_box3_to_box12dm1       — worst-case individual patient distance
=============================================================================
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score as sk_silhouette

from data_loader import load_data
# preprocessing.py must be present in the same folder.
# Expected API:
#   pp = Preprocessing()
#   pp.fit(X_fxN, methods)          # X is (features × N)
#   X_out = pp.transform(X_fxN, methods)
# Supported methods: "Normalization", "SNV", "Second Derivative", "EMSC"
# Baseline correction is NOT supported by the current preprocessing.py.
from preprocessing import Preprocessing


# =========================================================================
# CONFIG  —  update these paths before running
# =========================================================================

OUT_ROOT = Path(r"C:\Users\notha\OneDrive\Desktop\SERS_Cluster_work\sers_project\PCA_RESULTS\ALIGNMENT_SCORING")
BOX12_DATA_DIR = r"C:\Users\notha\OneDrive\Desktop\SERS_Cluster_work\sers_project\data\Box12_spectra"
BOX12_META_CSV = r"C:\Users\notha\OneDrive\Desktop\SERS_Cluster_work\sers_project\y_metadata.csv"

BOX3_DATA_DIR  = r"C:\Users\notha\OneDrive\Desktop\SERS_Cluster_work\sers_project\data\Box3_spectra"
BOX3_META_CSV  = r"C:\Users\notha\OneDrive\Desktop\SERS_Cluster_work\sers_project\y_metadata_test_updated_in_order.csv"
CONCEPTS = [
    "WEIGHTS__BOX12_DM1_ONLY",
    "WEIGHTS__BOX12_ALL",
]

SPECTRUM_MODES = [
    ("FULL_SPECTRUM",  None),
    ("TRIM_500_3100",  (500.0, 3100.0)),
    ("TRIM_700_2800",  (700.0, 2800.0)),
]

SUPPORTED_METHODS = [
    "Normalization",
    "SNV",
    "Second Derivative",
    "EMSC",
]

BASELINE_KWARGS = dict(lam=1e6, p=0.01, niter=10)

# chi2(0.95, df=2) threshold — boundary of the 95% confidence ellipse in 2D
CHI2_95 = chi2.ppf(0.95, df=2)   # ≈ 5.991


# =========================================================================
# UTILITY FUNCTIONS  (mirrored from the gradient script for consistency)
# =========================================================================

def sanitize_filename(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def detect_si_column(meta: pd.DataFrame) -> str:
    for c in ["target_SI", "SI", "SplicingIndex", "Splicing_Index"]:
        if c in meta.columns:
            return c
    for c in meta.columns:
        if "si" in c.lower():
            return c
    raise KeyError(f"Could not detect SI column. Columns={meta.columns.tolist()}")


def detect_dx_column(meta: pd.DataFrame) -> str:
    if "Type" in meta.columns:
        return "Type"
    for c in meta.columns:
        if c.lower() in ["dx", "diagnosis", "group", "class", "type"]:
            return c
    raise KeyError(f"Could not detect diagnosis column. Columns={meta.columns.tolist()}")


def get_dm1_control_masks(
    meta: pd.DataFrame, dx_col: str, require_controls: bool
) -> Tuple[np.ndarray, np.ndarray]:
    dx = meta[dx_col].astype(str).str.strip().str.lower()
    dm1_mask  = dx.str.contains("dm1").to_numpy()
    ctrl_mask = (dx.str.contains("control") | dx.str.contains("adco")).to_numpy()
    if require_controls and not ctrl_mask.any():
        raise ValueError(f"No Control rows detected using column '{dx_col}'.")
    if not dm1_mask.any():
        raise ValueError(f"No DM1 rows detected using column '{dx_col}'.")
    return dm1_mask, ctrl_mask


def safe_name(methods: List[str], do_baseline: bool) -> str:
    base = "+".join(m.upper() for m in methods) if methods else "RAW"
    return f"BASELINE__{base}" if do_baseline else base


def build_all_ordered_combos(methods: List[str]) -> List[List[str]]:
    combos: List[List[str]] = [[]]
    for m in methods:
        combos.append([m])
    if "Normalization" in methods and "EMSC" in methods:
        combos.append(["Normalization", "EMSC"])
    if "SNV" in methods and "Second Derivative" in methods:
        combos.append(["SNV", "Second Derivative"])
    if "Normalization" in methods and "SNV" in methods:
        combos.append(["Normalization", "SNV"])
    if "Normalization" in methods and "SNV" in methods and "Second Derivative" in methods:
        combos.append(["Normalization", "SNV", "Second Derivative"])
    uniq: List[List[str]] = []
    seen: set = set()
    for c in combos:
        key = tuple(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


def trim_region(
    wavenumbers: np.ndarray, X: np.ndarray, lo: float, hi: float
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (wavenumbers >= lo) & (wavenumbers <= hi)
    return wavenumbers[mask], X[mask, :]


def apply_pipeline(
    X: np.ndarray, methods: List[str], do_baseline: bool, baseline_kwargs: dict
) -> np.ndarray:
    if do_baseline:
        raise NotImplementedError(
            "do_baseline=True is not supported: baseline correction is not "
            "implemented in the current preprocessing.py API.  Keep "
            "BASELINE_FLAGS = [False] in the CONFIG section."
        )
    if not methods:
        return X.copy()
    # New API: Preprocessing().fit(X_fxN, methods) then .transform(X_fxN, methods)
    # X is (features × N); fit and transform on the same matrix (no held-out split
    # inside preprocessing for this PCA sweep).
    pp = Preprocessing()
    pp.fit(X, methods)
    return pp.transform(X, methods)


def compute_explained_variance(pca: PCA) -> Tuple[float, float]:
    evr = pca.explained_variance_ratio_
    return (float(evr[0]) if len(evr) > 0 else np.nan,
            float(evr[1]) if len(evr) > 1 else np.nan)


# =========================================================================
# CORE GEOMETRY HELPERS
# =========================================================================

def _safe_distribution(
    points: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return (mean, cov, cov_inv) for a 2-D point cloud, or None if degenerate.
    Requires >= 3 points and a non-singular, finite covariance matrix.
    """
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
    """
    Squared Mahalanobis distance for every row in query_points relative to
    a reference distribution defined by (ref_mean, ref_cov_inv).
    Returns shape (len(query_points),).
    """
    diffs = query_points - ref_mean
    # einsum: for each row i, compute diffs[i] @ cov_inv @ diffs[i]
    return np.einsum("ij,jk,ik->i", diffs, ref_cov_inv, diffs)


# ---- Mahalanobis: centroid-level ----------------------------------------

def mahalanobis_centroid_dist(
    query_points: np.ndarray, ref_points: np.ndarray
) -> float:
    """
    Mahalanobis distance from the CENTROID of query_points to the reference
    distribution estimated from ref_points.  A single cohort-level summary.
    """
    dist = _safe_distribution(ref_points)
    if dist is None or len(query_points) < 1:
        return np.nan
    ref_mean, _, ref_cov_inv = dist
    diff = query_points.mean(axis=0) - ref_mean
    return float(np.sqrt(max(0.0, diff @ ref_cov_inv @ diff)))


# ---- Mahalanobis: individual point statistics ---------------------------

def individual_mahalanobis_stats(
    query_points: np.ndarray, ref_points: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute per-point Mahalanobis distances from each query point to the
    distribution estimated from ref_points, then return (mean, median, max).

    These reveal whether Box3 aligns uniformly (low mean AND low max) or
    whether a subset of patients are outliers (low mean, elevated max).
    """
    dist = _safe_distribution(ref_points)
    if dist is None or len(query_points) < 1:
        return np.nan, np.nan, np.nan
    ref_mean, _, ref_cov_inv = dist
    mahal_sq = _mahal_sq_array(query_points, ref_mean, ref_cov_inv)
    mahal    = np.sqrt(np.maximum(mahal_sq, 0.0))
    return float(np.mean(mahal)), float(np.median(mahal)), float(np.max(mahal))


# ---- Ellipse containment ------------------------------------------------

def fraction_inside_ellipse(
    query_points: np.ndarray,
    ref_points: np.ndarray,
    chi2_threshold: float = CHI2_95,
) -> float:
    """
    Fraction of query_points whose squared Mahalanobis distance to the
    ref distribution is <= chi2_threshold (default: CHI2_95 ~= 5.991,
    corresponding to the 95% confidence ellipse in 2D).
    """
    dist = _safe_distribution(ref_points)
    if dist is None or len(query_points) < 1:
        return np.nan
    ref_mean, _, ref_cov_inv = dist
    mahal_sq = _mahal_sq_array(query_points, ref_mean, ref_cov_inv)
    return float(np.mean(mahal_sq <= chi2_threshold))


# ---- Convex hull containment --------------------------------------------

def fraction_inside_convex_hull(
    query_points: np.ndarray, ref_points: np.ndarray
) -> float:
    """
    Fraction of query_points that fall inside the convex hull of ref_points.
    Uses Delaunay triangulation; find_simplex >= 0 means inside.
    Returns nan if the hull cannot be computed (< 4 points, collinear, etc.).
    """
    if len(ref_points) < 4 or len(query_points) < 1:
        return np.nan
    try:
        tri = Delaunay(ref_points)
        return float(np.mean(tri.find_simplex(query_points) >= 0))
    except Exception:
        return np.nan


# ---- Nearest-neighbour distance -----------------------------------------

def mean_nn_dist(query_points: np.ndarray, ref_points: np.ndarray) -> float:
    """Mean Euclidean distance from each query point to its nearest ref neighbour."""
    if len(query_points) < 1 or len(ref_points) < 1:
        return np.nan
    return float(cdist(query_points, ref_points).min(axis=1).mean())


# ---- Spread-aware separation metrics ------------------------------------

def bhattacharyya_distance(
    points_a: np.ndarray, points_b: np.ndarray
) -> float:
    """
    Bhattacharyya distance between two 2D Gaussian distributions estimated
    from point clouds a and b:

        BD = (1/8)(μA − μB)ᵀ Σmid⁻¹ (μA − μB)
             + (1/2) ln( |Σmid| / √(|ΣA| |ΣB|) )
        where Σmid = (ΣA + ΣB) / 2.

    BD = 0  → complete distribution overlap.
    BD > 0  → increasing separation; no theoretical upper bound.

    This is the primary spread-aware separation metric: it accounts for
    BOTH centroid displacement AND the shape/spread of each group's
    covariance ellipse.  Returns nan if either distribution is degenerate.
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
        det_mid     = np.linalg.det(cov_mid)
        det_a       = np.linalg.det(cov_a)
        det_b       = np.linalg.det(cov_b)
    except np.linalg.LinAlgError:
        return np.nan

    # All determinants must be strictly positive
    if not all(np.isfinite(x) and x > 0 for x in [det_mid, det_a, det_b]):
        return np.nan

    diff  = mu_a - mu_b
    term1 = (1.0 / 8.0) * (diff @ cov_mid_inv @ diff)
    term2 = 0.5 * np.log(det_mid / np.sqrt(det_a * det_b))
    bd    = term1 + term2
    return float(bd) if np.isfinite(bd) else np.nan


def fisher_separation(
    points_a: np.ndarray, points_b: np.ndarray
) -> float:
    """
    Fisher-criterion separation score:

        F = centroid_dist² / pooled_within_variance

    where pooled_within = (n_a·var_a + n_b·var_b) / (n_a + n_b)
    and var_x = mean squared distance from the group centroid (scalar).

    Rewards configurations where the groups are far apart AND each group is
    internally compact.  Unlike bare centroid distance, a spread-out group
    is penalised even if its centroid is far away.  Returns nan if either
    group has < 2 points or zero within-group variance.
    """
    if len(points_a) < 2 or len(points_b) < 2:
        return np.nan

    mu_a = points_a.mean(axis=0)
    mu_b = points_b.mean(axis=0)
    var_a = float(np.mean(np.sum((points_a - mu_a) ** 2, axis=1)))
    var_b = float(np.mean(np.sum((points_b - mu_b) ** 2, axis=1)))

    n_a, n_b   = len(points_a), len(points_b)
    pooled_var = (n_a * var_a + n_b * var_b) / (n_a + n_b)

    if pooled_var <= 0.0:
        return np.nan
    result = float(np.sum((mu_a - mu_b) ** 2) / pooled_var)
    return result if np.isfinite(result) else np.nan


# =========================================================================
# METRIC BUNDLES
# =========================================================================

def compute_alignment_metrics(
    scores_box3_dm1:  np.ndarray,
    scores_box12_dm1: np.ndarray,
) -> Dict[str, float]:
    """
    All alignment metrics for Box3 DM1 relative to Box1-2 DM1
    in the PC1-PC2 score space.
    """
    nan_row = dict(
        centroid_dist_box3_to_box12dm1         = np.nan,
        mahalanobis_centroid_box3_to_box12dm1  = np.nan,
        mahalanobis_mean_box3_to_box12dm1      = np.nan,
        mahalanobis_median_box3_to_box12dm1    = np.nan,
        mahalanobis_max_box3_to_box12dm1       = np.nan,
        frac_box3_inside_dm1_95ellipse         = np.nan,
        frac_box3_inside_dm1_hull              = np.nan,
        mean_nn_dist_box3_to_box12dm1          = np.nan,
        frac_box12dm1_inside_own_95ellipse     = np.nan,
    )
    if len(scores_box3_dm1) == 0 or len(scores_box12_dm1) == 0:
        return nan_row

    c12 = scores_box12_dm1.mean(axis=0)
    c3  = scores_box3_dm1.mean(axis=0)

    mahal_mean, mahal_med, mahal_max = individual_mahalanobis_stats(
        scores_box3_dm1, scores_box12_dm1
    )

    # Self-consistency: what fraction of Box1-2 DM1 fall inside their own ellipse?
    # Expect ~0.95 for well-behaved Gaussian data.
    self_frac = fraction_inside_ellipse(scores_box12_dm1, scores_box12_dm1)

    return dict(
        centroid_dist_box3_to_box12dm1        = float(np.linalg.norm(c3 - c12)),
        mahalanobis_centroid_box3_to_box12dm1 = mahalanobis_centroid_dist(scores_box3_dm1, scores_box12_dm1),
        mahalanobis_mean_box3_to_box12dm1     = mahal_mean,
        mahalanobis_median_box3_to_box12dm1   = mahal_med,
        mahalanobis_max_box3_to_box12dm1      = mahal_max,
        frac_box3_inside_dm1_95ellipse        = fraction_inside_ellipse(scores_box3_dm1, scores_box12_dm1),
        frac_box3_inside_dm1_hull             = fraction_inside_convex_hull(scores_box3_dm1, scores_box12_dm1),
        mean_nn_dist_box3_to_box12dm1         = mean_nn_dist(scores_box3_dm1, scores_box12_dm1),
        frac_box12dm1_inside_own_95ellipse    = self_frac,
    )


def compute_separation_metrics(
    scores_box12_dm1:  np.ndarray,
    scores_box3_dm1:   np.ndarray,
    scores_box12_ctrl: np.ndarray,
) -> Dict[str, float]:
    """
    Control-vs-DM1 separation metrics.
    'DM1' = Box1-2 + Box3 combined for centroid / silhouette metrics.
    Bhattacharyya and Fisher use Box1-2 DM1 vs Box1-2 controls (training
    distributions only, for a clean comparison unconfounded by Box3 shift).
    Controls are Box1-2 only throughout.
    """
    nan_row = dict(
        control_dm1_centroid_dist          = np.nan,
        bhattacharyya_dm1_vs_ctrl          = np.nan,
        fisher_separation_dm1_vs_ctrl      = np.nan,
        silhouette_dm1_vs_ctrl             = np.nan,
        frac_controls_outside_dm1_95ellipse = np.nan,
    )
    if len(scores_box12_ctrl) == 0 or len(scores_box12_dm1) == 0:
        return nan_row

    # Combined DM1 for centroid / silhouette
    all_dm1 = (
        np.vstack([scores_box12_dm1, scores_box3_dm1])
        if len(scores_box3_dm1) > 0
        else scores_box12_dm1
    )
    dm1_centroid  = all_dm1.mean(axis=0)
    ctrl_centroid = scores_box12_ctrl.mean(axis=0)
    centroid_dist = float(np.linalg.norm(dm1_centroid - ctrl_centroid))

    # Bhattacharyya: Box1-2 DM1 vs Box1-2 controls (training distributions)
    bd = bhattacharyya_distance(scores_box12_dm1, scores_box12_ctrl)

    # Fisher: Box1-2 DM1 vs Box1-2 controls (training distributions)
    fisher = fisher_separation(scores_box12_dm1, scores_box12_ctrl)

    # Silhouette: all DM1 (label 0) vs controls (label 1)
    all_pts = np.vstack([all_dm1, scores_box12_ctrl])
    labels  = np.array([0] * len(all_dm1) + [1] * len(scores_box12_ctrl))
    sil = np.nan
    if len(np.unique(labels)) >= 2 and len(all_pts) >= 4:
        try:
            sil = float(sk_silhouette(all_pts, labels))
        except Exception:
            pass

    # Fraction of controls OUTSIDE the Box1-2 DM1 95% ellipse
    # (1 - fraction_inside): higher means controls are well-excluded from DM1 region
    frac_ctrl_inside = fraction_inside_ellipse(scores_box12_ctrl, scores_box12_dm1)
    frac_ctrl_outside = (1.0 - frac_ctrl_inside) if np.isfinite(frac_ctrl_inside) else np.nan

    return dict(
        control_dm1_centroid_dist           = centroid_dist,
        bhattacharyya_dm1_vs_ctrl           = bd,
        fisher_separation_dm1_vs_ctrl       = fisher if np.isfinite(fisher) else np.nan,
        silhouette_dm1_vs_ctrl              = sil,
        frac_controls_outside_dm1_95ellipse = frac_ctrl_outside,
    )


# =========================================================================
# VISUALIZATION
# =========================================================================

def _draw_confidence_ellipse(
    ax, points_2d: np.ndarray, confidence: float = 0.95, **kwargs
) -> bool:
    """
    Draw a confidence ellipse on ax for a 2-D point cloud.
    Returns True if drawn successfully.
    """
    if len(points_2d) < 3:
        return False
    mean = points_2d.mean(axis=0)
    cov  = np.cov(points_2d.T)
    if not np.all(np.isfinite(cov)) or cov.ndim < 2:
        return False
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order
    except np.linalg.LinAlgError:
        return False

    eigenvalues = np.maximum(eigenvalues, 0.0)
    chi2_val    = chi2.ppf(confidence, df=2)

    # eigh returns ascending order; largest eigenvalue = last column
    v_major = eigenvectors[:, -1]
    angle   = np.degrees(np.arctan2(v_major[1], v_major[0]))
    width   = 2.0 * np.sqrt(chi2_val * eigenvalues[-1])   # major axis
    height  = 2.0 * np.sqrt(chi2_val * eigenvalues[0])    # minor axis

    ax.add_patch(Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs))
    return True


def _fmt(val: float, fmt: str = ".2f") -> str:
    """Format a float; return 'N/A' for nan/inf."""
    return f"{val:{fmt}}" if np.isfinite(val) else "N/A"


def _pct(val: float) -> str:
    """Format a fraction as a percentage string; return 'N/A' for nan."""
    return f"{val * 100:.0f}%" if np.isfinite(val) else "N/A"


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
) -> None:
    """
    Enhanced PCA scatter plot.  Adds on top of the standard scatter:
      - 95% confidence ellipses: Box1-2 DM1 (blue dashed), controls (orange dashed)
      - Box1-2 DM1 centroid (diamond) and Box3 DM1 centroid (star)
      - Dashed line connecting the two DM1 centroids
      - Text box with the 5 most paper-relevant metrics
    """
    pc1_var, pc2_var = evr

    dm1_12, ctrl_12 = get_dm1_control_masks(meta_box12, dx_col, require_controls=True)
    dm1_3,  _       = get_dm1_control_masks(meta_box3,  dx_col, require_controls=False)

    si_12 = pd.to_numeric(meta_box12[si_col], errors="coerce").to_numpy(float)
    si_3  = pd.to_numeric(meta_box3[si_col],  errors="coerce").to_numpy(float)

    scores_dm1_12  = scores_box12[dm1_12]
    scores_ctrl_12 = scores_box12[ctrl_12]
    scores_dm1_3   = scores_box3[dm1_3]

    all_si = np.concatenate([si_12[dm1_12], si_3[dm1_3]])
    all_si = all_si[np.isfinite(all_si)]
    vmin = all_si.min() if len(all_si) > 0 else 0.0
    vmax = all_si.max() if len(all_si) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 7))

    # --- scatter points ---
    ax.scatter(
        scores_ctrl_12[:, 0], scores_ctrl_12[:, 1],
        c="orange", marker="o", s=18, alpha=0.85,
        label="Box1-2 Control", edgecolors="none", zorder=3,
    )
    sc1 = ax.scatter(
        scores_dm1_12[:, 0], scores_dm1_12[:, 1],
        c=si_12[dm1_12], cmap="viridis", vmin=vmin, vmax=vmax,
        marker="o", s=18, alpha=0.85,
        label="Box1-2 DM1", edgecolors="none", zorder=3,
    )
    ax.scatter(
        scores_dm1_3[:, 0], scores_dm1_3[:, 1],
        c=si_3[dm1_3], cmap="viridis", vmin=vmin, vmax=vmax,
        marker="s", s=50, alpha=0.95,
        label="Box3 DM1", edgecolors="k", linewidths=0.5, zorder=4,
    )
    plt.colorbar(sc1, ax=ax, label="SI (DM1 severity)")

    # --- 95% confidence ellipses ---
    _draw_confidence_ellipse(
        ax, scores_dm1_12, confidence=0.95,
        fill=False, edgecolor="steelblue", linewidth=1.5, linestyle="--",
        label="Box1-2 DM1 95% ellipse", zorder=2,
    )
    _draw_confidence_ellipse(
        ax, scores_ctrl_12, confidence=0.95,
        fill=False, edgecolor="darkorange", linewidth=1.5, linestyle="--",
        label="Box1-2 Control 95% ellipse", zorder=2,
    )

    # --- centroids + connecting line ---
    if len(scores_dm1_12) > 0 and len(scores_dm1_3) > 0:
        c12 = scores_dm1_12.mean(axis=0)
        c3  = scores_dm1_3.mean(axis=0)
        ax.plot(
            [c12[0], c3[0]], [c12[1], c3[1]],
            "k--", linewidth=1.0, alpha=0.5, zorder=2,
        )
        ax.scatter(
            *c12, marker="D", s=90, c="steelblue", edgecolors="k",
            linewidths=0.8, zorder=5, label="Box1-2 DM1 centroid",
        )
        ax.scatter(
            *c3, marker="*", s=160, c="darkred", edgecolors="k",
            linewidths=0.5, zorder=5, label="Box3 DM1 centroid",
        )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel(f"PC1 ({pc1_var * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var * 100:.1f}%)")
    ax.legend(loc="best", frameon=True, fontsize=6.5, ncol=2)

    # --- metrics text box ---
    # 5 metrics chosen for direct paper interpretability
    frac_el    = alignment_metrics.get("frac_box3_inside_dm1_95ellipse",      np.nan)
    frac_hull  = alignment_metrics.get("frac_box3_inside_dm1_hull",           np.nan)
    mahal_mean = alignment_metrics.get("mahalanobis_mean_box3_to_box12dm1",   np.nan)
    bd         = separation_metrics.get("bhattacharyya_dm1_vs_ctrl",          np.nan)
    sil        = separation_metrics.get("silhouette_dm1_vs_ctrl",             np.nan)
    frac_ctrl_out = separation_metrics.get("frac_controls_outside_dm1_95ellipse", np.nan)

    metrics_text = (
        f"── Alignment (Box3 → Box1-2 DM1) ──\n"
        f"Box3 in 95% ellipse : {_pct(frac_el)}\n"
        f"Box3 in hull        : {_pct(frac_hull)}\n"
        f"Mean Mahal dist     : {_fmt(mahal_mean)}\n"
        f"── Separation (DM1 vs Ctrl) ────────\n"
        f"Bhattacharyya dist  : {_fmt(bd)}\n"
        f"Silhouette          : {_fmt(sil, '.3f')}\n"
        f"Ctrl outside ellipse: {_pct(frac_ctrl_out)}"
    )
    ax.text(
        0.02, 0.98, metrics_text,
        transform=ax.transAxes, fontsize=7.0, verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88),
    )

    fig.tight_layout()
    fname = sanitize_filename(title.replace(" ", "_")) + "__alignment.png"
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)


# =========================================================================
# PCA RUNNERS
# =========================================================================

def run_patient_avg_alignment(
    out_dir: Path,
    run_name: str,
    concept_name: str,
    X_avg_train: np.ndarray,
    X_avg_test: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    si_col: str,
    dx_col: str,
    dm1_mask_train: np.ndarray,
    ctrl_mask_train: np.ndarray,
) -> Dict:
    """Fit PCA on Box1-2 (PATIENT_AVG mode), project Box3, compute metrics."""
    if concept_name == "WEIGHTS__BOX12_DM1_ONLY":
        X_fit = X_avg_train[:, dm1_mask_train].T
    elif concept_name == "WEIGHTS__BOX12_ALL":
        X_fit = X_avg_train.T
    else:
        raise ValueError(f"Unknown concept: {concept_name}")

    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_fit)

    scores_12 = pca.transform(X_avg_train.T)
    scores_3  = pca.transform(X_avg_test.T)
    evr = compute_explained_variance(pca)

    dm1_3, _ = get_dm1_control_masks(meta_test, dx_col, require_controls=False)

    scores_dm1_12  = scores_12[dm1_mask_train]
    scores_ctrl_12 = scores_12[ctrl_mask_train]
    scores_dm1_3   = scores_3[dm1_3]

    align = compute_alignment_metrics(scores_dm1_3, scores_dm1_12)
    sep   = compute_separation_metrics(scores_dm1_12, scores_dm1_3, scores_ctrl_12)

    plot_enhanced_pca_scatter(
        out_dir=out_dir,
        title=f"{run_name}__{concept_name}__PATIENT_AVG",
        scores_box12=scores_12, scores_box3=scores_3,
        meta_box12=meta_train, meta_box3=meta_test,
        si_col=si_col, dx_col=dx_col, evr=evr,
        alignment_metrics=align, separation_metrics=sep,
    )

    return dict(
        pc1_var=evr[0], pc2_var=evr[1],
        n_box12_dm1=int(dm1_mask_train.sum()),
        n_box3_dm1 =int(dm1_3.sum()),
        n_controls  =int(ctrl_mask_train.sum()),
        **align, **sep,
    )


def run_all_spectra_alignment(
    out_dir: Path,
    run_name: str,
    concept_name: str,
    X_all_train: np.ndarray,
    X_all_test: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    si_col: str,
    dx_col: str,
    dm1_mask_train_pat: np.ndarray,
    ctrl_mask_train_pat: np.ndarray,
) -> Dict:
    """
    Fit PCA on Box1-2 (ALL_SPECTRA mode — 9 spectra per patient).
    Patient-level masks are broadcast to spectrum level (repeated ×9).
    """
    n_pat_train = len(meta_train)
    n_pat_test  = len(meta_test)

    dm1_spec_train  = np.repeat(dm1_mask_train_pat,  9)
    ctrl_spec_train = np.repeat(ctrl_mask_train_pat, 9)

    if concept_name == "WEIGHTS__BOX12_DM1_ONLY":
        X_fit = X_all_train[:, dm1_spec_train].T
    elif concept_name == "WEIGHTS__BOX12_ALL":
        X_fit = X_all_train.T
    else:
        raise ValueError(f"Unknown concept: {concept_name}")

    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_fit)

    scores_12 = pca.transform(X_all_train.T)
    scores_3  = pca.transform(X_all_test.T)
    evr = compute_explained_variance(pca)

    dm1_3_pat, _ = get_dm1_control_masks(meta_test, dx_col, require_controls=False)
    dm1_3_spec   = np.repeat(dm1_3_pat, 9)

    scores_dm1_12_spec  = scores_12[dm1_spec_train]
    scores_ctrl_12_spec = scores_12[ctrl_spec_train]
    scores_dm1_3_spec   = scores_3[dm1_3_spec]

    align = compute_alignment_metrics(scores_dm1_3_spec, scores_dm1_12_spec)
    sep   = compute_separation_metrics(scores_dm1_12_spec, scores_dm1_3_spec, scores_ctrl_12_spec)

    # Pseudo-meta repeated ×9 so plot_enhanced_pca_scatter can apply masks correctly
    meta_train_spec = meta_train.loc[np.repeat(np.arange(n_pat_train), 9)].reset_index(drop=True)
    meta_test_spec  = meta_test.loc[np.repeat(np.arange(n_pat_test),  9)].reset_index(drop=True)

    plot_enhanced_pca_scatter(
        out_dir=out_dir,
        title=f"{run_name}__{concept_name}__ALL_SPECTRA",
        scores_box12=scores_12, scores_box3=scores_3,
        meta_box12=meta_train_spec, meta_box3=meta_test_spec,
        si_col=si_col, dx_col=dx_col, evr=evr,
        alignment_metrics=align, separation_metrics=sep,
    )

    return dict(
        pc1_var=evr[0], pc2_var=evr[1],
        n_box12_dm1=int(dm1_mask_train_pat.sum()),
        n_box3_dm1 =int(dm1_3_pat.sum()),
        n_controls  =int(ctrl_mask_train_pat.sum()),
        **align, **sep,
    )


# =========================================================================
# COMPOSITE SCORE  (computed post-hoc over the full sweep)
# =========================================================================

def add_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-normalise each contributing metric then compute a weighted composite.

    Rank convention: rank 1 = worst config, rank N = best.
    Normalised rank = rank / N, so values lie in (0, 1].

    Formula (exactly 50 % alignment / 50 % separation):

      ALIGNMENT (50 %):
        0.20 × rank_hi(frac_box3_inside_dm1_95ellipse)
        0.15 × rank_lo(mahalanobis_mean_box3_to_box12dm1)   ← lower is better
        0.15 × rank_lo(mean_nn_dist_box3_to_box12dm1)       ← lower is better

      SEPARATION (50 %):
        0.20 × rank_hi(bhattacharyya_dm1_vs_ctrl)
        0.15 × rank_hi(silhouette_dm1_vs_ctrl)
        0.15 × rank_hi(frac_controls_outside_dm1_95ellipse)

    rank_hi: higher metric value → higher normalised rank.
    rank_lo: lower metric value → higher normalised rank.
    NaN values are pushed to the bottom (na_option='bottom').
    """
    n = len(df)
    if n == 0:
        df["composite_score"] = np.nan
        return df

    def rank_hi(col: str) -> pd.Series:
        return df[col].rank(ascending=True,  na_option="bottom") / n

    def rank_lo(col: str) -> pd.Series:
        return df[col].rank(ascending=False, na_option="bottom") / n

    df["composite_score"] = (
        # alignment
        0.20 * rank_hi("frac_box3_inside_dm1_95ellipse")
        + 0.15 * rank_lo("mahalanobis_mean_box3_to_box12dm1")
        + 0.15 * rank_lo("mean_nn_dist_box3_to_box12dm1")
        # separation
        + 0.20 * rank_hi("bhattacharyya_dm1_vs_ctrl")
        + 0.15 * rank_hi("silhouette_dm1_vs_ctrl")
        + 0.15 * rank_hi("frac_controls_outside_dm1_95ellipse")
    )
    return df


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    wn_train, X_avg_train, X_all_train, meta_train = load_data(
        data_dir=BOX12_DATA_DIR,
        metadata_path=BOX12_META_CSV,
        include_types=["DM1", "Control"],
        strict=False,
    )
    meta_train["Box"] = "Box1-2"

    wn_test, X_avg_test, X_all_test, meta_test = load_data(
        data_dir=BOX3_DATA_DIR,
        metadata_path=BOX3_META_CSV,
        include_types=["DM1", "Control"],
        strict=False,
    )
    meta_test["Box"] = "Box3"

    if not np.allclose(wn_train, wn_test):
        raise ValueError("Wavenumber axes differ between Box1-2 and Box3.")

    wavenumbers_full = wn_train
    si_col  = detect_si_column(meta_train)
    dx_col  = detect_dx_column(meta_train)
    dm1_mask_train, ctrl_mask_train = get_dm1_control_masks(
        meta_train, dx_col, require_controls=True
    )

    print(f"Box1-2: {dm1_mask_train.sum()} DM1 patients, "
          f"{ctrl_mask_train.sum()} controls")
    print(f"Box3:   {len(meta_test)} total patients loaded")

    ALL_COMBOS     = build_all_ordered_combos(SUPPORTED_METHODS)
    BASELINE_FLAGS = [False]

    score_rows: List[Dict] = []

    for mode_name, trim_bounds in SPECTRUM_MODES:

        if trim_bounds is None:
            wn_use   = wavenumbers_full
            Xat, Xae = X_avg_train, X_avg_test
            Xlt, Xle = X_all_train, X_all_test
        else:
            lo, hi = trim_bounds
            wn_use, Xat = trim_region(wavenumbers_full, X_avg_train, lo, hi)
            _,      Xae = trim_region(wavenumbers_full, X_avg_test,  lo, hi)
            _,      Xlt = trim_region(wavenumbers_full, X_all_train, lo, hi)
            _,      Xle = trim_region(wavenumbers_full, X_all_test,  lo, hi)

        for concept in CONCEPTS:

            concept_root = OUT_ROOT / mode_name / concept
            out_avg = concept_root / "PATIENT_AVG"
            out_all = concept_root / "ALL_SPECTRA"
            out_avg.mkdir(parents=True, exist_ok=True)
            out_all.mkdir(parents=True, exist_ok=True)

            for do_baseline in BASELINE_FLAGS:
                for methods in ALL_COMBOS:

                    run_name = safe_name(methods, do_baseline=do_baseline)

                    run_dir_avg = out_avg / run_name
                    run_dir_all = out_all / run_name
                    run_dir_avg.mkdir(parents=True, exist_ok=True)
                    run_dir_all.mkdir(parents=True, exist_ok=True)

                    Xat_pp = apply_pipeline(Xat, methods, do_baseline, BASELINE_KWARGS)
                    Xae_pp = apply_pipeline(Xae, methods, do_baseline, BASELINE_KWARGS)
                    Xlt_pp = apply_pipeline(Xlt, methods, do_baseline, BASELINE_KWARGS)
                    Xle_pp = apply_pipeline(Xle, methods, do_baseline, BASELINE_KWARGS)

                    # --- PATIENT AVG ---
                    m_avg = run_patient_avg_alignment(
                        out_dir=run_dir_avg, run_name=run_name, concept_name=concept,
                        X_avg_train=Xat_pp, X_avg_test=Xae_pp,
                        meta_train=meta_train, meta_test=meta_test,
                        si_col=si_col, dx_col=dx_col,
                        dm1_mask_train=dm1_mask_train,
                        ctrl_mask_train=ctrl_mask_train,
                    )
                    score_rows.append(dict(
                        spectrum_mode=mode_name, concept=concept,
                        preprocessing=run_name, plot_type="PATIENT_AVG",
                        **m_avg,
                    ))

                    # --- ALL SPECTRA ---
                    m_all = run_all_spectra_alignment(
                        out_dir=run_dir_all, run_name=run_name, concept_name=concept,
                        X_all_train=Xlt_pp, X_all_test=Xle_pp,
                        meta_train=meta_train, meta_test=meta_test,
                        si_col=si_col, dx_col=dx_col,
                        dm1_mask_train_pat=dm1_mask_train,
                        ctrl_mask_train_pat=ctrl_mask_train,
                    )
                    score_rows.append(dict(
                        spectrum_mode=mode_name, concept=concept,
                        preprocessing=run_name, plot_type="ALL_SPECTRA",
                        **m_all,
                    ))

                    print(f"[DONE] {mode_name} | {concept} | {run_name}")

    # =========================================================================
    # Build and save leaderboards
    # =========================================================================

    if not score_rows:
        print("No results collected — check data paths.")
    else:
        # Canonical column order for all output CSVs
        col_order = [
            "spectrum_mode", "concept", "preprocessing", "plot_type",
            "pc1_var", "pc2_var",
            "n_box12_dm1", "n_box3_dm1", "n_controls",
            # --- alignment ---
            "centroid_dist_box3_to_box12dm1",
            "mahalanobis_centroid_box3_to_box12dm1",
            "mahalanobis_mean_box3_to_box12dm1",
            "mahalanobis_median_box3_to_box12dm1",
            "mahalanobis_max_box3_to_box12dm1",
            "frac_box3_inside_dm1_95ellipse",
            "frac_box3_inside_dm1_hull",
            "mean_nn_dist_box3_to_box12dm1",
            "frac_box12dm1_inside_own_95ellipse",   # self-consistency check
            # --- separation ---
            "control_dm1_centroid_dist",
            "bhattacharyya_dm1_vs_ctrl",
            "fisher_separation_dm1_vs_ctrl",
            "silhouette_dm1_vs_ctrl",
            "frac_controls_outside_dm1_95ellipse",
        ]

        df = pd.DataFrame(score_rows)
        df = df.reindex(columns=[c for c in col_order if c in df.columns])
        df = add_composite_score(df)

        # 1. Best Box3-DM1 alignment (primary: fraction inside 95% ellipse)
        df_align = df.sort_values("frac_box3_inside_dm1_95ellipse", ascending=False)
        df_align.to_csv(OUT_ROOT / "box3_alignment_leaderboard.csv", index=False)

        # 2. Best control–DM1 separation (primary: Bhattacharyya distance)
        df_sep = df.sort_values("bhattacharyya_dm1_vs_ctrl", ascending=False)
        df_sep.to_csv(OUT_ROOT / "control_separation_leaderboard.csv", index=False)

        # 3. Best combined composite (50 % alignment / 50 % separation)
        df_combined = df.sort_values("composite_score", ascending=False)
        df_combined.to_csv(
            OUT_ROOT / "combined_alignment_separation_leaderboard.csv", index=False
        )

        print(f"\nLeaderboards saved to: {OUT_ROOT}")

        preview_cols = [
            "spectrum_mode", "concept", "preprocessing", "plot_type",
            "frac_box3_inside_dm1_95ellipse",
            "mahalanobis_mean_box3_to_box12dm1",
            "bhattacharyya_dm1_vs_ctrl",
            "silhouette_dm1_vs_ctrl",
            "composite_score",
        ]
        print("\n--- Top 10 by composite score ---")
        print(
            df_combined[[c for c in preview_cols if c in df_combined.columns]]
            .head(10)
            .to_string(index=False)
        )

    print("\nDONE.")
