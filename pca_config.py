#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_config.py
=============
All configuration constants for the SERS DM1 PCA suite.
Edit the six path constants below before running.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

from scipy.stats import chi2

# =============================================================================
# PATHS  —  update before running
# =============================================================================

OUT_ROOT = Path(r"C:\Users\spect\Desktop\PCA NEW\PCA All results")

BOX12_DATA_DIR = r"C:\Users\spect\Desktop\SERS_Cluster_work\data"
BOX12_META_CSV = r"C:\Users\spect\Desktop\SERS_Cluster_work\y_metadata.csv"

BOX3_DATA_DIR  = r"C:\Users\spect\Desktop\SERS_Cluster_work\data_test_updated"
BOX3_META_CSV  = r"C:\Users\spect\Desktop\SERS_Cluster_work\y_metadata_test_updated_in_order.csv"

# =============================================================================
# SWEEP CONSTANTS
# =============================================================================

CONCEPTS: List[str] = [
    "WEIGHTS__BOX12_DM1_ONLY",   # PCA axes from DM1-internal variance  → Q2, Q5, Q6
    "WEIGHTS__BOX12_ALL",         # PCA axes from DM1+control variance   → Q3A, Q3B, Q4
]

SPECTRUM_MODES: List[Tuple[str, Optional[Tuple[float, float]]]] = [
    ("FULL_SPECTRUM",  None),
    ("TRIM_500_3100",  (500.0,  3100.0)),
    ("TRIM_700_2800",  (700.0,  2800.0)),
]

SUPPORTED_METHODS: List[str] = ["Normalization", "SNV", "Second Derivative", "EMSC"]

# Baseline correction not implemented; keep as [False]
BASELINE_FLAGS: List[bool] = [False]

# =============================================================================
# PLOT / EXPORT SETTINGS
# =============================================================================

TOP_K_WAVENUMBERS: int   = 25
GAP_THRESH:        float = 25.0   # cm⁻¹ gap that triggers a line break in overlays
FIGURE_DPI:        int   = 200
JOURNAL_DPI:       int   = 300

# 95% confidence ellipse threshold for 2-D chi-squared distribution
CHI2_95: float = chi2.ppf(0.95, df=2)   # ≈ 5.991

# =============================================================================
# COLOUR / MARKER SCHEME
# =============================================================================

COLOR_CTRL       = "orange"
COLOR_BOX12_DM1  = None          # viridis SI gradient
COLOR_BOX3_DM1   = None          # viridis SI gradient, black edge
MARKER_BOX12     = "o"
MARKER_BOX3      = "s"
SIZE_BOX12       = 18
SIZE_BOX3        = 50

GCOLOR_BOX12_DM1 = "steelblue"
GCOLOR_BOX3_DM1  = "crimson"
GCOLOR_CTRL      = "orange"

# =============================================================================
# QUESTION DEFINITIONS  (used for audit logs and README)
# =============================================================================

QUESTION_DEFINITIONS = {
    "Q1": "Native raw spectral geometry — no preprocessing, baseline reference.",
    "Q2": "Cohort 2 DM1 alignment / overlap with Cohort 1 DM1 — concept=WEIGHTS__BOX12_DM1_ONLY.",
    "Q3A": "Clean Controls vs Cohort 1 DM1 separation — concept=WEIGHTS__BOX12_ALL, Cohort 1 only.",
    "Q3B": "Holistic Controls vs all DM1 separation — concept=WEIGHTS__BOX12_ALL, Cohort 1 + Cohort 2 DM1.",
    "Q4": "Cohort shift within disease/control geometry — concept=WEIGHTS__BOX12_ALL, alignment metrics.",
    "Q5": "Severity gradient within DM1 — concept=WEIGHTS__BOX12_DM1_ONLY, Spearman rho.",
    "Q6": "Exploratory preprocessing-induced cohort separation — supplementary.",
}
