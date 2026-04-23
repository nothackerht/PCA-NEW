#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_utils.py
============
Lightweight, stateless utility helpers for the SERS DM1 PCA suite.
No heavy scientific logic here — just string handling, mask detection,
combo generation, and spectral-window trimming.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pca_config import SUPPORTED_METHODS


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def sanitize_filename(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def safe_name(methods: List[str], do_baseline: bool) -> str:
    base = "+".join(m.upper() for m in methods) if methods else "RAW"
    return f"BASELINE__{base}" if do_baseline else base


# ---------------------------------------------------------------------------
# Metadata column detection
# ---------------------------------------------------------------------------

def detect_si_column(meta: pd.DataFrame) -> str:
    for c in ["target_SI", "SI", "SplicingIndex", "Splicing_Index"]:
        if c in meta.columns:
            return c
    for c in meta.columns:
        if "si" in c.lower():
            return c
    raise KeyError(f"SI column not found. Columns: {meta.columns.tolist()}")


def detect_dx_column(meta: pd.DataFrame) -> str:
    if "Type" in meta.columns:
        return "Type"
    for c in meta.columns:
        if c.lower() in ["dx", "diagnosis", "group", "class", "type"]:
            return c
    raise KeyError(f"Diagnosis column not found. Columns: {meta.columns.tolist()}")


def get_dm1_control_masks(
    meta: pd.DataFrame, dx_col: str, require_controls: bool
) -> Tuple[np.ndarray, np.ndarray]:
    dx = meta[dx_col].astype(str).str.strip().str.lower()
    dm1_mask  = dx.str.contains("dm1").to_numpy()
    ctrl_mask = (dx.str.contains("control") | dx.str.contains("adco")).to_numpy()
    if require_controls and not ctrl_mask.any():
        raise ValueError(f"No Control rows in column '{dx_col}'.")
    if not dm1_mask.any():
        raise ValueError(f"No DM1 rows in column '{dx_col}'.")
    return dm1_mask, ctrl_mask


# ---------------------------------------------------------------------------
# Preprocessing combo generation
# ---------------------------------------------------------------------------

def build_all_ordered_combos(methods: List[str]) -> List[List[str]]:
    """Return all preprocessing combos (including RAW = empty list)."""
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
    seen: set = set()
    uniq: List[List[str]] = []
    for c in combos:
        key = tuple(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq


def _run_name_to_methods(run_name: str) -> List[str]:
    """Reverse-parse a run_name string back to the methods list."""
    name = run_name
    if name.startswith("BASELINE__"):
        name = name[len("BASELINE__"):]
    if name == "RAW":
        return []
    parts = [p.strip() for p in name.split("+")]
    lookup = {m.upper(): m for m in SUPPORTED_METHODS}
    lookup["SECOND_DERIVATIVE"] = "Second Derivative"
    lookup["SECOND DERIVATIVE"] = "Second Derivative"
    return [lookup.get(p, p) for p in parts]


# ---------------------------------------------------------------------------
# Spectral window trimming
# ---------------------------------------------------------------------------

def trim_region(
    wavenumbers: np.ndarray, X: np.ndarray, lo: float, hi: float
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (wavenumbers >= lo) & (wavenumbers <= hi)
    return wavenumbers[mask], X[mask, :]


# ---------------------------------------------------------------------------
# Edge-fraction helper  (used for artifact-detection columns in master table)
# ---------------------------------------------------------------------------

def compute_edge_fraction(wavenumbers: np.ndarray, top_wn: np.ndarray) -> float:
    """
    Fraction of top_wn values that fall in the outer 5% of the spectral range.
    High values suggest the loading is dominated by edge artefacts.
    """
    if len(top_wn) == 0:
        return np.nan
    wn_min, wn_max = float(wavenumbers.min()), float(wavenumbers.max())
    edge_size = 0.05 * (wn_max - wn_min)
    in_edge = (top_wn < wn_min + edge_size) | (top_wn > wn_max - edge_size)
    return float(in_edge.mean())
