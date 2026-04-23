# """
# run_pca_highlight_boxes123.py

# PCA visualizations for SERS Box1-2 (train) and Box3 (test), with:
# - Two PCA "concepts" (where PCA weights are learned from):
#     1) WEIGHTS__BOX12_DM1_ONLY  -> fit PCA on Box1-2 DM1 only; then project Box1-2 controls + Box3 DM1
#     2) WEIGHTS__BOX12_ALL       -> fit PCA on all Box1-2 (DM1+controls); then project Box3 DM1

# - Two plot modes:
#     PATIENT_AVG  -> each patient is one dot (averaged spectra)
#     ALL_SPECTRA  -> every spectrum is a dot (9 per patient), grouped coloring

# - Multiple spectral windows:
#     FULL_SPECTRUM
#     TRIM_500_3100
#     TRIM_700_2800

# - Preprocessing sweeps:
#     - RAW + preprocessing combos
#     - BASELINE-CORRECTED RAW
#     - BASELINE + preprocessing combos

# - Overlay plots:
#     PATIENT_AVG overlay uses averaged spectra (NOT all spectra)
#     ALL_SPECTRA overlay uses all spectra

# Outputs saved to:
#     C:\Users\spect\Desktop\SERS CODE USED FOR PAPER\PCA\PCA RESULTS
# """

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from scipy.stats import spearmanr

# local modules (must be in same folder)
from data_loader import load_data
from preprocessing import Preprocessing


# =============================
# CONFIG
# =============================

OUT_ROOT = Path(r"C:\Users\spect\Desktop\SERS CODE USED FOR PAPER\PCA\PCA RESULTS")

BOX12_DATA_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data"
BOX12_META_CSV = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata.csv"

BOX3_DATA_DIR = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\data_test_updated"
BOX3_META_CSV = r"C:\Users\spect\Desktop\MD-Analysis-main (3)\SERS_ML_Desktop\Modules for paper\data\y_metadata_test_updated_in_order.csv"

# PCA concepts
CONCEPTS = [
    "WEIGHTS__BOX12_DM1_ONLY",
    "WEIGHTS__BOX12_ALL",
]

# spectral windows
SPECTRUM_MODES = [
    ("FULL_SPECTRUM", None),
    ("TRIM_500_3100", (500.0, 3100.0)),
    ("TRIM_700_2800", (700.0, 2800.0)),
]

# preprocessing names that exist in preprocessing.py
SUPPORTED_METHODS = [
    "Normalization",
    "SNV",
    "Second Derivative",
    "EMSC",
]

# baseline correction settings (used by preprocessing module if you implement baseline there)
BASELINE_KWARGS = dict(lam=1e6, p=0.01, niter=10)

# plotting controls
TOP_K_WAVENUMBERS = 25
GAP_THRESH = 25.0  # if wavenumber jump > this, break line to avoid "connecting" windows

CONTROL_COLOR = "orange"  # outside gradient
BOX12_MARKER = "o"
BOX3_MARKER = "s"
BOX12_SIZE = 18
BOX3_SIZE = 50


# =============================
# UTILITIES
# =============================
import re

def sanitize_filename(s: str) -> str:
    # Replace forbidden Windows filename characters with underscores
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    # collapse repeated underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s
def detect_si_column(meta: pd.DataFrame) -> str:
    # Most common in your project is 'target_SI' but keep flexible
    candidates = ["target_SI", "SI", "SplicingIndex", "Splicing_Index"]
    for c in candidates:
        if c in meta.columns:
            return c
    # fallback: first column containing "si" and numeric
    for c in meta.columns:
        if "si" in c.lower():
            return c
    raise KeyError(f"Could not detect SI column in metadata. Columns={meta.columns.tolist()}")


def detect_dx_column(meta: pd.DataFrame) -> str:
    # Your metadata uses 'Type' (DM1 vs Control)
    if "Type" in meta.columns:
        return "Type"
    # fallback
    for c in meta.columns:
        if c.lower() in ["dx", "diagnosis", "group", "class", "type"]:
            return c
    raise KeyError(f"Could not detect diagnosis column in metadata. Columns={meta.columns.tolist()}")


def get_dm1_control_masks(meta: pd.DataFrame, dx_col: str, require_controls: bool) -> Tuple[np.ndarray, np.ndarray]:
    dx = meta[dx_col].astype(str).str.strip().str.lower()
    dm1_mask = dx.str.contains("dm1").to_numpy()

    # controls might be labeled "Control" or "AdCo"
    ctrl_mask = (dx.str.contains("control") | dx.str.contains("adco")).to_numpy()

    if require_controls and not ctrl_mask.any():
        raise ValueError(f"No Control rows detected in metadata using column '{dx_col}'.")

    if not dm1_mask.any():
        raise ValueError(f"No DM1 rows detected in metadata using column '{dx_col}'.")

    return dm1_mask, ctrl_mask


def safe_name(methods: List[str], do_baseline: bool) -> str:
    if len(methods) == 0:
        base = "RAW"
    else:
        base = "+".join([m.upper() for m in methods])

    if do_baseline:
        return f"BASELINE__{base}"
    return base


def build_all_ordered_combos(methods: List[str]) -> List[List[str]]:
    """
    Returns an ordered list of preprocessing pipelines including RAW ([]).
    We keep ordering stable and include singletons + some multi combos.
    """
    combos: List[List[str]] = [[]]

    # singletons
    for m in methods:
        combos.append([m])

    # common multi-combos you’ve been using
    if "Normalization" in methods and "EMSC" in methods:
        combos.append(["Normalization", "EMSC"])
    if "SNV" in methods and "SecondDerivative" in methods:
        combos.append(["SNV", "SecondDerivative"])
    if "Normalization" in methods and "SNV" in methods:
        combos.append(["Normalization", "SNV"])
    if "Normalization" in methods and "SNV" in methods and "SecondDerivative" in methods:
        combos.append(["Normalization", "SNV", "SecondDerivative"])

    # remove duplicates while preserving order
    uniq: List[List[str]] = []
    seen = set()
    for c in combos:
        key = tuple(c)
        if key not in seen:
            uniq.append(c)
            seen.add(key)

    return uniq


def trim_region(wavenumbers: np.ndarray, X: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (wavenumbers >= lo) & (wavenumbers <= hi)
    return wavenumbers[mask], X[mask, :]


def apply_pipeline(X: np.ndarray, methods: List[str], do_baseline: bool, baseline_kwargs: dict) -> np.ndarray:
    """
    Uses your preprocessing module.
    Assumes Preprocessing(X).preprocess(list_of_methods) exists.
    For baseline, assumes preprocessing module supports a "Baseline" step via a method,
    OR we apply baseline in a separate way if you implemented it there.
    """
    X_work = X.copy()

    # baseline-correct first if requested
    if do_baseline:
        # We assume your preprocessing module has baseline correction implemented
        # as a method called "Baseline". If not, replace this block with your
        # baseline function call.
        pp0 = Preprocessing(X_work)
        try:
            X_work = pp0.preprocess(["Baseline"], baseline_kwargs=baseline_kwargs)
        except TypeError:
            # if your preprocess() doesn't accept baseline_kwargs, try without
            X_work = pp0.preprocess(["Baseline"])

    # then apply the rest
    if len(methods) > 0:
        pp = Preprocessing(X_work)
        X_work = pp.preprocess(methods)

    return X_work


def compute_explained_variance(pca: PCA) -> Tuple[float, float]:
    evr = pca.explained_variance_ratio_
    pc1 = float(evr[0]) if len(evr) > 0 else np.nan
    pc2 = float(evr[1]) if len(evr) > 1 else np.nan
    return pc1, pc2


def break_by_gap(x: np.ndarray, gap_thresh: float) -> List[np.ndarray]:
    """
    Returns list of index arrays for continuous segments.
    """
    dx = np.diff(x)
    breaks = np.where(dx > gap_thresh)[0]
    if breaks.size == 0:
        return [np.arange(len(x))]

    segments = []
    start = 0
    for b in breaks:
        end = b + 1
        segments.append(np.arange(start, end))
        start = end
    segments.append(np.arange(start, len(x)))
    return segments


def plot_overlay(
    out_dir: Path,
    title: str,
    wavenumbers: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    wn_pc1: np.ndarray,
    wn_pc2: np.ndarray,
    gap_thresh: float = 25.0,
):
    """
    Overlay spectra and highlight top PC1 (blue) / PC2 (red) wavenumbers.
    IMPORTANT:
      - For PATIENT_AVG mode, pass averaged spectra matrices
      - For ALL_SPECTRA mode, pass all spectra matrices
    """
    fig = plt.figure(figsize=(14, 4.5))
    ax = plt.gca()

    segs = break_by_gap(wavenumbers, gap_thresh=gap_thresh)

    # plot train + test lightly
    for j in range(X_train.shape[1]):
        for seg in segs:
            ax.plot(wavenumbers[seg], X_train[seg, j], alpha=0.08, linewidth=0.8)
    for j in range(X_test.shape[1]):
        for seg in segs:
            ax.plot(wavenumbers[seg], X_test[seg, j], alpha=0.08, linewidth=0.8)

    # highlight bands
    for w in wn_pc1:
        ax.axvline(float(w), color="blue", alpha=0.15)
    for w in wn_pc2:
        ax.axvline(float(w), color="red", alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Intensity")
    fig.tight_layout()
    fname = sanitize_filename(title.replace(" ", "_")) + ".png"
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)


def compute_top_wavenumbers(wavenumbers: np.ndarray, pc_vec: np.ndarray, top_k: int) -> np.ndarray:
    idx = np.argsort(np.abs(pc_vec))[::-1][:top_k]
    return wavenumbers[idx]


def plot_loadings(out_dir: Path, title: str, wavenumbers: np.ndarray, pc1: np.ndarray, pc2: np.ndarray, gap_thresh: float):
    fig = plt.figure(figsize=(14, 4.5))
    ax = plt.gca()
    segs = break_by_gap(wavenumbers, gap_thresh=gap_thresh)

    for seg in segs:
        ax.plot(wavenumbers[seg], pc1[seg], label="PC1 loading", linewidth=1.2)
        ax.plot(wavenumbers[seg], pc2[seg], label="PC2 loading", linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel("Loading")
    ax.legend()
    fig.tight_layout()
    fname = sanitize_filename(title.replace(" ", "_")) + ".png"
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)


def plot_pca_scatter(
    out_dir: Path,
    title: str,
    scores_box12: np.ndarray,
    scores_box3: np.ndarray,
    meta_box12: pd.DataFrame,
    meta_box3: pd.DataFrame,
    si_col: str,
    dx_col: str,
    evr: Tuple[float, float],
):
    """
    Scatter:
      - Controls: orange
      - DM1: SI gradient
      - Box3 plotted as squares, Box1-2 as circles
    """
    pc1_var, pc2_var = evr
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()

    # masks
    dm1_12, ctrl_12 = get_dm1_control_masks(meta_box12, dx_col, require_controls=True)
    dm1_3, ctrl_3 = get_dm1_control_masks(meta_box3, dx_col, require_controls=False)  # <-- allow none
    # ctrl_3 may be all False if none exist

    si_12 = pd.to_numeric(meta_box12[si_col], errors="coerce").to_numpy(dtype=float)
    si_3 = pd.to_numeric(meta_box3[si_col], errors="coerce").to_numpy(dtype=float)

    # controls first (box1-2)
    ax.scatter(
        scores_box12[ctrl_12, 0],
        scores_box12[ctrl_12, 1],
        c=CONTROL_COLOR,
        marker=BOX12_MARKER,
        s=BOX12_SIZE,
        alpha=0.85,
        label="Box1-2 Control",
        edgecolors="none",
    )

    # DM1 (box1-2) gradient
    sc1 = ax.scatter(
        scores_box12[dm1_12, 0],
        scores_box12[dm1_12, 1],
        c=si_12[dm1_12],
        marker=BOX12_MARKER,
        s=BOX12_SIZE,
        alpha=0.85,
        label="Box1-2 DM1",
        edgecolors="none",
    )

    # Box3 DM1 gradient (squares)
    ax.scatter(
        scores_box3[dm1_3, 0],
        scores_box3[dm1_3, 1],
        c=si_3[dm1_3],
        marker=BOX3_MARKER,
        s=BOX3_SIZE,
        alpha=0.95,
        label="Box3 DM1",
        edgecolors="k",
        linewidths=0.3,
    )

    cbar = plt.colorbar(sc1, ax=ax)
    cbar.set_label("SI (DM1 severity)")

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pc1_var*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var*100:.1f}%)")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fname = sanitize_filename(title.replace(" ", "_")) + ".png"
    fig.savefig(out_dir / fname, dpi=200)
    plt.close(fig)


def save_top_wavenumbers_csv(out_dir: Path, title: str, wn1: np.ndarray, wn2: np.ndarray):
    df = pd.DataFrame({"top_PC1_wavenumbers": wn1, "top_PC2_wavenumbers": wn2})
    df.to_csv(out_dir / f"{title}__top_wavenumbers.csv", index=False)


def gradient_score_from_scores(scores_dm1: np.ndarray, si_dm1: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns:
      rho_pc1, rho_pc2, grad_score (max abs rho)
    """
    rho1 = spearmanr(scores_dm1[:, 0], si_dm1).correlation
    rho2 = spearmanr(scores_dm1[:, 1], si_dm1).correlation
    # guard
    rho1 = float(rho1) if rho1 is not None else np.nan
    rho2 = float(rho2) if rho2 is not None else np.nan
    grad = np.nanmax([abs(rho1), abs(rho2)])
    return rho1, rho2, float(grad)


def cv_r2_pc12_to_si(scores_dm1: np.ndarray, si_dm1: np.ndarray) -> float:
    """
    Simple linear R^2 of SI ~ [PC1, PC2] as a sanity measure.
    """
    X = scores_dm1[:, :2]
    y = si_dm1
    ok = np.isfinite(y)
    X = X[ok]
    y = y[ok]
    if len(y) < 5:
        return np.nan
    lr = LinearRegression().fit(X, y)
    return float(lr.score(X, y))


# =============================
# PCA RUNNERS
# =============================

def run_patient_avg_pca(
    out_dir: Path,
    run_name: str,
    concept_name: str,
    wavenumbers: np.ndarray,
    X_avg_train: np.ndarray,
    X_avg_test: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    si_col: str,
    dx_col: str,
    dm1_mask_train: np.ndarray,
    ctrl_mask_train: np.ndarray,
) -> Tuple[PCA, Tuple[float, float], Dict[str, float]]:
    """
    Fit PCA according to concept, project both Box1-2 and Box3.
    """
    # Build training matrix for PCA weights
    if concept_name == "WEIGHTS__BOX12_DM1_ONLY":
        X_fit = X_avg_train[:, dm1_mask_train].T
    elif concept_name == "WEIGHTS__BOX12_ALL":
        X_fit = X_avg_train.T
    else:
        raise ValueError(f"Unknown concept: {concept_name}")

    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_fit)

    scores_12 = pca.transform(X_avg_train.T)
    scores_3 = pca.transform(X_avg_test.T)

    evr = compute_explained_variance(pca)

    # save scatter
    plot_pca_scatter(
        out_dir=out_dir,
        title=f"{run_name}__{concept_name}__PATIENT_AVG",
        scores_box12=scores_12,
        scores_box3=scores_3,
        meta_box12=meta_train,
        meta_box3=meta_test,
        si_col=si_col,
        dx_col=dx_col,
        evr=evr,
    )

    # severity gradient score (DM1 only, Box1-2 only for stability)
    si_train = pd.to_numeric(meta_train[si_col], errors="coerce").to_numpy(dtype=float)
    rho1, rho2, grad = gradient_score_from_scores(scores_12[dm1_mask_train], si_train[dm1_mask_train])
    r2 = cv_r2_pc12_to_si(scores_12[dm1_mask_train], si_train[dm1_mask_train])

    metrics = dict(rho_pc1=rho1, rho_pc2=rho2, grad_score=grad, cv_r2_pc12_to_si=r2)
    return pca, evr, metrics


def run_all_spectra_pca(
    out_dir: Path,
    run_name: str,
    concept_name: str,
    wavenumbers: np.ndarray,
    X_all_train: np.ndarray,
    X_all_test: np.ndarray,
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    si_col: str,
    dx_col: str,
    dm1_mask_train_pat: np.ndarray,
    ctrl_mask_train_pat: np.ndarray,
) -> Tuple[PCA, Tuple[float, float], Dict[str, float]]:
    """
    All spectra mode:
    - X_all_train is (wn, 9*N_pat_train)
    - We repeat patient masks 9x to label each spectrum correctly
    """
    n_pat_train = len(meta_train)
    n_pat_test = len(meta_test)

    # repeat masks to spectra-level
    dm1_spec_train = np.repeat(dm1_mask_train_pat, 9)
    ctrl_spec_train = np.repeat(ctrl_mask_train_pat, 9)

    # Box3 is DM1-only in practice, but don’t assume
    dm1_pat_test, ctrl_pat_test = get_dm1_control_masks(meta_test, dx_col, require_controls=False)
    dm1_spec_test = np.repeat(dm1_pat_test, 9)
    ctrl_spec_test = np.repeat(ctrl_pat_test, 9)

    # Build PCA-fit matrix
    if concept_name == "WEIGHTS__BOX12_DM1_ONLY":
        X_fit = X_all_train[:, dm1_spec_train].T
    elif concept_name == "WEIGHTS__BOX12_ALL":
        X_fit = X_all_train.T
    else:
        raise ValueError(f"Unknown concept: {concept_name}")

    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_fit)

    scores_12 = pca.transform(X_all_train.T)
    scores_3 = pca.transform(X_all_test.T)
    evr = compute_explained_variance(pca)

    # Create pseudo-meta for spectra: repeat each patient row 9x
    meta_train_spec = meta_train.loc[np.repeat(np.arange(n_pat_train), 9)].reset_index(drop=True)
    meta_test_spec = meta_test.loc[np.repeat(np.arange(n_pat_test), 9)].reset_index(drop=True)

    plot_pca_scatter(
        out_dir=out_dir,
        title=f"{run_name}__{concept_name}__ALL_SPECTRA",
        scores_box12=scores_12,
        scores_box3=scores_3,
        meta_box12=meta_train_spec,
        meta_box3=meta_test_spec,
        si_col=si_col,
        dx_col=dx_col,
        evr=evr,
    )

    # gradient score (DM1 spectra in Box1-2)
    si_train = pd.to_numeric(meta_train[si_col], errors="coerce").to_numpy(dtype=float)
    si_train_spec = np.repeat(si_train, 9)

    rho1, rho2, grad = gradient_score_from_scores(scores_12[dm1_spec_train], si_train_spec[dm1_spec_train])
    r2 = cv_r2_pc12_to_si(scores_12[dm1_spec_train], si_train_spec[dm1_spec_train])

    metrics = dict(rho_pc1=rho1, rho_pc2=rho2, grad_score=grad, cv_r2_pc12_to_si=r2)
    return pca, evr, metrics


def save_loading_and_overlay(
    out_dir: Path,
    run_name: str,
    wavenumbers: np.ndarray,
    pca_model: PCA,
    X_overlay_train: np.ndarray,
    X_overlay_test: np.ndarray,
    top_k: int = 25,
    gap_thresh: float = 25.0,
):
    loadings = pca_model.components_
    pc1, pc2 = loadings[0], loadings[1]

    wn1 = compute_top_wavenumbers(wavenumbers, pc1, top_k=top_k)
    wn2 = compute_top_wavenumbers(wavenumbers, pc2, top_k=top_k)

    save_top_wavenumbers_csv(out_dir, run_name, wn1, wn2)
    plot_loadings(out_dir, f"{run_name}__LOADINGS", wavenumbers, pc1, pc2, gap_thresh=gap_thresh)
    plot_overlay(out_dir, f"{run_name}__Overlay | top-{top_k} PC1 blue / PC2 red",
                 wavenumbers, X_overlay_train, X_overlay_test, wn1, wn2, gap_thresh=gap_thresh)


# =============================
# MAIN
# =============================

if __name__ == "__main__":

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # -------------------
    # Load Box1-2 (train)
    # -------------------
    wn_train, X_avg_train, X_all_train, meta_train = load_data(
        data_dir=BOX12_DATA_DIR,
        metadata_path=BOX12_META_CSV,
        include_types=["DM1", "Control"],
        strict=False,
    )
    meta_train["Box"] = "Box1-2"

    # --------------
    # Load Box3 test
    # --------------
    wn_test, X_avg_test, X_all_test, meta_test = load_data(
        data_dir=BOX3_DATA_DIR,
        metadata_path=BOX3_META_CSV,
        include_types=["DM1", "Control"],
        strict=False,
    )
    meta_test["Box"] = "Box3"

    if not np.allclose(wn_train, wn_test):
        raise ValueError("Wavenumbers differ between Box1-2 and Box3. Cannot run shared PCA axis reliably.")

    wavenumbers_full = wn_train

    # Identify SI + dx columns
    si_col = detect_si_column(meta_train)
    dx_col = detect_dx_column(meta_train)

    dm1_mask_train, ctrl_mask_train = get_dm1_control_masks(meta_train, dx_col, require_controls=True)

    # preprocessing combos
    ALL_COMBOS = build_all_ordered_combos(SUPPORTED_METHODS)

    # We'll run:
    # - do_baseline=False: RAW + combos
    # - do_baseline=True: baseline-corrected RAW + baseline+combos
    BASELINE_FLAGS = [False, True]

    # collect score rows for a global CSV summary (optional but useful)
    score_rows = []

    for mode_name, trim_bounds in SPECTRUM_MODES:

        mode_root = OUT_ROOT / mode_name
        mode_root.mkdir(parents=True, exist_ok=True)

        # Select wavenumbers + matrices
        if trim_bounds is None:
            wn_use = wavenumbers_full
            X_avg_train_mode = X_avg_train
            X_avg_test_mode = X_avg_test
            X_all_train_mode = X_all_train
            X_all_test_mode = X_all_test
        else:
            lo, hi = trim_bounds
            wn_use, X_avg_train_mode = trim_region(wavenumbers_full, X_avg_train, lo, hi)
            _, X_avg_test_mode = trim_region(wavenumbers_full, X_avg_test, lo, hi)
            _, X_all_train_mode = trim_region(wavenumbers_full, X_all_train, lo, hi)
            _, X_all_test_mode = trim_region(wavenumbers_full, X_all_test, lo, hi)

        for concept in CONCEPTS:

            concept_root = mode_root / concept
            concept_root.mkdir(parents=True, exist_ok=True)

            out_avg_root = concept_root / "PATIENT_AVG"
            out_all_root = concept_root / "ALL_SPECTRA"
            out_avg_root.mkdir(parents=True, exist_ok=True)
            out_all_root.mkdir(parents=True, exist_ok=True)

            for do_baseline in BASELINE_FLAGS:
                for methods in ALL_COMBOS:

                    run_name = safe_name(methods, do_baseline=do_baseline)

                    run_dir_avg = out_avg_root / run_name
                    run_dir_all = out_all_root / run_name
                    run_dir_avg.mkdir(parents=True, exist_ok=True)
                    run_dir_all.mkdir(parents=True, exist_ok=True)

                    # Apply preprocessing to ALL matrices
                    X_avg_train_pp = apply_pipeline(X_avg_train_mode, methods, do_baseline, BASELINE_KWARGS)
                    X_avg_test_pp = apply_pipeline(X_avg_test_mode, methods, do_baseline, BASELINE_KWARGS)
                    X_all_train_pp = apply_pipeline(X_all_train_mode, methods, do_baseline, BASELINE_KWARGS)
                    X_all_test_pp = apply_pipeline(X_all_test_mode, methods, do_baseline, BASELINE_KWARGS)

                    # ----- PATIENT AVG PCA -----
                    pca_avg, evr_avg, metrics_avg = run_patient_avg_pca(
                        out_dir=run_dir_avg,
                        run_name=run_name,
                        concept_name=concept,
                        wavenumbers=wn_use,
                        X_avg_train=X_avg_train_pp,
                        X_avg_test=X_avg_test_pp,
                        meta_train=meta_train,
                        meta_test=meta_test,
                        si_col=si_col,
                        dx_col=dx_col,
                        dm1_mask_train=dm1_mask_train,
                        ctrl_mask_train=ctrl_mask_train,
                    )

                    # IMPORTANT: overlay uses averaged spectra for PATIENT_AVG
                    save_loading_and_overlay(
                        out_dir=run_dir_avg,
                        run_name=f"{run_name}__PATIENT_AVG",
                        wavenumbers=wn_use,
                        pca_model=pca_avg,
                        X_overlay_train=X_avg_train_pp,
                        X_overlay_test=X_avg_test_pp,
                        top_k=TOP_K_WAVENUMBERS,
                        gap_thresh=GAP_THRESH,
                    )

                    # ----- ALL SPECTRA PCA -----
                    pca_all, evr_all, metrics_all = run_all_spectra_pca(
                        out_dir=run_dir_all,
                        run_name=run_name,
                        concept_name=concept,
                        wavenumbers=wn_use,
                        X_all_train=X_all_train_pp,
                        X_all_test=X_all_test_pp,
                        meta_train=meta_train,
                        meta_test=meta_test,
                        si_col=si_col,
                        dx_col=dx_col,
                        dm1_mask_train_pat=dm1_mask_train,
                        ctrl_mask_train_pat=ctrl_mask_train,
                    )

                    # overlay uses ALL spectra for ALL_SPECTRA
                    save_loading_and_overlay(
                        out_dir=run_dir_all,
                        run_name=f"{run_name}__ALL_SPECTRA",
                        wavenumbers=wn_use,
                        pca_model=pca_all,
                        X_overlay_train=X_all_train_pp,
                        X_overlay_test=X_all_test_pp,
                        top_k=TOP_K_WAVENUMBERS,
                        gap_thresh=GAP_THRESH,
                    )

                    # collect scoring summary (optional)
                    score_rows.append({
                        "spectrum_mode": mode_name,
                        "concept": concept,
                        "preprocessing": run_name,
                        "plot_type": "PATIENT_AVG",
                        "pc1_var": evr_avg[0],
                        "pc2_var": evr_avg[1],
                        "n_dm1": int(dm1_mask_train.sum()),
                        **metrics_avg,
                    })
                    score_rows.append({
                        "spectrum_mode": mode_name,
                        "concept": concept,
                        "preprocessing": run_name,
                        "plot_type": "ALL_SPECTRA",
                        "pc1_var": evr_all[0],
                        "pc2_var": evr_all[1],
                        "n_dm1": int(dm1_mask_train.sum() * 9),
                        **metrics_all,
                    })

                    print(f"[SAVED] {mode_name} | {concept} | {run_name}")

    # write a leaderboard CSV
    if len(score_rows) > 0:
        df_scores = pd.DataFrame(score_rows)
        df_scores = df_scores.sort_values(by=["grad_score"], ascending=False)
        df_scores.to_csv(OUT_ROOT / "severity_gradient_leaderboard.csv", index=False)

    print("DONE. Saved ALL PCA concepts, windows, baseline+nonbaseline, and preprocessing combos.")