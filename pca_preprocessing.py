#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pca_preprocessing.py
====================
Thin wrapper around preprocessing.py that enforces train/test separation.
EMSC reference is always derived from training data only to prevent leakage.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from preprocessing import Preprocessing


def apply_pipeline_train_test(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    methods: List[str],
    do_baseline: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit preprocessing on X_train only, then transform both matrices.
    X shapes: (features, N).

    Raises NotImplementedError for do_baseline=True (not implemented in
    the current preprocessing.py — keep BASELINE_FLAGS = [False]).
    """
    if do_baseline:
        raise NotImplementedError(
            "Baseline correction is not implemented. Keep BASELINE_FLAGS = [False]."
        )
    if not methods:
        return X_train.copy(), X_test.copy()
    pp = Preprocessing()
    pp.fit(X_train, methods)
    return pp.transform(X_train, methods), pp.transform(X_test, methods)
