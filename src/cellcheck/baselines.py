"""Simple reference methods that every model has to beat."""

from __future__ import annotations

import numpy as np


def no_change(n_genes: int) -> np.ndarray:
    """Predict that a perturbation changes nothing relative to control."""
    return np.zeros(n_genes)


def additive(single_a: np.ndarray, single_b: np.ndarray) -> np.ndarray:
    """Predict a double perturbation as the sum of the two single effects.

    Inputs are changes relative to control.
    """
    return single_a + single_b


def fit_scale(additive_train: np.ndarray, truth_train: np.ndarray) -> float:
    """Single number that best shrinks (or stretches) the additive prediction.

    Fitted by least squares on training doubles only.
    """
    return float((additive_train * truth_train).sum() / (additive_train**2).sum())
