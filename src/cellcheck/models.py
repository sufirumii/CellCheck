"""A linear model on summed gene embeddings, plus controls with no gene knowledge."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV

ALPHAS = np.logspace(-3, 5, 17)


def feature(perturbation: str, table: dict, sep: str = "+") -> np.ndarray:
    """Represent a perturbation as the sum of the vectors of the genes it touches."""
    return sum(table[g] for g in perturbation.split(sep))


def fit_predict(train, test, table, y_train, sep: str = "+"):
    """Fit ridge regression on the training perturbations and predict the test ones."""
    x_train = np.vstack([feature(p, table, sep) for p in train])
    x_test = np.vstack([feature(p, table, sep) for p in test])
    ridge = RidgeCV(alphas=ALPHAS).fit(x_train, y_train)
    return ridge.predict(x_test), float(ridge.alpha_)


def shuffled_table(table: dict, genes: list, rng) -> dict:
    """Give every gene the vector that belongs to a different gene.

    The set of vectors is identical to the real one, so only the link between a
    gene and its vector is destroyed.
    """
    perm = rng.permutation(len(genes))
    return {g: table[genes[perm[i]]] for i, g in enumerate(genes)}


def random_table(genes: list, dim: int, scale: float, rng) -> dict:
    """Random Gaussian vectors with the same overall scale as the real ones."""
    return {g: rng.normal(size=dim) * scale for g in genes}


def one_hot_table(genes: list) -> dict:
    """One indicator dimension per gene: a linear model with no biology at all."""
    eye = np.eye(len(genes))
    return {g: eye[i] for i, g in enumerate(genes)}
