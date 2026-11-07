"""A small synthetic dataset used to check that the pipeline works.

The numbers produced from this data are not results. They only show that every
step runs and that the code recovers a known property of the synthetic data:
double perturbations are generated at 75 percent of the sum of the singles, so
the learned scale of the additive rule should land clearly below 1 and near
0.75. With few cells per perturbation the estimate is pulled lower, because
the single-perturbation averages are noisy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from . import embeddings

TRUE_SCALE = 0.75


def make_synthetic(n_genes: int = 300, n_singles: int = 24, n_doubles: int = 40,
                   cells_per_group: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(n_genes)]
    perturbed = [f"P{i}" for i in range(n_singles)]

    effect = {}
    for p in perturbed:
        e = np.zeros(n_genes)
        hit = rng.choice(n_genes, size=20, replace=False)
        e[hit] = rng.normal(0, 0.8, size=20)
        effect[p] = e

    pairs = set()
    while len(pairs) < n_doubles:
        a, b = rng.choice(n_singles, size=2, replace=False)
        pairs.add(tuple(sorted((int(a), int(b)))))
    doubles = [f"P{a}+P{b}" for a, b in sorted(pairs)]

    base = rng.normal(2.0, 0.3, size=n_genes)
    rows, labels = [], []

    def add(label, mean, n):
        rows.append(mean + rng.normal(0, 0.5, size=(n, n_genes)))
        labels.extend([label] * n)

    add("control", base, cells_per_group * 5)
    for p in perturbed:
        add(p, base + effect[p], cells_per_group)
    for d in doubles:
        a, b = d.split("+")
        interaction = rng.normal(0, 0.05, size=n_genes)
        add(d, base + TRUE_SCALE * (effect[a] + effect[b]) + interaction, cells_per_group)

    x = np.vstack(rows).astype(np.float32)
    obs = pd.DataFrame({"perturbation_name": labels})
    obs.index = [f"cell{i}" for i in range(len(labels))]
    adata = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=genes))

    vocab = genes + perturbed
    matrix = rng.normal(size=(len(vocab), 32))
    emb = embeddings.from_arrays("standin", vocab, matrix)
    return adata, emb
