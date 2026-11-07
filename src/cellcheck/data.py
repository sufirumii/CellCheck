"""Loading perturbation data and building one average profile per perturbation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_norman():
    """Load the Norman et al. 2019 dataset through pertpy."""
    try:
        import pertpy as pt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "pertpy is required for --dataset norman. "
            "Install it with: pip install 'cellcheck[data]'"
        ) from exc
    return pt.data.norman_2019()


def load_h5ad(path: str):
    """Load any AnnData file."""
    import anndata as ad

    return ad.read_h5ad(path)


def pseudobulk(adata, perturbation_col: str) -> pd.DataFrame:
    """Average expression per perturbation (rows) over all genes (columns).

    The average is taken over cells from ``adata.X``, which is expected to hold
    log-normalized expression values.
    """
    names = adata.obs[perturbation_col].astype(str)
    codes, uniq = pd.factorize(names)
    membership = sp.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(uniq), len(codes)),
    )
    sums = membership @ adata.X
    sums = sums.toarray() if sp.issparse(sums) else np.asarray(sums)
    counts = np.asarray(membership.sum(1)).ravel()
    return pd.DataFrame(
        sums / counts[:, None],
        index=uniq,
        columns=adata.var_names.astype(str),
    )
