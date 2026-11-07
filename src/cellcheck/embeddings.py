"""Gene embedding sources.

An embedding source answers two questions: which gene symbols does the model
know, and what is the vector for each one. Everything else in CellCheck works
with this small interface, so any model that can produce one vector per gene
can be tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class GeneEmbeddings:
    name: str
    vocabulary: set
    vector_fn: Callable[[str], np.ndarray]
    scale: float  # standard deviation used to size the random control
    dim: int

    def has(self, gene: str) -> bool:
        return gene in self.vocabulary

    def vector(self, gene: str) -> np.ndarray:
        return np.asarray(self.vector_fn(gene), dtype=float)


def from_arrays(name: str, genes, matrix) -> GeneEmbeddings:
    """Build an embedding source from a gene list and a matrix of vectors."""
    matrix = np.asarray(matrix, dtype=float)
    genes = [str(g) for g in genes]
    if matrix.shape[0] != len(genes):
        raise ValueError("Number of genes and number of rows in the matrix differ.")
    index = {g: i for i, g in enumerate(genes)}
    return GeneEmbeddings(
        name=name,
        vocabulary=set(index),
        vector_fn=lambda g: matrix[index[g]],
        scale=float(matrix.std()),
        dim=int(matrix.shape[1]),
    )


def from_file(path: str, name: str) -> GeneEmbeddings:
    """Read a CSV or TSV file: first column is the gene symbol, the rest are numbers."""
    sep = "\t" if str(path).lower().endswith((".tsv", ".txt")) else ","
    table = pd.read_csv(path, sep=sep, index_col=0)
    return from_arrays(name, table.index.astype(str), table.to_numpy())
