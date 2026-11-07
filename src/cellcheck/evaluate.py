"""The benchmark itself: build the task, run repeated splits, summarize."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import baselines, models
from .data import pseudobulk
from .embeddings import GeneEmbeddings


@dataclass
class Benchmark:
    profiles: pd.DataFrame  # perturbation x shared genes (average expression)
    control: np.ndarray
    genes: list  # genes every method is scored on
    singles: list
    doubles: list
    perturbed_genes: list
    sep: str = "+"

    def delta(self, perturbation: str) -> np.ndarray:
        return self.profiles.loc[perturbation].to_numpy() - self.control

    def additive_delta(self, double: str) -> np.ndarray:
        a, b = double.split(self.sep)
        return baselines.additive(self.delta(a), self.delta(b))

    def deltas(self, perturbations) -> np.ndarray:
        return np.vstack([self.delta(p) for p in perturbations])


def build_benchmark(adata, emb: GeneEmbeddings, perturbation_col: str,
                    control_label: str, sep: str = "+") -> Benchmark:
    """Turn a perturbation dataset into the prediction task.

    Fairness rule: every method is scored on the same genes, namely the genes
    the embedding source can represent.
    """
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    all_profiles = pseudobulk(adata, perturbation_col)
    if control_label not in all_profiles.index:
        raise SystemExit(f"Control label '{control_label}' not found in '{perturbation_col}'.")

    genes = [g for g in all_profiles.columns if emb.has(g)]
    profiles = all_profiles[genes]
    control = profiles.loc[control_label].to_numpy()

    perturbations = [p for p in profiles.index if p != control_label]
    singles = [p for p in perturbations if sep not in p and emb.has(p)]
    single_set = set(singles)
    doubles = [
        p for p in perturbations
        if sep in p and all(g in single_set for g in p.split(sep))
    ]
    perturbed_genes = sorted({g for p in singles + doubles for g in p.split(sep)})
    return Benchmark(profiles, control, genes, singles, doubles, perturbed_genes, sep)


def estimate_noise_floor(adata, perturbation_col: str, bench: Benchmark, seed: int = 1) -> dict:
    """Estimate the error that even a perfect predictor would show.

    Cells of each double perturbation are split into two random halves. The
    RMSE between the two half-averages, divided by two, approximates the noise
    in an average computed from all cells.
    """
    rng = np.random.default_rng(seed)
    x = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    names = adata.obs[perturbation_col].astype(str).to_numpy()
    col = {g: i for i, g in enumerate(adata.var_names)}
    gidx = [col[g] for g in bench.genes]
    floor = {}
    for d in bench.doubles:
        idx = np.where(names == d)[0]
        rng.shuffle(idx)
        h = len(idx) // 2
        m1 = np.asarray(x[idx[:h]][:, gidx].mean(0)).ravel()
        m2 = np.asarray(x[idx[h:]][:, gidx].mean(0)).ravel()
        floor[d] = float(np.sqrt(np.mean((m1 - m2) ** 2)) / 2)
    return {"per_double": floor, "mean": float(np.mean(list(floor.values())))}


def run_split(bench: Benchmark, emb: GeneEmbeddings, seed: int, n_shuffles: int = 2) -> dict:
    """One random train/test split of the double perturbations.

    Training uses every single perturbation plus half of the doubles. Testing
    uses the other half of the doubles, which the model never sees.
    """
    rng = np.random.default_rng(seed)
    test = sorted(rng.choice(bench.doubles, size=len(bench.doubles) // 2, replace=False).tolist())
    train_doubles = [d for d in bench.doubles if d not in set(test)]
    train = bench.singles + train_doubles
    y_train, y_test = bench.deltas(train), bench.deltas(test)

    def rmse(pred):
        return float(np.sqrt(((pred - y_test) ** 2).mean(1)).mean())

    add_test = np.vstack([bench.additive_delta(d) for d in test])
    add_train = np.vstack([bench.additive_delta(d) for d in train_doubles])
    scale = baselines.fit_scale(add_train, bench.deltas(train_doubles))

    real = {g: emb.vector(g) for g in bench.perturbed_genes}

    def fit(table):
        return rmse(models.fit_predict(train, test, table, y_train, bench.sep)[0])

    shuffled = [
        fit(models.shuffled_table(real, bench.perturbed_genes, rng)) for _ in range(n_shuffles)
    ]
    random_control = fit(models.random_table(bench.perturbed_genes, emb.dim, emb.scale, rng))

    return {
        "seed": seed,
        "no_change": rmse(np.zeros_like(y_test)),
        "additive": rmse(add_test),
        "scaled_additive": rmse(scale * add_test),
        "scale": scale,
        emb.name: fit(real),
        "shuffled": float(np.mean(shuffled)),
        "random": random_control,
    }


def _paired(diff: pd.Series) -> dict:
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "mean_improvement": float(diff.mean()),
        "range_95_low": float(lo),
        "range_95_high": float(hi),
        "better_in_fraction_of_splits": float((diff > 0).mean()),
    }


def summarize(per_split: pd.DataFrame, name: str, floor: float):
    """Summary table and paired comparisons across splits."""
    cols = ["no_change", "additive", "scaled_additive", "random", "shuffled", name]
    additive_mean = per_split["additive"].mean()
    summary = pd.DataFrame({
        "mean_rmse": per_split[cols].mean(),
        "std_across_splits": per_split[cols].std(),
        "gap_to_floor_closed_pct": 100 * (additive_mean - per_split[cols].mean()) / (additive_mean - floor),
    })
    summary.index.name = "method"
    paired = {
        "model_vs_shuffled": _paired(per_split["shuffled"] - per_split[name]),
        "model_vs_scaled_additive": _paired(per_split["scaled_additive"] - per_split[name]),
        "scale_learned": {
            "mean": float(per_split["scale"].mean()),
            "min": float(per_split["scale"].min()),
            "max": float(per_split["scale"].max()),
        },
    }
    return summary, paired
