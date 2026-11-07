"""End-to-end run: build the task, run the splits, summarize, save."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import __version__
from .embeddings import GeneEmbeddings
from .evaluate import build_benchmark, estimate_noise_floor, run_split, summarize


def run_all(adata, emb: GeneEmbeddings, perturbation_col: str, control_label: str,
            sep: str = "+", n_splits: int = 30, seed_start: int = 1000, log=print) -> dict:
    bench = build_benchmark(adata, emb, perturbation_col, control_label, sep)
    if len(bench.doubles) < 4:
        raise SystemExit("Fewer than 4 usable double perturbations were found. "
                         "Check the perturbation column, separator and embedding vocabulary.")
    info = {
        "n_cells": int(adata.n_obs),
        "n_genes_in_data": int(adata.n_vars),
        "n_shared_genes": len(bench.genes),
        "n_single_perturbations": len(bench.singles),
        "n_double_perturbations": len(bench.doubles),
        "n_test_doubles_per_split": len(bench.doubles) // 2,
    }
    log(f"cells: {info['n_cells']} | genes in data: {info['n_genes_in_data']} | "
        f"genes shared with the embedding: {info['n_shared_genes']}")
    log(f"usable singles: {info['n_single_perturbations']} | usable doubles: "
        f"{info['n_double_perturbations']} | test doubles per split: {info['n_test_doubles_per_split']}")

    floor = estimate_noise_floor(adata, perturbation_col, bench)
    log(f"estimated noise floor (RMSE): {floor['mean']:.4f}")

    rows = []
    for i, seed in enumerate(range(seed_start, seed_start + n_splits)):
        rows.append(run_split(bench, emb, seed))
        if (i + 1) % 10 == 0 or i + 1 == n_splits:
            log(f"splits done: {i + 1}/{n_splits}")
    per_split = pd.DataFrame(rows)
    summary, paired = summarize(per_split, emb.name, floor["mean"])
    return {"embedding": emb.name, "info": info, "n_splits": n_splits, "floor": floor,
            "per_split": per_split, "summary": summary, "paired": paired}


def format_text(result: dict) -> str:
    name = result["embedding"]
    lines = [result["summary"].round(4).to_string(), ""]
    for label, key in [(f"{name} vs shuffled embeddings", "model_vs_shuffled"),
                       (f"{name} vs scaled additive", "model_vs_scaled_additive")]:
        p = result["paired"][key]
        lines.append(
            f"{label}: mean improvement {p['mean_improvement']:.4f} | 95% of splits between "
            f"{p['range_95_low']:.4f} and {p['range_95_high']:.4f} | "
            f"{name} better in {100 * p['better_in_fraction_of_splits']:.0f}% of splits"
        )
    s = result["paired"]["scale_learned"]
    lines.append(f"scale learned across splits: mean {s['mean']:.3f} | min {s['min']:.3f} | max {s['max']:.3f}")
    return "\n".join(lines)


def save(result: dict, out_dir, provenance: str = "") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["per_split"].to_csv(out / "per_split.csv", index=False)
    result["summary"].round(6).to_csv(out / "summary.csv")
    pd.DataFrame({"double": list(result["floor"]["per_double"]),
                  "noise_floor_rmse": list(result["floor"]["per_double"].values())}
                 ).to_csv(out / "noise_floor.csv", index=False)
    payload = {
        "tool": "cellcheck",
        "version": __version__,
        "embedding": result["embedding"],
        "dataset": result["info"],
        "n_splits": result["n_splits"],
        "noise_floor_rmse": result["floor"]["mean"],
        "paired": result["paired"],
        "provenance": provenance,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    return out
