"""Command line interface: cellcheck run | report | selftest | env."""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from pathlib import Path

from . import __version__


def _cmd_env(_args) -> int:
    print(f"cellcheck {__version__}")
    print(f"python {platform.python_version()} on {platform.platform()}")
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "anndata", "matplotlib",
                "torch", "transformers", "huggingface_hub", "pertpy"]:
        try:
            from importlib.metadata import version
            print(f"  {pkg}: {version(pkg)}")
        except Exception:
            print(f"  {pkg}: not installed")
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU: {props.name} | {props.total_memory / 1e9:.1f} GB | "
                  f"compute capability {props.major}.{props.minor}")
        else:
            print("GPU: none detected (the benchmark itself runs on CPU)")
    except ImportError:
        print("GPU: torch not installed (the benchmark itself runs on CPU)")
    return 0


def _cmd_run(args) -> int:
    from . import data, embeddings, geneformer, pipeline, report

    if args.h5ad:
        adata = data.load_h5ad(args.h5ad)
        dataset_name = Path(args.h5ad).name
    else:
        adata = data.load_norman()
        dataset_name = "norman_2019 (via pertpy)"

    if args.embeddings_file:
        emb = embeddings.from_file(args.embeddings_file, args.embeddings_name)
    else:
        emb = geneformer.load(args.geneformer_subfolder)

    print(f"dataset: {dataset_name} | embeddings: {emb.name}")
    result = pipeline.run_all(adata, emb, args.perturbation_col, args.control_label,
                              args.separator, args.splits, args.seed_start)
    print()
    print(pipeline.format_text(result))
    out = pipeline.save(result, args.out, provenance=f"cellcheck run on {dataset_name}")
    print(f"\nsaved to {out}/")
    if not args.no_figures:
        figs = report.make_report(out, Path(out) / "figures")
        print("figures:", ", ".join(figs[:-1]))
    return 0


def _cmd_report(args) -> int:
    from . import report

    written = report.make_report(args.results, args.out)
    print("wrote:", ", ".join(written))
    return 0


def _cmd_selftest(args) -> int:
    from . import pipeline, report, selftest

    print("SELFTEST on synthetic data. These numbers are not results; they only check that the pipeline runs.")
    adata, emb = selftest.make_synthetic()
    result = pipeline.run_all(adata, emb, "perturbation_name", "control", "+", n_splits=args.splits, seed_start=1)
    print()
    print(pipeline.format_text(result))
    scale = result["paired"]["scale_learned"]["mean"]
    print(f"\ncheck: synthetic doubles were generated at {selftest.TRUE_SCALE} of the additive sum; "
          f"learned scale is {scale:.3f}")
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="cellcheck_selftest_"))
    pipeline.save(result, out, provenance="selftest on synthetic data; not a result")
    report.make_report(out, out / "figures")
    print(f"files written to {out}/")
    ok = 0.65 < scale < 0.85
    print("selftest passed" if ok else "selftest FAILED")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cellcheck",
                                description="Benchmark gene-perturbation prediction against simple methods and controls.")
    p.add_argument("--version", action="version", version=f"cellcheck {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="Show versions and GPU information.").set_defaults(func=_cmd_env)

    r = sub.add_parser("run", help="Run the benchmark.")
    r.add_argument("--h5ad", help="Path to your own AnnData file. Default: Norman 2019 through pertpy.")
    r.add_argument("--perturbation-col", default="perturbation_name")
    r.add_argument("--control-label", default="control")
    r.add_argument("--separator", default="+", help="Separator between genes in a double perturbation name.")
    r.add_argument("--embeddings-file", help="CSV/TSV of gene embeddings (first column: gene symbol). "
                                             "Default: Geneformer from Hugging Face.")
    r.add_argument("--embeddings-name", default="custom", help="Name used for your embeddings in the output.")
    r.add_argument("--geneformer-subfolder", default="Geneformer-V2-104M",
                   help="Only Geneformer-V2-104M has been tested.")
    r.add_argument("--splits", type=int, default=30)
    r.add_argument("--seed-start", type=int, default=1000)
    r.add_argument("--out", default="results/run")
    r.add_argument("--no-figures", action="store_true")
    r.set_defaults(func=_cmd_run)

    g = sub.add_parser("report", help="Draw figures and a results table from saved results.")
    g.add_argument("--results", default="results/run")
    g.add_argument("--out", default="docs/figures")
    g.set_defaults(func=_cmd_report)

    s = sub.add_parser("selftest", help="Run the pipeline on a small synthetic dataset (no downloads).")
    s.add_argument("--splits", type=int, default=5)
    s.add_argument("--out", default=None)
    s.set_defaults(func=_cmd_selftest)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
