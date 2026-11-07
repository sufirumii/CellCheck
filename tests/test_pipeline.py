import numpy as np
import pandas as pd
import pytest

from cellcheck import baselines, embeddings, models, pipeline, selftest
from cellcheck.data import pseudobulk
from cellcheck.evaluate import build_benchmark, run_split
from cellcheck.cli import build_parser


@pytest.fixture(scope="module")
def synthetic():
    return selftest.make_synthetic(n_genes=120, n_singles=12, n_doubles=16, cells_per_group=60, seed=3)


def test_pseudobulk_is_group_mean(synthetic):
    adata, _ = synthetic
    pb = pseudobulk(adata, "perturbation_name")
    mask = (adata.obs["perturbation_name"] == "P0").to_numpy()
    expected = np.asarray(adata.X)[mask].mean(0)
    np.testing.assert_allclose(pb.loc["P0"].to_numpy(), expected, rtol=1e-5)


def test_additive_and_scale():
    a, b = np.array([1.0, 2.0]), np.array([3.0, -1.0])
    np.testing.assert_allclose(baselines.additive(a, b), [4.0, 1.0])
    truth = 0.5 * np.array([[4.0, 1.0], [2.0, 6.0]])
    add = np.array([[4.0, 1.0], [2.0, 6.0]])
    assert baselines.fit_scale(add, truth) == pytest.approx(0.5)


def test_benchmark_uses_only_shared_genes_and_valid_doubles(synthetic):
    adata, emb = synthetic
    bench = build_benchmark(adata, emb, "perturbation_name", "control")
    assert set(bench.genes) == set(adata.var_names)
    single_set = set(bench.singles)
    assert all(g in single_set for d in bench.doubles for g in d.split("+"))
    assert "control" not in bench.singles


def test_vocabulary_restricts_genes(synthetic):
    adata, _ = synthetic
    keep = list(adata.var_names[:50]) + [f"P{i}" for i in range(12)]
    matrix = np.random.default_rng(0).normal(size=(len(keep), 8))
    small = embeddings.from_arrays("small", keep, matrix)
    bench = build_benchmark(adata, small, "perturbation_name", "control")
    assert len(bench.genes) == 50


def test_split_is_reproducible_and_test_is_held_out(synthetic):
    adata, emb = synthetic
    bench = build_benchmark(adata, emb, "perturbation_name", "control")
    r1 = run_split(bench, emb, seed=7, n_shuffles=1)
    r2 = run_split(bench, emb, seed=7, n_shuffles=1)
    assert r1 == r2
    assert r1["additive"] > 0 and r1["no_change"] > r1["additive"]


def test_shuffled_control_keeps_the_same_vectors_but_moves_them():
    genes = [f"g{i}" for i in range(10)]
    table = {g: np.full(3, float(i)) for i, g in enumerate(genes)}
    shuffled = models.shuffled_table(table, genes, np.random.default_rng(0))
    assert sorted(v[0] for v in shuffled.values()) == sorted(v[0] for v in table.values())
    assert any(not np.array_equal(shuffled[g], table[g]) for g in genes)


def test_one_hot_and_random_shapes():
    genes = ["a", "b", "c"]
    assert models.one_hot_table(genes)["b"].tolist() == [0.0, 1.0, 0.0]
    rand = models.random_table(genes, 5, 2.0, np.random.default_rng(0))
    assert rand["a"].shape == (5,)


def test_full_run_recovers_known_scale(tmp_path):
    adata, emb = selftest.make_synthetic()
    result = pipeline.run_all(adata, emb, "perturbation_name", "control", "+", n_splits=3, seed_start=1,
                              log=lambda *_: None)
    scale = result["paired"]["scale_learned"]["mean"]
    assert 0.65 < scale < 0.85  # synthetic doubles are built at 0.75 of the additive sum
    out = pipeline.save(result, tmp_path)
    for name in ["per_split.csv", "summary.csv", "noise_floor.csv", "results.json"]:
        assert (out / name).exists()
    summary = pd.read_csv(out / "summary.csv", index_col="method")
    assert summary.loc["scaled_additive", "mean_rmse"] < summary.loc["additive", "mean_rmse"]


def test_embedding_file_round_trip(tmp_path):
    path = tmp_path / "emb.csv"
    pd.DataFrame(np.eye(3), index=["A", "B", "C"]).to_csv(path)
    emb = embeddings.from_file(str(path), "mine")
    assert emb.has("B") and not emb.has("Z")
    assert emb.vector("B").tolist() == [0.0, 1.0, 0.0]


def test_cli_parser_accepts_documented_commands():
    parser = build_parser()
    assert parser.parse_args(["run", "--splits", "5", "--out", "x"]).splits == 5
    assert parser.parse_args(["report"]).command == "report"
    assert parser.parse_args(["selftest"]).command == "selftest"
