# Contributing

Contributions that make the benchmark fairer or broader are welcome.

Good first contributions:

- Add embeddings from another model (see "Test your own embeddings" in the README) and share the resulting `results.json`.
- Add a second perturbation dataset, ideally a knockout or knockdown screen, so conclusions are not tied to one experiment.
- Add a stronger simple baseline. A new baseline is only useful if it is simple, fitted on training data only, and documented in `docs/METHOD.md`.
- Report a result that contradicts ours. Please include the command, the environment (`cellcheck env`) and the seed range.

Before opening a pull request:

```bash
pip install -e ".[dev]"
pytest -q
cellcheck selftest
```

Keep claims in the README within what the saved results show.
