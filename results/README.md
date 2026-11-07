# Results

`norman_geneformer_104m/` holds the results that the main README refers to.

| File | What it is |
|---|---|
| `summary.csv` | Mean RMSE, standard deviation across splits, and share of the gap to the noise floor closed, per method |
| `results.json` | Dataset counts, noise floor estimate, paired comparisons across splits, provenance |
| `single_split_seed0.csv` | The single-split comparison that looked more favorable to Geneformer than the 30-split result |
| `RESULTS.md` | The summary as a Markdown table, written by `cellcheck report` |

Provenance: these values were produced by `notebooks/02_norman_benchmark.ipynb` on one Kaggle T4 session and copied
from its printed output, rounded to four decimals. The per-split table (`per_split.csv`) is not stored here.
Running `cellcheck run` writes it, together with `noise_floor.csv`, to the folder given by `--out`.

To redraw the figures from these files:

```bash
cellcheck report --results results/norman_geneformer_104m --out docs/figures
```
