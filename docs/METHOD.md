# Method

This document describes exactly what `cellcheck run` computes.

## Question

After a gene is perturbed, how well can a method predict the new average expression profile of the cells? In
particular: for pairs of perturbations that were never used for fitting, does a model do better than simple rules,
and better than controls that carry no gene-specific information?

## Data

The default dataset is Norman et al. 2019, a CRISPR activation screen in K562 cells with single and double
perturbations, loaded through `pertpy` (`norman_2019`). It has 111,255 cells and 19,018 genes. Any AnnData file can be
used instead with `--h5ad`, as long as one column of `obs` names the perturbation (`control` for controls and
`GENE_A+GENE_B` for doubles) and `X` holds log-normalized expression.

## Target

For every perturbation, the average of `X` over its cells (a pseudobulk profile). The quantity to predict is the
change relative to the average of the control cells.

## Fairness rule: one gene set for every method

Only genes the embedding source can represent are scored. For Geneformer V2-104M this is 13,534 of the 19,018 genes in
the Norman data, namely the symbols that map to an Ensembl ID that has a token. Every method, including the simple
baselines, is scored on exactly this set.

## Usable perturbations

A single perturbation is usable if its gene is in the embedding vocabulary. A double is usable if both of its genes
have a usable single perturbation. For Norman this gives 101 singles and 125 doubles (four perturbed genes are missing
from Geneformer's vocabulary: C19orf26, C3orf72, ELMSAN1, KIAA1804).

## Splits

Training uses all usable singles plus a random half of the usable doubles. Testing uses the other half of the doubles
(62 per split for Norman). The split is drawn with a seed, and the whole comparison is repeated for 30 seeds
(1000 to 1029 by default).

## Methods compared

| Method | Prediction for a double perturbation |
|---|---|
| No change | Same as control (change = 0) |
| Additive | Change of the first single plus change of the second single |
| Scaled additive | Additive multiplied by one number, fitted by least squares on the training doubles only |
| Embeddings + ridge | Ridge regression from the sum of the two gene vectors to the change in expression. Ridge strength is chosen by built-in leave-one-out cross-validation on the training set |
| Shuffled embeddings | Same as above, but each gene receives the vector of a different gene (average of 2 shuffles per split) |
| Random embeddings | Same as above, with Gaussian random vectors scaled like the real ones |

For Geneformer, a gene vector is its row in the input embedding table of the pretrained model. No model is trained or
fine-tuned. Only the ridge regression is fitted.

## Metrics

- RMSE: for each test double, the root mean squared difference over genes between predicted and observed change; then
  the mean over the test doubles; then the mean over splits. Lower is better.
- Gap to the noise floor closed: `100 * (RMSE_additive - RMSE_method) / (RMSE_additive - floor)`. Additive is 0 and the
  estimated noise floor is 100. Values can be negative.
- Paired comparisons: per split, the RMSE of the control minus the RMSE of the model. The tool reports the mean, the
  range that contains 95 percent of the splits, and the share of splits where the model is better.

## Noise floor

The cells of each double are split into two random halves. The RMSE between the two half-averages, divided by two,
approximates the error of an average computed from all cells. This is an estimate. It treats cells as independent
draws and ignores differences between batches.

## Reproducibility

Seeds are fixed. The split seeds, shuffle seeds and random-vector seeds are all derived from the split seed, so the
same command gives the same numbers on the same data and library versions.
