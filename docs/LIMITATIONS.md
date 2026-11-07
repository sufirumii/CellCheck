# Limitations

These limits apply to every statement in the README.

1. One dataset. All results come from the Norman screen, which activates genes (overexpression) in one cell line.
   Conclusions may not carry over to knockouts, other cell types or other perturbation types.
2. One form of the model. Only the static input embeddings of Geneformer V2-104M were tested, combined with a linear
   model. Geneformer used in other ways (embeddings computed from cells in context, fine-tuning), other Geneformer
   sizes, and other models were not tested. A negative result for this form does not show that the model is
   uninformative in general.
3. The target is an average. Predictions are for the average profile of a perturbation, not for individual cells.
4. The noise floor is an estimate. It assumes independent cells and may be optimistic or pessimistic. Reaching it may
   not be possible.
5. The splits overlap. The 30 splits reuse the same 125 doubles and the same genes, so they are not independent. The
   ranges reported are descriptive, not formal confidence intervals, and no formal significance claim is made.
6. The scaled-additive baseline is probably slightly pessimistic. Noise in the single-perturbation averages pulls the
   fitted scale toward zero. The synthetic selftest shows this effect: with few cells the learned scale falls below
   the true value of 0.75 and approaches it as the number of cells grows.
7. Interactions are not modeled directly. The linear model works on the sum of two gene vectors, so a gene pair can
   only differ from the sum of its parts through what the fitted model learns from the training doubles.
8. Some perturbed genes were dropped because the model's vocabulary does not contain them.
9. A single split can mislead. The first single-split comparison (`results/norman_geneformer_104m/single_split_seed0.csv`)
   suggested a small advantage for the real embeddings over the shuffled control. Over 30 splits that advantage was not
   reliable. Results from one split should not be reported on their own.
10. The top-50-gene diagnostic in the notebooks selects genes using the observed answers. It is a diagnostic only.
