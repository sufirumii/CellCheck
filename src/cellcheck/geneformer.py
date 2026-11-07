"""Geneformer gene embeddings, read from the official Hugging Face checkpoints.

Only the input embedding table of the pretrained model is used. Nothing is
trained or fine-tuned. The gene dictionaries come from the same repository and
define which gene symbols the model can represent.
"""

from __future__ import annotations

import pickle

import numpy as np

from .embeddings import GeneEmbeddings

REPO = "ctheodoris/Geneformer"
DEFAULT_SUBFOLDER = "Geneformer-V2-104M"
DICT_TAG = "gc104M"  # dictionaries that belong to the 104M-gene-corpus models


def _load_pickle(filename: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, f"geneformer/{filename}")
    with open(path, "rb") as handle:
        return pickle.load(handle)  # official file from the model repository


def load(subfolder: str = DEFAULT_SUBFOLDER) -> GeneEmbeddings:
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "transformers, torch and huggingface_hub are required for Geneformer. "
            "Install them with: pip install 'cellcheck[geneformer]'"
        ) from exc

    name_to_id = _load_pickle(f"gene_name_id_dict_{DICT_TAG}.pkl")  # symbol -> Ensembl ID
    token_dict = _load_pickle(f"token_dictionary_{DICT_TAG}.pkl")  # Ensembl ID -> token

    model = AutoModel.from_pretrained(REPO, subfolder=subfolder)
    table = model.embeddings.word_embeddings.weight.detach().float().cpu().numpy()
    del model

    vocabulary = {g for g, ens in name_to_id.items() if ens in token_dict}

    def vector(gene: str) -> np.ndarray:
        return table[token_dict[name_to_id[gene]]]

    return GeneEmbeddings(
        name="geneformer",
        vocabulary=vocabulary,
        vector_fn=vector,
        scale=float(table.std()),
        dim=int(table.shape[1]),
    )
