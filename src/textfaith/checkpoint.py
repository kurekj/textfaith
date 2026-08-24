"""Rebuild the text to probability pipeline from a saved training checkpoint."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

BATCH_SIZE = 64


def build_embedder(
    encoder: AutoModel,
    tokenizer: AutoTokenizer,
    query_prefix: str,
    maximum_token_count: int,
) -> Callable[[list[str]], np.ndarray]:
    """Build the function that encodes texts into L2 normalized, mean pooled embeddings.

    Batching follows length order to limit padding waste, and the original order
    is restored before returning. Training and loading must share this exact
    function: any drift between the two embedding paths would make the saved
    classification head consume vectors it was never trained on.
    """

    def embed(texts: list[str]) -> np.ndarray:
        length_order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        batches = []

        for start in range(0, len(length_order), BATCH_SIZE):
            batch_indices = length_order[start : start + BATCH_SIZE]
            tokenized = tokenizer(
                [query_prefix + texts[index] for index in batch_indices],
                padding=True,
                truncation=True,
                max_length=maximum_token_count,
                return_tensors="pt",
            )
            with torch.inference_mode():
                token_vectors = encoder(**tokenized).last_hidden_state

            mask = tokenized["attention_mask"].unsqueeze(-1)
            pooled = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            batches.append(torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())

        in_length_order = np.vstack(batches)
        in_original_order = np.empty_like(in_length_order)
        in_original_order[length_order] = in_length_order
        return in_original_order

    return embed


def load_predictor(
    checkpoint_directory: Path, thread_count: int = 8
) -> tuple[Callable[[Sequence[str]], np.ndarray], object]:
    """Load a checkpoint and return its predict_probabilities callable and tokenizer.

    The returned callable maps raw review texts to class probabilities of shape
    (n, 2). It is the unit that SHAP explains, because the classification head on
    its own consumes embeddings rather than text and so cannot be masked.
    """
    metadata = json.loads((checkpoint_directory / "metadata.json").read_text())

    torch.set_num_threads(thread_count)
    tokenizer = AutoTokenizer.from_pretrained(metadata["embedding_model_name"])
    encoder = AutoModel.from_pretrained(metadata["embedding_model_name"]).eval()
    head = joblib.load(checkpoint_directory / "head.joblib")

    embed = build_embedder(
        encoder, tokenizer, metadata["query_prefix"], metadata["maximum_token_count"]
    )

    def predict_probabilities(texts: Sequence[str]) -> np.ndarray:
        """Map raw review texts to class probabilities of shape (n, 2)."""
        return head.predict_proba(embed([str(text) for text in texts]))

    return predict_probabilities, tokenizer
