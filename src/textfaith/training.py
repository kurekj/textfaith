"""Train a binary text classification checkpoint the way notebook 01 trained IMDB.

The classifier exists only to be explained, so the pipeline optimizes for
comparability across datasets rather than accuracy: every dataset gets the same
frozen encoder, the same sampling protocol, the same head, and the same seeds.
Any faithfulness difference measured later is then attributable to the data and
the attribution method, not to pipeline drift.
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from transformers import AutoModel, AutoTokenizer

from textfaith.checkpoint import build_embedder

EMBEDDING_MODEL_NAME = "intfloat/e5-small-v2"
QUERY_PREFIX = "query: "
MAXIMUM_TOKEN_COUNT = 512
MINIMUM_WORD_COUNT = 20

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
ESCAPED_NEWLINE_PATTERN = re.compile(r"\\n|\\")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip HTML tags and CSV escape artifacts, then collapse whitespace.

    The fast.ai CSV exports encode newlines as a literal backslash n sequence
    and sometimes leave a bare backslash where a newline was. Left in place,
    those glue neighboring words into tokens like "dwindling\\band", which the
    whitespace masker would treat as single attributable units and quietly
    corrupt every downstream measurement.
    """
    without_html = HTML_TAG_PATTERN.sub(" ", text)
    without_escapes = ESCAPED_NEWLINE_PATTERN.sub(" ", without_html)
    return WHITESPACE_PATTERN.sub(" ", without_escapes).strip()


def build_checkpoint(
    reviews: pd.DataFrame,
    checkpoint_directory: Path,
    dataset_name: str,
    label_names: dict[int, str],
    sample_size: int = 2000,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.0,
    random_seed: int = 42,
    thread_count: int = 8,
) -> dict:
    """Train and save a checkpoint loadable by `load_predictor`, return its metadata.

    Expects a DataFrame with `text` (already cleaned) and binary `label`
    columns. The protocol mirrors notebook 01: deduplicate, drop texts over
    the token limit or under `MINIMUM_WORD_COUNT` words, stratified sample,
    stratified split, embed with the frozen encoder, fit the MLP head.

    With `validation_fraction` set, the split is three way: test is carved
    out first, then validation from the remainder, both stratified. The head
    never sees validation or test; its early stopping uses an internal slice
    of the training set, so the validation split stays clean for any future
    model selection and the test split exists solely for explanation
    evaluation. Validation accuracy is recorded next to test accuracy as the
    honesty check between them.

    Idempotent per directory: if `metadata.json` already exists the checkpoint
    is left untouched and its metadata returned, matching the cache discipline
    used everywhere else in this project.
    """
    metadata_path = checkpoint_directory / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())

    torch.set_num_threads(thread_count)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    encoder = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).eval()
    embed = build_embedder(encoder, tokenizer, QUERY_PREFIX, MAXIMUM_TOKEN_COUNT)

    deduplicated = reviews.drop_duplicates(subset="text").copy()
    deduplicated["token_count"] = [
        len(token_ids)
        for token_ids in tokenizer(
            deduplicated["text"].tolist(), truncation=False
        )["input_ids"]
    ]
    word_counts = deduplicated["text"].str.split().str.len()
    eligible = deduplicated[
        (deduplicated["token_count"] <= MAXIMUM_TOKEN_COUNT)
        & (word_counts >= MINIMUM_WORD_COUNT)
    ]

    sampled, _ = train_test_split(
        eligible,
        train_size=sample_size,
        random_state=random_seed,
        stratify=eligible["label"],
    )
    remainder, test_reviews = train_test_split(
        sampled,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=sampled["label"],
    )
    if validation_fraction > 0:
        train_reviews, validation_reviews = train_test_split(
            remainder,
            test_size=validation_fraction / (1 - test_fraction),
            random_state=random_seed,
            stratify=remainder["label"],
        )
    else:
        train_reviews, validation_reviews = remainder, None

    train_embeddings = embed(train_reviews["text"].tolist())
    test_embeddings = embed(test_reviews["text"].tolist())

    head = MLPClassifier(
        hidden_layer_sizes=(128,), early_stopping=True, random_state=random_seed
    )
    head.fit(train_embeddings, train_reviews["label"])
    test_accuracy = accuracy_score(
        test_reviews["label"], head.predict(test_embeddings)
    )

    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(head, checkpoint_directory / "head.joblib")
    train_reviews.to_parquet(checkpoint_directory / "train.parquet", index=False)
    test_reviews.to_parquet(checkpoint_directory / "test.parquet", index=False)
    np.save(checkpoint_directory / "train_embeddings.npy", train_embeddings)
    np.save(checkpoint_directory / "test_embeddings.npy", test_embeddings)

    metadata = {
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "maximum_token_count": MAXIMUM_TOKEN_COUNT,
        "query_prefix": QUERY_PREFIX,
        "random_seed": random_seed,
        "sample_size": sample_size,
        "train_size": len(train_reviews),
        "test_size": len(test_reviews),
        "test_accuracy": float(test_accuracy),
        "dataset": dataset_name,
        "label_names": {str(label): name for label, name in label_names.items()},
        "minimum_word_count": MINIMUM_WORD_COUNT,
    }

    if validation_reviews is not None:
        validation_embeddings = embed(validation_reviews["text"].tolist())
        validation_accuracy = accuracy_score(
            validation_reviews["label"], head.predict(validation_embeddings)
        )
        validation_reviews.to_parquet(
            checkpoint_directory / "validation.parquet", index=False
        )
        np.save(
            checkpoint_directory / "validation_embeddings.npy", validation_embeddings
        )
        metadata["validation_size"] = len(validation_reviews)
        metadata["validation_accuracy"] = float(validation_accuracy)

    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata
