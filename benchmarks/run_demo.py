"""Reproducible capsule: the full textfaith protocol in a few minutes on CPU.

The full benchmark of the accompanying article costs hours of attribution
computation, so this demo substitutes the smallest possible experiment that
still exercises every stage of the library: four attribution sources computed
through one shared masker at a counted budget, sufficiency and
comprehensiveness over a percentile sweep at two chunk sizes, paired Wilcoxon
tests against the random baseline, and the heat map rendering.

The corpus is synthetic on purpose. Reviews are built from templates around a
known sentiment vocabulary, so the demo needs no downloads, has no licensing
constraints, and the words a faithful attribution should find are known by
construction. The classifier is a TF-IDF logistic regression rather than the
article's frozen encoder for the same reason: the protocol only needs a
callable from texts to class probabilities, which is the point the demo makes.

Run from the repository root:

    uv run python benchmarks/run_demo.py

Outputs go to benchmarks/results/: a results table, the tests against the
random baseline, and a four panel heat map of one review.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from textfaith.experiment import compute_timed_attributions
from textfaith.faithfulness import (
    ABSOLUTE_SUM,
    build_word_masker,
    measure_comprehensiveness,
    measure_sufficiency,
)
from textfaith.rendering import render_words, word_figure_size

RANDOM_SEED = 42
REVIEW_COUNT = 240
EXPLAINED_COUNT = 24
EVALUATION_BUDGET = 100
PERCENTILES = [25, 50, 75]
CHUNK_SIZES = [1, 4]
RESULTS_DIRECTORY = Path(__file__).parent / "results"

POSITIVE_WORDS = [
    "wonderful", "excellent", "moving", "brilliant", "delightful",
    "gripping", "superb", "charming", "masterful", "uplifting",
]
NEGATIVE_WORDS = [
    "dreadful", "boring", "clumsy", "tedious", "shallow",
    "predictable", "lifeless", "grating", "incoherent", "forgettable",
]
FILLER_SENTENCES = [
    "the film follows a family through one long summer in the city",
    "the story is set in a small coastal town during the festival",
    "the plot centers on two friends who reopen an old theater",
    "the movie tracks a young musician moving between rented rooms",
    "the picture shows a slow week in the life of a night porter",
    "the narrative jumps between a courtroom and a quiet farmhouse",
]


def build_corpus(random_seed: int) -> tuple[list[str], np.ndarray]:
    """Synthetic reviews with a planted sentiment vocabulary.

    Each review is a neutral filler sentence with three class specific words
    appended in fixed slots. The filler carries no label signal, so a faithful
    attribution must concentrate on the planted words, which is what makes the
    demo's heat map readable as a correctness check by eye.
    """
    generator = np.random.default_rng(random_seed)
    texts = []
    labels = []
    for index in range(REVIEW_COUNT):
        label = index % 2
        vocabulary = POSITIVE_WORDS if label == 1 else NEGATIVE_WORDS
        planted = generator.choice(vocabulary, size=3, replace=False)
        filler = FILLER_SENTENCES[generator.integers(len(FILLER_SENTENCES))]
        texts.append(
            f"{filler} and it felt {planted[0]} with {planted[1]} scenes "
            f"and a {planted[2]} ending"
        )
        labels.append(label)
    return texts, np.array(labels)


def main() -> None:
    RESULTS_DIRECTORY.mkdir(exist_ok=True)

    texts, labels = build_corpus(RANDOM_SEED)
    split = int(REVIEW_COUNT * 0.75)
    classifier = make_pipeline(
        TfidfVectorizer(), LogisticRegression(random_state=RANDOM_SEED)
    )
    classifier.fit(texts[:split], labels[:split])
    accuracy = classifier.score(texts[split:], labels[split:])
    print(f"held out accuracy: {accuracy:.3f}")

    def predict_probabilities(input_texts) -> np.ndarray:
        return classifier.predict_proba(list(input_texts))

    explained_texts = texts[split : split + EXPLAINED_COUNT]
    masker = build_word_masker()
    original_probabilities = predict_probabilities(explained_texts)
    predicted_classes = original_probabilities.argmax(axis=1)
    original_confidence = original_probabilities[
        np.arange(len(explained_texts)), predicted_classes
    ]

    payloads = {}
    for method_name, budget in [
        ("shap", EVALUATION_BUDGET),
        ("lime", EVALUATION_BUDGET),
        ("occlusion", None),
        ("random", None),
    ]:
        payloads[method_name] = compute_timed_attributions(
            method_name, predict_probabilities, explained_texts, masker,
            budget, RANDOM_SEED,
        )
        print(
            f"{method_name}: {payloads[method_name]['wall_seconds']:.1f} s, "
            f"{payloads[method_name]['prediction_count']} predictions"
        )

    rows = []
    per_document = {}
    for method_name, payload in payloads.items():
        for chunk_size in CHUNK_SIZES:
            sufficiency = measure_sufficiency(
                predict_probabilities, explained_texts, PERCENTILES, masker,
                payload["attributions"], chunk_size, ABSOLUTE_SUM,
            )
            comprehensiveness = measure_comprehensiveness(
                predict_probabilities, explained_texts, PERCENTILES, masker,
                payload["attributions"], chunk_size, ABSOLUTE_SUM,
            )
            relative_sufficiency = (
                sufficiency.mean(axis=1) / original_confidence
            )
            relative_comprehensiveness = (
                comprehensiveness.mean(axis=1) / original_confidence
            )
            per_document[(method_name, chunk_size)] = (
                relative_sufficiency, relative_comprehensiveness,
            )
            rows.append({
                "method": method_name,
                "chunk_size": chunk_size,
                "sufficiency_percent": 100 * relative_sufficiency.mean(),
                "comprehensiveness_percent":
                    100 * relative_comprehensiveness.mean(),
                "wall_seconds": payload["wall_seconds"],
                "prediction_count": payload["prediction_count"],
            })

    results = pd.DataFrame(rows)
    results_path = RESULTS_DIRECTORY / "demo_results.csv"
    results.to_csv(results_path, index=False)
    print()
    print(results.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    comparison_rows = []
    for method_name in ["shap", "lime", "occlusion"]:
        for chunk_size in CHUNK_SIZES:
            method_sufficiency, method_comprehensiveness = per_document[
                (method_name, chunk_size)
            ]
            random_sufficiency, random_comprehensiveness = per_document[
                ("random", chunk_size)
            ]
            comparison_rows.append({
                "comparison": f"{method_name}_vs_random",
                "chunk_size": chunk_size,
                "sufficiency_advantage_percent":
                    100 * (random_sufficiency - method_sufficiency).mean(),
                "sufficiency_wilcoxon_p":
                    wilcoxon(method_sufficiency, random_sufficiency).pvalue,
                "comprehensiveness_advantage_percent":
                    100 * (method_comprehensiveness
                           - random_comprehensiveness).mean(),
                "comprehensiveness_wilcoxon_p":
                    wilcoxon(method_comprehensiveness,
                             random_comprehensiveness).pvalue,
            })
    comparisons = pd.DataFrame(comparison_rows)
    comparisons_path = RESULTS_DIRECTORY / "demo_vs_random.csv"
    comparisons.to_csv(comparisons_path, index=False)
    print()
    print(comparisons.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    example_index = 0
    words = masker.data_transform(explained_texts[example_index])[0]
    line_count = 3
    width, height = word_figure_size(70, line_count, 10)
    figure, axes = plt.subplots(
        len(payloads), 1, figsize=(width, height * len(payloads))
    )
    for axis, (method_name, payload) in zip(axes, payloads.items()):
        render_words(axis, words, payload["attributions"][example_index])
        axis.set_title(method_name, fontsize=10, loc="left")
    figure.tight_layout()
    heatmap_path = RESULTS_DIRECTORY / "demo_heatmap.png"
    figure.savefig(heatmap_path, dpi=200, bbox_inches="tight")
    print()
    print(f"wrote {results_path}, {comparisons_path} and {heatmap_path}")


if __name__ == "__main__":
    main()
