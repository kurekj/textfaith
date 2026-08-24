"""Word level attributions and measurements of whether they are faithful.

A readable heat map proves nothing on its own. These functions test the claim
directly: rank spans of text by attributed importance, cut some of them, and
watch how far the model output moves.

The measurements are method agnostic. They consume any per word attribution
arrays aligned with the word masker, which is what allows different attribution
methods, SHAP and LIME here, to be compared on equal footing: same reviews,
same cutting scheme, only the ranking differs.

Sufficiency cuts the least important spans. If the attributions are good, the
discarded text carried no signal and the output should barely move.
Comprehensiveness cuts the most important spans instead, where a good ranking
should move the output a lot. The two together catch a failure that sufficiency
alone cannot, namely an uninformative ranking that happens to leave the output
stable.

Spans are runs of consecutive words. A chunk size of 1 scores each word on its
own, which is the finest available granularity and reproduces word level
behavior exactly.

Neither number means anything in isolation, because masked text is off the
distribution the model was trained on. Compare against `shuffle_attributions`
to get a control that suffers the same distribution shift.
"""

from collections.abc import Callable, Sequence

import numpy as np
import shap
from lime.lime_text import IndexedString, LimeTextExplainer

WORD_SPLIT_PATTERN = r"\s+"

ABSOLUTE_SUM = "absolute_sum"
SUM_ABSOLUTE = "sum_absolute"
MAX_ABSOLUTE = "max_absolute"
COMBINATIONS = (ABSOLUTE_SUM, SUM_ABSOLUTE, MAX_ABSOLUTE)


def build_word_masker() -> shap.maskers.Text:
    """Build the masker that treats whole words, not subword tokens, as units.

    Splitting on whitespace rather than the more common non word character class
    keeps contractions intact. Splitting on non word characters would break
    "isn't" into "isn'" and "t", which destroys negation as an attributable unit
    and matters directly for sentiment.

    The same masker object must be used for explaining and for cutting. Shapley
    values are only defined relative to a perturbation scheme, so measuring
    against a different one would not test the attributions that were produced.
    """
    return shap.maskers.Text(WORD_SPLIT_PATTERN)


def compute_word_attributions(
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    masker: shap.maskers.Text,
    evaluation_budget: int | None = None,
    batch_size: int | None = None,
) -> list[np.ndarray]:
    """Return per observation signed SHAP values for the predicted class.

    Signed rather than absolute, because Shapley values are additive: the
    expected output shift from removing a set of words is the sum of their
    signed values. Discarding the signs here would make that aggregation
    impossible to recover downstream.

    `evaluation_budget` caps the model evaluations the partition explainer may
    spend per observation. It is a cap, not an exact count: short texts resolve
    their partition tree early and stop under budget, long texts spend it all
    and leave equal valued runs where the tree was never split. `batch_size`
    sets how many masked texts go into one prediction call; shap's default of 5
    wastes most of the embedder's own 64 wide batching. Both default to None,
    which keeps shap's defaults and the behavior of earlier notebooks.

    This is by far the expensive step, so callers should cache the result. Once
    it is cached, re-running the measurements at different percentiles, chunk
    sizes, and combinations is nearly free.
    """
    explainer = shap.Explainer(predict_probabilities, masker)
    call_options = {}
    if evaluation_budget is not None:
        call_options["max_evals"] = evaluation_budget
    if batch_size is not None:
        call_options["batch_size"] = batch_size
    explanation = explainer(list(texts), silent=True, **call_options)
    predicted_classes = predict_probabilities(texts).argmax(axis=1)

    return [
        explanation.values[index][:, predicted_classes[index]]
        for index in range(len(texts))
    ]


def compute_lime_word_attributions(
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    masker: shap.maskers.Text,
    random_seed: int,
    sample_count: int = 1000,
) -> list[np.ndarray]:
    """Per observation LIME weights for the predicted class, one per word position.

    LIME drops random subsets of words and fits a ridge regression from the
    dropout patterns to the model output, so unlike Shapley values the weights
    carry no additivity guarantee. They are whatever the local linear surrogate
    learned, which is exactly why measuring their faithfulness is worthwhile
    rather than assumed.

    Two choices make the output directly comparable to `compute_word_attributions`.
    Splitting on the masker's whitespace pattern with `bow=False` yields one
    weight per word position instead of one per unique word, aligning the array
    with the masker's word split. And dropped words are replaced with the same
    `...` marker the masker inserts, so both methods learn under the same
    replacement semantics; the one remaining difference is that LIME marks every
    dropped word while the masker collapses consecutive drops into one marker.

    Costs `sample_count` predictions per observation, comparable to the SHAP
    computation, so callers should cache the result.
    """
    explainer = LimeTextExplainer(
        split_expression=WORD_SPLIT_PATTERN,
        bow=False,
        mask_string="...",
        random_state=random_seed,
    )
    predicted_classes = predict_probabilities(texts).argmax(axis=1)

    attributions = []
    for text, predicted_class in zip(texts, predicted_classes):
        word_count = len(masker.data_transform(text)[0])
        indexed_text = IndexedString(
            text, bow=False, split_expression=WORD_SPLIT_PATTERN, mask_string="..."
        )
        if indexed_text.num_words() != word_count:
            raise ValueError(
                "LIME's word split does not match the masker's word count, "
                "the two must tokenize identically for the attributions to align."
            )
        explanation = explainer.explain_instance(
            text,
            predict_probabilities,
            labels=(int(predicted_class),),
            num_features=word_count,
            num_samples=sample_count,
        )
        attribution = np.zeros(word_count)
        for word_position, weight in explanation.as_map()[int(predicted_class)]:
            attribution[word_position] = weight
        attributions.append(attribution)

    return attributions


def compute_random_word_attributions(
    texts: Sequence[str],
    masker: shap.maskers.Text,
    random_seed: int,
) -> list[np.ndarray]:
    """Per observation standard gaussian noise, one draw per word position.

    The floor every real method must clear: a ranking that knows nothing about
    the model or the text. Unlike `shuffle_attributions`, which permutes a
    particular method's values and therefore inherits its value distribution,
    this baseline is method independent, mean zero and unit variance per word,
    so a single random arm serves every method in a comparison. It costs no
    model predictions at all.
    """
    generator = np.random.default_rng(random_seed)
    return [
        generator.standard_normal(len(masker.data_transform(text)[0]))
        for text in texts
    ]


def compute_occlusion_word_attributions(
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    masker: shap.maskers.Text,
) -> list[np.ndarray]:
    """Per observation drop in predicted class probability from masking each word alone.

    The most direct attribution there is: the value of word i is how far the
    predicted class probability falls when word i alone is replaced by the
    masker's `...` marker. No sampling, no surrogate, and no budget knob, the
    cost is fixed at one prediction per word plus one for the original, which
    makes it the baseline both SHAP and LIME are approximating shortcuts to.

    Its known blindness: each word is judged in isolation, so interactions
    and redundancy get no credit. Two words carrying the same signal each look
    cheap to remove because the other still stands. Whether that blindness
    costs measurable faithfulness is exactly what the deletion measurements
    can expose.
    """
    original_probabilities = predict_probabilities(texts)
    predicted_classes = original_probabilities.argmax(axis=1)

    attributions = []
    for text, original, predicted_class in zip(
        texts, original_probabilities, predicted_classes
    ):
        word_count = len(masker.data_transform(text)[0])
        occluded_texts = []
        for word_index in range(word_count):
            kept_words = np.ones(word_count, dtype=bool)
            kept_words[word_index] = False
            occluded_texts.append(masker(kept_words, text)[0][0])
        occluded_probabilities = predict_probabilities(occluded_texts)
        attributions.append(
            (original[predicted_class] - occluded_probabilities[:, predicted_class])
            .astype(np.float64)
        )
    return attributions


def paired_advantage(better: np.ndarray, worse: np.ndarray) -> float:
    """Mean paired advantage divided by its standard error, averaged over percentiles.

    Every arm of a comparison runs on the same reviews, so the difference is
    paired per review and the standard error belongs to the differences, not to
    the group means. Group means alone would overstate the evidence. Above
    roughly 2 the advantage is larger than sampling noise.
    """
    differences = better - worse
    standard_errors = differences.std(axis=0, ddof=1) / np.sqrt(differences.shape[0])
    return float(np.mean(differences.mean(axis=0) / standard_errors))


def shuffle_attributions(
    attributions: Sequence[np.ndarray], random_seed: int
) -> list[np.ndarray]:
    """Permute each observation's attributions to build a random ranking control.

    Permuting within an observation holds its value distribution fixed and
    randomizes only which word receives which value. The comparison therefore
    isolates the quality of the ranking instead of confounding it with scale.
    """
    generator = np.random.default_rng(random_seed)
    return [generator.permutation(attribution) for attribution in attributions]


def combine_into_chunks(
    attribution: np.ndarray, chunk_size: int, combination: str
) -> tuple[np.ndarray, np.ndarray]:
    """Score consecutive runs of words as single units.

    Returns the importance of each chunk together with the chunk index of every
    word, so a decision taken per chunk can be expanded back to word positions.

    The three combinations answer different questions, and they disagree exactly
    when a chunk holds attributions that cancel:

    - `absolute_sum` takes the size of the summed signed values. Shapley
      additivity makes this the predicted output shift from removing the whole
      chunk, so it is the theoretically motivated choice for this measurement.
      A chunk holding +0.05 and -0.05 scores 0, correctly marking it as cheap to
      remove.
    - `sum_absolute` totals the magnitudes, measuring how much attribution
      activity a chunk contains regardless of direction. The same cancelling
      chunk scores 0.10 and is treated as important.
    - `max_absolute` scores a chunk by its single strongest word, so one decisive
      word protects the words around it.

    At `chunk_size` 1 every chunk holds one word and all three reduce to the
    absolute value of that word, which makes them a useful consistency check.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")

    chunk_starts = np.arange(0, len(attribution), chunk_size)
    chunk_of_word = np.arange(len(attribution)) // chunk_size

    if combination == ABSOLUTE_SUM:
        chunk_importance = np.abs(np.add.reduceat(attribution, chunk_starts))
    elif combination == SUM_ABSOLUTE:
        chunk_importance = np.add.reduceat(np.abs(attribution), chunk_starts)
    elif combination == MAX_ABSOLUTE:
        chunk_importance = np.maximum.reduceat(np.abs(attribution), chunk_starts)
    else:
        raise ValueError(f"combination must be one of {COMBINATIONS}, got {combination!r}")

    return chunk_importance, chunk_of_word


def select_kept_words(
    attribution: np.ndarray,
    percentile: float,
    chunk_size: int = 1,
    combination: str = ABSOLUTE_SUM,
    cut_most_important: bool = False,
) -> np.ndarray:
    """Boolean word mask marking which words survive a percentile cut.

    The threshold is taken over this observation's own chunk importances, so
    the percentile is roughly the share of the text that gets cut and stays
    comparable across reviews of very different attribution scales. By default
    the least important chunks are cut, which is the sufficiency direction;
    with `cut_most_important` the most important ones are cut instead.

    The measurement functions below cut through this exact mask, so anything
    rendered with it shows precisely the text they scored.
    """
    chunk_importance, chunk_of_word = combine_into_chunks(
        attribution, chunk_size, combination
    )
    if cut_most_important:
        threshold = np.percentile(chunk_importance, 100 - percentile)
        kept_chunks = chunk_importance < threshold
    else:
        threshold = np.percentile(chunk_importance, percentile)
        kept_chunks = chunk_importance >= threshold
    return kept_chunks[chunk_of_word]


def measure_sufficiency(
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    percentiles: Sequence[float],
    masker: shap.maskers.Text,
    attributions: Sequence[np.ndarray],
    chunk_size: int = 1,
    combination: str = ABSOLUTE_SUM,
) -> np.ndarray:
    """Absolute probability shift after cutting the least important chunks.

    At each percentile the threshold is taken over that observation's own chunk
    importances, and every chunk below it is masked. Lower is better: a faithful
    ranking discards only noise.

    Returns an array of shape (len(texts), len(percentiles)).
    """
    return _measure_probability_shift(
        predict_probabilities,
        texts,
        percentiles,
        masker,
        attributions,
        chunk_size,
        combination,
        cut_most_important=False,
    )


def measure_comprehensiveness(
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    percentiles: Sequence[float],
    masker: shap.maskers.Text,
    attributions: Sequence[np.ndarray],
    chunk_size: int = 1,
    combination: str = ABSOLUTE_SUM,
) -> np.ndarray:
    """Absolute probability shift after cutting the most important chunks.

    The mirror image of sufficiency. Higher is better: removing the text the
    model genuinely relies on should disrupt its output.

    Returns an array of shape (len(texts), len(percentiles)).
    """
    return _measure_probability_shift(
        predict_probabilities,
        texts,
        percentiles,
        masker,
        attributions,
        chunk_size,
        combination,
        cut_most_important=True,
    )


def _measure_probability_shift(
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    percentiles: Sequence[float],
    masker: shap.maskers.Text,
    attributions: Sequence[np.ndarray],
    chunk_size: int,
    combination: str,
    cut_most_important: bool,
) -> np.ndarray:
    """Cut chunks at each percentile and report how far the output moved.

    Every reduced text across all observations and percentiles goes through a
    single prediction call, because the embedding path already batches and
    length sorts internally. Predicting one observation at a time would waste
    most of the available throughput.
    """
    if len(attributions) != len(texts):
        raise ValueError(
            f"Got {len(attributions)} attribution arrays for {len(texts)} texts"
        )

    original_probabilities = predict_probabilities(texts)
    predicted_classes = original_probabilities.argmax(axis=1)

    reduced_texts = []
    for text, attribution in zip(texts, attributions):
        if len(attribution) != len(masker.data_transform(text)[0]):
            raise ValueError(
                "Attribution length does not match the masker's word count. "
                "The attributions must come from the same masker used here."
            )
        for percentile in percentiles:
            kept_words = select_kept_words(
                attribution, percentile, chunk_size, combination, cut_most_important
            )
            reduced_texts.append(masker(kept_words, text)[0][0])

    reduced_probabilities = predict_probabilities(reduced_texts)
    reduced_by_observation = reduced_probabilities.reshape(
        len(texts), len(percentiles), -1
    )

    observation_indices = np.arange(len(texts))
    original_for_predicted_class = original_probabilities[
        observation_indices, predicted_classes
    ]
    reduced_for_predicted_class = reduced_by_observation[
        observation_indices, :, predicted_classes
    ]

    return np.abs(
        reduced_for_predicted_class - original_for_predicted_class[:, np.newaxis]
    )
