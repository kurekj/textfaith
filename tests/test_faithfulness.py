"""Unit tests for the faithfulness measurements on a deterministic toy model.

The toy classifier scores class 1 by the presence of the word "good", so the
ground truth attribution is known exactly: all importance sits on that one
word. Every property tested here follows from that construction rather than
from any real model, which keeps the suite fast and free of network downloads.
"""

import numpy as np
import pytest

from textfaith.experiment import PredictionCounter
from textfaith.faithfulness import (
    ABSOLUTE_SUM,
    MAX_ABSOLUTE,
    SUM_ABSOLUTE,
    build_word_masker,
    combine_into_chunks,
    compute_lime_word_attributions,
    compute_occlusion_word_attributions,
    compute_random_word_attributions,
    compute_word_attributions,
    measure_comprehensiveness,
    measure_sufficiency,
    paired_advantage,
    select_kept_words,
    shuffle_attributions,
)

RANDOM_SEED = 42


def predict_by_good(texts):
    """Class 1 probability is 0.9 when "good" survives anywhere in the text, 0.1 otherwise.

    Substring matching rather than token matching, because the masker glues its
    "..." marker to the following word ("a ...good movie"), which would make a
    token-based toy model flip when a neighbor of "good" is masked.
    """
    probabilities = []
    for text in texts:
        positive = 0.9 if "good" in text else 0.1
        probabilities.append([1.0 - positive, positive])
    return np.array(probabilities)


def test_masker_splits_on_whitespace():
    masker = build_word_masker()
    words = masker.data_transform("isn't a bad movie at all")[0]
    assert len(words) == 6


def test_combine_into_chunks_disagrees_exactly_on_cancellation():
    attribution = np.array([0.05, -0.05, 0.1, 0.2])
    absolute_sum, _ = combine_into_chunks(attribution, 2, ABSOLUTE_SUM)
    sum_absolute, _ = combine_into_chunks(attribution, 2, SUM_ABSOLUTE)
    max_absolute, _ = combine_into_chunks(attribution, 2, MAX_ABSOLUTE)
    np.testing.assert_allclose(absolute_sum, [0.0, 0.3])
    np.testing.assert_allclose(sum_absolute, [0.1, 0.3])
    np.testing.assert_allclose(max_absolute, [0.05, 0.2])


def test_combine_into_chunks_reduces_to_absolute_value_at_size_one():
    attribution = np.array([0.3, -0.7, 0.1])
    for combination in (ABSOLUTE_SUM, SUM_ABSOLUTE, MAX_ABSOLUTE):
        importance, chunk_of_word = combine_into_chunks(attribution, 1, combination)
        np.testing.assert_allclose(importance, np.abs(attribution))
        np.testing.assert_array_equal(chunk_of_word, [0, 1, 2])


def test_combine_into_chunks_rejects_bad_arguments():
    with pytest.raises(ValueError):
        combine_into_chunks(np.array([0.1]), 0, ABSOLUTE_SUM)
    with pytest.raises(ValueError):
        combine_into_chunks(np.array([0.1]), 1, "geometric_mean")


def test_select_kept_words_directions_are_mirrors():
    attribution = np.array([0.1, 0.9, 0.2, 0.8])
    kept_low_cut = select_kept_words(attribution, 50)
    kept_high_cut = select_kept_words(attribution, 50, cut_most_important=True)
    np.testing.assert_array_equal(kept_low_cut, [False, True, False, True])
    np.testing.assert_array_equal(kept_high_cut, [True, False, True, False])


def test_shuffle_attributions_permutes_and_reproduces():
    attributions = [np.array([1.0, 2.0, 3.0, 4.0])]
    shuffled = shuffle_attributions(attributions, RANDOM_SEED)
    assert sorted(shuffled[0].tolist()) == [1.0, 2.0, 3.0, 4.0]
    again = shuffle_attributions(attributions, RANDOM_SEED)
    np.testing.assert_array_equal(shuffled[0], again[0])


def test_random_attributions_match_word_counts_and_seed():
    masker = build_word_masker()
    texts = ["one two three", "four five"]
    first = compute_random_word_attributions(texts, masker, RANDOM_SEED)
    second = compute_random_word_attributions(texts, masker, RANDOM_SEED)
    assert [len(a) for a in first] == [3, 2]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_occlusion_finds_the_decisive_word():
    masker = build_word_masker()
    texts = ["a truly good movie"]
    attributions = compute_occlusion_word_attributions(predict_by_good, texts, masker)
    assert int(np.argmax(attributions[0])) == 2
    assert attributions[0][2] == pytest.approx(0.8)
    assert attributions[0][0] == pytest.approx(0.0)


def test_shap_attributions_align_with_the_masker():
    masker = build_word_masker()
    texts = ["a truly good movie", "such a bad film"]
    attributions = compute_word_attributions(
        predict_by_good, texts, masker, evaluation_budget=16
    )
    assert [len(a) for a in attributions] == [4, 4]
    assert int(np.argmax(np.abs(attributions[0]))) == 2


def test_lime_attributions_align_with_the_masker():
    masker = build_word_masker()
    texts = ["a truly good movie"]
    attributions = compute_lime_word_attributions(
        predict_by_good, texts, masker, RANDOM_SEED, sample_count=100
    )
    assert len(attributions[0]) == 4
    assert int(np.argmax(np.abs(attributions[0]))) == 2


def test_measurements_reward_the_true_ranking():
    masker = build_word_masker()
    texts = ["a truly good movie indeed", "just a good little film"]
    perfect = [
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    ]
    sufficiency = measure_sufficiency(
        predict_by_good, texts, [50], masker, perfect
    )
    comprehensiveness = measure_comprehensiveness(
        predict_by_good, texts, [50], masker, perfect
    )
    assert sufficiency.shape == (2, 1)
    assert comprehensiveness.shape == (2, 1)
    # Cutting the words the toy model ignores moves nothing; cutting "good"
    # collapses the prediction by construction.
    assert float(sufficiency.max()) == pytest.approx(0.0)
    assert float(comprehensiveness.min()) == pytest.approx(0.8)


def test_measurements_reject_mismatched_attributions():
    masker = build_word_masker()
    with pytest.raises(ValueError):
        measure_sufficiency(
            predict_by_good, ["two words"], [50], masker, [np.array([1.0])]
        )


def test_prediction_counter_counts_texts_not_calls():
    counter = PredictionCounter(predict_by_good)
    counter(["one", "two"])
    counter(["three"])
    assert counter.prediction_count == 3


def test_paired_advantage_is_positive_when_first_wins():
    generator = np.random.default_rng(RANDOM_SEED)
    worse = generator.normal(0.0, 0.01, size=(50, 4))
    better = worse + 0.05
    assert paired_advantage(better, worse) > 2.0
