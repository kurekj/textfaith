"""Run the multi dataset attribution benchmark: timed computation, caching, tidy results.

The benchmark's claims are comparative, SHAP versus LIME at an equal evaluation
budget, so everything here is built around fairness and reproducibility: both
methods drive the same counted predictor, timings are recorded next to the
attributions they measure, and every expensive result is cached under a
filename that encodes its parameters so a differently configured run can never
be silently reused.
"""

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import ttest_1samp, wilcoxon

from textfaith.faithfulness import (
    compute_lime_word_attributions,
    compute_occlusion_word_attributions,
    compute_random_word_attributions,
    compute_word_attributions,
    measure_comprehensiveness,
    measure_sufficiency,
    paired_advantage,
    shuffle_attributions,
)

SHAP_BATCH_SIZE = 64


class PredictionCounter:
    """Wrap a predictor and count how many texts it is asked to score.

    SHAP's evaluation budget is a cap rather than an exact count, short texts
    resolve their partition tree early and stop under budget, so the honest
    speed comparison needs the number of predictions actually made, not the
    number allowed.
    """

    def __init__(self, predict_probabilities: Callable[[Sequence[str]], np.ndarray]):
        self._predict_probabilities = predict_probabilities
        self.prediction_count = 0

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        self.prediction_count += len(texts)
        return self._predict_probabilities(texts)


def load_or_compute(cache_path: Path, compute: Callable[[], object]) -> object:
    """Return the cached result at `cache_path`, computing and caching it if absent."""
    if cache_path.exists():
        return joblib.load(cache_path)
    result = compute()
    joblib.dump(result, cache_path)
    return result


def compute_timed_attributions(
    method_name: str,
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    masker: shap.maskers.Text,
    evaluation_budget: int | None,
    random_seed: int,
) -> dict:
    """Compute one method's attributions with wall time and prediction count recorded.

    Timing lives inside the returned payload because it cannot be reconstructed
    from a cache after the fact. Every method runs through the same counting
    wrapper. The budget means max_evals for SHAP and sample_count for LIME,
    which is the equal footing the benchmark is about; occlusion has no budget
    knob, its cost is fixed at one prediction per word, so it ignores the
    argument and callers should pass None.
    """
    counter = PredictionCounter(predict_probabilities)
    start = time.perf_counter()

    if method_name == "shap":
        attributions = compute_word_attributions(
            counter,
            texts,
            masker,
            evaluation_budget=evaluation_budget,
            batch_size=SHAP_BATCH_SIZE,
        )
    elif method_name == "lime":
        attributions = compute_lime_word_attributions(
            counter,
            texts,
            masker,
            random_seed=random_seed,
            sample_count=evaluation_budget,
        )
    elif method_name == "occlusion":
        attributions = compute_occlusion_word_attributions(counter, texts, masker)
    elif method_name == "random":
        attributions = compute_random_word_attributions(texts, masker, random_seed)
    else:
        raise ValueError(
            f"method_name must be 'shap', 'lime', 'occlusion' or 'random', "
            f"got {method_name!r}"
        )

    return {
        "attributions": attributions,
        "wall_seconds": time.perf_counter() - start,
        "prediction_count": counter.prediction_count,
        "parameters": {
            "method": method_name,
            "evaluation_budget": evaluation_budget,
            "observation_count": len(texts),
            "random_seed": random_seed,
        },
    }


def sweep_faithfulness(
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    masker: shap.maskers.Text,
    attributions_by_method: dict[str, list[np.ndarray]],
    percentiles: Sequence[float],
    chunk_sizes: Sequence[int],
    combinations: Sequence[str],
    random_seed: int,
    include_shuffled_controls: bool = True,
) -> list[dict]:
    """Measure both directions for each attribution source over the grid.

    By default each source also gets its own shuffled control, because chunk
    combining consumes the attribution values, not only their ranks, so a fair
    control must preserve the source's own value distribution. When the
    comparison instead carries an explicit random attribution arm, pass
    `include_shuffled_controls=False` and the sources are measured as given.
    """
    sources = {}
    for method_name, attributions in attributions_by_method.items():
        sources[method_name] = attributions
        if include_shuffled_controls:
            sources[f"{method_name}_random"] = shuffle_attributions(
                attributions, random_seed
            )

    records = []
    for chunk_size in chunk_sizes:
        for combination in combinations:
            for source, attributions in sources.items():
                for direction, measure in [
                    ("sufficiency", measure_sufficiency),
                    ("comprehensiveness", measure_comprehensiveness),
                ]:
                    records.append(
                        {
                            "chunk_size": chunk_size,
                            "combination": combination,
                            "source": source,
                            "direction": direction,
                            "values": measure(
                                predict_probabilities,
                                texts,
                                percentiles,
                                masker,
                                attributions,
                                chunk_size=chunk_size,
                                combination=combination,
                            ),
                        }
                    )
    return records


def compute_original_probabilities(
    predict_probabilities: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
) -> pd.DataFrame:
    """One row per observation: the predicted class and its full text probability.

    This is the denominator that turns raw probability shifts into relative
    ones. It reproduces exactly the original probabilities the measurement
    functions computed internally, same texts, same model, and is cheap enough
    to cache once per dataset.
    """
    probabilities = predict_probabilities(texts)
    predicted_classes = probabilities.argmax(axis=1)
    return pd.DataFrame(
        {
            "observation_index": np.arange(len(texts)),
            "predicted_class": predicted_classes,
            "original_probability": probabilities[
                np.arange(len(texts)), predicted_classes
            ],
        }
    )


def attach_relative_shift(
    tidy_frame: pd.DataFrame, original_probabilities: pd.DataFrame
) -> pd.DataFrame:
    """Add `original_probability` and `relative_shift` columns to the tidy frame.

    `relative_shift` is the probability shift divided by the model's original
    predicted class probability: the share of the model's confidence the cut
    moved. Raw shifts are probability points and inherit each head's
    calibration, so they are not comparable across datasets whose heads differ
    in confidence; the relative form is bounded in [0, 1] because the
    predicted class starts at or above one half and can at most fall to zero.

    Within one observation every source shares the denominator, so the sign of
    every paired per review difference is unchanged by the rescaling; only the
    weighting of observations in aggregate statistics shifts.

    `original_probabilities` needs columns dataset, observation_index and
    original_probability.
    """
    merged = tidy_frame.merge(
        original_probabilities[
            ["dataset", "observation_index", "original_probability"]
        ],
        on=["dataset", "observation_index"],
        how="left",
    )
    if merged["original_probability"].isnull().any():
        raise ValueError(
            "Some observations have no original probability, "
            "the datasets or observation indices do not match"
        )
    merged["relative_shift"] = (
        merged["probability_shift"] / merged["original_probability"]
    )
    return merged


def average_shifts_over_thresholds(
    tidy_frame: pd.DataFrame, value_column: str = "probability_shift"
) -> pd.DataFrame:
    """Collapse the threshold dimension into one score per observation.

    The mean shift across the cut thresholds, known in the literature as
    AOPC, the area over the perturbation curve up to the deepest threshold
    measured. One scalar per observation and condition is the unit a paper's
    statistics want: the per review pairing stays explicit, and a comparison
    needs a single test per condition rather than an average of per threshold
    statistics.

    `value_column` selects the scale: `probability_shift` for raw probability
    points, `relative_shift` for the share of the model's original confidence,
    which is the scale that is comparable across datasets. The output column
    is named `average_shift` either way, so downstream comparisons work on
    both scales unchanged.
    """
    group_columns = [
        "dataset", "source", "method", "budget", "is_random",
        "chunk_size", "combination", "direction", "observation_index",
    ]
    return (
        tidy_frame.groupby(group_columns, as_index=False)[value_column]
        .mean()
        .rename(columns={value_column: "average_shift"})
    )


def compare_methods(
    average_shifts: pd.DataFrame,
    source_pairs: Sequence[tuple[str, str]] = (("shap_500", "lime_500"),),
) -> pd.DataFrame:
    """Pairwise comparison of attribution sources on the threshold averaged score.

    For every (first, second) pair, positive numbers favor `first` in both
    directions: sufficiency is better when lower, so its difference is second
    minus first, comprehensiveness is better when higher, so it is first minus
    second. Each condition gets the mean per observation difference, a single
    paired t statistic, a distribution free Wilcoxon signed rank p value on
    the same differences, and the share of observations where `first` is
    strictly better. Shuffled controls are excluded, this compares rankings,
    not the floors. Pairs can compare methods at one budget or one method
    across budgets, which is how the budget power question is asked.
    """
    scored_sources = average_shifts[~average_shifts["is_random"]]
    group_keys = ["dataset", "chunk_size", "combination", "direction"]
    rows = []
    for first, second in source_pairs:
        pair_scores = scored_sources[scored_sources["source"].isin([first, second])]
        for keys, group in pair_scores.groupby(group_keys, sort=False):
            by_source = group.pivot_table(
                index="observation_index", columns="source", values="average_shift"
            )
            direction = keys[group_keys.index("direction")]
            if direction == "sufficiency":
                differences = (by_source[second] - by_source[first]).to_numpy()
            else:
                differences = (by_source[first] - by_source[second]).to_numpy()
            if np.allclose(differences, 0):
                wilcoxon_p = 1.0
            else:
                wilcoxon_p = float(wilcoxon(differences).pvalue)
            rows.append(
                {"comparison": f"{first} vs {second}"}
                | dict(zip(group_keys, keys))
                | {
                    "average_shift_first": float(by_source[first].mean()),
                    "average_shift_second": float(by_source[second].mean()),
                    "first_advantage": float(differences.mean()),
                    "paired_t": float(ttest_1samp(differences, 0).statistic),
                    "wilcoxon_p": wilcoxon_p,
                    "first_win_share": float((differences > 0).mean()),
                }
            )
    return pd.DataFrame(rows).set_index(["comparison"] + group_keys)


def shift_matrix(
    tidy_frame: pd.DataFrame,
    dataset_name: str,
    source: str,
    chunk_size: int,
    combination: str,
    direction: str,
) -> np.ndarray:
    """Recover the (observations, percentiles) matrix for one condition.

    The inverse of `records_to_tidy_frame` for a single grid cell, which is
    what the paired statistics consume.
    """
    subset = tidy_frame[
        (tidy_frame["dataset"] == dataset_name)
        & (tidy_frame["source"] == source)
        & (tidy_frame["chunk_size"] == chunk_size)
        & (tidy_frame["combination"] == combination)
        & (tidy_frame["direction"] == direction)
    ]
    return subset.pivot_table(
        index="observation_index", columns="percentile", values="probability_shift"
    ).to_numpy()


def build_paired_grid(tidy_frame: pd.DataFrame, budget: int = 500) -> pd.DataFrame:
    """Summarize SHAP versus LIME at one budget into means and paired advantages.

    One row per dataset, chunk size and combination. The `*_shap_advantage`
    columns compare the methods head to head, positive favors SHAP in both
    directions because sufficiency is better when lower and comprehensiveness
    when higher. The `*_vs_control` columns compare each method against its
    own shuffled control, positive means the method beats a random ranking.
    Magnitudes above roughly 2 exceed sampling noise.
    """
    source_names = {
        "shap": f"shap_{budget}",
        "lime": f"lime_{budget}",
        "shap_random": f"shap_{budget}_random",
        "lime_random": f"lime_{budget}_random",
    }
    rows = []
    for dataset_name in tidy_frame["dataset"].unique():
        for chunk_size in sorted(tidy_frame["chunk_size"].unique()):
            for combination in tidy_frame["combination"].unique():
                matrices = {
                    (source, direction): shift_matrix(
                        tidy_frame,
                        dataset_name,
                        source_names[source],
                        chunk_size,
                        combination,
                        direction,
                    )
                    for source in ["shap", "lime", "shap_random", "lime_random"]
                    for direction in ["sufficiency", "comprehensiveness"]
                }
                rows.append(
                    {
                        "dataset": dataset_name,
                        "chunk_size": chunk_size,
                        "combination": combination,
                        "sufficiency_shap": matrices[("shap", "sufficiency")].mean(),
                        "sufficiency_lime": matrices[("lime", "sufficiency")].mean(),
                        "sufficiency_shap_advantage": paired_advantage(
                            matrices[("lime", "sufficiency")],
                            matrices[("shap", "sufficiency")],
                        ),
                        "sufficiency_shap_vs_control": paired_advantage(
                            matrices[("shap_random", "sufficiency")],
                            matrices[("shap", "sufficiency")],
                        ),
                        "sufficiency_lime_vs_control": paired_advantage(
                            matrices[("lime_random", "sufficiency")],
                            matrices[("lime", "sufficiency")],
                        ),
                        "comprehensiveness_shap": matrices[
                            ("shap", "comprehensiveness")
                        ].mean(),
                        "comprehensiveness_lime": matrices[
                            ("lime", "comprehensiveness")
                        ].mean(),
                        "comprehensiveness_shap_advantage": paired_advantage(
                            matrices[("shap", "comprehensiveness")],
                            matrices[("lime", "comprehensiveness")],
                        ),
                        "comprehensiveness_shap_vs_control": paired_advantage(
                            matrices[("shap", "comprehensiveness")],
                            matrices[("shap_random", "comprehensiveness")],
                        ),
                        "comprehensiveness_lime_vs_control": paired_advantage(
                            matrices[("lime", "comprehensiveness")],
                            matrices[("lime_random", "comprehensiveness")],
                        ),
                    }
                )
    return pd.DataFrame(rows).set_index(["dataset", "chunk_size", "combination"])


def records_to_tidy_frame(
    records: Sequence[dict], dataset_name: str, percentiles: Sequence[float]
) -> pd.DataFrame:
    """Flatten sweep records into one row per observation and percentile.

    The long format is the one statistics start from: `observation_index` keys
    the pairing across sources within a dataset, and `method`, `budget` and
    `is_random` make the contrasts plain groupbys. Source names follow the
    `method_budget` convention, for example `shap_500` or `lime_50`, with an
    optional `_random` suffix for shuffled controls; a source without a budget
    suffix, occlusion, gets the budget label `word_count` because its cost is
    one prediction per word.
    """
    rows = []
    for record in records:
        values = record["values"]
        source = record["source"]
        is_random = source.endswith("_random")
        base_source = source.removesuffix("_random")
        if "_" in base_source:
            method, budget = base_source.rsplit("_", 1)
        elif base_source == "random":
            method, budget = "random", "none"
        else:
            method, budget = base_source, "word_count"
        for observation_index in range(values.shape[0]):
            for percentile_index, percentile in enumerate(percentiles):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "source": source,
                        "method": method,
                        "budget": budget,
                        "is_random": is_random,
                        "chunk_size": record["chunk_size"],
                        "combination": record["combination"],
                        "direction": record["direction"],
                        "percentile": percentile,
                        "observation_index": observation_index,
                        "probability_shift": float(values[observation_index, percentile_index]),
                    }
                )
    return pd.DataFrame(rows)
