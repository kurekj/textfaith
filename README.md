# textfaith

Deletion faithfulness benchmark for word attribution methods in text classification. Scores SHAP, LIME, occlusion and a random baseline under identical removal semantics, at counted evaluation budgets, with paired per-document statistics.

## Quick Start

```python
import numpy as np
from textfaith.faithfulness import (
    build_word_masker,
    compute_word_attributions,       # SHAP partition explainer
    measure_sufficiency,
    measure_comprehensiveness,
)

# Any callable mapping a list of texts to an (n, classes) probability array works,
# scikit-learn pipelines and transformer classifiers alike.
def predict_probabilities(texts):
    ...  # your model here

texts = ["The movie was a complete waste of time.", "A quiet, wonderful little film."]
masker = build_word_masker()

attributions = compute_word_attributions(
    predict_probabilities, texts, masker, evaluation_budget=500,
)

# Cut the least important words: a faithful ranking barely moves the model (lower = better).
sufficiency = measure_sufficiency(
    predict_probabilities, texts, percentiles=[50], masker=masker, attributions=attributions,
)
# Cut the most important words: a faithful ranking moves it a lot (higher = better).
comprehensiveness = measure_comprehensiveness(
    predict_probabilities, texts, percentiles=[50], masker=masker, attributions=attributions,
)
```

LIME, leave-one-word-out occlusion and a gaussian random baseline are scored the same way through `compute_lime_word_attributions`, `compute_occlusion_word_attributions` and `compute_random_word_attributions`, so all four sources are compared on equal footing: same masker, same cutting scheme, only the ranking differs.

## Installation

```bash
git clone https://github.com/kurekj/textfaith.git
cd textfaith
uv sync            # Python 3.13, CPU-only PyTorch
```

Or with pip: `pip install .`

## Quick demo

A self-contained run of the whole protocol, about ten seconds on CPU, no downloads:

```bash
uv run python benchmarks/run_demo.py
```

It trains a small classifier on a synthetic corpus with planted sentiment words, scores SHAP, LIME, occlusion and the random baseline through the full measurement, tests each method against random, and renders a heat map. See `benchmarks/README.md`.

## What it measures

A word attribution method claims to rank words by their contribution to a classifier's prediction. `textfaith` tests that claim directly:

1. **Sufficiency** — delete the *least* important spans; a faithful ranking discards only noise, so the output should barely move (lower is better).
2. **Comprehensiveness** — delete the *most* important spans; removing the text the model relies on should disrupt the output (higher is better).

Deletions operate on **chunks** of 1, 4 or 8 consecutive words (so the surviving text still reads as language), chunk scores are built from word attributions by three combination rules (`absolute_sum`, `sum_absolute`, `max_absolute`), and all scores are relative to the model's original predicted-class confidence, making them comparable across tasks. Every comparison is paired per document and tested with a Wilcoxon signed-rank test with Holm correction.

## Benchmark results

Three binary classification tasks (IMDB and Yelp sentiment, AG News topic; frozen e5-small-v2 encoder + MLP head, test accuracy 0.89–0.96), 300 explained documents per dataset, budgets of 50/200/500 model calls per document, 54 conditions per comparison:

| Comparison | Result |
|---|---|
| SHAP vs random baseline | wins 54/54 Holm-corrected conditions, loses 0 |
| LIME vs random baseline | wins 54/54, loses 0 |
| Occlusion vs random baseline | wins 52/54, loses 0 |
| SHAP vs LIME (budget 500) | 31/54 conditions significant: 25 favor SHAP, 6 LIME |
| Cost per document (budget 500) | SHAP 7.8 s, LIME 18.7 s |
| `absolute_sum` vs other combinations | wins 24/48, loses 0 |

## Repository layout

| Path | Contents |
| --- | --- |
| `src/textfaith/` | The library: attribution sources, faithfulness measurements, training pipeline, checkpoint loader, heat-map renderer |
| `benchmarks/` | The reproducible capsule: a self-contained demo of the full protocol with committed results |
| `notebooks/` | The benchmark experiment, numbered in execution order |
| `reports/` | Generated HTML reports and the bibliography |
| `tests/` | Unit tests (`uv run pytest`) |

## Reproducing the benchmark

`notebooks/01_train.ipynb` and `notebooks/05_additional_datasets.ipynb` download the corpora (IMDB from Kaggle, Yelp and AG News from the fast.ai archives, about 700 MB on disk, no credentials needed), `notebooks/08_datasets_v2.ipynb` builds the checkpoints from them, `notebooks/09_benchmark_v2.ipynb` computes attributions and runs the deletion sweep (about 11 hours on an 8-core CPU at the full grid), and `notebooks/10_report_2.ipynb` turns the tidy result files into the generated report. Random seed is 42 throughout. See `DEVELOPMENT.md` for the full pipeline contract.

## Citation

If you use `textfaith` in your research, please cite:

> M. Bukowski, J. Kurek, "textfaith: measuring the deletion faithfulness of text classification explanations," *SoftwareX* (in preparation).

## License

MIT
