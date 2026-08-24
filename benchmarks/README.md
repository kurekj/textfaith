# Reproducible capsule

A complete run of the textfaith protocol that finishes in seconds on any CPU,
with no downloads: four attribution sources (SHAP, LIME, occlusion, random)
computed through one shared masker at a counted budget, sufficiency and
comprehensiveness over a percentile sweep at two chunk sizes, paired Wilcoxon
tests against the random baseline, and the heat map rendering.

```bash
uv run python benchmarks/run_demo.py
```

The corpus is synthetic: template reviews with three sentiment words planted in
known positions, classified by a TF-IDF logistic regression. Because the label
signal sits in known words, the heat map doubles as a correctness check by eye,
and because the model is just a callable returning class probabilities, the
demo also shows that the library is model agnostic.

`results/` holds the committed outputs of the demo (seed 42 throughout). A
rerun reproduces every measurement column exactly; only `wall_seconds` varies
between runs and machines.

| File | Contents |
| --- | --- |
| `demo_results.csv` | Mean relative sufficiency and comprehensiveness per source and chunk size, with wall time and counted predictions |
| `demo_vs_random.csv` | Paired Wilcoxon tests of each method against the random baseline |
| `demo_heatmap.png` | One review rendered under all four sources; the three real methods concentrate on the planted words, random does not |

The full benchmark of the accompanying article (three public datasets, 300
documents each, budgets 50/200/500) is reproduced by `notebooks/08_datasets_v2`
and `notebooks/09_benchmark_v2`; expect several hours of CPU time.
