# Training and research guide

[Back to the project](../README.md) · [Methodology](methodology.md) · [Measured results](evaluation.md)

## Training and evaluation

```bash
python -m ml.preprocessing.audit
python -m ml.train
python -m scripts.research_report
```

The authoritative supplied MatchID splits are preserved. The first half of validation series
selects models; the second half calibrates them. Test predictions are computed after that freeze.
`scripts.research_report` reads frozen predictions for error analysis and fits a **validation-only**
first-kill ablation; it does not retune or replace the final models. Completed identical training
versions are reused. ML code changes produce a new version; test-derived tuning should use a new
future holdout rather than repeatedly treating this test set as unseen.

Generated files:

```text
ml/data/<version>/round.parquet, player.parquet
ml/artifacts/<version>/feature-schema.json, audit.json, metrics.json
ml/artifacts/<version>/bundle.joblib, manifest.json, *.json (native tree models)
ml/artifacts/<version>/player-predictions.jsonl, round-predictions.parquet
ml/artifacts/<version>/calibration.png, feature-importance.png
ml/artifacts/current.json
docs/data-audit.json, evaluation.json, evaluation.md, research-diagnostics.json
```

For the separate, bounded improvement experiment (42 configurations across two chronological
training folds), run:

```bash
python -m ml.improve --output ml/artifacts/research/my-new-run --task both
python -m scripts.verify_research ml/artifacts/research/my-new-run
```

This excludes the original test partition from fitting, selection, calibration, and prediction.
Its confirmation segment is previously inspected validation data, so the results are development
evidence rather than a new independent test. Use a new output directory for each run. Research
bundles include additional historical/context inputs and are not automatically activated in the
dashboard. See the [declared protocol](experiments/2026-10-15-plan.md) and
[executed results](experiments/2026-10-15-results.md). A fresh chronological holdout is needed
before making new generalization claims.

For the subsequent **postgame player** experiment, including detailed individual features,
chronological early stopping, model averaging, and outcome-free peer-mean regression:

```bash
python -m ml.player_research --output ml/artifacts/research/my-player-run
python -m scripts.verify_player_research ml/artifacts/research/my-player-run --report docs/experiments/my-player-results.md
```

The [player research protocol](experiments/2026-10-16-plan.md) keeps outcome association and
expected-statistic benchmarks separate. It excludes original test rows and preserves a source
snapshot in each run. Models and keyed player comparisons are saved for review; research runs
do not replace the application's active model or turn win probabilities into impact scores.
See the [executed player results](experiments/2026-10-16-results.md) and the
[component-specific peer follow-up](experiments/2026-10-16-peer-followup-results.md).

The optional peer follow-up preserves a learned benchmark only where it improves every training
fold and the earlier selection segment:

```bash
python -m scripts.refine_peer_benchmarks ml/artifacts/research/my-player-run --output data/research/my-peer-run
```

The subsequent [alternative learner and calibration search](experiments/2026-10-16-final-search-results.md)
compares native categorical histogram boosting with the frozen XGBoost ensemble, then selects
calibration using chronologically separate fit/selection segments:

```bash
python -m scripts.optimize_player --parent ml/artifacts/research/player-20261016-v2 --output ml/artifacts/research/my-calibration-run
python -m scripts.verify_player_optimization ml/artifacts/research/my-calibration-run --report docs/experiments/my-calibration-results.md
```

`--parent` must point to a trusted completed `ml.player_research` run; substitute its path when
reproducing on another machine. This search retained the XGBoost ensemble and found only a tiny
calibration improvement on reused development validation. It does not update the active app.

The [contextual peer-range experiment](experiments/2026-10-17-peer-interval-results.md)
estimates 10th/50th/90th percentiles for six performance components using context only. It compares
72 quantile-regressor fits with empirical peer ranges, calibrates intervals on a separate period,
and reports coverage, tail misses, and interval scores, including components that worsened:

```bash
python -m scripts.peer_intervals ml/artifacts/research/player-20261016-v2 --output data/research/my-interval-run
python -m scripts.verify_peer_intervals data/research/my-interval-run --report docs/experiments/my-interval-results.md
```

The original test split is excluded. The dashboard can display the saved, experimental comparisons
for matching player-games in this research sample. `PEER_INTERVAL_DIR` defaults to
`data/research/peer-intervals-20261017`; it must contain `checksums.json`, `results.json`, and
`diagnostic-predictions.parquet`. The API verifies hashes and observation context before displaying
ranges and measured development coverage. Missing or invalid files leave ranges unavailable without
blocking ordinary analysis. Restart the API after changing research files. No research Joblib bundle
is loaded, no new research predictions are made, and the active score/model is not replaced.
Empirical coverage is not a guarantee for new matches or a validated causal-impact score.

Do not load externally supplied Joblib files. They are executable serialization. Manifest hashes
detect corruption; they are not cryptographic signatures. No API accepts uploaded model artifacts.
Keep data and generated models ignored until distribution rights are established.
