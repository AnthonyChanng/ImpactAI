# Component-specific player benchmarks — October 16, 2026

Completed the [exploratory follow-up protocol](2026-10-16-peer-followup-plan.md). The original benchmark won the first experiment overall; this follow-up uses learned means only for components that improved in all three training folds and the earlier selection segment. The rule and component choices were frozen before computing the combined model's final diagnostic predictions.

**This is reused historical development validation, not a new independent test.** The comparison covers 5,070 player-game observations across 243 series. No original test rows were fitted or predicted. The initial run had already inspected the hierarchical baseline on this segment.

Mean training-variance-normalized squared error decreased from **0.976376** to **0.974664**: **0.175% lower**. This is a small improvement in expected-statistic estimates, not evidence of a major improvement in overall player analysis.

| Component | Chosen source | Baseline error | Candidate error | Reduction |
|---|---|---:|---:|---:|
| Combat | boosted_medium | 1.101622 | 1.100265 | 0.123% |
| Openings | boosted_medium | 0.976879 | 0.974496 | 0.244% |
| Damage | boosted_medium | 1.165307 | 1.162913 | 0.205% |
| Assists | hierarchical | 0.869538 | 0.869538 | 0.000% |
| Objectives | hierarchical | 0.801336 | 0.801336 | 0.000% |
| Clutches | boosted_shallow | 0.943574 | 0.939434 | 0.439% |

Candidate-minus-baseline error: `-0.001712`; 95% series-bootstrap interval `[-0.0030982308917875505, -0.0002916321659665157]`. This describes the development comparison, not an independent generalization guarantee.

Combat and opening components may be signed; all expected values for nonnegative components were nonnegative on these 5,070 rows. Refits reproduced every earlier selection metric. Reloaded artifacts reproduced all predictions exactly, and changing observed player stats and outcome labels at inference did not change expected peer means.

The learned targets are combat, openings, damage, assists, objective participation, and clutch counts per round. Inputs are agent, map, patch, and match length. Neither team outcomes nor player/team identities are inputs. No new percentile distribution, causal contribution, or validated composite impact score is claimed.

Artifacts: `data/research/peer-components-20261016`. All 5 file hashes and the archived script hash were verified. The directory contains a portable model bundle, component sources, keyed actual/expected/baseline values, source snapshot, and frozen selection record.

```bash
python -m scripts.refine_peer_benchmarks ml/artifacts/research/player-20261016-v2 --output ml/artifacts/research/NEW_PEER_DIRECTORY
```

This follow-up fitted 12 additional boosted regressors. It did not revise any recipe or component choice after inspecting the combined model's diagnostic results. The candidate remains separate from the application's active heuristic index.
