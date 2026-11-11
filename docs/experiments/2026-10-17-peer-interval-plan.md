# Contextual peer intervals — October 17, 2026

## Purpose and limits

The recent win-association search has saturated. This experiment instead trains expected
statistic ranges for postgame player analysis: combat efficiency, openings, damage, assists,
objectives, and clutches. It asks whether comparable historical performances usually fall
within a predicted interval. It does not measure causal contribution or validate coaching advice.

This protocol is recorded before training. The original test partition is excluded from all
fits and predictions. The final diagnostic segment is **previously inspected development
validation**, not an independent test. No model is automatically activated in the application.

## Fixed protocol

1. Verify the source and ML-code fingerprints from `player-20261016-v2`. Preserve its original
   training partition and chronological 480-series selection, 238-series calibration, and
   243-series diagnostic partitions. Keep all rows from each series together.
2. Use only agent, map, patch, match-length band, and round count as inputs. Player/team
   identities, team outcomes, and the observed performance statistics are excluded from inputs.
3. Fit 10th, 50th, and 90th conditional quantiles for each of six components with one fixed
   histogram-gradient-boosting recipe: 300 iterations, 15 leaves, minimum 100 rows per leaf,
   learning rate 0.05, seed 42, no random early stopping. Fit categorical encoding on training
   rows only; unseen categories become missing. Three existing expanding chronological training
   folds plus the full training refit require **72 boosted regressor fits**.
4. Compare with empirical training-only quantiles: agent/map peers with at least 100 rows,
   then agent peers with at least 100, then the global training distribution. Order predicted
   quantiles deterministically and restrict inherently nonnegative components to their support.
5. Normalize pinball loss and interval score using training-only component standard deviations.
   Use the learned model for a component only when mean pinball loss is lower in every training
   fold and in the earlier selection period. Freeze that choice before calibration or final
   diagnostic inference. Do not tune the recipe based on diagnostic results.
6. Expand intervals using residuals from the separate calibration period. For nominal 80%
   coverage, use the finite-sample order statistic at rank `ceil((n + 1) * 0.8)`, capped at `n`,
   with expansion bounded below by zero. Calibrate the empirical baseline independently on
   the same period for a fair comparison. Preserve the median estimate.
7. Report raw and calibrated coverage, lower/upper miss rates, interval width, interval score,
   and pinball loss by component. Include coverage slices by map, agent, and match length and
   paired series-bootstrap uncertainty for interval-score differences.

Rows within a series are dependent and the historical population shifts through time. The
calibration is therefore assessed empirically; **no IID conformal coverage guarantee is claimed**.
Sparse objectives and clutches have ties and discrete support, so exact nominal coverage or
exact percentile frequencies should not be expected.

## Verification and outputs

Save a portable model bundle, frozen selection, protocol, script snapshot, source hashes,
metrics, keyed diagnostic predictions, and checksums under `data/research/peer-intervals-20261017`.
Check full prediction reload equality, allowed-input isolation, identity/outcome/statistic
mutation invariance, unknown-category handling, finite ordered intervals, support restrictions,
chronological isolation, empirical fallback, and calibration-rank arithmetic. Retain unsuccessful
results. The research interval estimates remain separate from the active heuristic index.
