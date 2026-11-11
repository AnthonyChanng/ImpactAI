# Measured evaluation

Frozen artifact version: `v1-1c8c8207cf73`. Seed: 42. No modern ranked claims.

| Selected calibrated model | Test rows | Log loss | Brier | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Round / xgboost_depth5 | 39,585 | 0.635601 | 0.223530 | 0.664986 | 62.1700% |
| Retrospective player / xgboost | 19,190 | 0.362281 | 0.119348 | 0.914554 | 82.3398% |

## Research questions

- XGBoost beat logistic regression on selection-validation log loss: round 0.63621 vs 0.66389; player 0.35991 vs 0.36667. This comparison is observational, not a significance claim.
- Removing only first-kill rate while retaining kills/deaths and first-death rate gives validation log loss 0.37059, versus 0.36667 for full logistic. The incremental first-kill evidence is small; this is a validation-only supplemental diagnostic, not test tuning.
- Removing both opening rates gives validation loss 0.37175. Context plus ACS gives 0.63123, versus 0.63872 for ACS alone.
- Other-nine-player controls alone yield validation loss 0.11395 and AUC 0.99243. Adding player statistics yields loss 0.10675. This strong scoreboard reconstruction is why these controls are diagnostic and never the displayed index.
- Removing opening, clutch, objective, HS, and length-band inputs gives conservative player validation loss 0.37256. All remain postgame models; even the conservative model is not a pregame predictor.
- Removing score/history from the round model gives validation loss 0.69320. Economy and current-round outcomes are excluded from every experiment, not merely from a final ablation.
- Maximum team-swap complement error: 1.19e-07. The constant prior is 0.5 after perspective balancing; the raw source-team win rate is 58.1276%.

## Held-out map errors

| Map | Round rows | Log loss | ROC-AUC |
|---|---:|---:|---:|
| Ascent | 7,534 | 0.63147 | 0.67208 |
| Bind | 5,852 | 0.63310 | 0.67044 |
| Breeze | 4,212 | 0.62962 | 0.66914 |
| Fracture | 1,828 | 0.63294 | 0.68092 |
| Haven | 7,446 | 0.64038 | 0.65895 |
| Icebox | 7,615 | 0.63713 | 0.66013 |
| Split | 5,098 | 0.64122 | 0.65349 |

These are chronological within-map slices. Fracture has no training examples and therefore uses unknown-category handling; map-specific performance should be interpreted with its sample count. No leave-one-map-out retraining was performed.

## Patch variation

| Patch | Round rows | Log loss | ROC-AUC |
|---|---:|---:|---:|
| Patch 3.06 | 519 | 0.65478 | 0.64431 |
| Patch 3.07 | 2,557 | 0.63611 | 0.65988 |
| Patch 3.08 | 6,548 | 0.63805 | 0.66310 |
| Patch 3.09 | 11,765 | 0.63093 | 0.66634 |
| Patch 3.10 | 7,156 | 0.64330 | 0.65475 |
| Patch 3.12 | 7,256 | 0.63062 | 0.67316 |
| unknown | 3,784 | 0.63789 | 0.66762 |

## Error analysis and limits

`research-diagnostics.json` contains the 15 largest saved test-round probability errors, with pre-score, side, map, patch, and observed result. These are inspection examples, not grounds for retuning this held-out experiment.
Round predictions miss unobserved team skill, lineups, economy, utility, and other states. Large confident errors can reflect recovery from a score deficit; the available fields cannot explain the underlying play.
Temperature calibration is fit only on the second half of validation series. Confidence intervals use 300 series-cluster bootstrap draws; they do not cover deployment drift or causal uncertainty.
The player index uses equal weights and is explicitly heuristic. Leave-one-component-out correlations are stored in evaluation.json; correlated combat/damage components and absent opportunity counts remain limitations.
Metrics, per-map/side/patch slices, calibration bins, SHAP importances, model manifests, and figures are generated from actual training outputs. Training and validation historical UI rows are not held-out predictions.
