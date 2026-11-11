# Methodology and boundaries

## Population and source integrity

The supplied cleaned CSVs describe professional VALORANT esports from **2020-07-31 through
2022-01-08**: 13,228 map-games, 6,404 series, 7,211 distinct player IDs, 132,280 player-game
observations, and 273,163 rounds. They do not represent modern ranked matchmaking. Dates are
source timestamps without a verified timezone; their original ordering is preserved.

`python -m ml.preprocessing.audit` validates required schemas, numeric types, missingness,
distributions, keys, complete five-person teams, source team outcomes, score increments,
game/round totals, regulation sides, cross-file context, and grouped chronology. It preserves
source bytes and records SHA-256 hashes. The source `build_impactai.py` needs four unavailable
raw CSVs and an original `/mnt/data` path. It is retained for provenance and never executed by
the clean-data pipeline. Rebuilding the original raw-to-clean filtering requires those raw inputs.

Discovered issues beyond the supplied quality report:

- Game **52669** is **14–14**, but the source assigns team 2 as winner through an `else` branch.
- Game **20210** is **5–6**, an incomplete final outcome.
- Their 20 player rows remain stored, with `team_won=null` and `outcome_reliable=false` in the
  normalized API; the exact original labels remain in raw JSON. These rows are excluded from
  supervised player outcome fitting/evaluation. Their consistent round histories remain eligible.
- Patch labels sometimes include descriptive text (up to 80 characters). They are preserved as
  supplied categorical labels, not truncated or silently normalized.
- Patch is missing in 6,540 player records; KAST in 129,250; headshot percentage in seven.
  Unknown categories remain explicit; numerical imputation fits only training rows. KAST and Econ
  are excluded. Headshot values are fractions, not whole percentages.
- 4,982 rounds have unknown sides, corresponding to overtime. No sides are invented.
- The raw source-team-1 round win rate is 58.13%, so order is not an exchangeable team label.
- Winner-to-abbreviation mappings can be checked for internal consistency, but cannot be
  independently rederived from original round-winner strings without the absent raw sources.

## Partition and evaluation protocol

| Supplied split | Series | Games | Source dates |
|---|---:|---:|---|
| Train | 4,482 | 9,118 | 2020-07-31 to 2021-07-20 15:00:00 |
| Validation | 961 | 2,191 | 2021-07-20 15:00:19 to 2021-10-12 |
| Test | 961 | 1,919 | 2021-10-13 to 2022-01-08 |

No row shuffling changes partition membership. All rows in a series stay together. Within
validation, chronological series 1–480 select minimum log loss; the other 481 fit a scalar
temperature in [0.5, 3] on logits. Temperature uses no intercept, preserving round symmetry.
Models, hyperparameters, and calibration are frozen before test evaluation. Every predeclared
candidate is evaluated on test for reporting, with no reselection. Confidence intervals resample
series 300 times (seed 42). Repeated player identities across periods are expected; no identity
feature is used. This is temporal generalization, not a new-player holdout.

The separate first-kill-only ablation in `scripts/research_report.py` uses validation only and
does not modify the selected model or repeat test evaluation. Error inspection reads already
saved predictions. For future development, reserve a new chronological holdout before tuning
against the findings of this test period.

## Pre-round probability

Explicit allowed inputs: map, patch, round number, pre-round team scores, score differential,
previous round's observed outcome (grouped shift within the same GameID), phase, and reconstructed
regulation side. The cumulative pre-scores are prior rounds won; no duplicate prior-win fields
are added. Round one has a neutral 0.5 previous outcome sentinel. Unknown overtime sides remain
`unknown`. Patch/map one-hot encoding handles unseen categories without using test-fitted means.

The cleaning script's pre-scores are independently recomputed from earlier outcomes during audit.
All current-round post-scores, win type, target, team IDs, and `ObservedTimingUnknown` economy
fields are excluded from feature construction. A test mutates outcome/economy columns and verifies
that the same round's features do not change.

Training duplicates each round with reversed team scores, sides, previous outcome, and target.
Inference computes `p_sym(x) = (p(x) + 1 - p(swap(x))) / 2`, then applies temperature calibration.
Thus team-order changes produce complementary predictions within floating-point tolerance. The
constant prior after balancing is 0.5. Logistic regression and two XGBoost depths are compared.
Score differential is redundant with pre-scores in the linear model; regularization stabilizes
coefficients, which are not interpreted causally. XGBoost uses a deterministic histogram method.

This estimates **pre-round win probability**, never event-level WPA. It does not observe team
strength, current lineup quality, verified economy, or tactical state. Calibration and map/side/
patch slices are reported, including unknown sides. See [measured evaluation](evaluation.md).

## Player association model

Each row contains one player's end-of-game statistics. Primary numerical inputs are KPR, DPR,
APR, ADR, first kills/round, first deaths/round, clutch count/round, plants+defuses/round, and HS
fraction. Context is agent, map, patch, and a postgame length band (≤18, 19–24, >24 rounds).
No player/team IDs, team-side labels, outcome-derived winner fields, KAST, Econ, plus-minus, or
simultaneously duplicated ACS/KD are primary inputs. Multi-kill rates are computed in the feature
dataset but omitted from the primary model to reduce overlap with kills and damage.

Baselines include prior, K/D, ACS, ADR, and ACS+context. Ablations remove openings, then remove
opening/objective/clutch/HS/length features. Other-nine-player controls use leave-self-out
teammate averages and opponent averages of KPR, DPR, ADR. Controls-only and controls+player
models diagnose incremental association conditional on the surrounding scoreboard. They are
never eligible for primary selection because the completed scoreboard nearly reconstructs the
outcome. Even the individual model is retrospective and must not be presented as a prediction
made before the game or as the player's share of team victory.

## Contextual index (explicitly heuristic)

Six components per player-game:

| Component | Observable quantity |
|---|---|
| Combat | (Kills − Deaths) / rounds |
| Openings | (FirstKills − FirstDeaths) / rounds |
| Damage | ADR |
| Assists | Assists / rounds |
| Objectives | (Plants + Defuses) / rounds |
| Clutches | (1v1 + 1v2 + 1v3 + 1v4 + 1v5 counts) / rounds |

References fit **training rows only**, including valid statistics from the incomplete training
game, without using its unreliable winner label. Start from global first and second moments,
then smooth through agent → agent/map → agent/map/patch → agent/map/patch/length band. Each level
uses weight `n / (n + 50)` for its local moments and the remaining weight for the parent. Variance
is reconstructed from second moments and floored at 10% of global variance. Standardized values
are clipped to [−3, 3]. The score is `clip(50 + 15 × mean(six component z-scores), 0, 100)`.

Equal weights are a transparent design choice, **not learned or validated causal weights**.
Empirical score percentiles use training scores from agent/map peers with at least 50 examples,
otherwise agent, otherwise the global reference. Midranks handle ties. Peer count, fallback level,
exact context count, raw component values, smoothed means, and z-scores are returned. Training
percentiles are descriptive in-sample references. Test scores never update reference distributions.
An average of game scores on a profile is a descriptive mean, not a percentile of player ability.

Damage and combat correlate; leave-one-component-out sensitivity is reported. Objective and clutch
counts lack opportunity denominators and can reflect role or assignment. Assists do not establish
utility quality. A low component is a review prompt, not proof of a mistake. This index is not MMR,
RR, Riot rank, causal attribution, or a validated universal skill score.

## Explanations and coaching

For selected tree models, TreeSHAP uses tree-path-dependent references and explains uncalibrated
log-odds. Local records store the baseline, eight largest terms, remaining-term sum, and total.
Applying the stored temperature to total logits reproduces the displayed association probability.
Global SHAP uses a deterministic 2,000-row test sample for the research tree; selected-model
global importance uses held-out rows. Linear fallback reports explicit standardized linear terms,
not mislabeled SHAP. Contributions explain model behavior, not causal effects.

Coaching retrieves a stored performance and up to nine same-agent games from strictly earlier
series. Templates render measured values and peer means with game/player/metric/version references.
A component at or below −0.5 standardized units is a review candidate. A repeated pattern requires
at least four prior games and at least 60% crossing that threshold; other candidates are labeled
isolated or limited-history. The report can also find no clear weakness. These thresholds are
heuristics awaiting human validation. Recommendations ask for VOD review without asserting unseen
actions. Cached reports refresh when their evidence signature or coaching revision changes.

An optional server-configured chat-completions-compatible provider may reorder evidence IDs for
presentation. Its free-form prose and invented numbers are never accepted. Invalid, duplicate,
missing, or unknown IDs trigger the deterministic fallback. This intentionally constrained MVP
does not offer unrestricted LLM narration. No paid provider was required for verification.

## Artifact and operational boundaries

Training writes versioned feature datasets, metrics, calibration figures, native XGBoost JSON,
materialized predictions, a Joblib bundle, and checksummed manifests. Source hashes plus ML code
identify a version; package versions are recorded. Joblib is executable serialization: load only
locally trained, trusted artifacts. Hashes detect corruption, not a maliciously rewritten manifest.

PostgreSQL stores normalized observations and separately versioned predictions. The importer is
transactional and idempotent; completed matching versions are verified by hashes and row counts.
To ingest changed source bytes under overlapping IDs, use a new database or an explicitly designed
source revision migration; the importer does not silently overwrite observations. Restart API
processes after changing the active artifact version. Overview caching lasts at most 120 seconds.

This is a local-first portfolio application with a deployable container configuration. Internet
deployment should add infrastructure-managed authentication, rate limiting, TLS, backups, and
observability according to the target environment. No Riot client access or scraping is present.

Implementation references: [Next.js installation](https://nextjs.org/docs/app/getting-started/installation),
[scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html), and
[XGBoost model serialization](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).
