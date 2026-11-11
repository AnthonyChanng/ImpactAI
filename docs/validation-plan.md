# Fresh-data and human-review protocol

Status: prepared November 10, 2026; not executed. Fresh authorized observations and independent
reviewers are not available in this checkout. Existing development results and software tests
must not be presented as a new external validation.

## Freeze before viewing new outcomes

Record the candidate and baseline artifact hashes, source/schema version, calibration parameters,
feature definitions, coach revision, and this protocol in a new immutable experiment directory.
Freeze primary metrics, acceptance criteria, analysis code, and reviewer rubric before evaluation.
Do not tune the −0.5 coaching threshold or the 60% repetition rule using the final comparison.

## Acquire and audit a chronological holdout

Use authorized matches occurring after every previously inspected development/test period.
Audit game/player/series identifiers, chronology, patch labels, score consistency, missingness, and
agent/map support. Keep every observation from a series together and remove overlap with all prior
experiments. Preserve source bytes and provenance. Choose sample size and stopping rules before
scoring; if the available sample is inadequate, report that limitation rather than repeatedly
checking results until they improve. Do not infer validity for ranked games from professional data.

## Evaluate once

- Pre-round and retrospective association: log loss and Brier as primary probability metrics;
  calibration, ROC-AUC, and performance by agent/map/patch as secondary diagnostics.
- Peer ranges: interval score and coverage with lower/upper tail misses, reported per component.
  Retain sparse and negative slices; do not average component coverage into “accuracy.”
- Compare frozen candidates and baselines on exactly the same keyed observations. Report paired
  series-bootstrap uncertainty and sample sizes. Predeclare acceptable degradation in each component.
- Keep test outputs immutable. Any tuning after inspection requires a separate later holdout.

## Blinded human review

Have at least two independent reviewers assess a preselected sample spanning agents, maps, support
levels, and positive/neutral/negative performances. Randomize report order and hide model identity.
Reviewers receive the same scoreboard, peer-reference definitions, and prior-game evidence.
Without VOD or event data, they must not rate claims about unobserved tactical causes as verified.

Record one row per report/reviewer with these fields:

| Field | Values |
|---|---|
| sample_id / reviewer_id / blinded_variant | Stable pseudonymous identifiers |
| factual_grounding | 1–5; are statements supported by the supplied evidence? |
| priority_relevance | 1–5; do priorities match the observed performance and history? |
| actionability | 1–5; is the suggested next review step specific and feasible? |
| appropriate_restraint | 1–5; are neutral/limited-history cases handled honestly? |
| unsupported_claim | Yes/no, with a quoted statement and reason |
| preferred_variant | A / B / tie |
| notes | Explanation and evidence references |

Define scoring anchors and acceptance criteria before review. Report agreement, disagreement,
paired preferences, and uncertainty; retain all reviews. These ratings evaluate report usefulness,
not improvement in actual gameplay. Testing coaching effectiveness requires a separate prospective
study with outcomes and an appropriate comparison group.
