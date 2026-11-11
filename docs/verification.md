# Implementation and verification record

Executed locally on October 15, 2026, using the supplied archive. This is a local repository build,
not a published website. Original CSVs, feature datasets, trained artifacts, caches, and databases
are ignored by Git. Git was initialized; no remote was created or pushed.

## Phase 1 — repository and data audit

Implemented `ml/preprocessing/audit.py`, safe archive extraction, dependency configuration,
directory structure, Alembic migrations, and source hash reporting. Preserved the original six
archive files. Commands: `python -m ml.preprocessing.audit`, `alembic upgrade head`.

Audit confirms all 13,228 games / 132,280 player rows / 273,163 rounds, complete teams,
consistent relationships, chronological MatchID splits, and side reconstruction. It discovers
the unreliable 14–14 and 5–6 game outcomes; these are preserved with explicit downstream handling.
The original cleaner cannot run without its four raw source files. This does not block the
clean-input workflow. Patch labels include descriptions up to 80 characters; a second migration
preserves them with a wider column.

## Phase 2 — reproducible ML

Implemented feature allowlists, symmetry, hierarchical context references, model comparison,
calibration, SHAP, sensitivity diagnostics, metrics, cluster intervals, materialized predictions,
native tree exports, and checksummed local bundles under `ml/`.

Commands: `python -m ml.train`, `pytest ml/tests -q`, `python -m scripts.research_report`.
The full run produced version `v1-1c8c8207cf73`; 8 initial ML tests passed. Final measured results
are in `evaluation.md` / `evaluation.json`, with actual calibration/importance PNGs in the ignored
artifact directory. The later first-kill-only diagnostic uses validation exclusively; it does not
change the frozen models or revisit model selection on test.

## Phase 3 — relational backend

Implemented SQLAlchemy observation/output tables, two migrations, idempotent transactional
import, typed FastAPI/Pydantic endpoints, request validation, inference reuse, materialized
predictions, SQL histogram aggregation, and short overview caching under `backend/` and
`scripts/import_data.py`.

Commands: migrations and full import on both SQLite and **PostgreSQL 17.11**; repeated imports;
`alembic check`; `pytest backend/tests -q`; live localhost HTTP checks.

Both databases imported all expected rows. A repeated full SQLite import completed without
duplicates. PostgreSQL repeat import reports matching source hashes and completed counts, then
returns without rewriting records. `alembic check` reports no schema drift. The initial 19 API
integration tests passed; these seed an isolated temporary database from real imported records.
Later tests also cover the optional provider's rejection of unknown evidence IDs.

The initial Homebrew installation failed while creating service/runtime links. The missing
PostgreSQL runtime links were repaired, then a project-local cluster was initialized under
`.cache/postgres`, bound to 127.0.0.1:55432. No system startup service was registered. The
Mac OpenMP dependency required by XGBoost was installed. These machine repairs are not needed
inside the provided Linux containers.

## Phases 4–5 — dashboard and evidence coach

Implemented Next.js App Router pages for overview, player search/profiles, match browsing/
analysis, agents, model transparency, and coaching; shared Recharts components, responsive dark
styles, empty/loading/error states, keyboard focus, and mobile navigation under `frontend/`.

The coach retrieves real materialized components before rendering templates. Optional external
LLM integration can only reorder existing evidence IDs. No paid provider was called; invalid
provider output and deterministic fallback are tested with a mocked provider.

Commands: `npm run typecheck`, `npm run lint`, `npm run build`, `npm run start`, and
`PLAYWRIGHT_CHANNEL=chrome npm run test:e2e`. Desktop Chrome and mobile Chromium emulation
passed the four workflows against the real PostgreSQL API, including player → match → round →
coach. Overview counts, model comparison, agent filtering, and viewport overflow checks passed.

The first browser run exposed contention while the SQLite repeat import held a write transaction.
It was corrected by completion of the import, faster completed-import detection, SQL aggregation,
short caching, and verification against PostgreSQL. Final initial PostgreSQL browser pass:
**4 passed in 10.3 seconds**. Desktop/mobile screenshots were visually inspected; a clipped large
chart tick was corrected with compact thousands formatting. No browser page errors were observed
in screenshot capture. Generated screenshots are under `.cache/`, not source assets.

The in-app browser execution tool was unavailable; the installed Chrome browser was used through
the project's Playwright dependency. The first preview start was rejected by automatic review
for binding to all interfaces; the native server now binds only to localhost and was approved.

## Phase 6 — initial checks and limits

| Final check | Result |
|---|---|
| `pytest -q` | **28 passed** in 11.15s; dependency deprecation warnings only |
| Ruff lint and format check | Passed; 37 Python files formatted |
| TypeScript and ESLint | Passed |
| Next.js production build | Passed; all eight page routes generated/compiled |
| Final desktop/mobile Playwright rerun | **4 passed** in 15.7s after final presentation/schema changes |
| PostgreSQL migration/import and repeated import | Passed; complete counts and hashes verified |
| SQLite and PostgreSQL `alembic check` | No schema drift |
| Docker Compose configuration | Valid; daemon/container runtime not exercised |
| Visual inspection | Desktop and mobile inspected; final desktop chart labels confirmed |

The final checks include Python tests, Ruff lint/format, TypeScript, ESLint, Next.js production
build, desktop/mobile workflows, native PostgreSQL schema/import checks, and
`docker compose config --quiet`. The root README documents clean-machine extraction, install,
training, migrations, import, development, production build, tests, API contracts, and limitations.

During the initial pass the **Docker daemon was stopped**, so only Compose configuration was
validated. Container execution was subsequently verified in the continuation below. The native
application, PostgreSQL database, production Next.js output, ML training, and browser workflows
were executed in the initial pass. No cloud
deployment, modern ranked validation, live Riot ingestion, or unrestricted paid LLM narration is
claimed. MLflow and the optional Random Forest comparison are deliberately deferred.

Model selection/evaluation uses historical data once per frozen training run. Formatting and
tests added after training can change a future code-derived version ID without changing the
statistical recipe; the reported metrics identify the actual saved run. Package versions and
source hashes are retained in its manifest. Do not use this test period as a fresh holdout for
future model tuning.

Temporary preview/API services and the project-local PostgreSQL process were stopped after
verification. Data, trained artifacts, and database files remain available; README startup commands
are the supported way to resume the application. The existing SQLite database is also ready for
the documented local smoke-test configuration.

## Continuation — integrity, readiness, and portable verification

Added composite foreign keys linking derived predictions and coaching reports to actual
`game_players` and `rounds` observations, plus bounds for scores, percentiles, and probabilities.
Migration `a83b91e274fc` preserves existing data. Applied it to the populated SQLite database;
`alembic check` reports no drift. Isolated tests exercise upgrade/downgrade/upgrade and reject
invalid combinations on both SQLite and PostgreSQL.

Added `/ready`, which requires the loaded model version, imported observations, and complete
active-version predictions matching the frozen audit counts. Compose now gates frontend startup
on that endpoint. During the real transactional import, `/health` returned 200 with loaded models
and `/ready` returned 503, correctly distinguishing process availability from usable data.

Artifact loading now requires checksums for every consumed model, metadata, and prediction file
before Joblib deserialization, and verifies directory/manifest/bundle version agreement. Tests
cover omitted checksums, changed files, unsafe paths, and version mismatch. This still assumes
trusted locally trained artifacts; hashes do not authenticate an attacker-controlled manifest.

The coach aborts an in-flight request when the selected match/player changes or the view unmounts,
and ignores stale results/errors. Added a desktop/mobile regression that holds a real coaching
response, changes the match, then verifies evidence belongs to the newly selected match.

Dataset-independent tests use small artificial fixtures only for software invariants. Product
analytics and published ML metrics remain entirely derived from the supplied historical data.
Dataset tests are explicitly marked; missing prerequisites skip by default and fail with
`--require-dataset`. The GitHub Actions workflow covers SQLite/PostgreSQL contracts, frontend
lint/build/type checks, and container builds. Its YAML was parsed locally; hosted GitHub Actions
has not run because the repository has not been pushed.

Executed checks after the changes:

| Check | Result |
|---|---|
| Full local `pytest -q --require-dataset` | **52 passed**, 15 dependency deprecation warnings |
| Source-only Linux container, SQLite | **26 passed, 26 explicitly skipped**; no data/models mounted |
| Source-only Linux container, PostgreSQL | **26 passed, 26 explicitly skipped**; isolated `impactai_test` database |
| Ruff lint and formatting | Passed |
| Frontend ESLint and TypeScript | Passed |
| API and frontend Docker image builds | Passed; Python 3.12 and Node 22 Linux images |
| Compose configuration | Passed |
| Container PostgreSQL migration and full import | Passed: 13,228 games / 132,280 player rows / 273,163 rounds |
| Live `/ready` after import | 200; all observation and active-version prediction counts match |
| Container repeated import | Source hashes/counts verified; skipped rewriting completed data |
| Container PostgreSQL `alembic check` | No schema drift |
| Desktop/mobile workflows against Docker | **6 passed in 17.4s**, including delayed coaching-response regression |

The deliberate frontend startup attempt during import was withheld by the unhealthy API dependency.
After the transaction committed, `/ready` became healthy and the frontend started successfully.
The documented setup order completes import before starting API/frontend, avoiding this startup wait.

Docker Compose services were stopped after verification without deleting volumes. The application
database remains in `impactai_postgres_data`; the artificial test database is isolated as
`impactai_test`. Restart the prepared local stack with `docker compose up -d`, then open
http://localhost:3000. No service was published to an external host.

Frozen model version `v1-1c8c8207cf73` and its published evaluation are unchanged. This continuation
does not retrain models or tune against the held-out test period.

## Subsequent requested training run

At the user's request, the full fixed-recipe pipeline was subsequently retrained from the supplied
CSVs in Linux. New candidate `v1-67a7c0cc8a82` completed successfully, with all prediction artifacts
verified and metrics reproduced within numerical precision. See the
[training run record](training-runs/2026-10-15-linux.md) for results, provenance, and saved paths.
This candidate is separate from the active application model and uses the same historical holdout.

## Requested model improvement experiment

Ran the [bounded development-only search](experiments/2026-10-15-plan.md): 42 configurations,
84 chronological screening fits, and 10 finalist/baseline refits. The original test split was
excluded. Both tasks improved on a fixed development confirmation segment; the gains are reported
against baselines scored on exactly the same rows, with matched calibration partitions.

The expanded Python suite passed **62 tests**, Ruff passed, and all research bundle/source hashes
were verified. Reloaded models reproduced all 10,342 round and 5,070 player confirmation predictions
within `1.2e-7`. Round symmetry and 512 player TreeSHAP explanations passed checks. Calibration
figures were rendered and inspected. Research candidates were saved separately; no API/frontend
activation or new held-out test performance is claimed. See the
[complete results and limitations](experiments/2026-10-15-results.md).

## Continued postgame training — October 16, 2026

Completed the [player-only search](experiments/2026-10-16-results.md) and a separately declared
[component-specific peer follow-up](experiments/2026-10-16-peer-followup-results.md). Together they
performed 145 boosted-model fits, including temporal early-stopping probes and individual component
regressors. Original test rows were excluded throughout.

The selected three-model average lowered log loss from 0.311092 to 0.306223 against the actual
frozen prior candidate, while accuracy at 0.5 fell from 85.07% to 84.52%. ROC-AUC rose from 0.937994
to 0.938737. Matched native-refit and frozen-artifact comparisons are both retained. Outcome-free
peer expected-value error decreased by 0.175% in the exploratory component follow-up. These are
small development gains, not an independently validated individual impact model.

`pytest --require-dataset -q` passed **75 tests**, and `pytest scripts/tests -q` passed **4 additional
follow-up tests**. New checks cover feature allowlists, unseen/missing context, chronological early
stopping, serialization, ensemble arithmetic, and component selection without diagnostic labels.
Ruff passed across 52 Python files. All 5,070 saved confirmation observations reproduced exactly;
TreeSHAP reconstruction passed on 512 rows for each of three selected members, with maximum
ensemble probability error below `6.56e-7`. Peer refits reproduced their selection metrics and
expected values were invariant to changed observed stats/outcome labels at inference.

A provenance check initially found an archived `.py` snapshot inside the ML artifact directory,
which the broad source fingerprint included. No recorded model source had changed. The immutable
peer run was relocated to `data/research/peer-components-20261016`, its existing hashes verified,
and future script snapshots use `.txt`. The complete source/hash/prediction check then passed.
Association artifacts are in `ml/artifacts/research/player-20261016-v2`. The candidates remain
separate from the application's active model. Both training processes completed successfully.

## Alternative learner and calibration continuation — October 16, 2026

Completed the [fixed alternative-learner search](experiments/2026-10-16-final-search-results.md)
and wrote a [project overview](project-overview.md). Six histogram-gradient-boosting recipes were
screened over three chronological windows. Early-stopping probes and finalist refits totaled
40 additional boosted-model fits. Twenty-eight model/mixture/calibration combinations were
compared on a selection segment that followed calibration fitting chronologically.

The previous XGBoost ensemble remained best in selection. A sigmoid calibration variant reduced
development log loss from 0.306223 to 0.306112; AUC stayed 0.938737 and accuracy changed from
84.52% to 84.50%. These small, mixed changes are explicitly reported. No original test rows were
fitted or predicted, and no fresh independent evaluation claim is made.

`pytest --require-dataset -q`: **93 passed**, including 14 new tests for calibration families,
native categorical unknown/missing handling, time separation, and mixture weights. Ruff lint and
format passed for **56 Python files**. Model/source/script hashes passed. Reloaded predictions
matched exactly on all 5,070 confirmation rows; changing IDs/outcomes left inference unchanged.
TreeSHAP checks covered 512 rows per member, with maximum reconstructed calibrated ensemble
probability error `6.56e-7`. The actual frozen previous predictions were matched by observation
keys. Training exited successfully, and the selected research bundle is saved at
`ml/artifacts/research/player-calibration-20261016`. The active application model remains v1.

## Peer ranges and application reliability — October 17, 2026

Completed the [contextual peer-range experiment](experiments/2026-10-17-peer-interval-results.md):
72 quantile-regressor fits, with train-only context encoding, three chronological screening folds,
earlier selection, separate interval calibration, and a reused development diagnostic period.
All original test rows were excluded. Calibrated normalized interval score decreased from
3.283993 to 3.160698 (**3.75%**); pinball loss decreased **2.05%**. Five components improved interval
score while objectives worsened. Combat/damage coverage remained below the nominal 80%; sparse
objective/clutch components exceeded 90%. Neither guaranteed coverage nor causal impact is claimed.

The independent verifier checked seven artifact hashes plus source/script/ML fingerprints,
reloaded all 5,070 diagnostic predictions exactly, rebuilt the empirical reference and calibration,
recomputed selection and interval metrics, and verified unchanged inference after removing every
noncontext input. Series-bootstrap coverage intervals and component/slice metrics are saved with
the report. The figure was rendered and visually inspected. Artifacts remain research-only in
`data/research/peer-intervals-20261017`; active model `v1-1c8c8207cf73` is unchanged.

End-to-end testing found and corrected application issues:

- Existing players without active-version predictions now receive an actionable 503 instead of
  an unhandled 500; nonexistent players still return 404.
- The round probability chart preserves missing estimates as gaps. It no longer substitutes 50%.
  Desktop/mobile regression tests exercise the missing-value case.
- SQLite sessions are serialized for the single-process local backend, preventing concurrent
  read/cache-write requests from locking each other. PostgreSQL retains a concurrent pool.
- Explicit prediction-key filters avoid unnecessarily scanning all predictions for a match or
  player. A full-data match query measured **26.20 seconds before / 0.076 seconds after**.
- Migration `b91537ac901e` adds covering summary indexes. SQLite migration/import also refreshes
  query statistics so the planner uses them. A full-data player-summary query measured
  **21.33 seconds before / 0.266 seconds after**. These are local observations, not latency SLAs.
- ESLint now excludes generated Playwright reports, which previously caused lint to inspect
  bundled trace-viewer code after a browser-test failure.

Initial browser runs exposed the database contention and summary-query bottlenecks; the final
run passed after both fixes. No test timeout was relaxed and no browser workflow was skipped.

| Check | Final result |
|---|---|
| `pytest --require-dataset -q` | **115 passed**, 15 dependency deprecation warnings |
| PostgreSQL contract suite | **14 passed** in isolated `impactai_test`; upgrade/downgrade/upgrade and no schema drift |
| Ruff lint and format | Passed across **61 Python files** |
| Frontend ESLint, TypeScript, production build | Passed |
| Desktop/mobile production-browser workflows | **8 passed in 18.6 seconds**, two concurrent workers |
| Existing full SQLite migration and `alembic check` | Passed; no schema drift |
| Repeated full-data import | Source hashes/counts matched; completed observations/predictions reused |
| Final live `/ready` | 200: 13,228 games, 132,280 player records/predictions, 273,163 rounds/predictions |

PostgreSQL verification used the existing native local cluster on port 55432 and an isolated test
database. The application's existing PostgreSQL observations were not modified. Apply the new
migration when updating an existing PostgreSQL installation, and rebuild container images to use
the latest code. Native API/frontend/PostgreSQL test services were stopped after checks; data and
artifacts were preserved. No public deployment or paid service was used.

## Navigation, comparisons, and coaching — November 10, 2026

Implemented URL-driven player/agent/match filters and selected player-game navigation; browser
Back/Forward and reload preserve the selected context. Searches debounce for 250 ms, retain
previous results while refreshing, and share a bounded two-minute response cache. Failed/non-JSON
responses provide a retry; obsolete requests cannot replace a newer view.

Added an observed-value/peer-mean table with optional saved research median, range, and measured
coverage. Research JSON/Parquet checksums, observation keys, context, and component values must
match before a range is displayed. This view loads no executable model bundles and makes no new
research predictions. The existing active model and index are unchanged. A direct comparison of
source game 53291 / player 11225 matched all six saved research components.

The coach now compares up to nine earlier same-agent games, excluding the selected series and
later/tied timestamps. It distinguishes repeated, isolated, and limited-history patterns and can
report no clear weakness. Thresholds are labeled heuristic. Cached reports refresh when their
coach revision or evidence signature changes. New tests cover chronology, source matching,
cache refresh, missing research files, corrupt checksums, and malformed intervals.

The browser suite now starts its own artificial fixture API and frontend; CI installs Chromium,
runs desktop/mobile workflows, and saves failure traces/screenshots. These fixtures never populate
application storage or reported ML metrics. `E2E_DATASET=1` retains the real-data integration mode.

Executed verification:

- Dataset-independent Python suite: **117 passed**, 27 dataset checks deliberately deselected.
- Python lint and formatting: passed.
- Frontend lint and TypeScript: passed.
- Browser workflows: all **26 checks passed** across desktop Chrome and mobile Chromium emulation.
  The full run passed 24; two search-clock assertions initially failed because the mock clock was
  installed after React initialized. Installing the clock before navigation fixed the test setup;
  both focused reruns passed. Application behavior was not relaxed to satisfy these assertions.
- Peer-comparison screenshots inspected at desktop/mobile sizes; tables scroll within their cards.
- Docker Compose configuration: passed.
- Production build (`next build --webpack`) and post-build TypeScript check: passed.

The local SQLite import was absent, so this update does not claim a fresh full-data API integration
run. The [validation protocol](validation-plan.md) is prepared only; no new holdout, independent
review, or coaching-effectiveness result is claimed. Earlier dated verification records above are
historical and describe the code and environment at those times.
