# Development guide

[Back to the project](../README.md) · [API reference](api.md) · [Research guide](research.md)

## Quick start with Docker Compose

Install Docker with a running daemon and Python 3 (only needed to extract the provided archive).
Supply your authorized `ImpactAI_Clean_Training_Data.zip`; data and trained artifacts are not
distributed in Git because source licensing was not supplied.

```bash
cp .env.example .env
python3 scripts/extract_data.py /path/to/ImpactAI_Clean_Training_Data.zip
docker compose build
docker compose up -d db
docker compose run --rm api python -m ml.preprocessing.audit
docker compose run --rm api python -m ml.train
docker compose run --rm api alembic upgrade head
docker compose run --rm api python -m scripts.import_data
docker compose up -d api frontend
```

Open **http://localhost:3000**. API docs: **http://localhost:8000/docs**.
Compose starts the frontend only after `/ready` confirms that observations and active-model
predictions are completely imported. `/health` checks database connectivity and reports whether
models are loaded; it does not imply an import is complete. Inspect startup with
`docker compose ps` and `docker compose logs api`.
`docker compose down` stops services without deleting the database volume. Do not use `down -v`
unless you intend to delete the local database. Default credentials are development-only; replace
them through environment variables for deployment. URL-encode special characters in connection URLs.

Training uses the full data, not a synthetic demo. It can take several minutes and several GB of
RAM including dependency/build caches. Leave a few GB of disk space for data, predictions, database,
feature datasets, and build artifacts. Models and calibration are trained before any import/API use.

## Native development

Requirements: Python 3.12, Node 22 (Node ≥20.9 supported by Next.js), npm, and PostgreSQL 17.
On macOS, install OpenMP for XGBoost: `brew install libomp`. Linux needs `libgomp1`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
cp .env.example .env
python scripts/extract_data.py /path/to/ImpactAI_Clean_Training_Data.zip
python -m ml.preprocessing.audit
python -m ml.train
```

Start PostgreSQL (for example `docker compose up -d db`) and set `DATABASE_URL` in `.env`.
Then:

```bash
alembic upgrade head
python -m scripts.import_data
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

For a database-free-service smoke test, set `DATABASE_URL=sqlite:///./data/impactai.db` in `.env`
before migration/import/startup. SQLite is a local convenience; PostgreSQL is the deployment
database. Avoid running a bulk SQLite import while serving traffic because its writer lock can
delay coaching writes. The local SQLite connection pool serializes database sessions; use one API
process with SQLite and PostgreSQL for concurrent deployment. PostgreSQL migrations and full
imports were independently verified.

`DATA_DIR` and `ARTIFACT_DIR` configure the importer/API. Training accepts `--data-dir` and
`--output` (defaults: `data/source`, `ml/artifacts`). Always run Python commands from the repo root.
Next.js proxies `/api/*` to `BACKEND_URL` (default `http://127.0.0.1:8000`). To change that target,
set it in the frontend process environment **before building**; Docker sets it through a build arg.
The repository-root `.env` is read by Python and Compose, not automatically by Next.js.

## Verification

Start with checks that work on a clean checkout without datasets or trained models:

```bash
pytest -q -m 'not dataset'
ruff check backend ml scripts conftest.py
ruff format --check backend ml scripts conftest.py
```

These use small artificial fixtures solely to test software invariants: migration upgrade/downgrade,
foreign keys, probability bounds, readiness, feature history, context scoring, serialization,
checksum validation, and provider fallback. They never populate the dashboard or reported metrics.
The GitHub Actions workflow runs these against SQLite and an isolated PostgreSQL database, plus
frontend lint/type/build checks and container builds. The workflow does not require private data
or run paid providers. To exercise the PostgreSQL test path locally, use an **empty database named
`impactai_test`** and set `TEST_DATABASE_URL` to its connection URL.

Dataset-marked ML tests need the supplied datasets and a completed training run. Real-record API
tests seed a temporary database from one match in the local SQLite import; they do not modify
your application PostgreSQL data. `pytest` skips these tests when prerequisites are missing;
`--require-dataset` fails instead, so a full integration run cannot silently omit them.
If you used PostgreSQL above, create the isolated SQLite fixture database once:

```bash
DATABASE_URL=sqlite:///./data/impactai.db alembic upgrade head
DATABASE_URL=sqlite:///./data/impactai.db python -m scripts.import_data
pytest -q --require-dataset
cd frontend
npm run typecheck
npm run lint
npm run build
```

The suite covers corrupted score histories, grouped chronology, current-round leakage, symmetry,
unknown-context fallbacks, deterministic fitting, schema mismatch, checksum rejection, SHAP
additivity, all major APIs, invalid/missing data, filtering, and coaching evidence/cache consistency.

For deterministic browser tests without private data or trained models:

```bash
cd frontend
npx playwright install chromium
npm run test:e2e
# Alternatively use an installed Chrome browser:
PLAYWRIGHT_CHANNEL=chrome npm run test:e2e
```

The test runner starts a test-only fixture API on port 18001 and Next.js on port 3100.
Artificial fixtures are confined to `frontend/tests`; they never populate application storage or
reported model metrics. CI runs these desktop/mobile workflows and retains failure traces/screenshots.
For integration against a running application with the original imported dataset, run
`E2E_DATASET=1 npm run test:e2e` (default port 3000; override with `TEST_BASE_URL`).

Tests cover player search → match selection → round analysis → coach evidence, pending-report
cancellation, URL navigation/Back/Forward, filter persistence, debouncing and caching, retries,
experimental ranges, overview counts, agent filtering, and viewport overflow. Test reports
are in `frontend/playwright-report`; failures include traces. [Verification record](verification.md)
distinguishes executed checks from configuration-only coverage.

## Production build

```bash
cd frontend
npm ci
npm run build
npm run start
```

The native start script serves the standalone Next.js output on **127.0.0.1:3000** and copies
static assets into that output. Container startup binds within the container; Compose exposes
ports only on the host loopback interface. API is a single FastAPI service; no queues or external
microservices are necessary. Add infrastructure-specific auth, TLS, backups, and rate limiting
before internet deployment. Restart the API after publishing a newly trained/imported model version.


## Source-data notes

The archive's original preprocessing script is retained in `data/source` for provenance. Its
missing raw inputs are not required for auditing or training from the supplied clean files.
The application does not scrape third-party sites or connect to a live Riot account.
