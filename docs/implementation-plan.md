# Implementation record

1. Audit authoritative cleaned CSVs; preserve inputs and series splits; build import validation.
2. Train reproducible round and retrospective player models, train-only peer references, explanations.
3. Migrate/import relational storage, expose paginated typed APIs and materialized predictions.
4. Build a Next.js dashboard connected to those APIs.
5. Add evidence-bound deterministic coaching with optional constrained narrative assistance.
6. Run tests, builds, browser workflow when available, and document actual results.

Initial constraints: workspace was empty. Source-cleaning script depends on four unavailable raw
CSVs and hardcodes `/mnt/data`; it is preserved, not executed or rewritten. The cleaned inputs
are authoritative. Docker daemon was not running. PostgreSQL remains the default deployment;
SQLite is an explicitly documented local verification option. User requested a local repository
with Python/PostgreSQL/Next.js, so hosted Sites/Cloudflare deployment is outside this build.
Dataset distribution rights are unspecified; original data and generated models stay ignored.
MLflow is deferred: versioned JSON manifests and metrics provide a smaller reproducible MVP.

All six phases are implemented and locally verified. See `verification.md` for executed commands,
phase results, test counts, environment repairs, and the unexecuted Docker runtime check;
`methodology.md` for score definitions and leakage controls; `evaluation.md` for measured findings.
