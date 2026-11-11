# API reference

[Back to the project](../README.md) · [Development guide](development.md)

## API surface

All successful responses use `{data, meta}` with model/coverage metadata; errors use
`{error: {message, ...}}`. OpenAPI is generated at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Database reachability and whether models are loaded |
| GET | `/ready` | Complete observation and active-model prediction coverage; 503 until ready |
| GET | `/api/overview` | Counts, distributions, agents, monthly coverage |
| GET | `/api/matches` | Paginated games; map/agent/player/event/date filters |
| GET | `/api/matches/{game_id}` | Map-game context, final score, reliability |
| GET | `/api/matches/{game_id}/players` | Ten-player scoreboard, index, explanations |
| GET | `/api/matches/{game_id}/rounds` | Observations and versioned pre-round probabilities |
| GET | `/api/players` | Name search; agent/map filters; pagination |
| GET | `/api/players/{player_id}` | Historical player summary |
| GET | `/api/players/{player_id}/matches` | Paginated performances; map/agent filters |
| GET | `/api/players/{player_id}/impact` | Latest or `?game_id=` contextual breakdown |
| GET | `/api/players/{player_id}/peer-ranges` | Optional saved experimental comparison; requires `?game_id=` |
| GET | `/api/players/{player_id}/trends` | Latest 200 chronological performances |
| GET | `/api/agents` | Agent metrics and distributions, optional map filter |
| GET | `/api/models` | Active model manifest and schema |
| GET | `/api/models/metrics` | Baselines, calibration, test slices, limitations |
| GET | `/api/analysis/global-features` | Selected-model and research TreeSHAP importance |
| POST | `/api/analysis/round` | Validated pre-round inference, team-symmetric probabilities |
| POST | `/api/analysis/coach` | `{player_id, game_id, use_llm?: false}` evidence report |

`/api/matches` represents map-games; `match_id` identifies their shared esports series.
Coach source references point to player IDs, game IDs, observed component metrics, and model versions.
The coach compares up to nine same-agent games from strictly earlier series. A current component
at or below −0.5 standardized units is a review candidate; a repeated pattern requires at least
four prior games and at least 60% crossing that threshold. Other cases are labeled isolated or
limited-history, and a performance can have no clear weakness. These thresholds are descriptive
heuristics awaiting human validation. Cached reports refresh when the evidence or coach revision
changes. See [the fresh-data and reviewer protocol](validation-plan.md) for remaining validation.
Optional `COACH_LLM_URL`, `COACH_LLM_API_KEY`, and `COACH_LLM_MODEL` remain server-side. The provider
can order verified evidence IDs only; unsupported or malformed output falls back deterministically.
This deliberately avoids ungrounded free-form statistics and observations.
