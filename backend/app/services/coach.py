"""Evidence is computed before narration. An optional LLM may order existing sentences only."""

import hashlib
import json
from statistics import median

import httpx

from backend.app.core.config import settings

SUGGESTIONS = {
    "Combat": "Review deaths and survival decisions in your own VOD; this scoreboard cannot identify their causes.",
    "Openings": "Review opening engagements and track first deaths relative to first kills in future games.",
    "Damage": "Review damage conversion in your own VOD and track damage per round over several games.",
    "Assists": "Track assist opportunities with teammates; the dataset cannot observe communication or utility timing.",
    "Objectives": "Review your role in plants and defuses; opportunity and assignment are not measured here.",
    "Clutches": "Review available clutch situations in your VOD; these counts do not measure clutch success rate.",
}


COACH_VERSION = "patterns-v2"
WEAKNESS_THRESHOLD = -0.5


def evidence_signature(performance, history):
    return hashlib.sha256(
        json.dumps([performance, history], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def evidence_coach(performance, history=()):
    """Descriptive heuristics over an explicitly bounded, earlier same-agent history."""
    impact = performance["impact"]
    components = impact["components"]
    patterns = {}
    for component, item in components.items():
        prior = [p["impact"]["components"][component]["z"] for p in history]
        below = sum(z <= WEAKNESS_THRESHOLD for z in prior)
        repeated = len(prior) >= 4 and below / len(prior) >= 0.6
        pattern = (
            ("repeated" if repeated else "limited_history" if len(prior) < 4 else "isolated")
            if item["z"] <= WEAKNESS_THRESHOLD
            else "within_reference"
        )
        patterns[component] = {
            "pattern": pattern,
            "prior_count": len(prior),
            "prior_below": below,
            "prior_median_z": median(prior) if prior else None,
        }
    ordered = sorted(components, key=lambda c: components[c]["z"])
    priorities = sorted(
        ordered, key=lambda c: (patterns[c]["pattern"] != "repeated", components[c]["z"])
    )
    strongest = ordered[-1]
    chosen = [strongest] + [c for c in priorities if c != strongest][:2]
    evidence = []
    for component in chosen:
        item = components[component]
        pattern = patterns[component]
        category = (
            "review_priority"
            if item["z"] <= WEAKNESS_THRESHOLD
            else "strength"
            if item["z"] >= 0.5
            else "context"
        )
        prior_statement = (
            (
                f" In {pattern['prior_below']} of {pattern['prior_count']} earlier same-agent games, "
                "this component was at least 0.5 standardized units below its peer mean."
            )
            if history
            else " No earlier same-agent games are available for comparison."
        )
        evidence.append(
            {
                "id": component.lower(),
                "component": component,
                "category": category,
                "value": item["value"],
                "peer_mean": item["peer_mean"],
                "z": item["z"],
                **pattern,
                "statement": f"{component}: {item['value']:.3f} versus a smoothed historical peer mean of {item['peer_mean']:.3f} ({item['z']:+.2f} standardized units)."
                + prior_statement,
                "suggestion": SUGGESTIONS[component]
                if category == "review_priority"
                else (
                    "Keep tracking this component across comparable games; this observation alone does not establish a coaching need."
                ),
                "source": {
                    "game_id": performance["game"]["id"],
                    "player_id": performance["player_id"],
                    "metric": component,
                    "model_version": impact["model_version"],
                },
            }
        )
    weaknesses = sum(c["z"] <= WEAKNESS_THRESHOLD for c in components.values())
    return {
        "player_id": performance["player_id"],
        "game_id": performance["game"]["id"],
        "player_name": performance["player_name"],
        "agent": performance["agent"],
        "mode": "deterministic_evidence",
        "model_version": impact["model_version"],
        "coach_version": COACH_VERSION,
        "history_signature": evidence_signature(performance, history),
        "history_count": len(history),
        "history_game_ids": [p["game"]["id"] for p in history],
        "assessment": (
            "No clear weakness in this performance under the heuristic threshold of −0.5 standardized units."
            if not weaknesses
            else f"{weaknesses} components fall below the heuristic review threshold of −0.5 standardized units."
        ),
        "score": impact["score"],
        "percentile": impact["percentile"],
        "peer_count": impact["peer_count"],
        "peer_level": impact["peer_level"],
        "evidence": evidence,
        "limitations": [
            "Historical professional esports (2020–2022), not modern ranked gameplay.",
            "History uses up to nine earlier same-agent games from earlier series; later and same-series games are excluded.",
            "Repeated means at least 60% of four or more prior games crossed −0.5 standardized units. These are descriptive heuristics, not validated coaching thresholds.",
            "No movement, position, aim placement, communication, utility timing, or event-level WPA is observable.",
            "The equal-weight contextual index is heuristic, not causal impact or Riot rank.",
        ],
    }


def optional_narration(report):
    if not all([settings.coach_llm_url, settings.coach_llm_api_key, settings.coach_llm_model]):
        return {**report, "llm_status": "unconfigured; deterministic evidence shown"}
    try:
        # Only known evidence IDs can affect the output. No provider prose/stats is accepted.
        response = httpx.post(
            settings.coach_llm_url,
            timeout=12,
            headers={"Authorization": f"Bearer {settings.coach_llm_api_key}"},
            json={
                "model": settings.coach_llm_model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": 'Order these evidence items for a clear coaching explanation. Return only JSON {"order":[IDs]}. Never generate statistics or advice.',
                    },
                    {"role": "user", "content": json.dumps(report["evidence"])},
                ],
            },
        )
        response.raise_for_status()
        order = json.loads(response.json()["choices"][0]["message"]["content"])["order"]
        lookup = {item["id"]: item for item in report["evidence"]}
        if not isinstance(order, list) or len(order) != len(lookup) or set(order) != set(lookup):
            raise ValueError("Ungrounded response")
        return {
            **report,
            "evidence": [lookup[key] for key in order],
            "mode": "llm_ordered_verified_evidence",
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
        return {
            **report,
            "llm_status": "provider response unavailable or invalid; deterministic evidence shown",
        }
