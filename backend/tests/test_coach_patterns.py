"""Behavior tests for descriptive patterns, chronology, and report cache upgrades."""

from copy import deepcopy
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.main import app
from backend.app.models import (
    Base,
    CoachingReport,
    Game,
    GamePlayer,
    ImpactPrediction,
    Match,
    ModelVersion,
    Player,
    Team,
)
from backend.app.services.coach import COACH_VERSION, evidence_coach
from ml.features.build import COMPONENTS


def performance(game_id=10, z=-0.8):
    return {
        "player_id": 1,
        "player_name": "Example",
        "agent": "jett",
        "game": {"id": game_id},
        "impact": {
            "model_version": "v1-pattern-test",
            "score": 50,
            "percentile": 50,
            "peer_count": 100,
            "peer_level": "agent_map",
            "components": {
                c: {"z": z if c != "Combat" else 0.8, "value": 0.1, "peer_mean": 0.2}
                for c in COMPONENTS
            },
        },
    }


def test_no_weakness_does_not_invent_review_priorities():
    report = evidence_coach(performance(z=0.2))
    assert report["assessment"].startswith("No clear weakness")
    assert all(e["category"] != "review_priority" for e in report["evidence"])


@pytest.mark.parametrize(
    "prior_z,count,expected",
    [(-0.8, 4, "repeated"), (0.1, 5, "isolated"), (-0.8, 3, "limited_history")],
)
def test_patterns_require_sufficient_repeated_prior_evidence(prior_z, count, expected):
    history = [performance(i, prior_z) for i in range(count)]
    report = evidence_coach(performance(), history)
    priorities = [e for e in report["evidence"] if e["category"] == "review_priority"]
    assert priorities
    assert all(e["pattern"] == expected for e in priorities)
    assert report["history_game_ids"] == list(range(count))


@pytest.fixture
def pattern_client(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all(
            [
                Player(id=1, name="Example"),
                Team(id=1, name="One"),
                Team(id=2, name="Two"),
                ModelVersion(id="v1-pattern-test", metrics={}, manifest={}),
            ]
        )
        # Target game 10; older same-series map 11, later 12, other agent 13, tied time 14 excluded.
        for game_id in range(1, 15):
            match_id = 10 if game_id == 11 else game_id
            if game_id != 11:
                day = 10 if game_id == 14 else min(game_id, 12)
                db.add(
                    Match(
                        id=match_id, date=datetime(2021, 1, day), event="test", split="validation"
                    )
                )
            db.add(
                Game(
                    id=game_id,
                    match_id=match_id,
                    map="Ascent",
                    team1_id=1,
                    team2_id=2,
                    team1_name="One",
                    team2_name="Two",
                    team1_score=13,
                    team2_score=7,
                    num_rounds=20,
                    winner_side=1,
                    outcome_reliable=True,
                    raw={},
                )
            )
            detail = performance(game_id)["impact"]
            db.add(
                GamePlayer(
                    game_id=game_id,
                    player_id=1,
                    team_id=1,
                    team_side=1,
                    agent="sage" if game_id == 13 else "jett",
                    team_won=True,
                    kills=10,
                    deaths=10,
                    assists=2,
                    acs=100,
                    adr=100,
                    first_kills=1,
                    first_deaths=1,
                    stats={"PlayerName": "Example", "HS_Percent": None},
                )
            )
            db.add(
                ImpactPrediction(
                    game_id=game_id,
                    player_id=1,
                    version="v1-pattern-test",
                    score=50,
                    percentile=50,
                    association_probability=0.5,
                    detail=detail,
                )
            )
        db.add(
            CoachingReport(
                game_id=10, player_id=1, version="v1-pattern-test", report={"coach_version": "old"}
            )
        )
        db.commit()
    monkeypatch.setattr(settings, "artifact_dir", tmp_path)
    monkeypatch.setattr(settings, "peer_interval_dir", tmp_path / "missing")

    def override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override
    try:
        with TestClient(app) as client:
            app.state.bundle = {}
            app.state.version = "v1-pattern-test"
            yield client, engine
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_coach_history_excludes_future_other_agent_and_same_series_and_upgrades_cache(
    pattern_client,
):
    client, engine = pattern_client
    response = client.post("/api/analysis/coach", json={"player_id": 1, "game_id": 10})
    assert response.status_code == 200, response.text
    report = response.json()["data"]
    assert report["history_game_ids"] == list(range(9, 0, -1))
    assert report["coach_version"] == COACH_VERSION
    with Session(engine) as db:
        cached = db.scalar(select(CoachingReport))
        old_signature = cached.report["history_signature"]
        impact = db.get(ImpactPrediction, (9, 1, "v1-pattern-test"))
        detail = deepcopy(impact.detail)
        detail["components"]["Openings"]["z"] = 0.4
        impact.detail = detail
        db.commit()
    assert (
        client.post("/api/analysis/coach", json={"player_id": 1, "game_id": 10}).status_code == 200
    )
    with Session(engine) as db:
        assert db.scalar(select(CoachingReport)).report["history_signature"] != old_signature


def test_peer_range_endpoint_is_optional_and_checks_performance_exists(pattern_client):
    client, _ = pattern_client
    response = client.get("/api/players/1/peer-ranges?game_id=10")
    assert response.status_code == 200
    assert response.json()["data"]["available"] is False
    assert client.get("/api/players/1/peer-ranges?game_id=999").status_code == 404
