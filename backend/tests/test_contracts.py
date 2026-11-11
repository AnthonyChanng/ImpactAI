"""Tiny artificial database fixtures test constraints, never product analytics or ML quality."""

import os
from datetime import datetime

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, event, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.main import app
from backend.app.models import (
    CoachingReport,
    Game,
    GamePlayer,
    ImpactPrediction,
    Match,
    ModelVersion,
    Player,
    Round,
    RoundPrediction,
    Team,
)
from backend.app.schemas import RoundRequest

VERSION = "v1-contract-test"


@pytest.fixture(scope="module")
def contract_engine(tmp_path_factory):
    # CI optionally supplies a dedicated empty PostgreSQL database. Never target application data.
    url = (
        os.environ.get("TEST_DATABASE_URL")
        or f"sqlite:///{tmp_path_factory.mktemp('contracts') / 'test.db'}"
    )
    engine = create_engine(url)
    if engine.url.get_backend_name() != "sqlite" and engine.url.database != "impactai_test":
        raise ValueError("TEST_DATABASE_URL must use the dedicated impactai_test database")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(settings, "database_url", url)
        config = Config("alembic.ini")
        command.upgrade(config, "head")
        command.downgrade(config, "6d8c25a5c354")
        command.upgrade(config, "head")
        command.check(config)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def enforce_foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")

    with engine.begin() as db:
        assert db.execute(select(Game.id)).first() is None, "Use an empty test database"
        db.execute(
            Match.__table__.insert(),
            dict(id=1, date=datetime(2021, 1, 1), event="test", split="train"),
        )
        db.execute(Team.__table__.insert(), [dict(id=i, name=f"test-{i}") for i in (1, 2)])
        db.execute(Player.__table__.insert(), [dict(id=i, name=f"test-{i}") for i in (1, 2)])
        for i in (1, 2):
            db.execute(
                Game.__table__.insert(),
                dict(
                    id=i,
                    match_id=1,
                    map="Ascent",
                    team1_id=1,
                    team2_id=2,
                    team1_name="test-1",
                    team2_name="test-2",
                    team1_score=13,
                    team2_score=0,
                    num_rounds=13,
                    winner_side=1,
                    outcome_reliable=True,
                    raw={},
                ),
            )
            db.execute(
                GamePlayer.__table__.insert(),
                dict(
                    game_id=i,
                    player_id=i,
                    team_id=1,
                    team_side=1,
                    agent="jett",
                    team_won=True,
                    kills=1,
                    deaths=1,
                    assists=1,
                    acs=1,
                    adr=1,
                    first_kills=0,
                    first_deaths=0,
                    stats={},
                ),
            )
            db.execute(
                Round.__table__.insert(),
                dict(
                    game_id=i,
                    number=i,
                    team1_pre_score=i - 1,
                    team2_pre_score=0,
                    team1_side="attack",
                    team1_won=True,
                    win_type="elim",
                    raw={},
                ),
            )
        db.execute(ModelVersion.__table__.insert(), dict(id=VERSION, metrics={}, manifest={}))
        for i in (1, 2):
            db.execute(
                ImpactPrediction.__table__.insert(),
                dict(
                    game_id=i,
                    player_id=i,
                    version=VERSION,
                    score=50,
                    percentile=50,
                    association_probability=0.5,
                    detail={},
                ),
            )
            db.execute(
                RoundPrediction.__table__.insert(),
                dict(
                    game_id=i,
                    round_number=i,
                    version=VERSION,
                    probability=0.5,
                ),
            )
    yield engine
    engine.dispose()


@pytest.mark.parametrize(
    "model,values",
    [
        (
            ImpactPrediction,
            dict(
                game_id=1,
                player_id=2,
                version=VERSION,
                score=50,
                percentile=50,
                association_probability=0.5,
                detail={},
            ),
        ),
        (CoachingReport, dict(game_id=1, player_id=2, version=VERSION, report={})),
        (RoundPrediction, dict(game_id=1, round_number=2, version=VERSION, probability=0.5)),
    ],
)
def test_outputs_cannot_reference_unobserved_combinations(contract_engine, model, values):
    with pytest.raises(IntegrityError), contract_engine.begin() as db:
        db.execute(model.__table__.insert(), values)


@pytest.mark.parametrize(
    "model,field,value",
    [
        (ImpactPrediction, "score", -1),
        (ImpactPrediction, "score", 101),
        (ImpactPrediction, "percentile", 101),
        (ImpactPrediction, "association_probability", 1.1),
        (RoundPrediction, "probability", -0.1),
        (RoundPrediction, "probability", 1.1),
    ],
)
def test_invalid_derived_values_are_rejected(contract_engine, model, field, value):
    with pytest.raises(IntegrityError), contract_engine.begin() as db:
        db.execute(model.__table__.update().values({field: value}))


@pytest.fixture
def readiness_client(contract_engine, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "artifact_dir", tmp_path)
    with contract_engine.connect() as connection, connection.begin() as transaction:

        def override():
            with Session(connection) as session:
                yield session

        app.dependency_overrides[get_db] = override
        try:
            with TestClient(app) as client:
                yield client, connection
        finally:
            app.dependency_overrides.clear()
            if transaction.is_active:
                transaction.rollback()


def activate_contract_model():
    # A sentinel is sufficient for readiness; these tests never serve model predictions.
    app.state.bundle = {}
    app.state.version = VERSION
    app.state.audit = {"counts": {"games": 2, "players": 2, "rounds": 2}}


def test_health_and_readiness_distinguish_missing_models(readiness_client):
    client, _ = readiness_client
    assert client.get("/health").json()["data"] == {"status": "ok", "models_loaded": False}
    assert client.get("/ready").status_code == 503
    activate_contract_model()
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["data"]["counts"]["round_predictions"] == 2


@pytest.mark.parametrize("model", [ImpactPrediction, RoundPrediction])
def test_readiness_rejects_partial_predictions(readiness_client, model):
    client, connection = readiness_client
    activate_contract_model()
    connection.execute(delete(model).where(model.game_id == 1))
    assert client.get("/health").status_code == 200
    response = client.get("/ready")
    assert response.status_code == 503
    assert "Import incomplete" in response.json()["error"]["message"]


def test_readiness_rejects_wrong_active_version(readiness_client):
    client, _ = readiness_client
    activate_contract_model()
    app.state.version = "v1-unimported"
    assert client.get("/ready").status_code == 503


def test_existing_player_without_active_predictions_is_temporarily_unavailable(readiness_client):
    client, connection = readiness_client
    activate_contract_model()
    connection.execute(delete(ImpactPrediction).where(ImpactPrediction.player_id == 1))
    response = client.get("/api/players/1")
    assert response.status_code == 503
    assert "Player predictions are not imported" in response.json()["error"]["message"]
    assert client.get("/api/players/999").status_code == 404


@pytest.mark.parametrize(
    "scores,previous_win",
    [
        ((13, 1), 1),
        ((1, 13), 0),
        ((14, 12), 1),
        ((3, 0), 0),
        ((0, 3), 1),
        ((13, 12), 0),
        ((12, 13), 1),
        ((14, 13), 0),
    ],
)
def test_round_endpoint_rejects_impossible_history(readiness_client, scores, previous_win):
    client, _ = readiness_client
    activate_contract_model()
    response = client.post(
        "/api/analysis/round",
        json={
            "map": "Ascent",
            "round_number": sum(scores) + 1,
            "team1_pre_score": scores[0],
            "team2_pre_score": scores[1],
            "team1_side": "unknown",
            "previous_win": previous_win,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "Invalid request"


@pytest.mark.parametrize(
    "scores,previous_win",
    [
        ((0, 0), 0.5),
        ((12, 0), 1),
        ((12, 12), 0),
        ((13, 12), 1),
        ((12, 13), 0),
        ((13, 13), 1),
        ((14, 13), 1),
        ((49, 49), 0),
    ],
)
def test_round_history_allows_regulation_and_extended_overtime(scores, previous_win):
    request = RoundRequest(
        map="Ascent",
        round_number=sum(scores) + 1,
        team1_pre_score=scores[0],
        team2_pre_score=scores[1],
        team1_side="unknown",
        previous_win=previous_win,
    )
    assert request.round_number == sum(scores) + 1


def test_search_interprets_sql_wildcards_as_literal_text(readiness_client):
    client, connection = readiness_client
    activate_contract_model()
    connection.execute(Player.__table__.update().where(Player.id == 1).values(name="test_50%"))
    connection.execute(Player.__table__.update().where(Player.id == 2).values(name="testX500"))
    connection.execute(Match.__table__.update().values(event="Test_50% Cup"))
    for query in ["_", "%", "TEST_50%"]:
        response = client.get("/api/players", params={"q": query})
        assert response.status_code == 200
        assert [row["id"] for row in response.json()["data"]] == [1]
    assert client.get("/api/players", params={"q": "test"}).json()["meta"]["total"] == 2
    for event_query in ["_", "%", "test_50%"]:
        assert (
            client.get("/api/matches", params={"event": event_query}).json()["meta"]["total"] == 2
        )
    for event_query in ["TestX", "Test_500", "Test%Cup"]:
        assert client.get("/api/matches", params={"event": event_query}).json()["data"] == []
