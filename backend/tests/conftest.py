"""API integration tests use a temporary database seeded with one real imported match."""

import json
import os
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///./data/impactai.db")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from backend.app.core.database import create_database_engine, get_db
from backend.app.main import app
from backend.app.models import (
    Base,
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


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    source_engine = create_engine("sqlite:///./data/impactai.db")
    engine = create_database_engine(f"sqlite:///{tmp_path_factory.mktemp('api') / 'test.db'}")

    Base.metadata.create_all(engine)
    version = json.loads(Path("ml/artifacts/current.json").read_text())["version"]
    with Session(source_engine) as source, engine.begin() as target:
        game = source.get(Game, 60894)
        assert game is not None, "Run the documented local import before integration tests"
        player_ids = source.scalars(
            select(GamePlayer.player_id).where(GamePlayer.game_id == game.id)
        ).all()
        specs = [
            (Match, Match.id == game.match_id),
            (Team, Team.id.in_([game.team1_id, game.team2_id])),
            (Player, Player.id.in_(player_ids)),
            (Game, Game.id == game.id),
            (GamePlayer, GamePlayer.game_id == game.id),
            (Round, Round.game_id == game.id),
            (ModelVersion, ModelVersion.id == version),
            (
                ImpactPrediction,
                (ImpactPrediction.game_id == game.id) & (ImpactPrediction.version == version),
            ),
            (
                RoundPrediction,
                (RoundPrediction.game_id == game.id) & (RoundPrediction.version == version),
            ),
        ]
        for model, condition in specs:
            rows = source.execute(select(model.__table__).where(condition)).mappings().all()
            if rows:
                target.execute(model.__table__.insert(), [dict(r) for r in rows])

    def override():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    engine.dispose()
    source_engine.dispose()
