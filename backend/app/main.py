import json
from contextlib import asynccontextmanager
from datetime import date, datetime, time
from time import monotonic

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import and_, case, func, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.database import get_db
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
)
from backend.app.repositories.analytics import (
    earlier_performances,
    game_out,
    impact_rows,
    performance_out,
    player_filters,
    player_query,
)
from backend.app.schemas import (
    AgentOut,
    CoachOut,
    CoachRequest,
    Envelope,
    GameOut,
    OverviewOut,
    PeerRangesOut,
    PerformanceOut,
    PlayerDetailOut,
    PlayerOut,
    RoundOut,
    RoundRequest,
    TrendOut,
)
from backend.app.services.coach import (
    COACH_VERSION,
    evidence_coach,
    evidence_signature,
    optional_narration,
)
from backend.app.services.peer_ranges import PeerRanges
from ml.models.inference import load_bundle, probabilities


@asynccontextmanager
async def lifespan(app):
    app.state.bundle = None
    app.state.report = None
    app.state.audit = None
    app.state.version = None
    app.state.overview_cache = None
    app.state.peer_ranges = PeerRanges(settings.peer_interval_dir)
    current = settings.artifact_dir / "current.json"
    if current.exists():
        version = json.loads(current.read_text())["version"]
        if not version.startswith("v1-") or "/" in version or ".." in version:
            raise ValueError("Invalid artifact version")
        directory = settings.artifact_dir / version
        app.state.bundle, app.state.manifest = load_bundle(directory)
        app.state.version = version
        app.state.report = json.loads((directory / "metrics.json").read_text())
        app.state.audit = json.loads((directory / "audit.json").read_text())
    yield


app = FastAPI(title="ImpactAI Historical Esports Analytics", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(HTTPException)
async def http_error(_, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.detail}})


@app.exception_handler(RequestValidationError)
async def validation_error(_, exc):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Invalid request",
                "fields": [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()],
            }
        },
    )


@app.exception_handler(SQLAlchemyError)
async def database_error(_, exc):
    return JSONResponse(
        status_code=503,
        content={"error": {"message": "Database unavailable; run migrations and import."}},
    )


def metadata(request, **kwargs):
    return {
        "model_version": request.app.state.version,
        "coverage": "Historical professional esports, 2020–2022",
        "score_kind": "heuristic contextual index",
        "player_prediction_kind": "retrospective association",
        **kwargs,
    }


def ready(request: Request, db: Session = Depends(get_db)):
    if request.app.state.bundle is None:
        raise HTTPException(503, "Models not trained. Run python -m ml.train and import data.")
    if (
        db.scalar(select(ModelVersion.id).where(ModelVersion.id == request.app.state.version))
        is None
    ):
        raise HTTPException(
            503, "Active model predictions are not imported. Run python -m scripts.import_data."
        )
    return request.app.state.version


DB = Depends(get_db)
Version = Depends(ready)


def require_player(db, player_id):
    player = db.get(Player, player_id)
    if player is None:
        raise HTTPException(404, "Player not found")
    return player


def require_game(db, game_id):
    game = db.get(Game, game_id)
    if game is None:
        raise HTTPException(404, "Game not found")
    return game


@app.get("/health", response_model=Envelope[dict])
def health(request: Request, db: Session = DB):
    db.execute(text("SELECT 1"))
    return {
        "data": {"status": "ok", "models_loaded": request.app.state.bundle is not None},
        "meta": metadata(request),
    }


@app.get("/ready", response_model=Envelope[dict])
def readiness(request: Request, db: Session = DB, version: str = Version):
    """Deployment readiness requires complete observations and active-version predictions."""
    expected = request.app.state.audit["counts"]
    checks = [
        (Game, "games", expected["games"]),
        (GamePlayer, "player_records", expected["players"]),
        (Round, "rounds", expected["rounds"]),
        (ImpactPrediction, "impact_predictions", expected["players"]),
        (RoundPrediction, "round_predictions", expected["rounds"]),
    ]
    counts = {}
    for model, label, total in checks:
        query = select(func.count()).select_from(model)
        if model in (ImpactPrediction, RoundPrediction):
            query = query.where(model.version == version)
        counts[label] = db.scalar(query)
        if counts[label] != total:
            raise HTTPException(
                503, "Import incomplete for active model. Run python -m scripts.import_data."
            )
    return {"data": {"status": "ready", "counts": counts}, "meta": metadata(request)}


@app.get("/api/overview", response_model=Envelope[OverviewOut])
def overview(request: Request, db: Session = DB, version: str = Version):
    cache = request.app.state.overview_cache
    if cache is not None and monotonic() - cache[0] < 120:
        return cache[1]

    def count(model):
        return db.scalar(select(func.count()).select_from(model))

    agents = (
        db.execute(
            select(GamePlayer.agent.label("name"), func.count().label("count"))
            .group_by(GamePlayer.agent)
            .order_by(func.count().desc())
        )
        .mappings()
        .all()
    )
    dates = db.execute(
        select(Match.date, func.count(Game.id)).join(Game).group_by(Match.date)
    ).all()
    months = {}
    for d, n in dates:
        month = d.strftime("%Y-%m")
        months[month] = months.get(month, 0) + n
    bucket = case(
        *[(ImpactPrediction.score < end, end - 10) for end in range(10, 100, 10)], else_=90
    )
    histogram = dict(
        db.execute(
            select(bucket, func.count()).where(ImpactPrediction.version == version).group_by(bucket)
        ).all()
    )
    bins = [{"name": f"{a}–{a + 10}", "count": histogram.get(a, 0)} for a in range(0, 100, 10)]
    result = {
        "data": {
            "games": count(Game),
            "series": count(Match),
            "players": count(Player),
            "rounds": count(Round),
            "player_records": count(GamePlayer),
            "agents": [dict(x) for x in agents],
            "trends": [{"month": k, "games": v} for k, v in sorted(months.items())],
            "distribution": bins,
            "maps": db.scalars(select(Game.map).distinct().order_by(Game.map)).all(),
            "coverage_start": str(db.scalar(select(func.min(Match.date)))),
            "coverage_end": str(db.scalar(select(func.max(Match.date)))),
            "round_model": request.app.state.report["round"]["selected"],
            "player_model": request.app.state.report["player"]["selected"],
            "warnings": request.app.state.audit["warnings"],
        },
        "meta": metadata(request),
    }
    request.app.state.overview_cache = (monotonic(), result)
    return result


@app.get("/api/matches", response_model=Envelope[list[GameOut]])
def matches(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    map: str | None = None,
    agent: str | None = None,
    player: int | None = None,
    event: str | None = Query(None, max_length=150),
    date_from: date | None = None,
    date_to: date | None = None,
    db: Session = DB,
):
    query = select(Game, Match).join(Match, Match.id == Game.match_id)
    if map:
        query = query.where(Game.map == map)
    if event:
        query = query.where(Match.event.icontains(event, autoescape=True))
    if date_from:
        query = query.where(Match.date >= datetime.combine(date_from, time.min))
    if date_to:
        query = query.where(Match.date <= datetime.combine(date_to, time.max))
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, "date_from must be on or before date_to")
    if player or agent:
        sub = select(GamePlayer.game_id)
        if player:
            sub = sub.where(GamePlayer.player_id == player)
        if agent:
            sub = sub.where(GamePlayer.agent == agent)
        query = query.where(Game.id.in_(sub))
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = db.execute(
        query.order_by(Match.date.desc(), Game.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return {
        "data": [game_out(*r) for r in rows],
        "meta": metadata(request, total=total, page=page, page_size=page_size),
    }


@app.get("/api/matches/{game_id}", response_model=Envelope[GameOut])
def match_detail(game_id: int, request: Request, db: Session = DB):
    game = require_game(db, game_id)
    return {"data": game_out(game, db.get(Match, game.match_id)), "meta": metadata(request)}


@app.get("/api/matches/{game_id}/players", response_model=Envelope[list[PerformanceOut]])
def match_players(game_id: int, request: Request, db: Session = DB, version: str = Version):
    require_game(db, game_id)
    return {
        "data": [performance_out(r) for r in db.execute(impact_rows(db, version, game_id=game_id))],
        "meta": metadata(request),
    }


@app.get("/api/matches/{game_id}/rounds", response_model=Envelope[list[RoundOut]])
def match_rounds(game_id: int, request: Request, db: Session = DB, version: str = Version):
    game = require_game(db, game_id)
    rows = db.execute(
        select(Round, RoundPrediction.probability)
        .outerjoin(
            RoundPrediction,
            and_(
                RoundPrediction.game_id == Round.game_id,
                RoundPrediction.round_number == Round.number,
                RoundPrediction.version == version,
            ),
        )
        .where(Round.game_id == game_id)
        .order_by(Round.number)
    ).all()
    return {
        "data": [
            {
                "number": r.number,
                "team1_pre_score": r.team1_pre_score,
                "team2_pre_score": r.team2_pre_score,
                "team1_side": r.team1_side,
                "team1_won": r.team1_won,
                "win_type": r.win_type,
                "probability": p,
            }
            for r, p in rows
        ],
        "meta": metadata(
            request,
            split=db.get(Match, game.match_id).split,
            prediction_timing="pre-round; prior recorded outcomes allowed",
            unknown_overtime_side=True,
        ),
    }


@app.get("/api/players", response_model=Envelope[list[PlayerOut]])
def players(
    request: Request,
    q: str | None = Query(None, max_length=100),
    map: str | None = None,
    agent: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = DB,
    version: str = Version,
):
    query = player_filters(player_query(version), q, agent, map)
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    rows = (
        db.execute(
            query.order_by(func.count(GamePlayer.game_id).desc(), Player.id)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .mappings()
        .all()
    )
    return {
        "data": [dict(r) for r in rows],
        "meta": metadata(request, total=total, page=page, page_size=page_size),
    }


@app.get("/api/players/{player_id}", response_model=Envelope[PlayerDetailOut])
def player_detail(player_id: int, request: Request, db: Session = DB, version: str = Version):
    require_player(db, player_id)
    summary = db.execute(player_query(version).where(Player.id == player_id)).mappings().first()
    if summary is None:
        raise HTTPException(
            503,
            "Player predictions are not imported for the active model. Run python -m scripts.import_data.",
        )
    agents = (
        db.execute(
            select(GamePlayer.agent.label("name"), func.count().label("count"))
            .where(GamePlayer.player_id == player_id)
            .group_by(GamePlayer.agent)
        )
        .mappings()
        .all()
    )
    teams = db.scalars(
        select(GamePlayer.team_id).where(GamePlayer.player_id == player_id).distinct()
    ).all()
    return {
        "data": {
            **dict(summary),
            "agents": [dict(a) for a in agents],
            "historical_team_ids": teams,
        },
        "meta": metadata(request),
    }


@app.get("/api/players/{player_id}/matches", response_model=Envelope[list[PerformanceOut]])
def player_matches(
    player_id: int,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    map: str | None = None,
    agent: str | None = None,
    db: Session = DB,
    version: str = Version,
):
    require_player(db, player_id)
    query = impact_rows(db, version, player_id=player_id)
    if map:
        query = query.where(Game.map == map)
    if agent:
        query = query.where(GamePlayer.agent == agent)
    total = db.scalar(select(func.count()).select_from(query.order_by(None).subquery()))
    rows = db.execute(query.offset((page - 1) * page_size).limit(page_size))
    return {
        "data": [performance_out(r) for r in rows],
        "meta": metadata(request, total=total, page=page, page_size=page_size),
    }


@app.get("/api/players/{player_id}/impact", response_model=Envelope[PerformanceOut])
def player_impact(
    player_id: int,
    request: Request,
    game_id: int | None = None,
    db: Session = DB,
    version: str = Version,
):
    require_player(db, player_id)
    row = db.execute(
        impact_rows(db, version, player_id=player_id, game_id=game_id).limit(1)
    ).first()
    if not row:
        raise HTTPException(404, "Player performance not found")
    return {"data": performance_out(row), "meta": metadata(request)}


@app.get("/api/players/{player_id}/peer-ranges", response_model=Envelope[PeerRangesOut])
def player_peer_ranges(
    player_id: int, game_id: int, request: Request, db: Session = DB, version: str = Version
):
    row = db.execute(impact_rows(db, version, player_id=player_id, game_id=game_id)).first()
    if row is None:
        raise HTTPException(404, "Player performance not found")
    return {
        "data": request.app.state.peer_ranges.for_performance(performance_out(row)),
        "meta": metadata(request),
    }


@app.get("/api/players/{player_id}/trends", response_model=Envelope[list[TrendOut]])
def player_trends(player_id: int, request: Request, db: Session = DB, version: str = Version):
    require_player(db, player_id)
    rows = db.execute(impact_rows(db, version, player_id=player_id).limit(200)).all()[::-1]
    return {
        "data": [
            {
                "game_id": g.id,
                "date": m.date.isoformat(),
                "map": g.map,
                "agent": gp.agent,
                "score": i.score,
                "acs": gp.acs,
                "adr": gp.adr,
            }
            for gp, _, g, m, i in rows
        ],
        "meta": metadata(request, window="latest 200 games"),
    }


@app.get("/api/models", response_model=Envelope[dict])
def models(request: Request, version: str = Version):
    report = request.app.state.report
    return {
        "data": {
            "version": version,
            "round": report["round"]["selected"],
            "player": report["player"]["selected"],
            "feature_schema": request.app.state.bundle["schema"],
            "manifest": request.app.state.manifest,
        },
        "meta": metadata(request),
    }


@app.get("/api/models/metrics", response_model=Envelope[dict])
def model_metrics(request: Request, version: str = Version):
    return {"data": request.app.state.report, "meta": metadata(request)}


@app.get("/api/analysis/global-features", response_model=Envelope[dict])
def global_features(request: Request, version: str = Version):
    report = request.app.state.report
    return {
        "data": {
            "selected_model": report["global_features"],
            "method": report["global_explanation_method"],
            "xgboost_shap": report["xgboost_shap"],
            "sample_n": report["xgboost_shap_sample_n"],
        },
        "meta": metadata(request),
    }


@app.get("/api/agents", response_model=Envelope[list[AgentOut]])
def agents(request: Request, map: str | None = None, db: Session = DB, version: str = Version):
    query = (
        select(
            GamePlayer.agent.label("name"),
            func.count().label("selections"),
            func.avg(GamePlayer.adr).label("adr"),
            func.avg(ImpactPrediction.score).label("score"),
            func.avg(GamePlayer.first_kills / (Game.num_rounds * 1.0)).label("fkpr"),
            func.avg(GamePlayer.first_deaths / (Game.num_rounds * 1.0)).label("fdpr"),
            func.min(ImpactPrediction.score).label("min_score"),
            func.max(ImpactPrediction.score).label("max_score"),
            *[
                func.sum(
                    case(
                        (
                            and_(
                                ImpactPrediction.score >= a,
                                ImpactPrediction.score < a + 10
                                if a < 90
                                else ImpactPrediction.score <= 100,
                            ),
                            1,
                        ),
                        else_=0,
                    )
                ).label(f"bin_{a}")
                for a in range(0, 100, 10)
            ],
        )
        .join(Game, Game.id == GamePlayer.game_id)
        .join(
            ImpactPrediction,
            and_(
                ImpactPrediction.game_id == GamePlayer.game_id,
                ImpactPrediction.player_id == GamePlayer.player_id,
                ImpactPrediction.version == version,
            ),
        )
        .group_by(GamePlayer.agent)
    )
    if map:
        query = query.where(Game.map == map)
    rows = db.execute(query.order_by(func.count().desc())).mappings().all()
    result = []
    for row in rows:
        item = dict(row)
        item["distribution"] = [
            {"name": f"{a}–{a + 10}", "count": item.pop(f"bin_{a}")} for a in range(0, 100, 10)
        ]
        result.append(item)
    return {
        "data": result,
        "meta": metadata(
            request, interpretation="Selections and observed performance, not causal agent strength"
        ),
    }


@app.post("/api/analysis/round", response_model=Envelope[dict])
def predict_round(body: RoundRequest, request: Request, version: str = Version):
    b = body
    x = pd.DataFrame(
        [
            {
                "Map": b.map,
                "Patch": b.patch or "unknown",
                "RoundNumber": b.round_number,
                "Team1_PreScore": b.team1_pre_score,
                "Team2_PreScore": b.team2_pre_score,
                "ScoreDiff": b.team1_pre_score - b.team2_pre_score,
                "Team1_Side": b.team1_side,
                "Phase": "first_half"
                if b.round_number <= 12
                else "second_half"
                if b.round_number <= 24
                else "overtime",
                "PreviousWin": b.previous_win,
            }
        ]
    )
    bundle = request.app.state.bundle
    p = probabilities(bundle["round_model"], x, True, bundle["round_temperature"])[0]
    known_maps = request.app.state.audit["profiles"]["rounds"]["categories"]["Map"]
    return {
        "data": {
            "team1_probability": float(p),
            "team2_probability": float(1 - p),
            "context": b.model_dump(),
            "data_quality": {
                "side_known": b.team1_side != "unknown",
                "map_in_dataset": b.map in known_maps,
                "economy_used": False,
                "historical_only": True,
            },
        },
        "meta": metadata(request),
    }


@app.post("/api/analysis/coach", response_model=Envelope[CoachOut])
def coach(body: CoachRequest, request: Request, db: Session = DB, version: str = Version):
    row = db.execute(
        impact_rows(db, version, player_id=body.player_id, game_id=body.game_id).limit(1)
    ).first()
    if not row:
        raise HTTPException(404, "Player performance not found")
    cached = db.scalar(
        select(CoachingReport).where(
            CoachingReport.game_id == body.game_id,
            CoachingReport.player_id == body.player_id,
            CoachingReport.version == version,
        )
    )
    performance = performance_out(row)
    history = earlier_performances(db, version, row)
    signature = evidence_signature(performance, history)
    fresh = (
        cached is not None
        and cached.report.get("coach_version") == COACH_VERSION
        and cached.report.get("history_signature") == signature
    )
    report = cached.report if fresh else evidence_coach(performance, history)
    if not fresh:
        if cached:
            cached.report = report
        else:
            db.add(
                CoachingReport(
                    game_id=body.game_id, player_id=body.player_id, version=version, report=report
                )
            )
        try:
            db.commit()
        except IntegrityError:
            db.rollback()  # Concurrent identical requests are safe; evidence is deterministic.
    if body.use_llm:
        # Release the read transaction before waiting on the provider. Cached
        # reports otherwise hold SQLite's sole pooled connection for the timeout.
        db.close()
        report = optional_narration(report)
    return {"data": report, "meta": metadata(request)}
