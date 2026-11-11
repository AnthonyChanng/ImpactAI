from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    event: Mapped[str] = mapped_column(String(500), index=True)
    patch: Mapped[str | None] = mapped_column(String(300), index=True)
    split: Mapped[str] = mapped_column(String(16), index=True)


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))


class Player(Base):
    __tablename__ = "players"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), index=True)


class Game(Base):
    __tablename__ = "games"
    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), index=True)
    map: Mapped[str] = mapped_column(String(40), index=True)
    team1_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    team2_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    team1_name: Mapped[str] = mapped_column(String(200))
    team2_name: Mapped[str] = mapped_column(String(200))
    team1_score: Mapped[int] = mapped_column(Integer)
    team2_score: Mapped[int] = mapped_column(Integer)
    num_rounds: Mapped[int] = mapped_column(Integer)
    winner_side: Mapped[int | None] = mapped_column(Integer)
    outcome_reliable: Mapped[bool]
    raw: Mapped[dict] = mapped_column(JSON)


class GamePlayer(Base):
    __tablename__ = "game_players"
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), primary_key=True, index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    team_side: Mapped[int] = mapped_column(Integer)
    agent: Mapped[str] = mapped_column(String(40), index=True)
    team_won: Mapped[bool | None]
    kills: Mapped[float] = mapped_column(Float)
    deaths: Mapped[float] = mapped_column(Float)
    assists: Mapped[float] = mapped_column(Float)
    acs: Mapped[float] = mapped_column(Float)
    adr: Mapped[float] = mapped_column(Float)
    first_kills: Mapped[float] = mapped_column(Float)
    first_deaths: Mapped[float] = mapped_column(Float)
    stats: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (
        Index(
            "ix_game_players_summary",
            "player_id",
            "game_id",
            "agent",
            "acs",
            "adr",
            "kills",
            "deaths",
            "first_kills",
            "first_deaths",
        ),
    )


class Round(Base):
    __tablename__ = "rounds"
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    number: Mapped[int] = mapped_column(Integer, primary_key=True)
    team1_pre_score: Mapped[int] = mapped_column(Integer)
    team2_pre_score: Mapped[int] = mapped_column(Integer)
    team1_side: Mapped[str | None] = mapped_column(String(16))
    team1_won: Mapped[bool]
    win_type: Mapped[str] = mapped_column(String(20))
    raw: Mapped[dict] = mapped_column(JSON)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    manifest: Mapped[dict] = mapped_column(JSON)


class ImpactPrediction(Base):
    __tablename__ = "impact_predictions"
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), primary_key=True, index=True)
    version: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, index=True)
    percentile: Mapped[float] = mapped_column(Float)
    association_probability: Mapped[float] = mapped_column(Float)
    detail: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (
        Index("ix_impact_version_player", "version", "player_id", "game_id", "score"),
        ForeignKeyConstraint(
            ["game_id", "player_id"],
            ["game_players.game_id", "game_players.player_id"],
            name="fk_impact_performance",
        ),
        CheckConstraint("score BETWEEN 0 AND 100", name="ck_impact_score"),
        CheckConstraint("percentile BETWEEN 0 AND 100", name="ck_impact_percentile"),
        CheckConstraint("association_probability BETWEEN 0 AND 1", name="ck_impact_probability"),
    )


class RoundPrediction(Base):
    __tablename__ = "round_predictions"
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), primary_key=True)
    round_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(ForeignKey("model_versions.id"), primary_key=True)
    probability: Mapped[float] = mapped_column(Float)
    __table_args__ = (
        ForeignKeyConstraint(
            ["game_id", "round_number"],
            ["rounds.game_id", "rounds.number"],
            name="fk_prediction_round",
        ),
        CheckConstraint("probability BETWEEN 0 AND 1", name="ck_round_probability"),
    )


class CoachingReport(Base):
    __tablename__ = "coaching_reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    version: Mapped[str] = mapped_column(ForeignKey("model_versions.id"))
    report: Mapped[dict] = mapped_column(JSON)
    __table_args__ = (
        UniqueConstraint("game_id", "player_id", "version"),
        ForeignKeyConstraint(
            ["game_id", "player_id"],
            ["game_players.game_id", "game_players.player_id"],
            name="fk_coach_performance",
        ),
    )
