"""Validated, transactional, idempotent imports. No source data is replaced or fabricated."""

import json

import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from backend.app.core.config import settings
from backend.app.core.database import engine
from backend.app.models import (
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
from ml.models.inference import load_bundle
from ml.preprocessing.audit import audit, load_data


def upsert(connection, model, records):
    if not records:
        return
    table = model.__table__
    insert = pg_insert if connection.dialect.name == "postgresql" else sqlite_insert
    for start in range(0, len(records), 500):
        statement = insert(table).values(records[start : start + 500])
        connection.execute(statement.on_conflict_do_nothing())


def clean_records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def refresh_query_statistics(connection):
    if connection.dialect.name == "sqlite":
        # Fresh imports need cardinalities so summary joins choose covering indexes.
        connection.execute(text("ANALYZE"))


def main():
    frames = load_data(settings.data_dir)
    report = audit(frames, settings.data_dir)
    if report["errors"]:
        raise ValueError(report["errors"])
    version = json.loads((settings.artifact_dir / "current.json").read_text())["version"]
    directory = settings.artifact_dir / version
    _, manifest = load_bundle(directory)
    if manifest["source_hashes"] != report["source_hashes"]:
        raise ValueError("Training artifacts do not match the source dataset")
    metrics = json.loads((directory / "metrics.json").read_text())
    g, p, r = (frames[k] for k in ["games", "players", "rounds"])
    unreliable = set(report["unreliable_outcome_game_ids"])
    games_by_id = g.set_index("GameID")
    with engine.begin() as conn:
        prior_manifests = conn.execute(select(ModelVersion.manifest)).scalars().all()
        if any(m["source_hashes"] != report["source_hashes"] for m in prior_manifests):
            raise ValueError(
                "Database contains another source revision. Use a new database or an explicit source migration; observations will not be silently overwritten."
            )
        existing = conn.execute(
            select(ModelVersion.manifest).where(ModelVersion.id == version)
        ).scalar()
        if existing:
            expected = [
                (Game, len(g)),
                (GamePlayer, len(p)),
                (Round, len(r)),
                (ImpactPrediction, len(p)),
                (RoundPrediction, len(r)),
            ]
            complete = True
            for model, size in expected:
                query = select(func.count()).select_from(model)
                if model in [ImpactPrediction, RoundPrediction]:
                    query = query.where(model.version == version)
                complete &= conn.scalar(query) == size
            if complete and existing["source_hashes"] == report["source_hashes"]:
                refresh_query_statistics(conn)
                print(f"Import already complete and source hashes verified: {version}")
                return
        upsert(
            conn,
            Match,
            [
                {
                    "id": int(row.MatchID),
                    "date": row.Date.to_pydatetime(),
                    "patch": None if pd.isna(row.Patch) else row.Patch,
                    "event": row.EventName,
                    "split": row.Split,
                }
                for row in g.drop_duplicates("MatchID").itertuples()
            ],
        )
        teams = {}
        for row in g.sort_values("Date").itertuples():
            teams[int(row.Team1ID)] = {"id": int(row.Team1ID), "name": row.Team1}
            teams[int(row.Team2ID)] = {"id": int(row.Team2ID), "name": row.Team2}
        upsert(conn, Team, list(teams.values()))
        upsert(
            conn,
            Player,
            [
                {"id": int(row.PlayerID), "name": row.PlayerName}
                for row in p.sort_values("Date")
                .drop_duplicates("PlayerID", keep="last")
                .itertuples()
            ],
        )
        upsert(
            conn,
            Game,
            [
                {
                    "id": row["GameID"],
                    "match_id": row["MatchID"],
                    "map": row["Map"],
                    "team1_id": row["Team1ID"],
                    "team2_id": row["Team2ID"],
                    "team1_name": row["Team1"],
                    "team2_name": row["Team2"],
                    "team1_score": row["Team1_Rounds"],
                    "team2_score": row["Team2_Rounds"],
                    "num_rounds": row["NumRounds"],
                    "winner_side": None if row["GameID"] in unreliable else row["GameWinnerSide"],
                    "outcome_reliable": row["GameID"] not in unreliable,
                    "raw": row,
                }
                for row in clean_records(g)
            ],
        )
        player_rows = []
        for row in clean_records(p):
            player_rows.append(
                {
                    "game_id": row["GameID"],
                    "player_id": row["PlayerID"],
                    "team_id": int(games_by_id.loc[row["GameID"], f"Team{row['TeamSide']}ID"]),
                    "team_side": row["TeamSide"],
                    "agent": row["Agent"],
                    "team_won": None if row["GameID"] in unreliable else bool(row["TeamWon"]),
                    **{
                        a: row[b]
                        for a, b in {
                            "kills": "Kills",
                            "deaths": "Deaths",
                            "assists": "Assists",
                            "acs": "ACS",
                            "adr": "ADR",
                            "first_kills": "FirstKills",
                            "first_deaths": "FirstDeaths",
                        }.items()
                    },
                    "stats": row,
                }
            )
        upsert(conn, GamePlayer, player_rows)
        print("Imported games and players", flush=True)
        upsert(
            conn,
            Round,
            [
                {
                    "game_id": row["GameID"],
                    "number": row["RoundNumber"],
                    "team1_pre_score": row["Team1_PreScore"],
                    "team2_pre_score": row["Team2_PreScore"],
                    "team1_side": row["Team1_Side"],
                    "team1_won": bool(row["Team1_RoundWon"]),
                    "win_type": row["WinType_POST_ROUND"],
                    "raw": row,
                }
                for row in clean_records(r)
            ],
        )
        upsert(conn, ModelVersion, [{"id": version, "metrics": metrics, "manifest": manifest}])
        predictions = []
        with (directory / "player-predictions.jsonl").open() as handle:
            for line in handle:
                item = json.loads(line)
                predictions.append(
                    {
                        "game_id": item["game_id"],
                        "player_id": item["player_id"],
                        "version": version,
                        "score": item["score"],
                        "percentile": item["percentile"],
                        "association_probability": item["association_probability"],
                        "detail": item,
                    }
                )
        upsert(conn, ImpactPrediction, predictions)
        upsert(
            conn,
            RoundPrediction,
            [
                {**row, "version": version}
                for row in clean_records(pd.read_parquet(directory / "round-predictions.parquet"))
            ],
        )
        refresh_query_statistics(conn)
    print(f"Import complete: {len(g)} games, {len(p)} player rows, {len(r)} rounds; {version}")


if __name__ == "__main__":
    main()
