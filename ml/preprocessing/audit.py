"""Independent fail-closed audit of the supplied cleaned dataset; never changes source rows."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    "games": "games_joined.csv",
    "players": "players_joined.csv",
    "rounds": "rounds_normalized.csv",
}
COMMON = ["GameID", "MatchID", "Date", "Patch", "Map", "Split"]
STATS = [
    "ACS",
    "Kills",
    "Deaths",
    "Assists",
    "ADR",
    "HS_Percent",
    "FirstKills",
    "FirstDeaths",
    "Num_2Ks",
    "Num_3Ks",
    "Num_4Ks",
    "Num_5Ks",
    "OnevOne",
    "OnevTwo",
    "OnevThree",
    "OnevFour",
    "OnevFive",
    "Plants",
    "Defuses",
]
REQUIRED = {
    "games": COMMON
    + [
        "Team1ID",
        "Team2ID",
        "Team1",
        "Team2",
        "EventName",
        "Team1_FirstSide",
        "Team2_FirstSide",
        "Team1_Rounds",
        "Team2_Rounds",
        "NumRounds",
        "GameWinnerSide",
    ],
    "players": COMMON
    + [
        "PlayerID",
        "PlayerName",
        "TeamSide",
        "TeamAbbreviation",
        "Agent",
        "TeamWon",
        "NumRounds",
        "KAST_Percent",
    ]
    + STATS,
    "rounds": COMMON
    + [
        "RoundNumber",
        "Team1_PreScore",
        "Team2_PreScore",
        "Team1_Side",
        "Team1_RoundWon",
        "RoundWinnerSide",
        "Team1_PostScore_DO_NOT_TRAIN",
        "Team2_PostScore_DO_NOT_TRAIN",
        "WinType_POST_ROUND",
    ],
}


def load_data(directory: Path):
    frames = {}
    for name, filename in FILES.items():
        frame = pd.read_csv(directory / filename)
        missing = set(REQUIRED[name]) - set(frame)
        if missing:
            raise ValueError(f"{filename}: missing columns {sorted(missing)}")
        frame["Date"] = pd.to_datetime(frame.Date, errors="raise")
        frames[name] = frame
    return frames


def audit(frames, source: Path | None = None):
    g, p, r = (frames[k] for k in FILES)
    errors, warnings = [], []

    def check(condition, message):
        if not bool(condition):
            errors.append(message)

    for name, frame in frames.items():
        check(
            frame[["GameID", "MatchID", "Date", "Split"]].notna().all().all(),
            f"{name}: missing relationship or split fields",
        )
        check(set(frame.Split) == {"train", "validation", "test"}, f"{name}: invalid splits")
        check(frame.groupby("MatchID").Split.nunique().max() == 1, f"{name}: series split leakage")
        for column in ["GameID", "MatchID"]:
            check(pd.api.types.is_integer_dtype(frame[column]), f"{name}: {column} not integer")
    check(not g.GameID.duplicated().any(), "duplicate GameID")
    check(not p.duplicated(["GameID", "PlayerID"]).any(), "duplicate player-game")
    check(not r.duplicated(["GameID", "RoundNumber"]).any(), "duplicate round")
    check(g.groupby("MatchID").Date.nunique().max() == 1, "series has inconsistent dates")
    for label, frame in [("players", p), ("rounds", r)]:
        check(set(frame.GameID) == set(g.GameID), f"{label}: GameID coverage differs")
        merged = frame.merge(g, on="GameID", suffixes=("", "_g"), validate="many_to_one")
        for col in ["MatchID", "Date", "Map", "Patch", "Split"]:
            check(
                merged[col].fillna("MISSING").eq(merged[f"{col}_g"].fillna("MISSING")).all(),
                f"{label}: inconsistent {col}",
            )
    check(p.TeamSide.isin([1, 2]).all(), "invalid TeamSide")
    counts = p.groupby(["GameID", "TeamSide"]).size().unstack(fill_value=0)
    check(counts.eq(5).all().all(), "not exactly five players per team")
    check(
        p.groupby(["GameID", "TeamSide"]).TeamAbbreviation.nunique().eq(1).all(),
        "inconsistent team abbreviations",
    )
    winner = g.set_index("GameID").GameWinnerSide
    check(
        p.TeamWon.eq(p.TeamSide.eq(p.GameID.map(winner)).astype(int)).all(),
        "player winner mapping inconsistent",
    )
    check(
        g.GameWinnerSide.eq(np.where(g.Team1_Rounds > g.Team2_Rounds, 1, 2)).all(),
        "game winner mapping inconsistent",
    )
    unreliable = g.Team1_Rounds.eq(g.Team2_Rounds) | g[["Team1_Rounds", "Team2_Rounds"]].max(
        axis=1
    ).lt(13)
    warnings.append(
        f"Unreliable final outcomes (draw/incomplete): {g.loc[unreliable, 'GameID'].tolist()}; preserve raw rows, exclude player outcome training"
    )
    check(g.NumRounds.eq(g.Team1_Rounds + g.Team2_Rounds).all(), "game length mismatch")
    check(
        p.NumRounds.eq(p.GameID.map(g.set_index("GameID").NumRounds)).all(),
        "player game length mismatch",
    )
    ordered = r.sort_values(["GameID", "RoundNumber"])
    check(
        ordered.RoundNumber.eq(ordered.groupby("GameID").cumcount() + 1).all(),
        "round ordering not consecutive",
    )
    check(ordered.Team1_RoundWon.isin([0, 1]).all(), "invalid round target")
    check(ordered.RoundWinnerSide.eq(2 - ordered.Team1_RoundWon).all(), "round target mismatch")
    previous1 = ordered.groupby("GameID").Team1_RoundWon.cumsum() - ordered.Team1_RoundWon
    previous2 = ordered.RoundNumber - 1 - previous1
    check(ordered.Team1_PreScore.eq(previous1).all(), "team1 pre-score leakage/inconsistency")
    check(ordered.Team2_PreScore.eq(previous2).all(), "team2 pre-score leakage/inconsistency")
    check(
        ordered.Team1_PostScore_DO_NOT_TRAIN.eq(previous1 + ordered.Team1_RoundWon).all(),
        "team1 post-score mismatch",
    )
    check(
        ordered.Team2_PostScore_DO_NOT_TRAIN.eq(previous2 + 1 - ordered.Team1_RoundWon).all(),
        "team2 post-score mismatch",
    )
    last = ordered.groupby("GameID").last()
    check(
        last.RoundNumber.eq(last.index.map(g.set_index("GameID").NumRounds)).all(),
        "round count differs from game",
    )
    for side in [1, 2]:
        check(
            last[f"Team{side}_PostScore_DO_NOT_TRAIN"]
            .eq(last.index.map(g.set_index("GameID")[f"Team{side}_Rounds"]))
            .all(),
            f"team{side} final round score mismatch",
        )
    first_side = r.GameID.map(g.set_index("GameID").Team1_FirstSide)
    expected = first_side.where(
        r.RoundNumber <= 12, first_side.map({"attack": "defend", "defend": "attack"})
    ).where(r.RoundNumber <= 24)
    check(
        r.Team1_Side.fillna("unknown").eq(expected.fillna("unknown")).all(),
        "regulation side reconstruction mismatch",
    )
    check(
        g.Team2_FirstSide.eq(g.Team1_FirstSide.map({"attack": "defend", "defend": "attack"})).all(),
        "initial sides not complementary",
    )
    check(r.loc[r.RoundNumber > 24, "Team1_Side"].isna().all(), "overtime side invented")
    intervals = {
        s: {
            "min": str(g[g.Split == s].Date.min()),
            "max": str(g[g.Split == s].Date.max()),
            "series": int(g[g.Split == s].MatchID.nunique()),
        }
        for s in ["train", "validation", "test"]
    }
    check(intervals["train"]["max"] <= intervals["validation"]["min"], "train chronology violated")
    check(intervals["validation"]["max"] <= intervals["test"]["min"], "test chronology violated")
    for col in STATS:
        check(pd.api.types.is_numeric_dtype(p[col]), f"player stat {col} nonnumeric")
        check((p[col].dropna() >= 0).all(), f"negative {col}")
    check(p.HS_Percent.dropna().between(0, 1).all(), "HS outside fractional scale")
    warnings += [
        f"Missing player patches: {int(p.Patch.isna().sum())}",
        f"Missing KAST: {int(p.KAST_Percent.isna().sum())}; excluded",
        f"Missing HS: {int(p.HS_Percent.isna().sum())}; train-median imputation",
        f"Unknown round sides: {int(r.Team1_Side.isna().sum())}; retained as unknown",
        "Original winner-abbreviation mapping cannot be independently rederived without raw files",
        "Team IDs/names may change historically; observations preserve per-game names",
        "Raw cleaning script requires absent inputs; clean-file validation is reproducible",
    ]
    rate = float(r.Team1_RoundWon.mean())
    warnings.append(f"Source team1 round win rate {rate:.4f}; symmetrize training and inference")
    profiles = {}
    for name, frame in frames.items():
        profiles[name] = {
            "rows": len(frame),
            "dtypes": frame.dtypes.astype(str).to_dict(),
            "missing": frame.isna().sum().astype(int).to_dict(),
            "max_text_lengths": {
                c: int(frame[c].dropna().astype(str).str.len().max())
                for c in frame.select_dtypes("object")
                if frame[c].notna().any()
            },
            "numeric": json.loads(frame.describe().to_json()),
            "categories": {
                c: {str(k): int(v) for k, v in frame[c].value_counts(dropna=False).items()}
                for c in ["Map", "Patch", "Split"] + (["Agent"] if name == "players" else [])
            },
        }
    report = {
        "schema_version": "1",
        "errors": errors,
        "warnings": warnings,
        "counts": {k: len(v) for k, v in frames.items()},
        "series": int(g.MatchID.nunique()),
        "unique_players": int(p.PlayerID.nunique()),
        "split_intervals": intervals,
        "team1_round_win_rate": rate,
        "profiles": profiles,
        "unreliable_outcome_game_ids": g.loc[unreliable, "GameID"].astype(int).tolist(),
    }
    if source:
        report["source_hashes"] = {
            f: hashlib.sha256((source / f).read_bytes()).hexdigest() for f in FILES.values()
        }
        report["source_quality_report"] = json.loads((source / "quality_report.json").read_text())
        expected_counts = report["source_quality_report"]["output_counts"]
        check(
            expected_counts
            == {
                "games": len(g),
                "player_game_rows": len(p),
                "rounds": len(r),
                "distinct_matches": g.MatchID.nunique(),
            },
            "source report counts differ",
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    parser.add_argument("--output", type=Path, default=Path("docs/data-audit.json"))
    args = parser.parse_args()
    report = audit(load_data(args.data_dir), args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != "profiles"}, indent=2))
    if report["errors"]:
        raise SystemExit("Dataset validation failed; inspect report before continuing")


if __name__ == "__main__":
    main()
