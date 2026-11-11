"""Explicit feature allowlists: data unavailable at prediction time cannot enter implicitly."""

import numpy as np
import pandas as pd

ROUND_CAT = ["Map", "Patch", "Team1_Side", "Phase"]
ROUND_NUM = ["RoundNumber", "Team1_PreScore", "Team2_PreScore", "ScoreDiff", "PreviousWin"]
ROUND_HISTORY_DIFFS = [
    "RecentWinDiff3",
    "RecentWinDiff5",
    "PreviousStreak",
    "HalfScoreDiff",
    "HistoricalEloDiff",
    "HistoricalMapEloDiff",
    "HistoricalExperienceDiff",
]
PLAYER_CAT = ["Agent", "Map", "Patch", "LengthBand"]
PLAYER_NUM = [
    "KPR",
    "DPR",
    "APR",
    "ADR",
    "FKPR",
    "FDPR",
    "ClutchRate",
    "ObjectiveRate",
    "HS_Percent",
]
COMPONENTS = ["Combat", "Openings", "Damage", "Assists", "Objectives", "Clutches"]


def round_features(raw):
    r = raw.sort_values(["GameID", "RoundNumber"]).copy()
    x = r[["Map", "Patch", "RoundNumber", "Team1_PreScore", "Team2_PreScore", "Team1_Side"]].copy()
    x["ScoreDiff"] = x.Team1_PreScore - x.Team2_PreScore
    x["PreviousWin"] = r.groupby("GameID").Team1_RoundWon.shift(1).fillna(0.5)
    x["Phase"] = np.select(
        [x.RoundNumber <= 12, x.RoundNumber <= 24],
        ["first_half", "second_half"],
        default="overtime",
    )
    x[ROUND_CAT] = x[ROUND_CAT].fillna("unknown")
    return x.reindex(raw.index)[ROUND_CAT + ROUND_NUM]


def swap_round(x):
    swapped = x.copy()
    swapped["Team1_PreScore"] = x.Team2_PreScore
    swapped["Team2_PreScore"] = x.Team1_PreScore
    swapped["ScoreDiff"] = -x.ScoreDiff
    swapped["PreviousWin"] = 1 - x.PreviousWin
    for column in ROUND_HISTORY_DIFFS:
        if column in x:
            swapped[column] = -x[column]
    swapped["Team1_Side"] = x.Team1_Side.map(
        {"attack": "defend", "defend": "attack", "unknown": "unknown"}
    ).fillna("unknown")
    return swapped


def player_features(p):
    x = p.copy()
    for col, name in [
        ("Kills", "KPR"),
        ("Deaths", "DPR"),
        ("Assists", "APR"),
        ("FirstKills", "FKPR"),
        ("FirstDeaths", "FDPR"),
    ]:
        x[name] = x[col] / x.NumRounds
    x["NetFirstEngagements"] = x.FKPR - x.FDPR
    x["ClutchRate"] = (
        x[["OnevOne", "OnevTwo", "OnevThree", "OnevFour", "OnevFive"]].sum(axis=1) / x.NumRounds
    )
    x["ObjectiveRate"] = (x.Plants + x.Defuses) / x.NumRounds
    x["MultiKillRate"] = x[["Num_2Ks", "Num_3Ks", "Num_4Ks", "Num_5Ks"]].sum(axis=1) / x.NumRounds
    x["KD"] = x.Kills / x.Deaths.clip(lower=1)
    x["LengthBand"] = pd.cut(
        x.NumRounds, [0, 18, 24, np.inf], labels=["short", "regulation", "overtime"]
    ).astype(str)
    x[PLAYER_CAT] = x[PLAYER_CAT].fillna("unknown")
    return x


def component_values(x):
    return pd.DataFrame(
        {
            "Combat": x.KPR - x.DPR,
            "Openings": x.FKPR - x.FDPR,
            "Damage": x.ADR,
            "Assists": x.APR,
            "Objectives": x.ObjectiveRate,
            "Clutches": x.ClutchRate,
        },
        index=x.index,
    )


def scoreboard_controls(x):
    """Other-nine-player controls are diagnostics, never the displayed individual index."""
    out = x.copy()
    for c in ["KPR", "DPR", "ADR"]:
        team = x.groupby(["GameID", "TeamSide"])[c].transform("sum")
        game = x.groupby("GameID")[c].transform("sum")
        out[f"Teammates_{c}"] = (team - x[c]) / 4
        out[f"Opponents_{c}"] = (game - team) / 5
    return out
