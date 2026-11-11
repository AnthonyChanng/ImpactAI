import copy

import numpy as np
import pandas as pd
import pytest

from ml.features.build import ROUND_CAT, ROUND_NUM, swap_round
from ml.features.history import TEAM_FEATURES, TeamHistory, recent_round_features
from ml.improve import candidates, design, paired_interval, time_boundaries


@pytest.fixture
def games():
    return pd.DataFrame(
        {
            "GameID": [1, 2, 3, 4],
            "MatchID": [10, 10, 20, 30],
            "Date": pd.to_datetime(["2021-01-01", "2021-01-01", "2021-01-02", "2021-01-02"]),
            "Map": ["Ascent", "Split", "Ascent", "Ascent"],
            "Team1ID": [1] * 4,
            "Team2ID": [2] * 4,
            "Team1_Rounds": [13] * 4,
            "Team2_Rounds": [0] * 4,
        }
    )


def test_same_series_and_simultaneous_games_do_not_see_current_outcomes(games):
    original = TeamHistory().fit_transform(games)
    assert (original.loc[[1, 2], TEAM_FEATURES] == 0).all().all()
    assert original.loc[3, "HistoricalEloDiff"] == pytest.approx(0.06)
    pd.testing.assert_series_equal(original.loc[3], original.loc[4], check_names=False)
    changed = games.copy()
    changed.loc[2, ["Team1_Rounds", "Team2_Rounds"]] = [0, 13]
    pd.testing.assert_frame_equal(original, TeamHistory().fit_transform(changed))


def test_frozen_history_ignores_validation_labels_and_preserves_state(games):
    model = TeamHistory()
    model.fit_transform(games.iloc[:2])
    before = copy.deepcopy(model.__dict__)
    observed = model.transform(games.iloc[2:])
    changed = games.iloc[2:].copy()
    changed[["Team1_Rounds", "Team2_Rounds"]] = [0, 13]
    pd.testing.assert_frame_equal(observed, model.transform(changed))
    assert model.__dict__ == before
    with pytest.raises(ValueError, match="after the training period"):
        model.transform(games.iloc[:1])


def test_unreliable_outcomes_do_not_update_ratings_and_unknown_teams_are_neutral(games):
    games.loc[:1, ["Team1_Rounds", "Team2_Rounds"]] = [14, 14]
    model = TeamHistory()
    model.fit_transform(games.iloc[:2])
    assert model.global_ratings == {} and model.counts == {}
    future = games.iloc[2:].copy()
    future[["Team1ID", "Team2ID"]] = [1000, 2000]
    assert (model.transform(future) == 0).all().all()


def test_history_is_team_symmetric(games):
    a = TeamHistory().fit_transform(games)
    reverse = games.copy()
    reverse[["Team1ID", "Team2ID"]] = games[["Team2ID", "Team1ID"]].to_numpy()
    reverse[["Team1_Rounds", "Team2_Rounds"]] = games[["Team2_Rounds", "Team1_Rounds"]].to_numpy()
    b = TeamHistory().fit_transform(reverse)
    np.testing.assert_allclose(a.iloc[:, :3], -b.iloc[:, :3])
    np.testing.assert_array_equal(a.HistoricalMinGames, b.HistoricalMinGames)


@pytest.fixture
def rounds():
    outcomes = [1, 1, 0, 1, 0, 0] * 4
    score = np.cumsum([0] + outcomes[:-1])
    frame = pd.DataFrame(
        {
            "GameID": [1] * 24,
            "RoundNumber": np.arange(1, 25),
            "Map": ["Ascent"] * 24,
            "Patch": ["test"] * 24,
            "Team1_PreScore": score,
            "Team2_PreScore": np.arange(24) - score,
            "Team1_Side": ["attack"] * 12 + ["defend"] * 12,
            "Team1_RoundWon": outcomes,
        }
    )
    return frame.sample(frac=1, random_state=42)


def test_recent_features_exclude_current_and_future_outcomes(rounds):
    x = recent_round_features(rounds)
    assert list(x.columns[: len(ROUND_CAT + ROUND_NUM)]) == ROUND_CAT + ROUND_NUM
    changed = rounds.copy()
    changed.loc[changed.RoundNumber.ge(8), "Team1_RoundWon"] = 0
    mask = rounds.RoundNumber.le(8)
    pd.testing.assert_frame_equal(x.loc[mask], recent_round_features(changed).loc[mask])
    assert x.loc[rounds.RoundNumber.eq(1), "PreviousStreak"].item() == 0
    assert x.loc[rounds.RoundNumber.eq(3), "PreviousStreak"].item() == 2
    assert x.loc[rounds.RoundNumber.eq(13), "HalfScoreDiff"].item() == 0


def test_recent_feature_perspective_swap_commutes(rounds):
    raw = rounds.copy()
    raw[["Team1_PreScore", "Team2_PreScore"]] = rounds[
        ["Team2_PreScore", "Team1_PreScore"]
    ].to_numpy()
    raw.Team1_RoundWon = 1 - raw.Team1_RoundWon
    raw.Team1_Side = raw.Team1_Side.map({"attack": "defend", "defend": "attack"})
    pd.testing.assert_frame_equal(
        swap_round(recent_round_features(rounds)), recent_round_features(raw), check_dtype=False
    )


def test_recent_history_resets_between_games(rounds):
    other = rounds.copy()
    other.GameID = 2
    combined = pd.concat([rounds, other], ignore_index=True)
    result = recent_round_features(combined)
    first = combined.RoundNumber.eq(1)
    assert (
        (
            result.loc[
                first, ["PreviousStreak", "RecentWinDiff3", "RecentWinDiff5", "HalfScoreDiff"]
            ]
            == 0
        )
        .all()
        .all()
    )


def test_research_rejects_original_test_rows(rounds):
    rounds["Split"] = "test"
    with pytest.raises(ValueError, match="Original test rows"):
        design(
            "round",
            rounds,
            pd.DataFrame(),
            rounds.GameID.eq(1),
            "base",
            recent_round_features(rounds),
        )


def test_temporal_boundaries_keep_tied_series_together():
    frame = pd.DataFrame(
        {
            "MatchID": range(40),
            "Date": pd.to_datetime([f"2021-01-{day:02}" for day in np.repeat(np.arange(1, 21), 2)]),
            "Split": ["train"] * 20 + ["validation"] * 20,
        }
    )
    boundaries = time_boundaries(frame)
    for boundary in boundaries.values():
        groups = frame.assign(before=frame.Date.lt(boundary)).groupby("Date").before.nunique()
        assert groups.max() == 1
    assert (
        boundaries["fold1"]
        < boundaries["fold2"]
        < boundaries["select_end"]
        < boundaries["calibrate_end"]
    )


def test_candidate_grid_and_paired_comparison():
    assert len(candidates("round")) == 24
    assert len(candidates("player")) == 18
    result = paired_interval(
        pd.Series([1, 1, 2, 2]),
        np.array([0, 1, 0, 1]),
        np.array([0.1, 0.9, 0.1, 0.9]),
        np.full(4, 0.5),
    )
    assert result["candidate_minus_baseline"] < 0
    assert result["series_bootstrap_95pct"][1] < 0
