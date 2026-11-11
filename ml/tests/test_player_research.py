"""Artificial fixtures test contracts; reported model quality uses real source matches."""

import copy

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit

from ml.features.build import player_features
from ml.features.postgame import PostgameFeatures
from ml.models.player_research import PeerMeanModel, ensemble_probability
from ml.player_research import (
    assert_development,
    fit_member,
    inner_masks,
    paired_mean_interval,
    temporal_folds,
)


@pytest.fixture
def players():
    rng = np.random.default_rng(42)
    n = 160
    raw = pd.DataFrame(
        {
            "GameID": np.repeat(np.arange(40), 4),
            "MatchID": np.repeat(np.arange(40), 4),
            "PlayerID": np.tile(np.arange(4), 40),
            "Date": np.repeat(pd.date_range("2021-01-01", periods=20), 8),
            "Split": ["train"] * 120 + ["validation"] * 40,
            "Agent": ["jett", "sage"] * 80,
            "Map": ["Ascent"] * n,
            "Patch": ["test"] * n,
            "TeamWon": [0, 1] * 80,
            "NumRounds": [24] * n,
            "Kills": rng.integers(1, 24, n),
            "Deaths": rng.integers(1, 24, n),
            "Assists": rng.integers(0, 10, n),
            "FirstKills": rng.integers(0, 4, n),
            "FirstDeaths": rng.integers(0, 4, n),
            "ADR": rng.normal(150, 30, n),
            "ACS": rng.normal(210, 40, n),
            "HS_Percent": rng.uniform(10, 40, n),
        }
    )
    for col in [
        "OnevOne",
        "OnevTwo",
        "OnevThree",
        "OnevFour",
        "OnevFive",
        "Plants",
        "Defuses",
        "Num_2Ks",
        "Num_3Ks",
        "Num_4Ks",
        "Num_5Ks",
    ]:
        raw[col] = rng.integers(0, 3, n)
    return player_features(raw)


def test_postgame_features_ignore_labels_ids_and_other_players(players):
    model = PostgameFeatures("expanded_context").fit(players.iloc[:120])
    expected = model.transform(players.iloc[120:])
    changed = players.iloc[120:].copy()
    changed["TeamWon"] = 1 - changed.TeamWon
    changed[["PlayerID", "GameID", "MatchID"]] = -1
    changed["Teammates_KPR"] = 1000
    changed["Econ"] = 999
    changed["Team1_Rounds"] = 13
    pd.testing.assert_frame_equal(expected, model.transform(changed))
    allowed = model.cat + model.num
    assert list(expected) == allowed
    assert not set(allowed) & {"TeamWon", "PlayerID", "TeamSide", "Econ", "KAST_Percent"}


def test_zero_openings_are_missing_not_failure_and_invalid_rounds_rejected(players):
    players.loc[0, ["FirstKills", "FirstDeaths", "FKPR", "FDPR"]] = 0
    model = PostgameFeatures("expanded").fit(players)
    assert np.isnan(model.transform(players).loc[0, "OpeningSuccess"])
    assert model.transform(players).loc[0, "OpeningInvolvement"] == 0
    players.loc[0, "NumRounds"] = 0
    with pytest.raises(ValueError, match="positive"):
        model.transform(players)


def test_context_references_are_frozen_and_inference_batch_independent(players):
    model = PostgameFeatures("context").fit(players.iloc[:120])
    before = copy.deepcopy(model.context.mean)
    small = model.transform(players.iloc[120:121])
    pd.testing.assert_frame_equal(small, model.transform(players).iloc[120:121])
    np.testing.assert_array_equal(model.context.mean, before)


def test_temporal_folds_and_inner_stopping_keep_series_and_ties_together(players):
    for fit, evaluate in temporal_folds(players):
        train = players.loc[fit]
        assert train.Date.max() < players.loc[evaluate, "Date"].min()
        assert set(train.MatchID).isdisjoint(players.loc[evaluate, "MatchID"])
        inner, stop = inner_masks(train)
        assert train.loc[inner, "Date"].max() < train.loc[stop, "Date"].min()
        assert set(train.loc[inner, "MatchID"]).isdisjoint(train.loc[stop, "MatchID"])
        assert not (fit & evaluate).any()


@pytest.mark.parametrize("failure", ["test", "split_series", "date_series", "future_train"])
def test_invalid_development_data_is_rejected(players, failure):
    if failure == "test":
        players.loc[0, "Split"] = "test"
    elif failure == "split_series":
        players.loc[0, "Split"] = "validation"
    elif failure == "date_series":
        players.loc[0, "Date"] += pd.Timedelta(days=1)
    else:
        players.loc[players.MatchID.eq(0), "Date"] = pd.Timestamp("2022-01-01")
    with pytest.raises(ValueError):
        assert_development(players)
    with pytest.raises(ValueError, match="original training"):
        inner_masks(players)


@pytest.mark.parametrize("kind", ["global", "hierarchical", "boosted"])
def test_peer_predictions_use_only_context_at_inference(players, kind, tmp_path):
    params = {"n_estimators": 5, "max_depth": 2} if kind == "boosted" else None
    model = PeerMeanModel(kind, params).fit(players.iloc[:120])
    validation = players.iloc[120:].copy()
    expected = model.predict(validation)
    validation[["KPR", "DPR", "FKPR", "FDPR", "ADR", "APR", "ObjectiveRate", "ClutchRate"]] = 1000
    validation["TeamWon"] = 1 - validation.TeamWon
    validation["PlayerID"] = -1
    np.testing.assert_allclose(model.predict(validation), expected)
    joblib.dump(model, tmp_path / "peer.joblib")
    np.testing.assert_array_equal(
        joblib.load(tmp_path / "peer.joblib").predict(validation), expected
    )


def test_early_stopping_refit_and_ensemble_probability_roundtrip(players, tmp_path):
    spec = {
        "name": "test",
        "features": "expanded_context",
        "early_stopping": True,
        "parameters": {"n_estimators": 8, "max_depth": 2},
    }
    a = fit_member(players.iloc[:120], spec)
    b = fit_member(players.iloc[:120], spec, 17, fixed_trees=a["trees"])
    assert 1 <= a["trees"] <= 8
    assert a["model"]["model"].get_booster().num_boosted_rounds() == a["trees"]
    x = players.iloc[120:]
    pa = a["model"].predict_proba(a["features"].transform(x))[:, 1]
    pb = b["model"].predict_proba(b["features"].transform(x))[:, 1]
    expected = expit(logit((pa + pb) / 2) / 0.8)
    np.testing.assert_allclose(ensemble_probability([a, b], x, 0.8), expected)
    joblib.dump([a, b], tmp_path / "ensemble.joblib")
    np.testing.assert_array_equal(
        ensemble_probability(joblib.load(tmp_path / "ensemble.joblib"), x),
        ensemble_probability([a, b], x),
    )
    with pytest.raises(ValueError):
        ensemble_probability([], x)


def test_paired_error_interval_uses_series_groups():
    result = paired_mean_interval([1, 1, 2, 2], [-0.1, -0.1, -0.1, -0.1])
    assert result["difference"] == pytest.approx(-0.1)
    np.testing.assert_allclose(result["series_bootstrap_95pct"], [-0.1, -0.1])
