"""Artificial contracts for contextual ranges; quality uses source matches only."""

import copy

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.features.build import COMPONENTS, component_values
from scripts.peer_intervals import (
    CONTEXT_COLUMNS,
    QUANTILES,
    RECIPE,
    apply_expansion,
    feature_matrix,
    fit_empirical,
    fit_expansion,
    fit_learned,
    metric_arrays,
    ordered_supported,
    predict_empirical,
    predict_learned,
    predict_selected,
    prepare_development,
    select_components,
    summarize,
)


def test_empirical_quantiles_and_fallback_use_only_training_context(players):
    train = players.iloc[:120].copy()
    train["Agent"] = ["a"] * 100 + ["b"] * 20
    train["Map"] = ["first"] * 50 + ["second"] * 50 + ["third"] * 20
    model = fit_empirical(train, min_peers=50)
    future = players.iloc[120:124].copy()
    future["Agent"] = ["a", "a", "b", "never_seen"]
    future["Map"] = ["first", "never_seen", "third", "first"]
    expected, levels = predict_empirical(model, future)
    assert levels == ["agent_map", "agent", "global", "global"]
    y = component_values(train).to_numpy()
    np.testing.assert_array_equal(expected[0], np.quantile(y[:50], QUANTILES, axis=0).T)
    np.testing.assert_array_equal(expected[1], np.quantile(y[:100], QUANTILES, axis=0).T)
    np.testing.assert_array_equal(expected[2], np.quantile(y, QUANTILES, axis=0).T)
    np.testing.assert_array_equal(expected[3], expected[2])
    changed = future.copy()
    changed[["KPR", "DPR", "ADR", "APR", "ObjectiveRate", "ClutchRate"]] = 9000
    changed[["TeamWon", "PlayerID", "GameID", "MatchID"]] = -99
    np.testing.assert_array_equal(predict_empirical(model, changed)[0], expected)


def test_pinball_interval_score_and_inclusive_coverage_have_known_values():
    prediction = np.tile(np.array([0.0, 1.0, 2.0]), (3, 6, 1))
    y = np.tile(np.array([-1.0, 1.0, 3.0])[:, None], (1, 6))
    arrays = metric_arrays(y, prediction, np.full(6, 2.0))
    np.testing.assert_allclose(
        arrays["pinball"][:, 0], [[0.9, 1.0, 0.3], [0.1, 0, 0.1], [0.3, 1, 0.9]]
    )
    np.testing.assert_allclose(arrays["normalized_interval_score"][:, 0], [6, 1, 6])
    summary = summarize(y, prediction, np.full(6, 2.0))
    assert summary["by_component"]["Combat"]["coverage"] == pytest.approx(1 / 3)
    assert summary["by_component"]["Combat"]["normalized_interval_score"] == pytest.approx(13 / 3)
    boundary_y = np.tile(np.array([0.0, 2.0])[:, None], (1, 6))
    assert metric_arrays(boundary_y, prediction[:2], np.ones(6))["coverage"].all()


@pytest.mark.parametrize("count,expected_rank", [(1, 1), (4, 4), (5, 5), (9, 8)])
def test_calibration_uses_finite_rank_and_never_shrinks(count, expected_rank):
    prediction = np.zeros((count, 6, 3))
    y = np.tile(np.arange(count)[:, None], (1, 6))
    calibration = fit_expansion(y, prediction)
    assert calibration["rank"] == expected_rank
    np.testing.assert_array_equal(calibration["expansion"], np.full(6, expected_rank - 1))
    inside = np.ones((count, 6))
    enclosing = np.tile([0.0, 1.0, 2.0], (count, 6, 1))
    np.testing.assert_array_equal(fit_expansion(inside, enclosing)["expansion"], np.zeros(6))


def test_ordering_support_and_expansion_preserve_median_and_signed_components():
    raw = np.tile([2.0, -3.0, 1.0], (2, 6, 1))
    ordered = ordered_supported(raw)
    np.testing.assert_array_equal(ordered[0, :2], [[-3, 1, 2], [-3, 1, 2]])
    np.testing.assert_array_equal(ordered[0, 2:], np.tile([0, 1, 2], (4, 1)))
    expanded = apply_expansion(ordered, {"expansion": np.full(6, 2.0)})
    np.testing.assert_array_equal(expanded[:, :, 1], ordered[:, :, 1])
    assert (expanded[:, :, 0] <= ordered[:, :, 0]).all()
    assert (expanded[:, :, 2] >= ordered[:, :, 2]).all()
    assert (expanded[:, 2:, :] >= 0).all()
    assert (np.diff(expanded, axis=2) >= 0).all()
    assert (raw[:, :, 0] == 2).all(), "Ordering must not mutate caller arrays"


def selection_result(empirical=1.0, learned=0.9):
    return {
        name: {"by_component": {c: {"normalized_pinball": loss} for c in COMPONENTS}}
        for name, loss in [("empirical", empirical), ("learned", learned)]
    }


def test_selection_requires_every_fold_and_earlier_selection_without_diagnostic_access():
    screening = [selection_result() for _ in range(3)]
    selection = selection_result()
    screening[0]["learned"]["by_component"]["Combat"]["normalized_pinball"] = 1.0
    selection["learned"]["by_component"]["Clutches"]["normalized_pinball"] = 1.1
    chosen = select_components(screening, selection)
    assert chosen["Combat"] == chosen["Clutches"] == "empirical"
    assert chosen["Damage"] == "learned"
    changed = copy.deepcopy(selection)
    changed["confirmation"] = {"learned": 0.0, "empirical": 1.0}
    assert select_components(screening, changed) == chosen


@pytest.mark.parametrize("fit_function", [fit_empirical, fit_learned])
@pytest.mark.parametrize("forbidden_split", ["validation", "test"])
def test_fit_rejects_any_nontraining_rows(players, fit_function, forbidden_split):
    bad = players.iloc[:30].copy()
    bad.loc[bad.index[-1], "Split"] = forbidden_split
    with pytest.raises(ValueError, match="original training"):
        fit_function(bad)


def test_preparation_excludes_original_test_before_feature_building(players):
    raw = players.copy()
    raw["Split"] = np.where(raw.MatchID < 28, "train", "validation")
    raw.loc[raw.index[-8:], "Split"] = "test"
    raw.loc[raw.Split.eq("test"), "NumRounds"] = 0
    games = raw[["GameID", "MatchID", "Date", "Split"]].drop_duplicates().copy()
    games["Team1_Rounds"], games["Team2_Rounds"] = 13, 11
    base, masks = prepare_development({"players": raw, "games": games})
    assert len(base) == len(raw) - 8
    assert not base.Split.eq("test").any()
    assert not set(raw.loc[raw.Split.eq("test"), "GameID"]) & set(base.GameID)
    assert np.stack([m.to_numpy() for m in masks.values()]).sum(axis=0).tolist() == [1] * len(base)
    previous = None
    for mask in masks.values():
        rows = base.loc[mask]
        if previous is not None:
            assert previous.Date.max() < rows.Date.min()
            assert set(previous.MatchID).isdisjoint(rows.MatchID)
        previous = rows


def test_learned_unknown_categories_invariance_and_portable_reload(players, tmp_path):
    train, future = players.iloc[:120], players.iloc[120:].copy()
    model = fit_learned(train, {**RECIPE, "max_iter": 3, "min_samples_leaf": 5})
    future["Agent"] = "unseen"
    future["Patch"] = None
    encoded = model["pre"].transform(feature_matrix(future))
    assert np.isnan(encoded[:, 0]).all()
    prediction = predict_learned(model, future)
    assert np.isfinite(prediction).all()
    assert (prediction[:, 2:, :] >= 0).all()
    changed = future[CONTEXT_COLUMNS].copy()
    np.testing.assert_array_equal(predict_learned(model, changed), prediction)
    empirical = fit_empirical(train)
    bundle = {
        "empirical": empirical,
        "learned": model,
        "component_sources": {
            c: "learned" if i % 2 else "empirical" for i, c in enumerate(COMPONENTS)
        },
    }
    expected = predict_selected(bundle, future)
    np.testing.assert_array_equal(expected[:, 1], prediction[:, 1])
    np.testing.assert_array_equal(expected[:, 0], predict_empirical(empirical, future)[0][:, 0])
    path = tmp_path / "portable.joblib"
    joblib.dump(bundle, path)
    np.testing.assert_array_equal(predict_selected(joblib.load(path), changed), expected)
    np.testing.assert_array_equal(predict_selected(bundle, future.iloc[:1]), expected[:1])


@pytest.mark.parametrize("round_count", [0, -1, np.nan, np.inf])
def test_invalid_round_counts_are_rejected(players, round_count):
    players["NumRounds"] = players.NumRounds.astype(float)
    players.loc[0, "NumRounds"] = round_count
    with pytest.raises(ValueError, match="finite and positive"):
        feature_matrix(players)


def test_calibration_and_metric_invalid_shapes_or_values_are_rejected():
    y, p = np.ones((5, 6)), np.ones((5, 6, 3))
    with pytest.raises(ValueError, match="between zero and one"):
        fit_expansion(y, p, 1)
    with pytest.raises(ValueError, match="shapes"):
        fit_expansion(y, p[:, :5])
    with pytest.raises(ValueError, match="finite"):
        fit_expansion(y * np.nan, p)
    with pytest.raises(ValueError, match="nonnegative"):
        apply_expansion(p, {"expansion": np.full(6, -1)})
    with pytest.raises(ValueError, match="finite"):
        ordered_supported(p * np.nan)


def test_unknown_category_encoding_is_fit_only_on_training(players):
    train = players.iloc[:120].copy()
    future = players.iloc[120:].copy()
    future["Map"] = "only_in_evaluation"
    model = fit_learned(train, {**RECIPE, "max_iter": 1, "min_samples_leaf": 5})
    assert "only_in_evaluation" not in model["pre"].named_transformers_["category"].categories_[1]
    assert np.isnan(model["pre"].transform(feature_matrix(future))[:, 1]).all()
    pd.testing.assert_frame_equal(train[CONTEXT_COLUMNS], feature_matrix(train))
