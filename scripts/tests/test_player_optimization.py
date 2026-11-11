import joblib
import numpy as np
import pytest

from scripts.optimize_player import (
    calibrated,
    fit_calibration,
    fit_member,
    group_probability,
    selection_halves,
)


@pytest.mark.parametrize("family", ["identity", "temperature", "sigmoid", "isotonic"])
def test_calibration_is_finite_monotone_and_serializable(family, tmp_path):
    p = np.repeat([0.1, 0.9], 100)
    y = np.array([0] * 70 + [1] * 30 + [0] * 30 + [1] * 70)
    calibration = fit_calibration(family, y, p)
    grid = np.linspace(0, 1, 101)
    expected = calibrated(calibration, grid)
    assert np.isfinite(expected).all()
    assert ((expected > 0) & (expected < 1)).all()
    assert (np.diff(expected) >= 0).all()
    joblib.dump(calibration, tmp_path / "calibration.joblib")
    np.testing.assert_array_equal(
        calibrated(joblib.load(tmp_path / "calibration.joblib"), grid), expected
    )
    if family != "identity":
        assert abs(calibrated(calibration, [0.1])[0] - 0.3) < abs(0.1 - 0.3)


@pytest.mark.parametrize(
    "probabilities,labels",
    [([np.nan, 0.9], [0, 1]), ([np.inf, 0.9], [0, 1]), ([0.1, 0.9], [1, 1]), ([0.1], [0, 1])],
)
def test_invalid_calibration_data_is_rejected(probabilities, labels):
    with pytest.raises(ValueError):
        fit_calibration("sigmoid", labels, probabilities)


def test_native_categorical_training_handles_unknown_categories_and_missing_values(
    players, tmp_path
):
    spec = {
        "name": "test",
        "features": "expanded",
        "parameters": dict(
            max_iter=6, learning_rate=0.1, max_leaf_nodes=3, min_samples_leaf=5, l2_regularization=2
        ),
    }
    member = fit_member(players.iloc[:120], spec)
    assert 1 <= member["iterations"] <= member["probe_iterations"] <= 6
    assert member["inner_fit_max_date"] < member["inner_stop_min_date"]
    assert member["model"]["model"].n_iter_ == member["iterations"]
    future = players.iloc[120:].copy()
    future["Agent"] = "never_in_training"
    future["HS_Percent"] = np.nan
    x = member["features"].transform(future)
    assert np.isnan(member["model"]["pre"].transform(x)[:, 0]).all()
    p = member["model"].predict_proba(x)
    assert np.isfinite(p).all()
    joblib.dump(member, tmp_path / "member.joblib")
    np.testing.assert_array_equal(
        joblib.load(tmp_path / "member.joblib")["model"].predict_proba(x), p
    )


def test_selection_calibration_and_choice_keep_timestamps_and_series_together(players):
    left, right = selection_halves(players, players.Split.eq("validation"))
    assert players.loc[left, "Date"].max() < players.loc[right, "Date"].min()
    assert set(players.loc[left, "MatchID"]).isdisjoint(players.loc[right, "MatchID"])
    assert not (left & right).any()


@pytest.mark.parametrize("weights", [[], [-0.1, 1.1], [0.2, 0.2], [np.nan, 1.0]])
def test_invalid_mixtures_fail_before_inference(weights):
    with pytest.raises(ValueError, match="weights"):
        group_probability(
            [{"source": str(i), "weight": w} for i, w in enumerate(weights)], {}, None
        )
