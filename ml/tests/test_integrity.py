import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.features.build import ROUND_CAT, ROUND_NUM, player_features, round_features, swap_round
from ml.models.context import ContextIndex
from ml.models.inference import load_bundle, probabilities
from ml.preprocessing.audit import audit, load_data
from ml.train import pipeline


@pytest.fixture(scope="module")
def frames():
    return load_data(Path("data/source"))


@pytest.fixture(scope="module")
def bundle():
    root = Path("ml/artifacts")
    version = json.loads((root / "current.json").read_text())["version"]
    return load_bundle(root / version)[0]


@pytest.mark.dataset
def test_source_relationships_chronology_and_discovered_outcomes(frames):
    report = audit(frames)
    assert report["errors"] == []
    assert set(report["unreliable_outcome_game_ids"]) == {52669, 20210}
    assert report["series"] == 6404


@pytest.mark.dataset
def test_round_allowlist_and_future_independence(frames):
    r = frames["rounds"].iloc[:50].copy()
    x = round_features(r)
    assert set(x) == set(ROUND_CAT + ROUND_NUM)
    mutated = r.copy()
    idx = mutated.index[5]
    mutated.loc[idx, "Team1_RoundWon"] = 1 - mutated.loc[idx, "Team1_RoundWon"]
    mutated["Team1Bank_ObservedTimingUnknown"] = 9999999
    mutated["Team1_PostScore_DO_NOT_TRAIN"] = -999
    changed = round_features(mutated)
    pd.testing.assert_series_equal(x.loc[idx], changed.loc[idx])
    assert not any("POST" in c or "Unknown" in c or "ID" in c for c in x)


@pytest.mark.dataset
def test_audit_rejects_corruption_and_split_leakage(frames):
    corrupt = {k: v.copy() for k, v in frames.items()}
    corrupt["rounds"].loc[0, "Team1_PreScore"] += 1
    corrupt["players"].loc[0, "Split"] = "train"
    errors = audit(corrupt)["errors"]
    assert "team1 pre-score leakage/inconsistency" in errors
    assert "players: series split leakage" in errors
    assert "players: inconsistent Split" in errors


@pytest.mark.dataset
def test_swapping_is_involution_and_model_is_complementary(frames, bundle):
    x = round_features(frames["rounds"].iloc[:1000])
    pd.testing.assert_frame_equal(x, swap_round(swap_round(x)))
    a = probabilities(bundle["round_model"], x, True, bundle["round_temperature"])
    b = probabilities(bundle["round_model"], swap_round(x), True, bundle["round_temperature"])
    np.testing.assert_allclose(a + b, 1, atol=1e-6)


@pytest.mark.dataset
def test_train_only_context_and_unknown_fallback(frames):
    p = player_features(frames["players"])
    train = p[p.Split == "train"].iloc[:3000]
    model = ContextIndex().fit(train)
    original = model.mean.copy()
    unknown = p.iloc[:1].copy()
    unknown["Agent"] = "unseen_agent"
    unknown["ADR"] = 9999
    out = model.transform(unknown)[0]
    assert out["peer_level"] == "global"
    assert out["peer_count"] == len(train)
    assert 0 <= out["score"] <= 100
    np.testing.assert_array_equal(model.mean, original)


@pytest.mark.dataset
def test_seeded_training_reproducibility_and_schema(frames):
    r = frames["rounds"].iloc[:1500]
    x = round_features(r)
    a = pipeline(ROUND_CAT, ROUND_NUM, "xgboost").fit(x, r.Team1_RoundWon)
    b = pipeline(ROUND_CAT, ROUND_NUM, "xgboost").fit(x, r.Team1_RoundWon)
    np.testing.assert_array_equal(a.predict_proba(x), b.predict_proba(x))
    with pytest.raises((ValueError, KeyError)):
        a.predict_proba(x.drop(columns=["ScoreDiff"]))


@pytest.mark.dataset
def test_shap_additivity_and_heuristic_independence():
    root = Path("ml/artifacts")
    version = json.loads((root / "current.json").read_text())["version"]
    with (root / version / "player-predictions.jsonl").open() as f:
        row = json.loads(next(f))
    ex = row["explanation"]
    total = (
        ex["base_log_odds"] + sum(v["contribution"] for v in ex["top"]) + ex["other_contribution"]
    )
    assert total == pytest.approx(ex["raw_log_odds"], abs=1e-5)
    from scipy.special import expit

    assert expit(total / ex["temperature"]) == pytest.approx(
        row["association_probability"], abs=1e-5
    )
    score = np.clip(50 + 15 * np.mean([v["z"] for v in row["components"].values()]), 0, 100)
    assert row["score"] == pytest.approx(score)
