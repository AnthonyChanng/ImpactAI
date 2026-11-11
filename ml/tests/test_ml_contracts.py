"""Artificial fixtures exercise software invariants; they are not evaluation data."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.evaluation.metrics import metrics
from ml.features.build import ROUND_CAT, ROUND_NUM, round_features, swap_round
from ml.models.context import ContextIndex
from ml.models.inference import load_bundle, probabilities
from ml.train import pipeline


@pytest.fixture
def artifact(tmp_path):
    directory = tmp_path / "v1-contract-test"
    directory.mkdir()
    joblib.dump({"version": directory.name}, directory / "bundle.joblib")
    for name in [
        "audit.json",
        "metrics.json",
        "feature-schema.json",
        "player-predictions.jsonl",
        "round-predictions.parquet",
    ]:
        (directory / name).write_text("{}")
    manifest = {
        "version": directory.name,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


def test_checked_bundle_round_trip(artifact):
    directory, manifest = artifact
    bundle, loaded_manifest = load_bundle(directory)
    assert bundle["version"] == manifest["version"]
    assert loaded_manifest == manifest


@pytest.mark.parametrize(
    "failure",
    [
        "missing_checksum",
        "missing_predictions",
        "corrupt_bundle",
        "corrupt_metadata",
        "path",
        "version",
    ],
)
def test_artifact_failure_prevents_deserialization(artifact, monkeypatch, failure):
    directory, manifest = artifact
    if failure == "missing_checksum":
        del manifest["sha256"]["bundle.joblib"]
    elif failure == "missing_predictions":
        del manifest["sha256"]["round-predictions.parquet"]
    elif failure == "corrupt_bundle":
        (directory / "bundle.joblib").write_bytes(b"invalid")
    elif failure == "corrupt_metadata":
        (directory / "audit.json").write_text('{"counts": {}}')
    elif failure == "path":
        manifest["sha256"]["../outside"] = "invalid"
    else:
        manifest["version"] = "v1-wrong-version"
    (directory / "manifest.json").write_text(json.dumps(manifest))

    def must_not_deserialize(*args, **kwargs):
        pytest.fail("Deserialization must follow complete manifest validation")

    monkeypatch.setattr(joblib, "load", must_not_deserialize)
    with pytest.raises(ValueError):
        load_bundle(directory)


def test_bundle_identity_must_match_manifest(artifact, monkeypatch):
    directory, _ = artifact
    monkeypatch.setattr(joblib, "load", lambda _: {"version": "v1-wrong"})
    with pytest.raises(ValueError, match="bundle version"):
        load_bundle(directory)


@pytest.fixture
def rounds():
    return pd.DataFrame(
        {
            "GameID": [2, 1, 2, 1],
            "RoundNumber": [2, 1, 1, 2],
            "Map": ["Ascent"] * 4,
            "Patch": [None] * 4,
            "Team1_PreScore": [1, 0, 0, 0],
            "Team2_PreScore": [0, 0, 0, 1],
            "Team1_Side": ["attack", "defend", "attack", "defend"],
            "Team1_RoundWon": [0, 0, 1, 1],
            "WinType_POST_ROUND": ["elim"] * 4,
        }
    )


def test_round_history_is_grouped_shifted_and_independent_of_current_outcome(rounds):
    x = round_features(rounds)
    assert list(x) == ROUND_CAT + ROUND_NUM
    assert x.PreviousWin.tolist() == [1, 0.5, 0.5, 0]
    mutated = rounds.copy()
    mutated.loc[0, "Team1_RoundWon"] = 1
    mutated["WinType_POST_ROUND"] = "changed"
    pd.testing.assert_frame_equal(x, round_features(mutated))
    pd.testing.assert_frame_equal(x, swap_round(swap_round(x)))


def test_fitted_pipeline_symmetry_schema_and_serialization(rounds, tmp_path):
    x = round_features(rounds)
    model = pipeline(ROUND_CAT, ROUND_NUM, "logistic").fit(x, rounds.Team1_RoundWon)
    p = probabilities(model, x, symmetric=True, temperature=0.8)
    inverse = probabilities(model, swap_round(x), symmetric=True, temperature=0.8)
    np.testing.assert_allclose(p + inverse, 1, atol=1e-12)
    joblib.dump(model, tmp_path / "pipeline.joblib")
    np.testing.assert_array_equal(
        model.predict_proba(x), joblib.load(tmp_path / "pipeline.joblib").predict_proba(x)
    )
    with pytest.raises((ValueError, KeyError)):
        model.predict_proba(x.drop(columns=["ScoreDiff"]))


def test_context_uses_frozen_references_and_bounded_midrank_scores():
    reference = pd.DataFrame(
        {
            "Agent": ["jett"] * 60,
            "Map": ["Ascent"] * 60,
            "Patch": ["test"] * 60,
            "LengthBand": ["regulation"] * 60,
            "KPR": [1.0] * 60,
            "DPR": [0.5] * 60,
            "FKPR": [0.1] * 60,
            "FDPR": [0.1] * 60,
            "ADR": [100.0] * 60,
            "APR": [0.1] * 60,
            "ObjectiveRate": [0.0] * 60,
            "ClutchRate": [0.0] * 60,
        }
    )
    model = ContextIndex().fit(reference)
    mean = model.mean.copy()
    expected = model.transform(reference.iloc[:1])[0]
    assert expected["score"] == pytest.approx(50)
    assert expected["percentile"] == pytest.approx(50)
    assert expected["peer_level"] == "agent_map"
    unseen = reference.iloc[:1].copy()
    unseen["Agent"] = "unseen"
    unseen["ADR"] = 1e9
    result = model.transform(unseen)[0]
    assert result["peer_level"] == "global"
    assert result["peer_count"] == 60
    assert result["components"]["Damage"]["z"] == 3
    assert 0 <= result["score"] <= 100
    np.testing.assert_array_equal(model.mean, mean)
    assert model.transform(reference.iloc[:1])[0] == expected


def test_metrics_have_known_values_and_handle_single_class_slices():
    result = metrics(np.array([0, 1]), np.array([0.25, 0.75]))
    assert result["log_loss"] == pytest.approx(-np.log(0.75))
    assert result["brier"] == pytest.approx(0.0625)
    assert result["roc_auc"] == 1
    assert result["accuracy"] == 1
    assert metrics(np.array([1, 1]), np.array([0.5, 0.75]))["roc_auc"] is None
