"""Research ranges stay optional, verified, and tied to matching observations."""

import hashlib
import json
from copy import deepcopy

import pandas as pd
import pytest

from backend.app.schemas import PeerRangesOut
from backend.app.services.peer_ranges import PeerRanges
from ml.features.build import COMPONENTS


def write_sample(root, *, duplicate=False, nominal=0.8, reversed_range=False):
    row = {
        "GameID": 1,
        "PlayerID": 2,
        "MatchID": 3,
        "Date": "2021-10-01",
        "Map": "Ascent",
        "Agent": "jett",
        "Patch": "3.0",
        "NumRounds": 20,
    }
    for component in COMPONENTS:
        row[f"{component}_observed"] = 0.2
        for q, value in zip(
            (0.1, 0.5, 0.9), (0.4, 0.2, 0.1) if reversed_range else (0.1, 0.2, 0.4)
        ):
            row[f"candidate_calibrated_{component}_q{q}"] = value
    pd.DataFrame([row, row] if duplicate else [row]).to_parquet(
        root / "diagnostic-predictions.parquet"
    )
    (root / "results.json").write_text(
        json.dumps(
            {
                "nominal_coverage": nominal,
                "confirmation": {
                    "candidate_calibrated": {
                        "by_component": {c: {"coverage": 0.77} for c in COMPONENTS}
                    }
                },
            }
        )
    )
    (root / "checksums.json").write_text(
        json.dumps(
            {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("results.json", "diagnostic-predictions.parquet")
            }
        )
    )
    return {
        "player_id": 2,
        "agent": "jett",
        "game": {
            "id": 1,
            "match_id": 3,
            "date": "2021-10-01",
            "map": "Ascent",
            "patch": "3.0",
            "num_rounds": 20,
            "split": "validation",
        },
        "impact": {"components": {c: {"value": 0.2} for c in COMPONENTS}},
    }


def test_verified_ranges_expose_measured_coverage_and_reject_source_mismatch(tmp_path):
    performance = write_sample(tmp_path)
    ranges = PeerRanges(tmp_path)
    result = PeerRangesOut.model_validate(ranges.for_performance(performance))
    assert result.available
    assert result.components["Combat"].measured_coverage == 0.77
    assert result.components["Combat"].low == 0.1
    for change in ("value", "map", "split", "date"):
        changed = deepcopy(performance)
        if change == "value":
            changed["impact"]["components"]["Combat"]["value"] = 10
        else:
            changed["game"][change] = {"map": "Breeze", "split": "test", "date": "2022-01-01"}[
                change
            ]
        assert not ranges.for_performance(changed)["available"]
    performance["game"]["id"] = 99
    assert "outside" in ranges.for_performance(performance)["reason"]


def test_checksum_rejection_and_missing_optional_files(tmp_path):
    performance = write_sample(tmp_path)
    with (tmp_path / "results.json").open("a") as f:
        f.write(" ")
    result = PeerRanges(tmp_path).for_performance(performance)
    assert not result["available"] and "failed verification" in result["reason"]
    assert not PeerRanges(tmp_path / "missing").for_performance(performance)["available"]


@pytest.mark.parametrize(
    "options", [{"duplicate": True}, {"nominal": 1.5}, {"reversed_range": True}]
)
def test_malformed_saved_ranges_fail_closed_with_valid_unavailable_response(tmp_path, options):
    performance = write_sample(tmp_path, **options)
    result = PeerRangesOut.model_validate(PeerRanges(tmp_path).for_performance(performance))
    assert not result.available
