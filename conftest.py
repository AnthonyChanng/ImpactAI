"""Keep private-data integration checks explicit and clean-clone tests reproducible."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-dataset",
        action="store_true",
        help="Fail instead of skipping dataset tests when local data/artifacts are absent.",
    )


def pytest_runtest_setup(item):
    if item.get_closest_marker("dataset") is None:
        return
    required = [
        "data/source/games_joined.csv",
        "data/source/players_joined.csv",
        "data/source/rounds_normalized.csv",
        "ml/artifacts/current.json",
        "data/impactai.db",
    ]
    missing = [name for name in required if not Path(name).is_file()]
    if missing:
        message = (
            "Dataset integration prerequisites missing; follow README verification: "
            + ", ".join(missing)
        )
        if item.config.getoption("--require-dataset"):
            pytest.fail(message)
        pytest.skip(message)
