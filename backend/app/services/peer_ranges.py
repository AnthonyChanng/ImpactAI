"""Read verified, saved research comparisons without loading executable model bundles."""

import hashlib
import json
import logging
import math
from pathlib import Path

import pandas as pd

from ml.features.build import COMPONENTS

logger = logging.getLogger(__name__)


class PeerRanges:
    def __init__(self, directory: Path):
        self.rows = {}
        self.coverage = {}
        self.run = directory.name
        self.nominal = 0.8
        self.reason = "Experimental ranges are unavailable; no saved research sample is configured."
        if not directory.exists():
            return
        try:
            hashes = json.loads((directory / "checksums.json").read_text())
            for name in ("results.json", "diagnostic-predictions.parquet"):
                if hashlib.sha256((directory / name).read_bytes()).hexdigest() != hashes[name]:
                    raise ValueError("Research checksum mismatch")
            result = json.loads((directory / "results.json").read_text())
            self.nominal = float(result["nominal_coverage"])
            if not 0 < self.nominal < 1:
                raise ValueError("Invalid nominal coverage")
            self.coverage = {
                c: float(
                    result["confirmation"]["candidate_calibrated"]["by_component"][c]["coverage"]
                )
                for c in COMPONENTS
            }
            if any(not 0 <= value <= 1 for value in self.coverage.values()):
                raise ValueError("Invalid measured coverage")
            frame = pd.read_parquet(directory / "diagnostic-predictions.parquet")
            if frame.duplicated(["GameID", "PlayerID"]).any():
                raise ValueError("Duplicate research observation")
            for row in frame.to_dict("records"):
                for component in COMPONENTS:
                    values = [
                        row[f"candidate_calibrated_{component}_q{q}"] for q in (0.1, 0.5, 0.9)
                    ]
                    if not all(math.isfinite(v) for v in values) or values != sorted(values):
                        raise ValueError("Invalid saved interval")
                self.rows[(int(row["GameID"]), int(row["PlayerID"]))] = row
            self.reason = "This performance is outside the saved research sample; experimental ranges are unavailable."
        except (OSError, ValueError, KeyError, TypeError):
            self.rows = {}
            self.nominal = 0.8
            self.coverage = {}
            self.reason = "Experimental ranges are unavailable because the saved research files failed verification."
            logger.warning("Peer-range research files failed verification", exc_info=True)

    def for_performance(self, performance):
        response = {
            "available": False,
            "reason": self.reason,
            "research_run": self.run,
            "nominal_coverage": self.nominal,
            "components": {},
            "limitation": "Experimental ranges of individual observations, evaluated on reused development validation. Coverage is not guaranteed; these are not confidence intervals or causal impact estimates.",
        }
        game = performance["game"]
        row = self.rows.get((game["id"], performance["player_id"]))
        if row is None:
            return response
        # Do not silently join research from a different source snapshot or feature definition.
        if (
            row["MatchID"] != game["match_id"]
            or pd.Timestamp(row["Date"]) != pd.Timestamp(game["date"])
            or row["Map"] != game["map"]
            or row["Agent"] != performance["agent"]
            or row["NumRounds"] != game["num_rounds"]
            or row["Patch"] != game["patch"]
            or game["split"] != "validation"
            or any(
                not math.isclose(
                    row[f"{c}_observed"],
                    performance["impact"]["components"][c]["value"],
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                )
                for c in COMPONENTS
            )
        ):
            return {
                **response,
                "reason": "The saved research observation does not match this performance.",
            }
        return {
            **response,
            "available": True,
            "reason": None,
            "components": {
                c: {
                    "low": float(row[f"candidate_calibrated_{c}_q0.1"]),
                    "median": float(row[f"candidate_calibrated_{c}_q0.5"]),
                    "high": float(row[f"candidate_calibrated_{c}_q0.9"]),
                    "measured_coverage": self.coverage[c],
                }
                for c in COMPONENTS
            },
        }
