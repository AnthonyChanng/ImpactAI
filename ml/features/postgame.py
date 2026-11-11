"""Individual postgame feature allowlists; never consult another player's scoreboard."""

import numpy as np

from ml.features.build import COMPONENTS, PLAYER_CAT, PLAYER_NUM
from ml.models.context import ContextIndex

EXPANDED_NUM = [c for c in PLAYER_NUM if c not in ["ClutchRate", "ObjectiveRate"]] + [
    "ACS",
    "NumRounds",
    "NetFirstEngagements",
    "OpeningInvolvement",
    "OpeningSuccess",
    "TwoKillRate",
    "ThreeKillRate",
    "FourPlusKillRate",
    "PlantRate",
    "DefuseRate",
    "SingleClutchRate",
    "MultiOpponentClutchRate",
]


class PostgameFeatures:
    def __init__(self, feature_set):
        if feature_set not in {"base", "context", "expanded", "expanded_context"}:
            raise ValueError("Unknown player feature set")
        self.feature_set = feature_set

    def fit(self, base):
        self.cat = PLAYER_CAT.copy()
        self.num = (EXPANDED_NUM if "expanded" in self.feature_set else PLAYER_NUM).copy()
        self.context = None
        if "context" in self.feature_set:
            self.context = ContextIndex().fit(base)
            self.num += [f"ContextZ_{c}" for c in COMPONENTS]
        return self

    def transform(self, base):
        x = base.copy()
        if "expanded" in self.feature_set:
            denominator = x.NumRounds
            if (denominator <= 0).any() or denominator.isna().any():
                raise ValueError("Postgame round counts must be positive")
            x["OpeningInvolvement"] = x.FKPR + x.FDPR
            engagements = x.FirstKills + x.FirstDeaths
            # No engagements means unavailable success percentage, not 0% success.
            x["OpeningSuccess"] = x.FirstKills / engagements.replace(0, np.nan)
            x["TwoKillRate"] = x.Num_2Ks / denominator
            x["ThreeKillRate"] = x.Num_3Ks / denominator
            x["FourPlusKillRate"] = (x.Num_4Ks + x.Num_5Ks) / denominator
            x["PlantRate"] = x.Plants / denominator
            x["DefuseRate"] = x.Defuses / denominator
            x["SingleClutchRate"] = x.OnevOne / denominator
            x["MultiOpponentClutchRate"] = (
                x.OnevTwo + x.OnevThree + x.OnevFour + x.OnevFive
            ) / denominator
        if self.context is not None:
            z, _, _ = self.context.standardize(base)
            x[[f"ContextZ_{c}" for c in COMPONENTS]] = z
        return x[self.cat + self.num]
