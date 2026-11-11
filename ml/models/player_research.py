"""Portable inference for research player candidates and outcome-free peer means."""

import numpy as np
from scipy.special import expit, logit
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from ml.features.build import PLAYER_CAT, component_values
from ml.models.context import ContextIndex
from ml.train import pipeline


def ensemble_probability(members, base, temperature=1.0):
    if not members or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("An ensemble needs members and a positive temperature")
    probabilities = [m["model"].predict_proba(m["features"].transform(base))[:, 1] for m in members]
    mean = np.mean(probabilities, axis=0)
    return expit(logit(np.clip(mean, 1e-6, 1 - 1e-6)) / temperature)


class PeerMeanModel:
    """Estimate expected component values from context only; not an impact score."""

    def __init__(self, kind="hierarchical", parameters=None):
        if kind not in {"hierarchical", "global", "boosted"}:
            raise ValueError("Unknown peer model")
        self.kind = kind
        self.parameters = parameters or {}

    def fit(self, base):
        target = component_values(base).to_numpy()
        self.mean = target.mean(axis=0)
        self.scale = np.sqrt(np.maximum(target.var(axis=0), 1e-8))
        self.context = ContextIndex().fit(base)
        if self.kind == "boosted":
            self.pre = pipeline(PLAYER_CAT, ["NumRounds"])["pre"]
            x = self.pre.fit_transform(base)
            self.model = MultiOutputRegressor(
                XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_jobs=4,
                    random_state=42,
                    **self.parameters,
                ),
                n_jobs=1,
            ).fit(x, (target - self.mean) / self.scale)
        return self

    def predict(self, base):
        if self.kind == "global":
            return np.tile(self.mean, (len(base), 1))
        if self.kind == "hierarchical":
            # Standardize also computes observed residuals, but only its context means
            # are returned; observed player statistics cannot affect those means.
            return self.context.standardize(base)[1]
        return self.model.predict(self.pre.transform(base)) * self.scale + self.mean


def peer_error(model, base):
    squared = ((component_values(base).to_numpy() - model.predict(base)) / model.scale) ** 2
    return squared.mean(axis=0), squared.mean(axis=1)
