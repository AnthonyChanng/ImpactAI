"""Train-only smoothed reference distributions and a transparent heuristic index."""

import numpy as np
import pandas as pd

from ml.features.build import COMPONENTS, component_values


class ContextIndex:
    levels = [
        ["Agent"],
        ["Agent", "Map"],
        ["Agent", "Map", "Patch"],
        ["Agent", "Map", "Patch", "LengthBand"],
    ]

    def fit(self, train):
        values = component_values(train)
        self.mean = values.mean().to_numpy()
        self.var = values.var(ddof=0).clip(lower=1e-6).to_numpy()
        self.groups = []
        for cols in self.levels:
            table = pd.concat([train[cols], values], axis=1)
            groups = {}
            for key, rows in table.groupby(cols, observed=True):
                key = key if isinstance(key, tuple) else (key,)
                a = rows[COMPONENTS].to_numpy()
                groups[key] = (len(a), a.mean(axis=0), np.mean(a * a, axis=0))
            self.groups.append(groups)
        self.reference = train[["Agent", "Map"]].copy()
        z, _, _ = self.standardize(train)
        self.reference["score"] = np.clip(50 + 15 * z.mean(axis=1), 0, 100)
        self.global_scores = np.sort(self.reference.score.to_numpy())
        self.peer_scores = {}
        for cols in [["Agent"], ["Agent", "Map"]]:
            for key, rows in self.reference.groupby(cols, observed=True):
                key = key if isinstance(key, tuple) else (key,)
                if len(rows) >= 50:
                    self.peer_scores[key] = np.sort(rows.score.to_numpy())
        return self

    def standardize(self, x):
        values = component_values(x).to_numpy()
        means = np.tile(self.mean, (len(x), 1))
        seconds = np.tile(self.var + self.mean**2, (len(x), 1))
        counts = np.zeros(len(x), dtype=int)
        for cols, lookup in zip(self.levels, self.groups):
            keys = list(x[cols].itertuples(index=False, name=None))
            n = np.array([lookup.get(k, (0, self.mean, self.var))[0] for k in keys])
            local_mean = np.array([lookup.get(k, (0, self.mean, self.var))[1] for k in keys])
            local_second = np.array(
                [lookup.get(k, (0, self.mean, self.var + self.mean**2))[2] for k in keys]
            )
            weight = (n / (n + 50))[:, None]
            means = weight * local_mean + (1 - weight) * means
            seconds = weight * local_second + (1 - weight) * seconds
            counts = n
        std = np.sqrt(np.maximum(seconds - means**2, self.var * 0.1))
        return np.clip((values - means) / std, -3, 3), means, counts

    def transform(self, x):
        z, means, counts = self.standardize(x)
        score = np.clip(50 + 15 * z.mean(axis=1), 0, 100)
        records = []
        raw = component_values(x).to_numpy()
        for i, row in enumerate(x.itertuples()):
            key = (row.Agent, row.Map)
            level = "agent_map"
            if key not in self.peer_scores:
                key, level = (row.Agent,), "agent"
            ref = self.peer_scores.get(key, self.global_scores)
            if key not in self.peer_scores:
                level = "global"
            # Midrank handles large masses of tied objective/clutch observations.
            rank = (
                np.searchsorted(ref, score[i], "left") + np.searchsorted(ref, score[i], "right")
            ) / 2
            records.append(
                {
                    "score": float(score[i]),
                    "percentile": float(100 * rank / len(ref)),
                    "peer_count": len(ref),
                    "peer_level": level,
                    "exact_context_count": int(counts[i]),
                    "components": {
                        c: {
                            "z": float(z[i, j]),
                            "value": float(raw[i, j]),
                            "peer_mean": float(means[i, j]),
                        }
                        for j, c in enumerate(COMPONENTS)
                    },
                }
            )
        return records
