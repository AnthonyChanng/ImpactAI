import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score


def metrics(y, p):
    y = np.asarray(y)
    actual, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "n": len(y),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "calibration": [
            {"predicted": float(a), "observed": float(b)} for a, b in zip(predicted, actual)
        ],
    }


def slices(frame, target, predictions, columns):
    out = {}
    for col in columns:
        out[col] = {}
        for value in sorted(frame[col].fillna("unknown").unique()):
            mask = frame[col].fillna("unknown").eq(value).to_numpy()
            out[col][str(value)] = metrics(target[mask], predictions[mask])
    return out


def grouped_interval(frame, y, p, seed=42):
    """Series-cluster bootstrap log loss interval; does not treat ten player rows as IID."""
    losses = -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
    grouped = frame[["MatchID"]].copy()
    grouped["loss"] = losses
    sums = grouped.groupby("MatchID").loss.agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(300):
        sample = sums[rng.integers(0, len(sums), len(sums))]
        draws.append(sample[:, 0].sum() / sample[:, 1].sum())
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]
