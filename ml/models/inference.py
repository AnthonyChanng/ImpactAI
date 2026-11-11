import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

from ml.features.build import swap_round


def probabilities(model, x, symmetric=False, temperature=1.0):
    p = model.predict_proba(x)[:, 1]
    if symmetric:
        p = (p + 1 - model.predict_proba(swap_round(x))[:, 1]) / 2
    return expit(logit(np.clip(p, 1e-6, 1 - 1e-6)) / temperature)


def load_bundle(directory: Path):
    """Only load locally trained, trusted artifacts. Hashes detect corruption, not hostile rewrites."""
    manifest = json.loads((directory / "manifest.json").read_text())
    hashes = manifest.get("sha256", {})
    required = {
        "bundle.joblib",
        "metrics.json",
        "audit.json",
        "feature-schema.json",
        "player-predictions.jsonl",
        "round-predictions.parquet",
    }
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise ValueError("Artifact checksums missing required files")
    if manifest.get("version") != directory.name:
        raise ValueError("Artifact version does not match directory")
    for name, expected in hashes.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe artifact path")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    bundle = joblib.load(directory / "bundle.joblib")
    if not isinstance(bundle, dict) or bundle.get("version") != manifest["version"]:
        raise ValueError("Model bundle version does not match manifest")
    return bundle, manifest
