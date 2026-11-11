"""One frozen, component-specific follow-up to the postgame peer experiment."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features.build import COMPONENTS, component_values, player_features
from ml.improve import write_json
from ml.models.player_research import PeerMeanModel, peer_error
from ml.player_research import assert_development, paired_mean_interval
from ml.preprocessing.audit import load_data


def select_components(result):
    chosen = {}
    for component in COMPONENTS:
        eligible = ["hierarchical"]
        for name in ["boosted_shallow", "boosted_medium"]:
            if all(
                a[component] < b[component]
                for a, b in zip(
                    result["screening"][name], result["screening"]["hierarchical"], strict=True
                )
            ):
                eligible.append(name)
        chosen[component] = min(
            eligible, key=lambda n: result["selection"][n]["by_component"][component]
        )
    return chosen


def predict_components(bundle, base):
    predicted = {name: model.predict(base) for name, model in bundle["models"].items()}
    return np.column_stack(
        [predicted[bundle["component_sources"][c]][:, i] for i, c in enumerate(COMPONENTS)]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use a new empty directory")
    parent_bytes = (args.parent / "experiment.json").read_bytes()
    checksums = json.loads((args.parent / "checksums.json").read_text())
    if hashlib.sha256(parent_bytes).hexdigest() != checksums["experiment.json"]:
        raise ValueError("Parent experiment hash mismatch")
    parent = json.loads(parent_bytes)
    for name, expected in parent["source_hashes"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Source mismatch")
    code_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path("ml").rglob("*.py")))
    ).hexdigest()
    if code_hash != parent["code_hash"]:
        raise ValueError("Use the original experiment's ML source")
    chosen = select_components(parent["peers"])
    frozen = {
        "component_sources": chosen,
        "parent_experiment_hash": hashlib.sha256(parent_bytes).hexdigest(),
        "ml_code_hash": code_hash,
        "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_basis": "Lower error in every training fold, then lowest earlier selection error.",
        "limitation": "Exploratory reused development validation; no independent test or causal impact claim.",
    }
    write_json(args.output / "selection-frozen.json", frozen)
    # A text snapshot must not become an extra live source file in ML fingerprints.
    (args.output / "script-snapshot.txt").write_bytes(Path(__file__).read_bytes())
    frames = load_data(args.data_dir)
    games = frames["games"]
    reliable = games.loc[
        (games.Team1_Rounds != games.Team2_Rounds)
        & (games[["Team1_Rounds", "Team2_Rounds"]].max(axis=1) >= 13),
        "GameID",
    ]
    raw = frames["players"]
    base = player_features(
        raw[raw.Split.isin(["train", "validation"]) & raw.GameID.isin(reliable)].reset_index(
            drop=True
        )
    )
    assert_development(base)
    fit = base.Split.eq("train")
    selection = base.Split.eq("validation") & base.Date.le(
        pd.Timestamp(parent["partitions"]["selection"]["max_date"])
    )
    confirm = base.Split.eq("validation") & base.Date.ge(
        pd.Timestamp(parent["partitions"]["confirmation"]["min_date"])
    )
    models = {}
    for name in ["hierarchical", "boosted_shallow", "boosted_medium"]:
        models[name] = PeerMeanModel(**parent["peer_specs"][name]).fit(base.loc[fit])
        errors, _ = peer_error(models[name], base.loc[selection])
        np.testing.assert_allclose(
            errors,
            [parent["peers"]["selection"][name]["by_component"][c] for c in COMPONENTS],
            atol=1e-8,
            rtol=0,
        )
        print(f"Refit {name}; selection metrics reproduced", flush=True)
    bundle = {"research_only": True, "component_sources": chosen, "models": models}
    joblib.dump(bundle, args.output / "peer-components.joblib", compress=3)
    data = base.loc[confirm]
    expected = predict_components(bundle, data)
    observed = component_values(data).to_numpy()
    baseline = models["hierarchical"].predict(data)
    scale = models["hierarchical"].scale
    error = ((observed - expected) / scale) ** 2
    previous_error = ((observed - baseline) / scale) ** 2
    result = {
        **frozen,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(data),
        "series": int(data.MatchID.nunique()),
        "normalized_mse": {
            "candidate": float(error.mean()),
            "baseline": float(previous_error.mean()),
        },
        "by_component": {
            c: {
                "candidate": float(error[:, i].mean()),
                "baseline": float(previous_error[:, i].mean()),
            }
            for i, c in enumerate(COMPONENTS)
        },
        "comparison": paired_mean_interval(
            data.MatchID, error.mean(axis=1) - previous_error.mean(axis=1)
        ),
        "negative_nonnegative_component_means": int((expected[:, 2:] < 0).sum()),
    }
    saved = data[["GameID", "PlayerID", "MatchID", "Agent", "Map", "Patch", "NumRounds"]].copy()
    for i, c in enumerate(COMPONENTS):
        saved[f"{c}_observed"] = observed[:, i]
        saved[f"{c}_expected"] = expected[:, i]
        saved[f"{c}_baseline"] = baseline[:, i]
    saved.to_parquet(args.output / "peer-confirmation.parquet", index=False)
    # Verify the bundle on all records and after changing every observed component/outcome.
    reloaded = joblib.load(args.output / "peer-components.joblib")
    np.testing.assert_array_equal(predict_components(reloaded, data), expected)
    mutated = data.copy()
    mutated[["KPR", "DPR", "FKPR", "FDPR", "ADR", "APR", "ObjectiveRate", "ClutchRate"]] = 1000
    mutated["TeamWon"] = 1 - mutated.TeamWon
    np.testing.assert_array_equal(predict_components(reloaded, mutated), expected)
    result["verification"] = {
        "reload_exact_match_rows": len(data),
        "context_only_inference": "passed",
        "selection_metrics_reproduced": "passed",
    }
    write_json(args.output / "results.json", result)
    write_json(
        args.output / "checksums.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.iterdir()
            if p.name != "checksums.json"
        },
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
