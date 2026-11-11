"""Fixed alternative-learner/calibration search; original test rows are excluded."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from ml.evaluation.metrics import metrics, slices
from ml.features.build import player_features
from ml.features.postgame import PostgameFeatures
from ml.improve import paired_interval, temperature, time_boundaries, write_json
from ml.models.player_research import ensemble_probability
from ml.player_research import assert_development, development_masks, inner_masks, temporal_folds
from ml.preprocessing.audit import load_data


def candidate_specs():
    recipes = {
        "compact": dict(
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=10,
            max_iter=1200,
            learning_rate=0.04,
        ),
        "flexible": dict(
            max_leaf_nodes=31,
            min_samples_leaf=120,
            l2_regularization=20,
            max_iter=1400,
            learning_rate=0.03,
        ),
        "regularized": dict(
            max_leaf_nodes=15,
            min_samples_leaf=250,
            l2_regularization=50,
            max_iter=1800,
            learning_rate=0.03,
        ),
    }
    return [
        {"name": f"{feature}__{name}", "features": feature, "parameters": params}
        for feature in ["expanded", "expanded_context"]
        for name, params in recipes.items()
    ]


def preprocessing(feature):
    return ColumnTransformer(
        [
            (
                "category",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                ),
                feature.cat,
            ),
            ("numeric", "passthrough", feature.num),
        ],
        remainder="drop",
    )


def estimator(spec, feature, iterations, stopping):
    params = {**spec["parameters"], "max_iter": iterations}
    return HistGradientBoostingClassifier(
        **params,
        categorical_features=list(range(len(feature.cat))),
        early_stopping=stopping,
        scoring="loss",
        n_iter_no_change=50,
        tol=1e-7,
        random_state=42,
    )


def fit_member(base, spec):
    train, stop = inner_masks(base)
    feature = PostgameFeatures(spec["features"]).fit(base.loc[train])
    x = feature.transform(base)
    pre = preprocessing(feature).fit(x.loc[train])
    probe = estimator(spec, feature, spec["parameters"]["max_iter"], True)
    probe.fit(
        pre.transform(x.loc[train]),
        base.loc[train, "TeamWon"],
        X_val=pre.transform(x.loc[stop]),
        y_val=base.loc[stop, "TeamWon"],
    )
    # Index zero is the unboosted model; refit only the best positive iteration count.
    iterations = int(np.argmax(probe.validation_score_[1:]) + 1)
    feature = PostgameFeatures(spec["features"]).fit(base)
    model = Pipeline(
        [("pre", preprocessing(feature)), ("model", estimator(spec, feature, iterations, False))]
    )
    model.fit(feature.transform(base), base.TeamWon)
    return {
        "features": feature,
        "model": model,
        "specification": spec,
        "iterations": iterations,
        "probe_iterations": int(probe.n_iter_),
        "inner_fit_max_date": str(base.loc[train, "Date"].max()),
        "inner_stop_min_date": str(base.loc[stop, "Date"].min()),
    }


def fit_calibration(kind, y, probability):
    p = np.asarray(probability)
    y = np.asarray(y)
    if (
        p.ndim != 1
        or not len(p)
        or len(p) != len(y)
        or not np.isfinite(p).all()
        or len(np.unique(y)) != 2
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Calibration needs aligned finite probabilities and both outcomes")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if kind == "identity":
        return {"kind": kind}
    if kind == "temperature":
        return {"kind": kind, "temperature": temperature(y, p)}
    if kind == "sigmoid":
        score = logit(p)
        result = minimize(
            lambda ab: log_loss(y, expit(ab[0] * score + ab[1]), labels=[0, 1]),
            np.array([1.0, 0.0]),
            bounds=[(0.1, 5.0), (-3.0, 3.0)],
            method="L-BFGS-B",
        )
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f"Sigmoid calibration failed: {result.message}")
        return {"kind": kind, "slope": float(result.x[0]), "intercept": float(result.x[1])}
    if kind == "isotonic":
        return {"kind": kind, "model": IsotonicRegression(out_of_bounds="clip").fit(p, y)}
    raise ValueError("Unknown calibration family")


def calibrated(calibration, probability):
    p = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    kind = calibration["kind"]
    if kind == "identity":
        return p
    if kind == "temperature":
        return expit(logit(p) / calibration["temperature"])
    if kind == "sigmoid":
        return expit(calibration["slope"] * logit(p) + calibration["intercept"])
    if kind == "isotonic":
        return np.clip(calibration["model"].predict(p), 1e-6, 1 - 1e-6)
    raise ValueError("Unknown calibration family")


def group_probability(group, members, base):
    weights = np.asarray([item["weight"] for item in group], dtype=float)
    if (
        not len(weights)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Mixture weights must be nonnegative and sum to one")
    return sum(
        item["weight"] * ensemble_probability(members[item["source"]], base) for item in group
    )


def selection_halves(base, selection):
    series = (
        base.loc[selection, ["MatchID", "Date"]].drop_duplicates().sort_values(["Date", "MatchID"])
    )
    boundary = series.iloc[len(series) // 2].Date
    left, right = selection & base.Date.lt(boundary), selection & base.Date.ge(boundary)
    if not left.any() or not right.any():
        raise ValueError("Insufficient selection timestamps")
    return left, right


def verify_parent(root):
    hashes = json.loads((root / "checksums.json").read_text())
    for name in [
        "experiment.json",
        "association-candidate.joblib",
        "association-confirmation.parquet",
    ]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("Prior artifact checksum mismatch")
    report = json.loads((root / "experiment.json").read_text())
    code_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path("ml").rglob("*.py")))
    ).hexdigest()
    if code_hash != report["code_hash"]:
        raise ValueError("Prior model source changed; use its recorded snapshot")
    return report, joblib.load(root / "association-candidate.joblib")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use a new empty experiment directory")
    parent, previous = verify_parent(args.parent)
    for name, expected in parent["source_hashes"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Source changed from parent experiment")
    frames = load_data(args.data_dir)
    frames = {
        n: f[f.Split.isin(["train", "validation"])].reset_index(drop=True)
        for n, f in frames.items()
    }
    games = frames["games"]
    reliable = games.loc[
        (games.Team1_Rounds != games.Team2_Rounds)
        & (games[["Team1_Rounds", "Team2_Rounds"]].max(axis=1) >= 13),
        "GameID",
    ]
    base = player_features(
        frames["players"].loc[frames["players"].GameID.isin(reliable)].reset_index(drop=True)
    )
    assert_development(base)
    masks = development_masks(base, time_boundaries(games))
    cal_select, choose = selection_halves(base, masks["selection"])
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parent": str(args.parent),
        "parent_experiment_hash": hashlib.sha256(
            (args.parent / "experiment.json").read_bytes()
        ).hexdigest(),
        "source_hashes": parent["source_hashes"],
        "ml_code_hash": parent["code_hash"],
        "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "specifications": candidate_specs(),
        "packages": parent["packages"],
        "partitions": {
            name: {
                "rows": int(mask.sum()),
                "series": int(base.loc[mask, "MatchID"].nunique()),
                "min_date": str(base.loc[mask, "Date"].min()),
                "max_date": str(base.loc[mask, "Date"].max()),
            }
            for name, mask in {
                **masks,
                "calibration_selection_fit": cal_select,
                "model_selection": choose,
            }.items()
        },
        "limitation": "Reused historical development validation; no independent test or causal impact claim.",
    }
    write_json(args.output / "protocol.json", manifest)
    (args.output / "script-snapshot.txt").write_bytes(Path(__file__).read_bytes())
    screen = {spec["name"]: {"specification": spec, "folds": []} for spec in candidate_specs()}
    for fold, (fit, evaluate) in enumerate(temporal_folds(base), 1):
        for spec in candidate_specs():
            member = fit_member(base.loc[fit], spec)
            p = ensemble_probability([member], base.loc[evaluate])
            result = metrics(base.loc[evaluate, "TeamWon"], p)
            screen[spec["name"]]["folds"].append(
                {
                    "log_loss": result["log_loss"],
                    "roc_auc": result["roc_auc"],
                    "iterations": member["iterations"],
                    "probe_iterations": member["probe_iterations"],
                    "fit_max_date": str(base.loc[fit, "Date"].max()),
                    "evaluation_min_date": str(base.loc[evaluate, "Date"].min()),
                    "inner_fit_max_date": member["inner_fit_max_date"],
                    "inner_stop_min_date": member["inner_stop_min_date"],
                }
            )
            print(
                f"HGB fold {fold} {spec['name']}: {member['iterations']} iterations, loss={result['log_loss']:.6f}",
                flush=True,
            )
            write_json(args.output / "screening.json", screen)
    for item in screen.values():
        item["mean_log_loss"] = float(np.mean([f["log_loss"] for f in item["folds"]]))
    finalists = sorted(screen, key=lambda n: screen[n]["mean_log_loss"])[:2]
    previous_group = previous["groups"][previous["selected"]]
    members = {"previous": previous_group["members"]}
    groups = {"previous": [{"source": "previous", "weight": 1.0}]}
    for name in finalists:
        members[name] = [fit_member(base.loc[masks["train"]], screen[name]["specification"])]
        groups[name] = [{"source": name, "weight": 1.0}]
        for weight in [0.25, 0.5]:
            groups[f"mix_{name}_{weight}"] = [
                {"source": name, "weight": weight},
                {"source": "previous", "weight": 1 - weight},
            ]
    selection = {}
    for name, group in groups.items():
        fit_p = group_probability(group, members, base.loc[cal_select])
        choose_p = group_probability(group, members, base.loc[choose])
        for family in ["identity", "temperature", "sigmoid", "isotonic"]:
            cal = fit_calibration(family, base.loc[cal_select, "TeamWon"], fit_p)
            result = metrics(base.loc[choose, "TeamWon"], calibrated(cal, choose_p))
            selection[f"{name}::{family}"] = result
            print(f"selection {name}::{family}: loss={result['log_loss']:.6f}", flush=True)
    selected = min(selection, key=lambda n: selection[n]["log_loss"])
    group_name, family = selected.rsplit("::", 1)
    frozen = {
        "selected": selected,
        "group": groups[group_name],
        "calibration_family": family,
        "selection": selection,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "finalist_iterations": {n: members[n][0]["iterations"] for n in finalists},
    }
    write_json(args.output / "selection-frozen.json", frozen)
    fit_p = group_probability(groups[group_name], members, base.loc[masks["calibration"]])
    calibration = fit_calibration(family, base.loc[masks["calibration"], "TeamWon"], fit_p)
    confirm = base.loc[masks["confirmation"]]
    p = calibrated(calibration, group_probability(groups[group_name], members, confirm))
    old_p = ensemble_probability(previous_group["members"], confirm, previous_group["temperature"])
    # Prove the baseline is the actual frozen artifact, aligned by observation keys.
    old = pd.read_parquet(args.parent / "association-confirmation.parquet").set_index(
        ["GameID", "PlayerID"]
    )
    saved_old = old.loc[
        pd.MultiIndex.from_frame(confirm[["GameID", "PlayerID"]]), previous["selected"]
    ].to_numpy()
    np.testing.assert_allclose(old_p, saved_old, atol=1e-6, rtol=0)
    saved = confirm[
        ["GameID", "PlayerID", "MatchID", "Date", "Agent", "Map", "Patch", "LengthBand", "TeamWon"]
    ].copy()
    saved["candidate"], saved["previous"] = p, old_p
    saved.to_parquet(args.output / "confirmation.parquet", index=False)
    bundle = {
        "research_only": True,
        "group": groups[group_name],
        "calibration": calibration,
        "members": {n: members[n] for n in {i["source"] for i in groups[group_name]}},
        "selected": selected,
    }
    joblib.dump(bundle, args.output / "candidate.joblib", compress=3)
    reloaded = joblib.load(args.output / "candidate.joblib")
    rp = calibrated(
        reloaded["calibration"], group_probability(reloaded["group"], reloaded["members"], confirm)
    )
    np.testing.assert_array_equal(rp, p)
    comparison = paired_interval(confirm.MatchID, confirm.TeamWon.to_numpy(), p, old_p)
    result = {
        **manifest,
        **frozen,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "screening": screen,
        "confirmation": {
            "candidate": metrics(confirm.TeamWon, p),
            "previous": metrics(confirm.TeamWon, old_p),
        },
        "slices": {
            "candidate": slices(
                confirm, confirm.TeamWon.to_numpy(), p, ["Map", "Agent", "Patch", "LengthBand"]
            ),
            "previous": slices(
                confirm, confirm.TeamWon.to_numpy(), old_p, ["Map", "Agent", "Patch", "LengthBand"]
            ),
        },
        "comparison": comparison,
        "reload_max_error": float(np.max(np.abs(rp - p))),
        "calibration_parameters": {k: v for k, v in calibration.items() if k != "model"},
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
    print(
        json.dumps(
            {
                "selected": selected,
                "comparison": comparison,
                "confirmation": result["confirmation"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
