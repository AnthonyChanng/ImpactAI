"""Continue postgame research with temporal early stopping and independent peer targets.

See docs/experiments/2026-10-16-plan.md. The original test split is never evaluated.
"""

import argparse
import hashlib
import importlib.metadata
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.evaluation.metrics import metrics, slices
from ml.features.build import COMPONENTS, player_features
from ml.features.postgame import PostgameFeatures
from ml.improve import paired_interval, temperature, time_boundaries, write_json
from ml.models.player_research import PeerMeanModel, ensemble_probability, peer_error
from ml.preprocessing.audit import FILES, load_data
from ml.train import pipeline

PREVIOUS = {
    "name": "previous_context",
    "features": "context",
    "early_stopping": False,
    "parameters": dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.025,
        min_child_weight=40,
        reg_lambda=10,
        subsample=0.85,
        colsample_bytree=0.9,
    ),
}
ORIGINAL = {
    "name": "original",
    "features": "base",
    "early_stopping": False,
    "parameters": dict(
        n_estimators=180, max_depth=3, learning_rate=0.05, min_child_weight=30, reg_lambda=5
    ),
}
PEER_SPECS = {
    "hierarchical": dict(kind="hierarchical"),
    "boosted_shallow": dict(
        kind="boosted",
        parameters=dict(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=100,
            reg_lambda=20,
            subsample=0.9,
            colsample_bytree=0.9,
        ),
    ),
    "boosted_medium": dict(
        kind="boosted",
        parameters=dict(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.025,
            min_child_weight=100,
            reg_lambda=30,
            subsample=0.9,
            colsample_bytree=0.9,
        ),
    ),
    "global": dict(kind="global"),
}


def candidate_specs():
    recipes = {
        "shallow": dict(
            max_depth=3, n_estimators=2400, learning_rate=0.025, min_child_weight=30, reg_lambda=10
        ),
        "medium": dict(
            max_depth=5, n_estimators=2000, learning_rate=0.02, min_child_weight=40, reg_lambda=15
        ),
        "deep": dict(
            max_depth=7, n_estimators=1400, learning_rate=0.025, min_child_weight=50, reg_lambda=20
        ),
        "regularized": dict(
            max_depth=6,
            n_estimators=2000,
            learning_rate=0.02,
            min_child_weight=120,
            reg_lambda=50,
            reg_alpha=1,
        ),
    }
    return [
        {
            "name": f"{feature}__{name}",
            "features": feature,
            "early_stopping": True,
            "parameters": dict(**params, subsample=0.85, colsample_bytree=0.9),
        }
        for feature in ["context", "expanded", "expanded_context"]
        for name, params in recipes.items()
    ]


def assert_development(raw):
    if not raw.Split.isin(["train", "validation"]).all():
        raise ValueError("Original test rows are forbidden")
    series = raw.groupby("MatchID").agg(splits=("Split", "nunique"), dates=("Date", "nunique"))
    if (series != 1).any().any():
        raise ValueError("Each series must have one partition and timestamp")
    if (
        raw.loc[raw.Split.eq("train"), "Date"].max()
        >= raw.loc[raw.Split.eq("validation"), "Date"].min()
    ):
        raise ValueError("Training must strictly precede validation")


def temporal_folds(raw):
    assert_development(raw)
    series = raw.loc[raw.Split.eq("train"), ["MatchID", "Date"]].drop_duplicates()
    series = series.sort_values(["Date", "MatchID"])
    cuts = [series.iloc[int(len(series) * fraction)].Date for fraction in [0.5, 0.67, 0.83]]
    folds = []
    for i, start in enumerate(cuts):
        fit = raw.Split.eq("train") & raw.Date.lt(start)
        evaluate = raw.Split.eq("train") & raw.Date.ge(start)
        if i < 2:
            evaluate &= raw.Date.lt(cuts[i + 1])
        if not fit.any() or not evaluate.any():
            raise ValueError("Insufficient timestamps for temporal folds")
        folds.append((fit, evaluate))
    return folds


def inner_masks(base):
    if not base.Split.eq("train").all():
        raise ValueError("Model fitting requires original training rows")
    series = base[["MatchID", "Date"]].drop_duplicates().sort_values(["Date", "MatchID"])
    boundary = series.iloc[int(len(series) * 0.85)].Date
    train, stop = base.Date.lt(boundary), base.Date.ge(boundary)
    if not train.any() or not stop.any():
        raise ValueError("Insufficient timestamps for early stopping")
    return train, stop


def fit_member(base, spec, seed=42, fixed_trees=None):
    train, stop = inner_masks(base)
    trees = fixed_trees or spec["parameters"]["n_estimators"]
    early = None
    if spec["early_stopping"] and fixed_trees is None:
        feature = PostgameFeatures(spec["features"]).fit(base.loc[train])
        x = feature.transform(base)
        probe = pipeline(feature.cat, feature.num, "xgboost")
        probe.set_params(**{f"model__{k}": v for k, v in spec["parameters"].items()})
        pre = probe["pre"].fit(x.loc[train])
        model = probe["model"].set_params(early_stopping_rounds=60, random_state=seed)
        model.fit(
            pre.transform(x.loc[train]),
            base.loc[train, "TeamWon"],
            eval_set=[(pre.transform(x.loc[stop]), base.loc[stop, "TeamWon"])],
            verbose=False,
        )
        trees = int(model.best_iteration + 1)
        early = {
            "trees": trees,
            "cap": spec["parameters"]["n_estimators"],
            "fitted_trees": int(model.get_booster().num_boosted_rounds()),
            "fit_max_date": str(base.loc[train, "Date"].max()),
            "stop_min_date": str(base.loc[stop, "Date"].min()),
            "stop_series": int(base.loc[stop, "MatchID"].nunique()),
        }
    feature = PostgameFeatures(spec["features"]).fit(base)
    model = pipeline(feature.cat, feature.num, "xgboost")
    model.set_params(**{f"model__{k}": v for k, v in spec["parameters"].items()})
    model.set_params(model__n_estimators=trees, model__random_state=seed)
    model.fit(feature.transform(base), base.TeamWon)
    return {
        "model": model,
        "features": feature,
        "specification": spec,
        "trees": trees,
        "seed": seed,
        "early_stopping": early,
    }


def compact(result):
    return {key: value for key, value in result.items() if key != "calibration"}


def development_masks(raw, boundaries):
    validation = raw.Split.eq("validation")
    return {
        "train": raw.Split.eq("train"),
        "selection": validation & raw.Date.lt(boundaries["select_end"]),
        "calibration": validation
        & raw.Date.ge(boundaries["select_end"])
        & raw.Date.lt(boundaries["calibrate_end"]),
        "confirmation": validation & raw.Date.ge(boundaries["calibrate_end"]),
    }


def run_association(base, masks, output):
    specs = candidate_specs() + [PREVIOUS]
    screening = {s["name"]: {"specification": s, "folds": []} for s in specs}
    for fold, (fit, evaluate) in enumerate(temporal_folds(base), 1):
        for spec in specs:
            member = fit_member(base.loc[fit], spec)
            p = ensemble_probability([member], base.loc[evaluate])
            result = metrics(base.loc[evaluate, "TeamWon"], p)
            screening[spec["name"]]["folds"].append(
                {
                    "metrics": compact(result),
                    "trees": member["trees"],
                    "early_stopping": member["early_stopping"],
                    "fit_rows": int(fit.sum()),
                    "evaluation_rows": int(evaluate.sum()),
                    "fit_max_date": str(base.loc[fit, "Date"].max()),
                    "evaluation_min_date": str(base.loc[evaluate, "Date"].min()),
                }
            )
            print(
                f"player fold {fold} {spec['name']}: trees={member['trees']} "
                f"loss={result['log_loss']:.6f}",
                flush=True,
            )
            write_json(output / "association-screening.json", screening)
    for item in screening.values():
        item["mean_fold_log_loss"] = float(
            np.mean([f["metrics"]["log_loss"] for f in item["folds"]])
        )
    ranked = sorted(
        [s["name"] for s in candidate_specs()],
        key=lambda name: screening[name]["mean_fold_log_loss"],
    )
    finalists = ranked[:3] + [PREVIOUS["name"]]
    members, groups, selection = {}, {}, {}
    for name in finalists:
        members[name] = fit_member(base.loc[masks["train"]], screening[name]["specification"])
        groups[name] = [name]
        p = ensemble_probability([members[name]], base.loc[masks["selection"]])
        selection[name] = metrics(base.loc[masks["selection"], "TeamWon"], p)
        print(f"player selection {name}: loss={selection[name]['log_loss']:.6f}", flush=True)
    best_single = min(selection, key=lambda name: selection[name]["log_loss"])
    seed_members = [best_single]
    for seed in [17, 89]:
        name = f"{best_single}__seed{seed}"
        members[name] = fit_member(
            base.loc[masks["train"]],
            members[best_single]["specification"],
            seed,
            fixed_trees=members[best_single]["trees"],
        )
        seed_members.append(name)
    groups["seed_average"] = seed_members
    groups["diverse_average"] = ranked[:3]
    for name in ["seed_average", "diverse_average"]:
        p = ensemble_probability([members[n] for n in groups[name]], base.loc[masks["selection"]])
        selection[name] = metrics(base.loc[masks["selection"], "TeamWon"], p)
        print(f"player selection {name}: loss={selection[name]['log_loss']:.6f}", flush=True)
    selected = min(selection, key=lambda name: selection[name]["log_loss"])
    frozen = {
        "selected": selected,
        "groups": groups,
        "selection": selection,
        "member_trees": {n: m["trees"] for n, m in members.items()},
        "member_early_stopping": {n: m["early_stopping"] for n, m in members.items()},
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "association-selection-frozen.json", frozen)
    members["original"] = fit_member(base.loc[masks["train"]], ORIGINAL)
    groups["original"] = ["original"]
    confirm = base.loc[masks["confirmation"]]
    calibration = base.loc[masks["calibration"]]
    predictions = confirm[
        ["GameID", "PlayerID", "MatchID", "Date", "Map", "Agent", "Patch", "LengthBand", "TeamWon"]
    ].copy()
    known_players = set(base.loc[masks["train"], "PlayerID"])
    predictions["PlayerCoverage"] = np.where(
        confirm.PlayerID.isin(known_players), "seen_in_training", "unseen_in_training"
    )
    evaluation, temperatures, saved_groups = {}, {}, {}
    for name in dict.fromkeys([selected, PREVIOUS["name"], "original"]):
        group = [members[n] for n in groups[name]]
        t = temperature(calibration.TeamWon, ensemble_probability(group, calibration))
        p = ensemble_probability(group, confirm, t)
        predictions[name] = p
        evaluation[name] = {
            "overall": metrics(confirm.TeamWon, p),
            "slices": slices(
                predictions,
                confirm.TeamWon.to_numpy(),
                p,
                ["Map", "Agent", "Patch", "LengthBand", "PlayerCoverage"],
            ),
        }
        temperatures[name] = t
        saved_groups[name] = {"members": group, "temperature": t}
    comparison = paired_interval(
        confirm.MatchID,
        confirm.TeamWon.to_numpy(),
        predictions[selected].to_numpy(),
        predictions[PREVIOUS["name"]].to_numpy(),
    )
    predictions.to_parquet(output / "association-confirmation.parquet", index=False)
    joblib.dump(
        {"research_only": True, "selected": selected, "groups": saved_groups},
        output / "association-candidate.joblib",
        compress=3,
    )
    result = {
        **frozen,
        "screening": screening,
        "temperature": temperatures,
        "confirmation": evaluation,
        "comparison_to_previous": comparison,
    }
    write_json(output / "association-results.json", result)
    return result


def paired_mean_interval(series, differences):
    grouped = (
        pd.DataFrame({"series": np.asarray(series), "difference": differences})
        .groupby("series")
        .difference.agg(["sum", "count"])
        .to_numpy()
    )
    rng = np.random.default_rng(42)
    draws = grouped[rng.integers(0, len(grouped), (1000, len(grouped)))]
    means = draws[:, :, 0].sum(axis=1) / draws[:, :, 1].sum(axis=1)
    return {
        "difference": float(grouped[:, 0].sum() / grouped[:, 1].sum()),
        "series_bootstrap_95pct": np.quantile(means, [0.025, 0.975]).tolist(),
        "series": len(grouped),
    }


def run_peers(base, masks, output):
    screening = {name: [] for name in PEER_SPECS}
    for fold, (fit, evaluate) in enumerate(temporal_folds(base), 1):
        for name, spec in PEER_SPECS.items():
            model = PeerMeanModel(**spec).fit(base.loc[fit])
            errors, _ = peer_error(model, base.loc[evaluate])
            screening[name].append(dict(zip(COMPONENTS, errors.tolist())))
            print(f"peers fold {fold} {name}: normalized_mse={errors.mean():.6f}", flush=True)
        write_json(output / "peer-screening.json", screening)
    models, selection = {}, {}
    for name, spec in PEER_SPECS.items():
        models[name] = PeerMeanModel(**spec).fit(base.loc[masks["train"]])
        errors, _ = peer_error(models[name], base.loc[masks["selection"]])
        selection[name] = {
            "normalized_mse": float(errors.mean()),
            "by_component": dict(zip(COMPONENTS, errors.tolist())),
        }
    # Global is retained as a diagnostic, not selected as a contextual model.
    selected = min(
        [n for n in selection if n != "global"], key=lambda name: selection[name]["normalized_mse"]
    )
    frozen = {
        "selected": selected,
        "selection": selection,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "peer-selection-frozen.json", frozen)
    confirm = base.loc[masks["confirmation"]]
    evaluation, row_errors = {}, {}
    predictions = confirm[["GameID", "PlayerID", "MatchID"]].copy()
    for name in dict.fromkeys([selected, "hierarchical", "global"]):
        errors, row_errors[name] = peer_error(models[name], confirm)
        predicted = models[name].predict(confirm)
        for i, component in enumerate(COMPONENTS):
            predictions[f"{name}_{component}"] = predicted[:, i]
        evaluation[name] = {
            "normalized_mse": float(errors.mean()),
            "by_component": dict(zip(COMPONENTS, errors.tolist())),
        }
    result = {
        **frozen,
        "screening": screening,
        "confirmation": evaluation,
        "comparison_to_hierarchical": paired_mean_interval(
            confirm.MatchID, row_errors[selected] - row_errors["hierarchical"]
        ),
    }
    joblib.dump(
        {"research_only": True, "selected": selected, "models": {n: models[n] for n in evaluation}},
        output / "peer-candidate.joblib",
        compress=3,
    )
    predictions.to_parquet(output / "peer-confirmation.parquet", index=False)
    write_json(output / "peer-results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory for an immutable experiment")
    source = load_data(args.data_dir)
    excluded = {name: int(frame.Split.eq("test").sum()) for name, frame in source.items()}
    frames = {
        name: frame[frame.Split.isin(["train", "validation"])].reset_index(drop=True)
        for name, frame in source.items()
    }
    del source
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
    boundaries = time_boundaries(games)
    masks = development_masks(base, boundaries)
    sources = sorted(Path("ml").rglob("*.py"))
    with zipfile.ZipFile(args.output / "ml-source.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
        for path in sources:
            snapshot.write(path, str(path))
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
            for name in FILES.values()
        },
        "code_hash": hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]
        },
        "original_test_rows_excluded": excluded,
        "partitions": {
            name: {
                "rows": int(mask.sum()),
                "series": int(base.loc[mask, "MatchID"].nunique()),
                "min_date": str(base.loc[mask, "Date"].min()),
                "max_date": str(base.loc[mask, "Date"].max()),
            }
            for name, mask in masks.items()
        },
        "candidate_specs": candidate_specs(),
        "peer_specs": PEER_SPECS,
        "limitation": "Reused development validation, not an independent new test. "
        "Win association and peer means do not establish causal player impact.",
    }
    write_json(args.output / "protocol.json", manifest)
    association = run_association(base, masks, args.output)
    peers = run_peers(base, masks, args.output)
    result = {
        **manifest,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "association": association,
        "peers": peers,
    }
    write_json(args.output / "experiment.json", result)
    write_json(
        args.output / "checksums.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.iterdir()
            if p.is_file() and p.name != "checksums.json"
        },
    )
    print(
        json.dumps(
            {
                "association": association["selected"],
                "association_change": association["comparison_to_previous"],
                "peer_model": peers["selected"],
                "peer_change": peers["comparison_to_hierarchical"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
