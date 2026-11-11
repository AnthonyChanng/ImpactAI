"""Bounded development-only search. Never evaluates or trains on the original test split.

Protocol is declared in docs/experiments/2026-10-15-plan.md. Research artifacts are intentionally
separate from the application's v1 contract and cannot silently replace its active model.
"""

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from ml.evaluation.metrics import metrics, slices
from ml.features.build import (
    COMPONENTS,
    PLAYER_CAT,
    PLAYER_NUM,
    ROUND_CAT,
    ROUND_NUM,
    player_features,
    swap_round,
)
from ml.features.history import RECENT_FEATURES, TEAM_FEATURES, TeamHistory, recent_round_features
from ml.models.context import ContextIndex
from ml.models.inference import probabilities
from ml.preprocessing.audit import FILES, load_data
from ml.train import pipeline


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def time_boundaries(games):
    """Boundaries operate on whole timestamps, so series and simultaneous matches stay together."""
    series = games[["MatchID", "Date", "Split"]].drop_duplicates().sort_values(["Date", "MatchID"])
    train = series[series.Split.eq("train")]
    validation = series[series.Split.eq("validation")]
    if len(train) < 10 or len(validation) < 10:
        raise ValueError("Too few development series for the declared temporal protocol")
    select_end = validation.iloc[len(validation) // 2].Date
    remaining = validation[validation.Date.ge(select_end)]
    return {
        "fold1": train.iloc[int(len(train) * 0.6)].Date,
        "fold2": train.iloc[int(len(train) * 0.8)].Date,
        "select_end": select_end,
        "calibrate_end": remaining.iloc[len(remaining) // 2].Date,
    }


def candidates(task):
    recipes = [
        (
            "reference",
            dict(
                n_estimators=180,
                max_depth=5 if task == "round" else 3,
                learning_rate=0.05,
                min_child_weight=30,
                reg_lambda=5,
            ),
        ),
        (
            "long_shallow",
            dict(
                n_estimators=600,
                max_depth=3,
                learning_rate=0.035,
                min_child_weight=30,
                reg_lambda=10,
            ),
        ),
        (
            "balanced",
            dict(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                min_child_weight=50,
                reg_lambda=10,
            ),
        ),
        (
            "flexible",
            dict(
                n_estimators=400,
                max_depth=7,
                learning_rate=0.04,
                min_child_weight=50,
                reg_lambda=10,
            ),
        ),
        (
            "regularized",
            dict(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.035,
                min_child_weight=100,
                reg_lambda=25,
            ),
        ),
        (
            "subsampled",
            dict(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.025,
                min_child_weight=40,
                reg_lambda=10,
                subsample=0.85,
                colsample_bytree=0.9,
            ),
        ),
    ]
    features = (
        ["base", "recent", "ratings", "recent_ratings"]
        if task == "round"
        else ["base", "context", "acs_multikill"]
    )
    return [
        {"name": f"{feature}__{name}", "features": feature, "parameters": params}
        for feature in features
        for name, params in recipes
    ]


def design(task, raw, games, training, feature_name, base):
    if not raw.Split.isin(["train", "validation"]).all():
        raise ValueError("Original test rows are forbidden in model improvement")
    if not raw.loc[training, "Split"].eq("train").all():
        raise ValueError("Estimator fitting must use original training rows only")
    state = {}
    if task == "round":
        cat, num = ROUND_CAT, ROUND_NUM.copy()
        x = base.copy()
        if "recent" in feature_name:
            num += RECENT_FEATURES
        if "ratings" in feature_name:
            train_games = games[games.GameID.isin(raw.loc[training, "GameID"])]
            other_games = games[~games.GameID.isin(train_games.GameID)]
            history = TeamHistory()
            lookup = pd.concat([history.fit_transform(train_games), history.transform(other_games)])
            added = lookup.reindex(raw.GameID).set_axis(raw.index)
            x = pd.concat([x, added], axis=1)
            num += TEAM_FEATURES
            state["team_history"] = history
    else:
        cat, num = PLAYER_CAT, PLAYER_NUM.copy()
        x = base.copy()
        if feature_name == "context":
            context = ContextIndex().fit(base.loc[training])
            z, _, _ = context.standardize(base)
            extra = [f"ContextZ_{name}" for name in COMPONENTS]
            x[extra] = z
            num += extra
            state["context_reference"] = context
        elif feature_name == "acs_multikill":
            num += ["ACS", "MultiKillRate"]
    return x[cat + num], cat, num, state


def fit_model(task, x, y, training, cat, num, spec):
    tx, ty = x.loc[training], y.loc[training]
    if task == "round":
        tx = pd.concat([tx, swap_round(tx)], ignore_index=True)
        ty = pd.concat([ty, 1 - ty], ignore_index=True)
    model = pipeline(cat, num, "xgboost")
    model.set_params(**{f"model__{key}": value for key, value in spec["parameters"].items()})
    return model.fit(tx, ty)


def temperature(y, p):
    logits = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return float(
        minimize_scalar(
            lambda t: log_loss(y, expit(logits / t), labels=[0, 1]),
            bounds=(0.5, 3),
            method="bounded",
        ).x
    )


def paired_interval(series, y, candidate, baseline):
    def losses(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))

    groups = (
        pd.DataFrame(
            {"series": series.to_numpy(), "difference": losses(candidate) - losses(baseline)}
        )
        .groupby("series")
        .difference.agg(["sum", "count"])
        .to_numpy()
    )
    rng = np.random.default_rng(42)
    sampled = groups[rng.integers(0, len(groups), (1000, len(groups)))]
    means = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return {
        "candidate_minus_baseline": float(groups[:, 0].sum() / groups[:, 1].sum()),
        "series_bootstrap_95pct": np.quantile(means, [0.025, 0.975]).tolist(),
        "series": len(groups),
        "interpretation": "Negative favors candidate; diagnostic development comparison, not a fresh test.",
    }


def run_task(task, frames, boundaries, output):
    games = frames["games"]
    raw = frames["rounds" if task == "round" else "players"].copy()
    if task == "player":
        reliable = games.loc[
            (games.Team1_Rounds != games.Team2_Rounds)
            & (games[["Team1_Rounds", "Team2_Rounds"]].max(axis=1) >= 13),
            "GameID",
        ]
        raw = raw[raw.GameID.isin(reliable)].reset_index(drop=True)
    y = raw.Team1_RoundWon if task == "round" else raw.TeamWon
    base = recent_round_features(raw) if task == "round" else player_features(raw)
    specs = candidates(task)
    screen = {s["name"]: {"specification": s, "folds": []} for s in specs}
    for fold in (1, 2):
        start = boundaries[f"fold{fold}"]
        fit = raw.Split.eq("train") & raw.Date.lt(start)
        evaluate = raw.Split.eq("train") & raw.Date.ge(start)
        if fold == 1:
            evaluate &= raw.Date.lt(boundaries["fold2"])
        cache = {}
        for spec in specs:
            feature = spec["features"]
            if feature not in cache:
                cache[feature] = design(task, raw, games, fit, feature, base)
            x, cat, num, _ = cache[feature]
            model = fit_model(task, x, y, fit, cat, num, spec)
            p = probabilities(model, x.loc[evaluate], task == "round")
            result = metrics(y.loc[evaluate], p)
            screen[spec["name"]]["folds"].append(
                {
                    "metrics": result,
                    "fit_rows": int(fit.sum()),
                    "evaluation_series": int(raw.loc[evaluate, "MatchID"].nunique()),
                }
            )
            print(
                f"{task} fold {fold} {spec['name']}: log_loss={result['log_loss']:.6f} auc={result['roc_auc']:.6f}",
                flush=True,
            )
            write_json(output / f"{task}-screening.json", screen)
    for item in screen.values():
        item["mean_fold_log_loss"] = float(
            np.mean([f["metrics"]["log_loss"] for f in item["folds"]])
        )
    ranked = sorted(screen, key=lambda name: screen[name]["mean_fold_log_loss"])
    finalists = ranked[:4]
    baseline_name = "base__reference"
    if baseline_name not in finalists:
        finalists.append(baseline_name)
    fit = raw.Split.eq("train")
    validation = raw.Split.eq("validation")
    select = validation & raw.Date.lt(boundaries["select_end"])
    calibrate = (
        validation
        & raw.Date.ge(boundaries["select_end"])
        & raw.Date.lt(boundaries["calibrate_end"])
    )
    confirm = validation & raw.Date.ge(boundaries["calibrate_end"])
    models, selection, cache = {}, {}, {}
    for name in finalists:
        spec = screen[name]["specification"]
        feature = spec["features"]
        if feature not in cache:
            cache[feature] = design(task, raw, games, fit, feature, base)
        x, cat, num, _ = cache[feature]
        model = fit_model(task, x, y, fit, cat, num, spec)
        p = probabilities(model, x.loc[select], task == "round")
        selection[name] = metrics(y.loc[select], p)
        models[name] = model
        print(f"{task} selection {name}: log_loss={selection[name]['log_loss']:.6f}", flush=True)
    selected = min(selection, key=lambda name: selection[name]["log_loss"])
    write_json(
        output / f"{task}-selection-frozen.json",
        {
            "selected": selected,
            "selection": selection,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    # The diagnostic confirmation segment is accessed only after candidate selection freezes.
    evaluation, saved, probabilities_by_name, calibration = {}, {}, {}, {}
    for name in dict.fromkeys([selected, baseline_name]):
        spec = screen[name]["specification"]
        x, cat, num, state = cache[spec["features"]]
        model = models[name]
        cp = probabilities(model, x.loc[calibrate], task == "round")
        t = temperature(y.loc[calibrate], cp)
        p = probabilities(model, x.loc[confirm], task == "round", t)
        probabilities_by_name[name] = p
        evaluation[name] = {
            "overall": metrics(y.loc[confirm], p),
            "by_map": slices(raw.loc[confirm], y.loc[confirm].to_numpy(), p, ["Map"]),
        }
        calibration[name] = {
            "temperature": t,
            "fit_rows": int(calibrate.sum()),
            "fit_series": int(raw.loc[calibrate, "MatchID"].nunique()),
        }
        saved[name] = {
            "model": model,
            "temperature": t,
            "features": cat + num,
            "feature_set": spec["features"],
            "state": state,
        }
        if task == "round":
            a = probabilities(model, swap_round(x.loc[confirm].iloc[:1000]), True, t)
            np.testing.assert_allclose(a + p[:1000], 1, atol=1e-6)
    comparison = paired_interval(
        raw.loc[confirm, "MatchID"],
        y.loc[confirm].to_numpy(),
        probabilities_by_name[selected],
        probabilities_by_name[baseline_name],
    )
    prediction_frame = raw.loc[confirm, ["GameID", "MatchID", "Date", "Map"]].copy()
    prediction_frame["RoundNumber" if task == "round" else "PlayerID"] = raw.loc[
        confirm, "RoundNumber" if task == "round" else "PlayerID"
    ]
    prediction_frame["observed"] = y.loc[confirm]
    prediction_frame["candidate"] = probabilities_by_name[selected]
    prediction_frame["baseline"] = probabilities_by_name[baseline_name]
    prediction_frame.to_parquet(output / f"{task}-confirmation.parquet", index=False)
    joblib.dump(
        {
            "task": task,
            "selected": selected,
            "baseline": baseline_name,
            "models": saved,
            "research_only": True,
        },
        output / f"{task}-candidate.joblib",
        compress=3,
    )
    result = {
        "selected": selected,
        "baseline": baseline_name,
        "screening": screen,
        "selection": selection,
        "calibration": calibration,
        "confirmation": evaluation,
        "comparison": comparison,
        "partition_rows": {
            name: int(mask.sum())
            for name, mask in [
                ("train", fit),
                ("selection", select),
                ("calibration", calibrate),
                ("confirmation", confirm),
            ]
        },
        "partition_series": {
            name: int(raw.loc[mask, "MatchID"].nunique())
            for name, mask in [
                ("train", fit),
                ("selection", select),
                ("calibration", calibrate),
                ("confirmation", confirm),
            ]
        },
    }
    write_json(output / f"{task}-results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=["round", "player", "both"], default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "experiment.json").exists():
        raise ValueError("Use a new output directory; completed experiment records are immutable")
    frames = load_data(args.data_dir)
    excluded = {name: int(frame.Split.eq("test").sum()) for name, frame in frames.items()}
    frames = {
        name: frame[frame.Split.isin(["train", "validation"])].reset_index(drop=True)
        for name, frame in frames.items()
    }
    boundaries = time_boundaries(frames["games"])
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "original_test_rows_excluded": excluded,
        "source_hashes": {
            name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
            for name in FILES.values()
        },
        "code_hash": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path("ml").rglob("*.py")))
        ).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "joblib"]
        },
        "time_boundaries": {name: str(value) for name, value in boundaries.items()},
        "limitation": "All confirmation rows belong to previously inspected historical validation. No new independent test claim.",
    }
    write_json(args.output / "protocol.json", manifest)
    tasks = ["round", "player"] if args.task == "both" else [args.task]
    results = {task: run_task(task, frames, boundaries, args.output) for task in tasks}
    write_json(
        args.output / "experiment.json",
        {**manifest, "completed_at": datetime.now(timezone.utc).isoformat(), "results": results},
    )
    write_json(
        args.output / "checksums.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.output.iterdir()
            if p.is_file() and p.name not in ["checksums.json", "training.log"]
        },
    )
    print(
        json.dumps(
            {
                task: {
                    "selected": r["selected"],
                    "comparison": r["comparison"],
                    "confirmation": {
                        name: value["overall"] for name, value in r["confirmation"].items()
                    },
                }
                for task, r in results.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
