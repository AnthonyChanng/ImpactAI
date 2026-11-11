"""Fixed context-only peer quantile experiment; original test rows are never predicted.

Intervals describe historical individual component distributions, not causal impact.
Calibration is empirical: dependent series and chronological drift preclude an IID
coverage guarantee. Run with bounded OpenMP threads in the environment.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from ml.features.build import COMPONENTS, PLAYER_CAT, component_values, player_features
from ml.improve import time_boundaries, write_json
from ml.player_research import (
    assert_development,
    development_masks,
    paired_mean_interval,
    temporal_folds,
)
from ml.preprocessing.audit import FILES, load_data

QUANTILES = np.array([0.1, 0.5, 0.9])
CONTEXT_COLUMNS = PLAYER_CAT + ["NumRounds"]
RECIPE = dict(
    max_iter=300,
    max_leaf_nodes=15,
    min_samples_leaf=100,
    learning_rate=0.05,
    early_stopping=False,
    random_state=42,
)
MIN_PEERS = 100
NOMINAL_COVERAGE = 0.8
LIMITATION = (
    "Reused historical development validation, not an independent test. "
    "Calibration treats player rows empirically; series dependence and temporal drift "
    "preclude an IID coverage guarantee. Component intervals do not validate causal "
    "impact, coaching advice, or a composite player score."
)


def feature_matrix(base):
    """Select context before transformation; observations and identities are ignored."""
    x = base[CONTEXT_COLUMNS].copy()
    for column in PLAYER_CAT:
        x[column] = x[column].astype(object).fillna("unknown")
    rounds = x.NumRounds.to_numpy(dtype=float)
    if not np.isfinite(rounds).all() or (rounds <= 0).any():
        raise ValueError("Context round counts must be finite and positive")
    return x


def training_targets(base):
    if not len(base) or not base.Split.eq("train").all():
        raise ValueError("Fitting requires original training rows only")
    y = component_values(base).to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y[:, 2:] < 0).any():
        raise ValueError("Component targets must be finite and respect nonnegative support")
    return y


def ordered_supported(predictions):
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (len(COMPONENTS), len(QUANTILES)):
        raise ValueError("Expected rows by six components by three quantiles")
    if not np.isfinite(p).all():
        raise ValueError("Quantile predictions must be finite")
    p = np.sort(p, axis=2)
    p[:, 2:, :] = np.maximum(p[:, 2:, :], 0)
    return p


def fit_empirical(base, min_peers=MIN_PEERS):
    y = training_targets(base)
    x = feature_matrix(base).reset_index(drop=True)
    if not isinstance(min_peers, int) or min_peers < 1:
        raise ValueError("Minimum peer count must be a positive integer")
    bundle = {
        "global": np.quantile(y, QUANTILES, axis=0).T,
        "agent": {},
        "agent_map": {},
        "min_peers": min_peers,
    }
    for columns, name in [(["Agent"], "agent"), (["Agent", "Map"], "agent_map")]:
        for key, rows in x.groupby(columns, observed=True, sort=True).indices.items():
            if len(rows) >= min_peers:
                key = key if isinstance(key, tuple) else (key,)
                bundle[name][key] = np.quantile(y[rows], QUANTILES, axis=0).T
    return bundle


def predict_empirical(bundle, base):
    x = feature_matrix(base)
    values, levels = [], []
    for agent, map_name in x[["Agent", "Map"]].itertuples(index=False, name=None):
        if (agent, map_name) in bundle["agent_map"]:
            value, level = bundle["agent_map"][(agent, map_name)], "agent_map"
        elif (agent,) in bundle["agent"]:
            value, level = bundle["agent"][(agent,)], "agent"
        else:
            value, level = bundle["global"], "global"
        values.append(value)
        levels.append(level)
    return ordered_supported(np.asarray(values)), levels


def fit_learned(base, recipe=None):
    y = training_targets(base)
    pre = ColumnTransformer(
        [
            (
                "category",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                    encoded_missing_value=np.nan,
                ),
                PLAYER_CAT,
            ),
            ("numeric", "passthrough", ["NumRounds"]),
        ],
        remainder="drop",
    )
    x = pre.fit_transform(feature_matrix(base))
    params = RECIPE if recipe is None else recipe
    if params.get("early_stopping") is not False:
        raise ValueError("Random early stopping is forbidden in this fixed temporal experiment")
    models = []
    for component in range(len(COMPONENTS)):
        models.append(
            [
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=float(quantile),
                    categorical_features=list(range(len(PLAYER_CAT))),
                    **params,
                ).fit(x, y[:, component])
                for quantile in QUANTILES
            ]
        )
    return {"pre": pre, "models": models, "recipe": params, "columns": CONTEXT_COLUMNS}


def predict_learned(bundle, base):
    x = bundle["pre"].transform(feature_matrix(base))
    predictions = np.stack(
        [np.column_stack([model.predict(x) for model in models]) for models in bundle["models"]],
        axis=1,
    )
    return ordered_supported(predictions)


def predict_selected(bundle, base):
    empirical, _ = predict_empirical(bundle["empirical"], base)
    learned = predict_learned(bundle["learned"], base)
    return np.stack(
        [
            (learned if bundle["component_sources"][name] == "learned" else empirical)[:, i, :]
            for i, name in enumerate(COMPONENTS)
        ],
        axis=1,
    )


def metric_arrays(observed, prediction, scale):
    y, p, scale = np.asarray(observed), np.asarray(prediction), np.asarray(scale)
    if y.shape != p.shape[:2] or p.shape[1:] != (6, 3) or scale.shape != (6,):
        raise ValueError("Observed, prediction, and scale shapes must align")
    if (
        not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
        or (np.diff(p, axis=2) < 0).any()
    ):
        raise ValueError("Metrics need nonempty finite observations and ordered quantiles")
    residual = y[:, :, None] - p
    pinball = np.maximum(QUANTILES * residual, (QUANTILES - 1) * residual)
    low, high = p[:, :, 0], p[:, :, 2]
    below, above = y < low, y > high
    width = high - low
    interval = width + 10 * np.maximum(low - y, 0) + 10 * np.maximum(y - high, 0)
    return {
        "pinball": pinball,
        "normalized_pinball": pinball / scale[None, :, None],
        "coverage": ~(below | above),
        "lower_tail": below,
        "upper_tail": above,
        "width": width,
        "normalized_width": width / scale,
        "interval_score": interval,
        "normalized_interval_score": interval / scale,
    }


def summarize(observed, prediction, scale):
    values = metric_arrays(observed, prediction, scale)
    by_component = {}
    for i, name in enumerate(COMPONENTS):
        row = {key: float(value[:, i].mean()) for key, value in values.items()}
        row["pinball_by_quantile"] = {
            str(q): float(values["pinball"][:, i, j].mean()) for j, q in enumerate(QUANTILES)
        }
        by_component[name] = row
    return {
        "rows": len(observed),
        "normalized_pinball": float(values["normalized_pinball"].mean()),
        "normalized_interval_score": float(values["normalized_interval_score"].mean()),
        "mean_component_coverage": float(values["coverage"].mean()),
        "by_component": by_component,
    }


def select_components(screening, selection):
    if len(screening) != 3:
        raise ValueError("Exactly three temporal screening folds are required")
    return {
        component: (
            "learned"
            if all(
                result["learned"]["by_component"][component]["normalized_pinball"]
                < result["empirical"]["by_component"][component]["normalized_pinball"]
                for result in [*screening, selection]
            )
            else "empirical"
        )
        for component in COMPONENTS
    }


def fit_expansion(observed, prediction, coverage=NOMINAL_COVERAGE):
    y, p = np.asarray(observed), np.asarray(prediction)
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one")
    metric_arrays(y, p, np.ones(len(COMPONENTS)))
    scores = np.maximum.reduce([p[:, :, 0] - y, y - p[:, :, 2], np.zeros_like(y)])
    rank = min(len(y), math.ceil((len(y) + 1) * coverage))
    return {"expansion": np.sort(scores, axis=0)[rank - 1], "rank": rank, "rows": len(y)}


def apply_expansion(prediction, calibration):
    p = ordered_supported(prediction)
    expansion = np.asarray(calibration["expansion"])
    if expansion.shape != (6,) or not np.isfinite(expansion).all() or (expansion < 0).any():
        raise ValueError("Interval expansions must be six finite nonnegative values")
    p[:, :, 0] -= expansion
    p[:, :, 2] += expansion
    return ordered_supported(p)


def prepare_development(frames):
    # Filter every source immediately; no original test observations reach feature builders.
    frames = {
        name: frame.loc[frame.Split.isin(["train", "validation"])].reset_index(drop=True)
        for name, frame in frames.items()
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
    return base, masks


def verify_parent(root, data_dir):
    checksums = json.loads((root / "checksums.json").read_text())
    contents = (root / "experiment.json").read_bytes()
    if hashlib.sha256(contents).hexdigest() != checksums["experiment.json"]:
        raise ValueError("Parent experiment checksum mismatch")
    parent = json.loads(contents)
    if set(parent["source_hashes"]) != set(FILES.values()):
        raise ValueError("Parent source manifest does not match required inputs")
    for name, expected in parent["source_hashes"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe source path")
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    code_hash = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(Path("ml").rglob("*.py")))
    ).hexdigest()
    if code_hash != parent["code_hash"]:
        raise ValueError("ML source changed; use the parent's archived source snapshot")
    return parent


def coverage_slices(base, observed, predictions, scale):
    result = {}
    for column in ["Map", "Agent", "LengthBand"]:
        result[column] = {}
        labels = base[column].fillna("unknown")
        for label in sorted(labels.unique()):
            mask = labels.eq(label).to_numpy()
            result[column][str(label)] = {
                "rows": int(mask.sum()),
                "series": int(base.loc[mask, "MatchID"].nunique()),
                "predictions": {
                    name: {
                        component: {
                            key: float(value[:, i].mean())
                            for key, value in metric_arrays(observed[mask], p[mask], scale).items()
                            if key in {"coverage", "lower_tail", "upper_tail", "width"}
                        }
                        for i, component in enumerate(COMPONENTS)
                    }
                    for name, p in predictions.items()
                },
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use a new empty output directory")
    parent = verify_parent(args.parent, args.data_dir)
    base, masks = prepare_development(load_data(args.data_dir))
    partitions = {
        name: {
            "rows": int(mask.sum()),
            "series": int(base.loc[mask, "MatchID"].nunique()),
            "min_date": str(base.loc[mask, "Date"].min()),
            "max_date": str(base.loc[mask, "Date"].max()),
        }
        for name, mask in masks.items()
    }
    if partitions != parent["partitions"]:
        raise ValueError("Development partitions differ from the recorded parent experiment")
    protocol = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "parent": str(args.parent),
        "parent_experiment_hash": hashlib.sha256(
            (args.parent / "experiment.json").read_bytes()
        ).hexdigest(),
        "source_hashes": parent["source_hashes"],
        "ml_code_hash": parent["code_hash"],
        "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partitions": partitions,
        "features": CONTEXT_COLUMNS,
        "components": COMPONENTS,
        "quantiles": QUANTILES.tolist(),
        "recipe": RECIPE,
        "minimum_empirical_peer_count": MIN_PEERS,
        "nominal_coverage": NOMINAL_COVERAGE,
        "planned_fits": 72,
        "selection_rule": "Learned only if lower normalized pinball in all three folds and selection",
        "calibration_rule": "Nonnegative expansion, rank min(n, ceil((n+1)*0.8)), calibration rows only",
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "joblib"]
        },
        "limitation": LIMITATION,
    }
    write_json(args.output / "protocol.json", protocol)
    (args.output / "snapshot.txt").write_bytes(Path(__file__).read_bytes())
    screening = []
    for fold, (fit, evaluate) in enumerate(temporal_folds(base), 1):
        training, evaluation = base.loc[fit], base.loc[evaluate]
        scale = np.maximum(training_targets(training).std(axis=0), 1e-6)
        empirical = fit_empirical(training)
        learned = fit_learned(training)
        observed = component_values(evaluation).to_numpy()
        row = {
            "fold": fold,
            "fit_series": int(training.MatchID.nunique()),
            "evaluation_series": int(evaluation.MatchID.nunique()),
            "fit_max_date": str(training.Date.max()),
            "evaluation_min_date": str(evaluation.Date.min()),
            "empirical": summarize(observed, predict_empirical(empirical, evaluation)[0], scale),
            "learned": summarize(observed, predict_learned(learned, evaluation), scale),
        }
        screening.append(row)
        write_json(args.output / "screening.json", screening)
        print(
            f"Fold {fold}: empirical pinball={row['empirical']['normalized_pinball']:.6f}; "
            f"learned={row['learned']['normalized_pinball']:.6f}",
            flush=True,
        )
    training, selection = base.loc[masks["train"]], base.loc[masks["selection"]]
    scale = np.maximum(training_targets(training).std(axis=0), 1e-6)
    empirical, learned = fit_empirical(training), fit_learned(training)
    observed = component_values(selection).to_numpy()
    selected_metrics = {
        "empirical": summarize(observed, predict_empirical(empirical, selection)[0], scale),
        "learned": summarize(observed, predict_learned(learned, selection), scale),
    }
    chosen = select_components(screening, selected_metrics)
    frozen = {
        "component_sources": chosen,
        "selection": selected_metrics,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": protocol["selection_rule"],
    }
    write_json(args.output / "selection-frozen.json", frozen)
    print(f"Frozen component sources: {chosen}", flush=True)
    bundle = {
        "research_only": True,
        "empirical": empirical,
        "learned": learned,
        "component_sources": chosen,
        "scale": scale,
        "quantiles": QUANTILES,
        "features": CONTEXT_COLUMNS,
        "limitation": LIMITATION,
    }
    calibration = base.loc[masks["calibration"]]
    observed_cal = component_values(calibration).to_numpy()
    bundle["calibration"] = {
        "candidate": fit_expansion(observed_cal, predict_selected(bundle, calibration)),
        "baseline": fit_expansion(observed_cal, predict_empirical(empirical, calibration)[0]),
    }
    joblib.dump(bundle, args.output / "peer-intervals.joblib", compress=3)
    confirm = base.loc[masks["confirmation"]]
    observed = component_values(confirm).to_numpy()
    baseline, levels = predict_empirical(empirical, confirm)
    candidate = predict_selected(bundle, confirm)
    predictions = {
        "baseline_raw": baseline,
        "candidate_raw": candidate,
        "baseline_calibrated": apply_expansion(baseline, bundle["calibration"]["baseline"]),
        "candidate_calibrated": apply_expansion(candidate, bundle["calibration"]["candidate"]),
    }
    evaluation = {name: summarize(observed, p, scale) for name, p in predictions.items()}
    comparisons = {}
    for suffix in ["raw", "calibrated"]:
        candidate_errors = metric_arrays(observed, predictions[f"candidate_{suffix}"], scale)
        baseline_errors = metric_arrays(observed, predictions[f"baseline_{suffix}"], scale)
        differences = (
            candidate_errors["normalized_interval_score"]
            - baseline_errors["normalized_interval_score"]
        )
        comparisons[suffix] = {
            "overall": paired_mean_interval(confirm.MatchID, differences.mean(axis=1)),
            "by_component": {
                name: paired_mean_interval(confirm.MatchID, differences[:, i])
                for i, name in enumerate(COMPONENTS)
            },
        }
    saved = confirm[
        [
            "GameID",
            "PlayerID",
            "MatchID",
            "Date",
            "Map",
            "Agent",
            "Patch",
            "LengthBand",
            "NumRounds",
        ]
    ].copy()
    saved["empirical_peer_level"] = levels
    for i, component in enumerate(COMPONENTS):
        saved[f"{component}_observed"] = observed[:, i]
        for name, prediction in predictions.items():
            for j, quantile in enumerate(QUANTILES):
                saved[f"{name}_{component}_q{quantile}"] = prediction[:, i, j]
    saved.to_parquet(args.output / "diagnostic-predictions.parquet", index=False)
    reloaded = joblib.load(args.output / "peer-intervals.joblib")
    np.testing.assert_array_equal(predict_selected(reloaded, confirm), candidate)
    np.testing.assert_array_equal(predict_empirical(reloaded["empirical"], confirm)[0], baseline)
    mutated = confirm.copy()
    for column in mutated.columns.difference(CONTEXT_COLUMNS):
        mutated[column] = "unused"
    np.testing.assert_array_equal(predict_selected(reloaded, mutated), candidate)
    for name in ["candidate", "baseline"]:
        raw = candidate if name == "candidate" else baseline
        np.testing.assert_array_equal(
            apply_expansion(raw, reloaded["calibration"][name]), predictions[f"{name}_calibrated"]
        )
    result = {
        **protocol,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_fits": 72,
        "screening": screening,
        "selection": frozen,
        "scale": scale.tolist(),
        "calibration": {
            name: {**value, "expansion": value["expansion"].tolist()}
            for name, value in bundle["calibration"].items()
        },
        "confirmation": evaluation,
        "comparison": comparisons,
        "slices": coverage_slices(confirm, observed, predictions, scale),
        "verification": {
            "all_predictions_reloaded_exactly": len(confirm),
            "all_noncontext_inputs_mutated": "unchanged predictions",
            "quantile_order_and_nonnegative_support": "enforced",
            "source_and_ml_code_hashes": "matched parent",
        },
    }
    write_json(args.output / "results.json", result)
    write_json(
        args.output / "checksums.json",
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.output.iterdir())
            if path.is_file() and path.name not in {"checksums.json", "training.log"}
        },
    )
    print(json.dumps({"selected": chosen, "comparison": comparisons["calibrated"]}, indent=2))


if __name__ == "__main__":
    main()
