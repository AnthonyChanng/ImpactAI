"""Run the complete predeclared experiment once, then publish immutable version artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml.evaluation.metrics import grouped_interval, metrics, slices
from ml.explainability.explain import explain
from ml.features.build import (
    PLAYER_CAT,
    PLAYER_NUM,
    ROUND_CAT,
    ROUND_NUM,
    player_features,
    round_features,
    scoreboard_controls,
    swap_round,
)
from ml.models.context import ContextIndex
from ml.models.inference import probabilities
from ml.preprocessing.audit import audit, load_data

SEED = 42


def pipeline(cat, num, kind="logistic", depth=3):
    pre = ColumnTransformer(
        [
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            ),
        ],
        remainder="drop",
    )
    if kind == "xgboost":
        model = XGBClassifier(
            n_estimators=180,
            max_depth=depth,
            learning_rate=0.05,
            min_child_weight=30,
            subsample=1,
            colsample_bytree=1,
            reg_lambda=5,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=4,
        )
    elif kind == "prior":
        model = DummyClassifier(strategy="prior")
    else:
        model = LogisticRegression(C=0.3, max_iter=600, random_state=SEED)
    return Pipeline([("pre", pre), ("model", model)])


def train_family(x, raw, target, configs, symmetric=False):
    train = raw.Split.eq("train")
    validation = raw[raw.Split.eq("validation")]
    series = validation[["MatchID", "Date"]].drop_duplicates().sort_values(["Date", "MatchID"])
    midpoint = len(series) // 2
    select = raw.MatchID.isin(series.MatchID.iloc[:midpoint])
    calibrate = raw.MatchID.isin(series.MatchID.iloc[midpoint:])
    tx, ty = x.loc[train], target.loc[train]
    if symmetric:
        tx = pd.concat([tx, swap_round(tx)], ignore_index=True)
        ty = pd.concat([ty, 1 - ty], ignore_index=True)
    models, comparison = {}, {}
    for name, config in configs.items():
        print(f"Training {'round' if symmetric else 'player'} / {name}", flush=True)
        model = pipeline(**config).fit(tx, ty)
        pred = probabilities(model, x.loc[select], symmetric)
        comparison[name] = metrics(target.loc[select], pred)
        models[name] = model
    eligible = [name for name in configs if not name.startswith("diagnostic_")]
    selected = min(eligible, key=lambda name: comparison[name]["log_loss"])
    model = models[selected]
    cp = probabilities(model, x.loc[calibrate], symmetric)
    cy = target.loc[calibrate].to_numpy()
    from scipy.special import expit, logit

    logits = logit(np.clip(cp, 1e-6, 1 - 1e-6))
    fit = minimize_scalar(
        lambda t: log_loss(cy, expit(logits / t), labels=[0, 1]), bounds=(0.5, 3), method="bounded"
    )
    temperature = float(fit.x)
    # Freeze model selection and temperature before accessing the test target.
    test = raw.Split.eq("test")
    test_predictions = probabilities(model, x.loc[test], symmetric, temperature)
    test_comparison = {
        name: metrics(target.loc[test], probabilities(m, x.loc[test], symmetric))
        for name, m in models.items()
    }
    report = {
        "selected": selected,
        "temperature": temperature,
        "validation_selection": comparison,
        "test_predeclared_models_uncalibrated": test_comparison,
        "test_selected": metrics(target.loc[test], test_predictions),
        "test_log_loss_95pct_series_bootstrap": grouped_interval(
            raw.loc[test], target.loc[test].to_numpy(), test_predictions
        ),
        "calibration_validation_before": metrics(cy, cp),
        "calibration_validation_after": metrics(cy, expit(logits / temperature)),
        "selection_series": midpoint,
        "calibration_series": len(series) - midpoint,
        "slices": slices(
            x.loc[test],
            target.loc[test].to_numpy(),
            test_predictions,
            ["Map", "Patch", "Team1_Side"] if symmetric else ["Map", "Patch", "Agent"],
        ),
    }
    if symmetric:
        sample = x.loc[test].iloc[:1000]
        a = probabilities(model, sample, True, temperature)
        b = probabilities(model, swap_round(sample), True, temperature)
        report["max_swap_error"] = float(np.max(np.abs(a + b - 1)))
    return model, temperature, report, models


def save_plots(report, directory):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, task in zip(axes, ["round", "player"]):
        curve = report[task]["test_selected"]["calibration"]
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.plot(
            [c["predicted"] for c in curve], [c["observed"] for c in curve], "o-", color="#e94b63"
        )
        ax.set(
            title=f"{task.title()} held-out calibration",
            xlabel="Predicted probability",
            ylabel="Observed fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
    fig.savefig(directory / "calibration.png", dpi=160)
    plt.close(fig)
    features = report["global_features"][:12][::-1]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.barh([f["feature"] for f in features], [f["importance"] for f in features], color="#e94b63")
    ax.set(
        xlabel="Mean absolute contribution (log-odds)",
        title="Player model behavior; not causal contribution",
    )
    fig.savefig(directory / "feature-importance.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts"))
    args = parser.parse_args()
    np.random.seed(SEED)
    frames = load_data(args.data_dir)
    checked = audit(frames, args.data_dir)
    if checked["errors"]:
        raise ValueError(checked["errors"])
    p, r = frames["players"], frames["rounds"]
    rx, px = round_features(r), scoreboard_controls(player_features(p))
    source = json.dumps(checked["source_hashes"], sort_keys=True)
    code = b"".join(f.read_bytes() for f in sorted(Path("ml").rglob("*.py")))
    version = "v1-" + hashlib.sha256(source.encode() + code).hexdigest()[:12]
    out = args.output / version
    if (out / "manifest.json").exists():
        print(f"Version {version} already complete; reuse it or change code/config for a new run.")
        return
    out.mkdir(parents=True, exist_ok=True)
    feature_dir = Path("ml/data") / version
    feature_dir.mkdir(parents=True, exist_ok=True)
    rx.join(r[["GameID", "MatchID", "Split", "Team1_RoundWon"]]).to_parquet(
        feature_dir / "round.parquet", index=False
    )
    px.to_parquet(feature_dir / "player.parquet", index=False)
    schema = {
        "version": version,
        "round_features": rx.dtypes.astype(str).to_dict(),
        "player_primary_features": PLAYER_CAT + PLAYER_NUM,
        "excluded_round": [c for c in r if "POST" in c or "DO_NOT_TRAIN" in c or "Unknown" in c],
        "excluded_player": [
            "KAST_Percent",
            "Econ",
            "PlusMinus",
            "FKFD_PlusMinus",
            "PlayerID",
            "TeamSide",
            "TeamAbbreviation",
        ],
        "seed": SEED,
    }
    (out / "feature-schema.json").write_text(json.dumps(schema, indent=2))
    round_configs = {
        "prior": dict(cat=ROUND_CAT, num=ROUND_NUM, kind="prior"),
        "logistic": dict(cat=ROUND_CAT, num=ROUND_NUM),
        "xgboost_depth3": dict(cat=ROUND_CAT, num=ROUND_NUM, kind="xgboost"),
        "xgboost_depth5": dict(cat=ROUND_CAT, num=ROUND_NUM, kind="xgboost", depth=5),
        "no_history": dict(cat=["Map", "Patch", "Team1_Side", "Phase"], num=["RoundNumber"]),
    }
    rm, rt, rr, _ = train_family(rx, r, r.Team1_RoundWon, round_configs, True)
    controls = [c for c in px if c.startswith(("Teammates_", "Opponents_"))]
    player_configs = {
        "prior": dict(cat=[], num=["KD"], kind="prior"),
        "kd_only": dict(cat=[], num=["KD"]),
        "acs_only": dict(cat=[], num=["ACS"]),
        "adr_only": dict(cat=[], num=["ADR"]),
        "acs_context": dict(cat=PLAYER_CAT, num=["ACS"]),
        "logistic": dict(cat=PLAYER_CAT, num=PLAYER_NUM),
        "xgboost": dict(cat=PLAYER_CAT, num=PLAYER_NUM, kind="xgboost"),
        "no_openings": dict(
            cat=PLAYER_CAT, num=[c for c in PLAYER_NUM if c not in ["FKPR", "FDPR"]]
        ),
        "conservative": dict(cat=["Agent", "Map", "Patch"], num=["KPR", "DPR", "APR", "ADR"]),
        "diagnostic_controls_only": dict(cat=PLAYER_CAT, num=controls),
        "diagnostic_controls_plus_player": dict(cat=PLAYER_CAT, num=controls + PLAYER_NUM),
    }
    eligible = ~p.GameID.isin(checked["unreliable_outcome_game_ids"])
    pm, pt, pr, all_player = train_family(
        px.loc[eligible], p.loc[eligible], p.loc[eligible].TeamWon, player_configs
    )
    pr["excluded_unreliable_outcome_rows"] = int((~eligible).sum())
    print("Computing training-only peer references and explanations", flush=True)
    context = ContextIndex().fit(px.loc[p.Split.eq("train")])
    contextual = context.transform(px)
    pp = probabilities(pm, px, temperature=pt)
    if pr["selected"] == "prior":
        raise RuntimeError("Selected prior has no feature explanation; inspect data")
    values, base, names, method = explain(pm, px)
    records = []
    for i, row in enumerate(p.itertuples()):
        top = np.argsort(np.abs(values[i]))[-8:][::-1]
        records.append(
            {
                "game_id": int(row.GameID),
                "player_id": int(row.PlayerID),
                "model_version": version,
                **contextual[i],
                "association_probability": float(pp[i]),
                "explanation": {
                    "method": method,
                    "base_log_odds": base,
                    "raw_log_odds": float(base + values[i].sum()),
                    "temperature": pt,
                    "top": [
                        {"feature": names[j], "contribution": float(values[i, j])} for j in top
                    ],
                    "other_contribution": float(values[i].sum() - values[i, top].sum()),
                },
            }
        )
    with (out / "player-predictions.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    round_preds = probabilities(rm, rx, True, rt)
    pd.DataFrame(
        {"game_id": r.GameID, "round_number": r.RoundNumber, "probability": round_preds}
    ).to_parquet(out / "round-predictions.parquet", index=False)
    test_values = values[p.Split.eq("test")]
    global_features = sorted(
        [
            {"feature": name, "importance": float(np.abs(test_values[:, i]).mean())}
            for i, name in enumerate(names)
        ],
        key=lambda x: -x["importance"],
    )
    # Always provide actual TreeSHAP research output, even when logistic wins selection.
    shap_sample = px.loc[p.Split.eq("test")].iloc[:2000]
    sv, sb, sn, sm = explain(all_player["xgboost"], shap_sample)
    tree_global = sorted(
        [{"feature": n, "importance": float(np.abs(sv[:, i]).mean())} for i, n in enumerate(sn)],
        key=lambda x: -x["importance"],
    )
    z, _, _ = context.standardize(px.loc[p.Split.eq("test")])
    full_score = z.mean(axis=1)
    score_sensitivity = {
        c: float(np.corrcoef(full_score, np.delete(z, i, axis=1).mean(axis=1))[0, 1])
        for i, c in enumerate(["Combat", "Openings", "Damage", "Assists", "Objectives", "Clutches"])
    }
    report = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round": rr,
        "player": pr,
        "global_features": global_features,
        "global_explanation_method": method,
        "xgboost_shap": tree_global,
        "xgboost_shap_sample_n": len(shap_sample),
        "heuristic_leave_one_component_out_correlation": score_sensitivity,
        "limitations": checked["source_quality_report"]["limitations"]
        + [
            "Player models observe the finished scoreboard; predictions are retrospective associations",
            "Heuristic index weights are equal, not learned causal contributions",
            "Clutch and objective counts lack opportunity denominators",
            "Confidence intervals resample series; limited to this historical test period",
            "Train/validation historical rows have in-sample/fitting-period predictions; test rows are held out",
            "Map slices measure temporal performance within maps, not unseen-map transfer",
        ],
        "protocol": "Train on supplied train. First half of validation series selects minimum log loss. Second half fits one temperature. Freeze; evaluate all predeclared candidates and selected calibrated model once on test. No test tuning.",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / "audit.json").write_text(json.dumps(checked, indent=2, allow_nan=False))
    bundle = {
        "version": version,
        "round_model": rm,
        "round_temperature": rt,
        "player_model": pm,
        "player_temperature": pt,
        "context": context,
        "schema": schema,
    }
    joblib.dump(bundle, out / "bundle.joblib", compress=3)
    for label, model in [("player-xgboost", all_player["xgboost"]), ("round-selected", rm)]:
        if hasattr(model.named_steps["model"], "save_model"):
            model.named_steps["model"].save_model(out / f"{label}.json")
    save_plots(report, out)
    manifest = {
        "version": version,
        "source_hashes": checked["source_hashes"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "joblib"]
        },
        "sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()
        },
        "trusted_local_artifacts_only": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "current.json").write_text(json.dumps({"version": version}))
    Path("docs/evaluation.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {"version": version, "round": rr["test_selected"], "player": pr["test_selected"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
