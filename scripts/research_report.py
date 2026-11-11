"""Summarize saved evaluation and a validation-only, one-feature opening-kill ablation.

Does not tune or replace the frozen model and does not repeat held-out model evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.evaluation.metrics import metrics
from ml.features.build import PLAYER_CAT, PLAYER_NUM, player_features
from ml.models.inference import probabilities
from ml.preprocessing.audit import load_data
from ml.train import pipeline


def main():
    root = Path("ml/artifacts")
    version = json.loads((root / "current.json").read_text())["version"]
    directory = root / version
    report = json.loads((directory / "metrics.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    frames = load_data(Path("data/source"))
    p = frames["players"]
    eligible = ~p.GameID.isin(audit["unreliable_outcome_game_ids"])
    p = p.loc[eligible]
    x = player_features(p)
    validation = p[p.Split == "validation"]
    series = validation[["MatchID", "Date"]].drop_duplicates().sort_values(["Date", "MatchID"])
    selection = p.MatchID.isin(series.MatchID.iloc[: len(series) // 2])
    training = p.Split.eq("train")
    model = pipeline(PLAYER_CAT, [c for c in PLAYER_NUM if c != "FKPR"]).fit(
        x.loc[training], p.loc[training, "TeamWon"]
    )
    ablation = metrics(p.loc[selection, "TeamWon"], probabilities(model, x.loc[selection]))
    full = report["player"]["validation_selection"]["logistic"]
    r = report["round"]
    q = report["player"]
    summary = {
        "version": version,
        "validation_only_no_first_kills": ablation,
        "full_logistic_validation": full,
        "note": "Supplemental validation-only diagnostic requested by the research questions. No held-out tuning or replacement of the frozen selected model.",
    }
    # Inspect saved errors, not retraining or reselecting using the test set.
    raw_rounds = frames["rounds"]
    saved = pd.read_parquet(directory / "round-predictions.parquet")
    joined = raw_rounds.merge(
        saved, left_on=["GameID", "RoundNumber"], right_on=["game_id", "round_number"]
    )
    joined = joined[joined.Split == "test"].copy()
    joined["error"] = np.abs(joined.Team1_RoundWon - joined.probability)
    summary["largest_round_errors"] = (
        joined.nlargest(15, "error")[
            [
                "GameID",
                "RoundNumber",
                "Map",
                "Patch",
                "Team1_PreScore",
                "Team2_PreScore",
                "Team1_Side",
                "Team1_RoundWon",
                "probability",
            ]
        ]
        .fillna("unknown")
        .to_dict("records")
    )
    Path("docs/research-diagnostics.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Measured evaluation",
        "",
        f"Frozen artifact version: `{version}`. Seed: 42. No modern ranked claims.",
        "",
        "| Selected calibrated model | Test rows | Log loss | Brier | ROC-AUC | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, task in [
        ("Round / " + r["selected"], r),
        ("Retrospective player / " + q["selected"], q),
    ]:
        m = task["test_selected"]
        lines.append(
            f"| {name} | {m['n']:,} | {m['log_loss']:.6f} | {m['brier']:.6f} | {m['roc_auc']:.6f} | {m['accuracy']:.4%} |"
        )
    lines += [
        "",
        "## Research questions",
        "",
        f"- XGBoost beat logistic regression on selection-validation log loss: round {r['validation_selection']['xgboost_depth5']['log_loss']:.5f} vs {r['validation_selection']['logistic']['log_loss']:.5f}; player {q['validation_selection']['xgboost']['log_loss']:.5f} vs {q['validation_selection']['logistic']['log_loss']:.5f}. This comparison is observational, not a significance claim.",
        f"- Removing only first-kill rate while retaining kills/deaths and first-death rate gives validation log loss {ablation['log_loss']:.5f}, versus {full['log_loss']:.5f} for full logistic. The incremental first-kill evidence is small; this is a validation-only supplemental diagnostic, not test tuning.",
        f"- Removing both opening rates gives validation loss {q['validation_selection']['no_openings']['log_loss']:.5f}. Context plus ACS gives {q['validation_selection']['acs_context']['log_loss']:.5f}, versus {q['validation_selection']['acs_only']['log_loss']:.5f} for ACS alone.",
        f"- Other-nine-player controls alone yield validation loss {q['validation_selection']['diagnostic_controls_only']['log_loss']:.5f} and AUC {q['validation_selection']['diagnostic_controls_only']['roc_auc']:.5f}. Adding player statistics yields loss {q['validation_selection']['diagnostic_controls_plus_player']['log_loss']:.5f}. This strong scoreboard reconstruction is why these controls are diagnostic and never the displayed index.",
        f"- Removing opening, clutch, objective, HS, and length-band inputs gives conservative player validation loss {q['validation_selection']['conservative']['log_loss']:.5f}. All remain postgame models; even the conservative model is not a pregame predictor.",
        f"- Removing score/history from the round model gives validation loss {r['validation_selection']['no_history']['log_loss']:.5f}. Economy and current-round outcomes are excluded from every experiment, not merely from a final ablation.",
        f"- Maximum team-swap complement error: {r['max_swap_error']:.3g}. The constant prior is 0.5 after perspective balancing; the raw source-team win rate is {audit['team1_round_win_rate']:.4%}.",
        "",
        "## Held-out map errors",
        "",
        "| Map | Round rows | Log loss | ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for name, m in r["slices"]["Map"].items():
        lines.append(f"| {name} | {m['n']:,} | {m['log_loss']:.5f} | {m['roc_auc']:.5f} |")
    lines += [
        "",
        "These are chronological within-map slices. Fracture has no training examples and therefore uses unknown-category handling; map-specific performance should be interpreted with its sample count. No leave-one-map-out retraining was performed.",
        "",
        "## Patch variation",
        "",
        "| Patch | Round rows | Log loss | ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for name, m in r["slices"]["Patch"].items():
        auc = "N/A" if m["roc_auc"] is None else f"{m['roc_auc']:.5f}"
        lines.append(f"| {name} | {m['n']:,} | {m['log_loss']:.5f} | {auc} |")
    lines += [
        "",
        "## Error analysis and limits",
        "",
        "`research-diagnostics.json` contains the 15 largest saved test-round probability errors, with pre-score, side, map, patch, and observed result. These are inspection examples, not grounds for retuning this held-out experiment.",
        "Round predictions miss unobserved team skill, lineups, economy, utility, and other states. Large confident errors can reflect recovery from a score deficit; the available fields cannot explain the underlying play.",
        "Temperature calibration is fit only on the second half of validation series. Confidence intervals use 300 series-cluster bootstrap draws; they do not cover deployment drift or causal uncertainty.",
        "The player index uses equal weights and is explicitly heuristic. Leave-one-component-out correlations are stored in evaluation.json; correlated combat/damage components and absent opportunity counts remain limitations.",
        "Metrics, per-map/side/patch slices, calibration bins, SHAP importances, model manifests, and figures are generated from actual training outputs. Training and validation historical UI rows are not held-out predictions.",
        "",
    ]
    Path("docs/evaluation.md").write_text("\n".join(lines))
    print(
        f"Wrote evaluation.md; first-kill ablation validation log loss={ablation['log_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
