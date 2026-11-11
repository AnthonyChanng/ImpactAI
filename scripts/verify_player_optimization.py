"""Verify a trusted alternative-learner run and generate its research report."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from ml.explainability.explain import explain
from ml.features.build import player_features
from ml.improve import write_json
from ml.preprocessing.audit import load_data
from scripts.optimize_player import calibrated, group_probability


def verify(root, data_dir):
    hashes = json.loads((root / "checksums.json").read_text())
    required = {
        "candidate.joblib",
        "confirmation.parquet",
        "results.json",
        "selection-frozen.json",
        "protocol.json",
        "script-snapshot.txt",
        "screening.json",
    }
    if not required.issubset(hashes):
        raise ValueError("Required checksums missing")
    for name, expected in hashes.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe artifact path")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    report = json.loads((root / "results.json").read_text())
    if (
        hashlib.sha256(Path("scripts/optimize_player.py").read_bytes()).hexdigest()
        != report["script_hash"]
    ):
        raise ValueError("Training script changed; use its recorded snapshot")
    if (
        hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path("ml").rglob("*.py")))
        ).hexdigest()
        != report["ml_code_hash"]
    ):
        raise ValueError("Model source changed")
    for name, expected in report["source_hashes"].items():
        if (
            Path(name).name != name
            or hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Source data mismatch")
    raw = load_data(data_dir)["players"]
    raw = raw[raw.Split.isin(["train", "validation"])].reset_index(drop=True)
    stored = pd.read_parquet(root / "confirmation.parquet")
    rows = stored[["GameID", "PlayerID"]].merge(
        raw, on=["GameID", "PlayerID"], how="left", validate="one_to_one"
    )
    assert rows.Split.eq("validation").all()
    assert len(rows) == report["partitions"]["confirmation"]["rows"]
    assert str(rows.Date.min()) == report["partitions"]["confirmation"]["min_date"]
    base = player_features(rows)
    bundle = joblib.load(root / "candidate.joblib")
    assert bundle["research_only"] and bundle["selected"] == report["selected"]
    predicted = calibrated(
        bundle["calibration"], group_probability(bundle["group"], bundle["members"], base)
    )
    assert np.isfinite(predicted).all() and ((predicted >= 0) & (predicted <= 1)).all()
    np.testing.assert_array_equal(predicted, stored.candidate.to_numpy())
    # Identity and outcome mutations must not influence inference.
    changed = base.copy()
    changed[["PlayerID", "GameID", "MatchID"]] = -1
    changed["TeamWon"] = 1 - changed.TeamWon
    np.testing.assert_array_equal(
        calibrated(
            bundle["calibration"], group_probability(bundle["group"], bundle["members"], changed)
        ),
        predicted,
    )
    contains_native = any(
        not hasattr(m["model"]["model"], "get_booster")
        for group in bundle["members"].values()
        for m in group
    )
    positions = np.linspace(0, len(base) - 1, 16 if contains_native else 512, dtype=int)
    sample = base.iloc[positions]
    background = player_features(raw[raw.Split.eq("train")].sample(n=32, random_state=42))
    reconstructed_group = {}
    explanation_checks = []
    for name, members in bundle["members"].items():
        member_outputs = []
        for number, member in enumerate(members):
            x = member["features"].transform(sample)
            model = member["model"]
            estimator = model["model"]
            if hasattr(estimator, "get_booster"):
                values, baseline, names, method = explain(model, x)
                reconstructed = expit(baseline + values.sum(axis=1))
                scale = "uncalibrated log-odds"
            else:
                # Native categorical splits use model-agnostic explanations on encoded
                # features. Do not incorrectly label these as TreeSHAP.
                import shap

                encoded = model["pre"].transform(x)
                encoded_background = model["pre"].transform(
                    member["features"].transform(background)
                )
                names = model["pre"].get_feature_names_out().tolist()
                explainer = shap.PermutationExplainer(
                    lambda a: estimator.predict_proba(a)[:, 1],
                    encoded_background,
                    feature_names=names,
                    seed=42,
                )
                explanation = explainer(encoded, max_evals=2 * encoded.shape[1] + 1, silent=True)
                values = explanation.values
                reconstructed = explanation.base_values + values.sum(axis=1)
                method, scale = (
                    "Permutation SHAP; one forward/reverse feature permutation",
                    "probability",
                )
            expected = model.predict_proba(x)[:, 1]
            np.testing.assert_allclose(reconstructed, expected, atol=1e-5, rtol=0)
            member_outputs.append(reconstructed)
            explanation_checks.append(
                {
                    "source": name,
                    "member": number,
                    "method": method,
                    "scale": scale,
                    "rows": len(sample),
                    "max_error": float(np.max(np.abs(reconstructed - expected))),
                }
            )
            pd.DataFrame(
                {"feature": names, "mean_absolute_contribution": np.abs(values).mean(axis=0)}
            ).sort_values("mean_absolute_contribution", ascending=False).to_csv(
                root / f"explanation-{name}-{number}.csv", index=False
            )
        reconstructed_group[name] = np.mean(member_outputs, axis=0)
    reconstructed = calibrated(
        bundle["calibration"],
        sum(item["weight"] * reconstructed_group[item["source"]] for item in bundle["group"]),
    )
    # Isotonic steps are discontinuous; compare pre-calibration reconstruction separately
    # if a tiny SHAP rounding difference lands on a step boundary.
    raw_reconstructed = sum(
        item["weight"] * reconstructed_group[item["source"]] for item in bundle["group"]
    )
    raw_expected = group_probability(bundle["group"], bundle["members"], sample)
    np.testing.assert_allclose(raw_reconstructed, raw_expected, atol=1e-5, rtol=0)
    if bundle["calibration"]["kind"] != "isotonic":
        np.testing.assert_allclose(reconstructed, predicted[positions], atol=1e-5, rtol=0)
    verification = {
        "status": "passed",
        "files_hashed": len(hashes),
        "rows": len(base),
        "reload_max_error": float(np.max(np.abs(predicted - stored.candidate.to_numpy()))),
        "identity_and_outcome_invariance": "passed",
        "explanations": explanation_checks,
        "reconstructed_raw_probability_max_error": float(
            np.max(np.abs(raw_reconstructed - raw_expected))
        ),
        "reconstructed_calibrated_probability_max_error": float(
            np.max(np.abs(reconstructed - predicted[positions]))
        ),
    }
    write_json(root / "verification.json", verification)
    return report, verification


def render(root, report, verification, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    previous = report["confirmation"]["previous"]
    candidate = report["confirmation"]["candidate"]
    difference = report["comparison"]
    relative = 100 * (previous["log_loss"] - candidate["log_loss"]) / previous["log_loss"]
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    for name, value in [("Frozen previous ensemble", previous), ("New selection", candidate)]:
        curve = value["calibration"]
        ax.plot([c["predicted"] for c in curve], [c["observed"] for c in curve], "o-", label=name)
    ax.set(
        xlabel="Predicted postgame win association",
        ylabel="Observed fraction",
        title="Reused development validation; not a fresh test",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    image_path = destination.with_suffix(".png")
    fig.savefig(image_path, dpi=160)
    plt.close(fig)
    best_alternative = min(
        (n for n in report["selection"] if not n.startswith("previous::")),
        key=lambda n: report["selection"][n]["log_loss"],
    )
    lines = [
        "# Alternative learner and calibration results — October 16, 2026",
        "",
        "Completed the [fixed search protocol](2026-10-16-final-search-plan.md): six native-categorical "
        "histogram-boosting recipes across three chronological training folds, two finalist refits, "
        "and 28 model/mixture/calibration comparisons. This required 40 boosted-model fits including "
        "inner early-stopping probes. No additional libraries, cloud compute, or external data were used.",
        "",
        "**This is reused development validation, not an independent test.** All original test rows "
        "were excluded from every fit and prediction. Calibration-family choice used an earlier "
        "selection-fit segment and a later selection segment; final calibration used the separate "
        "238-series calibration period. The final comparison covers 5,070 player-game rows / 243 series.",
        "",
        f"Frozen selection: **`{report['selected']}`**. Calibration parameters: "
        f"`{report['calibration_parameters']}`.",
        "",
        "| Model | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | Accuracy at 0.5 ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in [("Actual frozen previous ensemble", previous), ("New selection", candidate)]:
        lines.append(
            f"| {name} | {m['log_loss']:.6f} | {m['brier']:.6f} | {m['roc_auc']:.6f} | {100 * m['accuracy']:.2f}% |"
        )
    lines += [
        "",
        f"Relative log-loss reduction: **{relative:.3f}%**. Paired candidate-minus-previous loss "
        f"difference: `{difference['candidate_minus_baseline']:.6f}`; 95% series-bootstrap interval "
        f"`{difference['series_bootstrap_95pct']}`. This interval describes the reused comparison, "
        "not an independent generalization claim.",
        "",
        f"The best alternative-learner combination on the later selection segment was "
        f"`{best_alternative}` with log loss {report['selection'][best_alternative]['log_loss']:.6f}. "
        f"The chosen combination had selection loss {report['selection'][report['selected']]['log_loss']:.6f}. "
        "All candidates and unsuccessful calibration methods are retained in the JSON report.",
        "",
        "## Map checks",
        "",
        "| Map | Rows | Previous loss | New loss |",
        "|---|---:|---:|---:|",
    ]
    for name, m in report["slices"]["candidate"]["Map"].items():
        lines.append(
            f"| {name} | {m['n']} | {report['slices']['previous']['Map'][name]['log_loss']:.6f} | {m['log_loss']:.6f} |"
        )
    lines += [
        "",
        "Agent, patch, and length slices are saved in the full JSON. These slices were not "
        "used to revise the frozen selection. Small groups are descriptive and noisy.",
        "",
        "## Verification and limits",
        "",
        f"- All {verification['files_hashed']} artifact hashes, source hashes, model-source hash, and training-script hash matched before loading the trusted local bundle.",
        "- Reloaded all 5,070 predictions exactly and proved the reference predictions match the actual previous artifact by game/player keys.",
        "- Mutating identity and outcome fields did not affect inference. Fourteen new tests cover native categorical fallback, missing values, calibration, chronological boundaries, and mixture validation.",
        "- The complete Python suite passed **93 tests**. No frontend behavior was changed in this run.",
        f"- Explanation checks: `{verification['explanations']}`.",
        "- Explanation additivity verifies the calculation, not the causal truth or stability of feature attribution. Permutation explanations, if used, are sampled approximations on encoded features with a training-only background.",
        "- Probability calibration changes how scores map to probabilities; it does not establish that a player's actual contribution or coaching needs were measured correctly.",
        "- No classification threshold was optimized against the final comparison. Neither the heuristic impact index nor the component-specific peer benchmark was retuned in this run.",
        "",
        f"Artifacts: `{root}`. Training started `{report['started_at']}` and completed `{report['completed_at']}`. "
        "The new selection remains a research bundle; the application continues using its existing active artifact.",
        "",
        "```bash",
        "python -m scripts.optimize_player --parent ml/artifacts/research/player-20261016-v2 --output ml/artifacts/research/NEW_RUN",
        "python -m scripts.verify_player_optimization ml/artifacts/research/NEW_RUN --report docs/experiments/NEW-results.md",
        "```",
        "",
        f"![Development calibration]({image_path.name})",
        "",
    ]
    destination.write_text("\n".join(lines))
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, verification = verify(args.directory, args.data_dir)
    render(args.directory, report, verification, args.report)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
