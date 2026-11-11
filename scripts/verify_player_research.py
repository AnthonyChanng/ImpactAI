"""Verify a trusted local postgame run and produce a reviewable results report."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from ml.evaluation.metrics import metrics
from ml.explainability.explain import explain
from ml.features.build import COMPONENTS, component_values, player_features
from ml.improve import paired_interval, write_json
from ml.models.player_research import ensemble_probability, peer_error
from ml.preprocessing.audit import load_data


def verify(root, data_dir):
    checksums = json.loads((root / "checksums.json").read_text())
    required = {
        "experiment.json",
        "association-candidate.joblib",
        "peer-candidate.joblib",
        "association-confirmation.parquet",
        "peer-confirmation.parquet",
        "association-selection-frozen.json",
        "peer-selection-frozen.json",
        "ml-source.zip",
    }
    if not required.issubset(checksums):
        raise ValueError("Required artifact checksums are missing")
    for name, expected in checksums.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe artifact path")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    report = json.loads((root / "experiment.json").read_text())
    for name, expected in report["source_hashes"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe source path")
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    actual_code = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path("ml").rglob("*.py")))
    ).hexdigest()
    if actual_code != report["code_hash"]:
        raise ValueError("ML source changed; reproduce with the recorded ml-source.zip snapshot")
    raw = load_data(data_dir)["players"]
    raw = raw[raw.Split.eq("validation")].reset_index(drop=True)
    assert raw.Split.eq("validation").all()
    key = ["GameID", "PlayerID"]
    stored = pd.read_parquet(root / "association-confirmation.parquet")
    aligned = stored[key].merge(raw, on=key, how="left", validate="one_to_one")
    assert aligned.Split.eq("validation").all()
    assert len(aligned) == report["partitions"]["confirmation"]["rows"]
    assert str(aligned.Date.min()) == report["partitions"]["confirmation"]["min_date"]
    assert str(aligned.Date.max()) == report["partitions"]["confirmation"]["max_date"]
    base = player_features(aligned)
    bundle = joblib.load(root / "association-candidate.joblib")
    assert bundle["research_only"] and bundle["selected"] == report["association"]["selected"]
    verification = {
        "status": "passed",
        "manifest_files": len(checksums),
        "source_and_code_hashes": "matched",
        "partition": "validation only",
        "rows": len(base),
        "association_max_errors": {},
        "peer_max_errors": {},
    }
    for name, saved in bundle["groups"].items():
        prediction = ensemble_probability(saved["members"], base, saved["temperature"])
        assert np.isfinite(prediction).all() and ((prediction > 0) & (prediction < 1)).all()
        np.testing.assert_allclose(prediction, stored[name], atol=1e-6, rtol=0)
        verification["association_max_errors"][name] = float(
            np.max(np.abs(prediction - stored[name]))
        )
    selected = bundle["groups"][bundle["selected"]]
    positions = np.linspace(0, len(base) - 1, min(512, len(base)), dtype=int)
    sample = base.iloc[positions]
    explained_probabilities, explanations = [], []
    for number, member in enumerate(selected["members"]):
        x = member["features"].transform(sample)
        assert list(x) == member["model"].feature_names_in_.tolist()
        values, baseline, names, method = explain(member["model"], x)
        reconstructed = expit(baseline + values.sum(axis=1))
        expected = member["model"].predict_proba(x)[:, 1]
        np.testing.assert_allclose(reconstructed, expected, atol=1e-5, rtol=0)
        explained_probabilities.append(reconstructed)
        explanations.append(
            {
                "member": number,
                "method": method,
                "max_probability_error": float(np.max(np.abs(reconstructed - expected))),
            }
        )
        pd.DataFrame(
            {"feature": names, "mean_absolute_log_odds_contribution": np.abs(values).mean(axis=0)}
        ).sort_values("mean_absolute_log_odds_contribution", ascending=False).to_csv(
            root / f"player-explanations-member-{number}.csv", index=False
        )
    aggregate = expit(
        logit(np.clip(np.mean(explained_probabilities, axis=0), 1e-6, 1 - 1e-6))
        / selected["temperature"]
    )
    expected = ensemble_probability(selected["members"], sample, selected["temperature"])
    np.testing.assert_allclose(aggregate, expected, atol=1e-5, rtol=0)
    verification["explanations"] = {
        "rows": len(sample),
        "members": explanations,
        "ensemble_max_error": float(np.max(np.abs(aggregate - expected))),
    }
    peers = joblib.load(root / "peer-candidate.joblib")
    assert peers["research_only"] and peers["selected"] == report["peers"]["selected"]
    frozen_peer = (
        pd.read_parquet(root / "peer-confirmation.parquet")
        .set_index(key)
        .loc[pd.MultiIndex.from_frame(stored[key])]
    )
    peer_slices = {}
    for name, model in peers["models"].items():
        predicted = model.predict(base)
        frozen = frozen_peer[[f"{name}_{c}" for c in COMPONENTS]].to_numpy()
        assert np.isfinite(predicted).all()
        np.testing.assert_allclose(predicted, frozen, atol=1e-6, rtol=0)
        verification["peer_max_errors"][name] = float(np.max(np.abs(predicted - frozen)))
        mutated = base.copy()
        mutated["TeamWon"] = 1 - mutated.TeamWon
        mutated[["KPR", "DPR", "FKPR", "FDPR", "ADR", "APR", "ObjectiveRate", "ClutchRate"]] = 10000
        np.testing.assert_allclose(model.predict(mutated), predicted, atol=1e-9, rtol=0)
        error, rows = peer_error(model, base)
        np.testing.assert_allclose(
            error, list(report["peers"]["confirmation"][name]["by_component"].values())
        )
        peer_slices[name] = {}
        for column in ["Map", "Agent", "Patch", "LengthBand"]:
            view = pd.DataFrame({"group": base[column].fillna("unknown"), "error": rows})
            peer_slices[name][column] = {
                str(group): {"rows": len(frame), "normalized_mse": float(frame.error.mean())}
                for group, frame in view.groupby("group")
            }
    evidence = base[key + ["Agent", "Map", "Patch", "NumRounds"]].copy()
    actual = component_values(base).to_numpy()
    expected = peers["models"][peers["selected"]].predict(base)
    for i, component in enumerate(COMPONENTS):
        evidence[f"{component}_observed"] = actual[:, i]
        evidence[f"{component}_expected"] = expected[:, i]
        evidence[f"{component}_difference"] = actual[:, i] - expected[:, i]
    evidence.to_parquet(root / "player-peer-evidence.parquet", index=False)
    write_json(root / "peer-slices.json", peer_slices)
    write_json(root / "verification.json", verification)
    return report, verification


def compare_frozen_previous(root, previous_root, report):
    """Compare keyed frozen predictions; do not refit or retune either candidate."""
    hashes = json.loads((previous_root / "checksums.json").read_text())
    for name in ["experiment.json", "player-confirmation.parquet"]:
        if hashlib.sha256((previous_root / name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError("Previous run checksum mismatch")
    previous_report = json.loads((previous_root / "experiment.json").read_text())
    if previous_report["source_hashes"] != report["source_hashes"]:
        raise ValueError("Previous run used different source data")
    old = pd.read_parquet(previous_root / "player-confirmation.parquet")
    new = pd.read_parquet(root / "association-confirmation.parquet")
    joined = new.merge(
        old[["GameID", "PlayerID", "observed", "candidate"]],
        on=["GameID", "PlayerID"],
        validate="one_to_one",
        how="inner",
    )
    assert len(joined) == len(new) == len(old)
    np.testing.assert_array_equal(joined.TeamWon, joined.observed)
    selected = report["association"]["selected"]
    result = {
        "previous_directory": str(previous_root),
        "rows": len(joined),
        "previous": metrics(joined.observed, joined.candidate),
        "selected": metrics(joined.observed, joined[selected]),
        "comparison": paired_interval(
            joined.MatchID,
            joined.observed.to_numpy(),
            joined[selected].to_numpy(),
            joined.candidate.to_numpy(),
        ),
        "note": "The frozen prior candidate and its native refit have slightly different predictions; "
        "both comparisons are retained. No independent new test claim.",
    }
    write_json(root / "frozen-previous-comparison.json", result)
    report["frozen_previous_comparison"] = result


def render_report(root, report, verification, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    association, peers = report["association"], report["peers"]
    a_name, p_name = association["selected"], peers["selected"]
    candidate = association["confirmation"][a_name]["overall"]
    previous = association["confirmation"]["previous_context"]["overall"]
    original = association["confirmation"]["original"]["overall"]
    peer_candidate = peers["confirmation"][p_name]
    peer_previous = peers["confirmation"]["hierarchical"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    for label, value in [
        ("Original", original),
        ("Previous candidate", previous),
        ("New selection", candidate),
    ]:
        curve = value["calibration"]
        ax.plot([c["predicted"] for c in curve], [c["observed"] for c in curve], "o-", label=label)
    ax.set(
        xlabel="Predicted win association",
        ylabel="Observed fraction",
        title="Postgame association calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    ax = axes[1]
    changes = [
        100
        * (peer_previous["by_component"][c] - peer_candidate["by_component"][c])
        / peer_previous["by_component"][c]
        for c in COMPONENTS
    ]
    if p_name == "hierarchical":
        names = ["hierarchical", "boosted_shallow", "boosted_medium"]
        values = [peers["selection"][name]["normalized_mse"] for name in names]
        ax.plot(values, names, "o", color="#25855a")
        ax.axvline(values[0], color="gray", linestyle="--", linewidth=1)
        ax.set(
            xlabel="Selection normalized MSE (lower is better)",
            title="Peer benchmark: existing method retained",
        )
    else:
        ax.barh(COMPONENTS, changes, color=["#25855a" if x >= 0 else "#bf4450" for x in changes])
        ax.axvline(0, color="gray", linewidth=1)
        ax.set(
            xlabel="Squared-error reduction vs hierarchical means (%)",
            title="Outcome-free peer benchmarks",
        )
    fig.suptitle("Previously inspected development validation — not a fresh test")
    figure_path = destination.with_suffix(".png")
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False))
    change = 100 * (previous["log_loss"] - candidate["log_loss"]) / previous["log_loss"]
    peer_change = (
        100
        * (peer_previous["normalized_mse"] - peer_candidate["normalized_mse"])
        / peer_previous["normalized_mse"]
    )
    lines = [
        "# Postgame player training results — October 16, 2026",
        "",
        "Completed the [declared player-only experiment](2026-10-16-plan.md). "
        "All training used the supplied historical professional esports data and four local CPU threads. "
        "Original test rows were excluded from fitting, selection, calibration, and predictions.",
        "",
        "**The results below reuse historical validation; they are not an independent new test.** "
        "The 5,070 player rows cover 243 series. Selection used an earlier 480-series validation segment "
        "and temperature calibration a separate 238-series segment. Three expanding folds within "
        "original training screened candidates, and inner chronological windows selected tree counts.",
        "",
        "## Postgame outcome association",
        "",
        "| Model | Log loss ↓ | Brier ↓ | ROC-AUC ↑ | Accuracy ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    table_rows = [("Original specification", original)]
    if "frozen_previous_comparison" in report:
        table_rows.append(
            ("Frozen previous research candidate", report["frozen_previous_comparison"]["previous"])
        )
    table_rows += [
        ("Previous specification, native refit", previous),
        (f"New selection: {a_name}", candidate),
    ]
    for label, value in table_rows:
        lines.append(
            f"| {label} | {value['log_loss']:.6f} | {value['brier']:.6f} | {value['roc_auc']:.6f} | {100 * value['accuracy']:.2f}% |"
        )
    interval = association["comparison_to_previous"]
    candidate_slices = association["confirmation"][a_name]["slices"]
    previous_slices = association["confirmation"]["previous_context"]["slices"]
    map_gains = sum(
        value["log_loss"] < previous_slices["Map"][name]["log_loss"]
        for name, value in candidate_slices["Map"].items()
    )
    lines += [
        "",
        f"Relative log-loss reduction versus the previous specification refit: **{change:.2f}%**. "
        f"Candidate-minus-previous loss: {interval['candidate_minus_baseline']:.6f}; "
        f"95% series-bootstrap interval {interval['series_bootstrap_95pct']}. "
        "This interval describes the reused development comparison, not a fresh generalization estimate.",
        "",
        "Accuracy at the fixed 0.5 cutoff declined despite lower log loss. The selection objective "
        "was probability quality; the result is not an across-the-board improvement. "
        "The prior candidate's refit and its frozen artifact produce slightly different predictions, "
        "so both are retained when the previous run is supplied.",
        "",
        "The label is the team's completed-match outcome. These metrics do not validate individual "
        "causal impact, coaching quality, or a player's future win probability. Exact match length, "
        "plants, defuses, and survival-related statistics describe the completed match; their "
        "predictive association cannot be interpreted as a causal improvement prescription.",
        "",
        f"Log loss improved on {map_gains} of {len(candidate_slices['Map'])} map slices. "
        "Small slices can fluctuate substantially; these exploratory slice results were not used "
        "to revise the frozen selection.",
        "",
        "| Player coverage | Rows | Previous loss | Selected loss | Selected ROC-AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in candidate_slices["PlayerCoverage"].items():
        auc = f"{value['roc_auc']:.6f}" if value["roc_auc"] is not None else "Unavailable"
        lines.append(
            f"| {name} | {value['n']} | "
            f"{previous_slices['PlayerCoverage'][name]['log_loss']:.6f} | "
            f"{value['log_loss']:.6f} | {auc} |"
        )
    lines += [
        "",
        "Player identity is used only to define these evaluation groups; it is never a model input.",
        "",
        "## Outcome-free player comparisons",
        "",
        f"Selected benchmark: **{p_name}**. Mean training-variance-normalized squared error "
        f"changed from {peer_previous['normalized_mse']:.6f} to {peer_candidate['normalized_mse']:.6f} "
        f"(**{peer_change:.2f}% reduction**). "
        "The target is each player's measured performance component, not the match winner. "
        "At inference, only agent, map, patch, and match length determine expected values.",
        "",
        "| Component | Hierarchical baseline error | Selected error | Reduction |",
        "|---|---:|---:|---:|",
    ]
    for c, reduction in zip(COMPONENTS, changes):
        lines.append(
            f"| {c} | {peer_previous['by_component'][c]:.6f} | {peer_candidate['by_component'][c]:.6f} | {reduction:.2f}% |"
        )
    lines += [
        "",
        f"Paired mean-error comparison: `{peers['comparison_to_hierarchical']}`.",
        "",
        "Expected means support component-by-component comparisons. This experiment does not train "
        "a calibrated percentile distribution or validate new weights for a composite impact score. "
        "The existing heuristic impact score is not replaced by either model's win probability.",
        "",
        "## Verification and reproducibility",
        "",
        f"- Verified {verification['manifest_files']} artifact hashes and all source/code hashes before deserializing trusted local bundles.",
        f"- Reloaded all {verification['rows']:,} confirmation rows for every saved association/peer model and matched their predictions.",
        f"- Checked TreeSHAP reconstruction on {verification['explanations']['rows']} observations per selected member, including the actual ensemble probability/temperature arithmetic.",
        "- Verified peer means remain identical when observed stats and outcome labels are changed at inference.",
        "- Saved map, agent, patch, length, and seen/unseen-player association slices; peer slices and keyed observed/expected component evidence are also available.",
        "- Retained every screening/selection result, early-stopping count, specification, source snapshot, and negative finding.",
        "",
        f"Run directory: `{root}`. Started `{report['started_at']}`; completed `{report['completed_at']}`.",
        "",
        "Reproduce with `python -m ml.player_research --output ml/artifacts/research/NEW_DIRECTORY`. "
        "Verify using `python -m scripts.verify_player_research DIRECTORY --report docs/experiments/NEW-results.md`. "
        "A run's `ml-source.zip` preserves the exact source needed if later edits change the code hash.",
        "",
        "The saved models are research candidates. Activating them requires feature-serving integration "
        "and an explicit evaluation/release decision. Fresh authorized later matches and independently "
        "reviewed player-performance labels are needed before making stronger performance claims.",
        "",
        f"![Development calibration and peer component errors]({figure_path.name})",
        "",
    ]
    destination.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, help="Optional frozen prior research run for a keyed comparison"
    )
    args = parser.parse_args()
    report, verification = verify(args.directory, args.data_dir)
    if args.previous:
        compare_frozen_previous(args.directory, args.previous, report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    render_report(args.directory, report, verification, args.report)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
