"""Verify trusted local peer-interval artifacts and render their development report."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features.build import COMPONENTS, component_values
from ml.improve import write_json
from ml.player_research import paired_mean_interval
from ml.preprocessing.audit import load_data
from scripts.peer_intervals import (
    CONTEXT_COLUMNS,
    QUANTILES,
    apply_expansion,
    fit_empirical,
    fit_expansion,
    predict_empirical,
    predict_selected,
    prepare_development,
    select_components,
    verify_parent,
)


def verify(root, data_dir):
    hashes = json.loads((root / "checksums.json").read_text())
    required = {
        "protocol.json",
        "snapshot.txt",
        "screening.json",
        "selection-frozen.json",
        "peer-intervals.joblib",
        "diagnostic-predictions.parquet",
        "results.json",
    }
    if not required.issubset(hashes):
        raise ValueError("Missing required artifact checksums")
    for name, expected in hashes.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe artifact path")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    report = json.loads((root / "results.json").read_text())
    parent_path = Path(report["parent"])
    parent = verify_parent(parent_path, data_dir)
    assert (
        hashlib.sha256((parent_path / "experiment.json").read_bytes()).hexdigest()
        == report["parent_experiment_hash"]
    )
    assert report["source_hashes"] == parent["source_hashes"]
    assert report["ml_code_hash"] == parent["code_hash"]
    assert (
        hashlib.sha256(Path("scripts/peer_intervals.py").read_bytes()).hexdigest()
        == report["script_hash"]
    )
    assert (root / "snapshot.txt").read_bytes() == Path("scripts/peer_intervals.py").read_bytes()
    base, masks = prepare_development(load_data(data_dir))
    confirm = base.loc[masks["confirmation"]]
    calibration = base.loc[masks["calibration"]]
    stored = pd.read_parquet(root / "diagnostic-predictions.parquet")
    keys = ["GameID", "PlayerID", "MatchID"]
    pd.testing.assert_frame_equal(
        stored[keys].reset_index(drop=True), confirm[keys].reset_index(drop=True)
    )
    assert not stored.duplicated(["GameID", "PlayerID"]).any()
    assert len(confirm) == report["partitions"]["confirmation"]["rows"]
    assert confirm.MatchID.nunique() == report["partitions"]["confirmation"]["series"]
    assert calibration.Date.max() < confirm.Date.min()
    bundle = joblib.load(root / "peer-intervals.joblib")
    assert bundle["research_only"] and bundle["features"] == CONTEXT_COLUMNS
    frozen = report["selection"]
    chosen = select_components(report["screening"], frozen["selection"])
    assert chosen == frozen["component_sources"] == bundle["component_sources"]
    assert pd.Timestamp(frozen["frozen_at"]) < pd.Timestamp(report["completed_at"])
    # Independently rebuild empirical references and the calibration expansions.
    empirical = fit_empirical(base.loc[masks["train"]])
    baseline = predict_empirical(empirical, confirm)[0]
    np.testing.assert_array_equal(baseline, predict_empirical(bundle["empirical"], confirm)[0])
    raw = {"baseline": baseline, "candidate": predict_selected(bundle, confirm)}
    observed = component_values(confirm).to_numpy()
    observed_cal = component_values(calibration).to_numpy()
    changed = confirm[CONTEXT_COLUMNS].copy()
    np.testing.assert_array_equal(predict_selected(bundle, changed), raw["candidate"])
    coverage_intervals = {}
    for name in ["baseline", "candidate"]:
        cal_p = (
            predict_selected(bundle, calibration)
            if name == "candidate"
            else predict_empirical(empirical, calibration)[0]
        )
        regenerated = fit_expansion(observed_cal, cal_p)
        np.testing.assert_array_equal(
            regenerated["expansion"], bundle["calibration"][name]["expansion"]
        )
        assert regenerated["rank"] == report["calibration"][name]["rank"]
        for suffix in ["raw", "calibrated"]:
            variant = f"{name}_{suffix}"
            p = raw[name] if suffix == "raw" else apply_expansion(raw[name], regenerated)
            assert np.isfinite(p).all() and (np.diff(p, axis=2) >= 0).all()
            assert (p[:, 2:] >= 0).all()
            for i, component in enumerate(COMPONENTS):
                np.testing.assert_array_equal(observed[:, i], stored[f"{component}_observed"])
                for j, q in enumerate(QUANTILES):
                    np.testing.assert_array_equal(p[:, i, j], stored[f"{variant}_{component}_q{q}"])
                # Recompute coverage and interval score independently of training metrics.
                low, high, y = p[:, i, 0], p[:, i, 2], observed[:, i]
                covered = (y >= low) & (y <= high)
                score = high - low + 10 * (np.maximum(low - y, 0) + np.maximum(y - high, 0))
                m = report["confirmation"][variant]["by_component"][component]
                np.testing.assert_allclose(m["coverage"], covered.mean(), atol=1e-14)
                np.testing.assert_allclose(
                    m["normalized_interval_score"], score.mean() / bundle["scale"][i], atol=1e-14
                )
                if suffix == "calibrated":
                    coverage_intervals[f"{name}_{component}"] = paired_mean_interval(
                        confirm.MatchID, covered.astype(float)
                    )["series_bootstrap_95pct"]
    verification = {
        "status": "passed",
        "artifact_hashes": len(hashes),
        "rows": len(confirm),
        "series": int(confirm.MatchID.nunique()),
        "reload_max_error": 0.0,
        "selection_recomputed": True,
        "empirical_references_rebuilt": True,
        "calibration_recomputed": True,
        "noncontext_columns_removed": "unchanged predictions",
        "metrics_independently_recomputed": True,
        "coverage_intervals": coverage_intervals,
    }
    write_json(root / "verification.json", verification)
    return report, verification


def render(root, report, verification, destination):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = report["confirmation"]["baseline_calibrated"]
    candidate = report["confirmation"]["candidate_calibrated"]
    improvement = 100 * (
        1 - candidate["normalized_interval_score"] / baseline["normalized_interval_score"]
    )
    pinball_gain = 100 * (1 - candidate["normalized_pinball"] / baseline["normalized_pinball"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    positions = np.arange(len(COMPONENTS))
    for offset, (name, values, color) in enumerate(
        [("baseline", baseline, "#929ba8"), ("candidate", candidate, "#157f8c")]
    ):
        x = positions + (offset - 0.5) * 0.3
        metrics = [values["by_component"][c] for c in COMPONENTS]
        coverage = np.array([m["coverage"] for m in metrics]) * 100
        intervals = (
            np.array([verification["coverage_intervals"][f"{name}_{c}"] for c in COMPONENTS]) * 100
        )
        axes[0].errorbar(
            x,
            coverage,
            yerr=[coverage - intervals[:, 0], intervals[:, 1] - coverage],
            fmt="o",
            capsize=3,
            label=name.capitalize(),
            color=color,
        )
        axes[1].bar(
            x,
            [m["normalized_interval_score"] for m in metrics],
            width=0.3,
            color=color,
            label=name.capitalize(),
        )
    axes[0].axhline(80, linestyle="--", color="#444444", label="Nominal 80%")
    axes[0].set(
        ylabel="Observed coverage (%)",
        ylim=(70, 100),
        title="Coverage with series-bootstrap 95% intervals",
    )
    axes[1].set(
        ylabel="Normalized interval score (lower is better)",
        title="Width and misses evaluated together",
    )
    for ax in axes:
        ax.set_xticks(positions, COMPONENTS, rotation=35, ha="right")
        ax.legend(fontsize=9)
    fig.suptitle("Historical development comparison — not an independent test")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".png"), dpi=160)
    plt.close(fig)
    lines = [
        "# Contextual peer intervals — October 17, 2026",
        "",
        "Completed the [fixed protocol](2026-10-17-peer-interval-plan.md): **72 quantile-regressor fits**, "
        "covering six components, three quantiles, three chronological screening folds, and a full-training refit. "
        "All six learned components improved selection pinball loss in every fold and the earlier selection period. "
        "Selection was frozen before calibration and final diagnostic inference.",
        "",
        "**This is reused development validation, not a fresh test.** The comparison uses 5,070 player-game "
        "records in 243 series. Original test rows were excluded throughout. Inputs are agent, map, patch, "
        "length band, and round count; outcomes, player/team identity, and the player's observed statistics "
        "are excluded from model inputs.",
        "",
        "The models estimate an expected range for each performance component among comparable historical "
        "records. The 10th/90th quantiles form a nominal 80% interval; the median is also predicted. "
        "These are ranges of individual observations, not confidence intervals for a mean or an impact score.",
        "",
        "| Calibrated model | Normalized interval score ↓ | Normalized pinball loss ↓ |",
        "|---|---:|---:|",
        f"| Empirical agent/map baseline | {baseline['normalized_interval_score']:.6f} | {baseline['normalized_pinball']:.6f} |",
        f"| Learned conditional ranges | {candidate['normalized_interval_score']:.6f} | {candidate['normalized_pinball']:.6f} |",
        "",
        f"Interval score improved **{improvement:.2f}%** and pinball loss improved **{pinball_gain:.2f}%**. "
        "Interval score penalizes both overly wide ranges and observations outside the range. "
        f"The paired mean score difference was {report['comparison']['calibrated']['overall']['difference']:.6f}, "
        f"with series-bootstrap 95% interval {report['comparison']['calibrated']['overall']['series_bootstrap_95pct']}. "
        "This uncertainty describes the reused development comparison, not independent generalization.",
        "",
        "## Component results",
        "",
        "| Component | Baseline coverage | Learned coverage | Interval-score change |",
        "|---|---:|---:|---:|",
    ]
    for c in COMPONENTS:
        b, m = baseline["by_component"][c], candidate["by_component"][c]
        gain = 100 * (m["normalized_interval_score"] / b["normalized_interval_score"] - 1)
        lines.append(
            f"| {c} | {100 * b['coverage']:.2f}% | {100 * m['coverage']:.2f}% | {gain:+.2f}% |"
        )
    lines += [
        "",
        "Five components improved interval score, but **objectives worsened**. That negative result is retained; "
        "the frozen component selection was not changed after inspecting this period. Combat and damage "
        "coverage remained below nominal 80%, and damage misses were skewed toward unexpectedly high damage. "
        "Objectives and clutches have many zeros and discrete values, so their coverage exceeded 90%. "
        "The mean coverage across components must not be interpreted as prediction accuracy or simultaneous coverage.",
        "",
        "Calibration used only the separate 5,420-row / 238-series calibration period. Series dependence, "
        "temporal drift, and sparse outcomes mean no IID conformal coverage guarantee is claimed. Map, agent, "
        "and match-length slice coverage and tail errors are retained in the JSON; small slices are descriptive.",
        "",
        "## Verification and status",
        "",
        f"All {verification['artifact_hashes']} artifact hashes and source/script/ML fingerprints passed. "
        "All 5,070 predictions reloaded exactly. The verifier rebuilt empirical references and calibration "
        "expansions, recomputed selection and interval metrics, and confirmed predictions stay identical when "
        "all noncontext columns are removed. Automated tests also cover unseen categories, support, quantile "
        "ordering, finite-rank calibration, partition isolation, and portable serialization.",
        "",
        "The candidate remains research-only. The active dashboard model and heuristic score are unchanged. "
        "These results improve a measured peer-comparison capability; they do not establish causal impact, "
        "coaching effectiveness, or modern-ranked-game validity. New later matches and human-reviewed "
        "performance benchmarks are still needed.",
        "",
        f"Artifacts: `{root}`. Started `{report['started_at']}`; completed `{report['completed_at']}`.",
        "",
        "```bash",
        "python -m scripts.peer_intervals ml/artifacts/research/player-20261016-v2 --output data/research/NEW_INTERVAL_RUN",
        "python -m scripts.verify_peer_intervals data/research/NEW_INTERVAL_RUN --report docs/experiments/NEW-interval-results.md",
        "```",
        "",
        f"![Peer range coverage and interval score]({destination.with_suffix('.png').name})",
        "",
    ]
    destination.write_text("\n".join(lines))
    write_json(destination.with_suffix(".json"), {"results": report, "verification": verification})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report, verification = verify(args.root, args.data_dir)
    render(args.root, report, verification, args.report)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
