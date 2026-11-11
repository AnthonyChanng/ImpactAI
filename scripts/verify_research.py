"""Verify a trusted local improvement run without accessing original test rows for prediction."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from ml.explainability.explain import explain
from ml.features.build import COMPONENTS, player_features, swap_round
from ml.features.history import recent_round_features
from ml.improve import write_json
from ml.models.inference import probabilities
from ml.preprocessing.audit import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/source"))
    args = parser.parse_args()
    root = args.directory
    checksums = json.loads((root / "checksums.json").read_text())
    report = json.loads((root / "experiment.json").read_text())
    for task in report["results"]:
        if f"{task}-candidate.joblib" not in checksums:
            raise ValueError("Missing candidate checksum")
    for name, expected in checksums.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe artifact path")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    code_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path("ml").rglob("*.py")))
    ).hexdigest()
    if code_hash != report["code_hash"]:
        raise ValueError("ML source differs from the experiment's recorded implementation")
    for name, expected in report["source_hashes"].items():
        if hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source mismatch: {name}")
    frames = load_data(args.data_dir)
    frames = {
        name: frame[frame.Split.eq("validation")].reset_index(drop=True)
        for name, frame in frames.items()
    }
    verification = {
        "status": "passed",
        "manifest_files_checked": len(checksums),
        "ml_source_hash_matches": True,
        "prediction_partition": "validation only",
        "tasks": {},
    }
    for task, result in report["results"].items():
        bundle = joblib.load(root / f"{task}-candidate.joblib")
        if not bundle["research_only"] or bundle["selected"] != result["selected"]:
            raise ValueError("Research bundle identity mismatch")
        rows = frames["rounds" if task == "round" else "players"]
        base = recent_round_features(rows) if task == "round" else player_features(rows)
        key = ["GameID", "RoundNumber" if task == "round" else "PlayerID"]
        frozen = pd.read_parquet(root / f"{task}-confirmation.parquet")
        assert not frozen.duplicated(key).any()
        assert len(frozen) == result["partition_rows"]["confirmation"]
        assert frozen.Date.min() >= pd.Timestamp(report["time_boundaries"]["calibrate_end"])
        assert set(frozen.MatchID).issubset(set(rows.MatchID))
        errors = {}
        for name, saved in bundle["models"].items():
            x = base.copy()
            if task == "round" and "team_history" in saved["state"]:
                history = saved["state"]["team_history"]
                lookup = history.transform(frames["games"])
                x = pd.concat([x, lookup.reindex(rows.GameID).set_axis(rows.index)], axis=1)
            if task == "player" and "context_reference" in saved["state"]:
                z, _, _ = saved["state"]["context_reference"].standardize(base)
                x[[f"ContextZ_{component}" for component in COMPONENTS]] = z
            assert saved["features"] == saved["model"].feature_names_in_.tolist()
            x = x[saved["features"]]
            # Align by observation keys, never by incidental row positions.
            lookup = rows[key].copy()
            lookup["row_index"] = rows.index
            aligned = frozen[key].merge(lookup, on=key, how="left", validate="one_to_one")
            assert aligned.row_index.notna().all()
            x = x.loc[aligned.row_index.to_numpy()]
            p = probabilities(saved["model"], x, task == "round", saved["temperature"])
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            label = "candidate" if name == result["selected"] else "baseline"
            np.testing.assert_allclose(p, frozen[label], atol=1e-6, rtol=0)
            errors[name] = float(np.max(np.abs(p - frozen[label].to_numpy())))
            if task == "round":
                reverse = probabilities(
                    saved["model"], swap_round(x.iloc[:1000]), True, saved["temperature"]
                )
                np.testing.assert_allclose(reverse + p[:1000], 1, atol=1e-6)
            elif name == result["selected"]:
                positions = np.linspace(0, len(x) - 1, min(512, len(x)), dtype=int)
                values, baseline, names, method = explain(saved["model"], x.iloc[positions])
                explained = expit((baseline + values.sum(axis=1)) / saved["temperature"])
                np.testing.assert_allclose(explained, p[positions], atol=1e-5, rtol=0)
                importance = pd.DataFrame(
                    {
                        "feature": names,
                        "mean_absolute_log_odds_contribution": np.abs(values).mean(axis=0),
                    }
                ).sort_values("mean_absolute_log_odds_contribution", ascending=False)
                importance.to_csv(root / "player-global-explanations.csv", index=False)
                verification["player_explanation_check"] = {
                    "rows": len(positions),
                    "method": method,
                    "max_probability_error": float(np.max(np.abs(explained - p[positions]))),
                }
        verification["tasks"][task] = {
            "confirmation_rows": len(frozen),
            "max_reloaded_prediction_errors": errors,
        }
    write_json(root / "verification.json", verification)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(report["results"]),
        figsize=(6 * len(report["results"]), 5),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (task, result) in zip(axes[0], report["results"].items()):
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        for name, value in result["confirmation"].items():
            curve = value["overall"]["calibration"]
            label = "Selected candidate" if name == result["selected"] else "Original specification"
            ax.plot(
                [p["predicted"] for p in curve], [p["observed"] for p in curve], "o-", label=label
            )
        ax.set(
            title=f"{task.title()}: development confirmation",
            xlabel="Predicted probability",
            ylabel="Observed fraction",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend()
    fig.suptitle("Previously inspected historical validation; not an independent new test")
    fig.savefig(root / "development-calibration.png", dpi=150)
    plt.close(fig)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
