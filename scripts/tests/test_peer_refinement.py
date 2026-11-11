import copy

import numpy as np

from ml.features.build import COMPONENTS
from scripts.refine_peer_benchmarks import predict_components, select_components


def result_fixture():
    result = {"screening": {}, "selection": {}}
    for name, error in [("hierarchical", 1.0), ("boosted_shallow", 0.9), ("boosted_medium", 0.8)]:
        result["screening"][name] = [{c: error for c in COMPONENTS} for _ in range(3)]
        result["selection"][name] = {"by_component": {c: error for c in COMPONENTS}}
    return result


def test_component_selection_requires_improvement_in_every_earlier_fold():
    result = result_fixture()
    result["screening"]["boosted_medium"][2]["Damage"] = 1.1
    result["selection"]["boosted_medium"]["by_component"]["Damage"] = 0.1
    assert select_components(result)["Damage"] == "boosted_shallow"
    assert select_components(result)["Combat"] == "boosted_medium"


def test_component_selection_retains_baseline_when_later_selection_worsens():
    result = result_fixture()
    for name in ["boosted_shallow", "boosted_medium"]:
        result["selection"][name]["by_component"]["Assists"] = 1.1
    assert select_components(result)["Assists"] == "hierarchical"


def test_confirmation_results_cannot_change_component_selection():
    result = result_fixture()
    expected = select_components(result)
    changed = copy.deepcopy(result)
    changed["confirmation"] = {"boosted_medium": {"normalized_mse": 1000}}
    assert select_components(changed) == expected


class ConstantMeans:
    def __init__(self, offset):
        self.offset = offset

    def predict(self, base):
        return np.tile(np.arange(6) + self.offset, (len(base), 1))


def test_composed_predictions_preserve_each_components_source_and_order():
    sources = {c: "a" if i % 2 else "b" for i, c in enumerate(COMPONENTS)}
    bundle = {
        "component_sources": sources,
        "models": {"a": ConstantMeans(0), "b": ConstantMeans(10)},
    }
    np.testing.assert_array_equal(
        predict_components(bundle, [1, 2]), [[10, 1, 12, 3, 14, 5], [10, 1, 12, 3, 14, 5]]
    )
