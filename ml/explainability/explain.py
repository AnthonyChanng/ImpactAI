import numpy as np
import shap


def explain(model, frame, limit=None):
    """Contributions to the uncalibrated log-odds; never causal effects."""
    if limit:
        frame = frame.iloc[:limit]
    transformed = model.named_steps["pre"].transform(frame)
    estimator = model.named_steps["model"]
    names = model.named_steps["pre"].get_feature_names_out().tolist()
    if hasattr(estimator, "get_booster"):
        explainer = shap.TreeExplainer(estimator)
        values = np.asarray(explainer.shap_values(transformed))
        base = float(explainer.expected_value)
        method = "TreeSHAP (tree-path dependent)"
    else:
        # Training-centred linear terms: additive explanation, not mislabeled as TreeSHAP.
        values = transformed * estimator.coef_[0]
        base = float(estimator.intercept_[0])
        method = "Linear log-odds contributions (standardized training baseline)"
    return values, base, names, method
