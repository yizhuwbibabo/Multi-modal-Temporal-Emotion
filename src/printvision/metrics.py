"""Metrics used in the article-style binary/multiclass defect protocol."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, probabilities=None) -> dict[str, float]:
    """Return metrics that remain meaningful under class imbalance."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if probabilities is not None:
        probabilities = np.asarray(probabilities)
        try:
            if probabilities.ndim == 1 or probabilities.shape[1] == 2:
                score = probabilities if probabilities.ndim == 1 else probabilities[:, 1]
                result["roc_auc"] = float(roc_auc_score(y_true, score))
            else:
                result["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, probabilities, multi_class="ovr")
                )
        except ValueError:  # one-class validation folds have no defined AUC
            result["roc_auc"] = float("nan")
    return result
