import numpy as np

from printvision.metrics import classification_metrics


def test_imbalanced_metrics_are_finite():
    result = classification_metrics([0, 0, 1, 1], [0, 0, 1, 0], np.array([[.9, .1], [.8, .2], [.2, .8], [.7, .3]]))
    assert result["balanced_accuracy"] == 0.75
    assert 0 <= result["macro_f1"] <= 1
