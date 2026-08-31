#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import compute_metrics, get_classifier  # noqa: E402


def run_case(class_count):
    rng = np.random.default_rng(42 + class_count)
    labels = np.repeat(np.arange(class_count), 20)
    centers = np.eye(class_count, 6, dtype=np.float64) * 4.0
    features = np.vstack(
        [centers[label] + rng.normal(0.0, 0.3, size=6) for label in labels]
    )
    for classifier_type in (
        "logistic_regression",
        "nearest_centroid",
        "random_forest",
    ):
        classifier = get_classifier(classifier_type, random_seed=42)
        classifier.fit(features, labels)
        metrics = compute_metrics(classifier, features, labels)
        assert set(metrics) == {
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "macro_auroc",
            "macro_auprc",
        }
        assert np.isfinite(list(metrics.values())).all()
        assert metrics["accuracy"] > 0.9


def main():
    run_case(2)
    run_case(3)
    print("CLASSIFIER METRIC VALIDATION PASSED")


if __name__ == "__main__":
    main()
