from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import src.utils as utils


def get_classifier(classifier_type, random_seed):
    if classifier_type == "logistic_regression":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=random_seed),
        )
    elif classifier_type == "nearest_centroid":
        clf = make_pipeline(StandardScaler(), NearestCentroid())
    elif classifier_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=random_seed
        )
    else:
        raise ValueError(f"Unsupported classifier type {classifier_type}.")

    return clf


def _nearest_centroid_scores(clf, X):
    if len(clf) > 1:
        transformed = clf[:-1].transform(X)
    else:
        transformed = X

    centroids = clf[-1].centroids_
    distances = ((transformed[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)

    return -distances


def _final_estimator(clf):
    return clf[-1] if hasattr(clf, "steps") else clf


def get_class_scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)

    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        if scores.ndim == 1:
            scores = scores[:, None]
        return scores

    if isinstance(_final_estimator(clf), NearestCentroid):
        return _nearest_centroid_scores(clf, X)

    raise ValueError(
        f"Classifier {type(_final_estimator(clf)).__name__} does not provide class scores."
    )


def _macro_curve_metric(y_true, y_score, classes, metric_fn):
    values = []

    for class_idx, class_label in enumerate(classes):
        y_binary = (y_true == class_label).astype(int)
        if y_binary.min() == y_binary.max():
            continue
        values.append(metric_fn(y_binary, y_score[:, class_idx]))

    if not values:
        return float("nan")

    return sum(values) / len(values)


def compute_metrics_from_predictions(y_true, y_pred, y_score, classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_auroc": _macro_curve_metric(
            y_true, y_score, classes, roc_auc_score
        ),
        "macro_auprc": _macro_curve_metric(
            y_true, y_score, classes, average_precision_score
        ),
    }


def compute_metrics(clf, X, y):
    y_true = y.ravel()
    y_pred = clf.predict(X)
    classes = clf.classes_
    y_score = get_class_scores(clf, X)

    return compute_metrics_from_predictions(y_true, y_pred, y_score, classes)


def train_classifier(
    train_embeds,
    train_labels,
    test_embeds,
    test_labels,
    classifier_type,
    random_seed,
    val_ratio,
    val_embeds=None,
    val_labels=None,
):
    has_fixed_validation = val_embeds is not None or val_labels is not None
    if has_fixed_validation and (val_embeds is None or val_labels is None):
        raise ValueError("val_embeds and val_labels must be provided together")

    if has_fixed_validation:
        train = list(range(len(train_embeds)))
        val = list(range(len(val_embeds)))
        fit_embeds = train_embeds
        fit_labels = train_labels
        validation_embeds = val_embeds
        validation_labels = val_labels
    else:
        train, val = utils.get_split(
            train_embeds,
            frac=val_ratio,
            random_seed=random_seed,
        )
        fit_embeds = train_embeds[train]
        fit_labels = train_labels[train]
        validation_embeds = train_embeds[val]
        validation_labels = train_labels[val]

    clf = get_classifier(classifier_type, random_seed=random_seed)
    clf.fit(fit_embeds, fit_labels.ravel())

    val_metrics = compute_metrics(clf, validation_embeds, validation_labels)
    test_metrics = compute_metrics(clf, test_embeds, test_labels)

    return val_metrics, test_metrics, train, val
