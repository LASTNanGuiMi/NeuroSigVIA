from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


def train_classifier(
    train_embeds, train_labels, test_embeds, test_labels, classifier_type, random_seed
):
    train, val = utils.get_split(train_embeds)

    clf = get_classifier(classifier_type, random_seed=random_seed)
    clf.fit(train_embeds[train], train_labels[train].ravel())

    acc_val = clf.score(train_embeds[val], train_labels[val].ravel())
    acc_test = clf.score(test_embeds, test_labels.ravel())

    return acc_val, acc_test
