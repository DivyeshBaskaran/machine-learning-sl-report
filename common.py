"""Shared evaluation harness for the BRFSS GenHlth supervised learning report.

Every algorithm notebook (Decision Tree, kNN, SVM, NN-sklearn, NN-PyTorch) imports
from here so the split, preprocessing, tuning protocol, and metrics are identical
everywhere -- built and justified step by step in harness.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report,
)

SEED = 42
TARGET = "GenHlth"
CLASSES = [1, 2, 3, 4, 5]

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
CV_SCORING = "f1_macro"


def load_split():
    """Reload the sample and reconstruct the exact train/test split saved by harness.ipynb."""
    sample = pd.read_csv("data/sample_20k.csv")
    features = [c for c in sample.columns if c != TARGET]
    X = sample[features]
    y = sample[TARGET].astype(int)

    train_idx = np.load("results/train_idx.npy")
    test_idx = np.load("results/test_idx.npy")

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    return X_train, X_test, y_train, y_test


def scaled_pipeline(estimator):
    """Wrap a scale-sensitive estimator (kNN, SVM, NN) so preprocessing is refit
    on whatever training fold .fit() sees -- never on validation/test data.
    Decision Trees skip this; their splits are invariant to monotonic scaling."""
    return make_pipeline(StandardScaler(), estimator)


def summarize(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_F1": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }


def confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return pd.DataFrame(cm,
                         index=[f"true_{c}" for c in CLASSES],
                         columns=[f"pred_{c}" for c in CLASSES])


def per_class(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=CLASSES,
                                 output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.loc[[str(c) for c in CLASSES],
                                    ["precision", "recall", "f1-score", "support"]]


def full_report(name, y_true, y_pred):
    """The one metrics function every algorithm notebook calls for its final
    test-set evaluation: summary scores, confusion matrix, per-class table."""
    return summarize(name, y_true, y_pred), confusion(y_true, y_pred), per_class(y_true, y_pred)
