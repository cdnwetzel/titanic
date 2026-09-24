"""Evaluation harness behavior."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import cv_scores, paired_test


def test_paired_test_identical_is_null():
    a = np.array([0.8, 0.82, 0.79, 0.81, 0.8])
    diff, t, p = paired_test(a, a.copy())
    assert diff == 0.0
    # ttest_rel on identical arrays returns NaN (0/0): no evidence of difference
    assert np.isnan(p) or p == 1.0


def test_paired_test_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0.84, 0.02, 50)
    b = a - 0.01
    diff, t, p = paired_test(a, b)
    assert diff > 0
    assert p < 0.01


def test_cv_scores_is_deterministic():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=120, n_features=6, random_state=7)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression())])
    s1 = cv_scores(model, X, y, mode="screen")
    s2 = cv_scores(model, X, y, mode="screen")
    assert len(s1) == 15
    np.testing.assert_array_equal(s1, s2)
