"""Evaluation harness: CV protocols, fold scoring, paired tests.

Serial by design. Fit level parallelism (OpenMP inside the GBM libraries)
is enough at 891 rows, and every multiprocessing scheme tried (loky fork,
loky spawn, a persistent fork pool) eventually lost workers silently once
XGBoost, LightGBM, or CatBoost entered the workload. Robust beats fast here.

Gate rule used across the project: accept a change only if the mean paired
improvement is greater than 0.001 and the paired t test p value is under
0.05, both computed on CV_MAIN per fold scores from identical folds.
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.model_selection import RepeatedStratifiedKFold

# screening: cheap, for shortlists; half/full: 25/50 folds, for decisions
CV_SCREEN = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
CV_HALF = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
CV_MAIN = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
CV_FRESH = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2024)


def _cv_fit_score(payload):
    model, X, y, tr, te = payload
    return clone(model).fit(X.iloc[tr], y.iloc[tr]).score(X.iloc[te], y.iloc[te])


def cv_scores(model, X, y, mode="full", n_jobs=1):
    cv = {"screen": CV_SCREEN, "half": CV_HALF, "full": CV_MAIN, "fresh": CV_FRESH}[mode]
    payloads = [(model, X, y, tr, te) for tr, te in cv.split(X, y)]
    scores = [_cv_fit_score(p) for p in payloads]
    return np.asarray(scores, dtype=float)


def evaluate(model, X, y, label="", cv="full"):
    scores = cv_scores(model, X, y, mode=cv)
    sem = scores.std() / np.sqrt(len(scores))
    print(f"{label:50s} {scores.mean():.4f} (+/- {sem:.4f} SEM) [{len(scores)} folds]", flush=True)
    return scores


def paired_test(scores_a, scores_b):
    """Paired t test on per fold scores of two configs on identical folds."""
    diff = np.mean(scores_a) - np.mean(scores_b)
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return diff, t_stat, p_value


class SeedAveragedClassifier(BaseEstimator, ClassifierMixin):
    """Fit n_seeds clones of a base estimator (each with a different
    random_state) and average predicted probabilities (task 8 lever)."""

    def __init__(self, factory, n_seeds=10):
        self.factory = factory  # callable(seed) -> unfitted estimator
        self.n_seeds = n_seeds

    def fit(self, X, y):
        self.models_ = [clone(self.factory(seed)).fit(X, y) for seed in range(self.n_seeds)]
        self.classes_ = self.models_[0].classes_
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
