"""Shared experiment harness for tasks 5-16.

Provides:
- load_data() with switchable feature groups (tasks 9, 10, 13)
- cv_scores(): fold-parallel CV via ONE persistent fork pool. loky/joblib
  pool respawns deadlocked intermittently on this box; a single pool created
  once from the single-threaded main process does not.
- evaluate() returning per-fold scores for paired tests (task 15)
- SeedAveragedClassifier (task 8)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# screening: cheap, for shortlists; half/full: 25/50 folds, for accept/reject decisions
CV_SCREEN = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
CV_HALF = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
CV_MAIN = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
CV_FRESH = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2024)

BASE_NUM = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
BASE_CAT = ["Sex", "Embarked", "Title", "Deck"]


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    return df


def load_data(ticket=False, fare_per_person=False, hygiene=False):
    """Load train/test with switchable feature groups for tasks 9, 10, 13."""
    train = add_base_features(pd.read_csv("data/train.csv"))
    test = add_base_features(pd.read_csv("data/test.csv"))

    if ticket:
        # group sizes from combined train+test ticket identifiers (no target info)
        def ticket_key(df):
            return df["Ticket"].str.replace(" ", "", regex=False)

        counts = ticket_key(pd.concat([train, test], ignore_index=True)).value_counts()
        for df in (train, test):
            df["TicketGroupSize"] = ticket_key(df).map(counts).astype(int)
            df["TicketPrefix"] = (
                df["Ticket"].str.replace(r"[0-9.\s]+$", "", regex=True)
                .str.rstrip("/.")
                .replace("", "none")
            )

    if hygiene:
        # Fare == 0 is not a real fare (crew/placeholder); let imputation handle it
        for df in (train, test):
            df.loc[df["Fare"] <= 0, "Fare"] = np.nan

    if fare_per_person:
        for df in (train, test):
            df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

    return train, test


# ---------------------------------------------------------------- evaluation
# Serial by design. Fit-level parallelism (OpenMP inside RF/HGB/XGB/LGBM/CatBoost)
# is plenty for 891-row data, and every multiprocessing scheme tried — loky
# (fork), loky (spawn), and a persistent fork pool — eventually lost workers
# silently and hung once the GBM libraries entered the workload. Robust beats
# fast at this scale.

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
    """Paired t-test on per-fold scores of two configs on identical folds (task 15)."""
    diff = np.mean(scores_a) - np.mean(scores_b)
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return diff, t_stat, p_value


class SeedAveragedClassifier(BaseEstimator, ClassifierMixin):
    """Fit n_seeds clones of a base estimator (each with a different random_state)
    and average predicted probabilities (task 8)."""

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
