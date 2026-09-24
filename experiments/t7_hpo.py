"""Experiment: Optuna HPO for HGB and RF (task 7).

Standalone lever: adjust the search spaces below and rerun to explore new
regions, or raise n_trials. Winners should still pass the fresh-CV gate
before entering the champion. Run:
    OMP_NUM_THREADS=8 python experiments/t7_hpo.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")

import optuna
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import cv_scores, paired_test
from src.features import BASE_CAT, BASE_NUM, load_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, _ = load_data()
y = train["Survived"]
X = train[BASE_NUM + BASE_CAT]

PRE = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), BASE_NUM),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), BASE_CAT),
])

HGB_BASE = {"max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "min_samples_leaf": 20, "l2_regularization": 1.0}
RF_BASE = {"n_estimators": 200}


def objective_hgb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1500),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
    }
    pipe = Pipeline([("preprocess", PRE),
                     ("clf", HistGradientBoostingClassifier(random_state=42, **params))])
    return cv_scores(pipe, X, y, mode="screen").mean()


def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }
    pipe = Pipeline([("preprocess", PRE),
                     ("clf", RandomForestClassifier(random_state=42, **params))])
    return cv_scores(pipe, X, y, mode="screen").mean()


study_hgb = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
study_hgb.optimize(objective_hgb, n_trials=60)
print(f"Best HGB (screen {study_hgb.best_value:.4f}): {study_hgb.best_params}", flush=True)

study_rf = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=42))
study_rf.optimize(objective_rf, n_trials=40)
print(f"Best RF (screen {study_rf.best_value:.4f}): {study_rf.best_params}", flush=True)

for name, study, base in [("hgb", study_hgb, HGB_BASE), ("rf", study_rf, RF_BASE)]:
    factory = HistGradientBoostingClassifier if name == "hgb" else RandomForestClassifier
    untuned = Pipeline([("preprocess", PRE), ("clf", factory(random_state=42, **base))])
    tuned = Pipeline([("preprocess", PRE), ("clf", factory(random_state=42, **study.best_params))])
    s_un = cv_scores(untuned, X, y, mode="fresh")
    s_tu = cv_scores(tuned, X, y, mode="fresh")
    diff, t, p = paired_test(s_tu, s_un)
    print(f"FRESH 50-fold {name}: untuned {s_un.mean():.4f} tuned {s_tu.mean():.4f} "
          f"diff={diff:+.4f} p={p:.4f}", flush=True)
