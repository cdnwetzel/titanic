"""Task 7: Optuna hyperparameter search for HGB and RF.

Objective: 15-fold screening CV (fixed seed). Winners are re-validated on
FRESH 50-fold CV (different fold seed) against the untuned configs — this
guards against overfitting the search to one CV protocol. Best params are
saved to t7_params.json for use in the champion stack.
"""

import json
import warnings

warnings.filterwarnings("ignore")

import optuna
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from exputil import BASE_CAT, BASE_NUM, cv_scores, load_data, make_preprocessor, paired_test

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, _ = load_data()
y = train["Survived"]
X = train[BASE_NUM + BASE_CAT]
pre = make_preprocessor(BASE_NUM, BASE_CAT)


def make_hgb(params):
    return HistGradientBoostingClassifier(random_state=42, **params)


def make_rf(params):
    return RandomForestClassifier(random_state=42, **params)


def objective_hgb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1500),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
        "max_bins": trial.suggest_int("max_bins", 64, 255),
    }
    pipe = Pipeline([("preprocess", pre), ("clf", make_hgb(params))])
    return cv_scores(pipe, X, y, mode="screen").mean()


def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }
    pipe = Pipeline([("preprocess", pre), ("clf", make_rf(params))])
    return cv_scores(pipe, X, y, mode="screen").mean()


study_hgb = optuna.create_study(direction="maximize")
study_hgb.optimize(objective_hgb, n_trials=60)
print(f"Best HGB ({len(study_hgb.trials)} trials): {study_hgb.best_value:.4f}")
print("  params:", study_hgb.best_params)

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_rf, n_trials=40)
print(f"Best RF ({len(study_rf.trials)} trials): {study_rf.best_value:.4f}")
print("  params:", study_rf.best_params)

with open("t7_params.json", "w") as f:
    json.dump({"hgb": study_hgb.best_params, "rf": study_rf.best_params}, f, indent=2)

# Fresh-CV re-validation (task 7 caveat): different fold seed, paired test
HGB_BASE = {"max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "min_samples_leaf": 20, "l2_regularization": 1.0}
RF_BASE = {"n_estimators": 200}

for name, base_params, best_params, factory in [
    ("HGB", HGB_BASE, study_hgb.best_params, make_hgb),
    ("RF", RF_BASE, study_rf.best_params, make_rf),
]:
    untuned = Pipeline([("preprocess", pre), ("clf", factory(base_params))])
    tuned = Pipeline([("preprocess", pre), ("clf", factory(best_params))])
    s_un = cv_scores(untuned, X, y, mode="fresh")
    s_tu = cv_scores(tuned, X, y, mode="fresh")
    diff, t, p = paired_test(s_tu, s_un)
    print(f"FRESH 50-fold: {name} untuned {s_un.mean():.4f} | tuned {s_tu.mean():.4f} "
          f"| diff={diff:+.4f}, p={p:.4f}")
