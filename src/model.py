"""Model zoo, preprocessing, and champion stack construction.

Tuned parameters come from the task 7 Optuna search and were validated on a
fresh 50 fold protocol. CHAMPION_TUNED is what train.py ships; the pipeline
runner starts from empty overrides and refills them through its own gates.
"""

import functools

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.evaluate import SeedAveragedClassifier

HGB_BASE = {"max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "min_samples_leaf": 20, "l2_regularization": 1.0}
RF_BASE = {"n_estimators": 200}

# Task 7 Optuna winners, fresh CV validated (HGB 0.8320 vs 0.8176, RF 0.8305
# vs 0.8085, both p < 0.0001). Shipped by train.py.
CHAMPION_TUNED = {
    "hgb": {"learning_rate": 0.02220063792187545, "max_iter": 1165, "max_leaf_nodes": 8,
            "min_samples_leaf": 44, "l2_regularization": 4.0175271118619635, "max_bins": 77},
    "rf": {"n_estimators": 661, "max_depth": 15, "min_samples_leaf": 3,
           "max_features": 0.44766588900099735},
}

# Runner state starts empty; task 7 refills these through the fresh-CV gate
TUNED = {"hgb": {}, "rf": {}}


def make_member(name, seed, tuned=None):
    tuned = tuned if tuned is not None else TUNED
    if name == "rf":
        return RandomForestClassifier(random_state=seed, **{**RF_BASE, **tuned["rf"]})
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, **{**HGB_BASE, **tuned["hgb"]})
    if name == "lr":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "xgb":
        return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                             random_state=seed)
    if name == "cat":
        return CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                  verbose=0, random_state=seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32,), alpha=0.5, max_iter=2000,
                             random_state=seed)
    if name == "svc":
        return SVC(probability=True, random_state=seed)
    if name == "lgbm":
        return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15,
                              min_child_samples=20, random_state=seed, verbose=-1)
    raise ValueError(name)


def member_factory(name, tuned=None):
    return functools.partial(make_member, name, tuned=tuned)


def make_pre(num, cat, imputer="median", target_enc=False):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    if imputer == "iterative":
        imp = IterativeImputer(random_state=42)
    elif imputer == "passthrough":
        imp = "passthrough"  # for learners that handle NaN natively (task 3)
    else:
        imp = SimpleImputer(strategy="median")
    branches = [
        ("num", Pipeline([("impute", imp), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ]
    if target_enc:
        branches.append(("te", TargetEncoder(target_type="binary"), ["TicketPrefix"]))
    return ColumnTransformer(branches)


DEFAULT_SPEC = {
    "members": ["rf", "hgb", "lr"],
    "avg_seeds": {},          # task 8: member -> n_seeds
    "ticket": False,          # task 9
    "farepp": False,          # task 10 (requires ticket)
    "imputer": "median",      # task 11
    "target_enc": False,      # task 12 (requires ticket)
    "hygiene": False,         # task 13
}


def columns(spec=None):
    from src.features import BASE_CAT, BASE_NUM
    spec = spec or DEFAULT_SPEC
    num = list(BASE_NUM)
    cat = list(BASE_CAT)
    if spec["ticket"]:
        num.append("TicketGroupSize")
        cat.append("TicketPrefix")
    if spec["farepp"]:
        num.append("FarePerPerson")
    return num, cat


def build_stack(spec=None, tuned=None, passthrough=True):
    spec = spec or DEFAULT_SPEC
    num, cat = columns(spec)
    pre = make_pre(num, cat, imputer=spec["imputer"], target_enc=spec["target_enc"])
    estimators = []
    for name in spec["members"]:
        if spec["avg_seeds"].get(name):
            est = SeedAveragedClassifier(member_factory(name, tuned),
                                         n_seeds=spec["avg_seeds"][name])
        else:
            est = make_member(name, 42, tuned=tuned)
        estimators.append((name, est))
    return Pipeline([
        ("preprocess", pre),
        ("clf", StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, stack_method="predict_proba", passthrough=passthrough)),
    ])
