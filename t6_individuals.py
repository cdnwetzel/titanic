"""Task 6 step 1: screen candidate base learners (15-fold screen)."""

import warnings

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from exputil import BASE_CAT, BASE_NUM, evaluate, load_data, make_preprocessor

train, _ = load_data()
y = train["Survived"]
X = train[BASE_NUM + BASE_CAT]


def build(clf):
    return Pipeline([("preprocess", make_preprocessor(BASE_NUM, BASE_CAT)), ("clf", clf)])


candidates = {
    "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15,
                               min_child_samples=20, random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6,
                                   verbose=0, random_state=42),
    "MLP small": MLPClassifier(hidden_layer_sizes=(32,), alpha=0.5, max_iter=2000, random_state=42),
    "SVC rbf": SVC(probability=True, random_state=42),
}

evaluate(build(RandomForestClassifier(n_estimators=200, random_state=42)), X, y, "ref: RF")
evaluate(build(LogisticRegression(max_iter=1000, random_state=42)), X, y, "ref: LR")
evaluate(build(HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)), X, y, "ref: HGB")
for name, clf in candidates.items():
    evaluate(build(clf), X, y, f"Task 6 candidate: {name}")
