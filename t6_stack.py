"""Task 6 step 2: greedy stack composition.

Champion = task-5 stack (RF+HGB+LR, LR meta, passthrough). Try adding each
candidate base; keep additions that beat the champion in a paired t-test
(p < 0.05) on 50 identical CV folds.
"""

import warnings

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from exputil import BASE_CAT, BASE_NUM, evaluate, load_data, make_preprocessor, paired_test

train, _ = load_data()
y = train["Survived"]
X = train[BASE_NUM + BASE_CAT]

rf = ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
hgb = ("hgb", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=20, l2_regularization=1.0, random_state=42))
lr = ("lr", LogisticRegression(max_iter=1000, random_state=42))

candidates = {
    "xgb": ("xgb", XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=3,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)),
    "cat": ("cat", CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0, random_state=42)),
    "mlp": ("mlp", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.5, max_iter=2000, random_state=42)),
    "svc": ("svc", SVC(probability=True, random_state=42)),
    "lgbm": ("lgbm", LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15,
                                    min_child_samples=20, random_state=42, verbose=-1)),
}


def stack(bases):
    return Pipeline([
        ("preprocess", make_preprocessor(BASE_NUM, BASE_CAT)),
        ("clf", StackingClassifier(estimators=bases,
                                   final_estimator=LogisticRegression(max_iter=1000),
                                   cv=5, stack_method="predict_proba", passthrough=True)),
    ])


champ_bases = [rf, hgb, lr]
s_champ = evaluate(stack(champ_bases), X, y, "Champion: stack RF+HGB+LR+pt", cv="half")

accepted = list(champ_bases)
current = s_champ
for name, est in candidates.items():
    trial = accepted + [est]
    s_trial = evaluate(stack(trial), X, y, f"  + {name}", cv="half")
    diff, t, p = paired_test(s_trial, current)
    verdict = "ACCEPT" if (diff > 0 and p < 0.05) else "reject"
    print(f"    diff={diff:+.4f}, paired t={t:.2f}, p={p:.4f} -> {verdict}", flush=True)
    if verdict == "ACCEPT":
        accepted.append(est)
        current = s_trial

print("\nFinal stack members:", [n for n, _ in accepted])
