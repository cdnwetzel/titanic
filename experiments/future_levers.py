"""Future levers, filtered to honest ones: each has a mechanism by which it
could genuinely improve the champion, not merely re-measure it.

Gated against the shipped champion (tuned RF+HGB+LR stack) with the
standard rule: paired t test on 50 identical folds, accept only if
diff > 0.001 and p < 0.05. Nine gates at alpha 0.05 carry about 0.4
expected false accepts; anything that passes gets fresh-CV confirmation
before being believed (task 7 discipline).

Run from the repo root (about 30 min on a Ryzen 5950X):
    OMP_NUM_THREADS=8 python experiments/future_levers.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import socket
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (ExtraTreesClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.evaluate import cv_scores, paired_test
from src.features import load_data
from src.model import CHAMPION_TUNED, columns, make_member, make_pre

HOST = socket.gethostname()
JSONL = open("results/results.jsonl", "a", buffering=1)


def log(msg, **event):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)
    if event:
        JSONL.write(json.dumps({"t": stamp, "host": HOST, **event}) + "\n")


train, _ = load_data()
NUM, CAT = columns()
X, y = train[NUM + CAT], train["Survived"]


def stack(members, meta=None, cv=5):
    return Pipeline([
        ("preprocess", make_pre(NUM, CAT)),
        ("clf", StackingClassifier(
            estimators=members,
            final_estimator=meta or LogisticRegression(max_iter=1000),
            cv=cv, stack_method="predict_proba", passthrough=True)),
    ])


def tuned_members():
    return [(n, make_member(n, 42, CHAMPION_TUNED)) for n in ["rf", "hgb", "lr"]]


champion = stack(tuned_members())
s_champ = cv_scores(champion, X, y, mode="full")
log(f"champion: {s_champ.mean():.4f} (50 folds)", event="levers_champion",
    mean=float(s_champ.mean()), sem=float(s_champ.std() / np.sqrt(len(s_champ))))
np.save("results/scores_levers_champion.npy", s_champ)

levers = {}

# 1. ExtraTrees: more decorrelated bagging than RF, different error surface
levers["et_member"] = stack(tuned_members() + [
    ("et", ExtraTreesClassifier(n_estimators=500, random_state=42))])

# 2. KNN: maximally different inductive bias, wrong in new ways
levers["knn_member"] = stack(tuned_members() + [
    ("knn", KNeighborsClassifier(n_neighbors=15))])

# 3. Calibrated RF/HGB: probability quality into the meta learner
levers["calibrated_members"] = stack([
    ("rf", CalibratedClassifierCV(make_member("rf", 42, CHAMPION_TUNED), cv=3)),
    ("hgb", CalibratedClassifierCV(make_member("hgb", 42, CHAMPION_TUNED), cv=3)),
    make_member("lr", 42, CHAMPION_TUNED),
][0:2] + [("lr", make_member("lr", 42, CHAMPION_TUNED))])

# 4. ElasticNet meta: proper regularization on 25+ passthrough features
levers["meta_elasticnet"] = stack(tuned_members(), meta=LogisticRegression(
    penalty="elasticnet", solver="saga", max_iter=8000, l1_ratio=0.5,
    random_state=42))

# 5. AgeBand x Sex: the child survival mechanism, beyond Title's proxies
t5 = train.copy()
t5["AgeBand"] = pd.cut(t5["Age"], [0, 12, 18, 60, 120],
                       labels=["child", "teen", "adult", "elder"]).astype(object).fillna("unknown")
t5["SexAgeBand"] = t5["Sex"].astype(str) + "_" + t5["AgeBand"].astype(str)
pre5 = make_pre(NUM, CAT + ["AgeBand", "SexAgeBand"])
levers["ageband_feature"] = Pipeline([
    ("preprocess", pre5),
    ("clf", StackingClassifier(tuned_members(),
                               final_estimator=LogisticRegression(max_iter=1000),
                               cv=5, stack_method="predict_proba", passthrough=True)),
])
X5 = t5[NUM + CAT + ["AgeBand", "SexAgeBand"]]

# 6. log1p(Fare): heavy-tailed fare enters the linear path on a sane scale
t6 = train.copy()
t6["LogFare"] = np.log1p(t6["Fare"])
pre6 = make_pre(NUM + ["LogFare"], CAT)
levers["logfare_feature"] = Pipeline([
    ("preprocess", pre6),
    ("clf", StackingClassifier(tuned_members(),
                               final_estimator=LogisticRegression(max_iter=1000),
                               cv=5, stack_method="predict_proba", passthrough=True)),
])
X6 = t6[NUM + CAT + ["LogFare"]]

# 7. Weighted vote: a different combiner than the stack gate already tried.
# Each vote member is a full pipeline so CatBoost receives encoded features
# exactly as it did inside the stack gates.
cat_pipe = Pipeline([
    ("preprocess", make_pre(NUM, CAT)),
    ("clf", make_member("cat", 42)),
])
levers["vote_stack_cat_11"] = VotingClassifier(
    [("stack", champion), ("cat", cat_pipe)], voting="soft")
levers["vote_stack_cat_21"] = VotingClassifier(
    [("stack", champion), ("cat", cat_pipe)], voting="soft", weights=[2, 1])

for name, model in levers.items():
    Xd = {"ageband_feature": X5, "logfare_feature": X6}.get(name, X)
    s = cv_scores(model, Xd, y, mode="full")
    diff, t, p = paired_test(s, s_champ)
    accept = bool(diff > 0.001 and p < 0.05)
    log(f"{name}: {s.mean():.4f} diff={diff:+.4f} p={p:.4f} -> "
        f"{'ACCEPT (needs fresh-CV confirmation)' if accept else 'reject'}",
        event="lever_gate", lever=name, mean=float(s.mean()), diff=float(diff),
        p=float(p), decision=accept)
    np.save(f"results/scores_lever_{name}.npy", s)

print("DONE_LEVERS", flush=True)
