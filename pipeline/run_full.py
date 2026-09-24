"""Full battery: baseline and tasks 1 to 5 reconstructed, then the gated
pipeline (tasks 6 to 14) via pipeline.run_all.

Tasks 15 and 16 are methodology and policy: task 15 is the paired-test
discipline embedded in every gate here, and task 16 (submission budget) is
deliberately manual.

Run from the repo root:
    OMP_NUM_THREADS=8 python -m pipeline.run_full 2>&1 | tee results/runner_full.log

Verify wiring cheaply first (small models, 15-fold CV, about 3 min):
    python -m pipeline.run_full --smoke

Events append to results/results.jsonl with host tags and per-fold arrays to
results/scores_full_*.npy. Expected values from the original run appear in
the log labels as "(expect ~X)".
"""

import json
import os
import socket
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.evaluate import cv_scores, paired_test
from src.features import BASE_CAT, BASE_NUM, GroupMedianImputer, load_data
from src.model import HGB_BASE, build_stack, make_member, make_pre

OUT_DIR = os.environ.get("RUN_OUT_DIR", "results")
HOST = socket.gethostname()
SMOKE = "--smoke" in sys.argv
MODE = "screen" if SMOKE else "full"
# In smoke mode shrink the expensive members so the wiring check finishes
# in minutes. TUNED merges over the BASE params.
TUNED_SMOKE = {"hgb": {"max_iter": 50}, "rf": {"n_estimators": 20}}
TUNED = TUNED_SMOKE if SMOKE else {"hgb": {}, "rf": {}}

os.makedirs(OUT_DIR, exist_ok=True)
JSONL = open(os.path.join(OUT_DIR, "results.jsonl"), "a", buffering=1)


def log(msg, **event):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)
    if event:
        JSONL.write(json.dumps({"t": stamp, "host": HOST, **event}) + "\n")


def record_scores(event, task, label, scores):
    log(f"{label}: {scores.mean():.4f} ({len(scores)} folds)", event=event, task=task,
        mean=float(scores.mean()), sem=float(scores.std() / np.sqrt(len(scores))))
    np.save(os.path.join(OUT_DIR, f"scores_full_{event}_t{task}.npy"), scores)


def eval_row(model, X, y, event, task, label):
    scores = cv_scores(model, X, y, mode=MODE)
    record_scores(event, task, label, scores)
    return scores


train, _ = load_data()
y = train["Survived"]

BASE_NUM5 = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
BASE_CAT2 = ["Sex", "Embarked"]


def pre_unscaled(num, cat):
    """Baseline preprocessing: no scaling, sparse one-hot."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


# ================================================================ BASELINE
log("BASELINE: RF 200, unscaled, 5 numeric + 2 categorical (expect ~0.8134)",
    event="task_start", task=0)
Xb = train[BASE_NUM5 + BASE_CAT2]
baseline = Pipeline([
    ("preprocess", pre_unscaled(BASE_NUM5, BASE_CAT2)),
    ("clf", RandomForestClassifier(random_state=42, **TUNED.get("rf", {}))),
])
s_base = eval_row(baseline, Xb, y, "full_baseline", 0, "Baseline RF (CV)")
BASELINE_SCORES = s_base

# ================================================================ TASK 1
log("TASK 1: engineered feature variants (expect: none significant vs baseline)",
    event="task_start", task=1)
pre_scaled = make_pre(BASE_NUM5, BASE_CAT2)


def rf_scaled(num, cat):
    return Pipeline([
        ("preprocess", make_pre(num, cat)),
        ("clf", RandomForestClassifier(random_state=42, **TUNED.get("rf", {}))),
    ])


t1 = train.copy()
t1["FamGroup"] = pd.cut(t1["FamilySize"], bins=[0, 1, 4, 15],
                        labels=["alone", "small", "large"])
t1["HasCabin"] = t1["Cabin"].notna().astype(int)

t1_rows = [
    ("base", BASE_NUM5, BASE_CAT2, "~0.8134"),
    ("title", BASE_NUM5, BASE_CAT2 + ["Title"], "~0.812"),
    ("family", BASE_NUM5 + ["FamilySize", "IsAlone"], BASE_CAT2, "~0.810"),
    ("deck", BASE_NUM5, BASE_CAT2 + ["Deck"], "~0.800"),
    ("famgroup", BASE_NUM5, BASE_CAT2 + ["FamGroup"], "~0.810"),
    ("hascabin", BASE_NUM5 + ["HasCabin"], BASE_CAT2, "~0.812"),
    ("title_family", BASE_NUM5 + ["FamilySize", "IsAlone"], BASE_CAT2 + ["Title"], "~0.803"),
    ("title_deck", BASE_NUM5, BASE_CAT2 + ["Title", "Deck"], "~0.804"),
    ("all", BASE_NUM5 + ["FamilySize", "IsAlone"], BASE_CAT2 + ["Title", "Deck"], "~0.801"),
]
for name, num, cat, expect in t1_rows:
    eval_row(rf_scaled(num, cat), t1[num + cat], y, f"full_t1_{name}", 1,
             f"Task 1 {name} (expect {expect})")
log("TASK 1 done.", event="task_done", task=1)

# ================================================================ TASK 2
log("TASK 2: group-based Age imputation (expect Title x Pclass ~0.8153 accepted)",
    event="task_start", task=2)
t2 = train.copy()
for name, keys, expect in [("by_title", ["Title"], "~0.8103"),
                           ("by_sex_pclass", ["Sex", "Pclass"], "~0.8139"),
                           ("by_title_pclass", ["Title", "Pclass"], "~0.8153")]:
    pipe = Pipeline([
        ("age_impute", GroupMedianImputer("Age", keys)),
        ("preprocess", make_pre(BASE_NUM5, BASE_CAT2)),
        ("clf", RandomForestClassifier(random_state=42, **TUNED.get("rf", {}))),
    ])
    s = eval_row(pipe, t2[BASE_NUM5 + BASE_CAT2 + ["Title"]], y, f"full_t2_{name}", 2,
                 f"Task 2 {name} (expect {expect})")
    diff, t, p = paired_test(s, BASELINE_SCORES)
    log(f"  vs baseline: diff={diff:+.4f} p={p:.4f}", event="gate_result", task=2,
        variant=name, diff=float(diff), p=float(p), decision=bool(diff > 0.001 and p < 0.05))
log("TASK 2 done.", event="task_done", task=2)

# ================================================================ TASK 3
log("TASK 3: HistGradientBoosting variants (expect ~0.8184 / 0.8192 / 0.8203)",
    event="task_start", task=3)
FULL_NUM = list(BASE_NUM)
FULL_CAT = list(BASE_CAT)


def hgb_pipe(num, cat, params=None):
    merged = dict(params or {})
    if SMOKE:
        merged.update(TUNED["hgb"])
    return Pipeline([
        ("preprocess", make_pre(num, cat, imputer="passthrough")),
        ("clf", HistGradientBoostingClassifier(random_state=42, **merged)),
    ])


hgb_rows = [
    ("default_base", BASE_NUM5, BASE_CAT2, None, "~0.8184"),
    ("default_full", FULL_NUM, FULL_CAT, None, "~0.8192"),
    ("tuned_full", FULL_NUM, FULL_CAT, HGB_BASE, "~0.8203"),
]
for name, num, cat, params, expect in hgb_rows:
    eval_row(hgb_pipe(num, cat, params), train[num + cat], y, f"full_t3_{name}", 3,
             f"Task 3 {name} (expect {expect})")
log("TASK 3 done.", event="task_done", task=3)

# ================================================================ TASK 4
log("TASK 4: soft-voting ensembles (expect ~0.8219 / 0.8210 / 0.8314 / 0.8303)",
    event="task_start", task=4)
Xf = train[FULL_NUM + FULL_CAT]


def member(name):
    return make_member(name, 42, tuned=TUNED)


def vote(names, weights=None):
    return Pipeline([
        ("preprocess", make_pre(FULL_NUM, FULL_CAT)),
        ("clf", VotingClassifier([(n, member(n)) for n in names], voting="soft",
                                 weights=weights)),
    ])


t4_rows = [
    ("rf_hgb", ["rf", "hgb"], None, "~0.8219"),
    ("rf_hgb_w12", ["rf", "hgb"], [1, 2], "~0.8210"),
    ("rf_hgb_lr", ["rf", "hgb", "lr"], None, "~0.8314"),
    ("rf_hgb_lr_w121", ["rf", "hgb", "lr"], [1, 2, 1], "~0.8303"),
]
for name, names, weights, expect in t4_rows:
    eval_row(vote(names, weights), Xf, y, f"full_t4_{name}", 4,
             f"Task 4 {name} (expect {expect})")
log("TASK 4 done.", event="task_done", task=4)

# ================================================================ TASK 5
log("TASK 5: stacking vs voting (expect ~0.8337 / 0.8365)",
    event="task_start", task=5)
spec5 = {"members": ["rf", "hgb", "lr"], "avg_seeds": {}, "ticket": False,
         "farepp": False, "imputer": "median", "target_enc": False, "hygiene": False}
for name, passthrough, expect in [("no_passthrough", False, "~0.8337"),
                                  ("passthrough", True, "~0.8365")]:
    eval_row(build_stack(spec=spec5, tuned=TUNED, passthrough=passthrough), Xf, y,
             f"full_t5_{name}", 5, f"Task 5 {name} (expect {expect})")
log("TASK 5 done.", event="task_done", task=5)

# ================================================================ TASKS 6-14
if SMOKE:
    log("SMOKE MODE: tasks 6-14 skipped (run without --smoke for the full battery)",
        event="task_start", task=6)
    JSONL.close()
    sys.exit(0)

log("TASKS 6-14: handing off to the gated pipeline", event="task_start", task=6)
JSONL.close()

from pipeline.run_all import main as run_gated_pipeline  # noqa: E402

run_gated_pipeline()
