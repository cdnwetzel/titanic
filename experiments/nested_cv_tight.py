"""Tighter nested CV: 25 outer folds x 50-trial inner HPO per fold.

Same protocol as task 14, scaled up to cut the SEM on the honest estimate
from about 0.007 to about 0.004. Runtime about 80 min on a Ryzen 5950X.
Appends a nested_cv_tight event to results/results.jsonl.

Run from the repo root:
    OMP_NUM_THREADS=8 python experiments/nested_cv_tight.py
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
import optuna
from sklearn.model_selection import RepeatedStratifiedKFold

from src.evaluate import cv_scores
from src.features import load_data
from src.model import TUNED, build_stack, columns
from pipeline.run_all import make_objective

HOST = socket.gethostname()
JSONL = open("results/results.jsonl", "a", buffering=1)


def log(msg, **event):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)
    if event:
        JSONL.write(json.dumps({"t": stamp, "host": HOST, **event}) + "\n")


SPEC = {"members": ["rf", "hgb", "lr"], "avg_seeds": {}, "ticket": False,
        "farepp": False, "imputer": "median", "target_enc": False, "hygiene": False}

optuna.logging.set_verbosity(optuna.logging.WARNING)

train, _ = load_data()
num, cat = columns(SPEC)
X, y = train[num + cat], train["Survived"]

OUTER = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)
outer_scores = []
snapshot = {k: dict(v) for k, v in TUNED.items()}
for k, (tr, te) in enumerate(OUTER.split(X, y)):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]
    inner = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    inner.optimize(make_objective("hgb", X_tr, y_tr, num, cat), n_trials=50)
    TUNED["hgb"] = dict(inner.best_params)
    model = build_stack(spec=SPEC, tuned=TUNED)
    model.fit(X_tr, y_tr)
    outer_scores.append(model.score(X_te, y_te))
    log(f"  outer fold {k}: {outer_scores[-1]:.4f}", event="nested_tight_fold",
        fold=k, score=float(outer_scores[-1]))
TUNED.clear()
TUNED.update(snapshot)
outer_scores = np.array(outer_scores)
sem = outer_scores.std() / np.sqrt(len(outer_scores))
log(f"NESTED CV TIGHT: {outer_scores.mean():.4f} (+/- {sem:.4f} SEM, {len(outer_scores)} folds)",
    event="nested_cv_tight", mean=float(outer_scores.mean()), sem=float(sem))
np.save("results/scores_nested_cv_tight.npy", outer_scores)
print("DONE_NESTED_TIGHT", flush=True)
