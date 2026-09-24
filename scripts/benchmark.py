"""Standardized benchmark battery for cross-machine comparison.

Runs the same fixed evaluations on any machine and writes a JSON artifact
to results/. Commit the artifact; compare across machines by diffing.

Battery:
1. Champion 5-fold CV (identical protocol to train.py's printout)
2. Tuned HGB alone on the 15-fold screen protocol
3. Tuned RF alone on the 15-fold screen protocol

Scores should match across machines to within BLAS noise (watch the third
decimal). Wall times are the hardware signal. Run from the repo root:

    OMP_NUM_THREADS=8 python scripts/benchmark.py

Optionally set OMP_NUM_THREADS to the machine's physical core count before
comparing timings; scores are unaffected.
"""

import json
import pathlib
import socket
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import env_info
from src.evaluate import cv_scores
from src.features import load_data
from src.model import (
    CHAMPION_TUNED,
    HGB_BASE,
    RF_BASE,
    build_stack,
    columns,
    make_pre,
)


def timed(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.1f}s", flush=True)
    return result, elapsed


def main():
    host = socket.gethostname()
    train, _ = load_data()
    num, cat = columns()
    X, y = train[num + cat], train["Survived"]
    pre = make_pre(num, cat)

    report = {
        "hostname": host,
        "captured_by": "scripts/benchmark.py",
        "environment": env_info.gather(),
        "note": "Scores comparable across machines to BLAS noise; timings are the hardware signal.",
        "battery": {},
    }

    champion = build_stack(tuned=CHAMPION_TUNED)
    scores, elapsed = timed(
        "champion 5-fold CV",
        lambda: cross_val_score(champion, X, y, cv=5, scoring="accuracy"),
    )
    report["battery"]["champion_5fold_cv"] = {
        "wall_seconds": round(elapsed, 2),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "fold_scores": [float(s) for s in scores],
    }
    np.save(f"results/benchmark_champion_5fold_{host}.npy", scores)

    hgb = Pipeline([
        ("preprocess", pre),
        ("clf", HistGradientBoostingClassifier(random_state=42,
                                               **{**HGB_BASE, **CHAMPION_TUNED["hgb"]})),
    ])
    scores, elapsed = timed("tuned HGB 15-fold screen", lambda: cv_scores(hgb, X, y, mode="screen"))
    report["battery"]["tuned_hgb_screen"] = {
        "wall_seconds": round(elapsed, 2),
        "mean": float(scores.mean()),
        "fold_scores": [float(s) for s in scores],
    }

    rf = Pipeline([
        ("preprocess", pre),
        ("clf", RandomForestClassifier(random_state=42, **{**RF_BASE, **CHAMPION_TUNED["rf"]})),
    ])
    scores, elapsed = timed("tuned RF 15-fold screen", lambda: cv_scores(rf, X, y, mode="screen"))
    report["battery"]["tuned_rf_screen"] = {
        "wall_seconds": round(elapsed, 2),
        "mean": float(scores.mean()),
        "fold_scores": [float(s) for s in scores],
    }

    out = f"results/benchmark_{host}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
