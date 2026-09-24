"""Gated pipeline runner (tasks 6 to 14) for the Titanic project.

Runs every remaining task in order against the current champion, gating each
change with paired t tests on per fold CV scores (task 15), re validating
tuned configs on fresh folds (task 7 caveat), and finishing with a nested CV
estimate (task 14).

Logging:
- Human readable lines to stdout (tee to results/runner.log)
- Machine readable events to results/results.jsonl (one JSON object per
  line), each tagged with the host name for cross-machine runs

Run from the repo root with:
    OMP_NUM_THREADS=8 python -m pipeline.run_all 2>&1 | tee results/runner.log

Poll with:  tail -f results/runner.log
A killed run loses no completed decisions. See docs/TASKS.md for task specs.
Can also be imported and driven via main() from pipeline.run_full.
"""

import json
import os
import socket
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluate import (
    CV_FRESH,
    CV_HALF,
    CV_MAIN,
    CV_SCREEN,
    cv_scores,
    paired_test,
)
from src.features import BASE_CAT, BASE_NUM, load_data
from src.model import (
    HGB_BASE,
    RF_BASE,
    TUNED,
    build_stack,
    columns,
    make_pre,
)

OUT_DIR = os.environ.get("RUN_OUT_DIR", "results")
HOST = socket.gethostname()
JSONL = None


def log(msg, **event):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)
    if event:
        JSONL.write(json.dumps({"t": stamp, "host": HOST, **event}) + "\n")


def record_scores(event, task, label, scores):
    log(f"{label}: {scores.mean():.4f} ({len(scores)} folds)", event=event, task=task,
        mean=float(scores.mean()), sem=float(scores.std() / np.sqrt(len(scores))))
    np.save(os.path.join(OUT_DIR, f"scores_{event}_t{task}.npy"), scores)


# ---------------------------------------------------------------- runner state
SPEC = {
    "members": ["rf", "hgb", "lr"],
    "avg_seeds": {},
    "ticket": False,
    "farepp": False,
    "imputer": "median",
    "target_enc": False,
    "hygiene": False,
}


def load_xy(spec=None):
    spec = spec or SPEC
    train, _ = load_data(ticket=spec["ticket"], fare_per_person=spec["farepp"],
                         hygiene=spec["hygiene"])
    num, cat = columns(spec)
    return train[num + cat], train["Survived"]


def with_spec(spec_override, fn):
    saved = dict(SPEC)
    SPEC.update(spec_override)
    try:
        return fn()
    finally:
        SPEC.clear()
        SPEC.update(saved)


def full_eval(model, X, y, event, task, label):
    scores = cv_scores(model, X, y, mode="full")
    record_scores(event, task, label, scores)
    return scores


CHAMP = {"scores": None}


def gate(task, label, spec_override=None, event="gate"):
    """Challenger vs champion on identical 50 folds; accept only significant
    improvement (paired t test, task 15)."""

    def build():
        X, y = load_xy()
        return full_eval(build_stack(spec=SPEC, tuned=TUNED), X, y, event, task, label)

    scores = with_spec(spec_override or {}, build)
    diff, t, p = paired_test(scores, CHAMP["scores"])
    accept = diff > 0.001 and p < 0.05
    log(f"  gate: diff={diff:+.4f} t={t:.2f} p={p:.4f} -> {'ACCEPT' if accept else 'reject'}",
        event="gate_result", task=task, diff=float(diff), p=float(p), decision=bool(accept))
    if accept:
        SPEC.update(spec_override or {})
        CHAMP["scores"] = scores
        log(f"  new champion: {json.dumps(SPEC)}", event="champion", task=task,
            spec=json.loads(json.dumps(SPEC)))
    return accept


def make_objective(model_name, Xd, yd, num_cols, cat_cols):
    def objective(trial):
        if model_name == "hgb":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 1500),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                "max_bins": trial.suggest_int("max_bins", 64, 255),
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0),
            }
        pre = make_pre(num_cols, cat_cols)
        if model_name == "hgb":
            clf = HistGradientBoostingClassifier(random_state=42, **params)
        else:
            clf = RandomForestClassifier(random_state=42, **params)
        pipe = Pipeline([("preprocess", pre), ("clf", clf)])
        return cv_scores(pipe, Xd, yd, mode="screen").mean()
    return objective


def main():
    global JSONL
    os.makedirs(OUT_DIR, exist_ok=True)
    JSONL = open(os.path.join(OUT_DIR, "results.jsonl"), "a", buffering=1)

    # ================================================================ TASK 6
    log("TASK 6: greedy stack composition (25-fold screen -> 50-fold retest)",
        event="task_start", task=6)

    X0, y0 = load_xy()
    CHAMP["scores"] = full_eval(build_stack(spec=SPEC, tuned=TUNED), X0, y0,
                                "champion_baseline", 6,
                                "Champion stack RF+HGB+LR (50-fold)")
    champ_half = cv_scores(build_stack(spec=SPEC, tuned=TUNED), X0, y0, mode="half")
    log(f"Champion (25-fold screen): {champ_half.mean():.4f}", event="screen", task=6,
        mean=float(champ_half.mean()))

    for name in ["xgb", "cat", "mlp", "svc", "lgbm"]:
        def screen_build():
            X, y = load_xy()
            return cv_scores(build_stack(spec=SPEC, tuned=TUNED), X, y, mode="half")
        s_half = with_spec({"members": SPEC["members"] + [name]}, screen_build)
        diff, t, p = paired_test(s_half, champ_half)
        log(f"  +{name}: {s_half.mean():.4f} (25-fold) diff={diff:+.4f} p={p:.3f}",
            event="screen", task=6, member=name, diff=float(diff), p=float(p))
        if diff > 0.001 and p < 0.30:
            log(f"  +{name} borderline -> 50-fold retest", event="retest", task=6, member=name)
            gate(6, f"+{name} (retest)", spec_override={"members": SPEC["members"] + [name]},
                 event="retest")
            champ_half = cv_scores(build_stack(spec=SPEC, tuned=TUNED), *load_xy(), mode="half")
        else:
            log(f"  +{name} rejected at screen", event="screen_reject", task=6, member=name)

    log(f"TASK 6 done. members={SPEC['members']}", event="task_done", task=6)

    # ================================================================ TASK 7
    log("TASK 7: Optuna HPO for HGB and RF (screen objective, fresh-CV validation)",
        event="task_start", task=7)

    X, y = load_xy()
    num, cat = columns(SPEC)

    study_hgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_hgb.optimize(make_objective("hgb", X, y, num, cat), n_trials=60)
    study_rf = optuna.create_study(direction="maximize",
                                   sampler=optuna.samplers.TPESampler(seed=42))
    study_rf.optimize(make_objective("rf", X, y, num, cat), n_trials=40)
    log(f"HPO best screen: HGB {study_hgb.best_value:.4f} {study_hgb.best_params} | "
        f"RF {study_rf.best_value:.4f} {study_rf.best_params}",
        event="hpo_best", task=7, hgb=study_hgb.best_params, rf=study_rf.best_params)

    for name, study, base_params in [("hgb", study_hgb, HGB_BASE), ("rf", study_rf, RF_BASE)]:
        pre = make_pre(num, cat)
        if name == "hgb":
            untuned_clf = HistGradientBoostingClassifier(random_state=42, **base_params)
            tuned_clf = HistGradientBoostingClassifier(random_state=42, **study.best_params)
        else:
            untuned_clf = RandomForestClassifier(random_state=42, **base_params)
            tuned_clf = RandomForestClassifier(random_state=42, **study.best_params)
        untuned = Pipeline([("preprocess", pre), ("clf", untuned_clf)])
        tuned = Pipeline([("preprocess", pre), ("clf", tuned_clf)])
        s_un = cv_scores(untuned, X, y, mode="fresh")
        s_tu = cv_scores(tuned, X, y, mode="fresh")
        diff, t, p = paired_test(s_tu, s_un)
        ok = bool(diff > 0.001 and p < 0.05)
        log(f"FRESH 50-fold {name}: untuned {s_un.mean():.4f} tuned {s_tu.mean():.4f} "
            f"diff={diff:+.4f} p={p:.4f} -> {'ACCEPT' if ok else 'reject'}",
            event="fresh_validation", task=7, member=name, diff=float(diff), p=float(p),
            decision=ok)
        if ok:
            TUNED[name] = dict(study.best_params)

    gate(7, "champion with tuned HGB/RF", event="gate_tuned")
    log("TASK 7 done.", event="task_done", task=7)

    # ================================================================ TASK 8
    log("TASK 8: seed averaging (5 seeds, staged: 25-fold screen -> 50-fold confirm)",
        event="task_start", task=8)
    champ_half = cv_scores(build_stack(spec=SPEC, tuned=TUNED), *load_xy(), mode="half")
    log(f"Champion refresh (25-fold): {champ_half.mean():.4f}", event="screen", task=8,
        mean=float(champ_half.mean()))
    for name in [m for m in SPEC["members"] if m != "lr"]:
        override = {"avg_seeds": {**SPEC["avg_seeds"], name: 5}}
        s_half = with_spec(override, lambda: cv_scores(build_stack(spec=SPEC, tuned=TUNED),
                                                       *load_xy(), mode="half"))
        diff, t, p = paired_test(s_half, champ_half)
        log(f"  seed-avg x5 {name}: {s_half.mean():.4f} (25-fold) diff={diff:+.4f} p={p:.3f}",
            event="screen", task=8, member=name, diff=float(diff), p=float(p))
        if diff > 0.001 and p < 0.10:
            log(f"  seed-avg {name} promising -> 50-fold confirm", event="retest", task=8,
                member=name)
            gate(8, f"seed-avg x5 on {name} (confirm)", spec_override=override,
                 event="gate_seeds")
            champ_half = cv_scores(build_stack(spec=SPEC, tuned=TUNED), *load_xy(), mode="half")
        else:
            log(f"  seed-avg {name} rejected at screen", event="screen_reject", task=8,
                member=name)
    log("TASK 8 done.", event="task_done", task=8)

    # ================================================================ TASKS 9-13
    log("TASK 9: ticket group features", event="task_start", task=9)
    gate(9, "+TicketGroupSize, TicketPrefix", spec_override={"ticket": True},
         event="gate_ticket")
    log("TASK 9 done.", event="task_done", task=9)

    log("TASK 10: fare per person", event="task_start", task=10)
    if SPEC["ticket"]:
        gate(10, "+FarePerPerson", spec_override={"farepp": True}, event="gate_farepp")
    else:
        log("  skipped: requires TicketGroupSize (task 9 rejected)", event="skip", task=10)
    log("TASK 10 done.", event="task_done", task=10)

    log("TASK 11: IterativeImputer for numerics", event="task_start", task=11)
    gate(11, "iterative imputation", spec_override={"imputer": "iterative"},
         event="gate_iterimp")
    log("TASK 11 done.", event="task_done", task=11)

    log("TASK 12: target encoding of TicketPrefix", event="task_start", task=12)
    if SPEC["ticket"]:
        gate(12, "+TargetEncoder(TicketPrefix)", spec_override={"target_enc": True},
             event="gate_te")
    else:
        log("  skipped: ticket features not in champion", event="skip", task=12)
    log("TASK 12 done.", event="task_done", task=12)

    log("TASK 13: hygiene (Fare<=0 -> NaN)", event="task_start", task=13)
    gate(13, "hygiene fares", spec_override={"hygiene": True}, event="gate_hygiene")
    log("TASK 13 done.", event="task_done", task=13)

    # ================================================================ TASK 14
    log("TASK 14: nested CV with inner HPO (15 trials, seeded sampler)",
        event="task_start", task=14)
    X, y = load_xy()  # reload: features may have changed in tasks 9-13
    num, cat = columns(SPEC)
    OUTER = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)
    outer_scores = []
    TUNED_SNAPSHOT = {k: dict(v) for k, v in TUNED.items()}
    for k, (tr, te) in enumerate(OUTER.split(X, y)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        inner = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        inner.optimize(make_objective("hgb", X_tr, y_tr, num, cat), n_trials=15)
        TUNED["hgb"] = dict(inner.best_params)
        model = build_stack(spec=SPEC, tuned=TUNED)
        model.fit(X_tr, y_tr)
        outer_scores.append(model.score(X_te, y_te))
        log(f"  outer fold {k}: {outer_scores[-1]:.4f}", event="nested_fold", task=14,
            fold=k, score=float(outer_scores[-1]))
    TUNED.clear()
    TUNED.update(TUNED_SNAPSHOT)  # nested search params must not leak into champion.json
    outer_scores = np.array(outer_scores)
    log(f"NESTED CV: {outer_scores.mean():.4f} (+/- "
        f"{outer_scores.std()/np.sqrt(len(outer_scores)):.4f} SEM)",
        event="nested_cv", task=14, mean=float(outer_scores.mean()),
        sem=float(outer_scores.std() / np.sqrt(len(outer_scores))))
    log("TASK 14 done.", event="task_done", task=14)

    # ================================================================ FINAL
    log(f"FINAL SPEC: {json.dumps(SPEC)}", event="final_spec",
        spec=json.loads(json.dumps(SPEC)))
    log(f"FINAL 50-fold CV: {CHAMP['scores'].mean():.4f} (+/- "
        f"{CHAMP['scores'].std()/np.sqrt(len(CHAMP['scores'])):.4f})",
        event="final_cv", mean=float(CHAMP["scores"].mean()))
    with open(os.path.join(OUT_DIR, "champion.json"), "w") as f:
        json.dump({"spec": SPEC, "tuned": TUNED, "cv_mean": float(CHAMP["scores"].mean())},
                  f, indent=2)
    log("ALL DONE", event="all_done")
    JSONL.close()


if __name__ == "__main__":
    main()
