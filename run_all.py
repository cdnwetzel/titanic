"""Sequential runner for tasks 6-14 (baseline + tasks 1-16 evaluation).

Runs every remaining task in order against the current champion, gating each
change with paired t-tests on per-fold CV scores (task 15), re-validating
tuned configs on fresh folds (task 7 caveat), and finishing with a nested-CV
estimate (task 14).

Logging:
- Human-readable lines -> stdout (tee to runner.log)
- Machine-readable events -> results.jsonl (one JSON object per line)

Run out of band with:
    OMP_NUM_THREADS=4 .venv/bin/python run_all.py 2>&1 | tee runner.log

Poll with:  tail -f runner.log
"""

import functools
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

from exputil import (
    BASE_CAT,
    BASE_NUM,
    SeedAveragedClassifier,
    cv_scores,
    load_data,
    paired_test,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

JSONL = open("results.jsonl", "a", buffering=1)


def log(msg, **event):
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)
    if event:
        JSONL.write(json.dumps({"t": stamp, **event}) + "\n")


def record_scores(event, task, label, scores):
    log(f"{label}: {scores.mean():.4f} ({len(scores)} folds)", event=event, task=task,
        mean=float(scores.mean()), sem=float(scores.std() / np.sqrt(len(scores))))
    np.save(f"scores_{event}_t{task}.npy", scores)


# ---------------------------------------------------------------- model zoo
# Seed-aware factories at module level so multiprocessing can pickle them.

TUNED = {"hgb": {}, "rf": {}}  # filled by task 7

HGB_BASE = {"max_iter": 500, "learning_rate": 0.05, "max_leaf_nodes": 15,
            "min_samples_leaf": 20, "l2_regularization": 1.0}
RF_BASE = {"n_estimators": 200}


def make_member(name, seed):
    if name == "rf":
        params = {**RF_BASE, **TUNED["rf"]}
        return RandomForestClassifier(random_state=seed, **params)
    if name == "hgb":
        params = {**HGB_BASE, **TUNED["hgb"]}
        return HistGradientBoostingClassifier(random_state=seed, **params)
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


def member_factory(name):
    return functools.partial(make_member, name)


# ---------------------------------------------------------------- preprocessing
def make_pre(num, cat, imputer="median", target_enc=False):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    imp = IterativeImputer(random_state=42) if imputer == "iterative" \
        else SimpleImputer(strategy="median")
    branches = [
        ("num", Pipeline([("impute", imp), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ]
    if target_enc:
        branches.append(("te", TargetEncoder(target_type="binary"), ["TicketPrefix"]))
    return ColumnTransformer(branches)


# ---------------------------------------------------------------- champion spec
SPEC = {
    "members": ["rf", "hgb", "lr"],
    "avg_seeds": {},          # task 8: member -> n_seeds
    "ticket": False,          # task 9
    "farepp": False,          # task 10
    "imputer": "median",      # task 11
    "target_enc": False,      # task 12 (requires ticket)
    "hygiene": False,         # task 13
}


def columns(spec=None):
    spec = spec or SPEC
    num = list(BASE_NUM)
    cat = list(BASE_CAT)
    if spec["ticket"]:
        num.append("TicketGroupSize")
        cat.append("TicketPrefix")
    if spec["farepp"]:
        num.append("FarePerPerson")
    return num, cat


def build_stack(spec=None):
    spec = spec or SPEC
    num, cat = columns(spec)
    pre = make_pre(num, cat, imputer=spec["imputer"], target_enc=spec["target_enc"])
    estimators = []
    for name in spec["members"]:
        if spec["avg_seeds"].get(name):
            est = SeedAveragedClassifier(member_factory(name), n_seeds=spec["avg_seeds"][name])
        else:
            est = make_member(name, 42)
        estimators.append((name, est))
    return Pipeline([
        ("preprocess", pre),
        ("clf", StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, stack_method="predict_proba", passthrough=True)),
    ])


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


# ---------------------------------------------------------------- gates
CHAMP = {"scores": None}


def gate(task, label, spec_override=None, event="gate"):
    """Challenger vs champion on identical 50 folds; accept only significant
    improvement (paired t-test, task 15)."""

    def build():
        X, y = load_xy()
        return full_eval(build_stack(), X, y, event, task, label)

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


# ================================================================ TASK 6
log("TASK 6: greedy stack composition (25-fold screen -> 50-fold retest)",
    event="task_start", task=6)

X0, y0 = load_xy()
CHAMP["scores"] = full_eval(build_stack(), X0, y0, "champion_baseline", 6,
                            "Champion stack RF+HGB+LR (50-fold)")
champ_half = cv_scores(build_stack(), X0, y0, mode="half")
log(f"Champion (25-fold screen): {champ_half.mean():.4f}", event="screen", task=6,
    mean=float(champ_half.mean()))

for name in ["xgb", "cat", "mlp", "svc", "lgbm"]:
    def screen_build():
        X, y = load_xy()
        return cv_scores(build_stack(), X, y, mode="half")
    s_half = with_spec({"members": SPEC["members"] + [name]}, screen_build)
    diff, t, p = paired_test(s_half, champ_half)
    log(f"  +{name}: {s_half.mean():.4f} (25-fold) diff={diff:+.4f} p={p:.3f}",
        event="screen", task=6, member=name, diff=float(diff), p=float(p))
    if diff > 0.001 and p < 0.30:
        log(f"  +{name} borderline -> 50-fold retest", event="retest", task=6, member=name)
        gate(6, f"+{name} (retest)", spec_override={"members": SPEC["members"] + [name]},
             event="retest")
        champ_half = cv_scores(build_stack(), *load_xy(), mode="half")
    else:
        log(f"  +{name} rejected at screen", event="screen_reject", task=6, member=name)

log(f"TASK 6 done. members={SPEC['members']}", event="task_done", task=6)

# ================================================================ TASK 7
log("TASK 7: Optuna HPO for HGB and RF (screen objective, fresh-CV validation)",
    event="task_start", task=7)

X, y = load_xy()
num, cat = columns()


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


study_hgb = optuna.create_study(direction="maximize")
study_hgb.optimize(make_objective("hgb", X, y, num, cat), n_trials=60)
study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(make_objective("rf", X, y, num, cat), n_trials=40)
log(f"HPO best screen: HGB {study_hgb.best_value:.4f} {study_hgb.best_params} | "
    f"RF {study_rf.best_value:.4f} {study_rf.best_params}",
    event="hpo_best", task=7, hgb=study_hgb.best_params, rf=study_rf.best_params)

# fresh-CV validation of tuned vs untuned (untouched fold seed)
pre = make_pre(num, cat)
for name, study, base_params in [("hgb", study_hgb, HGB_BASE), ("rf", study_rf, RF_BASE)]:
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
        event="fresh_validation", task=7, member=name, diff=float(diff), p=float(p), decision=ok)
    if ok:
        TUNED[name] = dict(study.best_params)

gate(7, "champion with tuned HGB/RF", event="gate_tuned")
log("TASK 7 done.", event="task_done", task=7)

# ================================================================ TASK 8
log("TASK 8: seed averaging (5 seeds, staged: 25-fold screen -> 50-fold confirm)",
    event="task_start", task=8)
champ_half = cv_scores(build_stack(), *load_xy(), mode="half")
log(f"Champion refresh (25-fold): {champ_half.mean():.4f}", event="screen", task=8,
    mean=float(champ_half.mean()))
for name in [m for m in SPEC["members"] if m != "lr"]:
    override = {"avg_seeds": {**SPEC["avg_seeds"], name: 5}}
    s_half = with_spec(override, lambda: cv_scores(build_stack(), *load_xy(), mode="half"))
    diff, t, p = paired_test(s_half, champ_half)
    log(f"  seed-avg x5 {name}: {s_half.mean():.4f} (25-fold) diff={diff:+.4f} p={p:.3f}",
        event="screen", task=8, member=name, diff=float(diff), p=float(p))
    if diff > 0.001 and p < 0.10:
        log(f"  seed-avg {name} promising -> 50-fold confirm", event="retest", task=8, member=name)
        gate(8, f"seed-avg x5 on {name} (confirm)", spec_override=override, event="gate_seeds")
    else:
        log(f"  seed-avg {name} rejected at screen", event="screen_reject", task=8, member=name)
log("TASK 8 done.", event="task_done", task=8)

# ================================================================ TASKS 9-13
log("TASK 9: ticket group features", event="task_start", task=9)
gate(9, "+TicketGroupSize, TicketPrefix", spec_override={"ticket": True}, event="gate_ticket")
log("TASK 9 done.", event="task_done", task=9)

log("TASK 10: fare per person", event="task_start", task=10)
gate(10, "+FarePerPerson", spec_override={"farepp": True}, event="gate_farepp")
log("TASK 10 done.", event="task_done", task=10)

log("TASK 11: IterativeImputer for numerics", event="task_start", task=11)
gate(11, "iterative imputation", spec_override={"imputer": "iterative"}, event="gate_iterimp")
log("TASK 11 done.", event="task_done", task=11)

log("TASK 12: target encoding of TicketPrefix", event="task_start", task=12)
if SPEC["ticket"]:
    gate(12, "+TargetEncoder(TicketPrefix)", spec_override={"target_enc": True}, event="gate_te")
else:
    log("  skipped: ticket features not in champion", event="skip", task=12)
log("TASK 12 done.", event="task_done", task=12)

log("TASK 13: hygiene (Fare<=0 -> NaN)", event="task_start", task=13)
gate(13, "hygiene fares", spec_override={"hygiene": True}, event="gate_hygiene")
log("TASK 13 done.", event="task_done", task=13)

# ================================================================ TASK 14
log("TASK 14: nested CV with inner HPO (20 trials) for honest estimate",
    event="task_start", task=14)
X, y = load_xy()  # reload: features may have changed in tasks 9-13
num, cat = columns()
OUTER = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=7)
outer_scores = []
for k, (tr, te) in enumerate(OUTER.split(X, y)):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]

    inner = optuna.create_study(direction="maximize")
    inner.optimize(make_objective("hgb", X_tr, y_tr, num, cat), n_trials=15)
    TUNED["hgb"] = dict(inner.best_params)
    model = build_stack()
    model.fit(X_tr, y_tr)
    outer_scores.append(model.score(X_te, y_te))
    log(f"  outer fold {k}: {outer_scores[-1]:.4f}", event="nested_fold", task=14, fold=k,
        score=float(outer_scores[-1]))
outer_scores = np.array(outer_scores)
log(f"NESTED CV: {outer_scores.mean():.4f} (+/- {outer_scores.std()/np.sqrt(len(outer_scores)):.4f} SEM)",
    event="nested_cv", task=14, mean=float(outer_scores.mean()),
    sem=float(outer_scores.std() / np.sqrt(len(outer_scores))))
log("TASK 14 done.", event="task_done", task=14)

# ================================================================ FINAL
log(f"FINAL SPEC: {json.dumps(SPEC)}", event="final_spec", spec=json.loads(json.dumps(SPEC)))
log(f"FINAL 50-fold CV: {CHAMP['scores'].mean():.4f} (+/- "
    f"{CHAMP['scores'].std()/np.sqrt(len(CHAMP['scores'])):.4f})",
    event="final_cv", mean=float(CHAMP["scores"].mean()))
with open("champion.json", "w") as f:
    json.dump({"spec": SPEC, "tuned": TUNED, "cv_mean": float(CHAMP["scores"].mean())}, f, indent=2)
log("ALL DONE", event="all_done")
