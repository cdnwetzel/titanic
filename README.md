# Titanic: From Baseline to Tuned Stack — A Kaggle Learning Journey

My first Kaggle competition, worked end-to-end: environment setup → EDA →
baseline → sixteen iterative tasks → live submissions. The goal was never to
top the leaderboard — it was to learn the ML competition workflow and, more
importantly, to learn *how to evaluate claims of improvement*. The most
valuable results here are the ones that didn't pan out.

**Final model:** stacking ensemble (RF + gradient boosting + logistic
regression, LR meta-learner), 50-fold CV **0.8370 ± 0.0036**, nested CV with
inner HPO **0.8406 ± 0.0073**.

## Results at a glance

| Step | Change | Repeated CV | Public LB |
|---|---|---|---|
| Baseline | Random forest, 5 simple features | 0.8134 | **0.76555** |
| Tasks 1–2 | Title/FamilySize/Deck features, group imputation | 0.8153 | 0.74641 |
| Task 3 | HistGradientBoosting, full features | 0.8203 | 0.75358 |
| Task 4 | Soft vote RF+HGB+LR | 0.8314 | 0.76076 |
| Task 5 | **Stacking (LR meta, passthrough)** | **0.8365** | — |
| Task 7 | Optuna-tuned HGB/RF members | 0.8370 | 0.75837 |
| Task 14 | Nested CV estimate of final pipeline | 0.8406 ± 0.0073 | — |

CV arc: **0.8134 → 0.8370 (+2.4%)**, honest nested estimate 0.8406.
Public LB across all five submissions: 0.746–0.766 — a flat noise band that
never once reflected the real improvement. That is the central finding.

## Setup

- Python 3.12 venv: `kaggle`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`,
  `catboost`, `optuna`
- Data via Kaggle CLI (`kaggle competitions download -c titanic`)
- `eda.py` — quick exploration; `train.py` — final champion pipeline
- `run_all.py` / `run_resume.py` / `run_finish.py` — the sequential task
  runner: every candidate change is gated by a paired t-test on per-fold
  scores from 50 identical CV folds (5-fold × 10 repeats)
- `results.jsonl` — machine-readable log of every gate; `champion.json` —
  final winning configuration

## The data in one paragraph

891 training rows, 12 columns, 38% overall survival. The signal is real but
simple: 74% of females survived vs 19% of males; survival falls from 63%
(1st class) to 24% (3rd class). `Age` is 20% missing, `Cabin` 77% missing.
Everything past this point is squeezing fractions of a percent out of a
small, well-studied dataset.

## Step 0 — Baseline

Random forest on `Pclass, Age, SibSp, Parch, Fare, Sex, Embarked` with
median/mode imputation in a leak-free sklearn `Pipeline`. Repeated CV:
**0.8134** — the number every later step must honestly beat.

## Tasks 1–4 (features → ensemble)

Full detail in git history; summary: classic engineered features (Title,
FamilySize, Deck) showed **no significant gain** for a random forest (single
CV said +0.5%, repeated CV said noise — Lesson 1: confirm with repeated CV).
Group-based Age imputation (Title × Pclass) helped slightly (+0.2%). Gradient
boosting unlocked the task-1 features (+0.7%). A soft-voting ensemble of
three diverse model families added +1.1% — diversity of inductive bias, not
individual strength, is what ensembles buy (uniform weights beat
up-weighting the strongest member).

## Task 5 — Stacking beats voting

`StackingClassifier` with an LR meta-learner over out-of-fold predictions,
with `passthrough=True` (meta sees base predictions *and* raw features):
0.8365 vs 0.8314 soft vote, paired p = 0.0005. **Accepted.**

## Task 6 — Diverse learners: correlation beats strength

Solo 50-fold scores: XGBoost 0.8319, CatBoost 0.8321, SVC 0.8294, MLP 0.8282
— every candidate alone rivals the entire old ensemble. Added to the stack
one at a time with paired gates: **all rejected**. XGB (+0.0021, p=0.071)
and CatBoost (+0.0024, p=0.088) were consistently positive but never
significant — even added jointly (p=0.094). MLP/SVC/LightGBM were negative.

**Lesson:** a stack doesn't want the strongest models; it wants the most
*orthogonal errors*. XGBoost and CatBoost are excellent, but their mistakes
correlate with HGB's — the meta-learner already has that information.

## Task 7 — HPO: big solo gains, zero ensemble gain

Optuna (100 trials): tuned HGB **0.8320 vs 0.8176 untuned (+1.44%)**, tuned
RF **0.8305 vs 0.8085 (+2.20%)** — validated on a *fresh* 50-fold protocol to
guard against search-overfitting the CV (both p < 0.0001). Tuned RF alone
nearly matches the previous whole ensemble.

But swapped into the stack: 0.8370 vs 0.8365, **p = 0.74 — no significant
ensemble gain**. The ensemble was already extracting what tuning provides.

**Lesson:** single-model improvements don't automatically transfer to
ensembles. Measure at the level you ship.

## Task 8 — Seed averaging: nothing to average

Averaging 5 seeds per stochastic member: diff 0.0000, p = 1.000. At this
scale, with regularized configs, seed variance is already below the noise
floor. **Rejected** — cheap to test, cheaper to keep the simpler model.

## Tasks 9–13 — Features, imputation, hygiene: all rejected

| Task | Change | Gate result |
|---|---|---|
| 9 Ticket groups | +TicketGroupSize, TicketPrefix | −0.0019, p=0.36 |
| 10 Fare per person | — | skipped (depends on 9) |
| 11 Iterative imputation | IterativeImputer for numerics | −0.0024, p=0.07 |
| 12 Target encoding | — | skipped (depends on 9) |
| 13 Hygiene | Fare ≤ 0 → NaN | +0.0011, p=0.21 |

**Lesson:** the famous "secret features" of Titanic lore mostly duplicate
what `Sex`, `Pclass`, `Fare` and the task-1 features already encode. The
disciplined gates — not intuition — decide.

## Task 14 — The honest number: nested CV 0.8406

Final pipeline re-evaluated with HPO *inside* each outer fold (10 outer
folds × 15-trial inner search): **0.8406 ± 0.0073**. Slightly above the flat
50-fold 0.8370 — i.e., the hyperparameter search did **not** overfit the
evaluation protocol. This is the most trustworthy estimate of the model's
true skill.

## Task 15 — Paired tests: the discipline throughout

Every accept/reject decision above is a paired t-test on per-fold scores
from identical folds. Gate: improvement > 0.1% and p < 0.05, with a
25-fold screen → 50-fold retest stage for borderline cases. No decision was
made on a single CV run or on leaderboard feedback.

## Task 16 — The leaderboard cannot see any of this

Five submissions spanning CV 0.8134 → 0.8406 landed in a 0.746–0.766 band;
the displayed rank (~7,390 of 10,200, ahead of ~19%) is still held by the
day-one baseline. The public LB scores a few hundred rows — ±2% between two
models is ~8 passengers, indistinguishable from luck.

**Lesson:** model selection on the public leaderboard is fitting to noise.
Trust controlled CV; spend LB submissions sparingly as coarse sanity checks.
We used 5 of today's 10 and stopped — the discipline *is* the deliverable.

## Final lessons (the transferable set)

1. On small data, single CV runs lie. Repeated CV + paired tests or it didn't happen.
2. A feature's value is model-dependent; a model's value is ensemble-dependent.
3. Ensembles buy orthogonal errors, not individual accuracy.
4. Single-model gains don't transfer to ensembles automatically.
5. HPO is the easiest place to overfit your CV protocol — re-validate on fresh folds; nested CV for the final number.
6. At small scale, seed variance and "hygiene" fixes are below the noise floor.
7. The public leaderboard is a noisy oracle. CV is the compass; LB is a lighthouse glimpsed through fog.

Also worth knowing: the top of the Titanic leaderboard (~73 teams at a
perfect 1.00000) is overfit or junk entries. The legitimate modeling
ceiling is ~0.84; the nested estimate of 0.8406 says this pipeline is
essentially at it.

## Engineering note (for anyone rerunning this)

Parallelizing tiny fits across processes fought us all day: loky (fork and
spawn) and a persistent fork pool all eventually lost workers silently once
XGBoost/LightGBM/CatBoost entered the workload, hanging `pool.map` with no
error. The final runner is **serial** with OpenMP-parallel fits
(`OMP_NUM_THREADS=8`) — slower per evaluation, but deterministic and
hang-proof. On 891-row data, robust beats fast.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install kaggle pandas scikit-learn xgboost lightgbm catboost optuna
# ~/.kaggle/access_token or kaggle.json must be configured
.venv/bin/kaggle competitions download -c titanic -p data && cd data && unzip titanic.zip && cd ..
.venv/bin/python eda.py                            # quick data exploration
OMP_NUM_THREADS=8 .venv/bin/python train.py        # trains champion -> submission.csv
# full task pipeline (gates, ~3h serial):
OMP_NUM_THREADS=8 .venv/bin/python -u run_all.py 2>&1 | tee runner.log
.venv/bin/kaggle competitions submit -c titanic -f submission.csv -m "message"
```

## Files

- `train.py` — final champion: engineered features → median impute + scale +
  one-hot → stacking ensemble (tuned RF/HGB + LR, LR meta, passthrough)
- `eda.py` — data overview: shapes, missing values, survival rates
- `exputil.py` — harness: feature loaders, CV protocols, paired-test gate,
  seed-averaging wrapper
- `run_all.py`, `run_resume.py`, `run_finish.py` — the sequential task
  runners (kept for audit; run_all.py supersedes the other two)
- `results.jsonl` / `runner.log` — every gate decision; `champion.json` —
  final configuration
- `submission.csv` — predictions from the final champion
- `data/` — raw competition CSVs
