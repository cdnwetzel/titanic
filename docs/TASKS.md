# Task Reference: Baseline + Tasks 1 to 16

Complete specification for reproducing every step of this project. Companion
to `README.md` (overview) and `LEARNING_JOURNEY.md` (narrative). Every number
here was produced under the protocols below; the raw per fold evidence is in
`results/results.jsonl`, `results/runner.log`, and the `results/scores_*.npy`
arrays.

Author: Chris Wetzel. Twenty eight years in IT, three in AI, one in ML. I
write specs the way I learned to in infrastructure work: exact enough that a
stranger (or future me) can rerun them without asking questions.

## 0. Environment

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# pinned: Python 3.12, scikit-learn 1.9.1, pandas 3.0.6, xgboost 3.4.1,
# lightgbm 4.7.0, catboost 1.2.10, optuna 5.0.0 (full list: requirements.txt)
```

- Kaggle credentials at `~/.kaggle/access_token` (or `kaggle.json`), and the
  competition rules accepted on the Titanic page, then:
  `kaggle competitions download -c titanic -p data && cd data && unzip titanic.zip && cd ..`
- Run everything with `OMP_NUM_THREADS=8`. The pipeline is deliberately
  serial across folds; see the engineering note in `README.md`.

## 1. Evaluation methodology (applies to every task)

| Protocol | Definition | Use |
|---|---|---|
| `CV_SCREEN` | `RepeatedStratifiedKFold(5, 3, random_state=42)`, 15 fits | HPO objective, shortlists |
| `CV_HALF` | `RepeatedStratifiedKFold(5, 5, random_state=42)`, 25 fits | screening gates |
| `CV_MAIN` | `RepeatedStratifiedKFold(5, 10, random_state=42)`, 50 fits | all accept/reject decisions |
| `CV_FRESH` | `RepeatedStratifiedKFold(5, 10, random_state=2024)`, 50 fits | re-validating search-found configs |

- **Decision gate:** accept a challenger only if the mean paired improvement
  is **greater than 0.001 (0.1%)** and the **paired t test p < 0.05** on
  `CV_MAIN` per fold scores (identical folds for both configs, see
  `src/evaluate.py::paired_test`).
- **Two-stage screening:** changes are first screened on `CV_HALF`; if
  0.001 < diff and p < 0.30, they earn a `CV_MAIN` retest. Otherwise they
  are rejected at screen.
- **Search hygiene:** anything found by optimization (task 7) is
  re-validated on `CV_FRESH`, whose folds the search never saw.
- `src/evaluate.py::cv_scores(model, X, y, mode=...)` is the single
  evaluation entry point. Per fold scores are saved as
  `results/scores_<event>_t<task>.npy`.

## 2. Shared feature definitions

From `src/features.py` (applied identically to train and test; ticket counts
use the concatenated train+test identifiers only, never the target):

- `Title` = `Name.str.extract(r",\s*([^\.]+)\.")`, mapped `{Mlle,Ms -> Miss,
  Mme -> Mrs}`, everything outside `{Mr, Mrs, Miss, Master}` becomes `Rare`
- `FamilySize` = `SibSp + Parch + 1`; `IsAlone` = `FamilySize == 1`
- `Deck` = `Cabin.str[0]`, missing becomes `"U"`
- Task 9: `TicketGroupSize` = count of passengers sharing the same
  space-stripped `Ticket` across train+test; `TicketPrefix` = ticket with
  trailing `[0-9.\s]+$` removed, then `rstrip("/.")`, empty becomes `"none"`
- Task 10: `FarePerPerson` = `Fare / TicketGroupSize`
- Task 13: `Fare <= 0` becomes NaN

**Base columns** (champion feature set): numeric `[Pclass, Age, SibSp,
Parch, Fare, FamilySize, IsAlone]`, categorical `[Sex, Embarked, Title,
Deck]`.

**Champion preprocessing** (`src/model.py::make_pre`): numeric branch is
median imputation plus `StandardScaler`; categorical branch is most frequent
imputation plus one-hot (`handle_unknown="ignore", sparse_output=False`).

## 3. Baseline

- **Model:** `RandomForestClassifier(n_estimators=200, random_state=42)`
- **Features:** numeric `[Pclass, Age, SibSp, Parch, Fare]`, categorical
  `[Sex, Embarked]`; median/mode imputation; one-hot. No scaling.
- **Result:** `CV_MAIN` **0.8134 +/- 0.0036**; public LB **0.76555**.
- **Reproduce:** the RF reference row of `experiments/t6_individuals.py`
  shows the same model under the shared (scaled) preprocessing at 0.8080.
  The 0.8134 figure predates scaling and comes from the task 1 experiments
  under the exact spec above.

## 4. Task specifications

### Task 1: engineered features (REJECTED)

- **Added:** `Title`, `FamilySize`, `IsAlone`, `Deck`; variants `FamGroup`
  (`FamilySize` cut into alone/small/large), `HasCabin` (binary).
- **Result (CV_MAIN):** baseline 0.8134; feature additions alone 0.8091 to
  0.8153, none significant. A single 5-fold CV run had shown +0.5%; that
  mirage motivated the repeated-CV rule.
- **Reproduce:** the variant matrix ran inline during development; the
  equivalent machinery is the gate framework in `pipeline/run_all.py`.

### Task 2: group-based Age imputation (ACCEPTED, +0.2%)

- **Implementation:** a custom `GroupMedianImputer(column="Age",
  group_by=[Title, Pclass])` transformer *inside* the pipeline, so medians
  are computed per training fold only. Fallback: global median. (This class
  lived in an earlier `train.py`; the pattern is what matters and the gated
  equivalent is in the pipeline.)
- **Results (CV_MAIN):** by `Title` 0.8103 (rejected); by `Sex x Pclass`
  0.8139 (tie); by **`Title x Pclass` 0.8153 (accepted)**.
- **Public LB of that model:** 0.74641, the first live demonstration of LB
  noise.

### Task 3: gradient boosting (ACCEPTED, +0.5% over the task 2 line)

- **Model:** `HistGradientBoostingClassifier`; native NaN handling, so the
  numeric branch passes values through un-imputed.
- **Results (CV_MAIN):** base features 0.8184; plus task 1 features 0.8192;
  tuned (`max_iter=500, lr=0.05, max_leaf_nodes=15, min_samples_leaf=20,
  l2=1.0`) **0.8203**.
- **Public LB:** 0.75358.

### Task 4: soft-voting ensemble (ACCEPTED, +1.1%)

- **Model:** `VotingClassifier(voting="soft")`, uniform weights, members
  RF(200) + HGB(tuned) + LR(`max_iter=1000`), shared scaled preprocessing.
- **Variants tested:** RF+HGB 0.8219; HGB double weight 0.8210; plus LR
  **0.8314**; double HGB plus LR 0.8303. Uniform weights with LR won.
- **Public LB:** 0.76076.

### Task 5: stacking (ACCEPTED, p = 0.0005)

- **Model:** `StackingClassifier(estimators=[RF, HGB, LR],
  final_estimator=LogisticRegression(max_iter=1000), cv=5,
  stack_method="predict_proba", passthrough=True)`. Passthrough gives the
  meta learner raw features alongside out-of-fold base predictions.
- **Results (CV_MAIN):** without passthrough 0.8337 (p=0.019); with
  passthrough **0.8365 (p=0.0005 vs the task 4 vote)**. New champion.

### Task 6: diverse base learners (REJECTED)

- **Candidates (solo CV_MAIN, shared preprocessing):** XGBoost
  (`n_estimators=500, lr=0.05, max_depth=3, subsample=0.8,
  colsample_bytree=0.8, eval_metric="logloss"`) **0.8319**; CatBoost
  (`iterations=500, lr=0.05, depth=6`) **0.8321**; SVC rbf
  (`probability=True`) 0.8294; MLP `(32,), alpha=0.5, max_iter=2000` 0.8282;
  LightGBM (`n_estimators=500, lr=0.05, num_leaves=15,
  min_child_samples=20`) 0.8146.
- **Stack gates (CV_MAIN vs champion):** +XGB 0.8386 (+0.0021, p=0.071);
  +Cat 0.8388 (+0.0024, p=0.088); +MLP/+SVC/+LGBM negative at screen.
- **Post-hoc joint test:** +XGB+Cat together 0.8395 (+0.0025, p=0.094),
  rejected. (Ran in the finish phase of the original session.)
- **Lesson:** correlation of errors, not member strength.

### Task 7: hyperparameter search (members tuned; stack unchanged)

- **Search:** Optuna `TPESampler(seed=42)`; 60 HGB + 40 RF trials;
  objective is mean accuracy on `CV_SCREEN`.
  - HGB space: `lr in [0.01, 0.2] log`, `max_iter in [200, 1500]`,
    `max_leaf_nodes in [7, 63]`, `min_samples_leaf in [5, 60]`,
    `l2 in [1e-3, 10] log`, `max_bins in [64, 255]`
  - RF space: `n_estimators in [200, 800]`, `max_depth in [2, 20]`,
    `min_samples_leaf in [1, 30]`, `max_features in [0.3, 1.0]`
- **Fresh-fold validation (CV_FRESH, folds the search never saw):**
  HGB **0.8320 vs 0.8176** (+1.44%, p < 0.0001); RF **0.8305 vs 0.8085**
  (+2.20%, p < 0.0001). Both accepted into the tuned set.
  - Winners: HGB `lr=0.0222, max_iter=1165, max_leaf_nodes=8,
    min_samples_leaf=44, l2=4.018, max_bins=77`; RF `n_estimators=661,
    max_depth=15, min_samples_leaf=3, max_features=0.4477`. These are
    `CHAMPION_TUNED` in `src/model.py`.
- **Stack re-gate (CV_MAIN):** 0.8370 vs 0.8365, **p=0.74, no significant
  ensemble gain**. Tuned params kept (validated singles, nominal stack gain,
  no significant difference either way).
- **Reproduce:** `experiments/t7_hpo.py` standalone; integrated in
  `pipeline/run_all.py`.

### Task 8: seed averaging (REJECTED)

- **Implementation:** `SeedAveragedClassifier(factory, n_seeds=5)` fits 5
  clones (`random_state = 0..4`) and averages `predict_proba`; usable as a
  stacking member. In `src/evaluate.py`.
- **Results (CV_HALF screens):** RF diff 0.0000 (p=1.000); HGB diff 0.0000.
  Seed variance is below the noise floor at this scale.

### Task 9: ticket group features (REJECTED)

- **Implementation:** `TicketGroupSize`, `TicketPrefix` per section 2.
- **Result (CV_MAIN):** 0.8346, diff -0.0019, p=0.36.

### Task 10: fare per person (SKIPPED, dependency)

- `FarePerPerson` requires `TicketGroupSize` (task 9). The runner skips it
  automatically when task 9 is rejected; `load_data` raises if asked for it
  without the ticket flag.

### Task 11: iterative imputation (REJECTED)

- **Implementation:** `IterativeImputer(random_state=42)` (default
  BayesianRidge) replaces the median imputer in the numeric branch.
- **Result (CV_MAIN):** 0.8347, diff -0.0024, p=0.07.

### Task 12: target encoding (SKIPPED, dependency)

- `TargetEncoder(target_type="binary")` on `TicketPrefix`, placed in the
  `ColumnTransformer`. Fold-safe because the pipeline refits it on each
  training fold only. Requires task 9.

### Task 13: data hygiene (REJECTED)

- **Implementation:** `Fare <= 0` becomes NaN (12 placeholder fares), leaving
  imputation to the pipeline.
- **Result (CV_MAIN):** 0.8382, diff +0.0011, p=0.21.

### Task 14: nested CV honest estimate

- **Protocol:** outer `RepeatedStratifiedKFold(5, 2, random_state=7)`, 10
  outer folds. Inside each, a fresh 15-trial Optuna search
  (`TPESampler(seed=42)`, HGB space from task 7) runs on the outer training
  split only, then one champion fit scores the outer test split.
- **Result:** **0.8406 +/- 0.0073**. Per fold: 0.8603, 0.8258, 0.8034,
  0.8596, 0.8483, 0.8380, 0.8764, 0.8090, 0.8258, 0.8596. At or above the
  flat 50-fold 0.8370, so the HPO did not inflate the estimate.
- **Reproduce:** task 14 section of `pipeline/run_all.py` (~20 min).

### Task 15: paired-test discipline (methodology)

Every ACCEPT/REJECT above is a paired t test on per fold scores from
identical fold splits (`src/evaluate.py::paired_test`); thresholds in
section 1. No decision may rest on a single CV run or a leaderboard wiggle.

### Task 16: leaderboard budget (policy, executed)

| Submission | Public score |
|---|---|
| Baseline RF | 0.76555 |
| Task 2 group imputation | 0.74641 |
| Task 3 HGB | 0.75358 |
| Task 4 soft vote | 0.76076 |
| Final tuned stack (CV 0.8406) | 0.75837 |

Five of the daily ten submissions spent; the rest deliberately unused. Task
16's rule: the LB is a coarse sanity check, never a selection mechanism. All
five scores lie inside the +/-2% noise band of a few-hundred-row test
sample.

## 5. Final champion (what `train.py` builds)

- Pipeline: `make_pre` (section 2) feeding `StackingClassifier([tuned RF,
  tuned HGB, LR], LR meta, cv=5, predict_proba, passthrough=True)`.
- Estimates: CV_MAIN 0.8370 +/- 0.0036; nested CV 0.8406 +/- 0.0073.
- `results/champion.json` holds the exact spec and tuned parameters.

## 6. Reproducing the full pipeline

```bash
OMP_NUM_THREADS=8 python -m pipeline.run_all 2>&1 | tee results/runner.log
```

`pipeline/run_all.py` runs tasks 6 to 14 sequentially with the gates above
(about 3 h serial). Events stream to `results/results.jsonl` as they happen;
a killed run loses no completed decisions.

**Runtime map (serial, OMP=8):** champion baseline ~6 min; each 50-fold
stack gate ~6 to 13 min; HPO ~35 min; nested CV ~20 min.

## 7. Verifying an environment

```bash
.venv/bin/python -m pytest tests
```

Seventeen tests cover feature invariants, harness determinism, model
construction, an end to end champion fit, and a broad reference band for the
untuned HGB (the canary that catches library version drift).

## 8. Comparing across machines

1. `python scripts/env_info.py --out results/env_<hostname>.json` on each
   machine; commit the files.
2. Run `train.py` and the pytest suite on each machine.
3. Compare the `train.py` CV printout (0.8361 +/- BLAS noise), the pytest
   band test, and per evaluation wall clock from `results/runner.log`
   timestamps.
4. For a full comparison, run the pipeline and diff `results/results.jsonl`
   event by event. Gate scores should agree to the third decimal.

## 9. History: known issues fixed since the original run

The original session hit three real bugs, all fixed in the current
`pipeline/run_all.py` and recorded here so nobody rediscovers them:

1. `make_member` duplicate-kwargs `TypeError` when tuned params overlap
   defaults (fixed by merging dicts before the call).
2. Task 10 crashed when task 9 was rejected (fixed by the dependency guard;
   task 12 always had it).
3. Nested-CV search params leaked into `champion.json` (`TUNED` is now
   snapshotted and restored around task 14).
4. HPO studies and nested searches now all use `TPESampler(seed=42)`, so
   reruns are deterministic.

The two resume scripts from the original session (`run_resume.py`,
`run_finish.py`) were deleted in the repo cleanup. Their only unique logic
(the task 6 joint post-hoc test) is described in task 6 above, and the fixes
they carried are in the list above.

## 10. File map

| File | Role |
|---|---|
| `src/features.py` | feature engineering, data loading |
| `src/evaluate.py` | CV protocols, `cv_scores`, `paired_test`, seed averaging |
| `src/model.py` | model zoo, `make_pre`, `build_stack`, tuned parameters |
| `pipeline/run_all.py` | gated pipeline, tasks 6 to 14 |
| `experiments/` | standalone levers for repeat testing |
| `tests/` | pytest suite |
| `results/results.jsonl` | every gate decision, machine readable |
| `results/runner.log` | human readable run log |
| `results/champion.json` | final spec and tuned parameters |
| `results/env_*.json` | per machine hardware context |
| `requirements.txt` | pinned environment |
