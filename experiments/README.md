# Experiments

Standalone scripts for repeat testing and future levers. Each runs from any
directory (they add the repo root to `sys.path` themselves).

| Script | What it does | When to rerun |
|---|---|---|
| `t6_individuals.py` | Screens candidate base learners individually (50 fold CV) | After any change to features or preprocessing, to see who is worth a stack gate |
| `t6_stack.py` | Greedy stack composition with paired gates (25 fold screen) | To retest ensemble membership without running the full pipeline |
| `t7_hpo.py` | Optuna HPO for HGB and RF with fresh-CV validation | To explore new search spaces or more trials; winners must still pass the fresh-CV gate |

These mirror the logic embedded in `pipeline/run_all.py` and exist so the
levers can be pulled independently. See `docs/TASKS.md` for the recorded
results and `docs/LEARNING_JOURNEY.md` for what they taught.

## Future levers: executed and resolved (2026-09-24, Ryzen 5950X)

Eight honest levers, each with a genuine improvement mechanism, gated
against the shipped champion on 50 paired folds (`experiments/future_levers.py`).
All eight rejected; the champion stands. Level-2 seed stacks, a GBM meta
learner, and cv=10 OOF were excluded from testing as noise or as already
disproven (task 8 measured seed variance at exactly 0.0000).

| Lever | CV mean | paired diff | p | Decision |
|---|---|---|---|---|
| ExtraTrees member | 0.8361 | -0.0009 | 0.231 | reject |
| KNN member | 0.8358 | -0.0012 | 0.047 | reject |
| Calibrated RF/HGB members | 0.8347 | -0.0024 | 0.079 | reject |
| ElasticNet meta learner | 0.8371 | +0.0001 | 0.883 | reject |
| AgeBand x Sex feature | 0.8323 | -0.0047 | 0.0007 | reject, significantly worse |
| log1p(Fare) feature | 0.8355 | -0.0016 | 0.021 | reject, significantly worse |
| Vote stack+CatBoost 1:1 | 0.8383 | +0.0012 | 0.535 | reject |
| Vote stack+CatBoost 2:1 | 0.8370 | -0.0000 | 0.998 | reject |

Two findings worth keeping: explicit age banding destroys information the
trees already extract from continuous Age (significantly worse), and
member-level diversity beyond LR (ExtraTrees, KNN, CatBoost) consistently
adds nothing to this stack, corroborating the task 6 lesson with three
more model families.

A tighter honest estimate of the champion also ran
(`experiments/nested_cv_tight.py`, 25 outer folds x 50-trial inner HPO):
**0.8373 +/- 0.0043**. The original 10-outer nested CV read 0.8406 +/-
0.0073; the tighter protocol's central estimate sits at the flat 50-fold
CV value (0.8370), confirming the original's optimism was small-sample
noise in the outer folds, not HPO leakage.
