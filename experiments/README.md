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

## Future levers not yet implemented

The remaining ideas worth testing, in expected-value order, each a small
edit to `src/model.py` or a new script here:

1. Stacking meta learner variants (elastic net, shallow GBM) or `cv=10` OOF
2. `ExtraTreesClassifier` as a bagging member (more decorrelated than RF)
3. Full stacking level 2: stack of stacks with diverse seeds
4. Feature: `Age` banded into child/adult/elder plus interaction with `Sex`
5. Spline or polynomial features for `Fare` and `Age` in the LR member
6. Calibrated classifiers (`CalibratedClassifierCV`) as members
