# Titanic: From Baseline to Tuned Stack

A first Kaggle competition, worked end to end as a controlled experiment in
knowing when a model change is real: baseline, sixteen tasks, every candidate
improvement gated by paired statistical tests, five live submissions, and a
leaderboard that could not see any of it.

Author: Chris Wetzel. Twenty eight years in IT, three in AI, one in ML.
This repo is written for people like me: strong technical backgrounds, new to
machine learning, and suspicious of any claim that was not measured properly.
All results here trace to per fold score arrays and a machine readable
decision log.

**Final model:** stacking ensemble (random forest + gradient boosting +
logistic regression, logistic meta learner). 50 fold CV **0.8370 +/- 0.0036**;
nested CV with the hyperparameter search inside the validation loop
**0.8406 +/- 0.0073**.

## Results at a glance

| Step | Change | Repeated CV | Public LB |
|---|---|---|---|
| Baseline | Random forest, 5 simple features | 0.8134 | **0.76555** |
| Tasks 1 to 2 | Classic features, group imputation | 0.8153 | 0.74641 |
| Task 3 | HistGradientBoosting, full features | 0.8203 | 0.75358 |
| Task 4 | Soft vote RF+HGB+LR | 0.8314 | 0.76076 |
| Task 5 | **Stacking (LR meta learner)** | **0.8365** | 0.75837 (task 7 line) |
| Task 7 | Optuna tuned HGB/RF members | 0.8370 | 0.75837 |
| Task 14 | Nested CV estimate of final pipeline | 0.8406 +/- 0.0073 | 0.75837 |

CV arc: **0.8134 to 0.8370 (+2.4%)**, honest nested estimate 0.8406.
Public LB across all five submissions: 0.746 to 0.766, a flat noise band.
That contrast is the central finding of the project and the reason the
documents below exist.

## Repository layout

```
titanic/
├── README.md              this file
├── train.py               entry point: trains the champion, writes submission.csv
├── eda.py                 entry point: data exploration
├── requirements.txt       pinned environment
├── submission.csv         current champion predictions
├── docs/
│   ├── LEARNING_JOURNEY.md   narrative writeup of the full arc, for peers
│   └── TASKS.md              complete task by task reproducibility spec
├── src/                   library code (features, evaluation, models)
│   ├── features.py        feature engineering and data loading
│   ├── evaluate.py        CV protocols, fold scoring, paired tests
│   └── model.py           model zoo, preprocessing, champion stack builder
├── pipeline/
│   └── run_all.py         the gated full pipeline, tasks 6 to 14
├── experiments/           standalone levers: screens, stack composition, HPO
│   └── README.md          what each script is for and future levers
├── tests/                 pytest suite (features, harness, model, smoke)
├── scripts/
│   └── env_info.py        capture machine context for hardware comparison
└── results/               audit trail: results.jsonl, runner.log,
                         champion.json, per fold score arrays
```

Competition data is not included (Kaggle rules). Download it:

```bash
kaggle competitions download -c titanic -p data && cd data && unzip titanic.zip && cd ..
```

## Quick start (clone and reproduce)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python scripts/env_info.py --out results/env_$(hostname).json   # record the machine
.venv/bin/python train.py          # champion: prints CV, writes submission.csv
.venv/bin/python -m pytest tests   # suite: features, harness, model, smoke
```

Full pipeline (tasks 6 to 14, about 3 h serial):

```bash
OMP_NUM_THREADS=8 python -m pipeline.run_all 2>&1 | tee results/runner.log
```

Experiment levers (see experiments/README.md):

```bash
python experiments/t6_individuals.py    # screen candidate learners (50 fold)
python experiments/t6_stack.py          # greedy stack composition
OMP_NUM_THREADS=8 python experiments/t7_hpo.py   # HPO with fresh-CV validation
```

## Repeating on a second machine (hardware comparison)

The pipeline is serial by design and deterministic given the pinned
requirements, so gate scores should reproduce across machines to within BLAS
noise (watch the third decimal). To compare hardware properly:

1. Clone, create the venv from `requirements.txt`, download the data.
2. `python scripts/env_info.py --out results/env_<hostname>.json` on each
   machine, and commit the files.
3. Run the quick start above. Compare:
   - `train.py` CV printout (should match 0.8361 +/- small BLAS noise)
   - `pytest` results (the reference band test guards against version drift)
   - per evaluation wall clock, from the timestamps in `results/runner.log`
4. For the full comparison, run the pipeline on both machines and diff
   `results/results.jsonl` event by event.

On this project's original hardware (Xeon E5-2699 v4, 44 threads, dual RTX
A4500), the winning configuration turned out to be serial single process.
The bottleneck at 891 rows is information in the data, not FLOPs. A second
machine with different silicon is a useful check on reproducibility, not a
speedup.

## What the documents contain

- `docs/LEARNING_JOURNEY.md`: the story. What was tried, what the numbers
  said, and the seven lessons that transfer to any ML project.
- `docs/TASKS.md`: the spec. Every task with exact configurations, search
  spaces, gate criteria, results with p values, and reproduce commands.

## The short version of the findings

1. Single CV runs on small data lie. Repeated CV plus paired tests, or it
   did not happen.
2. A feature's value is model dependent; a model's value is ensemble
   dependent. Measure at the level you ship.
3. Ensembles buy orthogonal errors, not individual accuracy. Two solo-0.832
   gradient boosters were rejected from the stack three ways (p around 0.09).
4. Hyperparameter tuning gave +1.4% and +2.2% on fresh CV for the single
   models, and nothing significant in the ensemble (p = 0.74).
5. Seed averaging, ticket groups, iterative imputation, fare hygiene: all
   rejected by the gates. Famous tricks mostly duplicate existing signal.
6. The public leaderboard scored five models spanning CV 0.8134 to 0.8406
   inside one 0.746 to 0.766 band. Model selection against it is fitting to
   noise.
7. The top of the Titanic leaderboard (73 teams at 1.00000) is overfit or
   junk. The legitimate ceiling is about 0.84. The nested estimate says this
   pipeline is essentially there.

## Engineering note

Parallelizing tiny fits across processes fought this project for a full
day: loky (fork and spawn) and a persistent fork pool each eventually lost
worker processes silently, hanging `pool.map` without an error, always once
XGBoost, LightGBM, or CatBoost entered the workload (OpenMP runtimes and
forked workers do not mix reliably here). The final design is serial across
folds with OpenMP parallel fits. Slower per evaluation, deterministic,
hang proof.

## Files of record

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
