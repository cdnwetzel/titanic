# What Titanic Taught Me About Evaluating Machine Learning

*A first Kaggle competition, worked as a controlled experiment in model
improvement: sixteen tasks, every "obvious win" put through a statistical
gate, five live submissions, and a leaderboard that could not see any of it.*

Author: Chris Wetzel. Twenty eight years in IT, three in AI, one in ML. I
am writing for people with strong technical backgrounds who are relatively
new to machine learning: the infrastructure instincts transfer, the
evaluation instincts have to be built.

**Final model:** stacking ensemble (random forest + gradient boosting +
logistic regression, logistic meta learner). 50 fold CV **0.8370 +/-
0.0036**; nested CV with the hyperparameter search inside the validation
loop **0.8406 +/- 0.0073**. Public leaderboard: somewhere in a 0.746 to
0.766 noise band, best displayed score held by the day one baseline.

If that contrast surprises you, this document is for you.

---

## Why I did it this way

Titanic is the tutorial level of competitive ML. The dataset is tiny (891
rows), thoroughly explored by a decade of kagglers, and close to its
information ceiling. That makes it perfect for what I actually wanted to
learn: not feature engineering tricks, but the discipline of knowing when
a change is real.

So every task in this journey, from "add classic engineered features" to
"run a proper hyperparameter search", was gated the same way:

- **Repeated cross validation** (5-fold by 10 repeats, 50 held-out
  evaluations) as the measuring stick. Single CV runs on small data lie,
  and I demonstrate that below.
- **Paired t tests on per fold scores** for every accept/reject decision
  (improvement greater than 0.1% and p under 0.05, with a cheaper 25 fold
  screen and 50 fold retest for borderline cases).
- **Fresh fold re validation** for anything found by search, so tuning
  could not overfit the evaluation protocol itself.
- **Nested CV** (search inside each outer fold) for the final honest
  number.
- **Leaderboard submissions spent like money**: 5 of the daily 10, never
  for model selection.

The full pipeline is in the repo (`pipeline/run_all.py`, `src/`,
`results/results.jsonl`); this document is the story and the lessons.

## Results at a glance

| Step | Change | 50 fold CV | Public LB |
|---|---|---|---|
| Baseline | Random forest, 7 simple features | 0.8134 | 0.76555 |
| Tasks 1 to 2 | Classic features, group imputation | 0.8153 | 0.74641 |
| Task 3 | Gradient boosting, full features | 0.8203 | 0.75358 |
| Task 4 | Soft voting ensemble | 0.8314 | 0.76076 |
| Task 5 | **Stacking (LR meta learner)** | **0.8365** | 0.75837 |
| Task 7 | Optuna tuned ensemble members | 0.8370 | 0.75837 |
| Task 14 | Nested CV estimate of final pipeline | 0.8406 +/- 0.0073 | 0.75837 |
| Tasks 6, 8 to 13 | Six more serious attempts | all rejected | 0.75837 |

The CV line climbs 2.7% and the leaderboard never notices. That is the
experiment.

---

## The journey

### Baseline (0.8134)

Random forest on `Pclass, Age, SibSp, Parch, Fare, Sex, Embarked`, median
and mode imputation inside a leak free sklearn `Pipeline`. Nothing clever;
everything reproducible. This number became the bar every later step had to
clear statistically, not narratively.

### Tasks 1 to 2: the first mirage

I added the classic engineered features: `Title` from `Name`,
`FamilySize`/`IsAlone` from `SibSp`/`Parch`, `Deck` from `Cabin`, plus
group based Age imputation (median per Title by Pclass).

A single 5 fold CV run said `+FamilySize` improved the model: 0.8104 versus
0.8059. Looked like a win. Repeated CV (50 fits) said the opposite: the
baseline stayed on top at 0.8134 and every feature variant sat within
noise. Group imputation (Title by Pclass, about 12 groups) was the only
accepted change, worth about +0.2%.

**Lesson 1: single CV runs lie.** On 891 rows, the standard error of a
5 fold estimate is roughly +/-1.5%. Confirm every improvement with
repeated, paired evaluation before believing it. This lesson shaped the
rest of the project.

### Task 3: the right model unlocks the features

HistGradientBoosting on the full feature set: 0.8203. The features from
task 1 that the random forest could not use, boosting exploited through
interactions the bagged trees averaged away.

**Lesson 2: a feature's value is model dependent.** "Feature engineering
did not work" often means "the model could not express what the feature
encodes."

### Task 4: ensembles buy diversity, not accuracy

Soft voting over RF + HGB + LR: 0.8314 (+1.1% over HGB alone). The
logistic regression, clearly the weakest member solo, contributed the
most, because its errors look different from the trees'. Up weighting the
strongest member made things worse; uniform weights won.

### Task 5: stacking beats voting (0.8365, p = 0.0005)

A `StackingClassifier` with a logistic meta learner trained on out of
fold predictions, plus `passthrough`, so the meta learner sees the raw
features alongside base predictions. First unambiguous win of the second
phase.

### Task 6: the most instructive failure

This is where the project got interesting. I brought in the modern gradient
boosters as new ensemble members. Solo scores:

| Candidate | 50 fold CV alone |
|---|---|
| XGBoost | 0.8319 |
| CatBoost | 0.8321 |
| SVC (rbf) | 0.8294 |
| MLP (small) | 0.8282 |
| LightGBM | 0.8146 |

XGBoost and CatBoost alone matched or beat the entire previous ensemble.
Surely adding them to the stack would help?

No. Individually: +0.0021 (p = 0.071) and +0.0024 (p = 0.088). Jointly:
+0.0025 (p = 0.094). MLP, SVC, LightGBM: negative. All rejected.

**Lesson 3: a stack wants orthogonal errors, not strong members.** XGBoost
and CatBoost are excellent models whose mistakes correlate with HGB's. The
meta learner already had that information; the new members added variance,
not signal. "Add your best models to the ensemble" is half the advice; the
other half is "add models that are wrong in new ways."

### Task 7: big solo gains that vanish in the ensemble

Optuna search (100 trials) on HGB and RF, each winner re validated on a
fresh 50 fold protocol the search never saw:

- HGB: **0.8320 versus 0.8176 untuned (+1.44%, p < 0.0001)**
- RF: **0.8305 versus 0.8085 untuned (+2.20%, p < 0.0001)**

Tuned random forest alone nearly matched the previous whole ensemble. In
the stack: 0.8370 versus 0.8365, p = 0.74. No significant change.

**Lesson 4: single model improvements do not transfer to ensembles
automatically.** The ensemble was already extracting what tuning provided.
Measure at the level you actually ship.

### Task 8: nothing to average

Seed averaging (5 seeds per stochastic member): diff 0.0000, p = 1.000. At
this scale, with regularized configurations, seed variance is below the
noise floor. Cheap to test, and the simpler model wins by default.

### Tasks 9 to 13: the folklore features do not clear the bar

Ticket group features (-0.0019, p = 0.36), iterative imputation (-0.0024),
fare hygiene (+0.0011, p = 0.21). Fare per person and target encoding were
skipped automatically: their dependency (task 9) had been rejected, and the
runner knew it.

**Lesson 5: famous dataset tricks mostly duplicate what simpler features
already encode.** Let the gates, not the folklore, decide. Also: encode
task dependencies in your runner, or you will KeyError at 3 a.m.

### Task 14: the honest number

Final pipeline, re evaluated with the HPO running inside every outer fold
(10 outer folds by a 15 trial inner search): **0.8406 +/- 0.0073**.

Slightly above the flat 50 fold estimate, so the search never overfit the
evaluation protocol. This is the most trustworthy estimate of the
pipeline's true skill, and it says: essentially at Titanic's legitimate
ceiling (about 0.84; the 73 perfect scores atop the leaderboard are
overfit or junk entries).

### Task 15: the discipline, held throughout

Every decision above is a paired statistical test on identical folds. Not
one "it improved" claim in this document rests on a single CV run or a
leaderboard wiggle.

### Task 16: the leaderboard cannot see any of this

Five submissions, in chronological order:

| Model | Public score |
|---|---|
| Baseline random forest | 0.76555 |
| Group imputation | 0.74641 |
| Gradient boosting | 0.75358 |
| Soft voting ensemble | 0.76076 |
| Final tuned stack (CV 0.8406) | 0.75837 |

A genuine 2.7% improvement in controlled evaluation, invisible on the
public leaderboard. Every score lands in a two point band that, on a few
hundred scored rows, is about eight passengers of noise. The displayed
rank (about 7,390 of 10,200 teams) is still held by the baseline, meaning
by luck of the draw.

**Lesson 6: the public leaderboard is a noisy oracle.** Model selection
against it is fitting to noise. The working loop that transfers to every
competition: build, then repeated CV, then believe only consistent paired
gains, then spend LB submissions sparingly as coarse sanity checks.

## The transferable lessons

1. **Single CV runs lie.** Repeated CV and paired tests, or it did not
   happen.
2. **A feature's value is model dependent**; a model's value is ensemble
   dependent. Measure at the level you ship.
3. **Ensembles buy orthogonal errors, not individual accuracy.**
4. **HPO gains may not transfer** into your ensemble; re gate at ensemble
   level.
5. **Re validate tuned configs on fresh folds**; report nested CV for the
   final number, because it is the only estimate HPO cannot inflate.
6. **At small scale, seed variance and hygiene fixes are below the noise
   floor.** Test them; expect rejection.
7. **The leaderboard is a lighthouse glimpsed through fog.** CV is the
   compass.

## An honest engineering footnote

The unglamorous part of this project: parallelizing tiny fits across
processes fought me all day. loky (both fork and spawn start methods) and a
persistent fork pool each worked for a while and then silently lost worker
processes, no error, just a hung `pool.map`, always once XGBoost,
LightGBM, or CatBoost entered the workload (OpenMP runtimes and forked
workers do not mix reliably). The final runner is **serial** with
OpenMP parallel fits. Slower per evaluation, deterministic, hang proof.

The hardware irony: this ran on a dual GPU, 44 thread workstation, and the
winning configuration uses one core at a time. Titanic scale ML is not a
compute problem; the scarce resource is information in 891 rows. GPUs and
dozens of cores start mattering when the model is a neural net or the data
stops fitting in cache; neither applies here, and no amount of hardware
fixes that.

Postscript: the project later ran bit-for-bit identically on a second
workstation (Ryzen 9 5950X) about 3x faster, confirming that the serial,
seeded design makes results a property of the code, not the silicon. The
full comparison: `docs/HARDWARE_COMPARISON.md`.

## Reproduce

```bash
pip install -r requirements.txt
kaggle competitions download -c titanic -p data && cd data && unzip titanic.zip && cd ..
python scripts/env_info.py --out results/env_$(hostname).json
python train.py                     # final champion, writes submission.csv
python -m pytest tests              # environment verification
OMP_NUM_THREADS=8 python -m pipeline.run_all 2>&1 | tee results/runner.log
```

Every gate decision lands in `results/results.jsonl`; per fold score arrays
are saved next to it; the champion configuration is
`results/champion.json`.

## What I would carry to a real competition

The same harness, scaled up: leak free pipelines, repeated and paired CV,
fresh fold validation for anything found by search, nested CV for final
numbers, submission budget treated as finite. The models change with the
data; the discipline does not.

---

*Repo: [github.com/cdnwetzel/titanic](https://github.com/cdnwetzel/titanic).
Data: Kaggle Titanic, Machine Learning from Disaster (download via
`kaggle competitions download -c titanic`; not redistributed here per
competition rules). Every claim in this document is backed by a paired
statistical test; the rejections are as real as the accepts.*
