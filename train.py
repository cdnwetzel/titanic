"""Train the final champion model on the Titanic data and write submission.csv.

Champion (tasks 5 to 14 selection, see docs/TASKS.md): StackingClassifier
over tuned RF + HistGradientBoosting + LogisticRegression (LR meta learner,
passthrough), median imputation, scaling, one hot encoding.

Honest estimates: 50 fold CV 0.8370 (+/- 0.0036); nested CV with inner HPO
0.8406 (+/- 0.0073). Run from the repo root: python train.py
"""

import pandas as pd
from sklearn.model_selection import cross_val_score

from src.features import add_base_features, load_data
from src.model import CHAMPION_TUNED, build_stack, columns

model = build_stack(tuned=CHAMPION_TUNED)

train = add_base_features(pd.read_csv("data/train.csv"))
test = add_base_features(pd.read_csv("data/test.csv"))

num, cat = columns()
features = num + cat
X = train[features]
y = train["Survived"]

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

model.fit(X, y)
predictions = model.predict(test[features])

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
submission.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv with {len(submission)} predictions")
