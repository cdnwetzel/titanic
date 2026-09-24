"""Train the final champion model on the Titanic data and write submission.csv.

Champion (tasks 5-14 selection, see README): StackingClassifier over
RF + HistGradientBoosting + LogisticRegression (LR meta-learner, passthrough),
median imputation + scaling + one-hot encoding.

HGB/RF hyperparameters from Optuna search (task 7), validated on a fresh
50-fold protocol (HGB 0.8320 vs 0.8176 untuned, RF 0.8305 vs 0.8085,
both p < 0.0001). They did not significantly change the *stack* (p=0.74)
but are kept for the validated single-model margin. Honest estimates:
50-fold CV 0.8370 (+/- 0.0036); nested CV with inner HPO 0.8406 (+/- 0.0073).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
categorical_features = ["Sex", "Embarked", "Title", "Deck"]
features = numeric_features + categorical_features

HGB_PARAMS = {  # task 7 Optuna, fresh-CV validated
    "learning_rate": 0.02220063792187545,
    "max_iter": 1165,
    "max_leaf_nodes": 8,
    "min_samples_leaf": 44,
    "l2_regularization": 4.0175271118619635,
    "max_bins": 77,
}
RF_PARAMS = {  # task 7 Optuna, fresh-CV validated
    "n_estimators": 661,
    "max_depth": 15,
    "min_samples_leaf": 3,
    "max_features": 0.44766588900099735,
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Must be applied to train and test identically."""
    df = df.copy()
    # Title from Name: "Braund, Mr. Owen Harris" -> "Mr"
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    # Family size (including the passenger themselves) and "travelling alone" flag
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Cabin deck; most cabins are unknown -> "U"
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    return df


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        (
            "classifier",
            StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(random_state=RANDOM_STATE, **RF_PARAMS)),
                    (
                        "hgb",
                        HistGradientBoostingClassifier(random_state=RANDOM_STATE, **HGB_PARAMS),
                    ),
                    ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=5,
                stack_method="predict_proba",
                passthrough=True,
            ),
        ),
    ]
)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

train = add_features(train)
test = add_features(test)

X = train[features]
y = train["Survived"]

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"CV accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

model.fit(X, y)
predictions = model.predict(test[features])

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
submission.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv with {len(submission)} predictions")
