"""Feature engineering and data loading.

All transformations are pure functions of each row, so train and test are
processed identically. Ticket group sizes are computed from the concatenated
train and test identifiers only, never from the target.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

BASE_NUM = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
BASE_CAT = ["Sex", "Embarked", "Title", "Deck"]


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Title from Name: "Braund, Mr. Owen Harris" becomes "Mr"
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    return df


def load_data(ticket=False, fare_per_person=False, hygiene=False):
    """Load train/test with switchable feature groups (tasks 9, 10, 13).

    Flags compose only in dependency order: fare_per_person requires
    ticket (it divides by TicketGroupSize).
    """
    train = add_base_features(pd.read_csv("data/train.csv"))
    test = add_base_features(pd.read_csv("data/test.csv"))

    if ticket:
        def ticket_key(df):
            return df["Ticket"].str.replace(" ", "", regex=False)

        counts = ticket_key(pd.concat([train, test], ignore_index=True)).value_counts()
        for df in (train, test):
            df["TicketGroupSize"] = ticket_key(df).map(counts).astype(int)
            df["TicketPrefix"] = (
                df["Ticket"].str.replace(r"[0-9.\s]+$", "", regex=True)
                .str.rstrip("/.")
                .replace("", "none")
            )

    if hygiene:
        # Fare of zero is a placeholder, not a real fare; let imputation handle it
        for df in (train, test):
            df.loc[df["Fare"] <= 0, "Fare"] = np.nan

    if fare_per_person:
        if not ticket:
            raise ValueError("fare_per_person requires ticket=True (needs TicketGroupSize)")
        for df in (train, test):
            df["FarePerPerson"] = df["Fare"] / df["TicketGroupSize"]

    return train, test


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Impute a column with the median computed per group, falling back to
    the global median (task 2). Must sit inside the pipeline so group
    medians are computed on each training fold only."""

    def __init__(self, column, group_by):
        self.column = column
        self.group_by = group_by

    def _keys(self):
        return [self.group_by] if isinstance(self.group_by, str) else list(self.group_by)

    def fit(self, X, y=None):
        self.fallback_ = float(X[self.column].median())
        self.group_medians_ = X.groupby(self._keys())[self.column].median()
        return self

    def transform(self, X):
        X = X.copy()
        med = pd.Series(X.set_index(self._keys()).index.map(self.group_medians_), index=X.index)
        X[self.column] = X[self.column].fillna(med).fillna(self.fallback_)
        return X
