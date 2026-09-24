"""Feature engineering invariants."""

import pandas as pd
import pytest

from src.features import add_base_features, load_data


def test_base_shapes():
    train, test = load_data()
    assert train.shape == (891, 16)
    assert test.shape == (418, 15)  # no Survived column


def test_engineered_columns_exist():
    train, _ = load_data()
    for col in ["Title", "FamilySize", "IsAlone", "Deck"]:
        assert col in train.columns


def test_title_vocabulary_is_closed():
    train, _ = load_data()
    assert set(train["Title"].unique()) <= {"Mr", "Mrs", "Miss", "Master", "Rare"}


def test_deck_unknown_class_present():
    train, _ = load_data()
    assert "U" in set(train["Deck"].unique())


def test_family_size_is_consistent():
    train, _ = load_data()
    assert (train["FamilySize"] == train["SibSp"] + train["Parch"] + 1).all()
    assert set(train["IsAlone"].unique()) <= {0, 1}


def test_add_base_features_is_non_mutating():
    df = pd.read_csv("data/train.csv")
    before = df.columns.tolist()
    add_base_features(df)
    assert df.columns.tolist() == before


def test_ticket_features():
    train, test = load_data(ticket=True)
    for df in (train, test):
        assert (df["TicketGroupSize"] >= 1).all()
        assert pd.api.types.is_string_dtype(df["TicketPrefix"])
    # a known party: the Osborne family shares ticket 2142 variants
    party = train[train["Ticket"].str.replace(" ", "", regex=False) == "347082"]
    assert party["TicketGroupSize"].iloc[0] >= 4


def test_fare_per_person_requires_ticket():
    with pytest.raises(ValueError):
        load_data(fare_per_person=True, ticket=False)


def test_hygiene_zeroes_fares():
    train, _ = load_data(hygiene=True)
    assert (train["Fare"].dropna() > 0).all()
