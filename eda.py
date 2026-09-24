"""Quick first look at the Titanic training data."""

import pandas as pd

train = pd.read_csv("data/train.csv")

print(f"Shape: {train.shape} (rows, columns)\n")
print("Columns:", list(train.columns))
print("\nFirst 5 rows:")
print(train.head())

print("\nMissing values per column:")
print(train.isna().sum())

print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")

print("\nSurvival rate by Sex:")
print(train.groupby("Sex")["Survived"].mean().round(3))

print("\nSurvival rate by Passenger class (Pclass):")
print(train.groupby("Pclass")["Survived"].mean().round(3))

print("\nNumeric summary:")
print(train.describe().round(2))
