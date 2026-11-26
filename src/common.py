"""Shared helpers: paths, data loading and feature engineering for Voyage Analytics."""
import os
import pandas as pd

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# Regression feature definition (single source of truth for training AND serving)
CATEGORICAL = ["from", "to", "flightType", "agency"]
NUMERIC = ["distance", "month", "dayofweek"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price"


def load_users():
    return pd.read_csv(os.path.join(DATA_DIR, "users.csv"))


def load_flights():
    df = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


def load_hotels():
    df = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    return df


def add_date_features(df, date_col="date"):
    """Month and day-of-week from the flight date (year is excluded on purpose:
    a model should not extrapolate on a calendar year it has never seen)."""
    d = pd.to_datetime(df[date_col])
    out = df.copy()
    out["month"] = d.dt.month
    out["dayofweek"] = d.dt.dayofweek
    return out
