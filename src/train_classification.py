"""Gender classification (male / female / none) from user profile + travel behaviour.

Usage: PYTHONPATH=src python src/train_classification.py
"""
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import MODEL_DIR, load_flights, load_hotels, load_users

NUM = ["age", "n_flights", "avg_flight_price", "avg_distance", "share_first_class",
       "n_hotel_stays", "avg_hotel_total", "avg_stay_days"]
CAT = ["company"]


def build_user_table():
    users, flights, hotels = load_users(), load_flights(), load_hotels()
    f = flights.groupby("userCode").agg(
        n_flights=("travelCode", "size"), avg_flight_price=("price", "mean"),
        avg_distance=("distance", "mean"),
        share_first_class=("flightType", lambda s: (s == "firstClass").mean()))
    h = hotels.groupby("userCode").agg(
        n_hotel_stays=("travelCode", "size"), avg_hotel_total=("total", "mean"),
        avg_stay_days=("days", "mean"))
    t = users.rename(columns={"code": "userCode"}).merge(f, on="userCode", how="left") \
             .merge(h, on="userCode", how="left")
    return t.fillna(0)


def main():
    t = build_user_table()
    X, y = t[NUM + CAT], t["gender"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    prep = ColumnTransformer([("num", StandardScaler(), NUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6,
                                                min_samples_leaf=5, random_state=42),
    }
    results, fitted = {}, {}
    for name, m in models.items():
        pipe = Pipeline([("prep", prep), ("model", m)]).fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        results[name] = {"accuracy": float(accuracy_score(y_te, p)),
                         "macro_f1": float(f1_score(y_te, p, average="macro"))}
        fitted[name] = pipe
        print(name, results[name])
    best = max((k for k in results if k != "majority_baseline"),
               key=lambda k: results[k]["macro_f1"])
    print(classification_report(y_te, fitted[best].predict(X_te)))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"pipeline": fitted[best], "defaults": X_tr[NUM].median().to_dict(),
                 "companies": sorted(t["company"].unique())},
                os.path.join(MODEL_DIR, "gender_model.joblib"), compress=3)
    with open(os.path.join(MODEL_DIR, "classification_metrics.json"), "w") as fh:
        json.dump({"best_model": best, "results": results}, fh, indent=1)


if __name__ == "__main__":
    main()
