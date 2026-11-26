"""Train the flight price regression model, track every run in MLflow and
export the best pipeline as a joblib file for the Flask API.

Usage:  python src/train_regression.py
Env:    DATA_DIR, MODEL_DIR, MLFLOW_TRACKING_URI (default: local ./mlruns folder)
"""
import contextlib
import json
import os
import time

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import CATEGORICAL, FEATURES, MODEL_DIR, TARGET, add_date_features, load_flights

try:  # MLflow is optional locally, expected for the capstone run
    import mlflow
    import mlflow.sklearn
    HAVE_MLFLOW = True
except ImportError:
    HAVE_MLFLOW = False

EXPERIMENT = "voyage-flight-price"
REGISTERED_NAME = "flight-price-regressor"


def make_pipeline(model):
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",  # numeric columns go through unchanged
    )
    return Pipeline([("prep", prep), ("model", model)])


def candidates():
    return {
        "linear_regression": (LinearRegression(), {}),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(random_state=42),
            {"max_iter": 100},
        ),
        "random_forest": (
            RandomForestRegressor(n_estimators=60, max_depth=16, min_samples_leaf=3,
                                  n_jobs=-1, random_state=42),
            {"n_estimators": 60, "max_depth": 16, "min_samples_leaf": 3},
        ),
    }


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    df = add_date_features(load_flights())
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    if HAVE_MLFLOW:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
        mlflow.set_experiment(EXPERIMENT)

    results, fitted = {}, {}
    for name, (model, params) in candidates().items():
        run = mlflow.start_run(run_name=name) if HAVE_MLFLOW else contextlib.nullcontext()
        with run:
            t0 = time.time()
            pipe = make_pipeline(model).fit(X_tr, y_tr)
            metrics = evaluate(y_te, pipe.predict(X_te))
            metrics["train_seconds"] = round(time.time() - t0, 2)
            results[name], fitted[name] = metrics, pipe
            print(name, metrics)
            if HAVE_MLFLOW:
                mlflow.log_params({"model": name, "features": ",".join(FEATURES), **params})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipe, "model", input_example=X_te.head(3))

    best = min(results, key=lambda k: results[k]["rmse"])
    print("Best model:", best)

    # Register the winner so Jenkins/Kubernetes can refer to a model *version*
    if HAVE_MLFLOW:
        with mlflow.start_run(run_name=f"champion_{best}"):
            mlflow.log_params({"model": best, "role": "champion"})
            mlflow.log_metrics(results[best])
            try:
                mlflow.sklearn.log_model(fitted[best], "model",
                                         registered_model_name=REGISTERED_NAME)
            except Exception as exc:  # file store without registry support
                print("Model registry unavailable, logged without registering:", exc)
                mlflow.sklearn.log_model(fitted[best], "model")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(fitted[best], os.path.join(MODEL_DIR, "flight_price_model.joblib"), compress=3)
    routes = df.groupby(["from", "to"])["distance"].first().reset_index()
    route_map = {f"{r['from']}|{r['to']}": float(r["distance"]) for _, r in routes.iterrows()}
    with open(os.path.join(MODEL_DIR, "routes.json"), "w") as fh:
        json.dump(route_map, fh, indent=1)
    with open(os.path.join(MODEL_DIR, "regression_metrics.json"), "w") as fh:
        json.dump({"best_model": best, "results": results}, fh, indent=1)
    return best, results


if __name__ == "__main__":
    main()
