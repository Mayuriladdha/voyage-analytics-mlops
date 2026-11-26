"""Flask REST API for Voyage Analytics.

Endpoints
  GET  /health            liveness/readiness probe (used by Kubernetes)
  GET  /routes            list of valid origin/destination pairs
  POST /predict           flight price (regression)  - single JSON object or a list
  POST /predict_gender    gender class (classification)
  GET  /recommend         hotel suggestions: /recommend?userCode=12&k=3
"""
import json
import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import FEATURES  # noqa: E402
from train_recommender import recommend  # noqa: E402

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
app = Flask(__name__)

price_model = joblib.load(os.path.join(MODEL_DIR, "flight_price_model.joblib"))
with open(os.path.join(MODEL_DIR, "routes.json")) as fh:
    ROUTES = json.load(fh)
VALID_TYPES = {"economic", "premium", "firstClass"}
VALID_AGENCIES = {"Rainbow", "CloudFy", "FlyingDrops"}


def _optional(name):
    path = os.path.join(MODEL_DIR, name)
    return joblib.load(path) if os.path.exists(path) else None


gender_bundle = _optional("gender_model.joblib")
reco_model = _optional("recommender.joblib")


def _prepare_flight(payload):
    """Validate one request body and turn it into a model-ready row."""
    missing = [k for k in ("from", "to", "flightType", "agency", "date") if k not in payload]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    key = f"{payload['from']}|{payload['to']}"
    if key not in ROUTES:
        raise ValueError(f"unknown route '{key}'; see GET /routes")
    if payload["flightType"] not in VALID_TYPES:
        raise ValueError(f"flightType must be one of {sorted(VALID_TYPES)}")
    if payload["agency"] not in VALID_AGENCIES:
        raise ValueError(f"agency must be one of {sorted(VALID_AGENCIES)}")
    date = pd.to_datetime(payload["date"], errors="coerce")
    if pd.isna(date):
        raise ValueError("date must look like 2023-05-17 or 05/17/2023")
    return {"from": payload["from"], "to": payload["to"], "flightType": payload["flightType"],
            "agency": payload["agency"], "distance": ROUTES[key],  # distance comes from the route
            "month": int(date.month), "dayofweek": int(date.dayofweek)}


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/routes")
def routes():
    return jsonify(sorted(k.replace("|", " -> ") for k in ROUTES))


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if body is None:
        return jsonify(error="send a JSON body"), 400
    items = body if isinstance(body, list) else [body]
    try:
        rows = [_prepare_flight(i) for i in items]
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    preds = price_model.predict(pd.DataFrame(rows)[FEATURES])
    out = [round(float(p), 2) for p in preds]
    return jsonify(predicted_price=out[0] if not isinstance(body, list) else out)


@app.post("/predict_gender")
def predict_gender():
    if gender_bundle is None:
        return jsonify(error="gender model not available"), 503
    body = request.get_json(silent=True) or {}
    if "age" not in body or "company" not in body:
        return jsonify(error="age and company are required"), 400
    row = {**gender_bundle["defaults"], **body}
    pipe = gender_bundle["pipeline"]
    proba = pipe.predict_proba(pd.DataFrame([row]))[0]
    return jsonify(predicted_gender=str(pipe.classes_[proba.argmax()]),
                   probabilities={c: round(float(p), 3) for c, p in zip(pipe.classes_, proba)})


@app.get("/recommend")
def recommend_hotels():
    if reco_model is None:
        return jsonify(error="recommender not available"), 503
    try:
        user, k = int(request.args["userCode"]), int(request.args.get("k", 3))
    except (KeyError, ValueError):
        return jsonify(error="userCode (int) is required; k optional int"), 400
    return jsonify(userCode=user, recommendations=recommend(reco_model, user, k))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
