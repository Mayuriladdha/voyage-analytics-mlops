import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))
from app import app  # noqa: E402

client = app.test_client()
GOOD = {"from": "Recife (PE)", "to": "Florianopolis (SC)", "flightType": "firstClass",
        "agency": "FlyingDrops", "date": "2023-05-17"}


def test_health():
    assert client.get("/health").json == {"status": "ok"}


def test_predict_ok():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200 and r.json["predicted_price"] > 0


def test_first_class_costs_more_than_economic():
    eco = client.post("/predict", json={**GOOD, "flightType": "economic"}).json["predicted_price"]
    first = client.post("/predict", json=GOOD).json["predicted_price"]
    assert first > eco


def test_predict_batch():
    r = client.post("/predict", json=[GOOD, GOOD])
    assert len(r.json["predicted_price"]) == 2


def test_predict_bad_route():
    r = client.post("/predict", json={**GOOD, "to": "Mars"})
    assert r.status_code == 400


def test_recommend():
    r = client.get("/recommend?userCode=0&k=3")
    assert r.status_code == 200 and len(r.json["recommendations"]) == 3
