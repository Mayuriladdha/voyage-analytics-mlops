"""Voyage Analytics - Streamlit app: hotel recommendations, travel insights, price estimator.
Run from the repo root:  streamlit run streamlit_app/app.py
"""
import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import FEATURES  # noqa: E402
from train_recommender import recommend  # noqa: E402

st.set_page_config(page_title="Voyage Analytics", page_icon="✈️", layout="wide")


@st.cache_resource
def load_models():
    d = os.path.join(ROOT, "models")
    return (joblib.load(os.path.join(d, "recommender.joblib")),
            joblib.load(os.path.join(d, "flight_price_model.joblib")),
            json.load(open(os.path.join(d, "routes.json"))))


@st.cache_data
def load_data():
    d = os.path.join(ROOT, "data")
    return (pd.read_csv(os.path.join(d, "users.csv")), pd.read_csv(os.path.join(d, "hotels.csv")),
            pd.read_csv(os.path.join(d, "flights.csv")))


reco, price_model, routes = load_models()
users, hotels, flights = load_data()
st.title("✈️ Voyage Analytics: Integrating MLOps in Travel")
tab1, tab2, tab3 = st.tabs(["Hotel recommendations", "Travel insights", "Flight price estimator"])

with tab1:
    c1, c2 = st.columns([1, 2])
    user = c1.selectbox("User code", sorted(users["code"]))
    k = c1.slider("How many hotels?", 1, 5, 3)
    row = users[users.code == user].iloc[0]
    c1.caption(f"{row['name']} · {row['company']} · age {row['age']}")
    recs = pd.DataFrame(recommend(reco, int(user), k))
    c2.subheader("Suggested hotels")
    c2.dataframe(recs.rename(columns={"name": "hotel"}).round(1), hide_index=True)
    past = hotels[hotels.userCode == user].groupby("name").days.sum().sort_values(ascending=False)
    c2.subheader("Nights already spent")
    c2.bar_chart(past)

with tab2:
    a, b = st.columns(2)
    a.subheader("Average flight price by type")
    a.bar_chart(flights.groupby("flightType").price.mean())
    b.subheader("Hotel stays by place")
    b.bar_chart(hotels.groupby("place").size())
    a.subheader("Average hotel price/day by hotel")
    a.bar_chart(hotels.groupby("name").price.mean())
    flights["month"] = pd.to_datetime(flights["date"], format="%m/%d/%Y").dt.to_period("M").astype(str)
    b.subheader("Flights per month")
    b.line_chart(flights.groupby("month").size())

with tab3:
    c1, c2 = st.columns(2)
    route = c1.selectbox("Route", sorted(routes))
    ftype = c1.selectbox("Flight type", ["economic", "premium", "firstClass"])
    agency = c2.selectbox("Agency", ["Rainbow", "CloudFy", "FlyingDrops"])
    date = c2.date_input("Date")
    if st.button("Predict price"):
        origin, dest = route.split("|")
        x = pd.DataFrame([{"from": origin, "to": dest, "flightType": ftype, "agency": agency,
                           "distance": routes[route], "month": date.month,
                           "dayofweek": date.weekday()}])[FEATURES]
        st.metric("Estimated price", f"{price_model.predict(x)[0]:,.2f}")
