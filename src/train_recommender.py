"""Hotel recommender: item-based collaborative filtering on user x hotel stays,
with a popularity fallback for cold-start users.

Usage: PYTHONPATH=src python src/train_recommender.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from common import MODEL_DIR, load_hotels


def build_matrix(hotels):
    # implicit feedback = number of nights a user has spent in each hotel
    return hotels.pivot_table(index="userCode", columns="name", values="days",
                              aggfunc="sum", fill_value=0)


def fit(hotels):
    m = build_matrix(hotels)
    sim = pd.DataFrame(cosine_similarity(m.T), index=m.columns, columns=m.columns)
    info = hotels.groupby("name").agg(place=("place", "first"),
                                      avg_price_per_day=("price", "mean"),
                                      avg_stay_days=("days", "mean"),
                                      stays=("travelCode", "size"))
    popularity = info["stays"].sort_values(ascending=False)
    return {"matrix": m, "similarity": sim, "info": info, "popularity": popularity}


def recommend(model, user_code, k=3):
    m, sim, pop = model["matrix"], model["similarity"], model["popularity"]
    if user_code not in m.index:                      # cold start
        names = list(pop.index[:k])
    else:
        seen = m.loc[user_code]
        scores = sim.mul(seen, axis=0).sum(axis=0)    # similarity-weighted nights
        unseen = scores[seen == 0].sort_values(ascending=False)
        names = list(unseen.index[:k])
        if len(names) < k:                            # user visited (almost) everything
            names += [n for n in pop.index if n not in names][: k - len(names)]
    return model["info"].loc[names].reset_index().to_dict("records")


def evaluate(hotels, k=3, max_users=300):
    """Leave-last-hotel-out. Most users have visited 7-9 of the 9 hotels, which makes the
    task trivial, so we only use users with 3-5 distinct hotels (>= 5 unseen candidates).
    Baselines: random pick among unseen hotels, and most popular unseen hotels."""
    h = hotels.sort_values("date")
    first_visit = h.groupby(["userCode", "name"])["date"].min().reset_index()
    hits_cf = hits_pop = rand = n = 0
    for uid, g in first_visit.groupby("userCode"):
        if not 3 <= len(g) <= 5:
            continue
        held = g.sort_values("date").iloc[-1]["name"]
        train = h[~((h.userCode == uid) & (h.name == held))]
        model = fit(train)
        seen = set(model["matrix"].loc[uid][model["matrix"].loc[uid] > 0].index)
        unseen = [x for x in model["popularity"].index if x not in seen]
        hits_cf += held in [r["name"] for r in recommend(model, uid, k)]
        hits_pop += held in unseen[:k]
        rand += min(k, len(unseen)) / len(unseen)
        n += 1
        if n >= max_users:
            break
    return {"users_evaluated": n, f"hit_rate_at_{k}_cf": hits_cf / n,
            f"hit_rate_at_{k}_popularity_unseen": hits_pop / n,
            f"hit_rate_at_{k}_random_unseen": rand / n}


def main():
    hotels = load_hotels()
    metrics = evaluate(hotels)
    print(metrics)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(fit(hotels), os.path.join(MODEL_DIR, "recommender.joblib"), compress=3)
    with open(os.path.join(MODEL_DIR, "recommender_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)


if __name__ == "__main__":
    main()
