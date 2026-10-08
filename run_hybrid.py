import numpy as np
import pandas as pd

from data import load_data, temporal_split
from hybrid import HybridRecommender
from evaluate import evaluate

ratings, movies = load_data()
train, test = temporal_split(ratings)

# Same 1000 cold users as before (same seed)
rng = np.random.default_rng(42)
cold_users = set(rng.choice(train["user"].unique(), size=1000, replace=False))
is_cold = train["user"].isin(cold_users)
warm_users = list(set(train["user"]) - cold_users)

def make_popular(tr):
    order = tr["item"].value_counts().index.tolist()
    def recommend_popular(user, seen, k):
        recs = []
        for item in order:
            if item not in seen:
                recs.append(item)
                if len(recs) == k:
                    break
        return recs
    return recommend_popular

def compare(label, tr, users):
    h = HybridRecommender().fit(tr, movies)
    print(f"\n=== {label} ===")
    rows = [("Popularity", make_popular(tr)),
            ("CF only", h.cf.recommend),
            ("Content+pop only", h.content.recommend),
            ("HYBRID", h.recommend)]
    for name, fn in rows:
        r = evaluate(fn, tr, test, users=users, k=10)
        print(f"{name:18s} P@10={r['P@10']:.4f}  R@10={r['R@10']:.4f}  NDCG@10={r['NDCG@10']:.4f}")

for n in [1, 3, 5, 10]:
    cold_part = train[is_cold].groupby("user").head(n)
    train_cold = pd.concat([train[~is_cold], cold_part])
    compare(f"Cold users with {n} rating(s)", train_cold, list(cold_users))

compare("Warm users (full history)", train, warm_users)