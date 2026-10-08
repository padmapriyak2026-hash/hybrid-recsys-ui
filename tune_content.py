import numpy as np
import pandas as pd

from data import load_data, temporal_split
from content import ContentRecommender
from evaluate import evaluate

ratings, movies = load_data()
train, test = temporal_split(ratings)

rng = np.random.default_rng(42)
cold_users = set(rng.choice(train["user"].unique(), size=1000, replace=False))
is_cold = train["user"].isin(cold_users)

scenarios = {}
for n in [1, 5]:
    cold_part = train[is_cold].groupby("user").head(n)
    scenarios[f"cold, {n} rating(s)"] = (
        pd.concat([train[~is_cold], cold_part]), list(cold_users))
scenarios["warm, full history"] = (train, list(set(train["user"]) - cold_users))

for name, (tr, users) in scenarios.items():
    model = ContentRecommender().fit(tr, movies)
    print(f"\n{name}")
    for w in [0.02, 0.1, 0.3, 1.0, 3.0]:
        model.pop_weight = w
        r = evaluate(model.recommend, tr, test, users=users, k=10)
        print(f"  pop_weight={w}: NDCG@10 = {r['NDCG@10']:.4f}")