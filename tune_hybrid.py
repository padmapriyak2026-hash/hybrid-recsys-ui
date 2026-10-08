import numpy as np
import pandas as pd

from data import load_data, temporal_split
from hybrid import HybridRecommender
from evaluate import evaluate

ratings, movies = load_data()
train, test = temporal_split(ratings)

# The group we REPORT on (seed 42) is never used for tuning
rng = np.random.default_rng(42)
report_users = set(rng.choice(train["user"].unique(), size=1000, replace=False))

# A different group of users, used only for tuning
rest = np.array(sorted(set(train["user"]) - report_users))
tune_users = set(np.random.default_rng(7).choice(rest, size=1000, replace=False))
is_tune = train["user"].isin(tune_users)

for n in [1, 3, 5, 10]:
    tr = pd.concat([train[~is_tune], train[is_tune].groupby("user").head(n)])
    h = HybridRecommender().fit(tr, movies)

    cf = evaluate(h.cf.recommend, tr, test, users=list(tune_users))["NDCG@10"]
    print(f"\nTuning users with {n} rating(s)   (CF only = {cf:.4f})")
    for k in [5, 10, 20, 50, 100]:
        h.k_user = k          # only changes the blend, so no refit is needed
        r = evaluate(h.recommend, tr, test, users=list(tune_users))
        print(f"  k_user={k:3d}: NDCG@10 = {r['NDCG@10']:.4f}")