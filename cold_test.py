import numpy as np
import pandas as pd

from data import load_data, temporal_split
from cf import SVDRecommender
from evaluate import evaluate

ratings, movies = load_data()
train, test = temporal_split(ratings)

# Pick 1000 random users to turn into "new users"
rng = np.random.default_rng(42)
cold_users = set(rng.choice(train["user"].unique(), size=1000, replace=False))
is_cold = train["user"].isin(cold_users)

# Popularity ranking, for comparison
pop_order = train["item"].value_counts().index.tolist()

def recommend_popular(user, seen, k):
    recs = []
    for item in pop_order:
        if item not in seen:
            recs.append(item)
            if len(recs) == k:
                break
    return recs

# Warm users = everyone we did NOT shrink
warm_users = list(set(train["user"]) - cold_users)

for n in [1, 3, 5, 10]:
    # keep only each cold user's FIRST n ratings (train is already in time order)
    cold_part = train[is_cold].groupby("user").head(n)
    train_cold = pd.concat([train[~is_cold], cold_part])

    model = SVDRecommender(n_factors=50).fit(train_cold)

    cf_cold = evaluate(model.recommend, train_cold, test, users=list(cold_users))
    pop_cold = evaluate(recommend_popular, train_cold, test, users=list(cold_users))
    print(f"\n--- Cold users with only {n} rating(s) ---")
    print("CF (SVD):  ", cf_cold)
    print("Popularity:", pop_cold)

# For comparison: the warm users, same model (last loop's model)
print("\n--- Warm users (full history) ---")
print("CF (SVD):  ", evaluate(model.recommend, train_cold, test, users=warm_users))