import numpy as np

from data import load_data, temporal_split
from hybrid import HybridRecommender

K = 10
ratings, movies = load_data()
train, test = temporal_split(ratings)

# 1) Choose cold items: movies that users liked in the test set at least 3 times
liked_test = test[test["rating"] >= 4]
counts = liked_test["item"].value_counts()
candidates = counts[counts >= 3].index.values
rng = np.random.default_rng(42)
cold_list = np.sort(rng.choice(candidates, size=min(150, len(candidates)),
                               replace=False))
cold_items = set(cold_list)

# 2) Pretend they are brand new: remove ALL their ratings from training
train_cold = train[~train["item"].isin(cold_items)]
h = HybridRecommender(k_user=50).fit(train_cold, movies)
cold_idx = np.array([h.i_idx[m] for m in cold_list])

# 3) For each (user, liked cold movie), rank it among the cold movies
events = liked_test[liked_test["item"].isin(cold_items)]
hits, ranks = [], []
for u, item in zip(events["user"], events["item"]):
    s = h.scores(u)[cold_idx]                      # hybrid scores, cold movies only
    true_pos = np.where(cold_list == item)[0][0]
    rank = int((s > s[true_pos]).sum()) + 1        # 1 = best
    hits.append(rank <= K)
    ranks.append(rank)

n = len(cold_list)
print(f"Cold movies: {n} | test events: {len(events)}")
print(f"Hybrid (content only for these): Hit@{K} = {np.mean(hits):.3f}, "
      f"median rank = {np.median(ranks):.0f} of {n}")
print(f"Random guessing:                 Hit@{K} = {K / n:.3f}, "
      f"median rank = {n / 2:.0f} of {n}")