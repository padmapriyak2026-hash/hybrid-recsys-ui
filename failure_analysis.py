import numpy as np
import pandas as pd

from data import load_data, temporal_split
from hybrid import HybridRecommender

K = 10
ratings, movies = load_data()
train, test = temporal_split(ratings)
title = movies.set_index("item")["title"].to_dict()

# Same cold users as before (seed 42), shrunk to 3 ratings; everyone else is warm
rng = np.random.default_rng(42)
cold_users = set(rng.choice(train["user"].unique(), size=1000, replace=False))
is_cold = train["user"].isin(cold_users)
tr = pd.concat([train[~is_cold], train[is_cold].groupby("user").head(3)])

h = HybridRecommender(k_user=50).fit(tr, movies)
seen = tr.groupby("user")["item"].apply(set).to_dict()
liked = test[test["rating"] >= 4].groupby("user")["item"].apply(set).to_dict()
n_hist = tr.groupby("user").size()
item_cnt = tr["item"].value_counts()

user_rows, item_rows = [], []
recs_by_user = {}
all_recs = {"hybrid": [], "CF only": []}

for u, rel in liked.items():
    s = seen.get(u, set())
    recs = h.recommend(u, s, K)
    recs_by_user[u] = recs
    all_recs["hybrid"] += recs
    all_recs["CF only"] += h.cf.recommend(u, s, K)

    hits = [r in rel for r in recs]
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(hits) if x)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(rel), K)))
    user_rows.append((u, n_hist[u], dcg / ideal, sum(hits)))
    for item in rel:
        item_rows.append((u, item, item in recs, item_cnt.get(item, 0)))

# 1) Results by history length
U = pd.DataFrame(user_rows, columns=["user", "n_hist", "ndcg", "hits"])
U["bucket"] = pd.cut(U["n_hist"], [0, 3, 30, 100, 300, 10000],
                     labels=["3 (cold)", "4-30", "31-100", "101-300", "300+"])
print("\n=== 1) By number of training ratings ===")
print(U.groupby("bucket", observed=True).agg(
    users=("user", "count"),
    NDCG=("ndcg", "mean"),
    share_with_zero_hits=("hits", lambda x: (x == 0).mean())).round(3))

# 2) Does it find niche movies, or only popular ones?
I = pd.DataFrame(item_rows, columns=["user", "item", "hit", "n_item"])
I["tier"] = pd.cut(I["n_item"], [-1, 4, 50, 500, 100000],
                   labels=["0-4 ratings", "5-50", "51-500", "500+"])
print("\n=== 2) Liked test movies, by how popular the movie is ===")
t = I.groupby("tier", observed=True).agg(count=("hit", "size"),
                                         found_in_top10=("hit", "mean"))
t["share_of_all"] = t["count"] / t["count"].sum()
print(t.round(3))

# 3) Coverage and popularity bias
top100 = set(item_cnt.index[:100])
print("\n=== 3) Coverage ===")
for name, recs in all_recs.items():
    print(f"{name:8s} distinct movies recommended: {len(set(recs))} of {len(h.items)}"
          f" | share of recommendations from the 100 most popular movies:"
          f" {np.mean([r in top100 for r in recs]):.2f}")

# 4) Real examples where we got zero hits
def show(u):
    hist = tr[tr["user"] == u].sort_values("rating", ascending=False).head(5)
    print(f"\nUser {u} ({n_hist[u]} ratings in training)")
    print("  Rated:", "; ".join(f"{title[i]} ({r})" for i, r in zip(hist["item"], hist["rating"])))
    print("  We recommended:", "; ".join(title[i] for i in recs_by_user[u][:5]))
    print("  They liked next:", "; ".join(title[i] for i in liked[u]))

bad = U[U["hits"] == 0]
print("\n=== 4) Examples with zero hits: cold users ===")
for u in bad[bad["n_hist"] == 3].sample(2, random_state=1)["user"]:
    show(u)
print("\n=== 4) Examples with zero hits: warm users ===")
for u in bad[bad["n_hist"] >= 100].sample(2, random_state=1)["user"]:
    show(u)