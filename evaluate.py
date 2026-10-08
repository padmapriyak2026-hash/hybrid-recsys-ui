import numpy as np

def metrics_for_user(recommended, relevant, k):
    """recommended: ordered list of item ids. relevant: set of liked test items."""
    hits = [1 if item in relevant else 0 for item in recommended[:k]]
    n_hits = sum(hits)

    precision = n_hits / k
    recall = n_hits / len(relevant)

    dcg = sum(h / np.log2(rank + 2) for rank, h in enumerate(hits))
    ideal = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    ndcg = dcg / ideal

    return precision, recall, ndcg

def evaluate(recommend_fn, train, test, users=None, k=10):
    """recommend_fn(user, seen_items, k) -> ordered list of k item ids."""
    seen = train.groupby("user")["item"].apply(set).to_dict()
    relevant = (test[test["rating"] >= 4]
                .groupby("user")["item"].apply(set).to_dict())

    if users is None:
        users = list(relevant)
    else:
        users = [u for u in users if u in relevant]

    results = []
    for u in users:
        recs = recommend_fn(u, seen.get(u, set()), k)
        results.append(metrics_for_user(recs, relevant[u], k))

    p, r, n = np.mean(results, axis=0)
    return {"users": len(users), f"P@{k}": round(p, 4),
            f"R@{k}": round(r, 4), f"NDCG@{k}": round(n, 4)}
