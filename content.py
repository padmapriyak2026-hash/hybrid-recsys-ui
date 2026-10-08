import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from data import load_data, temporal_split
from evaluate import evaluate


class ContentRecommender:
    def __init__(self, pop_weight=0.02):
        self.pop_weight = pop_weight

    def fit(self, train, movies):
        movies = movies.reset_index(drop=True)
        self.items = movies["item"].values
        self.i_idx = {m: i for i, m in enumerate(self.items)}

        # 1) Describe every movie with numbers: genres + decade
        genres = movies["genres"].str.get_dummies(sep="|")
        year = movies["title"].str.extract(r"\((\d{4})\)")[0].astype(float)
        decade = (year // 10 * 10).fillna(0).astype(int).astype(str)
        decades = pd.get_dummies(decade, prefix="dec")
        X = np.hstack([genres.values.astype(float),
                       0.5 * decades.values.astype(float)])
        self.X = normalize(X)            # each movie = one row of numbers

        # 2) Popularity, used only as a tiny tie-breaker
        counts = train["item"].value_counts().reindex(self.items).fillna(0).values
        self.pop = np.log1p(counts) / np.log1p(counts.max())

        # 3) User profile = rating-weighted average of the movies they rated
        self.users = sorted(train["user"].unique())
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        rows = train["user"].map(self.u_idx).values
        cols = train["item"].map(self.i_idx).values
        vals = train["rating"].values.astype(float)
        R = csr_matrix((vals, (rows, cols)),
                       shape=(len(self.users), len(self.items)))
        self.profiles = normalize(R @ self.X)
        return self

    def recommend(self, user, seen, k):
        scores = self.pop_weight * self.pop
        if user in self.u_idx:
            # cosine similarity between user's profile and every movie
            scores = scores + self.X @ self.profiles[self.u_idx[user]]
        seen_idx = [self.i_idx[m] for m in seen if m in self.i_idx]
        scores[seen_idx] = -np.inf       # never recommend already-watched movies
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.items[i] for i in top]


if __name__ == "__main__":
    ratings, movies = load_data()
    train, test = temporal_split(ratings)

    model = ContentRecommender().fit(train, movies)
    print("Content-based, full history:",
          evaluate(model.recommend, train, test, k=10))

    # Same 1000 cold users as before (same seed = same users)
    rng = np.random.default_rng(42)
    cold_users = set(rng.choice(train["user"].unique(), size=1000, replace=False))
    is_cold = train["user"].isin(cold_users)

    for n in [1, 3, 5, 10]:
        cold_part = train[is_cold].groupby("user").head(n)
        train_cold = pd.concat([train[~is_cold], cold_part])
        m = ContentRecommender().fit(train_cold, movies)
        print(f"\nCold users, {n} rating(s):",
              evaluate(m.recommend, train_cold, test, users=list(cold_users)))