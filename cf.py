import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from data import load_data, temporal_split
from evaluate import evaluate


class SVDRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors

    def fit(self, train):
        # Give every user and movie a row/column number
        self.users = sorted(train["user"].unique())
        self.items = sorted(train["item"].unique())
        self.u_idx = {u: i for i, u in enumerate(self.users)}
        self.i_idx = {m: i for i, m in enumerate(self.items)}

        rows = train["user"].map(self.u_idx).values
        cols = train["item"].map(self.i_idx).values
        vals = train["rating"].values.astype(float)

        # The big (mostly empty) user x movie table
        R = csr_matrix((vals, (rows, cols)),
                       shape=(len(self.users), len(self.items)))

        # Squeeze it into hidden "taste" factors
        U, s, Vt = svds(R, k=self.n_factors)
        self.user_f = U * s          # one row of 50 numbers per user
        self.item_f = Vt.T           # one row of 50 numbers per movie
        return self

    def recommend(self, user, seen, k):
        if user not in self.u_idx:
            return []
        scores = self.item_f @ self.user_f[self.u_idx[user]]
        # never recommend what they've already watched
        seen_idx = [self.i_idx[m] for m in seen if m in self.i_idx]
        scores[seen_idx] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.items[i] for i in top]


if __name__ == "__main__":
    ratings, movies = load_data()
    train, test = temporal_split(ratings)

    for n in [20, 50, 100]:
        model = SVDRecommender(n_factors=n).fit(train)
        print(f"SVD with {n} factors:", evaluate(model.recommend, train, test, k=10))