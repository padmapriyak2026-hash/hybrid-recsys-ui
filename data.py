import pandas as pd

def load_data():
    ratings = pd.read_csv("data/ml-1m/ratings.dat", sep="::", engine="python",
                          names=["user", "item", "rating", "ts"])
    movies = pd.read_csv("data/ml-1m/movies.dat", sep="::", engine="python",
                         names=["item", "title", "genres"], encoding="latin-1")
    return ratings, movies

def temporal_split(ratings, n_test=5):
    # sort each user's ratings oldest -> newest
    df = ratings.sort_values(["user", "ts"], kind="stable").copy()
    # number each rating counting backwards from that user's newest one (0 = newest)
    df["rank_from_end"] = df.groupby("user").cumcount(ascending=False)
    test = df[df["rank_from_end"] < n_test].drop(columns="rank_from_end")
    train = df[df["rank_from_end"] >= n_test].drop(columns="rank_from_end")
    return train, test

if __name__ == "__main__":
    ratings, movies = load_data()
    train, test = temporal_split(ratings)

    print("Train ratings:", len(train))
    print("Test ratings: ", len(test))

    # In the test set, a movie counts as "liked" if rated 4 or 5
    liked = test[test["rating"] >= 4]
    print("Liked test ratings (the ones we try to find):", len(liked))

    # Which test movies never appear in training? (true cold items)
    unseen = set(test["item"]) - set(train["item"])
    print("Test movies with zero training ratings:", len(unseen))