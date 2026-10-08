from data import load_data, temporal_split
from evaluate import evaluate

ratings, movies = load_data()
train, test = temporal_split(ratings)

pop_order = train["item"].value_counts().index.tolist()

def recommend_popular(user, seen, k):
    recs = []
    for item in pop_order:
        if item not in seen:
            recs.append(item)
            if len(recs) == k:
                break
    return recs

print("Popularity baseline:", evaluate(recommend_popular, train, test, k=10))