"""
Train the hybrid model once on ALL the data and save it to disk.

Why this file exists: every other script in this project (run_hybrid.py,
cold_test.py, etc.) retrains a fresh model every time it runs, because they
are *measuring accuracy* and need a clean train/test split for that. The
Streamlit app (app.py) is different -- it just wants to hand out real
recommendations as fast as possible, and it shouldn't have to retrain a
model (a minute or two) every time someone moves a slider.

So this script trains one model on the full rating history and pickles it
to trained_model.pkl. The app then just loads that file, which takes a
second or two.

Usage:
    python train_and_save.py
    streamlit run app.py
"""

import pickle

from data import load_data
from hybrid import HybridRecommender

MODEL_PATH = "trained_model.pkl"


def main():
    print("Loading MovieLens data...")
    ratings, movies = load_data()

    print("Training the hybrid model on all ratings (this can take a minute)...")
    model = HybridRecommender().fit(ratings, movies)

    print(f"Saving model and data to {MODEL_PATH} ...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "ratings": ratings, "movies": movies}, f)

    print("Done. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
