Hybrid Recommendation Engine with Cold-Start Handling
A movie recommendation system using MovieLens 1M that combines:
- Collaborative Filtering (SVD)
- Content-based filtering using genres and decade
- Popularity-based recommendations
The system automatically adjusts the balance between collaborative and content-based methods based on the user's rating history, helping solve the cold-start problem.
Key Results
- Hybrid model performs well for users with few ratings.
- For cold-start movies, the hybrid model achieves 15.2% top-10 probability, compared with 6.7% for random guessing.
- Performance improves significantly when full user history is available.
Streamlit App
The interactive app allows users to:
- Select a MovieLens user or create a new user.
- Compare recommendations from different methods.
- Add ratings and see recommendations update instantly.
- Adjust diversity to get more varied movie suggestions.
Main Enhancements
1. Diversity Re-ranking: Uses MMR to avoid repetitive recommendations.
2. Fast Training: Trains once and saves the model for quick app use.
3. Live Cold-Start: Creates recommendations for completely new users using their initial ratings.
## Setup

Requires Python 3.10+ (developed on 3.13).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

On Windows, if `python` is not found, use `py -m venv venv`. If activation is blocked, run
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

## Data

1. Download **ml-1m.zip** from https://grouplens.org/datasets/movielens/1m/
2. Unzip it so the files sit at:

```
data/ml-1m/ratings.dat
data/ml-1m/movies.dat
data/ml-1m/users.dat
```

The `data/` folder is not committed to the repository.

## How to reproduce the results

Run from the repository root with the virtual environment active. Every random choice uses a
fixed seed, so the outputs are deterministic.

| Step | Command | What it does | Approx. time |
|---|---|---|---|
| 1 | `python explore.py` | Dataset statistics (users, items, sparsity, cold items) | seconds |
| 2 | `python data.py` | Builds the temporal train/test split and prints its sizes | seconds |
| 3 | `python baseline.py` | Popularity baseline on all users | ~1 min |
| 4 | `python cf.py` | SVD with 20 / 50 / 100 factors | ~2 min |
| 5 | `python cold_test.py` | CF vs popularity as history is truncated (1, 3, 5, 10 ratings) | a few min |
| 6 | `python content.py` | Content-based model, full and cold history | a few min |
| 7 | `python tune_content.py` | Tunes the popularity weight of the content model | a few min |
| 8 | `python tune_hybrid.py` | Tunes the blend parameter `k_user` on a separate user group | a few min |
| 9 | `python run_hybrid.py` | **Main results table**: popularity vs CF vs content vs hybrid | a few min |
| 10 | `python failure_analysis.py` | Failure analysis by history, item popularity, coverage, examples | ~2 min |
| 11 | `python cold_items.py` | Cold-**item** test (150 movies removed from training) | ~1 min |

### Re-running only the evaluation

The main table comes from `python run_hybrid.py`. To change the blend, edit the defaults of
`HybridRecommender` in `hybrid.py` (`k_user`, `k_item`, `n_factors`, `pop_weight`) and re-run it.
To evaluate a new model, write `recommend(user, seen_items, k) -> list_of_item_ids` and pass it to
`evaluate()` in `evaluate.py`.

## Project structure

```
data.py              load MovieLens, per-user temporal split
evaluate.py          Precision@K, Recall@K, NDCG@K
baseline.py          popularity baseline
cf.py                SVD collaborative filtering (scipy)
content.py           content-based model: genres + decade, popularity prior
hybrid.py            adaptive blend of CF and content
cold_test.py         cold-user simulation: CF vs popularity
tune_content.py      tunes the content model's popularity weight
tune_hybrid.py       tunes the blend parameter on separate users
run_hybrid.py        final comparison table
failure_analysis.py  where and why the hybrid fails
cold_items.py        cold-item evaluation
explore.py           dataset statistics
diversity.py         MMR re-ranking so recommendations aren't all the same popular movies
train_and_save.py    trains once on all data and pickles the model for the app
app.py               Streamlit UI: existing users, brand-new cold-start demo, diversity slider
WRITEUP.md           approach, decisions, results, failure analysis
```

## Method in brief

- **Split:** each user's 5 most recent ratings are the test set; everything earlier is training.
  This avoids leaking the future into training.
- **CF:** truncated SVD (50 factors) on the user-item rating matrix; score = dot product of user
  and item factors.
- **Content:** each movie is a vector of genres and decade; a user's profile is the
  rating-weighted average of the movies they rated; score = cosine similarity plus a popularity prior.
- **Hybrid:** `score = a * z(CF) + (1 - a) * z(content)`, where
  `a = n_user / (n_user + 50) * n_item / (n_item + 5)`. A user or movie with no history gets zero
  CF weight, so content decides.
- **Cold-start slices:** cold users are simulated by keeping only the first 1, 3, 5 or 10 training
  ratings of 1,000 random users; cold items by removing all training ratings of 150 movies.

## Limitations

See the write-up. In short: the content features (genre, decade) are weak, differences between
methods in the cold-user rows are small, hyperparameters were tuned without a separate
validation split in places, and evaluation uses one split with no confidence intervals. The
diversity re-ranking above is a simple MMR pass, not a re-trained model, so it trades a little
accuracy for spread rather than solving popularity bias at the source.
Demo video: see `Video Project (1).mp4` in this repository.
