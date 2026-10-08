# Hybrid Recommendation Engine with Cold-Start Handling

A movie recommender on **MovieLens 1M** that blends collaborative filtering (SVD) with a
content-based model (genres + decade). The blend weight **adapts to how much interaction
history a user and a movie have**, so the system degrades gracefully for new users and new movies
instead of breaking.

See [`WRITEUP.md`](WRITEUP.md) for the approach, key decisions, results, and failure analysis.

## Headline results

All numbers are top-10 ranking metrics (NDCG@10) on each user's 5 most recent ratings, held out
by time. A movie counts as relevant if the user rated it 4 or 5.

| Users | Popularity | CF only (SVD) | Content + popularity | **Hybrid** |
|---|---|---|---|---|
| Cold: 1 rating | 0.0159 | 0.0166 | 0.0154 | 0.0141 |
| Cold: 3 ratings | 0.0164 | 0.0191 | 0.0161 | **0.0200** |
| Cold: 5 ratings | 0.0169 | 0.0195 | 0.0171 | **0.0200** |
| Cold: 10 ratings | 0.0189 | 0.0263 | 0.0201 | 0.0259 |
| Warm: full history | 0.0357 | 0.0741 | 0.0427 | 0.0738 |

Cold **items** (150 movies with all ratings removed from training): the hybrid places a liked
cold movie in the top 10 of 150 with probability **0.152** vs **0.067** for random guessing.

Differences between methods in the cold rows are small and within noise for some rows; see the
write-up for an honest discussion.

## Try it yourself (Streamlit app)

The easiest way to explore the model is the interactive app — no command-line arguments, no
reading code, just click around.

```bash
pip install -r requirements.txt   # make sure streamlit is installed
python train_and_save.py          # trains once on all the data, ~10 seconds, saves trained_model.pkl
streamlit run app.py              # opens the app in your browser
```

What you can do in the app:

- **Pick any real MovieLens user** (by ID, or hit "random") and see the hybrid model's top picks,
  side-by-side with popularity-only, CF-only, and content-only recommendations.
- **Simulate a brand-new user** who has never rated anything: pick a few movies you like, rate
  them, and watch the recommendations update live. Because this "user" has zero training history,
  the adaptive blend weight drops to 0% CF automatically — this is the cold-start problem from the
  headline table, made tangible instead of just a number in a report.
- **Turn on diversity re-ranking** with a slider and watch the genre mix of the recommendations
  spread out instead of returning ten near-identical blockbusters (see "Enhancements" below).

If you skip `train_and_save.py`, the app will tell you to run it before it can start.

## Enhancements beyond the original write-up

Two things the write-up's limitations section flagged as "not implemented" or "optimistic" now have
working code:

1. **Diversity re-ranking (`diversity.py`).** The failure analysis found that 58% of recommendations
   came from just the 100 most popular movies. `diverse_top_k()` implements MMR (Maximal Marginal
   Relevance): it re-orders a model's top candidates so that each new pick is penalized for being too
   similar (by genre/decade vector) to what's already been chosen. A `diversity` slider (0 to 1)
   controls the trade-off; 0 is identical to the model's plain ranking.
2. **A fast path for real use (`train_and_save.py`).** The evaluation scripts intentionally retrain
   from scratch every run (they're measuring accuracy on fresh splits). For actually *using* the
   model — like the Streamlit app — retraining on every click is wasteful, so this script trains once
   on the full rating history and pickles the result; the app loads it in about a second.
3. **Live cold-start simulation (`HybridRecommender.scores_for_new_user` in `hybrid.py`).** The
   original code could only evaluate cold *users who already exist in MovieLens* with history
   truncated. The new method builds a taste profile on the fly from any dict of
   `{movie_id: star_rating}` — including a user who was never in the dataset at all — so the app can
   demo the cold-start blend in real time.

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
