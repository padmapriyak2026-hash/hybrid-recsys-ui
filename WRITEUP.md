# Write-up: Hybrid Recommender with Cold-Start Handling

## 1. Problem and approach

Collaborative filtering (CF) cannot recommend for users or items with no interaction history.
I built a hybrid that combines CF with a content-based model and shifts weight toward content as
history thins out. Data: MovieLens 1M (6,040 users, 3,706 rated movies, 1,000,209 ratings).

**Evaluation setup.** Each user's 5 most recent ratings form the test set; earlier ratings are
training (970,009 train / 30,200 test). A test movie is relevant if rated 4 or 5. Already-watched
movies are never recommended. Metrics are Precision@10, Recall@10 and NDCG@10, averaged over users
with at least one relevant test item. Ratings are treated as a ranking problem, not rating prediction.

**Models.**
- *Popularity baseline:* the most-rated unseen movies for everyone.
- *CF:* truncated SVD on the user-item matrix. 20, 50 and 100 factors were compared; 50 was best
  (NDCG@10 0.0747 vs 0.0354 for popularity, about 2x).
- *Content:* movie = genre multi-hot + decade vector; user profile = rating-weighted average of
  rated movies; score = cosine similarity + `pop_weight` x popularity prior.
- *Hybrid:* `score = a * z(CF) + (1 - a) * z(content)` with
  `a = n_user/(n_user + 50) * n_item/(n_item + 5)`. Scores are z-normalized per user so the two
  scales are comparable. A user or item with no history gets `a = 0`, so content decides. Unlike a
  fixed 50/50 mix, trust in CF grows smoothly with evidence on both the user and the item side.

## 2. Key decisions

1. **Temporal per-user split** rather than random, to avoid training on the future.
2. **Cold-start is simulated**, because every MovieLens-1M user has at least 20 ratings. Cold users:
   1,000 random users (seed 42) keep only their first 1, 3, 5 or 10 training ratings. Cold items:
   150 movies liked at least 3 times in the test set have all their training ratings removed.
3. **Content model needed a popularity prior.** Genre vectors create huge ties, and pure content
   scored *below* the popularity baseline (NDCG@10 0.0039 at 1 rating vs 0.0159). Raising the
   popularity weight from 0.02 to 3.0 lifted it to roughly popularity level (0.0154).
4. **Blend parameter tuned on separate users.** `k_user` was tuned on a different 1,000 users (seed 7)
   from the ones reported. The tuning favored slower CF trust (`k_user` 50 to 100 over my initial
   guess of 20); I used 50.

## 3. Results

NDCG@10 (P@10 and R@10 are in the repository output; they rank the methods the same way):

| Users | Popularity | CF only | Content + pop | Hybrid |
|---|---|---|---|---|
| Cold: 1 rating (933 users) | 0.0159 | 0.0166 | 0.0154 | 0.0141 |
| Cold: 3 ratings | 0.0164 | 0.0191 | 0.0161 | **0.0200** |
| Cold: 5 ratings | 0.0169 | 0.0195 | 0.0171 | **0.0200** |
| Cold: 10 ratings | 0.0189 | 0.0263 | 0.0201 | 0.0259 |
| Warm, full history (4,620 users) | 0.0357 | 0.0741 | 0.0427 | 0.0738 |

**Cold items:** with all ratings removed, the hybrid placed the liked movie in the top 10 of 150
candidates 15.2% of the time vs 6.7% for random (median rank 54 vs 75), over 1,558 events. CF cannot
score these items at all, so this isolates the content contribution.

**Reading the results honestly.**
- CF alone collapses for cold users: at 1 rating it is no better than popularity (0.0166 vs 0.0159),
  and at 10 ratings it reaches only about a third of its warm-user score (0.0263 vs 0.0741, although
  these are different user groups).
- The hybrid matches CF for warm users (0.0738 vs 0.0741), so adaptivity costs nothing there. It is
  best at 3 to 5 ratings, by a small margin (about 3 to 5% over CF), and tied at 10.
- **At 1 rating the hybrid is worse than CF** on the reported users, but better on the separate
  tuning users. With about 930 users per group and no confidence intervals, I treat differences of
  this size as noise and claim only that the hybrid tracks the best single method at each history length.
- The hybrid's main benefit is graceful behavior and the ability to rank brand-new items, not a
  large accuracy jump. Content+popularity is close to popularity alone, so the content features
  (genre and decade) add little beyond popularity.

## 4. Failure analysis (3 ratings for cold users, full history for the rest)

**By history.** Share of users with zero hits in the top 10: 91% for cold users (3 ratings), 58% for
4 to 30, 73% for 31 to 100, 80% for 101 to 300 and 83% for 300+. Heavy users do *worse* than light
ones (NDCG 0.048 vs 0.122). A plausible reason, which I did not test, is that heavy users have already
seen the popular movies, so what remains is the harder long tail.

**Popularity bias is the main failure.** Of liked test movies with more than 500 training ratings,
15.5% are found in the top 10; for movies with 51 to 500 ratings only 0.9%, and 0% below 50 ratings.
Only 889 of 3,883 movies are ever recommended, and 58% of recommendations come from the 100 most
popular movies (52% for CF alone). The popularity weight I tuned for accuracy also suppresses niche
items, an accuracy vs diversity trade-off.

**Examples (random zero-hit users):**
- *User 4044 (cold):* three ratings, all 3 stars or lower. No positive signal to personalize on, so
  the system showed popular classics.
- *User 3807 (cold):* rated horror and comedy; recommendations (Ghostbusters, Gremlins) were
  sensible, but the next liked movie (Demolition Man) was not predictable from three ratings.
- *User 1204 (warm, 131 ratings):* strong action/sci-fi taste (Empire Strikes Back, Terminator 2,
  Godfather), but recommendations drifted toward dramas and comedies. The blend diluted a clear taste.
- *User 3095 (warm):* romantic-comedy fan; recommendations matched that taste, but the next liked movies
  were newer releases (2000) with few training ratings.

A miss is not always a bad recommendation: unrated does not mean disliked, so offline metrics
understate quality. Why the hybrid still fails: genre and decade are too coarse to separate
within-genre taste; popularity dominates the content signal; cold users with no positive ratings give
nothing to personalize on; and recent releases are under-represented in training.

## 5. Limitations and next steps

- **Some hyperparameters were not tuned on separate data.** The SVD factor count was compared on the
  test set, and the content model's popularity weight (3.0) was tuned on the same cold and warm
  user groups that are reported. Only `k_user` was tuned on separate users. Reported numbers are
  therefore slightly optimistic; a proper validation split would fix this.
- One split, no confidence intervals; several cold-start differences are within likely noise.
- Content features are only genres and decade. Tags (MovieLens 25M) or text embeddings of plot
  descriptions would likely improve cold-item ranking.
- Not implemented: diversity or novelty re-ranking (e.g. MMR), which the popularity-bias finding
  motivates. The popularity baseline comparison is included throughout and quantifies the lift.
