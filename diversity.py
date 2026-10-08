"""
Diversity re-ranking (MMR: Maximal Marginal Relevance).

WRITEUP.md's failure analysis found a real problem: 58% of the hybrid's
recommendations come from just the 100 most popular movies. The model is
accurate, but it keeps showing everyone the same blockbusters instead of
digging into the long tail. This file fixes that -- it's an answer to the
"Not implemented: diversity or novelty re-ranking" line in the write-up's
limitations section.

The idea (MMR) is simple and doesn't need any new machine learning:
  1. Take the model's normal top scores (e.g. its top 50 candidates).
  2. Build the final list one movie at a time.
  3. Each time, pick the candidate that is a good mix of:
       - still scoring well, AND
       - not too similar to movies we've already picked.
  4. A "diversity" knob (0 to 1) controls that trade-off.
     diversity=0   -> ignore similarity completely (identical to plain top-k).
     diversity=1   -> only care about spreading out, ignore the score.

"Similar" is measured with cosine similarity on the content model's genre +
decade vectors, which is already sitting in the project (content.X), so no
extra data is needed.
"""

import numpy as np


def cosine_similarity(vector_a, vector_b):
    """A number from -1 to 1: 1 means "pointing the same way" (very similar
    movies), 0 means unrelated, -1 means opposite. Our genre vectors are
    non-negative, so in practice this stays between 0 and 1."""
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denom)


def diverse_top_k(scores, item_ids, feature_matrix, k, diversity=0.3, shortlist_size=50):
    """Re-rank a model's scores so the final top-k is less repetitive.

    Arguments:
        scores:         1D numpy array. scores[i] is the model's score for
                         item_ids[i]. Items already "seen" should already be
                         set to -inf by the caller, same as everywhere else
                         in this project.
        item_ids:       list/array of item ids, same order as `scores` and
                         the rows of `feature_matrix`.
        feature_matrix: 2D numpy array, one row of numbers per item, used
                         only to measure how similar two movies are to each
                         other (pass in content.X from content.py).
        k:              how many recommendations to return.
        diversity:      0 = plain top-k (no re-ranking). 1 = heavily favors
                         spreading out over raw score. 0.2-0.4 is a good
                         everyday range.
        shortlist_size: only consider the top N scoring items as candidates,
                         so this stays fast (MMR is O(k * shortlist_size)).

    Returns: a list of up to k item ids, best first.
    """
    if diversity <= 0:
        # No re-ranking requested: just return the plain top-k, same
        # ordering the rest of the project already uses.
        top = np.argsort(-scores)[:k]
        return [item_ids[i] for i in top]

    # Only look at the best `shortlist_size` candidates by raw score.
    # The full catalog has thousands of movies; we don't need to compare
    # all of them to each other, just the ones worth recommending at all.
    shortlist = np.argsort(-scores)[:shortlist_size].tolist()

    chosen = []          # positions (row indices) picked so far, in order
    remaining = shortlist

    while remaining and len(chosen) < k:
        best_position = None
        best_value = -np.inf

        for position in remaining:
            relevance = scores[position]

            if chosen:
                # How similar is this candidate to the closest movie we've
                # already chosen? That's the "redundancy" we want to avoid.
                similarities = [
                    cosine_similarity(feature_matrix[position], feature_matrix[c])
                    for c in chosen
                ]
                similarity_to_chosen = max(similarities)
            else:
                similarity_to_chosen = 0.0

            # Blend relevance (good score) against novelty (not similar to
            # what we already picked). This is the MMR formula.
            value = (1 - diversity) * relevance - diversity * similarity_to_chosen

            if value > best_value:
                best_value = value
                best_position = position

        chosen.append(best_position)
        remaining.remove(best_position)

    return [item_ids[p] for p in chosen]
