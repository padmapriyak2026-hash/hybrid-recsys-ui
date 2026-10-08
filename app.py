"""
Streamlit UI for the hybrid movie recommender.

What this app lets you do:
  1. Pick a real MovieLens user and see what the hybrid model recommends
     for them, and how that compares to popularity / CF-only / content-only.
  2. Simulate a BRAND NEW user who has no rating history at all, by picking
     a few movies you like -- this is a live demo of the cold-start problem
     this whole project is about. Watch the blend weight drop to 0% CF.
  3. Turn on "diversity" re-ranking and watch the recommendations spread
     out across genres instead of only showing the same popular hits.

Before running this app for the first time, train and save the model:
    python train_and_save.py
Then start the app:
    streamlit run app.py
"""

import pickle

import numpy as np
import pandas as pd
import streamlit as st

from diversity import diverse_top_k

MODEL_PATH = "trained_model.pkl"


# ---------------------------------------------------------------------------
# Loading the trained model (cached so this only happens once per session)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_everything():
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["ratings"], saved["movies"]


@st.cache_data
def build_lookup_tables(_ratings, _movies):
    """Small helper tables the app reuses a lot. Leading underscore on the
    arguments tells Streamlit's cache not to try hashing the big dataframes
    themselves -- we only need to compute this once anyway."""
    title_of = _movies.set_index("item")["title"].to_dict()
    genres_of = _movies.set_index("item")["genres"].to_dict()
    year_of = _movies["title"].str.extract(r"\((\d{4})\)")[0]
    year_of.index = _movies["item"].values
    year_of = year_of.to_dict()

    popularity_order = _ratings["item"].value_counts().index.tolist()
    n_ratings_per_item = _ratings["item"].value_counts()

    # A friendly, well-known shortlist of movies for the "brand new user"
    # picker, so people aren't scrolling through 3,700 obscure titles.
    well_known_items = n_ratings_per_item[n_ratings_per_item >= 200].index.tolist()
    well_known_titles = sorted(
        f"{title_of[i]}" for i in well_known_items if i in title_of
    )

    return title_of, genres_of, year_of, popularity_order, well_known_titles


def recs_to_table(item_ids, title_of, year_of, genres_of):
    """Turn a plain list of item ids into a nice dataframe for display."""
    rows = []
    for rank, item in enumerate(item_ids, start=1):
        rows.append({
            "#": rank,
            "Title": title_of.get(item, f"(unknown #{item})"),
            "Year": year_of.get(item, "?"),
            "Genres": genres_of.get(item, "?").replace("|", ", "),
        })
    return pd.DataFrame(rows).set_index("#")


def genre_breakdown(item_ids, genres_of):
    """Count how many recommended movies fall into each genre, for the
    little bar chart. One movie usually has several genres, so the counts
    add up to more than len(item_ids) -- that's expected."""
    counts = {}
    for item in item_ids:
        for genre in genres_of.get(item, "").split("|"):
            if genre:
                counts[genre] = counts.get(genre, 0) + 1
    return pd.Series(counts).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Hybrid Movie Recommender", page_icon="🎬", layout="wide")
st.title("🎬 Hybrid Movie Recommender")
st.caption(
    "Collaborative filtering (SVD) + content-based (genres & decade), blended "
    "with a weight that adapts to how much rating history a user and movie have. "
    "Built on MovieLens 1M. See WRITEUP.md for the full methodology and honest results."
)

try:
    model, ratings, movies = load_everything()
except FileNotFoundError:
    st.error(
        "No trained model found yet. Run **`python train_and_save.py`** in the "
        "project folder first, then reload this page."
    )
    st.stop()

title_of, genres_of, year_of, popularity_order, well_known_titles = build_lookup_tables(ratings, movies)
title_to_item = {title_of[i]: i for i in title_of}

# ---------------------------------------------------------------------------
# Sidebar: choose a user and settings
# ---------------------------------------------------------------------------

st.sidebar.header("1. Who are we recommending for?")
mode = st.sidebar.radio(
    "Mode",
    ["Existing MovieLens user", "Brand-new user (cold-start demo)"],
)

st.sidebar.header("2. Settings")
k = st.sidebar.slider("How many recommendations?", min_value=5, max_value=20, value=10)
diversity = st.sidebar.slider(
    "Diversity (spread out genres vs. pure accuracy)",
    min_value=0.0, max_value=1.0, value=0.0, step=0.1,
    help="0 = the model's plain top picks. Higher values trade a little "
         "accuracy for recommendations that aren't all the same genre.",
)

with st.sidebar.expander("How the blend works"):
    st.markdown(
        "`score = a * CF_score + (1 - a) * content_score`\n\n"
        "`a` grows with how many ratings the **user** and the **movie** have. "
        "No history on either side means `a = 0`, so the content model "
        "(genres + decade) makes the call instead of collaborative filtering, "
        "which has nothing to go on for a stranger."
    )

# ---------------------------------------------------------------------------
# Mode 1: an existing MovieLens user
# ---------------------------------------------------------------------------

if mode == "Existing MovieLens user":
    st.sidebar.subheader("Pick a user")
    if "user_id" not in st.session_state:
        st.session_state.user_id = int(ratings["user"].iloc[0])

    if st.sidebar.button("🎲 Pick a random user"):
        st.session_state.user_id = int(np.random.choice(ratings["user"].unique()))

    user_id = st.sidebar.number_input(
        "MovieLens user ID (1 to 6040)",
        min_value=1, max_value=int(ratings["user"].max()),
        key="user_id",
    )

    user_history = ratings[ratings["user"] == user_id]
    seen_items = set(user_history["item"])
    n_ratings = len(user_history)
    trust = model.user_trust(user_id)

    st.subheader(f"User {user_id}: {n_ratings} ratings in the training data")
    st.markdown(
        f"User-side trust in collaborative filtering: **{trust:.0%}** "
        f"(movies with little history still pull the final weight down further)."
    )

    favorite_movies = user_history.sort_values("rating", ascending=False).head(5)
    if len(favorite_movies) > 0:
        st.caption("A few movies this user rated highly:")
        st.write(", ".join(title_of.get(i, "?") for i in favorite_movies["item"]))

    raw_scores = model.scores(user_id)
    scored_items = model.items

    # Never recommend something they've already rated.
    masked_scores = raw_scores.copy()
    for item in seen_items:
        if item in model.i_idx:
            masked_scores[model.i_idx[item]] = -np.inf

    hybrid_recs = diverse_top_k(masked_scores, scored_items, model.content.X, k, diversity)

    st.subheader("🏆 Hybrid recommendations")
    st.dataframe(recs_to_table(hybrid_recs, title_of, year_of, genres_of), width="stretch")

    with st.expander("Compare against the other methods"):
        popularity_recs = [i for i in popularity_order if i not in seen_items][:k]
        cf_recs = model.cf.recommend(user_id, seen_items, k)
        content_recs = model.content.recommend(user_id, seen_items, k)

        col1, col2, col3, col4 = st.columns(4)
        for col, name, recs in [
            (col1, "Popularity", popularity_recs),
            (col2, "CF only", cf_recs),
            (col3, "Content only", content_recs),
            (col4, "Hybrid", hybrid_recs),
        ]:
            with col:
                st.markdown(f"**{name}**")
                for item in recs:
                    st.write(f"- {title_of.get(item, '?')}")

    st.subheader("What genres are we recommending?")
    st.bar_chart(genre_breakdown(hybrid_recs, genres_of))

# ---------------------------------------------------------------------------
# Mode 2: a brand-new user (cold-start demo)
# ---------------------------------------------------------------------------

else:
    st.subheader("Simulate a brand-new user")
    st.markdown(
        "Pick a few movies you like. This person does **not** exist in the "
        "training data at all -- no ratings, nothing. Watch how the system "
        "falls back entirely on genres and decade, because collaborative "
        "filtering has zero history to work with."
    )

    picked_titles = st.multiselect(
        "Movies you like (pick 2 or more for a decent taste profile):",
        options=well_known_titles,
    )

    if len(picked_titles) == 0:
        st.info("Pick at least one movie on the left to see recommendations.")
        st.stop()

    st.caption("How much did you like each one?")
    item_ratings = {}
    for title in picked_titles:
        item_id = title_to_item[title]
        stars = st.slider(title, min_value=1, max_value=5, value=5, key=f"rating_{item_id}")
        item_ratings[item_id] = stars

    st.markdown("User-side trust in collaborative filtering: **0%** (brand new user -> pure content model).")

    raw_scores = model.scores_for_new_user(item_ratings)
    for item_id in item_ratings:
        if item_id in model.i_idx:
            raw_scores[model.i_idx[item_id]] = -np.inf

    hybrid_recs = diverse_top_k(raw_scores, model.items, model.content.X, k, diversity)

    st.subheader("🏆 Recommendations for this brand-new user")
    st.dataframe(recs_to_table(hybrid_recs, title_of, year_of, genres_of), width="stretch")

    with st.expander("Compare against popularity"):
        seen_items = set(item_ratings)
        popularity_recs = [i for i in popularity_order if i not in seen_items][:k]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Popularity (ignores your picks)**")
            for item in popularity_recs:
                st.write(f"- {title_of.get(item, '?')}")
        with col2:
            st.markdown("**Hybrid (= content-only here)**")
            for item in hybrid_recs:
                st.write(f"- {title_of.get(item, '?')}")

    st.subheader("What genres are we recommending?")
    st.bar_chart(genre_breakdown(hybrid_recs, genres_of))
