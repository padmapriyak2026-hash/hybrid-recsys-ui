import pandas as pd

# Load the two files we need
ratings = pd.read_csv("data/ml-1m/ratings.dat", sep="::", engine="python",
                      names=["user", "item", "rating", "ts"])
movies = pd.read_csv("data/ml-1m/movies.dat", sep="::", engine="python",
                     names=["item", "title", "genres"], encoding="latin-1")

# Question 1: how big is the data?
print("Users:", ratings["user"].nunique())
print("Movies that got rated:", ratings["item"].nunique())
print("Total ratings:", len(ratings))

# Question 2: how many ratings does each user give?
per_user = ratings.groupby("user").size()
print("\nRatings per user:")
print(per_user.describe())

# Question 3: how many ratings does each movie get?
per_item = ratings.groupby("item").size()
print("\nMovies with fewer than 5 ratings:", (per_item < 5).sum())
print("Movies in the catalog:", len(movies))
print("Movies nobody ever rated:", len(movies) - ratings["item"].nunique())

# A peek at the movie info we'll use for the content-based part
print("\n", movies.head())