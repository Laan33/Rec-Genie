import pandas as pd
from io import StringIO

with open('user_ratings.txt', 'r') as file:
    user_ratings = file.read()

# Convert into a DataFrame
user_ratings_df = pd.read_csv(StringIO(user_ratings), header=None, names=["movieId", "movie_name", "rating"])

# Drop the movie_name column as it's unnecessary for appending to the ratings DataFrame
user_ratings_df = user_ratings_df.drop(columns=["movie_name"])

user_ratings_df['movieId'] = pd.to_numeric(user_ratings_df['movieId'], errors='coerce')
user_ratings_df = user_ratings_df.dropna(subset=['movieId'])
user_ratings_df['movieId'] = user_ratings_df['movieId'].astype(int)

# Testing user id = 999,999
user_ratings_df["userId"] = 999999

# Reorder columns to match the existing ratings DataFrame
user_ratings_df = user_ratings_df[["userId", "movieId", "rating"]]



