import pandas as pd
from io import StringIO

def load_user_profile():
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
    return user_ratings_df[["userId", "movieId", "rating"]]


genre_normalisation = 0.12


def create_user_profile(user_id, films_df, ratings_df, top_3_credits_df, genre_list_mlb):
    """Creates a user profile based on ratings for movies with shared cast/directors."""
    directors = []
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    profile = {}

    for _, rating_row in user_ratings.iterrows():
        movie_id = rating_row['movieId']
        rating = rating_row['rating']

        # find the index of the movie_id in top_3_credits_df
        try:
            movie_index = top_3_credits_df[top_3_credits_df['id'] == movie_id].index[0]
        except IndexError:
            print(f"Movie ID {movie_id} not found in top_3_credits_df")
            continue  # if the movie_id isn't in top_3_credits_df, skip this movie

        # Add cast and director IDs to user profile with weighted ratings
        cast_ids = top_3_credits_df['cast_info'].iloc[movie_index]
        for cast_member in cast_ids:
            profile[cast_member[1]] = profile.get(cast_member[1], 0) + rating

        director_info = top_3_credits_df['director_info'].iloc[movie_index]
        if director_info is not None:
            profile[director_info[0]] = profile.get(director_info[0], 0) + rating
            directors.append(director_info[0])

        # Add genre IDs to user profile with weighted ratings
        genre_score = genre_normalisation * rating
        for film_genre in genre_list_mlb:
            if films_df[film_genre].iloc[movie_index] == 1:
                profile[film_genre] = profile.get(film_genre, 0) + genre_score

    return profile, directors

