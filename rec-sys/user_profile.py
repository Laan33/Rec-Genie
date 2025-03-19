import pandas as pd
from io import StringIO

user_ratings = """
710,golden eye, 4
1770,Michael Collins, 5
76600,Avatar 2, 3
141052,Justice League, 2.5
284053,Thor: Ragnarok, 4
341013,Atomic Blonde, 3.8
374720,Dunkirk, 4.7
339403,Baby Driver, 4.7
324852,Despicable Me 3, 4
419192,McLaren, 3.5
283995,Guardians of the Galaxy Vol. 2, 3.8
324849,The Lego Batman Movie, 4.5
324552,John Wick: Chapter 2, 3.5
180863,T2 Trainspotting, 3.6
318846,The Big Short, 4.4
406,La Haine, 4.65
424,Schindler's List, 4.8
627,Trainspotting, 4.9
107,Snatch, 4.9
1429,25th Hour, 3.3
49517,Tinker Tailor Soldier Spy, 4.1
6538,Charlie Wilson's War, 3.9
10315,Fantastic Mr. Fox, 4.7
27205,Inception, 3.6
106646,The Wolf of Wall Street, 4.5
120467,The Grand Budapest Hotel, 4.6
157336,Interstellar, 3.9
261023,Black Mass, 3.6
278,The Shawshank Redemption, 3
4232,Scream, 2.4
1893,Star Wars: Episode I - The Phantom Menace, 3.8
550,Fight Club, 4.2
98,Gladiator, 4
508,Love Actually, 2.1
228967,The Interview, 2.7
1895,Star Wars: Episode III - Revenge of the Sith, 3.5
18785,The Hangover, 1.9
67913,The Guard, 3.7
40807,50/50, 2.5
70160,The Hunger Games, 2.4
77930,Magic Mike, 1
72190,World War Z, 2.5
187017,22 Jump Street, 1.9
198663,The Maze Runner, 2
207703,Kingsman: The Secret Service, 2.1
99861,Avengers: Age of Ultron, 2
167073,Brooklyn, 2.8
254470,Pitch Perfect 2, 1.9
271718,Trainwreck, 1
314365,Spotlight, 4.9
259693,The Conjuring 2, 2.3
308266,War Dogs, 2.1
324786,Hacksaw Ridge, 3.1
330459,Rogue One: A Star Wars Story, 2.9
339846,Baywatch, 2
"""

def load_user_profile():
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


def create_user_profile(user_id, films_df, ratings_df, genre_list_mlb):
    """Creates a user profile based on ratings for movies with shared cast/directors."""
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    profile = {}

    for _, rating_row in user_ratings.iterrows():
        movie_id = rating_row['movieId']
        rating = rating_row['rating']

        # Add cast and director IDs to user profile with weighted ratings
        cast_ids = films_df['cast_info'].iloc[movie_id]
        for cast_member in cast_ids:
            profile[cast_member[1]] = profile.get(cast_member[1], 0) + rating

        director_info = films_df['director_info'].iloc[movie_id]
        if director_info is not None:
            profile[director_info[0]] = profile.get(director_info[0], 0) + rating

        # Add genre IDs to user profile with weighted ratings
        genre_score = genre_normalisation * rating
        for film_genre in genre_list_mlb:
            if films_df[film_genre].iloc[movie_id] == 1:
                profile[film_genre] = profile.get(film_genre, 0) + genre_score

        profile.id = user_id

    return profile


