

def compute_content_scores(user_id, user_profile, films, credits_df, ratings_df, weights, genre_list_mlb):
    recommendations = []

    print("films shape: ", films.shape) # User rated films are ending up in the recommendations
    print("Num user rated films: ", len(ratings_df[ratings_df['userId'] == user_id]))
    user_rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    unrated_movies = films[~films['id'].isin(user_rated_movies)]

    print("Unrated movies shape: ", unrated_movies.shape)

    for _, movie_row in unrated_movies.iterrows():
        film_id = movie_row['id']
        score, cast_score, director_score, genre_score = 0, 0, 0, 0

        # Calculate cast score
        cast_ids = credits_df.loc[credits_df['id'] == film_id, 'cast_info'].values[0]
        for cast_member in cast_ids:
            cast_score += user_profile.get(cast_member[1], 0) * weights["cast_ft_weight"] * weights["content_weight"]

        # Calculate director score
        director_info = credits_df.loc[credits_df['id'] == film_id, 'director_info'].values[0]
        director_score += user_profile.get(director_info[0], 0) * weights["director_ft_weight"] * weights["content_weight"]

        # Calculate genre score
        matched_genres = [user_profile.get(film_genre, 0) for film_genre in genre_list_mlb if movie_row[film_genre] == 1]
        num_genres = sum(movie_row[genre] for genre in genre_list_mlb)  # Total genres for the movie
        if num_genres > 0:
            genre_score = sum(matched_genres) / num_genres  # Normalize by the number of genres
        genre_score *= weights["genre_ft_weight"] * weights["content_weight"]

        # Calculate final score
        score = cast_score + director_score + genre_score

        recommendations.append((film_id, round(score, 3), round(cast_score, 3), round(director_score, 3), round(genre_score, 3)))

    return recommendations