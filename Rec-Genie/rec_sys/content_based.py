

def compute_content_scores(user_id, user_profile, films, credits_df, ratings_df, weights, genre_list_mlb):
    # print("User profile type: ", type(user_profile))
    # print("User profile: ", user_profile)

    # user_profile = json.loads(user_profile)
    # print("User profile type (after json loads): ", type(user_profile))
    user_rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    unrated_movies = films[~films['id'].isin(user_rated_movies)]

    recommendations = []

    for _, movie_row in unrated_movies.iterrows():
        film_id = movie_row['id']
        score, cast_score, director_score, genre_score = 0, 0, 0, 0

        # find the index of the film_id in credits_df
        try:
            movie_index = credits_df[credits_df['id'] == film_id].index[0]
        except IndexError:
            print(f"Film ID {film_id} not found in credits_df")
            recommendations.append((film_id, 0, 0, 0, 0)) # Still append the movie ID with a score of 0
            continue

        # Calculate weighted score based on user profile and movie cast/director, genres,
        cast_ids = credits_df['cast_info'].iloc[movie_index]
        print("Credits type: ", type(credits_df))
        print("Credits head: ", credits_df.head())
        print("Credits columns: ", credits_df.columns)

        for cast_member in cast_ids:
            print("Cast ids: ", cast_ids)
            print("Cast ids type: ", type(cast_ids))
            print("Cast member: ", cast_member)
            print("Cast member[1]: ", cast_member[1])
            print("User profile type: ", type(user_profile))
            print("User profile: ", user_profile)

            cast_score += user_profile.get(cast_member[1], 0) * weights["cast_ft_weight"]
        score += cast_score

        director_info = credits_df["director_info"].iloc[movie_index]
        director_score += user_profile.get(director_info[0], 0) * weights["director_ft_weight"]
        score += director_score

        # **Normalize Genre Score**
        matched_genres = [user_profile.get(film_genre, 0) for film_genre in genre_list_mlb if
                          movie_row[film_genre] == 1]
        num_genres = sum(movie_row[genre] for genre in genre_list_mlb)  # Total genres for the movie

        if num_genres > 0:
            genre_score = sum(matched_genres) / num_genres  # Normalized score
        else:
            genre_score = 0  # Edge case: If no genres exist for this movie

        score += genre_score * weights["genre_ft_weight"]

        score = score * weights["content_weight"]
        recommendations.append((film_id, round(score, 3), round(cast_score, 3), round(director_score, 3), round(genre_score, 3)))

    recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    top_recommendations = recommendations

    top_recommendations = [(movie_id, score, cast_score, director_score, genre_score)
                           for movie_id, score, cast_score, director_score, genre_score in top_recommendations]

    return top_recommendations