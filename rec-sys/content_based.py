

def compute_content_scores(user_id, user_profile, films, credits_df, ratings_df, weights, genre_list_mlb, top_n=20):
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
            continue

        # Calculate weighted score based on user profile and movie cast/director, genres,
        cast_ids = credits_df['cast_info'].iloc[movie_index]
        for cast_member in cast_ids:
            cast_score += user_profile.get(cast_member[1], 0) * weights["cast_ft_weight"]
        score += cast_score

        director_info = films["director_info"].iloc[movie_index]
        director_score += user_profile.get(director_info[0], 0) * weights["director_ft_weight"]
        score += director_score

        # Calculate weighted score based on one hot-encoded genres
        # TODO - need to make this a normalised score - e.g. if you have 7 genres that all, you're greatly dominating other films with 1 genre that matches
        genre_score = sum \
            ([user_profile.get(film_genre, 0) for film_genre in genre_list_mlb if movie_row[film_genre] == 1])
        score += genre_score * weights["genre_ft_weight"]

        recommendations.append((film_id, score, cast_score, director_score, genre_score))


    recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    top_recommendations = recommendations[:top_n]

    top_recommendations = [(movie_id, score, cast_score, director_score, genre_score)
                           for movie_id, score, cast_score, director_score, genre_score in top_recommendations]

    return top_recommendations


# def recommend_movies(user_id, user_profile, films, top_3_credits_df, directors, ratings_df, top_n=20):
#
#
#     user_rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
#     unrated_movies = films[~films['id'].isin(user_rated_movies)]
#
#     print("Unrated movies dimensions:", unrated_movies.shape)
#     recommendations = []
#
#     user_user_scores = get_user_user_recs(user_id, 900)
#     print("User-User scores retrieved!")
#
#     for _, movie_row in unrated_movies.iterrows():
#         movie_id = movie_row['id']
#         score, cast_score, director_score, genre_score, user_user_score = 0, 0, 0, 0, 0
#
#         # Append on user-user collaborative filtering score
#         if movie_id in user_user_scores.item.values:
#             user_user_score = user_user_scores[user_user_scores['item'] == movie_id].score.values[0]
#
#         score += user_user_score * user_user_weight
#
#         # find the index of the movie_id in top_3_credits_df
#         try:
#             movie_index = top_3_credits_df[top_3_credits_df['id'] == movie_id].index[0]
#         except IndexError:
#             print(f"Movie ID {movie_id} not found in top_3_credits_df")
#             continue
#
#         # Calculate weighted score based on user profile and movie cast/director, genres,
#         cast_ids = top_3_credits_df['cast_info'].iloc[movie_index]
#         for cast_member in cast_ids:
#             cast_score += user_profile.get(cast_member[1], 0) * cast_ft_weight
#         score += cast_score
#
#         director_info = directors.iloc[movie_index]
#         director_score += user_profile.get(director_info[0], 0) * director_ft_weight
#         score += director_score
#
#         # Calculate weighted score based on one hot-encoded genres
#         # TODO - need to make this a normalised score - e.g. if you have 7 genres that all, you're greatly dominating other films with 1 genre that matches
#         genre_score = sum \
#             ([user_profile.get(film_genre, 0) for film_genre in mlb.classes_ if movie_row[film_genre] == 1])
#         score += genre_score * genre_ft_weight
#
#         recommendations.append((movie_id, score, cast_score, director_score, genre_score, user_user_score))
#
#     recommendations.sort(key=lambda x: x[1], reverse=True)  # Sort by score
#     top_recommendations = recommendations[:top_n]
#
#     top_recommendations = [(movie_id, score, cast_score, director_score, genre_score, user_user_score)
#                            for movie_id, score, cast_score, director_score, genre_score, user_user_score in top_recommendations]
#
#     # # recommendations.sort(key=lambda x: x[1])  # Sort in ascending order
#     # lowest_recommendations = recommendations[-top_n:]
#     # least_movie_recs = [(movie_id, score, cast_score, director_score, genre_score, user_user_score) for movie_id, score, cast_score, director_score, genre_score, user_user_score in lowest_recommendations]
#
#     # return movie_recs, least_movie_recs
#     return top_recommendations
