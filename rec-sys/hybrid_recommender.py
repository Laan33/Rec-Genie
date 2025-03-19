from collaborative import get_user_user_recs
from content_based import compute_content_scores

# Constants
weights = {
    'cast_ft_weight': 0.3,
    'director_ft_weight': 0.7,
    'genre_ft_weight': 0.07,
    'user_user_weight': 1,
    'content_weight': 0.7,
    'collab_weight': 1
}

num_recs = 400 # Number of recommendations to return

def hybrid_recommend(user_id, user_profile, films_df, credits_df, ratings_df, genre_list_mlb):
    content_scores = compute_content_scores(user_id, user_profile, films_df, credits_df, ratings_df, weights, genre_list_mlb)
    user_user_ratings = ratings_df.copy()
    collab_scores = get_user_user_recs(user_id, user_user_ratings)

    # Convert collab_scores into a dictionary for faster lookup
    collab_scores_dict = dict(zip(collab_scores['item'], collab_scores['score']))

    final_scores = {
        movie_id: {
            'final_score': weights['content_weight'] * content_score + weights['collab_weight'] * collab_scores_dict.get(movie_id, 0),  # Default to 0 if not found
            'content_score': content_score,
            'cast_score': cast_score,
            'director_score': director_score,
            'genre_score': genre_score,
            'collab_score': collab_scores_dict.get(movie_id, 0)  # Default to 0 if not found
        }
        for movie_id, content_score, cast_score, director_score, genre_score in content_scores
    }

    return sorted(final_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)[:400]


def score_breakdown(films_df, recommended_movies):
    # Extract the recommended movie IDs
    recommended_ids = [movie_id for movie_id, _ in recommended_movies]

    # Filter films_df to match the recommended IDs
    filtered_films = films_df[films_df['id'].isin(recommended_ids)][['id', 'title', 'release_date', 'vote_average', 'vote_count']]

    # Ensure that each movie_id appears only once
    filtered_films = filtered_films.drop_duplicates(subset=['id'])

    # Ensure that filtered_films and recommended_movies have the same number of entries
    recommended_movies_dict = dict(recommended_movies)  # Convert list of tuples to a dictionary

    # Assign the scores to the filtered dataframe
    scored_films = filtered_films.assign(
        score=[recommended_movies_dict[movie_id]['final_score'] for movie_id in filtered_films['id']],
        cast_score=[recommended_movies_dict[movie_id]['cast_score'] for movie_id in filtered_films['id']],
        director_score=[recommended_movies_dict[movie_id]['director_score'] for movie_id in filtered_films['id']],
        genre_score=[recommended_movies_dict[movie_id]['genre_score'] for movie_id in filtered_films['id']],
        user_user_score=[recommended_movies_dict[movie_id]['collab_score'] for movie_id in filtered_films['id']]
    )

    # Sort the DataFrame by the final_score in descending order
    return scored_films.sort_values(by='score', ascending=False)