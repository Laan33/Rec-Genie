from collaborative import get_user_user_recs
from content_based import compute_content_scores

# Constants
weights = {
    'cast_ft_weight': 0.3,
    'director_ft_weight': 0.3,
    'genre_ft_weight': 0.15,
    'user_user_weight': 1,
    'content_weight': 0.7,
    'collab_weight': 0.3
}

num_recs = 20 # Number of recommendations to return

def hybrid_recommend(user_id, user_profile, films_df, credits_df, ratings_df, genre_list_mlb):
    content_scores = compute_content_scores(user_id, user_profile, films_df, credits_df, ratings_df, weights, genre_list_mlb,top_n=num_recs)
    user_user_ratings = ratings_df.copy()
    collab_scores = get_user_user_recs(user_id, user_user_ratings)

    final_scores = {
        movie_id: {
            'final_score': weights['content_weight'] * content_score + weights['collab_weight'] * collab_scores.get(movie_id, 0),
            'content_score': content_score,
            'cast_score': cast_score,
            'director_score': director_score,
            'genre_score': genre_score,
            'collab_score': collab_scores.get(movie_id, 0)
        }
        for movie_id, content_score, cast_score, director_score, genre_score in content_scores
    }

    return sorted(final_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)[:num_recs]

def score_breakdown(films_df, recommended_movies):
    return films_df[films_df['id'].isin([movie_id for movie_id, _, _, _, _, _ in recommended_movies])][['title', 'release_date', 'vote_average', 'vote_count']].assign(
    score=[score for _, score, _, _, _, _ in recommended_movies],
    cast_score=[cast_score for _, _, cast_score, _, _, _ in recommended_movies],
    director_score=[director_score for _, _, _, director_score, _, _ in recommended_movies],
    genre_score=[genre_score for _, _, _, _, genre_score, _ in recommended_movies],
    user_user_score=[user_user_score for _, _, _, _, _, user_user_score in recommended_movies]
    )