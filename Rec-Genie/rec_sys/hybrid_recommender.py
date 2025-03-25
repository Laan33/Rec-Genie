from . import collaborative
from . import content_based
# from collaborative import get_user_user_recs
# from content_based import compute_content_scores

# Constants
# weights = {
#     'cast_ft_weight': 0.3,
#     'director_ft_weight': 0.4,
#     'genre_ft_weight': 0.4,
#     'content_weight': 0.7,
#     'collab_weight': 1,
#     'genre_normalisation': 0.12,
#     'average_rating_weight': 0.3
# }

num_recs = 400 # Number of recommendations to return

def hybrid_recommend(user_profile, films_df, credits_df, ratings_df, genre_list_mlb):
    weights = user_profile['weights']
    content_scores = content_based.compute_content_scores(user_profile['id'], user_profile['feature_profile'], films_df, credits_df, ratings_df, weights, genre_list_mlb)
    user_user_ratings = ratings_df.copy()
    user_user_ratings = remove_non_applicable_films(films_df, user_user_ratings)

    collab_scores = collaborative.get_user_user_recs(user_profile['id'], user_user_ratings)

    collab_scores_covered = fill_in_collab_scores(films_df, collab_scores, weights)

    # Convert collab_scores into a dictionary for faster lookup
    collab_scores_dict = dict(zip(collab_scores_covered['id'], collab_scores_covered['score']))

    final_scores = {
        movie_id: {
            'final_score': content_score + collab_scores_dict.get(movie_id, 0),  # Default to 0 if not found
            'content_score': content_score,
            'cast_score': cast_score,
            'director_score': director_score,
            'genre_score': genre_score,
            'collab_score': collab_scores_dict.get(movie_id, 0) * weights['collab_weight']  # Default to 0 if not found
        }
        for movie_id, content_score, cast_score, director_score, genre_score in content_scores
    }

    return sorted(final_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)[:num_recs]


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
        user_user_score=[recommended_movies_dict[movie_id]['collab_score'] for movie_id in filtered_films['id']],
        cast_proportion=[round(float(recommended_movies_dict[movie_id]['cast_score'] / recommended_movies_dict[movie_id]['final_score']),1) for movie_id in filtered_films['id']],
        director_proportion=[round(recommended_movies_dict[movie_id]['director_score'] / recommended_movies_dict[movie_id]['final_score'],1) for movie_id in filtered_films['id']],
        genre_proportion=[round(recommended_movies_dict[movie_id]['genre_score'] / recommended_movies_dict[movie_id]['final_score'],1) for movie_id in filtered_films['id']],
        user_user_proportion=[round(recommended_movies_dict[movie_id]['collab_score'] / recommended_movies_dict[movie_id]['final_score'],1) for movie_id in filtered_films['id']]
    )

    # Sort the DataFrame by the final_score in descending order
    return scored_films.sort_values(by='score', ascending=False)

def fill_in_collab_scores(films_df, collab_scores, weights):
    # Multiply the collaborative scores by the weight
    collab_scores['score'] = collab_scores['score'] * weights['collab_weight']

    # If no collaborative recommendations are available, add the vote_average as a fallback to the score
    collab_scores['score'] = collab_scores['score'].fillna(0)
    collab_scores = collab_scores.rename(columns={'item': 'id'})

    # Merge vote_average from films_df
    collab_scores = collab_scores.merge(films_df[['id', 'vote_average']], on='id', how='left')

    # Replace 0 scores with vote_average where applicable
    collab_scores['score'] = collab_scores['score'].where(collab_scores['score'] != 0,
                                                          (collab_scores['vote_average'].apply(punish_low_ratings) * weights['average_rating_weight']))
    # Drop the now-unneeded vote_average column
    collab_scores = collab_scores.drop(columns=['vote_average'])

    return collab_scores

def punish_low_ratings(rating):
    # Apply penalty: Negative weight for ratings below 2
    rating = (rating / 2)

    if rating < 2:
        return -abs(2 - rating)  # Negative penalty
    return rating

def remove_non_applicable_films(films_df, ratings_df):
    # Remove ratings that have film ids not in films_df
    valid_ids = films_df['id']
    ratings_df = ratings_df[ratings_df['movieId'].isin(valid_ids)]
    print("Ratings after removing non-applicable films:", ratings_df.shape)
    return ratings_df