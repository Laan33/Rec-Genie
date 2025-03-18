from collaborative import get_user_user_recs

def hybrid_recommend(user_id, user_profile, movies_df, credits_df, ratings_df, model, w_content=0.7, w_collab=0.3):
    content_scores = compute_content_scores(user_profile, movies_df, credits_df)
    collab_scores = get_user_user_recs(user_id, ratings_df)

    final_scores = {
        movie_id: (w_content * content + w_collab * collab_scores.get(movie_id, 0))
        for movie_id, content in content_scores
    }

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:20]
