from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser

def setup_user_user(ratings, nnbrs, min_nbrs):
    # These two numbers set the minimum (3) and maximum (15) number of neighbours to consider. These are considered "reasonable defaults", but you can experiment with others too
    user_user = UserUser(nnbrs, min_nbrs=min_nbrs)

    if 'timestamp' in ratings.columns:
        ratings = ratings.drop(columns=['timestamp'])
    ratings.rename(columns={'userId': 'user'}, inplace=True)
    ratings.rename(columns={'movieId': 'item'}, inplace=True)

    algo = Recommender.adapt(user_user)
    algo.fit(ratings)
    print("User-User algorithm set up!")
    return algo

def get_user_user_recs(user_id, user_ratings, nnbrs=15, min_nbrs=3):
    algo = setup_user_user(user_ratings, nnbrs, min_nbrs)
    return algo.recommend(user_id)


