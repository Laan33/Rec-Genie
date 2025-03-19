import pandas as pd

import data_loader
import hybrid_recommender
import preprocessing as pre
import user_profile as user_pf

# Change this to True if you are running on Google Colab
RUNNING_ON_COLAB = False

# Constants
USER_ID = 999,999



if RUNNING_ON_COLAB:
    data_loader.mount_drive()

# Load the data
films_df = data_loader.load_movies()
ratings_df = data_loader.load_ratings()
credits_df = data_loader.load_credits()

# Data preprocessing
# Drop the weird film entry
films_df = films_df.drop(35587)

# One-hot encode the genres
ohe_films_df = pre.one_hot_encode_genres(films_df)

# Process the credits metadata
credits_df = pre.condense_credits(credits_df)

# Tidy the noise and merge the credits metadata with the films DataFrame
films_df = pre.data_tidying(ohe_films_df, credits_df)

# Generate the user profile
user_profile_df = user_pf.load_user_ratings()
user_profile = user_pf.create_user_profile(USER_ID, films_df, ratings_df, user_profile_df)

# Append the user profile to the ratings DataFrame
ratings_df = pd.concat([ratings_df, user_profile_df], ignore_index=True)


# Generate recommendations
recommendations = hybrid_recommender.hybrid_recommend(USER_ID, user_profile, films_df, credits_df, ratings_df)
hybrid_recommender.score_breakdown(films_df, recommendations)




