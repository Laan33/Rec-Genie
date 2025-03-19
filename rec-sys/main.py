import importlib

import pandas as pd
import data_loader
import user_profile as user_pf
import preprocessing as pre


# Change this to True if you are running on Google Colab
RUNNING_ON_COLAB = False

# Constants
USER_ID = 999999

def data():
    if RUNNING_ON_COLAB:
        data_loader.mount_drive()

    # Load the data
    films_df = data_loader.load_movies()
    ratings_df = data_loader.load_ratings()
    credits_df = data_loader.load_credits()

    print("Data dimensions:")
    print("films_df: ", films_df.shape)
    print("ratings_df: ", ratings_df.shape)
    print("credits_df: ", credits_df.shape)

    return films_df, ratings_df, credits_df

# Data processing
def process_data(films_df, ratings_df, credits_df):
    films_df = pre.filter_films(films_df)

    ohe_films_df, genre_list_mlb = pre.one_hot_encode_genres(films_df)
    credits_df = pre.condense_credits(credits_df)

    films_df = pre.data_tidying(ohe_films_df, credits_df)

def user(user_id, films_df, genre_list_mlb):
    user_ratings_df = user_pf.load_user_ratings()
    user_profile = user_pf.user_feature_profile(user_id, films_df, user_ratings_df, genre_list_mlb)