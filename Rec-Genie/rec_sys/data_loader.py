import os
import pandas as pd

dataset_path = os.path.join(os.path.expanduser('~'), 'datasets', 'films')
num_lines_to_load = None  # Global variable to store the number of lines to load

def set_num_lines(num_lines):
    global num_lines_to_load
    num_lines_to_load = num_lines

def mount_drive():
    print("Mounting Google Drive...")
    # from google.colab import drive
    # drive.mount('/content/drive')
    # dataset_path = '/content/drive/MyDrive/Colab_Notebooks/Datasets/'

def load_movies():
    return pd.read_csv(dataset_path + "/TMDB_all_movies.csv",
                       usecols=['id', 'title', 'release_date', 'genres', 'popularity', 'vote_average', 'vote_count', 'imdb_id'],
                       nrows=num_lines_to_load)

def load_ratings():
    return pd.read_csv(dataset_path + "/ratings.csv", usecols=['userId', 'movieId', 'rating'], nrows=num_lines_to_load)

def load_ratings_timestamped():
    return pd.read_csv(dataset_path + "/ratings.csv", nrows=num_lines_to_load)

def load_credits(load_original=False):
    if not load_original:
        return pd.read_csv(dataset_path + "/gen_credits_df.csv", nrows=num_lines_to_load)
    else:
        return pd.read_csv(dataset_path + "/credits.csv", nrows=num_lines_to_load)

def load_credits_no_mod():
    return pd.read_csv(dataset_path + "/credits.csv", nrows=num_lines_to_load)

def save_credits(gen_credits_df):
    new_credits_file_path = dataset_path + "/gen_credits_df.csv"
    gen_credits_df.to_csv(new_credits_file_path, index=False)