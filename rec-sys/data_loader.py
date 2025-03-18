import os

import pandas as pd



dataset_path = os.path.join(os.path.expanduser('~'), 'datasets', 'films')

def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')
    dataset_path = '/content/drive/MyDrive/Colab_Notebooks/Datasets/'

def load_movies():
    # TODO - need to drop movies_df = movies_df.drop(35587) # A weird film entry is now gone!
    return pd.read_csv(dataset_path + "/movies_metadata.csv", usecols=['id', 'title', 'release_date', 'genres', 'popularity', 'vote_average', 'vote_count'])

def load_ratings():
    return pd.read_csv(dataset_path + "/ratings.csv", usecols=['userId', 'movieId', 'rating'])

def load_credits():
    return pd.read_csv(dataset_path + "/credits.csv")
