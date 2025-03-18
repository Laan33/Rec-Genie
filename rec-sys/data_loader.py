import pandas as pd

def load_movies(path):
    return pd.read_csv(path, usecols=['id', 'title', 'release_date', 'genres', 'popularity', 'vote_average', 'vote_count'])

def load_ratings(path):
    return pd.read_csv(path, usecols=['userId', 'movieId', 'rating'])

def load_credits(path):
    return pd.read_csv(path)

def load_tags(path):
    return pd.read_csv(path, usecols=['userId', 'movieId', 'tag'])
