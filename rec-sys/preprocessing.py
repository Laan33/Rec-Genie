import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer

def extract_genres(movies_df):
    movies_df['genres'] = movies_df['genres'].apply(lambda x: json.loads(x.replace("'", "\"")) if isinstance(x, str) else x)
    movies_df['genre_list'] = movies_df['genres'].apply(lambda x: [g['name'] for g in x] if isinstance(x, list) else [])
    return movies_df

def one_hot_encode_genres(movies_df):
    mlb = MultiLabelBinarizer()
    genre_ohe = pd.DataFrame(mlb.fit_transform(movies_df['genre_list']), columns=mlb.classes_)
    return pd.concat([movies_df, genre_ohe], axis=1)

def clean_credits(credits_df):
    credits_df['cast'] = credits_df['cast'].apply(lambda x: json.loads(x)[:3] if isinstance(x, str) else [])
    credits_df['directors'] = credits_df['crew'].apply(lambda x: [p['name'] for p in json.loads(x) if p['job'] == 'Director'] if isinstance(x, str) else [])
    return credits_df
