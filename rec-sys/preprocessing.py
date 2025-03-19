import ast

import pandas as pd
import json
from sklearn.preprocessing import MultiLabelBinarizer

genres_to_drop = [7759, 7760, 7761, 11602, 11176, 33751, 29812, 2883, 17161, 18012, 18013, 23822] # Remove low quality entries

def extract_genres(film_df):
    film_df['genres'] = film_df['genres'].apply(
        lambda x: json.loads(x.replace("'", "\"")) if isinstance(x, str) else x
    ).apply(
        lambda x: [gen for gen in x if gen['id'] not in genres_to_drop] if isinstance(x, list) else []
    )

    film_df['genre_list'] = film_df['genres'].apply(
        lambda x: [gen['name'] for gen in x] if isinstance(x, list) else [])
    return film_df

def one_hot_encode_genres(film_df):
    extract_genres(film_df)

    mlb = MultiLabelBinarizer()
    genre_ohe = pd.DataFrame(mlb.fit_transform(film_df['genre_list']),
                             columns=mlb.classes_,
                             index=film_df.index)
    genre_list_mlb = mlb.classes_

    # Concatenate the one-hot encoded genres to the original DataFrame
    ohe_film_df = pd.concat([film_df, genre_ohe], axis=1)
    return ohe_film_df.drop(columns=['genres', 'genre_list']), genre_list_mlb


# def clean_credits(credits_df):
#     credits_df['cast'] = credits_df['cast'].apply(lambda x: json.loads(x)[:3] if isinstance(x, str) else [])
#     credits_df['directors'] = credits_df['crew'].apply(lambda x: [p['name'] for p in json.loads(x) if p['job'] == 'Director'] if isinstance(x, str) else [])
#     return credits_df

def condense_credits(credits_df):
    top_3_credits_df = pd.concat([get_first_3_cast(credits_df['cast']), get_directors_from_crew(credits_df), credits_df['id']], axis=1)
    return top_3_credits_df


def get_first_3_cast(cast_series):
    """
    Extracts names and IDs of the first 3 cast members from a Pandas Series.

    Args:
        cast_series (pandas.Series): A Pandas Series containing cast information
                                      (string representation of list of dictionaries).

    Returns:
        pandas.DataFrame: DataFrame with a column 'cast_info' containing tuples of (name, ID)
                         for the first 3 cast members.
    """
    all_cast_info = []

    for cast_list_str in cast_series:
        cast_list = ast.literal_eval(cast_list_str)  # Convert string to list
        cast_info = []
        for i in range(min(3, len(cast_list))):
            cast_info.append((cast_list[i]['name'], cast_list[i]['id']))  # Create (name, ID) tuple
        all_cast_info.append(cast_info)

    return pd.DataFrame({'cast_info': all_cast_info})


def get_directors_from_crew(crew_series):
    """
    Extracts the names of the director(s) from the 'crew' column of a Pandas DataFrame.

    Args:
        film_credits (pandas.DataFrame): A Pandas DataFrame containing 'crew' column with crew information.

    Returns:
        pandas.DataFrame: DataFrame with a column 'director_info' containing tuples of (director name, film id).
    """
    crew_series = crew_series.set_index('id')

    director_info = []
    for idx, crew_list_str in crew_series['crew'].items():
        crew_list = ast.literal_eval(crew_list_str)  # Convert string to list
        director_name = next((member['name'] for member in crew_list if member['job'] == 'Director'), None)
        director_info.append((director_name, idx))
    return pd.DataFrame({'director_info': director_info})


def data_tidying(ohe_films_df, top_3_credits_df):
    # Convert the 'id' columns to numeric, forcing errors to NaN
    ohe_films_df.loc[:, 'id'] = pd.to_numeric(ohe_films_df['id'], errors='coerce')
    top_3_credits_df['id'] = pd.to_numeric(top_3_credits_df['id'], errors='coerce')

    # Drop rows with NaN values in the 'id' columns
    ohe_films_df = ohe_films_df.dropna(subset=['id'])
    top_3_credits_df = top_3_credits_df.dropna(subset=['id'])

    # Convert the 'id' columns to integers
    ohe_films_df.loc[:, 'id'] = ohe_films_df['id'].astype(int)
    top_3_credits_df['id'] = top_3_credits_df['id'].astype(int)

    # Merge the two DataFrames on the 'id' column
    ohe_films_df = ohe_films_df.merge(top_3_credits_df, on='id', how='inner')

    # Format looks like this: id, title, release_date, popularity, vote_average, vote_count, genre1, genre2, ..., genreN, cast_info, director_info
    return ohe_films_df




