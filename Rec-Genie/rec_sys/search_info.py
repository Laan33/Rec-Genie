

"""
Python file to search for information in the dataset - e.g. film id for a given title or vice versa.

TODO - If certain functions aren't working well (probably film id by title), might use fuzzy matching to find the closest match.
"""


def get_film_id_by_title(title, films_df):
    """Returns the film ID for a given title."""
    return films_df.loc[films_df['title'] == title, 'id'].values[0]

def get_title_by_film_id(film_id, films_df):
    """Returns the title for a given film ID."""
    return films_df.loc[films_df['id'] == film_id, 'title'].values[0]


def get_director_by_film_id(film_id, credits_df):
    """Returns the director for a given film ID."""
    try:
        return credits_df.loc[credits_df['id'] == film_id, 'director_info'].values[0][0]
    except IndexError:
        return None

def get_films_with_director(director, credits_df):
    """Returns the film IDs for a given director."""
    return credits_df[credits_df['director_info'].apply(lambda x: x[0] == director)]['id']


