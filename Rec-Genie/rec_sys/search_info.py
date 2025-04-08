from fuzzywuzzy import process

"""
Python file to search for information in the dataset - e.g. film id for a given title or vice versa.

TODO - If certain functions aren't working well (probably film id by title), might use fuzzy matching to find the closest match.
"""


def get_film_id_by_title(title, films_df, score_cutoff=0.95):
    """Returns the film ID for a given title."""
    film_id = films_df.loc[films_df['title'] == title, 'id'].values
    if len(film_id) == 0:
        best_match = process.extractOne(title, films_df['title'], score_cutoff=score_cutoff)
        if best_match:
            film_id = films_df.loc[films_df['title'] == best_match[0], 'id'].values[0]
        else:
            film_id = None
    else:
        film_id = film_id[0]
    # Print the title given, the film ID found, and the resulting title
    if film_id is not None:
        print("Film title found:", films_df.loc[films_df['id'] == film_id, 'title'].values[0])
    else:
        print("No film ID found for title:", title)
    return film_id

def get_actor_id_by_name(name, actors_series, score_cutoff=0.98):
    """Returns the actor ID for a given actor name."""
    print("Actor name:", name)

    # Filter actors_series to find the actor by name
    actor_id = actors_series.loc[actors_series.apply(lambda x: len(x) > 0 and x[0][0] == name)].index.values
    if len(actor_id) == 0:
        # Use fuzzy matching if exact match is not found
        best_match = process.extractOne(name, actors_series.apply(lambda x: x[0][0] if len(x) > 0 else ""), score_cutoff=score_cutoff)
        if best_match:
            actor_id = actors_series.loc[actors_series.apply(lambda x: len(x) > 0 and x[0][0] == best_match[0])].index.values[0]
        else:
            actor_id = None
    else:
        actor_id = actor_id[0]

    # Print the name given, the actor ID found, and the resulting actor name
    if actor_id is not None:
        print("Actor name found:", actors_series.loc[actor_id][0][0])
    else:
        print("No actor ID found for name:", name)

    return actor_id


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

def fuzzy_search_user_profile(user_profile, search_term, score_cutoff=0.95):
    """Returns the best match for a given search term in the user profile."""
    best_match = process.extractOne(search_term, user_profile.keys(), score_cutoff=score_cutoff)
    if best_match:
        return best_match[0]
    else:
        return None
