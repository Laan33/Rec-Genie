import ast
import os
import re
from collections import defaultdict
from pprint import pprint

from .search_info import get_film_id_by_title, fuzzy_search_user_profile, get_actor_id_by_name
import numpy as np

import pandas as pd
from io import StringIO
# import data_loader

# # TODO - burn once tested
# def load_data(num_lines=None):
#     if num_lines is not None:
#         data_loader.set_num_lines(num_lines)
#     films_df = data_loader.load_movies()
#     ratings_df = data_loader.load_ratings()
#     credits_df = data_loader.load_credits(True)
#
#     print("Data dimensions:")
#     print("films_df:", films_df.shape)
#     print("ratings_df:", ratings_df.shape)
#     print("credits_df:", credits_df.shape)
#
#     return films_df, ratings_df, credits_df


user_ratings = """
710,golden eye, 4
1770,Michael Collins, 5
76600,Avatar 2, 3
141052,Justice League, 2.5
284053,Thor: Ragnarok, 4
341013,Atomic Blonde, 3.8
374720,Dunkirk, 4.7
339403,Baby Driver, 4.7
324852,Despicable Me 3, 4
419192,McLaren, 3.5
283995,Guardians of the Galaxy Vol. 2, 3.8
324849,The Lego Batman Movie, 4.5
324552,John Wick: Chapter 2, 3.5
180863,T2 Trainspotting, 3.6
318846,The Big Short, 4.4
406,La Haine, 4.65
424,Schindler's List, 4.8
627,Trainspotting, 4.9
107,Snatch, 4.9
1429,25th Hour, 3.3
49517,Tinker Tailor Soldier Spy, 4.1
6538,Charlie Wilson's War, 3.9
10315,Fantastic Mr. Fox, 4.7
27205,Inception, 3.6
106646,The Wolf of Wall Street, 4.5
120467,The Grand Budapest Hotel, 4.6
157336,Interstellar, 3.9
261023,Black Mass, 3.6
278,The Shawshank Redemption, 3
4232,Scream, 2.4
1893,Star Wars: Episode I - The Phantom Menace, 3.8
550,Fight Club, 4.2
98,Gladiator, 4
508,Love Actually, 2.1
228967,The Interview, 2.7
1895,Star Wars: Episode III - Revenge of the Sith, 3.5
18785,The Hangover, 1.9
67913,The Guard, 3.7
40807,50/50, 2.5
70160,The Hunger Games, 2.4
77930,Magic Mike, 1
72190,World War Z, 2.5
187017,22 Jump Street, 1.9
198663,The Maze Runner, 2
207703,Kingsman: The Secret Service, 2.1
99861,Avengers: Age of Ultron, 2
167073,Brooklyn, 2.8
254470,Pitch Perfect 2, 1.9
271718,Trainwreck, 1
314365,Spotlight, 4.9
259693,The Conjuring 2, 2.3
308266,War Dogs, 2.1
324786,Hacksaw Ridge, 3.1
330459,Rogue One: A Star Wars Story, 2.9
339846,Baywatch, 2
"""

standard_weights = {
    'cast_ft_weight': 0.3,
    'director_ft_weight': 0.4,
    'genre_ft_weight': 0.4,
    'content_weight': 0.7,
    'collab_weight': 1,
    'genre_normalisation': 0.12,
    'average_rating_weight': 0.3
}

def load_user_ratings():
    # Convert into a DataFrame
    user_ratings_df = pd.read_csv(StringIO(user_ratings), header=None, names=["movieId", "movie_name", "rating"])

    # Drop the movie_name column as it's unnecessary for appending to the ratings DataFrame
    user_ratings_df = user_ratings_df.drop(columns=["movie_name"])

    user_ratings_df['movieId'] = pd.to_numeric(user_ratings_df['movieId'], errors='coerce')
    user_ratings_df = user_ratings_df.dropna(subset=['movieId'])
    user_ratings_df['movieId'] = user_ratings_df['movieId'].astype(int)

    # Testing user id = 999,999
    user_ratings_df["userId"] = 999999

    print("User ratings DataFrame shape:", user_ratings_df.shape)

    # Reorder columns to match the existing ratings DataFrame
    return user_ratings_df[["userId", "movieId", "rating"]]

def load_user_ratings_from_profile(user_profile, films_df):
    u_ratings = user_profile['feature_profile']

    # For all the films in the user_profile['feature_profile'], get all the numeric id's (as there is actors, directors, genres mixed in)
    u_ratings = {k: v for k, v in u_ratings.items() if isinstance(v, (int, float)) and v is not None}

    # Convert into a DataFrame
    u_ratings_df = pd.DataFrame(u_ratings.items(), columns=["movieId", "rating"])
    u_ratings_df['movieId'] = pd.to_numeric(u_ratings_df['movieId'], errors='coerce')
    u_ratings_df = u_ratings_df.dropna(subset=['movieId'])
    u_ratings_df['movieId'] = u_ratings_df['movieId'].astype(int)
    u_ratings_df["userId"] = user_profile['id']
    print("User ratings DataFrame shape with non-film items:", u_ratings_df.shape)

    # Filter out any id's that are not in the films_df
    u_ratings_df = u_ratings_df[u_ratings_df['movieId'].isin(films_df['id'])]
    u_ratings_df = u_ratings_df.drop_duplicates(subset=['userId', 'movieId'])
    print("User ratings DataFrame shape after filtering out non films: ", u_ratings_df.shape)
    print("------------------")
    print("User ratings DataFrame after filtering out non films: ", u_ratings_df)
    print("\n------------------\n")
    # Reorder columns to match the existing ratings DataFrame
    u_ratings_df = u_ratings_df[["userId", "movieId", "rating"]]
    return u_ratings_df

genre_normalisation = 0.12

def user_feature_profile(user_id, films_df, usr_ratings, genre_list_mlb, feature_profile=None):
    """Creates a user profile based on ratings for movies with shared cast/directors."""
    if feature_profile is None:
        features_profile = {}
        add_id_flag = True
    else :
        features_profile = feature_profile.copy()
        add_id_flag = False

    for _, rating_row in usr_ratings.iterrows():
        movie_id = int(rating_row['movieId'])
        rating = rating_row['rating']

        # Apply penalty: Negative weight for ratings below 2.5
        if rating < 2.5:
            rating = -abs(2.5 - rating)  # Negative penalty

        # Add cast and director IDs to user features_profile with weighted ratings
        cast_ids = films_df.loc[films_df['id'] == movie_id, 'cast_info'].values[0]
        for cast_member in cast_ids:
            features_profile[cast_member[1]] = features_profile.get(cast_member[1], 0) + rating

        director_info = films_df.loc[films_df['id'] == movie_id, 'director_info'].values[0]
        if director_info is not None:
            features_profile[director_info[0]] = features_profile.get(director_info[0], 0) + rating

        # Add genre IDs to user features_profile with weighted ratings
        genre_score = genre_normalisation * rating

        for film_genre in genre_list_mlb:
            if films_df.loc[films_df['id'] == movie_id, film_genre].values[0] == 1:
                features_profile[film_genre] = features_profile.get(film_genre, 0) + genre_score

        if add_id_flag:
            features_profile['user_id'] = user_id


        # Round all scores to 3 decimal places
        features_profile = {k: round(v, 3) if isinstance(v, (int, float)) and v is not None else v for k, v in features_profile.items()}

    return features_profile

def empty_user_profile(user_id):
    """Creates an empty user profile."""
    return {'id': user_id, 'weights': standard_weights, 'feature_profile': {'user_id': user_id}}

# TODO - fix this problem child, turned features into a string
def load_or_create_user_profile(user_id, films_df, usr_ratings, genre_list_mlb):
    """Loads the user profile from a CSV file or creates it if it doesn't exist."""
    profiles_dir = os.path.join(os.path.dirname(__file__), '..', 'userProfiles')
    os.makedirs(profiles_dir, exist_ok=True)
    try:
        profile = load_user_profile(user_id, profiles_dir)
        return profile
    except FileNotFoundError:
        print("user ratings: ", usr_ratings)
        if usr_ratings.empty:
            return empty_user_profile(user_id)
        profile = {'id': user_id, 'weights': standard_weights,
                   'feature_profile': user_feature_profile(user_id, films_df, usr_ratings, genre_list_mlb)}

        # Save the user profile to a CSV file
        save_user_profile(profile, profiles_dir)

        return profile

def create_user_profile(user_id, films_df, usr_ratings, genre_list_mlb):
    # Create the directory relative to the current script
    profiles_dir = os.path.join(os.path.dirname(__file__), '..', 'userProfiles')
    os.makedirs(profiles_dir, exist_ok=True)

    profile = {'id': user_id, 'weights': standard_weights,
               'feature_profile': user_feature_profile(user_id, films_df, usr_ratings, genre_list_mlb)}

    # Save the user profile to a CSV file
    profile_df = pd.DataFrame([profile])
    profile_df.to_csv(os.path.join(profiles_dir, f'user_profile_{user_id}.csv'), index=False)


    # print("features_profile type (after3): ", type(profile_df['feature_profile'])) # this was a series?
    return profile

def load_user_profile(user_id, profiles_dir):
    """Loads the user profile from a CSV file."""
    profile_df = pd.read_csv(os.path.join(profiles_dir, f'user_profile_{user_id}.csv'))

    # Convert feature_profile from string to dictionary
    profile_df['feature_profile'] = profile_df['feature_profile'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    profile_df['weights'] = profile_df['weights'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    user_profile = profile_df.to_dict(orient='records')[0]
    print("User_profile loaded: ", user_profile)
    return user_profile

"""
I think the user user score (collaborative filtering) is more important for me in a recommendation system than a content based one. Keep your response short
"""

def adjust_user_profile(user_profile, sentiment_response, films_df, credits_df, alpha=0.3):
    """Adjusts the user profile based on sentiment feedback while ensuring stability, by using tanh."""
    category_sentiment, item_sentiment = parse_semantic_breakdown(sentiment_response)

    print("Category sentiment: ", category_sentiment)
    print("Item sentiment: ", item_sentiment)

    # Create mapping between sentiment categories and user profile weight keys
    category_mapping = {
        'Films': None,  # No direct mapping
        'Actors': 'cast_ft_weight',
        'Genres': 'genre_ft_weight',
        'Directors': 'director_ft_weight',
        'Filtering': None  # Special handling for this category
    }

    def update_score(current_score, adjustment):
        # TODO - currently if the score goes below 0.0, films with good correlation will be disincentive - fix with a floor
        """Uses tanh to taper off values around 3 while allowing smooth updates."""
        return max(0, round(3 * np.tanh((current_score + alpha * adjustment) / 3), 3))

    # Adjust category weights
    for category, sentiment in category_sentiment.items():
        print("Category: ", category)
        if category_mapping.get(category) and category_mapping[category] in user_profile["weights"]:
            weight_key = category_mapping[category]
            print(f"Mapping category '{category}' to weight key '{weight_key}'")
            print("Category score before update: ", user_profile["weights"][weight_key])
            user_profile["weights"][weight_key] = update_score(user_profile["weights"][weight_key], sentiment)
            print("Category score after update: ", user_profile["weights"][weight_key])

    print("\n-------------------------------\n")

    # Special handling for Filtering category which maps to content_weight and collab_weight
    if 'Filtering' in item_sentiment:
        for filter_type, sentiment in item_sentiment['Filtering'].items():
            if filter_type == 'Content' and 'content_weight' in user_profile["weights"]:
                print("Content filter score before update: ", user_profile["weights"]["content_weight"])
                user_profile["weights"]["content_weight"] = update_score(user_profile["weights"]["content_weight"],
                                                                         sentiment)
                print("Content filter score after update: ", user_profile["weights"]["content_weight"])

            if filter_type == 'Collaborative' and 'collab_weight' in user_profile["weights"]:
                print("Collaborative filter score before update: ", user_profile["weights"]["collab_weight"])
                user_profile["weights"]["collab_weight"] = update_score(user_profile["weights"]["collab_weight"],
                                                                        sentiment)
                print("Collaborative filter score after update: ", user_profile["weights"]["collab_weight"])

    # TODO - the user items are not in categories, tidying it will be handy
    # Regular item updates - handle films, genres, directors, actors
    for category, items in item_sentiment.items():
        # Turn all non numeric values ("None", "nan", "-", "", " ") into 0.0
        items = {k: v if isinstance(v, (int, float)) else 0.0 for k, v in items.items()}

        if category != 'Filtering':
            for item, sentiment in items.items():
                print(f"Item: '{item}' in category '{category}', sentiment: {sentiment}")
                if item in user_profile["feature_profile"]:
                    print(f"For item: '{item}' in category '{category}, score was {user_profile['feature_profile'][item]}")
                    user_profile["feature_profile"][item] = update_score(
                        user_profile["feature_profile"][item], sentiment
                    )
                    print("Item score after update: ", user_profile["feature_profile"][item])
                elif category == "Films":
                    # Handle films separately
                    film_id = get_film_id_by_title(item, films_df)
                    if film_id in user_profile["feature_profile"]:
                        print("Item score before update: ", user_profile["feature_profile"][film_id])
                        user_profile["feature_profile"][film_id] = update_score(
                            user_profile["feature_profile"][film_id], sentiment
                        )
                        print("Item score after update: ", user_profile["feature_profile"][film_id])
                    elif film_id is not None:
                        print(f"Film '{item}' not found in user profile, but ID found, film added to profile.")
                        # Initialize new item with a default score
                        user_profile["feature_profile"][film_id] = update_score(3, sentiment)
                    else:
                        print(f"Film '{item}' not found in films_df, skipping it.")
                elif category == "Actors":
                    # Handle actors separately
                    actor_id = get_actor_id_by_name(item, credits_df['cast_info'])
                    if actor_id in user_profile["feature_profile"]:
                        print("Item score before update: ", user_profile["feature_profile"][actor_id])
                        user_profile["feature_profile"][actor_id] = update_score(
                            user_profile["feature_profile"][actor_id], sentiment
                        )
                        print("Item score after update: ", user_profile["feature_profile"][actor_id])
                    elif actor_id is not None:
                        print(f"Actor '{item}' not found in user profile, adding it with a default score. Actor ID: {actor_id}")
                        # Initialize new item with a default score
                        user_profile["feature_profile"][actor_id] = update_score(3, sentiment)
                    else:
                        print(f"Actor '{item}' not found in user profile, skipping it.")
                else:
                    print(
                        f"Item '{item}' not found in user profile category, adding it with a default score.")

                    if category == "Directors":
                        # Firstly, check with fuzzy matching for director name
                        fuzzy_result = fuzzy_search_user_profile(user_profile["feature_profile"], item)
                        print("Fuzzy result: ", fuzzy_result)
                        if fuzzy_result:
                            print("Fuzzy match found for director: ", fuzzy_result)
                            user_profile["feature_profile"][fuzzy_result] = update_score(user_profile["feature_profile"][fuzzy_result], sentiment)
                            # TODO - future improvement - could add synonyms for items in the profile (jesus this'd be a pain)
                        else:
                            # If no fuzzy match, initialise new item with a default score
                            new_item_score = update_score(1.0, sentiment)
                            user_profile["feature_profile"][item] = new_item_score
                            print(f"Director '{item}' not found in user profile, adding it with a default score of {new_item_score}")

                    else:
                        # Initialize new item with a default score
                        print(f"Item '{item}' in category '{category}' not found in user profile, adding it with a default score.")
                        user_profile["feature_profile"][item] = update_score(1.0, sentiment)

    # print("User profile after adjustment: ")
    # pprint(user_profile)

    # Save the updated user profile
    profiles_dir = os.path.join(os.path.dirname(__file__), '..', 'userProfiles', 'adjustedProfiles')
    os.makedirs(profiles_dir, exist_ok=True)

    # Save the user profile to a CSV file - change the ID to the current timestamp
    profile = user_profile.copy()
    profile['id'] = int(pd.Timestamp.now().timestamp())
    print("Profile ID: ", profile['id'])

    # Save the user profile to a CSV file
    save_user_profile(profile, profiles_dir)
    return user_profile, profiles_dir


def save_user_profile(user_profile, profiles_dir):
    # Save the user profile to a CSV file
    profile_df = pd.DataFrame([user_profile])
    # profile_df.to_csv(f'/userProfiles/user_profile_{user_profile["id"]}.csv', index=False)
    profile_df.to_csv(os.path.join(profiles_dir, f'user_profile_{user_profile["id"]}.csv'), index=False)

    return user_profile

"""
Sample inputs for parse_semantic_breakdown function
 Films; 1.0, Blade Runner 2049: 0.95 \nDirectors; 0.4, Ridley Scott: -0.2 \nActors; 0.8, Ryan Gosling: 0.7
 
"""

def parse_semantic_breakdown(text):
    """Parses a semantic breakdown and extracts categories, items, and scores separately.
    Designed to be robust against various LLM output formats.
    """
    # Initialize results
    categories = {"Films": 0.0, "Actors": 0.0, "Genres": 0.0, "Directors": 0.0, "Filtering": 0.0}
    items_data = defaultdict(dict)

    # Clean and normalise input
    text = re.sub(r'\\n', '\n', text)
    print("Semantic breakdown:\n", text)

    # Extract category blocks from the text
    lines = text.strip().split('\n')

    # Define category mapping for singular/plural forms
    category_map = {
        "film": "Films", "films": "Films",
        "actor": "Actors", "actors": "Actors",
        "genre": "Genres", "genres": "Genres",
        "director": "Directors", "directors": "Directors",
        "filtering": "Filtering", "filter": "Filtering"
    }

    current_category = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for a new category
        found_category = False
        for cat_variant, std_cat in category_map.items():
            # Match at start of line or after punctuation, case insensitive
            pattern = rf'(^|\s|[,;])({cat_variant})\s*[;:]?\s*([-\d.]+)?'
            match = re.search(pattern, line.lower())

            if match:
                current_category = std_cat
                found_category = True

                # Extract category score if available
                if match.group(3):
                    try:
                        score = float(match.group(3))
                        categories[current_category] = score
                    except ValueError:
                        pass

                # Process items after the category declaration
                item_section = line[match.end():].strip()
                if item_section.startswith(','):
                    item_section = item_section[1:].strip()

                # Extract items with scores
                extract_items(item_section, current_category, items_data)
                break

        # If no category found, this line contains items for the current category
        if not found_category and current_category:
            extract_items(line, current_category, items_data)

    return categories, dict(items_data)

def extract_items(text, category, items_data):
    """Extract items and their scores from text"""
    if not text or not category:
        return

    # Match "Item: score" pattern
    item_score_pairs = re.findall(r'([^:,]+):\s*([-\d.]+)', text)

    if item_score_pairs:
        for item, score in item_score_pairs:
            item = item.strip()
            if item:
                try:
                    items_data[category][item] = float(score)
                except ValueError:
                    items_data[category][item] = 0.0
    else:
        # Handle comma-separated items without scores
        items = re.split(r',\s*', text)
        for item in items:
            item = item.strip()
            if item and not re.match(r'^[-\d.]+$', item):  # Make sure it's not just a number
                # Look for embedded scores like "Science Fiction 0.7"
                score_match = re.search(r'(.*)\s+([-\d.]+)$', item)
                if score_match:
                    try:
                        item_name = score_match.group(1).strip()
                        item_score = float(score_match.group(2))
                        items_data[category][item_name] = item_score
                    except ValueError:
                        items_data[category][item] = 0.0
                else:
                    items_data[category][item] = 0.0




# I really like the film the godfather tbh. The directing in it is great, so is the cast. I don't usually like the genres with it, so it's a surprising like of mine