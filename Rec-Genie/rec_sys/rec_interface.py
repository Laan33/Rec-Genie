import re
from collections import defaultdict
import pandas as pd
from . import data_loader
from . import user_profile as user_pf
from . import preprocessing as pre
from . import hybrid_recommender as hyb
# import AttributeSearch


# Configuration
RUNNING_ON_COLAB = False
# USER_ID = 999999
# load_original_credits = True
load_original_credits = False


def load_data(num_lines=None):
    if RUNNING_ON_COLAB:
        data_loader.mount_drive()
    if num_lines is not None:
        data_loader.set_num_lines(num_lines)
    films_df = data_loader.load_movies()
    ratings_df = data_loader.load_ratings()
    credits_df = data_loader.load_credits(load_original_credits)

    print("-------\nData dimensions:")
    print("films_df:", films_df.shape)
    print("ratings_df:", ratings_df.shape)
    print("credits_df:", credits_df.shape)
    print("-------\n")

    return films_df, ratings_df, credits_df

def parse_semantic_breakdown(text):
    """Parses a semantic breakdown and extracts categories, items, and scores."""
    parsed_data = defaultdict(dict)  # Dictionary to store parsed categories and scores

    for line in text.strip().split('\n'):
        if not line:
            continue

        match = re.match(r"(\w+);\s*([\d.-]*)\s*(.*)", line)
        if match:
            category, category_score, items = match.groups()
            category = category.strip()

            # Store category score if available
            if category_score:
                parsed_data[category]['_score'] = float(category_score)

            # Extract items and their scores
            if items:
                item_matches = re.findall(r"([^:,]+):\s*([-\d.]+)", items)
                for item, score in item_matches:
                    parsed_data[category][item.strip()] = float(score)

    return dict(parsed_data)


class RecInterface:

    def __init__(self, user_id):
        self.user_profile = None
        self.user_ratings_df = None
        self.user_id = user_id
        # self.films_df, self.ratings_df, self.credits_df = load_data()
        self.films_df, self.ratings_df, self.credits_df = load_data(num_lines=None)
        # self.attribute_search = AttributeSearch('path/to/films.csv', 'path/to/credits.csv')
        self.films_df, self.credits_df, self.genre_list_mlb = self.process_data()
        print("User ID:", user_id)

        self.update_user_profile(user_id)
        # print("User profile user_id:", self.user_profile['id'])
        # self.user_ratings_df = self.update_user_profile()
        # self.user_ratings_df = self.ratings_df[self.ratings_df['userId'] == user_id]


        # self.user_profile = user_pf.load_or_create_user_profile(user_id, self.films_df, self.user_ratings_df, self.genre_list_mlb)
        # print("User profile user_id:", self.user_profile['id'])
        print("\nRecInterface initialized")

    def process_data(self):
        films_df = pre.filter_films(self.films_df)
        ohe_films_df, genre_list_mlb = pre.one_hot_encode_genres(films_df)

        if load_original_credits:
            self.credits_df = pre.condense_credits(self.credits_df)

        films_df = pre.data_tidying(ohe_films_df, self.credits_df)
        return films_df, self.credits_df, genre_list_mlb

    def update_user_profile(self, user_id):
        if load_original_credits:
            self.user_ratings_df = user_pf.load_user_ratings()
            self.ratings_df = pd.concat([self.ratings_df, self.user_ratings_df], ignore_index=True)
            print("Ratings df shape: ", self.ratings_df.shape)
            self.ratings_df = self.ratings_df.drop_duplicates(subset=['userId', 'movieId'])
            print("Ratings df shape after dropping duplicates: ", self.ratings_df.shape)
            self.user_profile = user_pf.create_user_profile(user_id, self.films_df, self.user_ratings_df, self.genre_list_mlb)
            # print("Feature user profile type4: ", type(self.user_profile['feature_profile'])) # this is a dict
            # print("Feature user profile: ", self.user_profile['feature_profile'])
        else:
            print("Loading user ratings from file")
            # user_profile  = user_pf.load_or_create_user_profile(user_id, self.films_df, self.ratings_df, self.genre_list_mlb)
            # Convert from a string
            self.user_profile = user_pf.load_or_create_user_profile(user_id, self.films_df, self.ratings_df, self.genre_list_mlb)

            self.user_ratings_df = user_pf.load_user_ratings()

            # print("Ratings df shape: ", self.ratings_df.shape)
            self.ratings_df = pd.concat([self.ratings_df, self.user_ratings_df], ignore_index=True)
            # print("Ratings df shape after concatenation: ", self.ratings_df.shape)
            self.ratings_df = self.ratings_df.drop_duplicates(subset=['userId', 'movieId'])
            # print("Ratings df shape after dropping duplicates: ", self.ratings_df.shape)

            # print("Feature user profile type4: ", type(self.user_profile['feature_profile'])) # this is a dict
            # print("Feature user profile: ", self.user_profile['feature_profile'])

    def recommend(self, num_recommendations=5):
        print("Generating recommendations for user ID:", self.user_id)
        if self.user_profile is None:
            raise ValueError("User profile not initialized. Call update_user_profile() first.")
        recommendations = hyb.hybrid_recommend(self.user_profile, self.films_df, self.credits_df, self.ratings_df, self.genre_list_mlb)
        return recommendations[:num_recommendations]

    def score_breakdown(self, recommendations):
        return hyb.score_breakdown(self.films_df, recommendations)

    def implement_user_feedback(self, session_id, semantics_response):
        # Implement user feedback using the semantics_response
        # This could involve updating the user profile or modifying the recommendation algorithm
        feedback_feature_weights, feedback_items = parse_semantic_breakdown(semantics_response)


