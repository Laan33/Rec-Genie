import pandas as pd
from . import data_loader
from . import user_profile as user_pf
from . import preprocessing as pre
from . import hybrid_recommender as hyb


"""
Scenario 1: ID:1,111,111 - comedy and romcom
Scenario 2: ID:2,222,222 - children's films
Scenario 3: ID:3,333,333 - action and thriller
"""
# IMPORTANT - Choose scenario here
SCENARIO_CHOICE = 20

# Configuration
RUNNING_ON_COLAB = False
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

def get_user_id_from_scenario(scenario_number):
    scenario_map = {
        1: 1111111, # Comedy and Romcom
        2: 2222222, # Children's Films
        3: 3333333 # Action and Thriller
    }
    return scenario_map.get(scenario_number, 999999)  # Default to 999,999 if scenario_number is not in the map

class RecInterface:

    def __init__(self, session_id):
        self.session_id = session_id

        # IMPORTANT - Choose scenario here
        self.user_id = get_user_id_from_scenario(SCENARIO_CHOICE)

        self.current_profile_path = None
        self.user_profile = None
        self.user_ratings_df = None
        self.films_df, self.ratings_df, self.credits_df = load_data(num_lines=None)
        self.films_df, self.credits_df, self.genre_list_mlb = self.process_data()
        print("User ID:", self.user_id)

        self.update_user_profile()
        print("\nRecInterface initialized")

    def process_data(self):
        films_df = pre.filter_films(self.films_df)
        ohe_films_df, genre_list_mlb = pre.one_hot_encode_genres(films_df)

        if load_original_credits:
            self.credits_df = pre.condense_credits(self.credits_df)

        films_df = pre.data_tidying(ohe_films_df, self.credits_df)
        return films_df, self.credits_df, genre_list_mlb

    def update_user_profile(self):
        if load_original_credits:
            self.user_ratings_df = user_pf.load_user_ratings(self.user_id)
            self.ratings_df = pd.concat([self.ratings_df, self.user_ratings_df], ignore_index=True)
            print("Ratings df shape: ", self.ratings_df.shape)
            self.ratings_df = self.ratings_df.drop_duplicates(subset=['userId', 'movieId'])
            print("Ratings df shape after dropping duplicates: ", self.ratings_df.shape)
            self.user_profile = user_pf.create_user_profile(self.user_id, self.films_df, self.user_ratings_df, self.genre_list_mlb)
        else:
            print("Loading user ratings from file")
            self.user_ratings_df = user_pf.load_user_ratings(user_id=self.user_id)

            # Convert from a string
            self.user_profile = user_pf.load_or_create_user_profile(self.user_id, self.films_df, self.user_ratings_df, self.genre_list_mlb)

            self.ratings_df = pd.concat([self.ratings_df, self.user_ratings_df], ignore_index=True)
            self.ratings_df = self.ratings_df.drop_duplicates(subset=['userId', 'movieId'])



    def recommend(self, num_recommendations=5):
        print("Generating recommendations for user ID:", self.user_id)
        if self.user_profile is None:
            raise ValueError("User profile not initialized. Call update_user_profile() first.")
        recommendations = hyb.hybrid_recommend(self.user_profile, self.films_df, self.credits_df, self.ratings_df, self.genre_list_mlb)
        return recommendations[:num_recommendations]

    def score_breakdown(self, recommendations):
        return hyb.score_breakdown(self.films_df, recommendations)

    def implement_user_feedback(self, session_id, sentiment_response):
        # Adjust the weights on the user profile
        self.user_profile, self.current_profile_path = user_pf.adjust_user_profile(self.user_profile, sentiment_response, self.films_df, self.credits_df)


    def reload_user_profile(self):
        # Reload the user profile from the file
        if self.current_profile_path:
            print("\nReloading user ratings from path:", self.current_profile_path)
            print("Ratings df shape before concatenation: ", self.ratings_df.shape)
            print("Self.user_ratings_df shape before concatenation: ", self.user_ratings_df.shape)
            new_user_ratings = user_pf.load_user_ratings_from_profile(self.user_profile, self.films_df)

            print("Building new user profile feature profile")
            new_features_profile = (
                user_pf.user_feature_profile(self.user_profile, self.films_df, new_user_ratings, self.genre_list_mlb, feature_profile=self.user_profile['feature_profile']))
            # print("User profile feature shape after reloading: ", self.user_profile['feature_profile'].shape)
            self.user_profile['feature_profile'] = new_features_profile
            self.user_ratings_df = pd.concat([self.user_ratings_df, new_user_ratings], ignore_index=True)
            self.user_ratings_df = self.user_ratings_df.drop_duplicates(subset=['userId', 'movieId'])
            print("User ratings df shape after dropping duplicates: ", self.user_ratings_df.shape)

            self.ratings_df = pd.concat([self.ratings_df, self.user_ratings_df], ignore_index=True)
            self.ratings_df = self.ratings_df.drop_duplicates(subset=['userId', 'movieId'])
            print("Ratings df shape after concatenation & duplicate removal: ", self.ratings_df.shape)

        else:
            print("\nWARNING: No user profile path found. Please update the user profile first.\n")


# I'm really vibing with tarantino recently, he's a great director, and I'm loving all his films. I also like the Cornetto Triology Films, but I don't like the John Wick series
