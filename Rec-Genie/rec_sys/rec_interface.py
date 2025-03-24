import importlib
import pandas as pd
from . import data_loader
from . import user_profile as user_pf
from . import preprocessing as pre
from . import hybrid_recommender as hyb
# import AttributeSearch


# Configuration
RUNNING_ON_COLAB = False
USER_ID = 999999
load_original_credits = True

class RecInterface:
    # def __init__(self):
    #     self.films_df, self.ratings_df, self.credits_df = self.load_data()
    #     # self.attribute_search = AttributeSearch('path/to/films.csv', 'path/to/credits.csv')
    #     self.films_df, self.credits_df, self.genre_list_mlb = self.process_data()
    #     self.user_profile = user_pf.load_or_create_user_profile(USER_ID, self.films_df, self.ratings_df, self.genre_list_mlb)
    #     print("RecInterface initialized")

    def __init__(self):# placeholder for testing - no need to load everything
        self.films_df, self.ratings_df, self.credits_df = self.load_data(num_lines=5000)
        # self.attribute_search = AttributeSearch('path/to/films.csv', 'path/to/credits.csv')
        self.films_df, self.credits_df, self.genre_list_mlb = self.process_data()

        self.user_ratings_df = self.ratings_df[self.ratings_df['userId'] == USER_ID]

        self.user_profile = user_pf.load_or_create_user_profile(USER_ID, self.films_df, self.user_ratings_df, self.genre_list_mlb)
        print("RecInterface initialized")

    def load_data(self, num_lines=None):
        if RUNNING_ON_COLAB:
            data_loader.mount_drive()
        if num_lines is not None:
            data_loader.set_num_lines(num_lines)
        films_df = data_loader.load_movies()
        ratings_df = data_loader.load_ratings()
        credits_df = data_loader.load_credits(load_original_credits)

        print("Data dimensions:")
        print("films_df:", films_df.shape)
        print("ratings_df:", ratings_df.shape)
        print("credits_df:", credits_df.shape)

        return films_df, ratings_df, credits_df

    def process_data(self):
        films_df = pre.filter_films(self.films_df)
        ohe_films_df, genre_list_mlb = pre.one_hot_encode_genres(films_df)

        if load_original_credits:
            self.credits_df = pre.condense_credits(self.credits_df)

        films_df = pre.data_tidying(ohe_films_df, self.credits_df)
        return films_df, self.credits_df, genre_list_mlb

    def update_user_profile(self, user_id):
        user_ratings_df = user_pf.load_user_ratings()
        self.ratings_df = pd.concat([self.ratings_df, user_ratings_df], ignore_index=True)
        self.ratings_df = self.ratings_df.drop_duplicates(subset=['userId', 'movieId'])
        self.user_profile = user_pf.create_user_profile(user_id, self.films_df, user_ratings_df, self.genre_list_mlb)

    def recommend(self, num_recommendations=5):
        if self.user_profile is None:
            raise ValueError("User profile not initialized. Call update_user_profile() first.")
        recommendations = hyb.hybrid_recommend(self.user_profile, self.films_df, self.credits_df, self.ratings_df, self.genre_list_mlb)
        return recommendations[:num_recommendations]



# def main():
#     rec_system = RecInterface()
#     rec_system.update_user_profile(USER_ID)
#     recommendations = rec_system.recommend()
#     print(recommendations)
#     return recommendations
#
# if __name__ == "__main__":
#     main()
