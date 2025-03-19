import importlib

import pandas as pd
import data_loader
import user_profile as user_pf
import preprocessing as pre
import hybrid_recommender as hyb



# Change this to True if you are running on Google Colab
RUNNING_ON_COLAB = False

# Constants
USER_ID = 999999
load_original_credits = True


def load_data():
    if RUNNING_ON_COLAB:
        data_loader.mount_drive()

    # Load the data
    films_df = data_loader.load_movies()
    ratings_df = data_loader.load_ratings()
    credits_df = data_loader.load_credits(load_original_credits)

    print("Data dimensions:")
    print("films_df: ", films_df.shape)
    print("ratings_df: ", ratings_df.shape)
    print("credits_df: ", credits_df.shape)

    return films_df, ratings_df, credits_df

# Data processing
def process_data(films_df, ratings_df, credits_df):
    films_df = pre.filter_films(films_df)

    ohe_films_df, genre_list_mlb = pre.one_hot_encode_genres(films_df)
    if load_original_credits:
        credits_df = pre.condense_credits(credits_df)

    films_df = pre.data_tidying(ohe_films_df, credits_df)
    return films_df, credits_df, genre_list_mlb

def user(user_id, films_df, ratings_df, genre_list_mlb):
    user_ratings_df = user_pf.load_user_ratings()
    ratings_df = pd.concat([ratings_df, user_ratings_df], ignore_index=True).drop_duplicates(subset=['userId', 'movieId'])
    user_profile = user_pf.create_user_profile(user_id, films_df, user_ratings_df, genre_list_mlb)
    return user_profile, ratings_df

def recommend(user_profile, films_df, credits_df, ratings_df, genre_list_mlb):
    recommendations = hyb.hybrid_recommend(user_profile, films_df, credits_df, ratings_df, genre_list_mlb)
    return recommendations

def __init__():
    importlib.reload(data_loader)
    importlib.reload(user_pf)
    importlib.reload(pre)
    importlib.reload(hyb)
    films_df, ratings_df, credits_df = load_data()
    films_df, credits_df, genre_list_mlb = process_data(films_df, ratings_df, credits_df)


def main():
    films_df, ratings_df, credits_df = load_data()
    films_df, credits_df, genre_list_mlb = process_data(films_df, ratings_df, credits_df)
    user_profile, ratings_df = user(USER_ID, films_df, ratings_df, genre_list_mlb)
    recommendations = recommend(user_profile, films_df, credits_df, ratings_df, genre_list_mlb)
    return recommendations



class Chatbot:
    def __init__(self):
        print("Loading data!")
        self.films_df, self.ratings_df, self.credits_df = load_data()
        print("Data loaded successfully.")
        self.films_df, self.credits_df, self.genre_list_mlb = process_data(self.films_df, self.ratings_df, self.credits_df)

        print("Data processed successfully.")
    #
    # def load_data(self):
    #     # Implement your data loading logic here
    #     films_df, ratings_df, credits_df = load_data()
    #     return data

    def handle_interaction(self, user_input):
        # Implement your chatbot interaction logic here
        response = f"Received: {user_input}"
        return response

def chatbot_main():
        chatbot = Chatbot()
        print("Chatbot is running. Type 'exit' to stop.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Stopping chatbot.")
                break
            response = chatbot.handle_interaction(user_input)
            print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot_main()


