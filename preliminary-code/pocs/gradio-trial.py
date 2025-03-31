# import numpy as np
#
#
# def adjust_user_profile(user_profile, category_sentiment, item_sentiment, alpha=0.5):
#     """Adjusts the user profile based on sentiment feedback while ensuring stability."""
#
#     def update_score(current_score, adjustment):
#         """Uses tanh to taper off values around 3 while allowing smooth updates."""
#         return round(3 * np.tanh((current_score + alpha * adjustment) / 3), 2)
#
#     # Adjust category weights
#     for category, sentiment in category_sentiment.items():
#         if category in user_profile["categories"]:
#             user_profile["categories"][category] = update_score(user_profile["categories"][category], sentiment)
#
#     # Adjust item scores
#     for category, items in item_sentiment.items():
#         if category in user_profile["items"]:
#             for item, sentiment in items.items():
#                 if item in user_profile["items"][category]:
#                     user_profile["items"][category][item] = update_score(user_profile["items"][category][item],
#                                                                          sentiment)
#                 else:
#                     # Initialize new item with a default score (e.g., 1.0)
#                     user_profile["items"][category][item] = update_score(1.0, sentiment)
#
#     return user_profile
#
#
# # Example user profile
# user_profile = {
#     "categories": {"Films": 1.5, "Actors": 0.8, "Genres": 1.0, "Directors": 2.0},
#     "items": {"Films": {"Shrek": 1.2}, "Actors": {"Brad Pitt": 0.9}}
# }
#
# print("Initial User Profile:")
# print("Categories:", user_profile["categories"])
# print("Items:", user_profile["items"])
#
#
# # Example sentiment adjustments
# category_sentiment = {"Films": 0.5, "Actors": -0.9, "Genres": 0.9}
# item_sentiment = {"Films": {"Shrek": 0.2}, "Actors": {"Brad Pitt": -0.4}}
#
# updated_profile = adjust_user_profile(user_profile, category_sentiment, item_sentiment)
#
# print("\nUpdated User Profile:")
# print("Categories:", updated_profile["categories"])
# print("Items:", updated_profile["items"])
