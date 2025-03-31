# import re
# from collections import defaultdict
#
#
# def parse_semantic_breakdown(text):
#     """Parses a semantic breakdown and extracts categories, items, and scores separately."""
#     categories = {"Films": 0.0, "Actors": 0.0, "Genres": 0.0, "Directors": 0.0}  # Default weights
#     items_data = defaultdict(dict)  # Dictionary to store parsed categories and scores
#
#     for line in text.strip().split('\n'):
#         line = line.strip().lstrip('#').strip()  # Remove leading '#' and spaces
#         if not line:
#             continue
#
#         match = re.match(r"(\w+);\s*([\d.-]*)\s*(.*)", line)
#         if match:
#             category, category_score, items = match.groups()
#             category = category.strip()
#
#             # Store category score if available and belongs to the four main categories
#             if category in categories and category_score:
#                 categories[category] = float(category_score)
#
#             # Extract items and their scores
#             if items:
#                 item_matches = re.findall(r"([^:,]+):\s*([-\d.]+)", items)
#                 for item, score in item_matches:
#                     items_data[category][item.strip()] = float(score)
#
#     return categories, dict(items_data)
#
#
# # Example usage
# sample_1 = """
# # Films; 1.0, Shrek: 0.9, Donkey: 0.9
# # Directors;
# # Actors; 0.2
# # Genres;
# """
#
# sample_2 = """
# # Directors; 0.9, Denis Villeneuve: 0.9
# # Films; 1.0, Blade Runner 2049: 0.8
# # Actors; 1.0, Harrison Ford: 0.9, Ryan Gosling: 0.7, Ana de Armas: 0.8
# """
# sample_1_categories, sample_1_items = parse_semantic_breakdown(sample_1)
# sample_2_categories, sample_2_items = parse_semantic_breakdown(sample_2)
#
# print("Sample 1 Categories:", sample_1_categories)
# print("Sample 1 Items:", sample_1_items)
# print("Sample 2 Categories:", sample_2_categories)
# print("Sample 2 Items:", sample_2_items)
