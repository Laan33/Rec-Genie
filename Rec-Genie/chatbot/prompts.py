from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


film_chat_explore_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You're an assistant who's good at talking to users to find out about their film interests and what matters to them in a film"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

explain_rec_chain = PromptTemplate(
    input_variables=["title", "release_date", "score", "cast_score", "director_score", "genre_score", "user_user_score", "user_question"],
    template=(
    "You are talking to the user, briefly explain the recommendation given to the user for the given film. These are all attributes based off the users profile.\n\n"
    "Film Recommendation:\n"
    "Title: {title} ({release_date})\n"
    "Total Score: {score}\n"
    "Cast Proportion: {cast_proportion}\n"
    "Director Proportion: {director_proportion}\n"
    "Genre Proportion: {genre_proportion}\n"
    "Collaborative Filtering Proportion: {user_user_proportion}\n\n"
    "User Question: {user_question}\n\n"
    "Explain why this movie is recommended based on the given scores."
    )
)

glean_feed_back_chain = PromptTemplate(
    input_variables=["input_text"],
    template=(
        "Extract film preferences, filtering preferences, and sentiment scores from this user message. "
        "FORMAT THE OUTPUT WITH CATEGORIES AND SCORES LIKE THIS EXAMPLE, BUT ONLY USE ITEMS ACTUALLY MENTIONED OR IMPLIED IN THE USER MESSAGE BELOW: "
        "\n"
        "RULES:\n"
        "1. Each category score shows how important this category is to the user (0.1-1.0)\n"
        "2. Each item score shows how much the user likes that specific item (0.1-1.0)\n"
        "3. IMPORTANT: IGNORE THE EXAMPLE ITEMS AND ONLY EXTRACT ITEMS FROM THE USER MESSAGE\n"
        "4. Also include implied items by name not directly mentioned but associated with mentioned items:\n"
        "   a. For mentioned films, include their directors with a score of 60% of the film's score\n"
        "   b. For mentioned films, include their main actors with a score of 50% of the film's score\n"
        "   c. For mentioned films, include their genres with a score of 50% of the film's score\n"
        "   d. For mentioned directors, include their notable films with a score of 70% of the director's score\n"
        "   e. For mentioned actors, include their notable films with a score of 60% of the actor's score\n"
        "5. FORMAT MUST BE EXACTLY: 'Category; score, Item1: score, Item2: score \\n'\n"
        "6. No explanations or additional text - only the structured data\n"
        "7. Use negative scores (-0.1 to -0.5) for things the user explicitly dislikes or has negative sentiment about\n"
        "8. IF AN ITEM IS MENTIONED BUT NO CLEAR SENTIMENT IS GIVEN, DEFAULT TO 0.5\n"
        "9. ENSURE EVERY ITEM HAS A NUMERIC SCORE\n"
        "10. For implied items, multiply the sentiment score of the source item by the relationship factor\n"
        "11. ENSURE ALL EXPLICITLY MENTIONED FILMS, DIRECTORS, ACTORS AND GENRES ARE INCLUDED, even if sentiment is negative\n"
        "\n"
        "EXAMPLE FORMAT (DO NOT COPY THESE ITEMS, ONLY THE FORMAT):\n"
        "Directors; 0.5, Some Director: 0.7 \\n"
        "Films; 0.8, Some Film: 0.9, Another Film: -0.4 \\n"
        "Actors; 0.3, Some Actor: 0.5 \\n"
        "Genres; 0.6, Some Genre: 0.7 \\n"
        "Filtering; Content: 0.5, Collaborative: 0.6\\n"
        "\n"
        "USER MESSAGE: {input_text}"
    )
)


# Create a routing LLM with context about existing recommendations
# router_prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are a router that determines what the user is asking for. "
#      f"IMPORTANT: {'Recommendations HAVE already been generated and are available to explain.' if has_recommendations else 'No recommendations have been generated yet.'} "
#      "Respond ONLY with one of these exact categories: "
#      "- RECOMMEND: If the user is asking for new film recommendations "
#      "- EXPLAIN: If the user is asking for an explanation of the existing recommendation scores "
#      "- FEEDBACK: If the user is providing feedback or having a general conversation about films "
#      "- OTHER: If the query doesn't fit into any of the above categories"
#      ),
#     ("human", "{question}")
# ])

