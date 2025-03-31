from typing import Annotated, TypedDict

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



# TyperdDict
class SentimentScores(TypedDict):
    """Sentiment scores for items within features, and feature importance scores."""

    films: float # Sentiment score for films
    actors: float # Sentiment score for actors
    directors: float # Sentiment score for directors
    genres: float # Sentiment score for genres



film_chat_explore_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You're an assistant who's good at talking to users to find out about their film interests and what matters to them in a film"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

explain_rec_chain = PromptTemplate(
    input_variables=["title", "release_date", "score", "cast_score", "director_score", "genre_score", "user_user_score"],
    template=(
        "You are talking to the user, briefly explain the recommendation for the given film. These are all attributes based off the users profile.\n\n"
        "Film Recommendation:\n"
        "Title: {title} ({release_date})\n"
        "Total Score: {score}\n"
        "Cast Score: {cast_score}\n"
        "Director Score: {director_score}\n"
        "Genre Score: {genre_score}\n"
        "Collaborative Filtering Score: {user_user_score}\n\n"
        "Explain why this movie is recommended based on the given scores."
    )
)

glean_feed_back_chain = PromptTemplate(
    input_variables=["input_text"],
    template=(
        "Extract film preferences, filtering preferences, and sentiment scores from this user message. "
        "FORMAT THE OUTPUT EXACTLY LIKE THIS EXAMPLE, ONLY USE ITEMS MENTIONED IN THE USER MESSAGE: "
        "Directors; 0.9, Christopher Nolan: 0.9 \n"
        "Films; 1.0, Interstellar: 0.8, Inception: 0.8 \n"
        "Actors; 0.2, Brad Pitt: 0.7 \n"
        "Genres; 0.5, Action: 0.6, Comedy: 0.3 \n"
        "Filtering; Content: 0.7, Collaborative: 0.8 \n"
        "\n"
        "RULES:\n"
        "1. Each category score shows how important this category is to the user (0.1-1.0)\n"
        "2. Each item score shows how much the user likes that specific item (0.1-1.0)\n"
        "3. Only include categories and items actually mentioned in the message\n"
        "4. Format must be exactly: 'Category; score, Item1: score, Item2: score \\n'\n"
        "5. No explanations or additional text - only the structured data\n"
        "6. Use negative scores (-0.1 to -1.0) for things the user dislikes\n"
        "7. If the user doesn't express a preference for a category, provide a neutral score of 0.0\n"
        "8. For the Filtering category, extract 'Content' score (how much the user values film attributes) "
        "   and 'Collaborative' score (how much the user values what others with similar taste liked)\n"
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

