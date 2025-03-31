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
        "Analyze the sentiment of the following sentence. "
        "Provide sentiment scores on the importance of the following features followed with the sentiment scores for items within these features: "
        "- Films "
        "- Actors "
        "- Directors "
        "- Genres "
        "Add a `\n` before each feature, and a `,` between each item rating. "
        
        "- For item ratings, just write i: score, e.g. 'item1: 0.5'. "
        "- For feature importance, just write f; score, e.g. 'feature1; 0.5'. "
        "For example: "
        "\n Films; 0.5 item1: 0.9, item2: 0.2 \n Actors; 0.3 item1: 0.5, item2: 0.1 "
        "ONLY provide the item names and associated scores, no other chit chat. "
        "Sentence: {input_text}"
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

