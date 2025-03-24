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

DEPRECATED_film_chat_explore_chain = PromptTemplate.from_template(
    """You are a friendly and knowledgeable assistant specializing in films. Your goal is to engage users in conversations to understand their film preferences, interests, and what aspects of films are most important to them. \
    Keep the conversation a reasonably short length, and ask questions to get more information from the user. \

    Respond to the following question:

    Question: {question}
    Answer:
    """
) | OllamaLLM(model="llama3.2:3b-instruct-q4_K_M")

DEPRECATED_explain_rec_chain = PromptTemplate.from_template(
    """You are talking to the user, briefly explain the recommendation score breakdown to the user for the given film. 
    The recommendation score format is:
    `Title (Release Date): total_score, cast_score, director_score, genre_score, collaborative_filtering_score`. 
    total_score is how likely this film is to be recommended to the user.
    the other scores are the content based scores of the user with collaborative filtering score based on other uses reviews.
    Given the following breakdown:

    {recommendation_text}

    You need to concisely and briefly say to the user why the film might be recommended. e.g. "you liked members of the cast in other films before" if the cast score is high.

    Answer:
    """
) | OllamaLLM(model="llama3.2:3b-instruct-q4_K_M")

explain_rec_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are talking to the user, briefly explain the recommendation_text to the user for the given film.",
     "The recommendation score format is: `Title (Release Date): total_score, cast_score, director_score, genre_score, collaborative_filtering_score`. "
     "total_score is how highly recommended this item is to the user."
     "The collaborative filtering score is based on other uses reviews, given the users ratings."
     "The other scores are all content based scores for the user, based on the user's previous ratings and preferences.",
     ),
    MessagesPlaceholder(variable_name="recommendation_text"),
    # MessagesPlaceholder(variable_name="history")
])

glean_feed_back_chain = ChatPromptTemplate.from_messages([
    ("system",
     "Analyze the sentiment of the following sentence.",
     "Provide sentiment scores on the importance of the following features followed with the sentiment scores for items within these features:",
     "- Films",
     "- Actors",
     "- Directors",
     "- Genres",
     "Add a `\n` before each feature, and a `,` between each item rating.",
     "Don't be verbose, just provide the scores with no other info.",
     "- For item ratings, just write i: score, e.g. 'item1: 0.5'.",
     "- For feature importance, just write f; score, e.g. 'feature1; 0.5'.",
     "For example:",
     "\n Films; 0.5 item1: 0.9, item2: 0.2 \n Actors; 0.3 item1: 0.5, item2: 0.1",
     "Sentence: {input_text}",
     ),
    ("human", "{question}"),
])

DEPRECATED_glean_feed_back_chain = PromptTemplate.from_template(
    """
    Analyze the sentiment of the following sentence.

    Provide sentiment scores on the importance of the following features followed with the sentiment scores for items within these features:
    - Films
    - Actors
    - Directors
    - Genres
    Add a `\n` before each feature, and a `,` between each item rating.

    Don't be verbose, just provide the scores with no other info.
    - For item ratings, just write i: score, e.g. "item1: 0.5".
    - For feature importance, just write f; score, e.g. "feature1; 0.5".
    For example:
    \n Films; 0.5 item1: 0.9, item2: 0.2 \n Actors; 0.3 item1: 0.5, item2: 0.1

    Sentence: {input_text}
    """
) | OllamaLLM(model="llama3.2:3b-instruct-q4_K_M")
