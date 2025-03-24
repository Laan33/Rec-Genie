from random import randint
import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine

from rec_sys.rec_interface import RecInterface
from chatbot.prompts import film_chat_explore_chain, explain_rec_chain, glean_feed_back_chain
# from chatbot import

sample_json = {
                    "id": 1,
                    "feature_profile": {
                        "cast": ["Tom Cruise", "Nicole Kidman"],
                        "director": ["Steven Spielberg"],
                        "genre": ["Action", "Adventure"]
                    },
                    "weights": {
                        "cast_ft_weight": 0.3,
                        "director_ft_weight": 0.4,
                        "genre_ft_weight": 0.4,
                        "content_weight": 0.7,
                        "collab_weight": 1,
                        "genre_normalisation": 0.12,
                        "average_rating_weight": 0.3
                    }
}

# model_name = "llama3.2:latest"
model_name = "llama3.2:3b-instruct-q4_K_M"

llm = ChatOllama(model=model_name)


class GradioFilmRec:
    def __init__(self):
        self.model_name = model_name
        self.rec_interface = RecInterface()
        self.session_id = randint(1000000, 9999999)

    def recommend_films(self):
        # RecInterface.recommend(user_id=session_id_num.value)
        print("Recommendations generated!!!!.")

        recommendations = RecInterface.recommend(self.rec_interface, user_id=self.session_id, num_recommendations=5)

        return recommendations, score_explainer(self, recommendations)


    def custom_chatbot(self, input_value, history, session_id):
        with_message_history = update_chain(film_chat_explore_chain)
        response = with_message_history.stream(
            {"ability": "everything", "question": input_value},
            config={"configurable": {"session_id": session_id}},
        )
        full_response = ''
        for item in response:
            full_response += item
            yield full_response
        semantics_scraper(self, input_value, history)
        yield full_response


# Create a connection using SQLAlchemy
engine = create_engine("sqlite:///sqlite.db")

def update_chain(chain):
    chain = chain | llm.bind(stop=["<|eot_id|>"]) | StrOutputParser()
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: SQLChatMessageHistory(
            session_id=session_id, connection=engine
        ),
        input_messages_key="question",
        output_messages_key="output",
        history_messages_key="history"
    )


def recommend_films(self, num_recommendations=5):
    # RecInterface.recommend(user_id=session_id_num.value)
    print("Recommendations generated!!!!.")

    scores = RecInterface.recommend(self.rec_interface, user_id=session_id_num.value)

    return scores, score_explainer(self, scores)


def custom_chatbot(self, input_value, history, session_id):
    with_message_history = update_chain(film_chat_explore_chain)

    response = with_message_history.stream(
        {"ability": "everything", "question": input_value},
        config={"configurable": {"session_id": session_id}},
    )
    full_response = ''
    for item in response:
        full_response += item
        yield full_response
    print("Debug test (this happens once a response correct?)")
    semantics_scraper(self, input_value, history) # Only call after the visible response has been generated
    yield full_response

def score_explainer(self, breakdown):
    with_message_history = update_chain(explain_rec_chain)

    response = with_message_history.stream(
        {"ability": "everything", "score": breakdown},
        config={"configurable": {"session_id": self.session_id}},
    )

    yield response

def semantics_scraper(self, input_value, history):
    """
    Outputs the sentiment of the user's message for feature and item sentiment extraction.
    """
    history = history # Placeholder for now

    with_message_history = update_chain(glean_feed_back_chain)

    response = with_message_history.stream(
        {"ability": "everything", "input_text": input_value},
        config={"configurable": {"session_id": self.session_id}},
    )

    full_response = ''
    for item in response:
        full_response += item
        # yield full_response
    print("Full response: ", full_response)
    yield full_response





with gr.Blocks(theme="Soft") as demo:
    # user_profile = gr.Markdown()
    with gr.Row():
        gr.Markdown("# Film Recommendation Chatbot - Llama 3.2")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Session ID")
            session_id_num = gr.Number(
                value=randint(1000000, 9999999), # Biggest User ID in the dataset is 999,999
                label="Session ID",
                interactive=True,
                info="Session ID to use for chat history",
                minimum=1,
                maximum=1000000,
                step=1
            )
            gr.Markdown("Generate recommendations")
            generate_recommendations = gr.Button(
                value="Generate recommendations",
                # label="Generate recommendations",
            )
        with gr.Column(scale=2):
            gr.Markdown("### User profile")
            user_profile = gr.JSON(
                value=[sample_json],
                label="User profile"
            )
        with gr.Column(scale=2):
            gr.Markdown("### User profile")
            message_semantics = gr.JSON(
                value=[sample_json],
                label="Message semantics"
            )

    chatbot_interface = gr.ChatInterface(
        fn=custom_chatbot,
        chatbot=gr.Chatbot(type="messages", show_copy_button=True),
        editable=True,
        type="messages",
        additional_inputs=[session_id_num],
        examples=[
            ["I love the Barbie film, I'm just Ken in a Barbie world"],
            ["I really like Eddie Murphy as Donkey in Shrek, really, it's my favourite film"],
            ["I think the director is really important in making or breaking a film"],
            ["I liked the cast in the last film I saw, but they the casting didn't make the film for me"],
        ]
    )
    generate_recommendations.click(
        fn=recommend_films,
        inputs=[],
        outputs=[]
    )



demo.launch()

