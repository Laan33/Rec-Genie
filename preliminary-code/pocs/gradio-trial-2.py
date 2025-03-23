from random import randint
import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine


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


model_name = "llama3.2:latest"

llm = ChatOllama(model=model_name)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You're an assistant who's good at talking to users to find out about their film interests and what matters to them in a film"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm.bind(stop=["<|eot_id|>"]) | StrOutputParser()

# Create a connection using SQLAlchemy
engine = create_engine("sqlite:///sqlite.db")

with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection=engine
    ),
    input_messages_key="question",
    output_messages_key="output",
    history_messages_key="history"
)

def route(input_value, history, session_id):
    if "breakdown" in input_value.lower():
        return score_explainer()
    # elif "semantics" in input_value.lower():
    #     return semantics_scraper(
    else:
        semantics_scraper(input_value, history)
        return custom_chatbot(input_value, history, session_id)

def custom_chatbot(input_value, history, session_id):
    response = with_message_history.stream(
        {"ability": "everything", "question": input_value},
        config={"configurable": {"session_id": session_id}},
    )
    full_response = ''
    for item in response:
        full_response += item
        # yield full_response
    yield full_response

def score_explainer():
    response = ""

    yield response

def semantics_scraper(input_value, history):
    response = ""

    yield response


with gr.Blocks(theme="Soft") as demo:
    # user_profile = gr.Markdown()
    with gr.Row():
        gr.Markdown("# Film Recommendation Chatbot - Llama 3.2")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Session ID")
            session_id_num = gr.Number(
                value=randint(200, 1000),
                label="Session ID",
                interactive=True,
                info="Session ID to use for chat history",
                minimum=1,
                maximum=1000000,
                step=1
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
        # additional_outputs=[user_profile],
        examples=[
            ["I love the Barbie film"],
            ["I really like Eddie Murphy in Shrek, and it's my favourite film"],
            ["I think the director is really important in making or breaking a film"],
            ["I liked the cast in the last film I saw, but they the casting didn't make the film for me"],
        ]
        )



demo.launch()

