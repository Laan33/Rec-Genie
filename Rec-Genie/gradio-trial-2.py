import gradio as gr
from random import randint
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import SQLChatMessageHistory
from sqlalchemy import create_engine

from rec_sys import rec_interface
from chatbot.prompts import film_chat_explore_chain, explain_rec_chain, glean_feed_back_chain

# Initialize database connection
engine = create_engine("sqlite:///sqlite.db")

# Sample user profile (for testing UI)
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

# Define LLM model
MODEL_NAME = "llama3.2:3b-instruct-q4_K_M"
llm = ChatOllama(model=MODEL_NAME)


class GradioFilmRec:
    def __init__(self):
        self.current_recommendations, self.scores = None, None
        self.semantic_full_response = None


        # self.session_id = randint(1000000, 9999999)  # Random unique session ID
        self.session_id = 999999

        self.model_name = MODEL_NAME
        print(f"Using model: {self.model_name}")
        self.rec_interface = rec_interface.RecInterface(user_id=self.session_id)

    def get_message_history(self):
        """Retrieve or initialise message history for this session."""
        return SQLChatMessageHistory(session_id=str(self.session_id), connection=engine)

    def update_chain(self, chain):
        """Wraps a LangChain pipeline with message history tracking."""
        chain = chain | llm.bind(stop=["<|eot_id|>"]) | StrOutputParser()
        return RunnableWithMessageHistory(
            chain,
            lambda _: self.get_message_history(),
            input_messages_key="question",
            output_messages_key="output",
            history_messages_key="history"
        )

    def recommend_films(self):
        """Generates film recommendations based on user history."""
        print("Generating recommendations...")
        recs = self.rec_interface.recommend(num_recommendations=5)
        scores = self.rec_interface.score_breakdown(recs)
        # return recommendations, self.score_explainer(recommendations)
        self.current_recommendations = recs
        self.scores = scores
        return recs

    def custom_chatbot(self, input_value, history, session_id):
        """Handles user queries with persistent chat history."""
        self.session_id = session_id
        chat_chain = self.update_chain(film_chat_explore_chain)
        response = chat_chain.stream(
            {"ability": "everything", "question": input_value},
            config={"configurable": {"session_id": str(self.session_id)}},
        )

        full_response = ""
        for item in response:
            full_response += item
            yield full_response, self.current_recommendations, self.semantic_full_response  # Add None for additional outputs

        # Semantic scraping and returning results
        # feedback_chain = self.update_chain(glean_feed_back_chain)
        feedback_chain =  glean_feed_back_chain | llm.bind(stop=["<|eot_id|>"]) | StrOutputParser()
        feedback_chain = RunnableWithMessageHistory(
            feedback_chain,
            lambda _: self.get_message_history(),
            input_messages_key="question",
            output_messages_key="output",
            history_messages_key="history"
        )

        semantic_response = feedback_chain.stream(
            {"input_text": input_value},
            config={"configurable": {"session_id": str(self.session_id)}},
        )

        self.semantic_full_response = ''.join(semantic_response)
        yield full_response, self.current_recommendations, self.semantic_full_response

    def score_explainer(self, breakdown):
        """Explains why a recommendation was made."""
        explain_chain = self.update_chain(explain_rec_chain)
        response = explain_chain.stream(
            {"ability": "everything", "score": breakdown},
            config={"configurable": {"session_id": str(self.session_id)}},
        )
        full_response = ''
        for item in response:
            full_response += item
            # yield full_response
        # print("Full response: ", full_response)
        yield full_response
        # return ''.join(response)

    def semantics_scraper(self, input_value, history):
        """Extracts sentiment & features from user messages."""
        feedback_chain = self.update_chain(glean_feed_back_chain)
        response = feedback_chain.stream(
            {"input_text": input_value},  # Remove 'ability' and 'question'
            config={"configurable": {"session_id": str(self.session_id)}},
        )
        print("Semantic Analysis:", ''.join(response))


# Instantiate the chatbot system
film_rec_bot = GradioFilmRec()

# --- 🎨 Gradio UI ---
with gr.Blocks(theme="Soft") as demo:
    gr.Markdown("# 🎬 Film Recommendation Chatbot - Llama 3.2")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Session ID")
            session_id_num = gr.Number(
                # value=film_rec_bot.session_id,
                value=999999,
                label="Session ID",
                interactive=True,
                info="Unique session ID to maintain chat history",
                minimum=999998,
                maximum=10000000,
                step=1
            )
            gr.Markdown("### Generate Recommendations")
            generate_recommendations = gr.Button(value="Generate Recommendations")

        # with gr.Column(scale=2):
        #     gr.Markdown("### User Profile")
        #     user_profile = gr.JSON(value=[sample_json], label="User Profile")

        with gr.Column(scale=2):
            gr.Markdown("### Recommendations")
            recommendations = gr.JSON(label="recommendations")

        with gr.Column(scale=2):
            gr.Markdown("### Message Semantics")
            message_semantics = gr.Text(label="Extracted Insights")

    # Chat Interface
    chatbot_interface = gr.ChatInterface(
        fn=film_rec_bot.custom_chatbot,
        chatbot=gr.Chatbot(type="messages", show_copy_button=True),
        editable=True,
        type="messages",
        additional_inputs=[session_id_num],
        additional_outputs=[recommendations, message_semantics],
        examples=[
            ["I love the Barbie film, I'm just Ken in a Barbie world"],
            ["I really like Eddie Murphy as Donkey in Shrek, really, it's my favourite film"],
            ["I think the director is really important in making or breaking a film"],
            ["I liked the cast in the last film I saw, but they the casting didn't make the film for me"],
        ]
    )

    # Link button to recommendation function
    generate_recommendations.click(
        fn=film_rec_bot.recommend_films,
        inputs=[],
        outputs=[recommendations]
    )

# Launch app
demo.launch()
