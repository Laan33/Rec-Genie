import time

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
        self.basic_rec_list = None
        # self.recs = None
        self.current_recommendations, self.scores = None, None
        self.semantic_full_response = None
        self.score_explanation = None


        # self.session_id = randint(1000000, 9999999)  # Random unique session ID
        self.session_id = 999999

        self.model_name = MODEL_NAME
        print(f"Using model: {self.model_name}")

        # Timer to measure rec_interface initialisation time
        start_time = time.time()
        self.rec_interface = rec_interface.RecInterface(user_id=self.session_id)
        print("RecInterface initialised in", round((time.time() - start_time), 1), "seconds\n")

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
        scores, basic_rec_list = self.rec_interface.score_breakdown(recs)

        # Convert DataFrame to list of dictionaries for easy JSON rendering
        recommendations_list = scores.to_dict('records')
        basic_rec_list = basic_rec_list.to_dict('records')
        # self.recs  = recs
        self.current_recommendations = recommendations_list
        self.scores = scores
        self.score_explanation = None  # Reset score explanation
        print("Current recommendations type:", type(self.current_recommendations))
        print("Current recommendations:", self.current_recommendations)

        print("\n")

        convert_json_to_markdown = "\n".join([
            f"{i + 1}. **{rec['title']}** (Released: {rec['release_date']}) - Score: **{rec['score']:.2f}**"
            for i, rec in enumerate(recommendations_list)
        ])
        print("Recommendations: \n", convert_json_to_markdown)
        # self.basic_rec_list = basic_rec_list

        # Call custom_chatbot to output "Do you want a breakdown of the recommendation scores?"
        self.custom_chatbot("", None, self.session_id, scores=recommendations_list)

        return convert_json_to_markdown

    def custom_chatbot(self, input_value, history, session_id, **kwargs):
        """Handles user queries with persistent chat history."""
        self.session_id = session_id

        if kwargs.get("scores"):
            print("Prompting for score explanation...")
            # Output a prompt to the user to ask if they want a breakdown of the recommendation scores
            self.scores = kwargs["scores"]
            return (
                "Would you like to know about the recommendation scores?",
                self.current_recommendations,
                self.semantic_full_response
                # examples=[
                #     ["Yes, I would like to know about the recommendation scores"],
                #     ["No, I'm good"]
                # ]
            )

        # Check if user wants score explanation
        if input_value.lower() in ['yes', 'y', 'Yes, I would like to know about the recommendation scores']:
            print("User wants score explanation...")
            if self.scores:
                # Generate score explanation
                explain_generator = self.score_explainer(self.scores)
                self.score_explanation = next(explain_generator)
                return (
                    f"Here's a breakdown of the recommendation scores:\n\n{self.score_explanation}",
                    self.current_recommendations,
                    self.semantic_full_response
                )
            else:
                return (
                    "Sorry, there are no recent recommendations to explain.",
                    self.current_recommendations,
                    self.semantic_full_response
                )

        # Regular chat handling
        chat_chain = self.update_chain(film_chat_explore_chain)
        response = chat_chain.stream(
            {"ability": "everything", "question": input_value},
            config={"configurable": {"session_id": str(self.session_id)}},
        )

        full_response = ""
        for item in response:
            full_response += item
            yield full_response, self.current_recommendations, self.semantic_full_response

        # Semantic scraping
        feedback_chain = glean_feed_back_chain | llm.bind(stop=["<|eot_id|>"]) | StrOutputParser()
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
        return full_response

    def semantics_scraper(self, input_value, history):
        """Extracts sentiment & features from user messages."""
        feedback_chain = self.update_chain(glean_feed_back_chain)
        response = feedback_chain.stream(
            {"input_text": input_value},
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
            # recommendations = gr.JSON(label="recommendations")
            recommendations = gr.Markdown(label="recommendaitions", value="No recommendations generated, click the button to the left")

        with gr.Column(scale=2):
            gr.Markdown("### Message Semantics")
            message_semantics = gr.Text(label="Extracted Message Semantics", placeholder="No message semantics yet")

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