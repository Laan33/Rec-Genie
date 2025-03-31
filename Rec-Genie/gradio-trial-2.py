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

# Define LLM model
MODEL_NAME = "llama3.2:3b-instruct-q4_K_M"
llm = ChatOllama(model=MODEL_NAME)

def json_to_markdown(recommendations_list):
    """Converts a JSON object to a markdown list."""
    convert_json_to_markdown = "\n".join([
        f"{i + 1}. **{rec['title']}** (Released: {int(rec['release_date'])}) - Score: {rec['score']:.2f}"
        for i, rec in enumerate(recommendations_list)
    ])
    return convert_json_to_markdown

class GradioFilmRec:
    def __init__(self):
        self.current_recommendations = "No recommendations generated, click the button to the left"
        self.scores = None
        self.semantic_full_response = None

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
        scores, _ = self.rec_interface.score_breakdown(recs)

        # # Convert DataFrame to list of dictionaries for easy JSON rendering
        recommendations_list = scores.to_dict('records')
        convert_json_to_markdown = json_to_markdown(recommendations_list)
        # self.recs  = recs
        self.current_recommendations = convert_json_to_markdown
        self.scores = recommendations_list

        # Add on the line: "\n you can ask in chat for an explanation of these recommendations"
        convert_json_to_markdown = convert_json_to_markdown + "\n\n### Ask in the chat for an explanation of these recommendations!"

        print("\n")
        print("Recommendations: \n", convert_json_to_markdown)

        self.current_recommendations = convert_json_to_markdown

        return convert_json_to_markdown

    def router(self, input_value, history, session_id):
        """
        Routes the input to the appropriate function based on user query content.
        - If the user is asking for recommendations: call the recommend_films function
        - If the user is asking for an explanation: call the score_explainer function
        - If the user is giving feedback or chatting: call the custom_chatbot function
        - If none of the above: provide a default response
        """
        self.session_id = session_id
        print("Router session ID:", self.session_id)

        # # Check if the input is empty
        # if not input_value:
        #     return "Please enter a message."

        # Check if recommendations exist
        has_recommendations = self.scores is not None
        print("Has recommendations:", has_recommendations)

        # Create a routing LLM with context about existing recommendations
        router_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a router that determines what the user is asking for. "
             f"IMPORTANT: {'Recommendations HAVE already been generated and are available to explain.' if has_recommendations else 'No recommendations have been generated yet.'} "
             "Respond ONLY with one of these exact categories: "
             "- FEEDBACK: If the user is talking about films, providing feedback on their film tastes, or having a general conversation"
             "- EXPLAIN: ONLY, If the user is asking for an explanation of the existing recommendation scores"
             ),
            ("human", "{question}")
        ])

        # f"IMPORTANT: {'Recommendations HAVE already been generated and are available to explain.' if has_recommendations else 'No recommendations have been generated yet.'} "
        # "- RECOMMEND: If the user is asking for new film recommendations "
        # "- OTHER: If the query doesn't fit into any of the above categories"

        # Create the routing chain
        routing_chain = router_prompt | llm | StrOutputParser()

        # Timing the routing chain
        start_time = time.time()

        # Get the route category
        route_category = routing_chain.invoke({"question": input_value}).strip().upper()
        print(f"Router determined category: {route_category}")

        # Print the time taken for routing
        print("Routing time:", round((time.time() - start_time), 2), "seconds")

        # Route to the appropriate function based on the category
        # if "RECOMMEND" in route_category:
        #     print("Routing to recommend_films")
        #     return self.recommend_films()
        if "EXPLAIN" in route_category:
            # If we have current recommendations to explain
            if has_recommendations:
                for response in self.score_explainer(self.scores):
                    # Each yield becomes a return from the router function
                    yield response
            else:
                return "I don't have any recommendations to explain yet. Would you like me to recommend some films first?"
        elif "FEEDBACK" in route_category:
            for response in self.custom_chatbot(input_value, history, session_id):
                # Each yield becomes a return from the router function
                yield response
        else:
            print("Unknown category, returning default response")
            # Simple response, not a generator
            return "I'm not sure how to help with that. I can recommend films, explain recommendations, or chat about your film preferences."

    def custom_chatbot(self, input_value, history, session_id):
        """Handles user queries with persistent chat history."""
        self.session_id = session_id

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
        feedback_chain = self.update_chain(glean_feed_back_chain)

        semantic_response = feedback_chain.stream(
            {"input_text": input_value},
            config={"configurable": {"session_id": str(self.session_id)}},
        )

        self.semantic_full_response = ''.join(semantic_response)

        # Update the user profile weights and items
        # self.rec_interface.implement_user_feedback(self.session_id, self.semantic_full_response)

        yield full_response, self.current_recommendations, self.semantic_full_response

    def score_explainer(self, breakdown):
        """Explains why a recommendation was made."""
        print("Breakdown:", breakdown)

        # Update the chain with the correct prompt template
        explain_chain = self.update_chain(explain_rec_chain)

        # Process the first recommendation in the list as an example
        # You might want to expand this to explain all recommendations
        if breakdown and len(breakdown) > 0:
            item = breakdown[0]

            # Extract the relevant fields from the breakdown
            # Adapt these field names to match your actual data structure
            explanation_input = {
                "title": item.get('title', 'Unknown film'),
                "release_date": item.get('year', 'Unknown year'),
                "score": item.get('score', 0),
                "cast_score": item.get('actor_score', 0),
                "director_score": item.get('director_score', 0),
                "genre_score": item.get('genre_score', 0),
                "user_user_score": item.get('collab_score', 0)
            }

            # Get the explanation
            response = explain_chain.stream(
                explanation_input,
                config={"configurable": {"session_id": str(self.session_id)}},
            )
            full_response = ""
            for item in response:
                full_response += item
                yield full_response, self.current_recommendations, self.semantic_full_response

            # Return the expected values
            yield response, self.current_recommendations, self.semantic_full_response
        else:
            return "No recommendations to explain.", self.current_recommendations, self.semantic_full_response

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

        with gr.Column(scale=2):
            gr.Markdown("### Recommendations")
            # recommendations = gr.JSON(label="recommendations")
            recommendations = gr.Markdown(label="recommendations", value=film_rec_bot.current_recommendations)

        with gr.Column(scale=2):
            gr.Markdown("### Message Semantics")
            message_semantics = gr.Text(label="Extracted Message Semantics", placeholder="No message semantics yet")

    # Chat Interface
    chatbot_interface = gr.ChatInterface(
        fn=film_rec_bot.router,
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



# TODO - add in 3 sample profiles. E.g. One for a kid (Disney), one for someone into action films, and one for someone into romcoms.

