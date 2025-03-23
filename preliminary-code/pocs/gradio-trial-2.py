from random import randint
import gradio as gr
# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.chat_message_histories import SQLChatMessageHistory

model_name = "llama3.2:latest"

llm = ChatOllama(model=model_name)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You're an assistant who's good at talking to users to find out about their film interests and what matters to them in a film"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm.bind(stop=["<|eot_id|>"]) | StrOutputParser()

with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),
    input_messages_key="question",
    output_messages_key="output",
    history_messages_key="history"
)

def chatbot(input_value, history, session_id):
    response = with_message_history.stream(
            {"ability": "everything", "question": input_value},
            config={"configurable": {"session_id": session_id}},
            )
    full_response = ''
    for item in response:
        full_response += item
        yield full_response
    yield full_response

session_id_num = gr.Number(
    value=randint(200, 1000),
    label="Input",
    interactive=True,
    info="Session ID to use for chat history",
    minimum=1,
    maximum=1000000,
    step=1
)

state = gr.State(value=session_id_num.value)

def update_state(session_id):
    state.value = session_id
    return state.value

iface = gr.ChatInterface(fn=chatbot,
                         title="🦙💬 Chatbot using Llama3 via Ollama",
                         additional_inputs=[session_id_num],
                         examples=[["I love the Barbie film"], ["I think the director is really important"]]
                         )

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Session ID")
            session_id_num.render()
            session_id_num.change(fn=update_state, inputs=session_id_num, outputs=state)

        with gr.Column(scale=3):
            gr.Markdown("Chatbot")
            iface.render()

demo.launch()