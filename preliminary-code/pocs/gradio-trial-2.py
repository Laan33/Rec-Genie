from random import randint

import gradio as gr
from langchain_community.chat_models import ChatOllama
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

# accordian = gr.Accordion("Advanced Options", open=True)


session_id_num = gr.Number(
    value=randint(200, 1000),
    label="Input",
    interactive=True,
    info="Session ID to use for chat history",
    minimum=1,
    maximum=1000000,
    step=1)



iface = gr.ChatInterface(fn=chatbot,
                         title="🦙💬 Chatbot using Llama3 via Ollama",
                         # additional_inputs_accordion=[accordian],
                         # additional_inputs=[session_id_num],
                         examples=[["I love the Barbie film"], ["I think the director is really important"]]
                         )
# iface = gr.ChatInterface(fn=chatbot, title="🦙💬 Chatbot using Llama3 via Ollama", additional_inputs=[session_id_num])
# iface.launch(inbrowser=True)
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Session ID")
            session_id_num.render()

        with gr.Column(scale=3):
            gr.Markdown("Chatbot")
            iface.render()

demo.launch()

#
# import gradio as gr
#
# python_code = """
# def fib(n):
#     if n <= 0:
#         return 0
#     elif n == 1:
#         return 1
#     else:
#         return fib(n-1) + fib(n-2)
# """
#
# js_code = """
# function fib(n) {
#     if (n <= 0) return 0;
#     if (n === 1) return 1;
#     return fib(n - 1) + fib(n - 2);
# }
# """
#
# def chat(message, history):
#     if "python" in message.lower():
#         return "Type Python or JavaScript to see the code.", gr.Code(language="python", value=python_code)
#     elif "javascript" in message.lower():
#         return "Type Python or JavaScript to see the code.", gr.Code(language="javascript", value=js_code)
#     else:
#         return "Please ask about Python or JavaScript.", None
#
# with gr.Blocks() as demo:
#     code = gr.Code(render=False)
#     with gr.Row():
#         with gr.Column():
#             gr.Markdown("<center><h1>Write Python or JavaScript</h1></center>")
#             gr.ChatInterface(
#                 chat,
#                 examples=["Python", "JavaScript"],
#                 additional_outputs=[code],
#                 type="messages"
#             )
#         with gr.Column():
#             gr.Markdown("<center><h1>Code Artifacts</h1></center>")
#             code.render()
#
# demo.launch()