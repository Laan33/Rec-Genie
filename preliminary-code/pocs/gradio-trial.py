import sys
from ollama import Client
import gradio as gr

host_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434/"

client = Client(host=host_url)

model_list = client.list()
model_names = [model['model'] for model in model_list['models']]

def chat_ollama(user_input, history, Model):
    stream = client.chat(
        model=Model,
        messages=[
                {
                    'role': 'user',
                    'content': user_input
                },
            ],
        stream=True,
    )

    partial_message = ""
    for chunk in stream:
        if len(chunk['message']['content']) != 0:
            partial_message = partial_message + chunk['message']['content']
            yield partial_message

with gr.Blocks(title="Ollama Chat", fill_height=True) as demo:
    gr.Markdown("# Ollama Chat")
    model_list = gr.Dropdown(model_names, value="llama3.2:latest", label="Model", info="Model to chat with")
    gr.ChatInterface(chat_ollama, additional_inputs=model_list)

if __name__ == "__main__":
    demo.launch()