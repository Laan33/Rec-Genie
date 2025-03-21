import gradio as gr
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# Initialize the OllamaLLM model
llm = OllamaLLM(model="llama3.2:3b-instruct-q4_K_M")

# Define a chat template
chat_template = """<|im_start|>{role}\n{content}\n"""

# Define the predict function
def predict(message, history):
    if history is None:
        history = []
    history.append({"role": "user", "content": message})
    input_text = chat_template.format(role="user", content=message)

    # Debug: Print input text
    print(f"Input Text: {input_text}")

    # Use the OllamaLLM model to generate a response
    try:
        response = llm.invoke({"prompt": input_text})
        # Debug: Print response
        print(f"Response: {response}")
    except Exception as e:
        # Debug: Print error
        print(f"Error: {e}")
        response = "An error occurred while generating the response."

    history.append({"role": "assistant", "content": response})
    return response, history

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=["text", "state"],
    outputs=["text", "state"],
    live=True
)

iface.launch()