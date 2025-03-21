from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr


checkpoint = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"
device = "cpu"  # "cuda" or "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

def predict(message, history):
    history.append({"role": "user", "content": message})
    input_text = tokenizer.apply_chat_template(history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        temperature=0.7,  # Increase temperature for more diverse responses
        top_p=0.9,       # Use nucleus sampling
        repetition_penalty=1.2,  # Penalize repetition
        do_sample=True
    )
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    return response

demo = gr.ChatInterface(predict, type="messages")

demo.launch()
