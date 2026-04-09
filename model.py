
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# ------------------------
# Load model and tokenizer
# ------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)

# ------------------------
# Generate response
# ------------------------
def generate_response(prompt, max_tokens=150):
    # 1. Format the prompt
    formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    # 2. Encode the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1] # Measure the length of your prompt

    # 3. Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 4. Slice the output to only include the NEW tokens
    # outputs[0] contains [prompt tokens + generated tokens]
    # We only want [generated tokens], which start at index 'input_length'
    new_tokens = outputs[0][input_length:]
    
    # 5. Decode only those new tokens
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return response

# ------------------------
# Flask routes
# ------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    
    if not user_input:
        return jsonify({"response": "Please enter a message."})
    
    try:
        response = generate_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
