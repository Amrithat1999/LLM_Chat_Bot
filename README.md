🤖 TinyLlama Flask Chatbot
A simple, lightweight web application that allows users to chat with the TinyLlama-1.1B large language model in real-time. The project uses Flask for the backend and Hugging Face Transformers to run the model locally.

🚀 Features
Local Inference: No API keys required; the model runs entirely on your hardware.

Smart Formatting: Uses the ChatML-style prompting specifically for TinyLlama.

GPU Accelerated: Automatically detects and uses NVIDIA GPUs (via CUDA) for faster responses.

Responsive API: A JSON-based chat endpoint for easy integration.

🛠️ Prerequisites
Before running the project, ensure you have the following installed:

Python 3.8+

PyTorch (With CUDA support if you have an NVIDIA GPU)

Pip (Python package manager)

pip install flask torch transformers accelerate

Python
python -c "import torch; print(torch.cuda.is_available())"
🚦 How to Run
Start the Flask server:

python model.py
Access the Chatbot:
Open your browser and navigate to:
http://127.0.0.1:5000

Wait for Model Loading:
On the first run, the script will download approximately 2.2GB of model weights from Hugging Face. Subsequent runs will be nearly instant.

📂 Project Structure
Plaintext
├── app.py              # Main Flask application and LLM logic
├── templates/
│   └── index.html      # The frontend chat interface
└── README.md           # Project documentation
⚙️ Technical Details
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0

Prompt Template:

Plaintext
<|system|>
You are a helpful assistant.</s>
<|user|>
{prompt}</s>
<|assistant|>
Hyperparameters:

Temperature: 0.7 (Balanced between creative and focused)

Max New Tokens: 150

Sampling: Enabled (do_sample=True)

⚠️ Notes
Hardware: Running this model on a CPU is possible but will be significantly slower than running on a GPU.

Knowledge Cutoff: TinyLlama's knowledge is based on its training data; it may not be aware of very recent events from 2025 or 2026.
