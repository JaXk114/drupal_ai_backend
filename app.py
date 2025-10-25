import os, requests
from flask import Flask, request, jsonify

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model_loaded = False  # track whether model initialized

@app.route("/ask", methods=["POST"])
def ask():
    global model_loaded
    q = request.json.get("question", "")
    if not q:
        return jsonify({"error": "No question provided"}), 400

    # Lazy init: only connect to Hugging Face once the first query comes in
    if not model_loaded:
        print("Initializing model lazily...")
        model_loaded = True  # pretend loaded, real call happens below

    r = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": q},
        timeout=30
    )

    if r.status_code != 200:
        return jsonify({"error": r.text}), r.status_code

    data = r.json()
    return jsonify({"embedding_preview": data[:3]})

@app.route("/")
def home():
    return "Flask proxy for Hugging Face inference (lazy load)"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
