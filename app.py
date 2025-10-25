import os, requests
from flask import Flask, request, jsonify

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in Render dashboard
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@app.route("/ask", methods=["POST"])
def ask():
    q = request.json.get("question", "")
    if not q:
        return jsonify({"error": "No question provided"}), 400

    # Call Hugging Face hosted model
    r = requests.post(
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": q},
        timeout=30
    )

    if r.status_code != 200:
        return jsonify({"error": r.text}), r.status_code

    data = r.json()
    return jsonify({"embedding_preview": data[:3]})  # simple test

@app.route("/")
def home():
    return "Flask proxy for Hugging Face inference"

@app.route("/ask", methods=["POST"])
def ask():
    try:
        q = request.json.get("question", "")
        if not q:
            return jsonify({"error": "No question provided"}), 400
        # ... your embedding / search / inference code here ...
        return jsonify({"answer": "Response goes here"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
