from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch
import os
import json

app = Flask(__name__)

# Load your embedding model (small + CPU friendly)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load text documents
data_folder = "."
documents = []
for file in os.listdir(data_folder):
    if file.endswith(".txt"):
        with open(os.path.join(data_folder, file), "r", encoding="utf-8", errors="ignore") as f:

            documents.append({"name": file, "text": f.read()})

# Precompute embeddings
corpus_embeddings = model.encode([d["text"] for d in documents], convert_to_tensor=True)

@app.route("/ask", methods=["POST"])
def ask():
    try:
        q = request.json.get("question", "")
        if not q:
            return jsonify({"error": "No question provided"}), 400

        # Get embedding for question
        query_emb = model.encode(q, convert_to_tensor=True)

        # Compute similarity
        scores = util.cos_sim(query_emb, corpus_embeddings)[0]
        best_idx = torch.argmax(scores).item()

        best_doc = documents[best_idx]
        context = best_doc["text"][:1500]  # clip long text

        # Construct response (you can later replace this with an LLM API call)
        return jsonify({
            "document": best_doc["name"],
            "answer": f"Most relevant section from {best_doc['name']}:\n\n{context}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Flask document chatbot running."

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
