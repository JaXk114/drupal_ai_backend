import os, requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Debug print to confirm app starts
print("🚀 Flask app initializing...")

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@app.route("/ask", methods=["POST"])
def ask():
    print("⚙️ /ask endpoint called")
    q = request.json.get("question", "")
    if not q:
        return jsonify({"error": "No question provided"}), 400

    if not HF_TOKEN:
        return jsonify({"error": "Missing HF_TOKEN in environment"}), 500

    try:
        print(f"🔍 Sending to Hugging Face: model={MODEL}, token={'SET' if HF_TOKEN else 'MISSING'}")

        r = requests.post(
    f"https://api-inference.huggingface.co/models/{MODEL}",
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
    json={"inputs": q},
    timeout=30
)

        )

        if r.status_code != 200:
            print(f"❌ HF error: {r.status_code} - {r.text}")
            return jsonify({"error": r.text}), r.status_code

        data = r.json()
        print("✅ HF response received successfully")
        return jsonify({"embedding_preview": data[:3]})
    except Exception as e:
        print(f"🔥 Exception in /ask: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    print("🏠 Root route hit")
    return "Flask proxy for Hugging Face inference"
    
@app.route("/debug")
def debug_routes():
    routes = [str(r) for r in app.url_map.iter_rules()]
    return jsonify({"routes": routes})

@app.route("/debug")
def debug_routes():
    routes = [str(r) for r in app.url_map.iter_rules()]
    return jsonify({"routes": routes})
