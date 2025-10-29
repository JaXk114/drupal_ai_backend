import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Debug print to confirm app starts
print("🚀 Flask app initializing...")

# Load environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@app.route("/ask", methods=["POST"])
def ask():
    print("⚙️ /ask triggered successfully")
    data = request.get_json(force=True)
    print("📦 Request data:", data)
    return jsonify({"ok": True, "question": data.get("question", None)})



@app.route("/")
def home():
    print("🏠 Root route hit")
    return "Flask proxy for Hugging Face inference"


@app.route("/debug")
def debug_routes():
    routes = [str(r) for r in app.url_map.iter_rules()]
    print(f"🧭 Registered routes: {routes}")
    return jsonify({"routes": routes})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
