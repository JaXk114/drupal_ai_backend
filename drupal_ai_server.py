from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, GPT2TokenizerFast

app = Flask(__name__)

# ----  SETUP ON STARTUP ----
PDF_PATH = "therman-44-foa-report-wf538235.pdf"
print("Loading and indexing PDF ...")

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
text = "\n".join([p.page_content for p in pages])

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def count_tokens(t): return len(tokenizer.encode(t))

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=24, length_function=count_tokens)
chunks = splitter.create_documents([text])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)

hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=hf_pipeline)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# ----  API ENDPOINT ----
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    q = data.get("question", "")
    if not q:
        return jsonify({"error": "No question provided"}), 400
    try:
        a = qa.run(q)
        return jsonify({"answer": a})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
