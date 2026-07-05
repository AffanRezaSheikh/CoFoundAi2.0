"""
CoFoundAI — Startup OS web backend.

Serves a single-page app with three agents:
  - Finance : preset queries + Gemini function-calling chat
  - HR      : bias-aware CatBoost shortlist over a chosen dataset
  - Legal   : systematic search + Gemini RAG over ChromaDB
"""
import os
import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from finance.services import finance_service
from HR import hr_pipeline
from legal import legal_summary

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
HAS_LLM = bool(GEMINI_KEY)

app = Flask(__name__)

HR_DIR = os.path.join(os.path.dirname(__file__), "HR")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Finance agent — preset options mapped to service functions
# ---------------------------------------------------------------------------
FINANCE_OPTIONS = {
    "summary": {
        "label": "Full financial overview",
        "fn": finance_service.generate_finance_summary,
    },
    "cash_balance": {
        "label": "Current cash balance",
        "fn": lambda: {"cash_balance": finance_service.calculate_cash_balance()},
    },
    "burn_rate": {
        "label": "Monthly burn rate",
        "fn": lambda: {"monthly_burn_rate": round(finance_service.calculate_burn_rate(), 2)},
    },
    "runway": {
        "label": "Runway (months left)",
        "fn": lambda: {"runway_months": finance_service.calculate_runway()},
    },
    "monthly_profit": {
        "label": "Monthly profit & margin",
        "fn": finance_service.calculate_monthly_profit,
    },
    "top_expenses": {
        "label": "Top expense categories",
        "fn": finance_service.top_expense_categories,
    },
}


@app.route("/api/finance/options")
def finance_options():
    return jsonify([{"key": k, "label": v["label"]} for k, v in FINANCE_OPTIONS.items()])


@app.route("/api/finance/query", methods=["POST"])
def finance_query():
    key = (request.json or {}).get("option")
    opt = FINANCE_OPTIONS.get(key)
    if not opt:
        return jsonify({"error": f"Unknown finance option: {key}"}), 400
    try:
        return jsonify({"option": key, "label": opt["label"], "data": opt["fn"]()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Finance query failed: {e}"}), 500


# ---------------------------------------------------------------------------
# HR agent — dataset shortlist
# ---------------------------------------------------------------------------
@app.route("/api/hr/datasets")
def hr_datasets():
    out = []
    for fname, preset in hr_pipeline.DATASET_PRESETS.items():
        if os.path.exists(os.path.join(HR_DIR, fname)):
            out.append({"file": fname, **preset})
    return jsonify(out)


@app.route("/api/hr/shortlist", methods=["POST"])
def hr_shortlist():
    body = request.json or {}
    fname = body.get("dataset")
    preset = hr_pipeline.DATASET_PRESETS.get(fname)
    if not preset:
        return jsonify({"error": f"Unknown dataset: {fname}"}), 400

    csv_path = os.path.join(HR_DIR, fname)
    if not os.path.exists(csv_path):
        return jsonify({"error": f"Dataset file not found: {fname}"}), 404

    top_n = int(body.get("top_n", 50))
    try:
        result = hr_pipeline.run_shortlist(
            csv_path=csv_path,
            gender_col=preset["gender_col"],
            target_col=preset["target_col"],
            positive_val=preset["positive_val"],
            top_n=top_n,
        )
        if "error" in result:
            return jsonify(result), 400
        result["dataset_label"] = preset["label"]
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Shortlist failed: {e}"}), 500


# ---------------------------------------------------------------------------
# Legal agent — systematic knowledge-base output
# ---------------------------------------------------------------------------
@app.route("/api/legal/overview")
def legal_overview():
    try:
        return jsonify(legal_summary.overview())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Legal overview failed: {e}"}), 500


@app.route("/api/legal/query", methods=["POST"])
def legal_query():
    query = (request.json or {}).get("query", "").strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400
    try:
        return jsonify(legal_summary.search(query))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Legal query failed: {e}"}), 500


@app.route("/api/legal/category", methods=["POST"])
def legal_category():
    category = (request.json or {}).get("category", "").strip()
    if not category:
        return jsonify({"error": "Empty category"}), 400
    try:
        return jsonify(legal_summary.by_category(category))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Legal category failed: {e}"}), 500


# ---------------------------------------------------------------------------
# LLM chat endpoints (only active when a Gemini API key is provided)
# ---------------------------------------------------------------------------
@app.route("/api/llm/status")
def llm_status():
    return jsonify({"available": HAS_LLM})


@app.route("/api/finance/chat", methods=["POST"])
def finance_chat():
    if not HAS_LLM:
        return jsonify({"error": "No API key configured"}), 503
    query = (request.json or {}).get("message", "").strip()
    if not query:
        return jsonify({"error": "Empty message"}), 400
    try:
        from finance.agents.finance_agent import run_agent
        answer = run_agent(query)
        return jsonify({"answer": answer})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Finance chat failed: {e}"}), 500


@app.route("/api/legal/chat", methods=["POST"])
def legal_chat():
    if not HAS_LLM:
        return jsonify({"error": "No API key configured"}), 503
    query = (request.json or {}).get("message", "").strip()
    if not query:
        return jsonify({"error": "Empty message"}), 400
    try:
        answer, sources = _legal_rag(query)
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Legal chat failed: {e}"}), 500


_legal_retriever = None

def _legal_rag(query: str):
    """Simple RAG: retrieve from ChromaDB, send context + query to Gemini."""
    import google.generativeai as genai
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import Chroma

    global _legal_retriever
    if _legal_retriever is None:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GEMINI_KEY,
        )
        chroma_dir = os.path.join(os.path.dirname(__file__), "legal", "chroma_db")
        vectorstore = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        _legal_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    docs = _legal_retriever.invoke(query)

    context = "\n\n".join(
        f"[{d.metadata.get('category','?')} — {d.metadata.get('id','?')}] {d.page_content}"
        for d in docs
    )
    sources = [
        {"source": d.metadata.get("source"), "id": d.metadata.get("id"),
         "category": d.metadata.get("category")}
        for d in docs
    ]

    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (
        "You are a legal support assistant for a startup. "
        "Use ONLY the context below to answer the question. "
        "If the answer isn't in the context, say so. "
        "Always cite which document/section your answer comes from.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    resp = model.generate_content(prompt)
    return resp.text, sources


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
