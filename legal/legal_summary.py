"""
Systematic legal knowledge-base reader.

Reads the mock legal RAG dataset and produces structured, deterministic output
(overview + keyword search) without requiring an LLM or API key. If a Gemini key
is available the separate legal_agent.py RAG chain can be used for free-form Q&A.
"""
import json
import os
import re
from collections import Counter

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "mock_legal_rag_dataset_large.json")

_CACHE = None


def _load():
    global _CACHE
    if _CACHE is None:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _CACHE = json.load(f)["documents"]
    return _CACHE


def _doc_text(doc):
    """Flatten a document's searchable fields into one lowercase string."""
    parts = []
    for k, v in doc.items():
        if k in ("id", "category"):
            continue
        if isinstance(v, list):
            parts.append(" ".join(map(str, v)))
        else:
            parts.append(str(v))
    return " ".join(parts).lower()


def _doc_title(doc):
    cat = doc.get("category")
    if cat in ("contract_template", "policy"):
        return doc.get("title", doc["id"])
    if cat == "contract_review":
        return doc.get("clause", doc["id"])
    if cat == "faq":
        return doc.get("question", doc["id"])
    if cat == "compliance":
        return doc.get("task", doc["id"])
    return doc["id"]


def _doc_body(doc):
    cat = doc.get("category")
    if cat == "contract_review":
        return (f"Risk: {doc.get('risk', 'N/A')}. {doc.get('content', '')} "
                f"Recommendation: {doc.get('recommendation', '')}")
    if cat == "faq":
        return doc.get("answer", "")
    if cat == "compliance":
        return f"Frequency: {doc.get('frequency', 'N/A')}. {doc.get('description', '')}"
    return doc.get("content", "")


CATEGORY_LABELS = {
    "contract_template": "Contract Templates",
    "contract_review": "Contract Reviews",
    "faq": "Legal FAQs",
    "compliance": "Compliance Tasks",
    "policy": "Company Policies",
}


def overview():
    """Structured snapshot of the whole legal knowledge base."""
    docs = _load()
    cats = Counter(d.get("category") for d in docs)

    reviews = [d for d in docs if d.get("category") == "contract_review"]
    risk_dist = Counter(d.get("risk", "Unknown") for d in reviews)

    templates = [d.get("title") for d in docs if d.get("category") == "contract_template"]

    compliance = [
        {"task": d.get("task"), "frequency": d.get("frequency")}
        for d in docs if d.get("category") == "compliance"
    ]

    policies = [d.get("title") for d in docs if d.get("category") == "policy"]

    return {
        "total_documents": len(docs),
        "categories": [
            {"key": k, "label": CATEGORY_LABELS.get(k, k), "count": c}
            for k, c in cats.most_common()
        ],
        "risk_distribution": dict(risk_dist),
        "contract_templates": templates,
        "compliance_calendar": compliance[:15],
        "policies": policies[:15],
    }


def search(query, limit=6):
    """Deterministic keyword search across all documents, ranked by term overlap."""
    docs = _load()
    terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    if not terms:
        return {"query": query, "results": [], "message": "Please enter a more specific query."}

    scored = []
    for doc in docs:
        text = _doc_text(doc)
        score = sum(text.count(t) for t in terms)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:limit]:
        results.append({
            "id": doc["id"],
            "category": CATEGORY_LABELS.get(doc.get("category"), doc.get("category")),
            "title": _doc_title(doc),
            "body": _doc_body(doc),
            "relevance": score,
        })

    return {
        "query": query,
        "match_count": len(scored),
        "results": results,
        "message": None if results else "No matching legal documents found.",
    }


def by_category(category, limit=20):
    """Return documents in a given category, formatted for display."""
    docs = _load()
    items = [d for d in docs if d.get("category") == category][:limit]
    return {
        "category": CATEGORY_LABELS.get(category, category),
        "items": [
            {"id": d["id"], "title": _doc_title(d), "body": _doc_body(d)}
            for d in items
        ],
    }
