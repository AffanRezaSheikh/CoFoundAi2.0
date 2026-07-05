"""
LangGraph orchestrator that merges CoFoundAI (HR pipeline) and Legal-AI-agent
(legal RAG assistant) into one graph.

Flow:

    START -> router -> hr_node   -> END
                     -> legal_node -> END
                     -> both_node  -> END   (rare: query touches both domains)

`router` is a small Gemini call that classifies the incoming user request as
"hr", "legal", or "both". Each downstream node calls straight into the
original CoFoundAI / Legal-AI-agent code via agents/hr_agent.py and
agents/legal_agent.py — nothing about the underlying logic was rewritten.
"""
import json
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
import google.generativeai as genai

from agents.hr_agent import run_hr_task
from agents.legal_agent import ask_legal_question

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

_ROUTER_MODEL = os.getenv("GEMINI_ROUTER_MODEL", "gemini-flash-latest")


class GraphState(TypedDict, total=False):
    user_input: str
    route: Literal["hr", "legal", "both"]
    hr_payload: dict          # optional structured args for HR tasks (task, payload)
    hr_result: dict
    legal_result: dict
    final_answer: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: GraphState) -> GraphState:
    """Classify the request: hr / legal / both, using Gemini."""
    text = state["user_input"]
    model = genai.GenerativeModel(_ROUTER_MODEL)
    prompt = (
        "Classify this request into exactly one label: hr, legal, or both.\n"
        "hr = hiring/recruiting/CV/interview/bias-audit questions.\n"
        "legal = contracts, compliance, policy, legal-document questions.\n"
        "both = the request needs both domains (e.g. 'review this candidate's "
        "employment contract for legal risk AND assess their CV fit').\n"
        f"Request: {text}\n"
        "Respond with only the single label word."
    )
    try:
        resp = model.generate_content(prompt)
        label = resp.text.strip().lower()
    except Exception:
        label = "legal"  # safe default if the router call fails
    if label not in ("hr", "legal", "both"):
        label = "legal"
    return {"route": label}


def hr_node(state: GraphState) -> GraphState:
    payload = state.get("hr_payload") or {"task": "shortlist", "payload": {}}
    result = run_hr_task(payload["task"], payload.get("payload", {}))
    return {"hr_result": result}


def legal_node(state: GraphState) -> GraphState:
    result = ask_legal_question(state["user_input"])
    return {"legal_result": result}


def both_node(state: GraphState) -> GraphState:
    hr_state = hr_node(state)
    legal_state = legal_node(state)
    return {**hr_state, **legal_state}


def finalize_node(state: GraphState) -> GraphState:
    parts = []
    if "hr_result" in state:
        parts.append(f"[HR agent]\n{json.dumps(state['hr_result'], indent=2, default=str)}")
    if "legal_result" in state:
        parts.append(f"[Legal agent]\n{state['legal_result']['answer']}")
    return {"final_answer": "\n\n".join(parts)}


def _route_choice(state: GraphState) -> str:
    return state["route"]


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("hr", hr_node)
    graph.add_node("legal", legal_node)
    graph.add_node("both", both_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", _route_choice, {
        "hr": "hr",
        "legal": "legal",
        "both": "both",
    })
    graph.add_edge("hr", "finalize")
    graph.add_edge("legal", "finalize")
    graph.add_edge("both", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    out = app.invoke({"user_input": "Can I add a non-compete clause to a contractor agreement?"})
    print(out["final_answer"])
