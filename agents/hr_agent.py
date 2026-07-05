"""
HR Agent — wraps the bias-aware hiring pipeline (reweight + CatBoost shortlist)
as functions that a LangGraph node can call.

Pipeline:
  1. reweight.py  — applies fairness reweighting to remove gender bias from a dataset
  2. catshortlist.py — trains CatBoost on the reweighted data and shortlists candidates
  3. audit_report.py — generates an HTML fairness audit report
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "HR"))


def run_hr_task(task: str, payload: dict) -> dict:
    """
    Single dispatch entrypoint for the LangGraph HR node.

    task: one of "reweight" | "shortlist"
    payload: task-specific arguments
    """
    if task == "reweight":
        return {"status": "reweight pipeline ready", "script": "HR/reweight.py", "usage": "python HR/reweight.py"}
    if task == "shortlist":
        return {"status": "shortlist pipeline ready", "script": "HR/catshortlist.py", "usage": "python HR/catshortlist.py"}
    return {"error": f"unknown HR task: {task}"}


if __name__ == "__main__":
    print(json.dumps({"note": "HR agent ready — reweight + catshortlist pipeline"}, indent=2))
