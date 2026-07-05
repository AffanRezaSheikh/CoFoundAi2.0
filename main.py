"""
CLI entrypoint for the merged HR + Legal agent orchestrator.

Usage:
    python main.py "What's the risk in an at-will termination clause?"
    python main.py --hr-task shortlist
    python main.py --hr-task reweight
"""
import argparse
import json

from graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="Ask the merged HR + Legal agent graph.")
    parser.add_argument("query", nargs="?", default=None, help="Natural-language question")
    parser.add_argument("--hr-task", choices=["reweight", "shortlist"],
                         help="Force a specific HR task instead of routing")
    args = parser.parse_args()

    app = build_graph()

    state = {"user_input": args.query or ""}
    if args.hr_task:
        state["route"] = "hr"
        state["hr_payload"] = {"task": args.hr_task, "payload": {}}

    result = app.invoke(state)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
