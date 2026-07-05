import json
import os

import google.generativeai as genai
from dotenv import load_dotenv

from finance.tools.finance_tools import FINANCE_TOOLS

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

TOOL_MAP = {tool["name"]: tool["function"] for tool in FINANCE_TOOLS}

TOOL_DECLARATIONS = [
    genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=genai.protos.Schema(type=genai.protos.Type.OBJECT, properties={}),
            )
            for tool in FINANCE_TOOLS
        ]
    )
]

SYSTEM_PROMPT = (
    "You are a Finance Agent for a startup. "
    "You help the founder understand the company's financial health. "
    "Use the available tools to answer questions about cash balance, burn rate, runway, expenses, and profits. "
    "Always be concise and data-driven."
)


def run_agent(user_query: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash", tools=TOOL_DECLARATIONS, system_instruction=SYSTEM_PROMPT)
    chat = model.start_chat()
    response = chat.send_message(user_query)

    while response.candidates[0].content.parts:
        function_call = None
        for part in response.candidates[0].content.parts:
            if part.function_call:
                function_call = part.function_call
                break

        if not function_call:
            break

        tool_name = function_call.name
        if tool_name not in TOOL_MAP:
            break

        result = TOOL_MAP[tool_name]()
        response = chat.send_message(
            genai.protos.Content(
                parts=[
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": json.loads(json.dumps(result, default=str))},
                        )
                    )
                ]
            )
        )

    return response.candidates[0].content.parts[0].text


if __name__ == "__main__":
    print("Finance Agent Ready. Type 'quit' to exit.\n")
    while True:
        query = input("Founder: ")
        if query.lower() in ("quit", "exit"):
            break
        answer = run_agent(query)
        print(f"\nAgent: {answer}\n")
