"""
Legal Agent — wraps Legal-AI-agent's Chroma + Gemini RAG chain as a single
callable function, so a LangGraph node can invoke it without the original
input()-loop from query.py.
"""
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

_CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "legal", "chroma_db")

_PROMPT = PromptTemplate(
    template="""You are a legal support assistant. Use ONLY the context below to answer the question.
If the answer isn't in the context, say "I don't have enough information in the provided documents to answer that."
Always cite which document/section your answer comes from.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

_qa_chain = None


def _get_chain() -> RetrievalQA:
    global _qa_chain
    if _qa_chain is None:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        vectorstore = Chroma(persist_directory=_CHROMA_DIR, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": _PROMPT},
            return_source_documents=True,
        )
    return _qa_chain


def ask_legal_question(question: str) -> dict:
    """Answer a legal question using the RAG index built from legal_ai_agent/data."""
    chain = _get_chain()
    result = chain.invoke({"query": question})
    sources = [
        {"source": d.metadata.get("source"), "id": d.metadata.get("id"), "category": d.metadata.get("category")}
        for d in result["source_documents"]
    ]
    return {"answer": result["result"], "sources": sources}


if __name__ == "__main__":
    print(ask_legal_question("What is a non-compete clause?"))
