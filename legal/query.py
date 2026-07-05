import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

prompt_template = """You are a legal support assistant. Use ONLY the context below to answer the question.
If the answer isn't in the context, say "I don't have enough information in the provided documents to answer that."
Always cite which document/section your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

while True:
    question = input("\nAsk a legal question (or 'quit'): ")
    if question.lower() == "quit":
        break
    result = qa_chain.invoke({"query": question})
    print("\n--- Answer ---")
    print(result["result"])
    print("\n--- Sources ---")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source')} (page {doc.metadata.get('page')})")