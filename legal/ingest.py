import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil
from datetime import datetime

load_dotenv()

# 1. Load your JSON dataset
with open("data/mock_legal_rag_dataset_large.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Convert each record into a LangChain Document based on its category
documents = []
for record in data["documents"]:
    category = record.get("category", "unknown")

    if category in ("contract_template", "policy"):
        title = record.get("title", "")
        content = record.get("content", "")
        text = f"{title}\n\n{content}"

    elif category == "contract_review":
        clause = record.get("clause", "")
        risk = record.get("risk", "")
        content = record.get("content", "")
        recommendation = record.get("recommendation", "")
        text = (
            f"Clause: {clause}\n"
            f"Risk level: {risk}\n"
            f"Details: {content}\n"
            f"Recommendation: {recommendation}"
        )

    elif category == "faq":
        question = record.get("question", "")
        answer = record.get("answer", "")
        text = f"Q: {question}\nA: {answer}"

    elif category == "compliance":
        task = record.get("task", "")
        frequency = record.get("frequency", "")
        description = record.get("description", "")
        text = (
            f"Compliance task: {task}\n"
            f"Frequency: {frequency}\n"
            f"Description: {description}"
        )

    else:
        # fallback: just dump whatever fields exist
        text = " | ".join(f"{k}: {v}" for k, v in record.items() if k != "id")

    doc = Document(
        page_content=text,
        metadata={
            "id": record.get("id", ""),
            "category": category
        }
    )
    documents.append(doc)

print(f"Loaded {len(documents)} documents")

import time
import traceback
import re
import math

# 3. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 4. Embed and store in vector DB — in small batches to respect free tier rate limits
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = None
# number of items to send per embed request
batch_size = 10  # stay safely under the 100/min free tier limit

# configurable sleep between batches (seconds). Set SLEEP_SECONDS env var to override.
env_sleep = os.environ.get("SLEEP_SECONDS")
sleep_seconds = int(env_sleep) if env_sleep and env_sleep.isdigit() else None
# ensure sleep is at least large enough to keep requests/min <= 100 for free tier
min_sleep = math.ceil(60 * batch_size / 100) + 1
if sleep_seconds is None:
    sleep_seconds = min_sleep
else:
    sleep_seconds = max(sleep_seconds, min_sleep)

# ensure existing Chroma DB won't cause rust-binding panics — backup if present
persist_dir = "chroma_db"
if os.path.isdir(persist_dir) and os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_dir = f"{persist_dir}_backup_{timestamp}"
    print(f"Existing Chroma DB found — moving {persist_dir} -> {backup_dir}")
    shutil.move(persist_dir, backup_dir)

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    batch_no = i // batch_size + 1
    total_batches = (len(chunks) - 1) // batch_size + 1
    print(f"Embedding batch {batch_no} of {total_batches}...")

    # try several retries for quota / transient errors
    max_retries = 5
    attempt = 0
    while True:
        attempt += 1
        try:
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=persist_dir
                )
            else:
                vectorstore.add_documents(batch)
            break
        except Exception as e:
            print(f"Error while writing to Chroma (attempt {attempt}): {e}")
            traceback.print_exc()
            err_str = str(e).lower()
            # handle API quota / rate-limit errors by sleeping the suggested retry window
            m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", err_str)
            if m:
                wait = float(m.group(1)) + 1.0
                print(f"API advises retry after {wait}s — sleeping and retrying")
                time.sleep(wait)
                if attempt >= max_retries:
                    raise
                continue

            # handle chroma rust panics by backing up DB and retrying once
            if "panic" in err_str or "pyo3_runtime.panicexception" in err_str or "range start index" in err_str:
                if os.path.isdir(persist_dir):
                    ts = datetime.now().strftime("%Y%m%d%H%M%S")
                    bdir = f"{persist_dir}_backup_error_{ts}"
                    print(f"Detected Chroma panic — backing up {persist_dir} -> {bdir} and retrying")
                    shutil.move(persist_dir, bdir)
                    vectorstore = None
                    if attempt >= max_retries:
                        raise
                    continue

            # for other errors, don't retry
            raise

    try:
        time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("Interrupted by user — persisting vectorstore and exiting")
        try:
            if vectorstore is not None and hasattr(vectorstore, "persist"):
                vectorstore.persist()
        except Exception:
            pass
        raise

if vectorstore is not None and hasattr(vectorstore, "persist"):
    try:
        vectorstore.persist()
    except Exception as e:
        print("Warning: error while persisting vectorstore:", e)

print(f"Done! Vector store saved to ./{persist_dir}")