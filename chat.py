# chat.py

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# STEP 1: Load environment variables

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

# STEP 2: Load embeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# STEP 3: Load ChromaDB

CHROMA_PATH = BASE_DIR / "chroma_db"

if not CHROMA_PATH.exists():
    raise FileNotFoundError("ChromaDB not found. Run ingest.py first.")

vector_store = Chroma(
    collection_name="rag_documents",
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)

# STEP 4: Create retriever

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# STEP 5: Connect Groq LLM

llm = ChatGroq(
    model=model_name,
    groq_api_key=api_key,
    temperature=0
)

# STEP 6: Create prompt

prompt = ChatPromptTemplate.from_template("""
You are an AI document assistant.

Answer using only the provided context.

If the answer is unavailable,
say you don't know based on the document.

Context:
{context}

Question:
{question}

Answer:
""")

# STEP 7: Ask questions

while True:

    question = input("\nAsk your question: ")

    if question.lower() in ["exit", "quit"]:
        print("Chat ended!")
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    if not docs:
        print("\nNo relevant information found.")
        continue

    # Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Prepare prompt
    formatted_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    # Generate answer
    response = llm.invoke(formatted_prompt)

    print("\nAI Answer:")
    print(response.content)