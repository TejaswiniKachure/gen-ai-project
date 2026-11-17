
from pathlib import Path
import os
import chromadb

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

PDF_PATH = BASE_DIR / "data" / "sample.pdf"
DB_PATH = BASE_DIR / "Chroma_db"

COLLECTION_NAME = "rag_documents"


def ingest_documents():

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    print("Loading PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Total pages loaded: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No text found in PDF.")

    print("Generating Gemini embeddings...")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key
    )

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    existing = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing:
        client.delete_collection(name=COLLECTION_NAME)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH)
    )

    print("Cloud vector database created successfully!")
    print(f"Database location: {DB_PATH}")


if __name__ == "__main__":
    ingest_documents()
