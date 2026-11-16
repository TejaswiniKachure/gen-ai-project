
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import chromadb


# Project paths
BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "sample.pdf"
DB_PATH = BASE_DIR / "Chroma_db"

COLLECTION_NAME = "rag_documents"


def ingest_documents():

    # 1. Check PDF file
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print("Loading PDF...")

    # 2. Load PDF
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Total pages loaded: {len(documents)}")

    # 3. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        # maximum target chunk length in characters
        chunk_overlap=100
        # Reapeated text between neighboring chunks

    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No text found in PDF.")

    # 4. Generate embeddings using Ollama
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    # ollama embeddings-- convert chunks into numerical vectors


    # 5. Remove old collection if it exists
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    existing = [
        collection.name
        for collection in client.list_collections()
    ]

    if COLLECTION_NAME in existing:
        client.delete_collection(
            name=COLLECTION_NAME
        )

    # 6. Store chunks and embeddings in ChromaDB

    #  chroma.from_documents()--- creates and store vector records

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH)
    )

    # persist_directory -- saves the database locally 

    print("Embeddings generated successfully!")
    print("Documents stored in ChromaDB.")

    print(f"Database location: {DB_PATH}")


if __name__ == "__main__":
    ingest_documents()

