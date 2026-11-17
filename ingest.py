# ingest.py

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# STEP 1: Project paths

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "book.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "rag_documents_v2"


# STEP 2: Load PDF

if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

loader = PyPDFLoader(str(PDF_PATH))
documents = loader.load()

print("Total pages:", len(documents))

# Check extracted text
for i, doc in enumerate(documents[:2]):
    print(f"\nPage {i + 1}:")
    print(doc.page_content[:500])


# STEP 3: Remove empty pages

documents = [
    doc for doc in documents
    if doc.page_content.strip()
]

print("\nNon-empty pages:", len(documents))


# STEP 4: Split into chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print("\nTotal chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content[:300])
    print("Metadata:", chunk.metadata)


# STEP 5: Load embedding model

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# STEP 6: Rebuild the ChromaDB collection

# This removes the previous version of this collection.
# It does not delete other ChromaDB collections.

old_db = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)

old_db.delete_collection()

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_PATH),
    collection_name=COLLECTION_NAME
)

print("\nPDF indexed successfully!")
print("Total stored chunks:", vector_store._collection.count())
print("Database:", CHROMA_PATH)