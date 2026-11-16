
from pathlib import Path

from langchain_ollama import OllamaEmbeddings, ChatOllama
# ollamaEmbedding--- coverts into the numerical vector(embeddings).
# ChatOllama --- connects application to a locally ollama LLM.
# here we use the qwen 2.5 to generate answers.
from langchain_chroma import Chroma
# connects application to chromaDB
#  chromaDB store embeddings and documents chunks 




# define project paths 
# Project paths
BASE_DIR = Path(__file__).resolve().parent

# __file__ returns the location of the current py file 

# .resolve() -- return the absolute path 

# .parent -- it return the parent directory containing py file 

DB_PATH = BASE_DIR / "Chroma_db"

COLLECTION_NAME = "rag_documents"


# 1. Initialize embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 2. Connect to ChromaDB
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_PATH),
# DB_PATH is path object str() converts it into a normal string path 
    embedding_function=embeddings
)

# 3. Initialize Qwen LLM
llm = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
    # temp = 0 makes the ans more consistent and less random. 
)


def answer_question(question):

    # 4. Retrieve relevant chunks
    retrieved_docs = vectorstore.similarity_search(
        question,
        k=4
    )

    if not retrieved_docs:
        return "No relevant information found in the PDF.", []

    # 5. Prepare context
    context = ""

    sources = []

    for doc in retrieved_docs:

        context += doc.page_content + "\n\n"

        page = doc.metadata.get("page", "Unknown")

        if isinstance(page, int):
            page = page + 1

        source = {
            "page": page,
            "file": Path(
                doc.metadata.get("source", "Unknown")
            ).name
        }

        if source not in sources:
            sources.append(source)

    # 6. Create prompt
    prompt = f"""
    You are a helpful PDF question-answering assistant.

    Answer the question using ONLY the provided context.

    If the answer is not available in the context,
    clearly say that the information is not available
    in the provided PDF.

    Do not make up information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # 7. Generate answer using Qwen
    response = llm.invoke(prompt)

    return response.content, sources


if __name__ == "__main__":

    print("PDF RAG Chatbot")

    while True:

        question = input("\nAsk a question (type exit to stop): ")

        if question.lower() == "exit":
            break

        answer, sources = answer_question(question)

        print("\nAnswer:")
        print(answer)

        print("\nSources:")
        for source in sources:
            print(source)
