
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# STEP 1: STREAMLIT CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Chatbot")
st.write("Ask questions about your PDF documents.")



# ==========================================
# STEP 2: LOAD API KEYS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def get_secret(key):
    try:
        value = st.secrets.get(key)
        if value:
            return value
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass

    return os.getenv(key, "")


groq_api_key = get_secret("GROQ_API_KEY")
google_api_key = get_secret("GOOGLE_API_KEY")

model_name = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

if not groq_api_key:
    st.error("GROQ_API_KEY not found.")
    st.stop()

if not google_api_key:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()



# ==========================================
# STEP 3: LOAD GEMINI EMBEDDINGS
# ==========================================

@st.cache_resource
def load_embeddings(api_key):

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )


with st.spinner("Loading Gemini embeddings..."):
    embeddings = load_embeddings(google_api_key)


# ==========================================
# STEP 4: LOAD CHROMADB
# ==========================================

from ingest_cloud import ingest_documents

CHROMA_PATH = BASE_DIR / "Chroma_db"
COLLECTION_NAME = "rag_documents"

# Automatically create ChromaDB if it does not exist
if not CHROMA_PATH.exists() or not any(CHROMA_PATH.iterdir()):

    with st.spinner("Creating ChromaDB using Gemini embeddings..."):
        try:
            ingest_documents()
            st.success("ChromaDB created successfully!")

        except Exception as e:
            st.error(f"Database creation failed: {e}")
            st.stop()


@st.cache_resource
def load_vector_store(_embeddings):

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PATH),
        embedding_function=_embeddings
    )


vector_store = load_vector_store(embeddings)

document_count = vector_store._collection.count()

if document_count == 0:
    st.error("ChromaDB is empty. Run ingest_cloud.py first.")
    st.stop()

st.sidebar.success(
    f"PDF chunks available: {document_count}"
)


# ==========================================
# STEP 5: CREATE RETRIEVER
# ==========================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# ==========================================
# STEP 6: CONNECT GROQ LLM
# ==========================================

@st.cache_resource
def load_llm(api_key, model):

    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=0
    )


llm = load_llm(groq_api_key, model_name)


# ==========================================
# STEP 7: CREATE PROMPT
# ==========================================

prompt = ChatPromptTemplate.from_template("""
You are an AI assistant that answers questions
based on the provided PDF documents.

Context:
{context}

Question:
{question}

Instructions:
1. Use only the provided context.
2. Do not invent information.
3. Explain answers in simple language.
4. If the answer is not available, respond:
   I don't know based on the provided documents.

Answer:
""")


# ==========================================
# STEP 8: INITIALIZE CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# STEP 9: DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# ==========================================
# STEP 10: USER QUESTION
# ==========================================

question = st.chat_input("Ask something about your PDF...")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching PDF and generating answer..."):

            try:

                # Retrieve relevant chunks
                docs = retriever.invoke(question)

                if not docs:

                    answer = (
                        "I don't know based on the provided documents."
                    )

                else:

                    context = "\n\n".join(
                        doc.page_content
                        for doc in docs
                        if doc.page_content.strip()
                    )

                    if not context.strip():

                        answer = (
                            "I don't know based on the provided documents."
                        )

                    else:

                        formatted_prompt = prompt.invoke({
                            "context": context,
                            "question": question
                        })

                        # Generate answer using Groq
                        response = llm.invoke(formatted_prompt)

                        answer = response.content

                        if not isinstance(answer, str):
                            answer = str(answer)

                st.write(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                # Display sources
                if docs:

                    with st.expander("📄 View Sources"):

                        for i, doc in enumerate(docs, start=1):

                            st.write(f"**Source {i}**")

                            page = doc.metadata.get("page")

                            if isinstance(page, int):
                                st.write(f"Page: {page + 1}")
                            else:
                                st.write("Page: Unknown")

                            st.write(doc.page_content)
                            st.divider()

            except Exception as e:

                st.error(f"Error generating response: {str(e)}")
                print("ERROR:", str(e))
