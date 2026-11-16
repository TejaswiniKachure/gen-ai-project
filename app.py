
import streamlit as st

from chat import answer_question


# Page configuration
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="centered"
)


# Application title
st.title("📚 PDF RAG Chatbot")

st.write(
    "Ask questions about your PDF document "
    "and get answers using RAG."
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
question = st.chat_input("Ask something about your PDF...")


if question:

    # Display user question
    st.chat_message("user").markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Searching PDF and generating answer..."):

            try:
                answer, sources = answer_question(question)

                st.markdown(answer)

                # Display sources
                if sources:

                    st.markdown("**Sources:**")

                    for source in sources:

                        st.write(
                            f"📄 {source['file']} - "
                            f"Page {source['page']}"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(f"Error: {e}")
