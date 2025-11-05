import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from backend.db_manager import create_vectorstore_from_text, get_vectorstore
from backend.main import get_response
from backend.config import settings

st.set_page_config(page_title="Chat with PDF", page_icon="🤖", layout="wide")
st.title("Chat with PDF 📄")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()

# Initialize session state for model and RAG
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-2.0-flash"
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True

with st.sidebar:
    st.header("Settings")

    if (
        not settings.GOOGLE_API_KEY
        or settings.GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE"
    ):
        st.error(
            "Google API Key not found! Please create a .env file with your GOOGLE_API_KEY."
        )
        st.stop()
    else:
        st.success("Google API Key loaded.")

    # Model selection dropdown
    st.subheader("Model Selection")
    model_options = ["gemini-2.0-flash", "gemini-pro", "gemini-1.5-flash"]
    selected_model = st.selectbox(
        "Choose AI Model:",
        options=model_options,
        index=model_options.index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model
    st.info(f"Selected model: {selected_model}")

    # RAG toggle
    st.subheader("RAG Settings")
    rag_enabled = st.toggle("Enable RAG", value=st.session_state.rag_enabled)
    st.session_state.rag_enabled = rag_enabled
    
    if rag_enabled:
        st.success("RAG is enabled - Answers will use your document content")
    else:
        st.warning("RAG is disabled - Answers will use only the model's knowledge")

    if st.session_state.vector_store:
        st.info("Existing database loaded from disk.")
    else:
        st.warning("No database found. Upload a file to start.")

    uploaded_file = st.file_uploader("Upload a text file (.txt)", type=["txt"])

    if uploaded_file:
        if st.button("Process File"):
            with st.spinner("Processing file and building vector store..."):
                text_content = uploaded_file.getvalue().decode("utf-8")

                vector_store = create_vectorstore_from_text(text_content)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    st.session_state.chat_history = [
                        AIMessage(
                            content="Hello! The file has been processed. How can I help you?"
                        ),
                    ]
                    st.success("Vector store created successfully!")
                    st.rerun()

if "chat_history" not in st.session_state:
    if st.session_state.vector_store and st.session_state.rag_enabled:
        st.session_state.chat_history = [
            AIMessage(
                content="Hello! An existing database is loaded. Ask me anything about it."
            )
        ]
    else:
        st.session_state.chat_history = [
            AIMessage(content="Hello! Please upload a text file to start chatting.") if st.session_state.rag_enabled else
            AIMessage(content="Hello! I'm ready to answer your questions using my general knowledge.")
        ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query.strip() != "":
    if st.session_state.rag_enabled and not st.session_state.vector_store:
        st.error("Please upload and process a file before asking questions.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.write(user_query)

        with st.spinner("Thinking..."):
            response = get_response(
                user_query, 
                st.session_state.vector_store, 
                st.session_state.chat_history,
                st.session_state.selected_model,
                st.session_state.rag_enabled
            )

        st.session_state.chat_history.append(AIMessage(content=response))
        with st.chat_message("AI"):
            st.write(response)