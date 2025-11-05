import streamlit as st
import os
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from backend.config import settings


def _ensure_event_loop():
    """Ensures that there is a running event loop in the current thread."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # 'RuntimeError: There is no current event loop...'
        # If there is no current event loop, create one and set it.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)


def create_vectorstore_from_text(text_content: str):
    """
    Creates a new vector store from raw text content, overwriting any existing one.

    Args:
        text_content (str): The raw text extracted from the uploaded file.

    Returns:
        Chroma: The Chroma vector store, or None on failure.
    """
    _ensure_event_loop()

    if not text_content:
        st.warning("The uploaded file appears to be empty.")
        return None

    try:
        # Wrap the raw text in a LangChain Document object
        documents = [Document(page_content=text_content)]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
        )
        document_chunks = text_splitter.split_documents(documents)

        # FIX: Use settings.GOOGLE_API_KEY instead of hard-coded key
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            document_chunks,
            embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )

        return vector_store
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        return None


def get_vectorstore():
    """
    Loads an existing vector store from the persistent directory.

    Returns:
        Chroma: The Chroma vector store, or None if it doesn't exist.
    """
    _ensure_event_loop()

    if not os.path.exists(settings.CHROMA_PERSIST_DIRECTORY):
        return None

    try:
        # FIX: Use settings.GOOGLE_API_KEY instead of hard-coded key
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        vector_store = Chroma(
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
        return vector_store
    except Exception as e:
        st.error(f"An error occurred while loading the vector store: {e}")
        return None
