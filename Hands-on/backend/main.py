from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from backend.config import settings


def get_context_retriever_chain(vector_store, model_name):
    """
    Creates a retriever chain that is aware of the conversation history.

    Args:
        vector_store (Chroma): The vector store containing document embeddings.
        model_name (str): The name of the model to use.

    Returns:
        RetrievalChain: The history-aware retriever chain.
    """
    # FIX: Use settings.GOOGLE_API_KEY and updated model name
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain, model_name):
    """
    Creates a conversational RAG chain for answering questions.

    Args:
        retriever_chain (RetrievalChain): The history-aware retriever chain.
        model_name (str): The name of the model to use.

    Returns:
        RetrievalChain: The conversational RAG chain.
    """
    # FIX: Use settings.GOOGLE_API_KEY and updated model name
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_llm_only_response(user_input, chat_history, model_name):
    """
    Gets a response using only the LLM without RAG.

    Args:
        user_input (str): The user's question.
        chat_history (list): The conversation history.
        model_name (str): The name of the model to use.

    Returns:
        str: The generated answer from the LLM.
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.GOOGLE_API_KEY,
    )

    # Create a simple prompt for LLM-only responses
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. Answer the user's questions based on your general knowledge.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    chain = prompt | llm
    
    response = chain.invoke(
        {"chat_history": chat_history, "input": user_input}
    )

    return response.content


def get_response(user_input, vector_store, chat_history, model_name, rag_enabled=True):
    """
    Gets a response from either the conversational RAG chain or LLM only.

    Args:
        user_input (str): The user's question.
        vector_store (Chroma): The vector store for retrieval.
        chat_history (list): The conversation history.
        model_name (str): The name of the model to use.
        rag_enabled (bool): Whether to use RAG or LLM only.

    Returns:
        str: The generated answer.
    """
    if rag_enabled and vector_store:
        # Use RAG with document retrieval
        retriever_chain = get_context_retriever_chain(vector_store, model_name)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain, model_name)

        response = conversation_rag_chain.invoke(
            {"chat_history": chat_history, "input": user_input}
        )

        return response.get("answer", "Sorry, I could not find an answer.")
    else:
        # Use LLM only without RAG
        return get_llm_only_response(user_input, chat_history, model_name)