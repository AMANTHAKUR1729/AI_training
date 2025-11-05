from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import HuggingFaceHub
import os

class ChatEngine:
    def __init__(self, retriever):
        self.retriever = retriever
        
    def ask_question(self, question):
        try:
            
            vector_store = self.retriever.vectorstore
            docs = vector_store.similarity_search(question, k=4)
            
            if not docs:
                return "I couldn't find relevant information in the uploaded documents. Please try asking about topics mentioned in your market reports."
            
            
            context_parts = []
            sources = set()
            
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                sources.add(source)
                context_parts.append(f"**From {source}:**\n{doc.page_content}")
            
            context = "\n\n".join(context_parts)
            
            response = f"""Based on your market research documents:

{context}

**Sources:** {', '.join(sorted(sources))}"""
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
