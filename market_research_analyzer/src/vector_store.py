from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import Config

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.persist_directory = Config.PERSIST_DIRECTORY
    
    def add_documents(self, documents):
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        return vector_store
    
    def get_retriever(self):
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return vector_store.as_retriever(search_kwargs={"k": 4})