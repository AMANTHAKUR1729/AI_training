import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    PERSIST_DIRECTORY = "./chroma_db"