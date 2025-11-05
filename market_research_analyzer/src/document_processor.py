import os
import csv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
    
    def process_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        return self._split_text(text, source=os.path.basename(file_path))
    
    def process_csv(self, file_path):
        text = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                text += " ".join(row) + "\n"
        return self._split_text(text, source=os.path.basename(file_path))
    
    def process_text(self, text, source="web"):
        return self._split_text(text, source=source)
    
    def _split_text(self, text, source):
        if not text.strip():
            return []
        documents = self.text_splitter.split_text(text)
        return [Document(page_content=doc, metadata={"source": source}) for doc in documents]