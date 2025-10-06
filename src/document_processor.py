# Handle document loading and text processing.

from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    # Process PDF and Word documents into text chunks.
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        # Initialize document processor.
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        # Load and split PDF document.
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_word(self, word_path: str) -> List[Document]:
        # Load and split Word document.
        logger.info(f"Loading Word document: {word_path}")
        loader = Docx2txtLoader(word_path)
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
    
    def load_multiple(self, file_paths: List[str]) -> List[Document]:
        # Load multiple documents (PDF and Word).
        all_docs = []
        for path in file_paths:
            if path.endswith('.pdf'):
                all_docs.extend(self.load_pdf(path))
            elif path.endswith('.docx'):
                all_docs.extend(self.load_word(path))
            else:
                logger.warning(f"Unsupported file type: {path}")
        return all_docs
