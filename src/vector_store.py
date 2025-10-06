# Vector store management using FAISS.

from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import logging
from src.config import EMBEDDING_MODEL


logger = logging.getLogger(__name__)


class VectorStoreManager:
    # Manage FAISS vector store for document embeddings.
    
    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        # Initialize vector store manager.
        logger.info(f"Initializing embeddings with model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        # Create FAISS vector store from documents.
        if not documents:
            raise ValueError("Cannot create vector store from empty documents list")
        logger.info(f"Creating vector store with {len(documents)} documents")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self.vector_store
    
    def get_retriever(self, k: int = 3):
        # Get retriever for similarity search.
        if self.vector_store is None:
            raise ValueError("Vector store not created yet")
        return self.vector_store.as_retriever(search_kwargs={"k": k})
