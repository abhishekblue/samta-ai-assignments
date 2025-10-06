# Task 2: RAG system with PDF and Word documents.

import logging
from dotenv import load_dotenv
from pathlib import Path
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, MODEL, TEMPERATURE
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine


load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Run Task 2: PDF + Word document RAG system.
    logger.info("Initializing RAG system for Task 2...")
    
    doc_processor = DocumentProcessor(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Load multiple documents
    file_paths = [
    "data/sample.pdf",           # Original PDF
    "data/extracted_text.docx"   # Word file created in Task 1
    ]
    
    for path in file_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    logger.info("Loading and processing documents...")
    documents = doc_processor.load_multiple(file_paths)
    logger.info(f"Loaded {len(documents)} document chunks from {len(file_paths)} files")
    
    # Create vector store
    vector_manager = VectorStoreManager()
    vector_manager.create_vector_store(documents)
    retriever = vector_manager.get_retriever(k=TOP_K)
    
    # Initialize RAG engine
    rag_engine = RAGEngine(
        retriever=retriever,
        model_name=MODEL,
        temperature=TEMPERATURE
    )
    
    # Interactive query loop
    logger.info("\n" + "="*50)
    logger.info("RAG System Ready! Ask questions (type 'quit' to exit)")
    logger.info("="*50 + "\n")
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result = rag_engine.query(question)
            
            print(f"\nAnswer: {result['answer']}\n")
            print(f"Sources: {len(result['sources'])} documents used")
            
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break

if __name__ == "__main__":
    main()
