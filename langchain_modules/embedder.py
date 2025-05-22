from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document  # Changed from langchain.docstore.document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.logger import logger  # Added import

# Initialize the embedding model globally
embedding_model = HuggingFaceBgeEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

# Global variable to hold the FAISS vector store
vector_store = None  # Renamed from vectorstore


def embed_text_to_faiss(text: str):
    global vector_store  # Declare that we are using the global vector_store

    full_doc = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents([full_doc])

    # Create/replace the vector store with the new document chunks
    vector_store = FAISS.from_documents(chunks, embedding_model)  # Renamed from vectorstore
    logger.info(f"embedder.py: vector_store initialized, id: {id(vector_store)}")  # Added logging

    return {"message": f"{len(chunks)} chunks embedded successfully"}