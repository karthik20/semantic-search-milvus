import os
from typing import Dict, Any, List, Optional, Tuple
from langchain_milvus import Milvus
from langchain_core.documents import Document
from app.embeddings.onnx_embeddings import OnnxEmbeddings

# Collection names (from env or defaults)
HELP_COLLECTION = os.getenv("HELP_COLLECTION", "help_support")
SERVICES_COLLECTION = os.getenv("SERVICES_COLLECTION", "services")

# Milvus connection settings
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_URI = os.getenv("MILVUS_URI") 

# ONNX model settings
MODEL_DIR = os.getenv("MODEL_DIR", "/home/karthik/projects/ai-models/onnx/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
ONNX_PROVIDER = os.getenv("ORT_PROVIDER", "AUTO")  # AUTO | CUDA | CPU

# Default embedding model
def get_embeddings():
    """Get the ONNX embeddings model."""
    return OnnxEmbeddings(
        model_dir=MODEL_DIR,
        embedding_dim=EMBEDDING_DIM,
        provider=ONNX_PROVIDER,
    )

def get_connection_args() -> Dict[str, Any]:
    """Get Milvus connection arguments based on environment."""
    # Use host/port connection
    conn_args = {"host": MILVUS_HOST, "port": str(MILVUS_PORT)}
    return conn_args

def get_vectorstore(collection_name: str, drop_old: bool = False) -> Milvus:
    """Get a Milvus vector store for a specific collection."""
    embeddings = get_embeddings()
    # Explicitly set primary key field name for each collection
    if collection_name == HELP_COLLECTION:
        primary_field = "id"
    elif collection_name == SERVICES_COLLECTION:
        primary_field = "service_id"
    else:
        primary_field = "id"  # fallback

    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args=get_connection_args(),
        drop_old=drop_old,
        auto_id=False,  # Preserve document IDs
        primary_field=primary_field,
    )

def add_texts_to_collection(
    collection_name: str,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
    drop_old: bool = False
) -> List[str]:
    """Add texts to a specific Milvus collection."""
    vectorstore = get_vectorstore(collection_name, drop_old=drop_old)
    return vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

def similarity_search(
    collection_name: str,
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """Search for similar documents in a collection."""
    vectorstore = get_vectorstore(collection_name)
    return vectorstore.similarity_search(query, k=k, filter=filter)

def similarity_search_with_score(
    collection_name: str,
    query: str,
    k: int = 4,
    filter: Optional[Dict[str, Any]] = None
) -> List[Tuple[Document, float]]:
    """Search for similar documents and return with scores."""
    vectorstore = get_vectorstore(collection_name)
    if filter:
        return vectorstore.similarity_search_with_score(query, k=k, filter=filter)
    return vectorstore.similarity_search_with_score(query, k=k)
    
