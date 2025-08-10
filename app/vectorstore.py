import os
from typing import Dict, Any, List, Optional, Tuple
from app.embeddings.local_minilm_embeddings import LocalMiniLMEmbeddings
from pymilvus import AnnSearchRequest, RRFRanker
from app.milvus_client import get_milvus_client

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

# Default embedding model
def get_embeddings():
    from app.embeddings.onnx_embeddings import OnnxEmbeddings
    ONNX_PROVIDER = os.getenv("ORT_PROVIDER", "AUTO")  # AUTO | CUDA | CPU
    return OnnxEmbeddings(
        model_dir=MODEL_DIR,
        embedding_dim=EMBEDDING_DIM,
        provider=ONNX_PROVIDER,
    )
    # """Get the local all-MiniLM-L6-v2 ONNX model."""
    # return LocalMiniLMEmbeddings(
    #     model_dir=MODEL_DIR,
    # )

def get_connection_args() -> Dict[str, Any]:
    """Get Milvus connection arguments based on environment."""
    # Use host/port connection
    conn_args = {"host": MILVUS_HOST, "port": str(MILVUS_PORT)}
    return conn_args

def add_texts_to_collection(
    collection_name: str,
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
    dense_embeddings: Optional[List[List[float]]] = None,
    drop_old: bool = False
) -> List[str]:
    """Add texts to a specific Milvus collection with dense and sparse fields."""
    client = get_milvus_client()
    data = []
    for i, text in enumerate(texts):
        item = {
            "id": ids[i] if ids else i,
            "text": text,
            "text_dense": dense_embeddings[i] if dense_embeddings else None,
        }
        if metadatas:
            item.update(metadatas[i])
        data.append(item)
    client.insert(collection_name=collection_name, data=data)
    return [str(d["id"]) for d in data]

def hybrid_search(collection_name: str, query_text: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Perform hybrid search using both dense and sparse vectors."""
    client = get_milvus_client()
    print(f"Performing hybrid search on collection '{collection_name}' with query: {query_text}")
    # Compute dense embedding from query_text
    dense_query = get_embeddings().embed_query(query_text)
    req_dense = AnnSearchRequest(data=[dense_query], anns_field="text_dense", param={"nprobe": 10}, limit=k)
    req_sparse = AnnSearchRequest(data=[query_text], anns_field="text_sparse", param={"drop_ratio_search": 0.2}, limit=k)
    ranker = RRFRanker(100)

    # Set output_fields based on collection
    if collection_name == HELP_COLLECTION:
        output_fields = [
            "id", "text", "title", "url", "content", "tags"
        ]
    elif collection_name == SERVICES_COLLECTION:
        output_fields = [
            "id", "text", "name", "url", "description", "intent_entity"
        ]
    else:
        output_fields = ["id", "text"]

    results = client.hybrid_search(
        collection_name=collection_name,
        output_fields=output_fields,
        reqs=[req_dense, req_sparse],
        ranker=ranker,
        limit=k
    )[0]
    hits = []

    for hit in results:
        print(f"Hybrid search results: {hit}")
        hit_data = {'distance': getattr(hit, 'distance', None)}
        for key in output_fields:
            hit_data[key] = getattr(hit, key, None)
        hits.append(hit_data)
    return hits