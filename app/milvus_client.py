from pymilvus import MilvusClient
import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_URI = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

def get_milvus_client():
    """Return a singleton MilvusClient for standalone mode (no token)."""
    if not hasattr(get_milvus_client, "_client"):
        get_milvus_client._client = MilvusClient(uri=MILVUS_URI)
    return get_milvus_client._client
