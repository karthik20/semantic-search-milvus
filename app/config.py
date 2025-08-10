import os

class Settings:
    app_name: str = os.getenv("APP_NAME", "semantic-search-milvus")
    milvus_host: str = os.getenv("MILVUS_HOST", "localhost")
    milvus_port: int = int(os.getenv("MILVUS_PORT", "19530"))
    model_dir: str = os.getenv("MODEL_DIR", "/models/all-MiniLM-L6-v2-onnx")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    index_type: str = os.getenv("INDEX_TYPE", "IVF_FLAT")
    metric_type: str = os.getenv("METRIC_TYPE", "L2")

settings = Settings()