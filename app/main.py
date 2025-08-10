import os
import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Literal, Dict, Any
from app.milvus_client import MilvusClient, CollectionType
from app.schemas import (
    IngestHelpSupportItem,
    IngestServicesItem,
    QueryRequest,
    QueryResponse,
)
from app.embeddings.onnx_embedder import OnnxSentenceEmbedder

APP_NAME = os.getenv("APP_NAME", "semantic-search-milvus")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MODEL_DIR = os.getenv("MODEL_DIR", "/models/all-MiniLM-L6-v2-onnx")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
INDEX_TYPE = os.getenv("INDEX_TYPE", "IVF_FLAT")  # Or HNSW
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")  # Requested: L2
PROVIDER = os.getenv("ORT_PROVIDER", "AUTO")  # AUTO | CUDA | CPU

app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
    description="FastAPI service providing semantic search over two Milvus collections using all-MiniLM-L6-v2 embeddings (ONNX).",
)

# CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedder (ONNX Runtime with optional GPU)
embedder = OnnxSentenceEmbedder(
    model_dir=MODEL_DIR,
    embedding_dim=EMBEDDING_DIM,
    provider=PROVIDER,  # AUTO attempts CUDA else CPU
)

# Initialize Milvus client and ensure collections exist
milvus = MilvusClient(
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    embedding_dim=EMBEDDING_DIM,
    metric_type=METRIC_TYPE,
    index_type=INDEX_TYPE,
)

@app.on_event("startup")
def _startup():
    milvus.ensure_collections()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/ingest/{collection}", response_model=dict)
def ingest(
    collection: Literal["help_support", "services"] = Path(..., description="Target collection type"),
    items: List[Dict[str, Any]] = Body(..., description="List of items to ingest, schema depends on collection type"),
):
    try:
        if collection == "help_support":
            docs = [IngestHelpSupportItem(**it) for it in items]
            texts = [f"{d.title}\n\n{d.content}" for d in docs]
            vectors = embedder.embed(texts)
            records = []
            for d, v in zip(docs, vectors):
                records.append({
                    "id": d.id,
                    "url": d.url,
                    "title": d.title,
                    "content": d.content,
                    "tags": ",".join(d.tags or []),
                    "embedding": v,
                })
            count = milvus.upsert_documents(CollectionType.HELP_SUPPORT, records)
            return {"ingested": count, "collection": CollectionType.HELP_SUPPORT.value}
        else:
            docs = [IngestServicesItem(**it) for it in items]
            texts = [f"{d.name}\n{d.description}\n{d.intent_entity}" for d in docs]
            vectors = embedder.embed(texts)
            records = []
            for d, v in zip(docs, vectors):
                records.append({
                    "service_id": d.service_id,
                    "url": d.url,
                    "name": d.name,
                    "description": d.description,
                    "intent_entity": d.intent_entity,
                    "embedding": v,
                })
            count = milvus.upsert_documents(CollectionType.SERVICES, records)
            return {"ingested": count, "collection": CollectionType.SERVICES.value}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if req.collection not in ("help_support", "services"):
        raise HTTPException(status_code=400, detail="Invalid collection. Use 'help_support' or 'services'.")
    try:
        vector = embedder.embed([req.query])[0]
        out = milvus.search(
            collection=CollectionType(req.collection),
            vector=vector,
            page=req.page,
            page_size=req.page_size,
            metadata_filter=req.metadata_filter,
            output_fields=req.output_fields,
            params=req.search_params or {},
        )
        return out
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=False)