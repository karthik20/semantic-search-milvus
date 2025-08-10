import os
import uvicorn
from fastapi import FastAPI, HTTPException, Path, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Literal, Dict, Any
from app.schemas import (
    IngestHelpSupportItem,
    IngestServicesItem,
    QueryRequest,
    QueryResponse,
    Hit,
)
from app.vectorstore import (
    add_texts_to_collection,
    similarity_search_with_score,
    HELP_COLLECTION,
    SERVICES_COLLECTION,
)

APP_NAME = os.getenv("APP_NAME", "semantic-search-milvus")

app = FastAPI(
    title=APP_NAME,
    version="0.2.0",
    description="FastAPI service providing semantic search over two Milvus collections using LangChain and Milvus.",
)

# CORS (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            
            # Prepare metadata
            metadatas = []
            ids = []
            for d in docs:
                metadatas.append({
                    "url": d.url,
                    "title": d.title,
                    "content": d.content,
                    "tags": ",".join(d.tags or []),
                })
                ids.append(d.id)
                
            # Add to Milvus via LangChain
            add_texts_to_collection(
                collection_name=HELP_COLLECTION,
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return {"ingested": len(texts), "collection": HELP_COLLECTION}
        else:
            docs = [IngestServicesItem(**it) for it in items]
            texts = [f"{d.name}\n{d.description}\n{d.intent_entity}" for d in docs]
            
            # Prepare metadata
            metadatas = []
            ids = []
            for d in docs:
                metadatas.append({
                    "url": d.url,
                    "name": d.name,
                    "description": d.description,
                    "intent_entity": d.intent_entity,
                })
                ids.append(d.service_id)
                
            # Add to Milvus via LangChain
            add_texts_to_collection(
                collection_name=SERVICES_COLLECTION,
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            return {"ingested": len(texts), "collection": SERVICES_COLLECTION}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if req.collection not in ("help_support", "services"):
        raise HTTPException(status_code=400, detail="Invalid collection. Use 'help_support' or 'services'.")
    try:
        collection_name = HELP_COLLECTION if req.collection == "help_support" else SERVICES_COLLECTION
        
        # Use LangChain for similarity search
        if req.metadata_filter:
            docs_and_scores = similarity_search_with_score(
                collection_name=collection_name,
                query=req.query,
                k=req.page_size,
                filter=req.metadata_filter
            )
        else:
            docs_and_scores = similarity_search_with_score(
                collection_name=collection_name,
                query=req.query,
                k=req.page_size
            )
        
        # Convert results to our response format
        results = []
        for doc, score in docs_and_scores:
            # Create a Hit from document metadata
            hit_data = {"distance": score}
            hit_data.update(doc.metadata)
            
            # Add document ID based on collection type
            if req.collection == "help_support":
                hit_data["id"] = doc.metadata.get("id", None)
            else:
                hit_data["service_id"] = doc.metadata.get("service_id", None)
                
            results.append(Hit(**hit_data))
            
        return QueryResponse(
            collection=req.collection,
            page=req.page,
            page_size=req.page_size,
            count=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=False)