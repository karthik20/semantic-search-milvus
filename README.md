# Semantic Search with Milvus using LangChain and ONNX

## Overview

This service provides semantic search over two Milvus collections:

- help_support: ~200+ help/support documents (title, content, url, tags)
- services: consumer/retail banking services (service_id, url, name, description, intent_entity e.g. "account_opening-savings")

Embeddings are computed with all-MiniLM-L6-v2 via ONNX Runtime, integrated with LangChain through a custom embeddings class. Vector operations are handled by langchain-milvus.

## API

- GET /healthz
- POST /ingest/{collection}
  - collection: help_support | services
  - Body (help_support):
    ```json
    [
      {"id": "doc-001", "url":"https://...", "title":"...", "content":"...", "tags":["billing","cards"]},
      ...
    ]
    ```
  - Body (services):
    ```json
    [
      {"service_id":"svc-001","url":"https://...","name":"Open Savings Account","description":"...","intent_entity":"account_opening-savings"},
      ...
    ]
    ```
- POST /query
  - Request:
    ```json
    {
      "collection": "help_support",
      "query": "How do I reset my debit card PIN?",
      "page": 1,
      "page_size": 5,
      "metadata_filter": {"tags": "cards"},
      "search_params": {"nprobe": 16}
    }
    ```
  - Response:
    ```json
    {
      "collection": "help_support",
      "page": 1,
      "page_size": 5,
      "count": 5,
      "results": [
        { "id":"doc-001","url":"...","title":"...","content":"...","tags":"cards,billing","distance":0.23 }
      ]
    }
    ```

Notes:
- "Hybrid" here means dense semantic vector search with optional scalar metadata filters (e.g., intent_entity, tags).
- Integration is handled through langchain-milvus.

## Local Development

1) Prepare ONNX model

Place an ONNX-exported all-MiniLM-L6-v2 model at:
```
models/all-MiniLM-L6-v2-onnx/
  tokenizer.json
  model.onnx (or model_quant.onnx)
```

If you don't have the model, you can export one using Sentence Transformers or HuggingFace Optimum.

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Generate sample data and initialize collections

```bash
# Generate sample data and optionally load into Milvus collections
python load_sample_data.py               # Generate data only
python load_sample_data.py --ingest      # Generate data and ingest into collections
python load_sample_data.py --ingest --drop  # Drop collections, recreate, and ingest data
python load_sample_data.py --ingest --api   # Use API for ingestion (requires running API server)

# Alternatively, you can use the separate scripts:
python data/generate_sample_data.py  # Generate data only
python init_collections.py --drop    # Initialize collections (with drop option)
python ingestion/ingest_help_support.py --data data/help_support_data.json --direct
python ingestion/ingest_services.py --data data/services_data.json --direct
```

4) Run the application

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5) Test with sample queries

```bash
# Run default test queries against both collections (direct mode)
python test_queries.py

# Use API mode (requires the API server to be running)
python test_queries.py --api

# Run a specific query against a collection
python test_queries.py --query "How do I reset my password?" --collection help_support
```

5) Docker Compose

```
docker compose up --build
```

- App: http://localhost:8000
- Milvus: grpc on localhost:19530

GPU acceleration:
- Set `ORT_PROVIDER=CUDA` to explicitly use GPU for ONNX Runtime
- Ensure NVIDIA drivers + nvidia-container-toolkit installed when using Docker

3) Ingest sample data

```
python ingestion/ingest_help_support.py --data path/to/help_support.jsonl
python ingestion/ingest_services.py --data path/to/services.jsonl
```

Or use curl directly against /ingest/help_support and /ingest/services.

4) Query

```
curl -X POST http://localhost:8000/query -H 'content-type: application/json' -d '{
  "collection": "services",
  "query": "I want to open a savings account",
  "page": 1,
  "page_size": 5,
  "metadata_filter": {"intent_entity":"account_opening-savings"}
}'
```

## Kubernetes (K8s)

- Deploy Milvus and the app:

```
kubectl apply -f k8s/milvus-standalone.yaml
kubectl apply -f k8s/app-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

Provide the ONNX model via:
- CSI volume mount from an object store,
- ConfigMap (small models only),
- InitContainer that downloads the model into an EmptyDir,
- Or bake model into a custom image.

Update `image` in `k8s/app-deployment.yaml` to your published container.

## OpenShift

```
oc process -f openshift/template.yaml -p NAMESPACE=semantic-search -p IMAGE=ghcr.io/your-org/semantic-search-milvus:latest -p HOST=semantic-search.apps.example.com | oc apply -f -
```

## Configuration

Environment variables:
- MILVUS_HOST (default: localhost)
- MILVUS_PORT (default: 19530)
- MILVUS_URI (optional for Zilliz Cloud)
- MILVUS_TOKEN (optional for authentication)
- MODEL_DIR (default: /models/all-MiniLM-L6-v2-onnx)
- EMBEDDING_DIM (default: 384)
- ORT_PROVIDER (AUTO | CUDA | CPU, default: AUTO)
- ONNX_MODEL_FILENAME (optional, auto-detected if not specified)
- HELP_COLLECTION (default: help_support)
- SERVICES_COLLECTION (default: services)

Advanced configuration:
- Index params and search params can be adjusted through the Milvus UI or with custom scripts
- See langchain-milvus documentation for additional configuration options

## Notes and Future Enhancements

- Sparse+dense hybrid (BM25/SPLADE) using Milvus sparse vectors for improved keyword matching
- Reranking using a cross-encoder
- Background ingestion workers / streaming ingestion
- AuthN/AuthZ / multi-tenant partitions

## License

MIT (adjust as needed)