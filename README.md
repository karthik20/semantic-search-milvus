# Semantic Search with Milvus

A Python-based semantic search service using FastAPI, Milvus as the vector store, and ONNX Runtime embedding pipeline for all-MiniLM-L6-v2 model.

## Features

- **FastAPI**: High-performance web framework with automatic API documentation
- **Milvus**: Vector database for efficient similarity search with persistent storage
- **ONNX Runtime**: Optimized embedding generation with CUDA support
- **Two Collections**: Help/support content and services/menu metadata
- **Hybrid Search**: Dense vector search with scalar metadata filtering
- **GPU Acceleration**: Automatic CUDA detection and usage when available
- **Production Ready**: Docker, Kubernetes, and OpenShift deployment configurations

## Quick Start

### Prerequisites

1. **ONNX Model Setup**: Download the all-MiniLM-L6-v2 ONNX model:
   ```bash
   mkdir -p models/all-MiniLM-L6-v2-onnx
   # Download model.onnx (or model_quant.onnx) and tokenizer.json to the models directory
   # You can convert from Hugging Face using optimum: 
   # pip install optimum[exporters,onnxruntime]
   # optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2-onnx/
   ```

2. **Docker and Docker Compose**: Ensure you have Docker and Docker Compose installed.

### Local Development with Docker Compose

1. **Clone and setup**:
   ```bash
   git clone <repo-url>
   cd semantic-search-milvus
   ```

2. **Start services**:
   ```bash
   docker-compose up -d
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Milvus: localhost:19530

## API Endpoints

### Health Check
```bash
GET /healthz
```

### Ingest Data
```bash
# Help/Support content
POST /ingest/help_support
Content-Type: application/json

[
  {
    "id": "help-001",
    "url": "https://example.com/help/account-opening",
    "title": "How to Open an Account",
    "content": "Step-by-step guide to opening a new account...",
    "tags": ["account", "banking", "guide"]
  }
]

# Services metadata
POST /ingest/services
Content-Type: application/json

[
  {
    "service_id": "svc-001",
    "url": "https://example.com/services/savings",
    "name": "Savings Account",
    "description": "High-yield savings account with competitive rates",
    "intent_entity": "account_opening-savings"
  }
]
```

### Search
```bash
POST /query
Content-Type: application/json

{
  "collection": "help_support",
  "query": "How do I open a savings account?",
  "page": 1,
  "page_size": 5,
  "metadata_filter": {"tags": "account"},
  "output_fields": ["id", "title", "url"],
  "search_params": {"nprobe": 16}
}
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus server host |
| `MILVUS_PORT` | `19530` | Milvus server port |
| `MODEL_DIR` | `/models/all-MiniLM-L6-v2-onnx` | Path to ONNX model directory |
| `EMBEDDING_DIM` | `384` | Embedding dimension |
| `INDEX_TYPE` | `IVF_FLAT` | Milvus index type (`IVF_FLAT` or `HNSW`) |
| `METRIC_TYPE` | `L2` | Distance metric |
| `ORT_PROVIDER` | `AUTO` | ONNX Runtime provider (`AUTO`, `CUDA`, `CPU`) |

## Data Ingestion

Use the provided scripts to ingest data:

```bash
# Help/Support documents
python ingestion/ingest_help_support.py --data data/help_support.json --api http://localhost:8000/ingest/help_support

# Services metadata
python ingestion/ingest_services.py --data data/services.json --api http://localhost:8000/ingest/services
```

Data formats:
- Help/Support: `id`, `url`, `title`, `content`, `tags[]`
- Services: `service_id`, `url`, `name`, `description`, `intent_entity`

## Deployment

### Kubernetes

1. **Deploy Milvus**:
   ```bash
   kubectl apply -f k8s/milvus-standalone.yaml
   ```

2. **Deploy Application**:
   ```bash
   kubectl apply -f k8s/app-deployment.yaml
   kubectl apply -f k8s/ingress.yaml
   ```

### OpenShift

```bash
oc process -f openshift/template.yaml \
  -p NAMESPACE=semantic-search \
  -p IMAGE=your-registry/semantic-search-milvus:latest \
  -p HOST=semantic-search.apps.your-cluster.com \
  | oc apply -f -
```

## GPU Support

To enable CUDA acceleration:

1. **Docker Compose**: Uncomment the GPU configuration in `docker-compose.yml`
2. **Kubernetes**: Add GPU resource requests to the deployment
3. **Environment**: Set `ORT_PROVIDER=CUDA` to force CUDA usage

## Collections Schema

### Help/Support Collection
- `id` (VARCHAR, primary): Unique document ID
- `url` (VARCHAR): Document URL
- `title` (VARCHAR): Document title
- `content` (VARCHAR): Document content
- `tags` (VARCHAR): Comma-separated tags
- `embedding` (FLOAT_VECTOR): 384-dim embedding

### Services Collection
- `service_id` (VARCHAR, primary): Unique service ID
- `url` (VARCHAR): Service URL
- `name` (VARCHAR): Service name
- `description` (VARCHAR): Service description
- `intent_entity` (VARCHAR): Intent-entity mapping
- `embedding` (FLOAT_VECTOR): 384-dim embedding

## Development

### Local Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Milvus locally**:
   ```bash
   docker run -d --name milvus -p 19530:19530 -p 9091:9091 \
     -v milvus_data:/var/lib/milvus \
     milvusdb/milvus:2.4.7
   ```

3. **Run the application**:
   ```bash
   export MODEL_DIR=./models/all-MiniLM-L6-v2-onnx
   python -m uvicorn app.main:app --reload
   ```

### Testing

Example test queries:

```bash
# Ingest sample data
curl -X POST "http://localhost:8000/ingest/help_support" \
  -H "Content-Type: application/json" \
  -d '[{"id":"test-1","url":"http://example.com","title":"Account Opening","content":"How to open an account","tags":["account"]}]'

# Search
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"collection":"help_support","query":"open account","page":1,"page_size":5}'
```

## License

MIT License - see LICENSE file for details.
