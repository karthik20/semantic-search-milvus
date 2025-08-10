.PHONY: help setup build run test clean docker-build docker-up docker-down

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Set up the development environment
	pip install -r requirements.txt
	./setup_model.sh

build: ## Build Docker image
	docker build -t semantic-search-milvus .

run: ## Run the application locally
	@echo "Make sure Milvus is running (docker-compose up -d milvus-standalone)"
	@echo "And MODEL_DIR is set to ./models/all-MiniLM-L6-v2-onnx"
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run basic smoke tests
	@echo "Testing syntax..."
	python -c "import ast; [ast.parse(open(f).read()) for f in ['app/main.py', 'app/milvus_client.py', 'app/embeddings/onnx_embedder.py', 'app/schemas.py']]"
	@echo "✅ All Python files have valid syntax"

docker-build: ## Build Docker image
	docker build -t semantic-search-milvus .

docker-up: ## Start services with Docker Compose
	docker-compose up -d

docker-down: ## Stop services with Docker Compose
	docker-compose down

docker-logs: ## Show Docker Compose logs
	docker-compose logs -f

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	docker-compose down -v

ingest-sample: ## Ingest sample data (requires running service)
	@echo "Ingesting sample help/support data..."
	python ingestion/ingest_help_support.py --data data/help_support_sample.json
	@echo "Ingesting sample services data..."
	python ingestion/ingest_services.py --data data/services_sample.json
	@echo "✅ Sample data ingested successfully"

test-query: ## Test query endpoint (requires running service)
	@echo "Testing help/support query..."
	curl -X POST "http://localhost:8000/query" \
		-H "Content-Type: application/json" \
		-d '{"collection":"help_support","query":"how to open account","page":1,"page_size":3}'
	@echo ""
	@echo "Testing services query..."
	curl -X POST "http://localhost:8000/query" \
		-H "Content-Type: application/json" \
		-d '{"collection":"services","query":"savings account","page":1,"page_size":3}'