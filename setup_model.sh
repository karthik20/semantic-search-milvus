#!/bin/bash

# Script to download and convert the all-MiniLM-L6-v2 model to ONNX format
# Requires: pip install optimum[exporters,onnxruntime]

MODEL_DIR="models/all-MiniLM-L6-v2-onnx"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

echo "Setting up ONNX model for all-MiniLM-L6-v2..."

# Create model directory
mkdir -p "$MODEL_DIR"

# Check if optimum is installed
if ! python -c "import optimum" 2>/dev/null; then
    echo "Installing optimum..."
    pip install "optimum[exporters,onnxruntime]"
fi

# Export model to ONNX
echo "Exporting model to ONNX format..."
optimum-cli export onnx --model "$MODEL_NAME" "$MODEL_DIR"

# Verify files exist
if [[ -f "$MODEL_DIR/model.onnx" && -f "$MODEL_DIR/tokenizer.json" ]]; then
    echo "✅ Model setup complete!"
    echo "Files created:"
    ls -la "$MODEL_DIR"
else
    echo "❌ Model setup failed. Please check the output above."
    exit 1
fi

echo ""
echo "You can now run the semantic search service:"
echo "export MODEL_DIR=./models/all-MiniLM-L6-v2-onnx"
echo "docker-compose up -d"