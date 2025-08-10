#!/usr/bin/env python
"""
Helper script to initialize Milvus collections with langchain-milvus.
"""

import os
import argparse
from app.vectorstore import get_vectorstore, HELP_COLLECTION, SERVICES_COLLECTION

def init_collections(drop_old: bool = False):
    """Initialize Milvus collections."""
    print(f"Initializing Milvus collections (drop_old={drop_old})")
    
    # Initialize help_support collection
    print(f"Creating {HELP_COLLECTION} collection...")
    vs_help = get_vectorstore(HELP_COLLECTION, drop_old=drop_old)
    print(f"  - Collection '{HELP_COLLECTION}' ready")
    
    # Initialize services collection
    print(f"Creating {SERVICES_COLLECTION} collection...")
    vs_services = get_vectorstore(SERVICES_COLLECTION, drop_old=drop_old)
    print(f"  - Collection '{SERVICES_COLLECTION}' ready")
    
    print("Initialization complete!")
    return vs_help, vs_services

def check_environment():
    """Check and print environment settings."""
    print("=== Environment Configuration ===")
    print(f"Milvus Host: {os.getenv('MILVUS_HOST', 'localhost')}")
    print(f"Milvus Port: {os.getenv('MILVUS_PORT', '19530')}")
    print(f"Milvus URI: {os.getenv('MILVUS_URI', 'None')}")
    print(f"Help Collection: {HELP_COLLECTION}")
    print(f"Services Collection: {SERVICES_COLLECTION}")
    print(f"ONNX Model Dir: {os.getenv('MODEL_DIR', '/models/all-MiniLM-L6-v2-onnx')}")
    print(f"ONNX Model Filename: {os.getenv('ONNX_MODEL_FILENAME', 'Auto-detected')}")
    print(f"ONNX Provider: {os.getenv('ORT_PROVIDER', 'AUTO')}")
    print(f"Embedding Dim: {os.getenv('EMBEDDING_DIM', '384')}")
    print("==============================")

def main():
    parser = argparse.ArgumentParser(description="Initialize Milvus collections for semantic search")
    parser.add_argument("--drop", action="store_true", help="Drop existing collections before initialization")
    parser.add_argument("--check", action="store_true", help="Only check environment, don't initialize")
    args = parser.parse_args()
    
    check_environment()
    
    if not args.check:
        init_collections(drop_old=args.drop)

if __name__ == "__main__":
    main()
