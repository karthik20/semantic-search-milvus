#!/usr/bin/env python
"""
Helper script to generate and load sample data into Milvus collections.

This script combines the following operations:
1. Initializing Milvus collections
2. Generating realistic sample data for help_support and services collections
3. Ingesting the generated data into Milvus

Usage:
  python load_sample_data.py               # Generate data only
  python load_sample_data.py --ingest      # Generate and ingest data
  python load_sample_data.py --ingest --drop  # Drop collections, recreate, generate, and ingest data
"""

import os
import sys
import argparse
import subprocess
import time

def ensure_dependencies():
    """Ensure required dependencies are installed"""
    try:
        # Try importing important packages
        import faker
        print("All required packages found!")
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faker"])

def check_milvus():
    """Check if Milvus is running"""
    print("Checking Milvus connection...")
    try:
        from pymilvus import connections
        if os.getenv("MILVUS_URI"):
            connections.connect(uri=os.getenv("MILVUS_URI"))
        else:
            connections.connect(
                host=os.getenv("MILVUS_HOST", "localhost"),
                port=os.getenv("MILVUS_PORT", "19530")
            )
        print("Successfully connected to Milvus.")
        return True
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        print("Please ensure Milvus is running and accessible.")
        print("You can start Milvus with: docker compose up -d milvus-standalone")
        return False

def run_command(cmd, desc=None):
    """Run a command with proper output handling"""
    if desc:
        print(f"Running: {desc}")
    else:
        print(f"Running: {' '.join(cmd)}")
        
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate and load sample data into Milvus collections")
    parser.add_argument("--ingest", action="store_true", help="Ingest data into collections")
    parser.add_argument("--drop", action="store_true", help="Drop existing collections before initialization")
    parser.add_argument("--init", action="store_true", help="Initialize collections before generating or ingesting data")
    parser.add_argument("--api", action="store_true", help="Use API for ingestion (requires running API server)")
    args = parser.parse_args()

    # Ensure we have all necessary packages
    ensure_dependencies()

    # Always check Milvus if we need to init or ingest
    if args.init or args.ingest:
        if not check_milvus():
            print("Milvus check failed. Please ensure Milvus is running.")
            sys.exit(1)

        # Initialize collections if --init or --drop is set
        init_cmd = [sys.executable, "init_collections.py"]
        if args.drop:
            init_cmd.append("--drop")
            desc = "Initializing collections (dropping existing ones)"
        else:
            desc = "Initializing collections"

        if not run_command(init_cmd, desc):
            sys.exit(1)

    # Generate the sample data
    generate_cmd = [sys.executable, "data/generate_sample_data.py"]
    if not run_command(generate_cmd, "Generating sample data"):
        sys.exit(1)

    # If we need to ingest, call the ingest scripts
    if args.ingest:
        # Help support data ingestion
        help_cmd = [sys.executable, "ingestion/ingest_help_support.py", "--data", "data/help_support_data.json"]
        if not args.api:
            help_cmd.append("--direct")

        if not run_command(help_cmd, "Ingesting help_support data"):
            sys.exit(1)

        # Services data ingestion
        services_cmd = [sys.executable, "ingestion/ingest_services.py", "--data", "data/services_data.json"]
        if not args.api:
            services_cmd.append("--direct")

        if not run_command(services_cmd, "Ingesting services data"):
            sys.exit(1)

        print("\n✅ Success! Sample data has been generated and ingested into Milvus.")
        print("You can now start the API server with: uvicorn app.main:app --reload")
    else:
        print("\n✅ Success! Sample data has been generated.")
        print("To initialize collections, run: python load_sample_data.py --init")
        print("To ingest the data, run: python load_sample_data.py --ingest")
    
if __name__ == "__main__":
    main()
