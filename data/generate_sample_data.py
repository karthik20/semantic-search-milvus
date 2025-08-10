#!/usr/bin/env python
"""
Generate sample data for both collections and optionally ingest it.
This script creates realistic sample data for help_support and services collections
and optionally initializes Milvus collections and ingests the data.
"""

import os
import json
import sys
import subprocess
import argparse

# Check if Faker is installed and install if needed
try:
    from faker import Faker
    print("Faker package found, continuing...")
except ImportError:
    print("Installing Faker package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faker"])
    from faker import Faker

# Import our data generators
from generate_help_data import generate_help_support_data
from generate_services_data import generate_services_data

def generate_data():
    """Generate sample data for both collections."""
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    print("Generating sample data for help_support collection...")
    
    # Load existing samples
    try:
        with open('data/help_support_sample.json', 'r') as f:
            help_samples = json.load(f)
    except FileNotFoundError:
        print("Error: help_support_sample.json not found!")
        help_samples = []
    
    # Generate additional help_support data (100 items)
    help_data = generate_help_support_data(100)
    
    # Combine and save
    all_help_data = help_samples + help_data
    with open('data/help_support_data.json', 'w') as f:
        json.dump(all_help_data, f, indent=2)
    print(f"Generated {len(all_help_data)} help support documents in data/help_support_data.json")
    
    print("\nGenerating sample data for services collection...")
    
    # Load existing samples
    try:
        with open('data/services_sample.json', 'r') as f:
            services_samples = json.load(f)
    except FileNotFoundError:
        print("Error: services_sample.json not found!")
        services_samples = []
    
    # Generate additional services data (100 items)
    services_data = generate_services_data(100)
    
    # Combine and save
    all_services_data = services_samples + services_data
    with open('data/services_data.json', 'w') as f:
        json.dump(all_services_data, f, indent=2)
    print(f"Generated {len(all_services_data)} services in data/services_data.json")
    
    print("\nSample data generation complete!")
    
    return all_help_data, all_services_data

def init_milvus(drop=False):
    """Initialize Milvus collections."""
    print("\nInitializing Milvus collections...")
    cmd = [sys.executable, "init_collections.py"]
    if drop:
        cmd.append("--drop")
    
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        print("Failed to initialize collections!")
        sys.exit(1)
    
    print("Collections initialized successfully.")

def ingest_data(use_direct=True):
    """Ingest data into collections."""
    print("\nIngesting help support data...")
    help_cmd = [
        sys.executable, 
        "ingestion/ingest_help_support.py", 
        "--data", "data/help_support_data.json"
    ]
    if use_direct:
        help_cmd.append("--direct")
    
    result = subprocess.run(help_cmd, check=True)
    if result.returncode != 0:
        print("Failed to ingest help support data!")
        sys.exit(1)
    
    print("\nIngesting services data...")
    services_cmd = [
        sys.executable, 
        "ingestion/ingest_services.py", 
        "--data", "data/services_data.json"
    ]
    if use_direct:
        services_cmd.append("--direct")
    
    result = subprocess.run(services_cmd, check=True)
    if result.returncode != 0:
        print("Failed to ingest services data!")
        sys.exit(1)
    
    print("Data ingestion complete!")

def main():
    """Generate sample data and optionally ingest it."""
    parser = argparse.ArgumentParser(
        description="Generate and optionally ingest sample data for Milvus collections."
    )
    parser.add_argument(
        "--ingest", 
        action="store_true", 
        help="Ingest the generated data after creation"
    )
    parser.add_argument(
        "--init", 
        action="store_true", 
        help="Initialize collections before ingestion"
    )
    parser.add_argument(
        "--drop", 
        action="store_true", 
        help="Drop existing collections before initialization"
    )
    parser.add_argument(
        "--api", 
        action="store_true", 
        help="Use API for ingestion instead of direct LangChain (requires running API server)"
    )
    args = parser.parse_args()
    
    # Generate the data
    help_data, services_data = generate_data()
    
    # If ingestion is requested
    if args.ingest:
        # Initialize collections if requested
        if args.init:
            init_milvus(drop=args.drop)
        
        # Ingest the data
        ingest_data(use_direct=not args.api)
        
        print("\nData generation and ingestion complete. Your collections are ready to use!")
    else:
        print("\nData generation complete. To ingest data, run with --ingest flag.")
        print("Example: python data/generate_sample_data.py --ingest --init")

if __name__ == "__main__":
    main()
