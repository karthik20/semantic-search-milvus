#!/usr/bin/env python
"""
Helper script to run test queries against the Milvus semantic search service.

This script performs test queries against both collections:
- help_support: For help content and support documents
- services: For banking services

Usage:
  python test_queries.py                # Run default test queries
  python test_queries.py --api          # Use API endpoint instead of direct LangChain
  python test_queries.py --query "How do I reset my password?" --collection help_support
"""

import argparse
import json
import os
import sys
import requests

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Sample test queries for each collection
DEFAULT_QUERIES = {
    "help_support": [
        "How do I reset my password?", 
        "What are the security best practices?", 
        "How to make mobile deposits?",
        "What are the fees for account maintenance?"
    ],
    "services": [
        "I need a savings account with high interest",
        "Looking for credit cards with travel rewards",
        "Need information about mortgage loans",
        "What investment options do you offer?"
    ]
}

def run_query_via_api(collection, query, api_url="http://localhost:8000/query"):
    """Run a query using the API endpoint"""
    payload = {
        "collection": collection,
        "query": query,
        "page": 1,
        "page_size": 3
    }
    
    print(f"\nQuerying {collection} via API for: '{query}'")
    try:
        resp = requests.post(api_url, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        
        print(f"Found {result.get('count', 0)} results:")
        for i, item in enumerate(result.get("results", []), 1):
            print(f"\n--- Result {i} (distance: {item.get('distance', 'N/A')}) ---")
            if collection == "help_support":
                print(f"Title: {item.get('title', 'N/A')}")
                print(f"URL: {item.get('url', 'N/A')}")
                print(f"Tags: {item.get('tags', 'N/A')}")
                content = item.get('content', '')
                print(f"Content: {content[:150]}..." if len(content) > 150 else f"Content: {content}")
            else:
                print(f"Name: {item.get('name', 'N/A')}")
                print(f"Intent: {item.get('intent_entity', 'N/A')}")
                print(f"URL: {item.get('url', 'N/A')}")
                desc = item.get('description', '')
                print(f"Description: {desc[:150]}..." if len(desc) > 150 else f"Description: {desc}")
        
        return True
    except Exception as e:
        print(f"Error running query via API: {e}")
        return False

def run_query_direct(collection, query):
    """Run a query using direct LangChain access"""
    from app.vectorstore import similarity_search
    
    print(f"\nQuerying {collection} directly for: '{query}'")
    try:
        results = similarity_search(collection, query, k=3)
        
        print(f"Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            distance = getattr(doc, 'distance', 'N/A')
            
            print(f"\n--- Result {i} (distance: {distance}) ---")
            if collection == "help_support":
                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"URL: {metadata.get('url', 'N/A')}")
                print(f"Tags: {metadata.get('tags', 'N/A')}")
                content = metadata.get('content', '')
                print(f"Content: {content[:150]}..." if len(content) > 150 else f"Content: {content}")
            else:
                print(f"Name: {metadata.get('name', 'N/A')}")
                print(f"Intent: {metadata.get('intent_entity', 'N/A')}")
                print(f"URL: {metadata.get('url', 'N/A')}")
                desc = metadata.get('description', '')
                print(f"Description: {desc[:150]}..." if len(desc) > 150 else f"Description: {desc}")
        
        return True
    except Exception as e:
        print(f"Error running direct query: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run test queries against the semantic search service")
    parser.add_argument("--api", action="store_true", help="Use the API endpoint instead of direct LangChain")
    parser.add_argument("--query", help="Specific query to run")
    parser.add_argument("--collection", choices=["help_support", "services"], help="Collection to query against")
    parser.add_argument("--api-url", default="http://localhost:8000/query", help="API URL for queries")
    args = parser.parse_args()
    
    # If specific query is provided, run only that
    if args.query:
        if not args.collection:
            print("Error: When providing a query, you must specify a collection with --collection")
            sys.exit(1)
            
        if args.api:
            run_query_via_api(args.collection, args.query, args.api_url)
        else:
            run_query_direct(args.collection, args.query)
        return
    
    # Otherwise run the default test queries
    print("Running test queries...")
    for collection, queries in DEFAULT_QUERIES.items():
        for query in queries:
            if args.api:
                run_query_via_api(collection, query, args.api_url)
            else:
                run_query_direct(collection, query)
            print("\n" + "-"*50)
    
    print("\nTest queries complete!")

if __name__ == "__main__":
    main()
