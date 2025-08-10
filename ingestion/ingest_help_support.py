import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import requests
import sys

# Add the parent directory to the path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Option to use API or direct LangChain import
USE_API = os.getenv("USE_API", "true").lower() == "true"

def load_items(path: Path) -> List[Dict[str, Any]]:
    # Supports .jsonl or .json (list)
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items
    else:
        return json.loads(path.read_text())

def ingest_via_api(items: List[Dict[str, Any]], api_url: str):
    resp = requests.post(api_url, json=items, timeout=600)
    resp.raise_for_status()
    print(resp.json())

def ingest_direct(items: List[Dict[str, Any]]):
    from app.vectorstore import add_texts_to_collection, HELP_COLLECTION
    from app.schemas import IngestHelpSupportItem
    
    # Process items
    docs = [IngestHelpSupportItem(**it) for it in items]

    # Skip duplicate IDs
    seen = set()
    texts = []
    metadatas = []
    ids = []
    for d in docs:
        if d.id in seen:
            continue
        seen.add(d.id)
        texts.append(f"{d.title}\n\n{d.content}")
        metadatas.append({
            "url": d.url,
            "title": d.title,
            "content": d.content,
            "tags": ",".join(d.tags or []),
        })
        ids.append(d.id)

    if len(ids) != len(texts):
        print("Mismatch between number of IDs and texts after removing duplicates!")
        sys.exit(1)

    # Add to vector store in batches
    batch_size = 50
    total_inserted = 0
    for i in range(0, len(ids), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_result = add_texts_to_collection(
            collection_name=HELP_COLLECTION,
            texts=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        total_inserted += len(batch_result)
        print(f"Inserted batch {i//batch_size+1}: {len(batch_result)} documents")
    print(f"Ingested {total_inserted} documents to {HELP_COLLECTION}")

def main():
    parser = argparse.ArgumentParser(description="Ingest help & support documents via API or directly")
    parser.add_argument("--data", required=True, help="Path to JSONL/JSON with fields: id,url,title,content,tags[]")
    parser.add_argument("--api", default="http://localhost:8000/ingest/help_support")
    parser.add_argument("--direct", action="store_true", help="Ingest directly using LangChain instead of API")
    args = parser.parse_args()

    items = load_items(Path(args.data))
    
    # Decide whether to use API or direct ingestion
    if args.direct or not USE_API:
        ingest_direct(items)
    else:
        ingest_via_api(items, args.api)

if __name__ == "__main__":
    main()