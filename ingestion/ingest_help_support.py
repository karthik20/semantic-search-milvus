import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import requests

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

def main():
    parser = argparse.ArgumentParser(description="Ingest help & support documents via API")
    parser.add_argument("--data", required=True, help="Path to JSONL/JSON with fields: id,url,title,content,tags[]")
    parser.add_argument("--api", default="http://localhost:8000/ingest/help_support")
    args = parser.parse_args()

    items = load_items(Path(args.data))
    resp = requests.post(args.api, json=items, timeout=600)
    resp.raise_for_status()
    print(resp.json())

if __name__ == "__main__":
    main()