import os
from enum import Enum
from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType, Collection,
    utility,
)

DEFAULT_NLIST = int(os.getenv("MILVUS_NLIST", "1024"))
DEFAULT_NPROBE = int(os.getenv("MILVUS_NPROBE", "16"))
DEFAULT_M = int(os.getenv("MILVUS_HNSW_M", "16"))
DEFAULT_EF = int(os.getenv("MILVUS_HNSW_EF", "64"))

HELP_COLLECTION = os.getenv("HELP_COLLECTION", "help_support")
SERVICES_COLLECTION = os.getenv("SERVICES_COLLECTION", "services")


class CollectionType(str, Enum):
    HELP_SUPPORT = HELP_COLLECTION
    SERVICES = SERVICES_COLLECTION


class MilvusClient:
    def __init__(self, host: str, port: int, embedding_dim: int, metric_type: str = "L2", index_type: str = "IVF_FLAT"):
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim
        self.metric_type = metric_type
        self.index_type = index_type

        connections.connect(alias="default", host=self.host, port=str(self.port))

    def ensure_collections(self):
        # HELP/SUPPORT collection
        if not utility.has_collection(HELP_COLLECTION):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=512),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="Consumer/Retail bank help & support content")
            coll = Collection(HELP_COLLECTION, schema)
            self._create_index(coll)
            coll.load()
        else:
            Collection(HELP_COLLECTION).load()

        # SERVICES collection
        if not utility.has_collection(SERVICES_COLLECTION):
            fields = [
                FieldSchema(name="service_id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=256),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="intent_entity", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="Consumer/Retail bank menu & services")
            coll = Collection(SERVICES_COLLECTION, schema)
            self._create_index(coll)
            coll.load()
        else:
            Collection(SERVICES_COLLECTION).load()

    def _create_index(self, collection: Collection):
        # Vector index
        if self.index_type.upper() == "HNSW":
            index_params = {
                "index_type": "HNSW",
                "params": {"M": DEFAULT_M, "efConstruction": DEFAULT_EF},
                "metric_type": self.metric_type,
            }
        else:
            index_params = {
                "index_type": "IVF_FLAT",
                "params": {"nlist": DEFAULT_NLIST},
                "metric_type": self.metric_type,
            }
        collection.create_index(field_name="embedding", index_params=index_params)

    def upsert_documents(self, collection: CollectionType, records: List[Dict[str, Any]]) -> int:
        coll = Collection(collection.value)
        # Upsert: Milvus doesn't enforce uniqueness except primary key; we can delete before insert if key exists.
        # For simplicity, assume ids are unique or we're inserting new data.
        if collection == CollectionType.HELP_SUPPORT:
            data = [
                [r["id"] for r in records],
                [r.get("url", "") for r in records],
                [r.get("title", "") for r in records],
                [r.get("content", "") for r in records],
                [r.get("tags", "") for r in records],
                [r["embedding"] for r in records],
            ]
        else:
            data = [
                [r["service_id"] for r in records],
                [r.get("url", "") for r in records],
                [r.get("name", "") for r in records],
                [r.get("description", "") for r in records],
                [r.get("intent_entity", "") for r in records],
                [r["embedding"] for r in records],
            ]
        mr = coll.insert(data)
        coll.flush()
        return mr.insert_count

    def search(
        self,
        collection: CollectionType,
        vector: List[float],
        page: int = 1,
        page_size: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        coll = Collection(collection.value)
        coll.load()
        expr = self._build_expr(metadata_filter)
        # Search params
        if self.index_type.upper() == "HNSW":
            search_params = {"metric_type": self.metric_type, "params": {"ef": params.get("ef", DEFAULT_EF)} if params else {"ef": DEFAULT_EF}}
        else:
            nprobe = params.get("nprobe", DEFAULT_NPROBE) if params else DEFAULT_NPROBE
            search_params = {"metric_type": self.metric_type, "params": {"nprobe": nprobe}}
        offset = max(0, (page - 1) * page_size)
        limit = page_size
        fields = output_fields or (["url", "title", "content", "tags"] if collection == CollectionType.HELP_SUPPORT else ["service_id", "url", "name", "description", "intent_entity"])
        results = coll.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            offset=offset,
            expr=expr,
            output_fields=fields,
        )
        hits = results[0]
        items = []
        for h in hits:
            row = {"distance": float(h.distance)}
            for f in fields:
                row[f] = h.entity.get(f)
            # Include primary key
            if collection == CollectionType.HELP_SUPPORT:
                row["id"] = h.entity.get("id")
            else:
                row["service_id"] = h.entity.get("service_id")
            items.append(row)

        return {
            "collection": collection.value,
            "page": page,
            "page_size": page_size,
            "count": len(items),
            "results": items,
        }

    def _build_expr(self, filt: Optional[Dict[str, Any]]) -> Optional[str]:
        if not filt:
            return None
        parts = []
        for k, v in filt.items():
            if v is None:
                continue
            if isinstance(v, str):
                # Escape single quotes
                vv = v.replace("'", "\\'")
                parts.append(f"{k} == '{vv}'")
            elif isinstance(v, (int, float)):
                parts.append(f"{k} == {v}")
            elif isinstance(v, list):
                # IN list for strings/numbers
                vals = []
                for item in v:
                    if isinstance(item, str):
                        escaped_item = item.replace("'", "\\'")
                        vals.append(f"'{escaped_item}'")
                    else:
                        vals.append(str(item))
                parts.append(f"{k} in [{', '.join(vals)}]")
        return " and ".join(parts) if parts else None