from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# Ingestion schemas
class IngestHelpSupportItem(BaseModel):
    id: str = Field(..., description="Unique id for the page")
    url: str
    title: str
    content: str
    tags: Optional[List[str]] = Field(default_factory=list)


class IngestServicesItem(BaseModel):
    service_id: str = Field(..., description="Unique service identifier")
    url: str
    name: str
    description: str
    intent_entity: str = Field(..., description="Intent-entity mapping like 'account_opening-savings'")


# Query schema
class QueryRequest(BaseModel):
    collection: Literal["help_support", "services"]
    query: str
    page: int = 1
    page_size: int = 5
    metadata_filter: Optional[Dict[str, Any]] = Field(default=None, description="Milvus scalar filter map")
    output_fields: Optional[List[str]] = None
    search_params: Optional[Dict[str, Any]] = Field(default=None, description="Milvus index search params override, e.g. {'nprobe': 16}")


class Hit(BaseModel):
    distance: float
    # Flexible payload - depends on collection
    id: Optional[str] = None
    service_id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    intent_entity: Optional[str] = None


class QueryResponse(BaseModel):
    collection: str
    page: int
    page_size: int
    count: int
    results: List[Hit]