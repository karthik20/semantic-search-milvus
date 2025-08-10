from pymilvus import DataType, Function, FunctionType
from app.milvus_client import get_milvus_client

HELP_COLLECTION = "help_support"
SERVICES_COLLECTION = "services"


def create_help_support_schema(client, dim_dense):
    schema = client.create_schema(auto_id=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000, enable_analyzer=True)
    schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=dim_dense)
    schema.add_field(field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="tags", datatype=DataType.VARCHAR, max_length=500)
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["text_sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)
    return schema

def create_services_schema(client, dim_dense):
    schema = client.create_schema(auto_id=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000, enable_analyzer=True)
    schema.add_field(field_name="text_dense", datatype=DataType.FLOAT_VECTOR, dim=dim_dense)
    schema.add_field(field_name="text_sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="url", datatype=DataType.VARCHAR, max_length=500)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=2000)
    schema.add_field(field_name="intent_entity", datatype=DataType.VARCHAR, max_length=500)
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["text_sparse"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)
    return schema

def init_hybrid_collection(collection_name, dim_dense, drop_old=False):
    client = get_milvus_client()
    if drop_old and client.has_collection(collection_name):
        print(f"Dropping collection: {collection_name}")
        client.drop_collection(collection_name)
    if collection_name == HELP_COLLECTION:
        schema = create_help_support_schema(client, dim_dense)
    elif collection_name == SERVICES_COLLECTION:
        schema = create_services_schema(client, dim_dense)
    else:
        raise ValueError(f"Unknown collection name: {collection_name}")
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="text_dense", index_name="text_dense_index", index_type="AUTOINDEX", metric_type="IP")
    index_params.add_index(field_name="text_sparse", index_name="text_sparse_index", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
