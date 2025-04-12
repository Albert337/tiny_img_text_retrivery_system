from pymilvus import connections, db
from pymilvus import Collection, utility, connections, db
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection, db, connections
import numpy as np


def crete_database():
    conn = connections.connect(host="0.0.0.0", port=19530)
    database = db.create_database("text_image_db")
    
    db.using_database("text_image_db")
    print(db.list_database())


def create_collection():
    conn = connections.connect(host="0.0.0.0", port=19530)
    db.using_database("text_image_db")
    
    m_id = FieldSchema(name="m_id", dtype=DataType.INT64, is_primary=True,)
    embeding_img = FieldSchema(name="embeding_img", dtype=DataType.FLOAT_VECTOR,dim=512,)
    path = FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=256,)
    schema = CollectionSchema(
        fields=[m_id, embeding_img, path],
        description="text to image embeding search",
        enable_dynamic_field=True
    )
    
    collection_name = "text_image_vector"
    collection = Collection(name=collection_name, schema=schema, using='default', shards_num=2)
    print(db.connections.list_connections())    


def create_index():
    conn = connections.connect(host="0.0.0.0", port=19530)
    db.using_database("text_image_db")
    

    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    
    collection = Collection("text_image_vector")
    if collection.has_index():
        collection.drop_index()
    collection.create_index(
        field_name="embeding_img",
        index_params=index_params
    )
    
    utility.index_building_progress("text_image_vector")


def insert_data():
    conn = connections.connect(host="0.0.0.0", port=19530)
    db.using_database("text_image_db")
    
    collection = Collection("text_image_vector")
    mids, embedings, paths = [], [], []
    data_num = 10
    for idx in range(0, data_num):
        mids.append(idx)
        embedings.append(np.random.normal(0, 0.1, 512).tolist())
        paths.append(f'path: random num {idx}')

    
    collection.insert([mids, embedings, paths])
    print(collection.num_entities)

def search_collection():
    conn = connections.connect(host="0.0.0.0", port=19530)
    db.using_database("text_image_db")
    
    collection = Collection("text_image_vector")
    collection.load()
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    query_embedding = np.random.normal(0, 0.1, 768).tolist()
    results = collection.search(
        data=[query_embedding],
        anns_field="embeding",
        param=search_params,
        limit=10,
        expr=None,
        output_fields=None,
        timeout=None,
        round_decimal=-1
    )
    print(results)

def delete_collection():
    conn = connections.connect(host="0.0.0.0", port=19530)
    db.using_database("text_image_db")
    
    collection = Collection("text_image_vector")
    collection.drop() 



if __name__ == '__main__':
    # crete_database()
    delete_collection()
    create_collection()
    create_index()

    # insert_data()
    # search_collection()
