# weaviate_handler.py
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter   

class WeaviateHandler:
    def __init__(self, collection_name: str, client: weaviate.WeaviateClient):
        self.client = client
        self.collection_name = collection_name # A collection is basically a schema/ blueprint. 

        if self.collection_name not in self.client.collections.list_all():
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(), #We are going to provide the embeddings as we want full control over the process. Normally, a model gives the values from their side. 
                vector_index_config=wvc.config.Configure.VectorIndex.hnsw(), # A popular algorithm for nearest neighbour search. 
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="page", data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="chunk_index", data_type=wvc.config.DataType.INT),
                    wvc.config.Property(name="summary",  data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="file_name", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="section", data_type=wvc.config.DataType.TEXT),
                ]
            )

        self.collection = self.client.collections.get(self.collection_name)

    def document_already_exists(self, file_name):
         flt = Filter.by_property("file_name").equal(file_name)
         return bool(self.collection.query.fetch_objects(limit=1, filters=flt).objects)
    
    def insert_chunks(self, chunks: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        
        if self.document_already_exists(metadatas[0]["file_name"]):
            print(f"⚠️ File '{metadatas[0]['file_name']}' already ingested. Skipping.")
            return
        
        for chunk, vector, metadata in zip(chunks, embeddings, metadatas): #zip is converting them into a tuple. 
            self.collection.data.insert(
                properties={**metadata, "text": chunk, "summary": ""},
                vector=vector
            )
        print(f"✅ Inserted {len(chunks)} chunks into Weaviate!")
    

    def close(self):
        self.client.close()