from typing import List
import weaviate.classes as wvc
from weaviate_handler import WeaviateHandler


class RAGRetriever:
    def __init__(self, collection_name: str, embedding_model, client):
        """
        collection_name : Weaviate collection to search
        embedding_model : any object with .encode(list[str]) → vectors
        client          : an **open** weaviate.WeaviateClient instance
        """
        self.embedding_model = embedding_model
        self.weaviate_handler = WeaviateHandler(collection_name, client)

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        vector = self.embedding_model.encode([query])[0]  # list[float]
        collection = self.weaviate_handler.collection
        result = collection.query.near_vector(
            near_vector=vector,
            limit=k,
            return_properties=["text","page","summary"],
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )

        return [
            {"uuid": o.uuid,   
             "text": o.properties["text"],
             "summary": o.properties.get("summary", ""),
             "page": o.properties.get("page", 0),
             "distance": o.metadata.distance}
            for o in result.objects
            ]
