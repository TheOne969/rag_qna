import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateBaseError

# Connect to local Weaviate
client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=10)),
)

if "LectureSlides" in client.collections.list_all():
    client.collections.delete("LectureSlides")
    print("Deleted existing collection ✅")
else:
    print("Collection did not exist ❕")


client.close()