import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateBaseError

# Connect to local Weaviate
client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    additional_config=wvc.init.AdditionalConfig(timeout=wvc.init.Timeout(init=10)),
)

# Try to delete if exists
try:
    client.collections.delete("TestCollection")
except WeaviateBaseError:
    pass  # If not found or already deleted

# Create a schema
client.collections.create(
    name="TestCollection",
    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    properties=[
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT)
    ],
)

# Insert dummy vector
collection = client.collections.get("TestCollection")
collection.data.insert(
    properties={"content": "Weaviate local test successful."},
    vector=[0.1] * 384
)

print("✅ Inserted successfully into Weaviate!")

# Delete the test collection after successful insertion
try:
    client.collections.delete("TestCollection")
    print("🗑️  Test collection deleted")
except WeaviateBaseError as e:
    print(f"Failed to delete collection: {e}")
finally:
    client.close()


