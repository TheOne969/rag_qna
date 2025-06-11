# test_rag.py 
""" didn't used this, so it's not modified accordingly, hence unreliable """ 
from rag import RAGRetriever
from dotenv import load_dotenv
import os
from weaviate import connect_to_local
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_KEY")


query = "What is the main topic of the lecture?"
with connect_to_local(port=8080, grpc_port=50051) as client:
        retriever = RAGRetriever(COLLECTION_NAME, embedding_model, client)
        results = retriever.retrieve(query, k=k)


print("\n🔍 Top Retrieved Chunks:\n")
for i, chunk in enumerate(results, 1):
    print(f"[{i}] {chunk[:300]}...\n")  # print only first 300 chars