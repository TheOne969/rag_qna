from hf_embedder import HFEmbedderAPI
from dotenv import load_dotenv
load_dotenv() 
embedder = HFEmbedderAPI()
test_embeddings = embedder.encode(["test sentence"])
print(len(test_embeddings[0]))  # Should output 384 (dimension of MiniLM-L6-v2)
