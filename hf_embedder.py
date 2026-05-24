# hf_embedder.py
import os, requests
from typing import List

class HFEmbedderAPI:
    def __init__(self, api_key: str, model_id="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_id = model_id
        self.api_key  = api_key
        
        if not self.api_key:
            raise ValueError("Hugging Face API key must be provided to the embedder.")

    def encode(self, texts: List[str]):
        url = (
            f"https://router.huggingface.co/hf-inference/models/"
            f"{self.model_id}/pipeline/feature-extraction"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        resp = requests.post(url, headers=headers, json={"inputs": texts})
        
        if resp.status_code != 200:
            raise RuntimeError(
                f"Hugging Face API error {resp.status_code}: {resp.text}"
            )
        return resp.json()          # list[list[float]]
