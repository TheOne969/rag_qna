# summarizer.py  
import os, requests


HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
# Even if you just import the get_or... function, value of headers variable 
# would be passed along as well. 

API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"


def summarise_via_api(text: str,  max_tokens: int = 60) -> str:
    """Return an abstractive summary from the HF Inference API."""
    payload = {
        "inputs": text,
        "parameters": {
            "max_new_tokens": max_tokens,
            "min_length": 25,
            "do_sample": False
        },
        "options": {"wait_for_model": True}
    }
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=40)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and "summary_text" in data[0]:
        return data[0]["summary_text"]
    raise RuntimeError(f"Summarisation failed: {data}")


def get_or_create_summary(hit: dict, collection):
    if hit["summary"]:
        return hit["summary"]                         # already cached

    summary = summarise_via_api(hit["text"])          # call HF API once
    collection.data.update(oid=hit["uuid"],
                           properties={"summary": summary})
    return summary
