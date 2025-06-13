# generator.py
import os
import requests

def generate_answer_hf_api(
        query: str,
        retrieved_chunks: list[str],
        model_id: str = "HuggingFaceH4/zephyr-7b-beta",
        max_tokens: int = 180,
        temperature: float = 0.2,
        top_p: float= 0.9, 
        no_repeat_ngram_size: int = 5, 
        repetition_penalty: float = 1.2,
        timeout: int = 60
) -> str:
    """
    Call the HF Inference API to generate an answer from retrieved context.
    Raises RuntimeError on HTTP / JSON problems.
    """
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_key:
        raise RuntimeError("HUGGINGFACE_API_KEY not set")

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_key}"}

    context = "\n\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens, 
            "temperature": temperature,
            "top_p": top_p,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty    
            },
        "options": {"wait_for_model": True}
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Hugging Face request failed: {e}") from e

    data = resp.json()

    # Successful generation returns a list of dicts:
    if isinstance(data, list) and data and "generated_text" in data[0]:
        raw= data[0]["generated_text"].strip()
        # split at last occurrence of "Answer:"
        answer = raw.split("Answer:")[-1].strip()
        return answer

    # Model still loading or other server-side message:
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"HF API error: {data['error']}")

    raise RuntimeError(f"Unexpected response format: {data}")
