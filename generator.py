# generator.py
import openai
import os
import time
import random

def generate_answer_hf_api(
    query: str,
    retrieved_chunks: list[str],
    model_id: str = "openai/gpt-oss-120b:cerebras",
    max_tokens: int = 300,  # Increased from 180
    temperature: float = 0.2,
    **kwargs
) -> str:
    """
    Fixed version with higher token limits for complete answers
    """
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_key:
        raise RuntimeError("HUGGINGFACE_API_KEY not set")
    
    context = "\n\n".join(retrieved_chunks)
    
    # Enhanced prompt for complete responses
    system_prompt = """You are a helpful assistant that answers questions based on provided context. 
    Provide complete, well-structured answers using only the information from the context. 
    Always finish your sentences and provide comprehensive responses."""
    
    user_prompt = f"""Context:
{context}

Question: {query}

Please provide a complete and detailed answer based on the context above."""
    
    # Try GPT-OSS models with higher token limits
    client = openai.OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_key
    )
    
    models = [
        "openai/gpt-oss-120b:cerebras",
        "openai/gpt-oss-20b:fireworks-ai"
    ]
    
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,  # Use the increased limit
                temperature=temperature,
                # Add stop sequences to prevent abrupt cutoffs
                stop=None  # Let it finish naturally
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Check if answer seems complete (basic validation)
            if len(answer) > 20 and not answer.endswith(('—', '-', 'who is', 'that is', 'which is')):
                return answer
            else:
                print(f"⚠️  {model} gave incomplete answer: {answer[-50:]}")
                continue
                
        except Exception as e:
            print(f"❌ Model {model} failed: {e}")
            continue
    
    raise RuntimeError("All GPT-OSS models failed or gave incomplete answers")

def try_gpt_oss_models(hf_key, system_prompt, user_prompt, max_tokens, temperature):
    """Try GPT-OSS models with fresh client"""
    client = openai.OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_key
    )
    
    models = [
        "openai/gpt-oss-120b:cerebras",
        "openai/gpt-oss-20b:fireworks-ai"
    ]
    
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"   Model {model} failed: {e}")
            continue
    return None

def try_gpt_oss_with_retry(hf_key, system_prompt, user_prompt, max_tokens, temperature):
    """Retry GPT-OSS with backoff and jitter"""
    for attempt in range(3):
        try:
            # Add random delay to avoid rate limits
            time.sleep(random.uniform(1, 3))
            
            client = openai.OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_key
            )
            
            # Try with reduced parameters to increase success rate
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b:fireworks-ai",  # Use smaller model
                messages=[{"role": "user", "content": user_prompt}],  # Simplified
                max_tokens=min(max_tokens, 100),  # Reduce token limit
                temperature=0.1  # Lower temperature
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"   Retry attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None

def try_local_fallback(hf_key, system_prompt, user_prompt, max_tokens, temperature):
    """Fallback to local transformers when HF API fails"""
    try:
        from transformers import pipeline
        
        # Use a small, fast model
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        
        # Simplified prompt for local model
        prompt = user_prompt.replace("Context:", "Answer this question using the context:")
        
        result = generator(
            prompt, 
            max_length=min(max_tokens + 50, 200),
            temperature=temperature,
            do_sample=temperature > 0
        )
        
        return result[0]['generated_text'].strip()
        
    except Exception as e:
        print(f"   Local fallback failed: {e}")
        return None
