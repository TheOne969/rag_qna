# strategy.py
def choose_strategy(num_chunks: int,
                    max_tokens_per_chunk: int = 512,
                    model_context_tokens: int = 2048) -> str:
    """
    Decide which token-budget strategy to use.
    Returns: 'full_text' | 'sliding_window' | 'summarize'
    """
    total_tokens = num_chunks * max_tokens_per_chunk

    if total_tokens > model_context_tokens * 2:      # very large doc
        return "summarize"
    elif total_tokens > model_context_tokens:        # medium
        return "sliding_window"
    else:                                            # small
        return "full_text"
