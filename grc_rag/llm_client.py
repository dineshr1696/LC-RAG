# grc_rag/llm_client.py
from huggingface_hub import InferenceClient

def init_llm(model_name, token):
    """Initialize HuggingFace LLM client."""
    return InferenceClient(model=model_name, token=token)

def generate_answer(llm_client, question, docs):
    """Run RAG generation using retrieved docs."""
    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
    messages = [
        {"role": "system", "content": (
            "You are a factual assistant specialized in governance, risk, and compliance (GRC). "
            "Answer only using the provided context. "
            "If the answer isn't in context, say: 'The provided documents do not contain that information.'"
        )},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{question}"}
    ]
    try:
        response = llm_client.chat.completions.create(
            model=llm_client.model,
            messages=messages,
            max_tokens=512,
            temperature=0.0
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"LLM Error: {e}"
