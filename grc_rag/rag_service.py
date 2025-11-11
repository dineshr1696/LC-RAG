# grc_rag/rag_service.py
from grc_rag.llm_client import generate_answer

def run_query(llm_client, retriever, user_query):
    """Retrieve context and generate answer."""
    retrieved_docs = retriever.invoke(user_query)
    answer = generate_answer(llm_client, user_query, retrieved_docs)
    return answer, retrieved_docs
