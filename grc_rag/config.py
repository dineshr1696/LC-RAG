# grc_rag/config.py
import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

# Default model
EMBEDDING_MODEL = "intfloat/e5-base-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Basic validation
if not all([PINECONE_API_KEY, HUGGINGFACEHUB_API_TOKEN, PINECONE_INDEX_NAME]):
    raise EnvironmentError(" Missing API keys or index name in .env file")
