import os
import sys
import tempfile
import streamlit as st
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from grc_rag.config import *
from grc_rag.pdf_loader import load_pdfs
from grc_rag.vectorstore_manager import init_pinecone, rebuild_pinecone
from grc_rag.llm_client import init_llm
from grc_rag.rag_service import run_query

sys.path.append(str(Path(__file__).resolve().parent))

st.set_page_config(page_title=" GRC RAG Assistant", layout="wide")
st.title(" Global GRC RAG Assistant")
st.markdown("Ask GRC-related questions across multiple country policy PDFs.")

@st.cache_resource
def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    retriever, pc, vectorstore = init_pinecone(
        PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_REGION, PINECONE_METRIC, embeddings
    )
    llm_client = init_llm(LLM_MODEL, HUGGINGFACEHUB_API_TOKEN)
    return retriever, pc, vectorstore, embeddings, llm_client

retriever, pc, vectorstore, embeddings, llm_client = initialize_system()

st.sidebar.header(" Upload Policy PDFs")
uploaded_files = st.sidebar.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)

temp_dir = tempfile.mkdtemp()
all_chunks = []

if uploaded_files:
    st.sidebar.info("Processing uploaded PDFs...")
    all_chunks = load_pdfs(uploaded_files, temp_dir)
    st.sidebar.success(f"{len(all_chunks)} chunks prepared (ready for indexing).")

if st.sidebar.button(" Rebuild Pinecone Index"):
    if not all_chunks:
        st.sidebar.warning("Please upload files first.")
    else:
        with st.spinner("Rebuilding Pinecone index..."):
            rebuild_pinecone(pc, PINECONE_INDEX_NAME, PINECONE_REGION, PINECONE_METRIC, embeddings, all_chunks)
            st.sidebar.success(" Index rebuilt successfully!")

st.header(" Ask a Question")
user_query = st.text_input("Enter your question:")

if st.button(" Retrieve & Generate Answer"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            answer, docs = run_query(llm_client, retriever, user_query)

        st.subheader(" Retrieved Context")
        for i, doc in enumerate(docs):
            st.markdown(f"**Match {i+1}:** {doc.metadata.get('source', 'Unknown PDF')}")
            st.text_area(f"Snippet {i+1}", doc.page_content[:700] + "...", height=150)

        st.subheader(" Model Answer")
        st.success(answer)

st.markdown("---")
st.caption("Developed by Dinesh R | Powered by LangChain ⚡ Pinecone ⚡ Mistral 7B")
