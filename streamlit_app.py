import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import InferenceClient
from tqdm import tqdm
import tempfile

# --- Load environment ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")

# --- Streamlit UI Config ---
st.set_page_config(page_title="GRC RAG Assistant", layout="wide")
st.title(" Global GRC RAG Assistant")
st.markdown("Ask compliance or governance-related questions across multiple country policy PDFs.")

# --- Cache Initialization ---
@st.cache_resource
def init_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        st.warning(f"Index '{PINECONE_INDEX_NAME}' not found. Creating new one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
        st.info(f" Created new Pinecone index: {PINECONE_INDEX_NAME}")
    else:
        st.info(f" Using existing Pinecone index: {PINECONE_INDEX_NAME}")

    vectorstore = PineconeVectorStore.from_existing_index(
        PINECONE_INDEX_NAME, embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever, embeddings, pc, vectorstore


retriever, embeddings, pc, vectorstore = init_retriever()

# --- LLM Setup ---
@st.cache_resource
def init_llm():
    return InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2", token=HUGGINGFACEHUB_API_TOKEN)


llm_client = init_llm()

# --- Mistral Conversation ---
def mistral_conversation(question, context_docs):
    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in context_docs])
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
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=messages,
            max_tokens=512,
            temperature=0.0
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f" LLM Error: {e}"

# --- Sidebar: Upload & Index ---
st.sidebar.header(" Upload Policy PDFs")
uploaded_files = st.sidebar.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)

# Prepare upload temp dir
temp_dir = tempfile.mkdtemp()
all_chunks = []

if uploaded_files:
    st.sidebar.info("Processing uploaded PDFs...")
    for file in uploaded_files:
        file_path = Path(temp_dir) / file.name
        file_path.write_bytes(file.read())
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file.name
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    st.sidebar.success(f" {len(all_chunks)} chunks prepared from uploaded PDFs (ready for indexing).")

# --- Sidebar: Rebuild Pinecone Index ---
if st.sidebar.button("🔄 Rebuild Pinecone Index"):
    if not all_chunks:
        st.sidebar.warning("No new PDFs uploaded. Please upload files first.")
    else:
        with st.spinner("Rebuilding Pinecone index... this may take a few minutes."):
            try:
                if PINECONE_INDEX_NAME in pc.list_indexes().names():
                    st.sidebar.info(f" Deleting existing index '{PINECONE_INDEX_NAME}'...")
                    pc.delete_index(PINECONE_INDEX_NAME)

                st.sidebar.info(" Creating new index...")
                pc.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=768,
                    metric=PINECONE_METRIC,
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
                )

                st.sidebar.info(" Uploading document chunks...")
                new_vectorstore = PineconeVectorStore.from_existing_index(
                    PINECONE_INDEX_NAME, embedding=embeddings
                )
                for i in tqdm(range(0, len(all_chunks), 5)):
                    new_vectorstore.add_documents(all_chunks[i:i + 5])
                st.sidebar.success(" Index rebuilt successfully!")
            except Exception as e:
                st.sidebar.error(f"Error rebuilding index: {e}")

# --- Main QA Interface ---
st.header(" Ask a Question")
user_query = st.text_input("Enter your question here")

if st.button(" Retrieve & Generate Answer"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant context and generating answer..."):
            retrieved_docs = retriever.invoke(user_query)
            answer = mistral_conversation(user_query, retrieved_docs)

        st.subheader(" Retrieved Context")
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Match {i+1}:** {doc.metadata.get('source', 'Unknown PDF')}")
            st.text_area(f"Snippet {i+1}", doc.page_content[:700] + "...", height=150)

        st.subheader(" Model Answer")
        st.success(answer)

st.markdown("---")


