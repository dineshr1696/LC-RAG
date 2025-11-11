# grc_rag/vectorstore_manager.py
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

def init_pinecone(api_key, index_name, region, metric, embeddings):
    pc = Pinecone(api_key=api_key)
    existing_indexes = pc.list_indexes().names()

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region=region),
        )

    vectorstore = PineconeVectorStore.from_existing_index(index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever, pc, vectorstore

def rebuild_pinecone(pc, index_name, region, metric, embeddings, chunks):
    """Deletes and rebuilds the index with new chunks."""
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=768,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region=region),
    )

    vectorstore = PineconeVectorStore.from_existing_index(index_name, embedding=embeddings)
    for i in tqdm(range(0, len(chunks), 5)):
        vectorstore.add_documents(chunks[i:i + 5])

    return vectorstore
