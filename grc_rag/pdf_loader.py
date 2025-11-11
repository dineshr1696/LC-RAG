# grc_rag/pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

def load_pdfs(uploaded_files, temp_dir):
    """Reads uploaded PDFs, extracts pages, and splits into chunks."""
    all_chunks = []
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

    return all_chunks
