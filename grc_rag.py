import os
from dotenv import load_dotenv
from pathlib import Path
import sys
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_huggingface import HuggingFaceEndpoint
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from datasets import Dataset

load_dotenv()
os.environ["OPENAI_API_KEY"] = "dummy-key"

def require_env(name: str):
    v = os.getenv(name)
    if not v:
        print(f" ERROR: {name} not set in .env.")
        sys.exit(1)
    return v

PINECONE_API_KEY = require_env("PINECONE_API_KEY")
PINECONE_INDEX_NAME = require_env("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
HUGGINGFACEHUB_API_TOKEN = require_env("HUGGINGFACEHUB_API_TOKEN")


#PDF LOad
PDF_DIR = Path(r"D:\LC GRC RAG\pdf")  

if not PDF_DIR.exists():
    print(f" Folder not found: {PDF_DIR}")
    sys.exit(1)

all_docs = []
for pdf_file in PDF_DIR.glob("*.pdf"):
    print(f" Loading {pdf_file.name} ...")
    try:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = pdf_file.name  
        all_docs.extend(docs)
    except Exception as e:
        print(f" Error loading {pdf_file.name}: {e}")

print(f" Loaded {len(all_docs)} total pages from {len(list(PDF_DIR.glob('*.pdf')))} PDFs")
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
chunks = splitter.split_documents(all_docs)
print(f" Split into {len(chunks)} chunks total")





#Embeddings

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
#Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
existing_indexes = pc.list_indexes().names()

if PINECONE_INDEX_NAME not in existing_indexes:
    print(f" Creating new Pinecone index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
    )
else:
    print(f" Reusing existing Pinecone index: {PINECONE_INDEX_NAME}")

index = pc.Index(PINECONE_INDEX_NAME)
print(f" Connected to Pinecone index: {PINECONE_INDEX_NAME}")



#Vectors
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)
print(" LangChain vectorstore connected to Pinecone index")
#Upload if empty
stats = index.describe_index_stats()
if stats.total_vector_count == 0:
    print(" Uploading document chunks to Pinecone (first-time setup)...")
    for i in tqdm(range(0, len(chunks), 5)):
        vectorstore.add_documents(chunks[i:i + 5])
    print(" All chunks uploaded to Pinecone.")
else:
    print(f"ℹ Pinecone already has {stats.total_vector_count} vectors.")


#Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(" Retriever created (k=3)")


#LLM
llm_client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HUGGINGFACEHUB_API_TOKEN,
)
print(" Mistral LLM initialized successfully")


#
def mistral_conversation(inputs):
    docs = inputs.get("context", [])
    if not docs:
        print(" No matching context retrieved from vectorstore.")
        return " No relevant information found in the documents."

    print("\n Retrieved Context Details:")
    for i, doc in enumerate(docs):
        src = doc.metadata.get("source", "Unknown PDF")
        pg = doc.metadata.get("page", "Unknown Page")
        print(f"   Match {i+1}: {src}, Page {pg}")
        print(f"      {doc.page_content[:200].strip()}...\n")

    context_text = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
    question = inputs.get("question", "").strip()

    messages = [
        {"role": "system", "content": (
            "You are a factual assistant. "
            "Answer strictly using the provided context. "
            "If it is not in the context, say: "
            "'The provided documents do not contain that information.'"
        )},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{question}"}
    ]

    try:
        response = llm_client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            top_p=0.9,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f" LLM error: {e}")
        return " LLM inference failed."



#Query
while True:
    user_query = input("\n Ask a question (or type 'exit' to quit): ").strip()
    if user_query.lower() in ["exit", "quit", "q"]:
        print(" Exiting RAG system. Goodbye!")
        break

    print("\n Retrieving and generating answer...\n")
    try:
        retrieved_docs = retriever.invoke(user_query)
        print(f" Retrieved {len(retrieved_docs)} matching chunks for query: '{user_query}'")

        result = mistral_conversation({"question": user_query, "context": retrieved_docs})

        print("\n----- ANSWER START ---")
        print(result)
        print("----- ANSWER END ----\n")

    except Exception as e:
        print(f" Error during query: {e}")



#Dataset
print("\n Building evaluation dataset...")
page_map = {}
for d in all_docs:
    p = d.metadata.get("page")
    source = d.metadata.get("source", "unknown")
    key = f"{source}_page_{p}"
    page_map[key] = d.page_content


eval_items = [
    {
        "question": "What are the NRI investment limits in private sector banks?",
        "gold_pages": ["rbi-guidelines.pdf_page_3.0"],
        "gold_answer": "The limit for individual NRI portfolio investment is 5%, aggregate 10%, extendable to 24%."
    },
    {
        "question": "What are the corporate governance guidelines for banks in UAE?",
        "gold_pages": ["2019-corporate-gov-standards_UAE.pdf_page_10.0"],
        "gold_answer": "UAE Central Bank mandates banks to comply with CBUAE Corporate Governance Standards 2019, covering board independence, audit, and risk management."
},

]


eval_data = {"question": [], "contexts": [], "answer": [], "reference": []}

for item in eval_items:
    q = item["question"]
    retrieved_docs = retriever.invoke(q)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    model_answer = mistral_conversation({"question": q, "context": retrieved_docs})
    gold_contexts = [page_map.get(p, "") for p in item["gold_pages"]]

    eval_data["question"].append(q)
    eval_data["contexts"].append(retrieved_texts)
    eval_data["answer"].append(model_answer)
    eval_data["reference"].append("\n".join(gold_contexts))

dataset = Dataset.from_dict(eval_data)

print("\n Evaluation dataset built successfully:")
print(dataset.to_pandas().head())

print("\n Running Local Retriever Evaluation (E5 embeddings)...")

def cosine_sim(a, b):
    a_vec = embeddings.embed_query(a)
    b_vec = embeddings.embed_query(b)
    return float(cosine_similarity([a_vec], [b_vec])[0][0])

def context_precision(retrieved_contexts, reference):
    sims = [cosine_sim(ctx, reference) for ctx in retrieved_contexts]
    return np.mean(sims)

def context_recall(retrieved_contexts, reference):
    combined = " ".join(retrieved_contexts)
    return cosine_sim(combined, reference)



results = []
for i in range(len(dataset)):
    q = dataset[i]["question"]
    ctx = dataset[i]["contexts"]
    ref = dataset[i]["reference"]
    prec = context_precision(ctx, ref)
    rec = context_recall(ctx, ref)
    results.append({"question": q, "context_precision": prec, "context_recall": rec})

df = pd.DataFrame(results)
print("\n  Retriever Evaluation ")
print(df.to_string(index=False))
print(" ")

print("\n Evaluating Generator Quality (faithfulness & answer relevance)...")

def mistral_judge(question, context, answer):
    """Use Mistral as a local evaluator for faithfulness & relevance."""
    prompt = f"""
You are an evaluator. Rate on a 0-1 scale.

Question: {question}
Context: {context}
Answer: {answer}

1️ Faithfulness: Is the answer strictly based on the given context (no hallucinations)?
2️ Answer Relevance: Does the answer actually answer the question?

Reply in JSON:
{{"faithfulness": <float>, "answer_relevance": <float>}}
"""
    try:
        response = llm_client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        text = response.choices[0].message["content"]
        import json, re
        text = re.search(r"\{.*\}", text, re.S).group()
        scores = json.loads(text)
        return scores.get("faithfulness", 0), scores.get("answer_relevance", 0)
    except Exception as e:
        print(f" Judge error: {e}")
        return 0, 0

gen_metrics = []
for i in range(len(dataset)):
    q = dataset[i]["question"]
    ctx = " ".join(dataset[i]["contexts"])
    ans = dataset[i]["answer"]
    f, a = mistral_judge(q, ctx, ans)
    gen_metrics.append({"question": q, "faithfulness": f, "answer_relevance": a})

gen_df = pd.DataFrame(gen_metrics)
print("\n  Generator Evaluation ")
print(gen_df.to_string(index=False))
print(" ")

final_df = pd.merge(df, gen_df, on="question")
print("\n  Overall RAG Evaluation Summary ")
print(final_df.to_string(index=False))
print(" ")


