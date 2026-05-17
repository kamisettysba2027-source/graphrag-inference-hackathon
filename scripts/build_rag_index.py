"""
Pipeline 2 setup: Build FAISS index from 95 papers.
Chunks documents, embeds with OpenAI text-embedding-3-small, saves index.
Run ONCE - the index is reused by pipeline_2 for all queries.
"""
import os
import json
import pickle
import time
from openai import OpenAI
import faiss
import numpy as np

OPENAI_API_KEY = "sk-proj-REDACTED"   # <-- replace with your working OpenAI key
SRC_DIR = "ingestion_txt"
INDEX_PATH = "rag_index.faiss"
CHUNKS_PATH = "rag_chunks.pkl"

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000      # chars per chunk (~250 tokens)
CHUNK_OVERLAP = 150    # char overlap between chunks

client = OpenAI(api_key=OPENAI_API_KEY)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def embed_batch(texts):
    """Embed a list of texts in one API call."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    files = [f for f in os.listdir(SRC_DIR) if f.endswith(".txt")]
    print(f"Found {len(files)} documents")

    all_chunks = []   # list of dicts: {doc_id, chunk_idx, text}
    for fname in files:
        doc_id = os.path.splitext(fname)[0]
        with open(os.path.join(SRC_DIR, fname), "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            all_chunks.append({"doc_id": doc_id, "chunk_idx": i, "text": ch})

    print(f"Created {len(all_chunks)} chunks total")

    # Embed in batches (OpenAI allows up to ~2048 inputs/request; use 100 to be safe)
    print("Embedding chunks (this calls OpenAI - costs ~$0.01-0.03 total)...")
    embeddings = []
    BATCH = 100
    t0 = time.time()
    for i in range(0, len(all_chunks), BATCH):
        batch_texts = [c["text"] for c in all_chunks[i:i+BATCH]]
        embs = embed_batch(batch_texts)
        embeddings.extend(embs)
        print(f"  Embedded {min(i+BATCH, len(all_chunks))}/{len(all_chunks)}")
        time.sleep(0.3)  # gentle pacing

    print(f"Embedding done in {time.time()-t0:.1f}s")

    # Build FAISS index
    emb_array = np.array(embeddings, dtype=np.float32)
    dim = emb_array.shape[1]
    print(f"Embedding dimension: {dim}")

    index = faiss.IndexFlatIP(dim)   # inner product (cosine if normalized)
    faiss.normalize_L2(emb_array)
    index.add(emb_array)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nSaved index: {INDEX_PATH} ({index.ntotal} vectors)")
    print(f"Saved chunks: {CHUNKS_PATH} ({len(all_chunks)} chunks)")
    print("Pipeline 2 index ready.")


if __name__ == "__main__":
    main()