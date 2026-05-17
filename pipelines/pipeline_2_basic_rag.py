"""
Pipeline 2: Basic RAG
Retrieves top-k chunks from FAISS, stuffs them into the prompt, answers with gpt-4o-mini.
Tracks tokens, latency, cost - same structure as Pipeline 1 for fair comparison.
"""
import os
import time
import json
import pickle
from openai import OpenAI
import faiss
import numpy as np
import tiktoken

OPENAI_API_KEY = "sk-proj-REDACTED"   # <-- same working OpenAI key

INDEX_PATH = "rag_index.faiss"
CHUNKS_PATH = "rag_chunks.pkl"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 5

# gpt-4o-mini pricing (USD per 1M tokens)
PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60

enc = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = """You are a medical research assistant specializing in Type 2 Diabetes drug interactions and pharmacology. Answer the question using ONLY the provided context passages. If the context does not contain the answer, say so clearly. Be concise and factual."""


class BasicRAGPipeline:
    def __init__(self, api_key=OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        self.results = []
        print(f"Loaded index: {self.index.ntotal} vectors, {len(self.chunks)} chunks")

    def count_tokens(self, text):
        return len(enc.encode(text))

    def calculate_cost(self, input_tokens, output_tokens):
        return (input_tokens / 1_000_000) * PRICE_INPUT_PER_1M + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M

    def retrieve(self, question, top_k=TOP_K):
        emb_resp = self.client.embeddings.create(model=EMBED_MODEL, input=[question])
        q_emb = np.array([emb_resp.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, top_k)
        retrieved = []
        for rank, i in enumerate(idxs[0]):
            c = self.chunks[i]
            retrieved.append({
                "rank": rank + 1,
                "doc_id": c["doc_id"],
                "chunk_idx": c["chunk_idx"],
                "score": float(scores[0][rank]),
                "text": c["text"]
            })
        return retrieved

    def answer(self, question, top_k=TOP_K):
        start_time = time.time()

        retrieved = self.retrieve(question, top_k)
        context = "\n\n".join([f"[Source: {r['doc_id']}]\n{r['text']}" for r in retrieved])

        user_msg = f"Context passages:\n{context}\n\nQuestion: {question}"
        full_input = SYSTEM_PROMPT + "\n\n" + user_msg
        input_tokens = self.count_tokens(full_input)

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0
            )
            answer_text = resp.choices[0].message.content
            error = None
        except Exception as e:
            answer_text = ""
            error = str(e)

        latency = time.time() - start_time
        output_tokens = self.count_tokens(answer_text) if answer_text else 0
        cost = self.calculate_cost(input_tokens, output_tokens)

        result = {
            "pipeline": "basic_rag",
            "model": LLM_MODEL,
            "question": question,
            "answer": answer_text,
            "retrieved_sources": [r["doc_id"] for r in retrieved],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_seconds": round(latency, 2),
            "cost_usd": round(cost, 6),
            "error": error
        }
        self.results.append(result)
        return result

    def run_batch(self, questions, save_path=None):
        print("=" * 60)
        print(f"Pipeline 2: Basic RAG - Running on {len(questions)} questions")
        print("=" * 60)

        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q[:60]}...")
            r = self.answer(q)
            if r["error"]:
                print(f"   ERROR: {r['error']}")
            else:
                print(f"   OK | Tokens: {r['total_tokens']} | "
                      f"Latency: {r['latency_seconds']}s | "
                      f"Cost: ${r['cost_usd']:.6f} | "
                      f"Sources: {r['retrieved_sources']}")

        valid = [r for r in self.results if not r["error"]]
        if valid:
            print("=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Successful:          {len(valid)}/{len(questions)}")
            print(f"Total input tokens:  {sum(r['input_tokens'] for r in valid)}")
            print(f"Total output tokens: {sum(r['output_tokens'] for r in valid)}")
            print(f"Total cost:          ${sum(r['cost_usd'] for r in valid):.4f}")
            print(f"Avg latency:         {sum(r['latency_seconds'] for r in valid)/len(valid):.2f}s")

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {save_path}")

        return self.results


if __name__ == "__main__":
    test_questions = [
        "What is metformin's primary mechanism of action?",
        "Which drugs were assessed for drug-drug interaction potential with Lotiglipron?",
        "How do SGLT2 inhibitors compare to GLP-1 receptor agonists in Type 2 Diabetes management?"
    ]
    pipeline = BasicRAGPipeline()
    pipeline.run_batch(test_questions, save_path="results/pipeline_2_test_results.json")