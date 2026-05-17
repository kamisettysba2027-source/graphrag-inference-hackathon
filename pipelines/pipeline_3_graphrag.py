"""
Pipeline 3: GraphRAG
Queries the TigerGraph knowledge graph via /graphrag/answerquestion.
Tracks tokens, latency, cost - same structure as Pipelines 1 & 2.
"""
import time
import json
import requests
from requests.auth import HTTPBasicAuth
import tiktoken

BASE_URL = "http://localhost:8000"
GRAPHNAME = "GraphRAG"
USERNAME = "tigergraph"
TOKEN = "REDACTED_TOKEN"

# gpt-4o-mini pricing (USD per 1M tokens) - GraphRAG uses this internally
PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60

enc = tiktoken.get_encoding("cl100k_base")


class GraphRAGPipeline:
    def __init__(self, method="hybrid"):
        self.auth = HTTPBasicAuth(USERNAME, TOKEN)
        self.headers = {"accept": "application/json", "Content-Type": "application/json"}
        self.method = method
        self.results = []

    def count_tokens(self, text):
        return len(enc.encode(text or ""))

    def calculate_cost(self, input_tokens, output_tokens):
        return (input_tokens / 1_000_000) * PRICE_INPUT_PER_1M + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M

    def answer(self, question):
        start_time = time.time()

        payload = {
            "question": question,
            "method": self.method,
            "method_params": {
                "indices": ["DocumentChunk"],
                "top_k": 5,
                "num_hops": 1,
                "num_seen_min": 1,
                "similarity_threshold": 0.90,
                "chunk_only": False,
                "doc_only": False,
                "expand": False,
                "combine": False,
                "verbose": False
            }
        }

        try:
            r = requests.post(
                f"{BASE_URL}/{GRAPHNAME}/graphrag/answerquestion",
                auth=self.auth, headers=self.headers,
                data=json.dumps(payload),
                timeout=180
            )
            if r.status_code == 200:
                resp = r.json()
                answer_text = resp.get("response") or resp.get("natural_language_response") or ""
                sources = resp.get("retrieved") or resp.get("query_sources") or {}
                error = None
            else:
                answer_text = ""
                sources = {}
                error = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            answer_text = ""
            sources = {}
            error = str(e)

        latency = time.time() - start_time

        # GraphRAG's token cost: the context it sends internally + the answer.
        # We approximate input as the retrieved context length + question.
        ctx_str = json.dumps(sources) if sources else ""
        input_tokens = self.count_tokens(question + ctx_str)
        output_tokens = self.count_tokens(answer_text)
        cost = self.calculate_cost(input_tokens, output_tokens)

        result = {
            "pipeline": f"graphrag_{self.method}",
            "model": "gpt-4o-mini",
            "question": question,
            "answer": answer_text,
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
        print(f"Pipeline 3: GraphRAG (method={self.method}) - {len(questions)} questions")
        print("=" * 60)

        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q[:55]}...")
            r = self.answer(q)
            if r["error"]:
                print(f"   ERROR: {r['error']}")
            else:
                print(f"   OK | Tokens: {r['total_tokens']} | "
                      f"Latency: {r['latency_seconds']}s | "
                      f"Cost: ${r['cost_usd']:.6f}")

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
    pipeline = GraphRAGPipeline(method="hybrid")
    pipeline.run_batch(test_questions, save_path="results/pipeline_3_test_results.json")