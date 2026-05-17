"""
Pipeline 1: LLM-Only Baseline (OpenAI gpt-4o-mini)
Sends questions directly to the LLM with no retrieval.
Uses same model as Pipelines 2 & 3 for fair comparison.
"""
import os
import time
import json
from openai import OpenAI
import tiktoken

OPENAI_API_KEY = "sk-proj-REDACTED"
LLM_MODEL = "gpt-4o-mini"

PRICE_INPUT_PER_1M = 0.15
PRICE_OUTPUT_PER_1M = 0.60

enc = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = """You are a medical research assistant specializing in Type 2 Diabetes drug interactions and pharmacology. Answer questions accurately based on established medical knowledge. If you don't know something, say so clearly. Provide concise, factual answers."""


class LLMOnlyPipeline:
    def __init__(self, api_key=OPENAI_API_KEY):
        self.client = OpenAI(api_key=api_key)
        self.results = []

    def count_tokens(self, text):
        return len(enc.encode(text or ""))

    def calculate_cost(self, input_tokens, output_tokens):
        return (input_tokens / 1_000_000) * PRICE_INPUT_PER_1M + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_1M

    def answer(self, question):
        full_input = SYSTEM_PROMPT + "\n\n" + question
        input_tokens = self.count_tokens(full_input)
        start_time = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
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
            "pipeline": "llm_only",
            "model": LLM_MODEL,
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
        print(f"Pipeline 1: LLM-Only (gpt-4o-mini) - {len(questions)} questions")
        print("=" * 60)
        for i, q in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {q[:55]}...")
            r = self.answer(q)
            if r["error"]:
                print(f"   ERROR: {r['error'][:120]}")
            else:
                print(f"   OK | Tokens: {r['total_tokens']} | "
                      f"Latency: {r['latency_seconds']}s | Cost: ${r['cost_usd']:.6f}")
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
        "What are common adverse effects of SGLT2 inhibitors?"
    ]
    p = LLMOnlyPipeline()
    p.run_batch(test_questions, save_path="results/pipeline_1_test_results.json")