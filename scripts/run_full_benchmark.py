"""
MASTER BENCHMARK RUNNER
Runs all 3 pipelines on the 30 benchmark questions, saves results.
Then evaluation/evaluate.py scores them.
"""
import json
import sys
import os

sys.path.insert(0, "pipelines")

# Load the 30 benchmark questions
with open("benchmark/benchmark_questions_FINAL.json", "r", encoding="utf-8") as f:
    bm = json.load(f)
questions_data = bm["questions"] if isinstance(bm, dict) and "questions" in bm else bm
questions = [q["question"] for q in questions_data]

print(f"Loaded {len(questions)} benchmark questions")
print("=" * 60)

os.makedirs("results", exist_ok=True)

# ---- PIPELINE 1: LLM-Only ----
print("\n### PIPELINE 1: LLM-Only ###")
from pipeline_1_llm_only import LLMOnlyPipeline
p1 = LLMOnlyPipeline()
p1.run_batch(questions, save_path="results/pipeline_1_benchmark.json")

# ---- PIPELINE 2: Basic RAG ----
print("\n### PIPELINE 2: Basic RAG ###")
from pipeline_2_basic_rag import BasicRAGPipeline
p2 = BasicRAGPipeline()
p2.run_batch(questions, save_path="results/pipeline_2_benchmark.json")

# ---- PIPELINE 3: GraphRAG ----
print("\n### PIPELINE 3: GraphRAG ###")
from pipeline_3_graphrag import GraphRAGPipeline
p3 = GraphRAGPipeline(method="hybrid")
p3.run_batch(questions, save_path="results/pipeline_3_benchmark.json")

print("\n" + "=" * 60)
print("ALL 3 PIPELINES COMPLETE")
print("Results saved:")
print("  results/pipeline_1_benchmark.json")
print("  results/pipeline_2_benchmark.json")
print("  results/pipeline_3_benchmark.json")
print("=" * 60)
print("\nNext: run scoring with evaluation/evaluate.py")