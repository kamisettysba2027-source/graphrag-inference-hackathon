import json, sys
sys.path.insert(0, "pipelines")
from pipeline_1_llm_only import LLMOnlyPipeline

with open("benchmark/benchmark_questions_FINAL.json", "r", encoding="utf-8") as f:
    bm = json.load(f)
qs = bm["questions"] if isinstance(bm, dict) and "questions" in bm else bm
questions = [q["question"] for q in qs]

print(f"Re-running Pipeline 1 on {len(questions)} questions (OpenAI)")
p1 = LLMOnlyPipeline()
p1.run_batch(questions, save_path="results/pipeline_1_benchmark.json")