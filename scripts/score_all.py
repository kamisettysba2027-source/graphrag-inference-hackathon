"""Score all 3 benchmark result files against ground truth."""
import sys
sys.path.insert(0, "evaluation")
from evaluate import evaluate_pipeline

BENCH = "benchmark/benchmark_questions_FINAL.json"

for n in [1, 2, 3]:
    print(f"\n{'#'*60}\n# SCORING PIPELINE {n}\n{'#'*60}")
    evaluate_pipeline(
        f"results/pipeline_{n}_benchmark.json",
        BENCH,
        f"results/pipeline_{n}_scored.json"
    )