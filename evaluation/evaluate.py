"""
Evaluation Framework (SPEC-COMPLIANT per official Accuracy Evaluation Guide)
  Method 1: LLM-as-a-Judge -> huggingface_hub.InferenceClient,
            meta-llama/Llama-3.1-8B-Instruct, PASS/FAIL verdict
  Method 2: BERTScore -> roberta-large F1 (unchanged, already correct)
"""
import os
import json
import time
from huggingface_hub import InferenceClient
from bert_score import score as bertscore_fn

HF_TOKEN = "hf_REDACTED"          # <-- paste your free HF token
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

hf_client = InferenceClient(token=HF_TOKEN)

JUDGE_PROMPT = """You are grading whether a pipeline's answer is correct.

You will be given a QUESTION, the correct GROUND TRUTH answer, and a PIPELINE ANSWER.

Decide if the PIPELINE ANSWER is factually correct and adequately addresses the question, based on the GROUND TRUTH. Minor wording differences are fine. The answer fails if it is wrong, irrelevant, hallucinated, contradicts the ground truth, or says it cannot answer when the ground truth has a clear answer.

Respond with EXACTLY one word: PASS or FAIL.

QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

PIPELINE ANSWER:
{pipeline_answer}

Verdict (PASS or FAIL):"""


def llm_judge(question, ground_truth, pipeline_answer):
    """Returns ('PASS'|'FAIL', raw_text)."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        pipeline_answer=pipeline_answer if pipeline_answer else "[NO ANSWER]"
    )
    try:
        resp = hf_client.chat_completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        raw = resp.choices[0].message.content.strip().upper()
        verdict = "PASS" if "PASS" in raw else "FAIL"
        return verdict, raw
    except Exception as e:
        return "ERROR", f"judge_error: {e}"


def compute_bertscore(candidates, references):
    P, R, F1 = bertscore_fn(candidates, references, lang="en", verbose=False)
    return [float(f) for f in F1]


def evaluate_pipeline(results_path, benchmark_path, out_path):
    with open(results_path, "r", encoding="utf-8") as f:
        pipeline_results = json.load(f)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        bm = json.load(f)
    benchmark = bm["questions"] if isinstance(bm, dict) and "questions" in bm else bm
    gt_by_q = {b["question"].strip(): b for b in benchmark}

    candidates, references, idx_map = [], [], []
    for i, r in enumerate(pipeline_results):
        q = r["question"].strip()
        b = gt_by_q.get(q)
        if not b:
            print(f"  WARN: no benchmark match: {q[:50]}")
            continue
        ans = r.get("answer", "") or ""
        candidates.append(ans if ans else "[NO ANSWER]")
        references.append(b["ground_truth_answer"])
        idx_map.append((i, b))

    print(f"Computing BERTScore for {len(candidates)} answers...")
    bert_f1s = compute_bertscore(candidates, references) if candidates else []

    print(f"Running LLM-Judge (Llama-3.1-8B via HF) for {len(candidates)}...")
    scored = []
    for j, (orig_i, b) in enumerate(idx_map):
        r = pipeline_results[orig_i]
        q = r["question"].strip()
        gt = b["ground_truth_answer"]
        ans = r.get("answer", "") or ""
        verdict, raw = llm_judge(q, gt, ans)
        time.sleep(1.0)  # gentle on free HF inference
        scored.append({
            "id": b.get("id", f"Q{j+1}"),
            "hop_type": b.get("hop_type", "unknown"),
            "question": q,
            "ground_truth": gt,
            "pipeline_answer": ans,
            "bertscore_f1": round(bert_f1s[j], 4),
            "llm_judge_verdict": verdict,
            "llm_judge_pass": 1 if verdict == "PASS" else 0,
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "total_tokens": r.get("total_tokens", 0),
            "latency_seconds": r.get("latency_seconds", 0),
            "cost_usd": r.get("cost_usd", 0),
        })
        print(f"  [{j+1}/{len(idx_map)}] {b.get('id','?')} ({b.get('hop_type','?')}): "
              f"BERT={bert_f1s[j]:.3f}  Judge={verdict}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scored, f, indent=2)

    if scored:
        n = len(scored)
        avg_bert = sum(s["bertscore_f1"] for s in scored) / n
        passes = sum(s["llm_judge_pass"] for s in scored)
        pass_rate = 100 * passes / n
        print("\n" + "=" * 50)
        print(f"RESULTS ({n} questions)")
        print("=" * 50)
        print(f"BERTScore F1 (avg):  {avg_bert:.4f}")
        print(f"LLM-Judge PASS rate: {pass_rate:.1f}%  ({passes}/{n})")
        print(f"Bonus bar: Judge >=90% AND BERT raw >=0.88")
        for hop in ["single", "two", "three"]:
            sub = [s for s in scored if s["hop_type"] == hop]
            if sub:
                ab = sum(s["bertscore_f1"] for s in sub) / len(sub)
                ap = 100 * sum(s["llm_judge_pass"] for s in sub) / len(sub)
                print(f"  {hop:6s} ({len(sub)}): BERT={ab:.3f}  Pass={ap:.0f}%")
    print(f"\nSaved: {out_path}")
    return scored


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    db = {"questions": [
        {"id": "T1", "hop_type": "single",
         "question": "What is metformin's primary mechanism of action?",
         "ground_truth_answer": "Metformin reduces hepatic glucose production and improves insulin sensitivity."}]}
    dr = [{"question": "What is metformin's primary mechanism of action?",
           "answer": "Metformin lowers hepatic glucose output and increases insulin sensitivity.",
           "input_tokens": 50, "output_tokens": 20, "total_tokens": 70,
           "latency_seconds": 1.2, "cost_usd": 0.0001}]
    with open("results/_db.json", "w") as f: json.dump(db, f)
    with open("results/_dr.json", "w") as f: json.dump(dr, f)
    print("SELF-TEST")
    evaluate_pipeline("results/_dr.json", "results/_db.json", "results/_ds.json")