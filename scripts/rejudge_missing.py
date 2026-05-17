"""
Re-judge ONLY the answers whose verdict is ERROR. Keep valid PASS/FAIL as-is.
Gentle pacing + retry. Writes after every call (crash-safe / resumable).
"""
import json, time
from huggingface_hub import InferenceClient

HF_TOKEN = "hf_REDACTED"
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
client = InferenceClient(token=HF_TOKEN)

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


def judge_with_retry(q, gt, ans, max_retries=3):
    prompt = JUDGE_PROMPT.format(question=q, ground_truth=gt,
                                 pipeline_answer=ans if ans else "[NO ANSWER]")
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat_completion(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10, temperature=0.0
            )
            raw = resp.choices[0].message.content.strip().upper()
            return "PASS" if "PASS" in raw else "FAIL"
        except Exception as e:
            wait = 5 * attempt
            print(f"      retry {attempt}/{max_retries} after error ({str(e)[:60]}) - wait {wait}s")
            time.sleep(wait)
    return "ERROR"


for n in [1, 2, 3]:
    path = f"results/pipeline_{n}_scored.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [d for d in data if d.get("llm_judge_verdict") not in ("PASS", "FAIL")]
    print(f"\n{'='*55}\nPIPELINE {n}: {len(missing)} answers to re-judge\n{'='*55}")

    for d in data:
        if d.get("llm_judge_verdict") in ("PASS", "FAIL"):
            continue  # keep valid verdict
        v = judge_with_retry(d["question"], d["ground_truth"], d.get("pipeline_answer", ""))
        d["llm_judge_verdict"] = v
        d["llm_judge_pass"] = 1 if v == "PASS" else 0
        print(f"  {d.get('id','?')} ({d.get('hop_type','?')}): {v}")
        # write after each (crash-safe)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        time.sleep(4)  # gentle pacing

    # recompute summary
    valid = [d for d in data if d.get("llm_judge_verdict") in ("PASS", "FAIL")]
    passes = sum(d["llm_judge_pass"] for d in data)
    avg_bert = sum(d["bertscore_f1"] for d in data) / len(data)
    print(f"\n  PIPELINE {n} FINAL:")
    print(f"    Valid verdicts: {len(valid)}/30")
    print(f"    PASS rate: {100*passes/len(data):.1f}%  ({passes}/30)")
    print(f"    BERTScore F1: {avg_bert:.4f}")
    for hop in ["single", "two", "three"]:
        sub = [d for d in data if d.get("hop_type") == hop]
        sp = sum(d["llm_judge_pass"] for d in sub)
        sb = sum(d["bertscore_f1"] for d in sub)/len(sub)
        print(f"      {hop:6s}: Pass={100*sp/len(sub):.0f}%  BERT={sb:.3f}")

print("\nAll pipelines re-judged. Scored files updated in place.")