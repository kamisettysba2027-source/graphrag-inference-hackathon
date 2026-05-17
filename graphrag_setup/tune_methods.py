"""
Compare GraphRAG retrieval METHODS on benchmark questions.
Each method gets its correct params (per supportai.py source).
Tests across all 3 hop types to see which method wins which question type.
"""
import json, time, requests
from requests.auth import HTTPBasicAuth
import tiktoken
from bert_score import score as bertscore_fn

BASE_URL = "http://localhost:8000"
GRAPHNAME = "GraphRAG"
USERNAME = "tigergraph"
TOKEN = "REDACTED_TOKEN"

auth = HTTPBasicAuth(USERNAME, TOKEN)
headers = {"accept": "application/json", "Content-Type": "application/json"}
enc = tiktoken.get_encoding("cl100k_base")

with open("benchmark/benchmark_questions_FINAL.json", "r", encoding="utf-8") as f:
    bm = json.load(f)
questions = bm["questions"] if isinstance(bm, dict) and "questions" in bm else bm

# 6 tuning questions: 2 per hop type
tuning = []
for hop in ["single", "two", "three"]:
    tuning.extend([q for q in questions if str(q.get("hop_type","")).lower()==hop][:2])

# Each method with its CORRECT params (from supportai.py source)
METHODS = {
    "hybrid_hops1": {
        "method": "hybrid",
        "method_params": {"indices": ["DocumentChunk"], "top_k": 5, "num_hops": 1,
                          "num_seen_min": 1, "similarity_threshold": 0.90,
                          "chunk_only": False, "doc_only": False,
                          "expand": False, "combine": False, "verbose": False}
    },
    "similarity": {
        "method": "similarity",
        "method_params": {"index": "DocumentChunk", "top_k": 5,
                          "withHyDE": False, "expand": False, "verbose": False}
    },
    "community": {
        "method": "community",
        "method_params": {"community_level": 2, "top_k": 5,
                          "similarity_threshold": 0.90, "expand": False,
                          "with_chunk": True, "with_doc": False,
                          "combine": False, "verbose": False}
    },
    "entityrelationship": {
        "method": "entityrelationship",
        "method_params": {"top_k": 5}
    },
}

def query(question, method, mp):
    payload = {"question": question, "method": method, "method_params": mp}
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/{GRAPHNAME}/graphrag/answerquestion",
                          auth=auth, headers=headers, data=json.dumps(payload), timeout=180)
        lat = time.time()-t0
        if r.status_code == 200:
            resp = r.json()
            return (resp.get("response") or resp.get("natural_language_response") or ""), lat, None
        return "", lat, f"HTTP {r.status_code}: {r.text[:150]}"
    except Exception as e:
        return "", time.time()-t0, str(e)

all_results = {}
for mname, cfg in METHODS.items():
    print(f"\n{'='*55}\nMETHOD: {mname}\n{'='*55}")
    per_q = []
    answers, refs = [], []
    for q in tuning:
        ans, lat, err = query(q["question"], cfg["method"], cfg["method_params"])
        tok = len(enc.encode(ans or ""))
        if err:
            print(f"  {q.get('id','?')} ({q['hop_type']}): ERROR {err}")
            answers.append("[NO ANSWER]")
        else:
            print(f"  {q.get('id','?')} ({q['hop_type']}): {tok} tok, {lat:.1f}s")
            answers.append(ans if ans else "[NO ANSWER]")
        refs.append(q["ground_truth_answer"])
        per_q.append({"id": q.get("id"), "hop": q["hop_type"], "tok": tok,
                      "lat": lat, "err": bool(err)})
        time.sleep(0.5)
    P,R,F1 = bertscore_fn(answers, refs, lang="en", verbose=False)
    for i, f in enumerate(F1):
        per_q[i]["bert"] = float(f)
    all_results[mname] = per_q

# Overall comparison
print(f"\n{'='*55}\nOVERALL (avg across 6 questions)\n{'='*55}")
print(f"{'Method':22s} {'BERT':>8s} {'Tokens':>8s} {'Latency':>8s} {'Err':>4s}")
for m, rows in all_results.items():
    valid = [r for r in rows if not r["err"]]
    n = len(rows)
    avg_b = sum(r.get("bert",0) for r in rows)/n
    avg_t = sum(r["tok"] for r in rows)/n
    avg_l = sum(r["lat"] for r in rows)/n
    errs = sum(1 for r in rows if r["err"])
    print(f"{m:22s} {avg_b:>8.4f} {avg_t:>8.0f} {avg_l:>7.1f}s {errs:>4d}")

# Per-hop-type breakdown (the key insight)
print(f"\n{'='*55}\nBY HOP TYPE (which method wins which question type)\n{'='*55}")
for hop in ["single", "two", "three"]:
    print(f"\n{hop.upper()}-HOP:")
    for m, rows in all_results.items():
        sub = [r for r in rows if r["hop"] == hop and not r["err"]]
        if sub:
            ab = sum(r["bert"] for r in sub)/len(sub)
            at = sum(r["tok"] for r in sub)/len(sub)
            print(f"  {m:22s} BERT={ab:.4f}  tokens={at:.0f}")
        else:
            errs = sum(1 for r in rows if r["hop"]==hop and r["err"])
            print(f"  {m:22s} (all {errs} errored)")

print("\nINTERPRETATION:")
print("- If similarity wins single-hop (fact-lookup) -> use it for those")
print("- If hybrid wins multi-hop -> use it for those")
print("- This per-type finding is a strong hackathon result")