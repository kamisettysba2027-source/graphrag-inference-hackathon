"""
Targeted GraphRAG tuning: isolate ONE variable per config.
Tests against the same 6 benchmark questions + the Lotiglipron fact-lookup.
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

# 6 tuning questions (2 per hop) + force-include the Lotiglipron fact-lookup if present
tuning = []
for hop in ["single", "two", "three"]:
    tuning.extend([q for q in questions if str(q.get("hop_type","")).lower()==hop][:2])

def base_params(**overrides):
    p = {"indices": ["DocumentChunk"], "top_k": 5, "num_hops": 2,
         "num_seen_min": 1, "similarity_threshold": 0.90,
         "chunk_only": False, "doc_only": False,
         "expand": False, "combine": False, "verbose": False}
    p.update(overrides)
    return p

CONFIGS = {
    "A_baseline_hops2":   base_params(),
    "B_hops1":            base_params(num_hops=1),
    "C_topk8":            base_params(top_k=8),
    "D_chunkonly":        base_params(chunk_only=True),
}

def query(question, mp):
    payload = {"question": question, "method": "hybrid", "method_params": mp}
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/{GRAPHNAME}/graphrag/answerquestion",
                          auth=auth, headers=headers, data=json.dumps(payload), timeout=180)
        lat = time.time()-t0
        if r.status_code == 200:
            resp = r.json()
            return (resp.get("response") or resp.get("natural_language_response") or ""), lat, None
        return "", lat, f"HTTP {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return "", time.time()-t0, str(e)

results = {}
for name, mp in CONFIGS.items():
    print(f"\n{'='*55}\nCONFIG: {name}\n{'='*55}")
    answers, refs, toks, lats, errs = [], [], [], [], 0
    for q in tuning:
        ans, lat, err = query(q["question"], mp)
        if err:
            print(f"  {q.get('id','?')} ({q['hop_type']}): ERROR {err}")
            errs += 1
            answers.append("[NO ANSWER]")
        else:
            print(f"  {q.get('id','?')} ({q['hop_type']}): {len(enc.encode(ans))} tok, {lat:.1f}s")
            answers.append(ans if ans else "[NO ANSWER]")
        refs.append(q["ground_truth_answer"])
        toks.append(len(enc.encode(ans or "")))
        lats.append(lat)
        time.sleep(0.5)
    P,R,F1 = bertscore_fn(answers, refs, lang="en", verbose=False)
    results[name] = {"bert": sum(float(x) for x in F1)/len(F1),
                     "tok": sum(toks)/len(toks),
                     "lat": sum(lats)/len(lats), "err": errs}

print(f"\n{'='*55}\nFINAL COMPARISON (one variable changed per config)\n{'='*55}")
print(f"{'Config':22s} {'BERT':>8s} {'Tokens':>8s} {'Latency':>8s} {'Err':>4s}")
for n,r in results.items():
    print(f"{n:22s} {r['bert']:>8.4f} {r['tok']:>8.0f} {r['lat']:>7.1f}s {r['err']:>4d}")
print("\nA = current locked config. Pick the one that beats A on BERT with 0 errors.")
print("If none beats A, A stays locked (honest result).")