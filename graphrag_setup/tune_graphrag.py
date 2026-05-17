"""
A/B test GraphRAG retrieval configs on 6 representative benchmark questions.
Picks the config with the best accuracy/token trade-off.
"""
import json
import time
import requests
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

# Load benchmark, pick 2 of each hop type
with open("benchmark/benchmark_questions_FINAL.json", "r", encoding="utf-8") as f:
    bm = json.load(f)
questions = bm["questions"] if isinstance(bm, dict) and "questions" in bm else bm

tuning_set = []
for hop in ["single", "two", "three"]:
    matches = [q for q in questions if str(q.get("hop_type","")).lower() == hop][:2]
    tuning_set.extend(matches)

print(f"Tuning on {len(tuning_set)} questions (2 single, 2 two, 2 three)")

# The 3 configs to test
CONFIGS = {
    "hybrid_default": {
        "method": "hybrid",
        "method_params": {"indices": ["DocumentChunk"], "top_k": 5, "num_hops": 2,
                          "num_seen_min": 1, "similarity_threshold": 0.90,
                          "chunk_only": False, "doc_only": False,
                          "expand": False, "combine": False, "verbose": False}
    },
    "hybrid_morechunks": {
        "method": "hybrid",
        "method_params": {"indices": ["DocumentChunk"], "top_k": 8, "num_hops": 2,
                          "num_seen_min": 1, "similarity_threshold": 0.80,
                          "chunk_only": True, "doc_only": False,
                          "expand": True, "combine": True, "verbose": False}
    },
    "community": {
        "method": "community",
        "method_params": {"community_level": 2, "top_k": 5,
                          "similarity_threshold": 0.90, "expand": False,
                          "with_chunk": True, "with_doc": False,
                          "combine": False, "verbose": False}
    }
}


def query(question, method, method_params):
    payload = {"question": question, "method": method, "method_params": method_params}
    t0 = time.time()
    try:
        r = requests.post(f"{BASE_URL}/{GRAPHNAME}/graphrag/answerquestion",
                          auth=auth, headers=headers, data=json.dumps(payload), timeout=180)
        lat = time.time() - t0
        if r.status_code == 200:
            resp = r.json()
            ans = resp.get("response") or resp.get("natural_language_response") or ""
            return ans, lat, None
        return "", lat, f"HTTP {r.status_code}: {r.text[:150]}"
    except Exception as e:
        return "", time.time()-t0, str(e)


results = {}
for cfg_name, cfg in CONFIGS.items():
    print(f"\n{'='*55}\nCONFIG: {cfg_name}\n{'='*55}")
    answers, refs, toks, lats = [], [], [], []
    errors = 0
    for q in tuning_set:
        ans, lat, err = query(q["question"], cfg["method"], cfg["method_params"])
        if err:
            print(f"  {q.get('id','?')}: ERROR {err}")
            errors += 1
            answers.append("[NO ANSWER]")
        else:
            print(f"  {q.get('id','?')} ({q['hop_type']}): {len(enc.encode(ans))} tok, {lat:.1f}s")
            answers.append(ans if ans else "[NO ANSWER]")
        refs.append(q["ground_truth_answer"])
        toks.append(len(enc.encode(ans or "")))
        lats.append(lat)
        time.sleep(0.5)

    # Score with BERTScore
    P, R, F1 = bertscore_fn(answers, refs, lang="en", verbose=False)
    avg_bert = sum(float(f) for f in F1) / len(F1)
    avg_tok = sum(toks) / len(toks)
    avg_lat = sum(lats) / len(lats)

    results[cfg_name] = {"bert": avg_bert, "tokens": avg_tok, "latency": avg_lat, "errors": errors}
    print(f"\n  AVG BERTScore: {avg_bert:.4f} | AVG tokens: {avg_tok:.0f} | "
          f"AVG latency: {avg_lat:.1f}s | errors: {errors}")

print(f"\n{'='*55}\nFINAL COMPARISON\n{'='*55}")
print(f"{'Config':22s} {'BERTScore':>10s} {'AvgTokens':>10s} {'Latency':>9s} {'Err':>4s}")
for name, r in results.items():
    print(f"{name:22s} {r['bert']:>10.4f} {r['tokens']:>10.0f} {r['latency']:>8.1f}s {r['errors']:>4d}")
print("\nPick the config with highest BERTScore and lowest errors.")
print("If tie on accuracy, prefer lower tokens (token reduction = 30% of judging).")