import json, re

print("="*60)
print("1. What num_hops is pipeline_3 ACTUALLY set to right now?")
print("="*60)
src = open("pipelines/pipeline_3_graphrag.py").read()
m = re.search(r'"num_hops":\s*(\d)', src)
print("   pipeline_3_graphrag.py num_hops =", m.group(1) if m else "NOT FOUND")
mm = re.search(r'"method":\s*"(\w+)"', src) or re.search(r'method="(\w+)"', src)
print("   method =", mm.group(1) if mm else "check file")

print("\n" + "="*60)
print("2. Compare GraphRAG answers: tuning Qs vs benchmark Qs")
print("="*60)
# benchmark scored results
with open("results/pipeline_3_scored.json") as f:
    p3 = json.load(f)

# the 6 tuning question ids
tuning_ids = ["Q01","Q02","Q11","Q12","Q21","Q22"]
print("\nBERTScore on the SAME 6 questions used in tuning (from full run):")
sub = [d for d in p3 if d.get("id") in tuning_ids]
for d in sub:
    print(f"   {d['id']} ({d['hop_type']}): BERT={d['bertscore_f1']:.3f}")
if sub:
    avg6 = sum(d['bertscore_f1'] for d in sub)/len(sub)
    print(f"   --> avg over these 6 in FULL run: {avg6:.4f}")
    print(f"   (tuning runs showed ~0.870-0.872 on these same 6)")

avg30 = sum(d['bertscore_f1'] for d in p3)/len(p3)
print(f"\n   avg over ALL 30: {avg30:.4f}")
print(f"   avg over other 24 (non-tuning):", end=" ")
other = [d for d in p3 if d.get("id") not in tuning_ids]
print(f"{sum(d['bertscore_f1'] for d in other)/len(other):.4f}")

print("\n" + "="*60)
print("3. Sample 2 GraphRAG benchmark answers (are they real/non-empty?)")
print("="*60)
for d in p3[:2]:
    print(f"\n   {d['id']}: {d['question'][:70]}")
    print(f"   ANSWER: {d['pipeline_answer'][:300]}")
    print(f"   BERT={d['bertscore_f1']:.3f} Judge={d.get('llm_judge_verdict')}")