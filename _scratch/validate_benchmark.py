import json
import os

# Auto-find the json file in benchmark/
bench_dir = "benchmark"
json_files = [f for f in os.listdir(bench_dir) if f.endswith(".json")]
print(f"JSON files found in benchmark/: {json_files}")

if not json_files:
    print("ERROR: no .json file in benchmark/ folder")
    raise SystemExit(1)

# Use the first one (or the one matching 'benchmark')
fname = json_files[0]
for f in json_files:
    if "benchmark" in f.lower() or "question" in f.lower():
        fname = f
        break

path = os.path.join(bench_dir, fname)
print(f"\nValidating: {path}")
print("=" * 55)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Handle both {questions:[...]} and bare [...]
if isinstance(data, dict) and "questions" in data:
    questions = data["questions"]
elif isinstance(data, list):
    questions = data
else:
    print(f"ERROR: unexpected structure. Top-level keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    raise SystemExit(1)

print(f"Total questions: {len(questions)}")

# Validate each
required = ["question", "hop_type", "ground_truth_answer"]
issues = []
hop_counts = {"single": 0, "two": 0, "three": 0, "other": 0}
empty_answers = 0
no_source = 0

for i, q in enumerate(questions):
    qid = q.get("id", f"#{i+1}")

    for field in required:
        if field not in q or not str(q.get(field, "")).strip():
            issues.append(f"  {qid}: missing/empty '{field}'")

    hop = str(q.get("hop_type", "")).lower().strip()
    if hop in hop_counts:
        hop_counts[hop] += 1
    else:
        hop_counts["other"] += 1
        issues.append(f"  {qid}: invalid hop_type '{q.get('hop_type')}' (must be single/two/three)")

    ans = str(q.get("ground_truth_answer", "")).strip()
    if not ans or ans.upper().startswith("EXAMPLE"):
        empty_answers += 1
        issues.append(f"  {qid}: ground_truth_answer empty or still EXAMPLE placeholder")

    src = q.get("source_pmc_ids", [])
    if not src:
        no_source += 1

print(f"\nHop type distribution:")
for h, c in hop_counts.items():
    print(f"  {h:8s}: {c}")

print(f"\nEmpty/placeholder answers: {empty_answers}")
print(f"Questions without source_pmc_ids: {no_source}")

if issues:
    print(f"\n!!! {len(issues)} ISSUES FOUND:")
    for iss in issues[:40]:
        print(iss)
    if len(issues) > 40:
        print(f"  ... and {len(issues)-40} more")
else:
    print("\nNO ISSUES - benchmark is clean and ready")

# Show a sample
print("\n" + "=" * 55)
print("SAMPLE (first question of each hop type):")
for hop in ["single", "two", "three"]:
    for q in questions:
        if str(q.get("hop_type","")).lower().strip() == hop:
            print(f"\n[{hop}] {q.get('id','?')}")
            print(f"  Q: {q.get('question','')[:120]}")
            print(f"  A: {q.get('ground_truth_answer','')[:150]}")
            print(f"  Src: {q.get('source_pmc_ids', [])}")
            break