import json
import os

SRC = "ingestion_ready"
DST = "ingestion_txt"

os.makedirs(DST, exist_ok=True)

files = [f for f in os.listdir(SRC) if f.endswith(".json")]
print(f"Found {len(files)} JSON files to convert")

converted = 0
for fname in files:
    src_path = os.path.join(SRC, fname)
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = data.get("doc_id", os.path.splitext(fname)[0])
    text = data.get("text", "")

    if not text.strip():
        print(f"  SKIP {fname} - empty text")
        continue

    # Write clean text file named by doc_id (PMC ID)
    out_path = os.path.join(DST, f"{doc_id}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    converted += 1

print(f"Converted {converted} files to {DST}/")
print(f"Sample: {os.listdir(DST)[:3]}")

# Verify one
sample = os.listdir(DST)[0]
with open(os.path.join(DST, sample), "r", encoding="utf-8") as f:
    content = f.read()
print(f"\nSample file: {sample}")
print(f"Length: {len(content)} chars")
print(f"Preview:\n{content[:300]}")