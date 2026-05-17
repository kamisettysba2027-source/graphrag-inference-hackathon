import requests
from requests.auth import HTTPBasicAuth
import json
import time

BASE_URL = "http://localhost:8000"
GRAPHNAME = "GraphRAG"
USERNAME = "tigergraph"
TOKEN = "REDACTED_TOKEN"

auth = HTTPBasicAuth(USERNAME, TOKEN)
headers = {"accept": "application/json", "Content-Type": "application/json"}

print("=" * 60)
print("PHASE 1: FULL INGESTION - 95 documents")
print("=" * 60)

# STEP 1: create_ingest
print("\n[1/2] create_ingest (processing 95 files to JSONL)...")
t0 = time.time()

create_payload = {
    "data_source": "server",
    "data_source_config": {"data_path": "/code/ingestion_full"},
    "file_format": "multi"
}

r1 = requests.post(
    f"{BASE_URL}/{GRAPHNAME}/supportai/create_ingest",
    auth=auth, headers=headers,
    data=json.dumps(create_payload),
    timeout=1800
)

print(f"Status: {r1.status_code} | Time: {time.time()-t0:.1f}s")
if r1.status_code != 200:
    print("FAILED:", r1.text)
    raise SystemExit(1)

create_result = r1.json()
print(json.dumps(create_result, indent=2))

# STEP 2: ingest
print("\n[2/2] ingest (loading documents into graph)...")
t0 = time.time()

ingest_payload = {
    "load_job_id": create_result["load_job_id"],
    "data_source_id": create_result["data_source_id"],
    "file_path": create_result.get("data_path", "in_temp_storage")
}

r2 = requests.post(
    f"{BASE_URL}/{GRAPHNAME}/supportai/ingest",
    auth=auth, headers=headers,
    data=json.dumps(ingest_payload),
    timeout=1800
)

print(f"Status: {r2.status_code} | Time: {time.time()-t0:.1f}s")
if r2.status_code != 200:
    print("FAILED:", r2.text)
    raise SystemExit(1)

result = r2.json()
print(f"\nSummary: {result.get('summary')}")
print(f"Document count: {result.get('document_count')}")

failed = [f for f in result.get('ingested_files', []) if f.get('status') != 'success']
if failed:
    print(f"\nWARNING: {len(failed)} files failed:")
    for f in failed:
        print(f"  - {f.get('jsonl_file')}: {f.get('error')}")
else:
    print("\nAll files ingested successfully - 0 failures")

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE")
print("=" * 60)