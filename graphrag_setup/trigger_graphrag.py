import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "http://localhost:8000"
GRAPHNAME = "GraphRAG"
USERNAME = "tigergraph"
TOKEN = "REDACTED_TOKEN"

auth = HTTPBasicAuth(USERNAME, TOKEN)

print("Triggering GraphRAG extraction (Phase 2)...")
print("This starts background entity/relationship/community extraction.")
print("=" * 60)

# forceupdate with method=graphrag triggers full GraphRAG build
r = requests.get(
    f"{BASE_URL}/{GRAPHNAME}/graphrag/forceupdate",
    auth=auth,
    headers={"accept": "application/json"},
    timeout=120
)

print(f"Status: {r.status_code}")
print(f"Response: {r.text}")

if r.status_code == 200:
    print("\n" + "=" * 60)
    print("PHASE 2 TRIGGERED - extraction running in background")
    print("Monitor progress with: docker logs graphrag-ecc --tail 30")
    print("=" * 60)
else:
    print("\nTrigger failed - share this output")