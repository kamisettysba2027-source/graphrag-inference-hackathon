import requests
from requests.auth import HTTPBasicAuth

BASE_URL = "http://localhost:8000"
GRAPHNAME = "GraphRAG"
USERNAME = "tigergraph"
TOKEN = "REDACTED_TOKEN"

print("Calling initialize with explicit Basic auth header...")
print("=" * 60)

response = requests.post(
    f"{BASE_URL}/{GRAPHNAME}/graphrag/initialize",
    auth=HTTPBasicAuth(USERNAME, TOKEN),
    headers={"accept": "application/json"},
    timeout=300
)

print(f"Status Code: {response.status_code}")
print(f"Response Body:")
print(response.text)