import pyTigerGraph as tg

conn = tg.TigerGraphConnection(
    host="https://tg-831bc2a8-c1fa-4324-a6d2-2ecf5ee3ce74.tg-2635877100.i.tgcloud.io",
    graphname="GraphRAG",
    apiToken="REDACTED_TOKEN",
    restppPort="443",
    gsPort="443"
)

print("Current Document vertices:")
docs = conn.getVertices("Document")
for d in docs:
    print(f"  ID: {d['v_id']}")

print(f"\nDocument count: {conn.getVertexCount('Document')}")
print(f"Content count: {conn.getVertexCount('Content')}")

# Try explicit GSQL delete instead
print("\nAttempting GSQL-based delete...")
res = conn.gsql("""USE GRAPH GraphRAG
DELETE FROM Document
DELETE FROM Content""")
print(res)

print("\nAfter GSQL delete:")
print(f"Document count: {conn.getVertexCount('Document')}")
print(f"Content count: {conn.getVertexCount('Content')}")