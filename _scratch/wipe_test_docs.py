import pyTigerGraph as tg

conn = tg.TigerGraphConnection(
    host="https://tg-831bc2a8-c1fa-4324-a6d2-2ecf5ee3ce74.tg-2635877100.i.tgcloud.io",
    graphname="GraphRAG",
    apiToken="REDACTED_TOKEN",
    restppPort="443",
    gsPort="443"
)

print("Before wipe:")
print(f"  Document: {conn.getVertexCount('Document')}")
print(f"  Content:  {conn.getVertexCount('Content')}")

# Delete all Document and Content vertices (clean slate for full ingest)
print("\nDeleting all Document vertices...")
print(conn.delVertices("Document"))
print("Deleting all Content vertices...")
print(conn.delVertices("Content"))

print("\nAfter wipe:")
print(f"  Document: {conn.getVertexCount('Document')}")
print(f"  Content:  {conn.getVertexCount('Content')}")
print("\nClean slate ready for 95-paper ingest")