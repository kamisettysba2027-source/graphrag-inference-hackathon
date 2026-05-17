import pyTigerGraph as tg

conn = tg.TigerGraphConnection(
    host="https://tg-831bc2a8-c1fa-4324-a6d2-2ecf5ee3ce74.tg-2635877100.i.tgcloud.io",
    graphname="GraphRAG",
    apiToken="REDACTED_TOKEN",
    restppPort="443",
    gsPort="443"
)

print("=" * 50)
print("PHASE 1 VERIFICATION")
print("=" * 50)

doc_count = conn.getVertexCount("Document")
content_count = conn.getVertexCount("Content")
print(f"Document vertices: {doc_count}")
print(f"Content vertices:  {content_count}")

print("\nSample 5 Document IDs:")
docs = conn.getVertices("Document", limit=5)
for d in docs:
    print(f"  - {d['v_id']}")

print("\nSample Content (first 200 chars each):")
contents = conn.getVertices("Content", limit=3)
for c in contents:
    txt = c['attributes'].get('text', '')
    print(f"  [{c['v_id']}]")
    print(f"    {txt[:200]}")

print("\n" + "=" * 50)
if doc_count == 95 and content_count == 95:
    print("PHASE 1 VERIFIED: 95 docs + 95 content, clean")
else:
    print(f"COUNT MISMATCH: docs={doc_count}, content={content_count} (may be cache lag - rerun if needed)")
print("=" * 50)