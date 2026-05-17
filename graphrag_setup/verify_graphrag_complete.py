import pyTigerGraph as tg

conn = tg.TigerGraphConnection(
    host="https://tg-831bc2a8-c1fa-4324-a6d2-2ecf5ee3ce74.tg-2635877100.i.tgcloud.io",
    graphname="GraphRAG",
    apiToken="REDACTED_TOKEN",
    restppPort="443",
    gsPort="443"
)

print("=" * 55)
print("GRAPHRAG COMPLETION VERIFICATION")
print("=" * 55)

print("\nVertex counts:")
for v in ["Document", "DocumentChunk", "Content", "Entity", "Community"]:
    print(f"  {v:15s}: {conn.getVertexCount(v)}")

print("\nEdge counts (the knowledge graph connections):")
for e in ["HAS_CHILD", "CONTAINS_ENTITY", "RELATIONSHIP", "IN_COMMUNITY", "HAS_CONTENT"]:
    try:
        print(f"  {e:18s}: {conn.getEdgeCount(e)}")
    except Exception as ex:
        print(f"  {e:18s}: {ex}")

print("\nSample entities (first 15):")
ents = conn.getVertices("Entity", limit=15)
for e in ents:
    print(f"  - {e['v_id']}  (type: {e['attributes'].get('entity_type','?')})")

print("\nSample communities (first 5, with descriptions):")
comms = conn.getVertices("Community", limit=5)
for c in comms:
    desc = c['attributes'].get('description', '')[:150]
    print(f"  [{c['v_id']}] {desc}")

print("\n" + "=" * 55)
print("If RELATIONSHIP edges > 0 and communities have descriptions,")
print("GraphRAG is fully built and query-ready.")
print("=" * 55)