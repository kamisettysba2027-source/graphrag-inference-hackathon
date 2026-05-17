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

import datetime
print(f"Progress @ {datetime.datetime.now().strftime('%H:%M:%S')}")
print("=" * 40)
for vtype in ["Document", "DocumentChunk", "Entity", "RelationshipType", "Content", "Community"]:
    try:
        print(f"  {vtype:18s}: {conn.getVertexCount(vtype)}")
    except Exception as e:
        print(f"  {vtype:18s}: {e}")
print("\nComplete when Entity, DocumentChunk, Community all > 0 and stable")