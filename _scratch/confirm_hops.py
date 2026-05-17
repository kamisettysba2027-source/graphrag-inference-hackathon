import re
c = open("pipelines/pipeline_3_graphrag.py").read()
m = re.search(r'"num_hops":\s*(\d)', c)
mth = re.search(r'GraphRAGPipeline\(method="(\w+)"\)', c)
print("num_hops:", m.group(1) if m else "NOT FOUND")
print("method:", mth.group(1) if mth else "check default in __init__")