import re
t1 = re.search(r'apiToken="([^"]+)"', open("verify_graphrag_complete.py").read()).group(1)
t2 = re.search(r'TOKEN = "([^"]+)"', open("tune_methods.py").read()).group(1)
print("len verify:", len(t1), "| len tune:", len(t2))
print("MATCH" if t1 == t2 else "MISMATCH")