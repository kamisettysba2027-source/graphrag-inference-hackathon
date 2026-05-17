import re

path = "pipelines/pipeline_3_graphrag.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

before = re.search(r'"num_hops":\s*\d', content)
print("Before:", before.group(0) if before else "NOT FOUND")

# Replace num_hops: 2 with num_hops: 1 (only the param line)
new_content = re.sub(r'("num_hops":\s*)2', r'\g<1>1', content)

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(new_content)

after = re.search(r'"num_hops":\s*\d', new_content)
print("After: ", after.group(0) if after else "NOT FOUND")

# Verify syntax still valid
import ast
try:
    ast.parse(new_content)
    print("SYNTAX OK")
except SyntaxError as e:
    print("SYNTAX ERROR:", e)