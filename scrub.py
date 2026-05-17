import re, glob
pats = [
    (re.compile(r'sk-proj-[A-Za-z0-9_\-]{20,}'), "sk-proj-REDACTED"),
    (re.compile(r'eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+'), "REDACTED_TOKEN"),
    (re.compile(r'REDACTED_SECRET'), "REDACTED_SECRET"),
    (re.compile(r'hf_[A-Za-z0-9]{20,}'), "hf_REDACTED"),
]
n = 0
for fp in glob.glob("**/*.py", recursive=True) + glob.glob("**/*.json", recursive=True):
    try:
        t = open(fp, encoding="utf-8").read()
    except:
        continue
    o = t
    for rx, s in pats:
        t = rx.sub(s, t)
    if t != o:
        open(fp, "w", encoding="utf-8").write(t)
        print("scrubbed", fp)
        n += 1
print(n, "files scrubbed")