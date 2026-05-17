"""
Controlled probe: make 12 rapid HF judge calls, print FULL error on any failure.
Tells us EXACTLY what kind of limit we're hitting (429 rate vs 402 quota vs other).
"""
import time
from huggingface_hub import InferenceClient

HF_TOKEN = "hf_REDACTED"   # <-- same token as evaluate.py
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

client = InferenceClient(token=HF_TOKEN)

print(f"Token: {HF_TOKEN[:6]}... len={len(HF_TOKEN)}")
print(f"Model: {JUDGE_MODEL}")
print("Making 12 rapid calls (1s apart) to trigger/observe the limit...\n")

ok = 0
for i in range(1, 13):
    try:
        resp = client.chat_completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": f"Reply with one word: TEST{i}"}],
            max_tokens=5,
            temperature=0.0
        )
        txt = resp.choices[0].message.content.strip()
        print(f"  [{i:2d}] OK -> {txt!r}")
        ok += 1
    except Exception as e:
        print(f"  [{i:2d}] ERROR")
        print(f"       TYPE: {type(e).__name__}")
        print(f"       MSG : {str(e)[:400]}")
        # check for HTTP status code
        code = getattr(getattr(e, "response", None), "status_code", None)
        print(f"       HTTP STATUS: {code}")
    time.sleep(1)

print(f"\n{ok}/12 succeeded")
print("If failures show HTTP 429 -> rate limit (fixable: slower pacing + retry)")
print("If HTTP 402 -> monthly credit exhausted (need reset or small HF credit)")
print("If 503/model loading -> transient (fixable: retry with wait)")