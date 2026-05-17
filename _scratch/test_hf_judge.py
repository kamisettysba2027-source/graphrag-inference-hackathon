"""Isolated test of the HF judge call - shows the FULL error."""
from huggingface_hub import InferenceClient

HF_TOKEN = "hf_REDACTED"   # <-- paste the SAME token you put in evaluate.py
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

print("Token starts with:", HF_TOKEN[:6], "| length:", len(HF_TOKEN))

client = InferenceClient(token=HF_TOKEN)

print("\nAttempting chat_completion with", JUDGE_MODEL, "...")
try:
    resp = client.chat_completion(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly one word: PASS"}],
        max_tokens=10,
        temperature=0.0
    )
    print("SUCCESS")
    print("Response:", resp.choices[0].message.content)
except Exception as e:
    print("FULL ERROR TYPE:", type(e).__name__)
    print("FULL ERROR MESSAGE:")
    print(repr(e))
