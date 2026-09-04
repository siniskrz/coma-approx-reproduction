from llmlingua import PromptCompressor
import argparse

ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
c = PromptCompressor(model_name=a.model, use_llmlingua2=True, device_map="cuda:0")
r = c.compress_prompt("The user must confirm before sharing confidential data.", rate=0.6)
print({k: r.get(k) for k in ("origin_tokens", "compressed_tokens", "compressed_prompt")})
