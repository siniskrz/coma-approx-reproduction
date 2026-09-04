#!/usr/bin/env python3
"""Victim-compressor pass for official-runner outputs; no backend/API calls."""
import argparse, json, os
from pathlib import Path

def source_text(x, task):
    if task == "qa": return x["context"]
    if task == "spc": return x["system_prompt"]
    return "\n\n".join(x.get(f"demo_{i}", "") for i in range(1, 6))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--task",required=True,choices=["ats","qa","spc"])
    ap.add_argument("--input",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--model-path",required=True); ap.add_argument("--rate",type=float,default=.6)
    args=ap.parse_args(); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    from llmlingua import PromptCompressor
    compressor=PromptCompressor(model_name=args.model_path, use_llmlingua2=True, device_map="cuda:0")
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    with open(args.input,encoding="utf-8") as fh, out.open("w",encoding="utf-8") as oh:
        for line in fh:
            if not line.strip(): continue
            x=json.loads(line); original=source_text(x,args.task); attacked=x.get("attacked_context",original)
            rec={"reconstruction_id":x.get("reconstruction_id"),"task":args.task,"rate":args.rate}
            variants = [("original", original), ("attacked", attacked)]
            if args.task == "spc" and x.get("target_prompt"):
                variants.append(("target", x["target_prompt"]))
            for label,text in variants:
                try:
                    r=compressor.compress_prompt(text, rate=args.rate)
                    rec[label]={"text":r.get("compressed_prompt",text),"origin_tokens":r.get("origin_tokens"),"compressed_tokens":r.get("compressed_tokens")}
                except Exception as e: rec[label]={"text":text,"error":str(e)}
            oh.write(json.dumps(rec,ensure_ascii=False)+"\n"); oh.flush(); print(rec["reconstruction_id"],flush=True)
if __name__ == "__main__": main()
