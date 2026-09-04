#!/usr/bin/env python3
"""Experimental Track B scaffold: freeze system text and record query-only suffixes."""
import argparse, json
from pathlib import Path

def records(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",required=True); ap.add_argument("--output",required=True); ap.add_argument("--suffix",default="")
    a=ap.parse_args(); out=[]
    for x in records(a.data):
        query=x.get("adversarial_query","")
        out.append({"sample_id":x["reconstruction_id"],"track":"paper_threat_model_faithful_experimental_scaffold",
                    "system_prompt_frozen":x["system_prompt"],"original_query":query,
                    "adversarial_suffix":a.suffix,"attacked_query":query+a.suffix,
                    "system_prompt_optimized":False,
                    "status":"SCAFFOLD_ONLY_ALGORITHM_DETAILS_UNRESOLVED"})
    p=Path(a.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"rows":len(out),"output":str(p),"system_prompts_frozen":all(x["system_prompt_optimized"] is False for x in out)},indent=2))
if __name__=="__main__": main()
