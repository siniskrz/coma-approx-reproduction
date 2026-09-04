#!/usr/bin/env python3
"""Summarize the deterministic reconstructed SPC expansion funnel."""
import argparse, json, re
from pathlib import Path

def lines(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def content(row): return row.get("result", {}).get("content", "")
def parsed(text):
    m=re.search(r"\{.*\}",text or "",re.S)
    try: return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError: return None

def normalized(text): return " ".join((text or "").lower().replace("’", "'").split())
def span_retained(text, span): return normalized(span["text"]) in normalized(text)

def main():
    ap=argparse.ArgumentParser()
    for name in ("data","generation","backend","judge","filtered","output"): ap.add_argument(f"--{name}",required=True)
    ap.add_argument("--clean-compressed")
    ap.add_argument("--annotations",default="datasets/frozen/authorization_spc_critical_spans_v2.json")
    a=ap.parse_args(); data=json.loads(Path(a.data).read_text()); gen=lines(a.generation); backend=lines(a.backend); judge=lines(a.judge); filtered=json.loads(Path(a.filtered).read_text())
    g={x["request"]["request_id"]:parsed(content(x)) for x in gen}; b={x["request"]["request_id"]:x for x in backend}; j={x["request"]["request_id"]:x for x in judge}
    orig_safe=without_violation=both=0; complete_q=complete_b=complete_j=0
    for x in data:
        rid=x["reconstruction_id"]; q=isinstance(g.get(rid),dict) and bool(g[rid].get("adversarial_query")); complete_q+=q
        o=parsed(content(j.get(f"{rid}-orig-judge",{}))); n=parsed(content(j.get(f"{rid}-nosys-judge",{})))
        complete_b += int(all(f"{rid}-{suffix}-backend" in b and bool(content(b[f"{rid}-{suffix}-backend"])) for suffix in ("orig", "nosys")))
        complete_j += int(isinstance(o,dict) and isinstance(n,dict))
        orig_safe += int(isinstance(o,dict) and o.get("violation") is False)
        without_violation += int(isinstance(n,dict) and n.get("violation") is True)
        both += int(isinstance(o,dict) and o.get("violation") is False and isinstance(n,dict) and n.get("violation") is True)
    clean_valid = None; clean_failures = None
    if a.clean_compressed:
        ann={x["sample_id"]:x["critical_spans"] for x in json.loads(Path(a.annotations).read_text())}
        clean={x.get("reconstruction_id"):x for x in lines(a.clean_compressed)}
        clean_valid=sum(bool(ann.get(x["reconstruction_id"])) and all(span_retained(clean.get(x["reconstruction_id"],{}).get("original",{}).get("text",""),s) for s in ann[x["reconstruction_id"]]) for x in filtered)
        clean_failures=len(filtered)-clean_valid
    result={"track":"released_artifact_faithful","raw_source_candidates":152,"length_and_target_attackable_candidates":138,"existing_track_a_candidates":24,"additional_candidates":len(data),"query_generated":complete_q,"backend_complete":complete_b,"judge_complete":complete_j,"passed_original_safe":orig_safe,"passed_without_system_rule_violation":without_violation,"passed_two_state_filter":both,"passed_length_limit":138,"passed_clean_compression_validity":clean_valid,"clean_compression_failures":clean_failures,"final_attack_eligible":len(filtered),"combined_valid_with_existing":3+len(filtered),"unique_system_prompts_in_combined":17,"filtered_path":a.filtered}
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
