#!/usr/bin/env python3
"""Small OpenAI-compatible batch client; credentials are read only from env."""
import argparse, json, os, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def call(item, key):
    body = {"model": item["model"], "messages": item["messages"],
            "temperature": item.get("temperature", 0),
            "max_tokens": item.get("max_tokens", 512)}
    if item.get("top_p") is not None: body["top_p"] = item["top_p"]
    if item.get("seed") is not None: body["seed"] = item["seed"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False,
                                     dir=Path(item.get("temp_dir", ".")).resolve()) as fh:
        json.dump(body, fh, ensure_ascii=False); body_path = fh.name
    cfg = ('header = "Content-Type: application/json"\n'
           f'header = "Authorization: Bearer {key}"\n')
    try:
        p = subprocess.run(["curl", "-sS", "--max-time", str(item.get("timeout", 120)),
                            "--config", "-", "--data-binary", f"@{body_path}",
                            f'{item.get("base_url", "https://api.ofox.ai/v1")}/chat/completions'],
                           input=cfg, text=True, capture_output=True)
    finally:
        try: os.unlink(body_path)
        except OSError: pass
    if p.returncode: raise RuntimeError(p.stderr.strip() or f"curl exit {p.returncode}")
    try: obj = json.loads(p.stdout)
    except json.JSONDecodeError as e: raise RuntimeError(f"non-json response: {p.stdout[:300]!r}") from e
    if obj.get("error"): raise RuntimeError(json.dumps(obj["error"], ensure_ascii=False))
    choice = (obj.get("choices") or [{}])[0]
    return {"request_id": item.get("request_id"), "model": obj.get("model", item["model"]),
            "response": obj, "content": choice.get("message", {}).get("content", "")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--credential-env", default="OPENAI_API_KEY"); ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(); key = os.environ.get(args.credential_env)
    if not key: raise SystemExit(f"missing credential env {args.credential_env}")
    requests = [json.loads(x) for x in Path(args.input).read_text().splitlines() if x.strip()]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    def run(item):
        rec = {"request": item}
        try: rec["result"] = call(item, key)
        except Exception as e: rec["error"] = str(e)
        return rec
    if args.workers == 1:
        done = set()
        if args.resume and out.exists():
            for line in out.read_text(encoding="utf-8").splitlines():
                try: done.add(json.loads(line)["request"]["request_id"])
                except (ValueError, KeyError, TypeError): pass
        todo = [x for x in requests if x.get("request_id") not in done]
        mode = "a" if args.resume else "w"
        with out.open(mode, encoding="utf-8") as fh:
            for n, item in enumerate(todo, 1):
                rec = run(item); fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                print(f"{n}/{len(todo)} {item.get('request_id')} {'ok' if 'result' in rec else 'ERROR'}", flush=True)
                if n < len(todo): time.sleep(args.sleep)
        return
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex: records = list(ex.map(run, requests))
    else:
        records = []
        for n, item in enumerate(requests, 1):
            records.append(run(item)); print(f"{n}/{len(requests)} {item.get('request_id')} {'ok' if 'result' in records[-1] else 'ERROR'}", flush=True)
            if n < len(requests): time.sleep(args.sleep)
    with out.open("w", encoding="utf-8") as fh:
        for n, rec in enumerate(records, 1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{n}/{len(records)} {rec['request'].get('request_id')} {'ok' if 'result' in rec else 'ERROR'}", flush=True)

if __name__ == "__main__": main()
