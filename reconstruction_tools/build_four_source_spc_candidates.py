#!/usr/bin/env python3
"""Build a balanced, provenance-bound SPC candidate pool from the paper's four repos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_construction.extract_guardrails import extract_guardrail_pairs, read_prompt_from_file
from reconstruction_tools.fetch_public_assets import SOURCES


TEXT_EXTENSIONS = {".md", ".txt", ".json"}
SKIP_NAMES = {"readme.md", "readme.txt", "license", "license.md", "copying"}
LOW_RISK_TERMS = {
    "answer", "citation", "code", "emoji", "format", "instruction", "json", "language",
    "link", "markdown", "mention", "output", "prompt", "question", "quote", "repeat",
    "reveal", "response", "source", "system", "text", "word",
}
EXCLUDED_TERMS = {
    "biological", "bomb", "child sexual", "credential", "exploit", "hatred", "malware",
    "medical", "nuclear", "password", "race", "ransomware", "religion", "self-harm",
    "suicide", "terror", "token", "weapon",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def collect(repo: Path) -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(repo.rglob("*")):
        if (not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_EXTENSIONS
                or path.name.lower() in SKIP_NAMES or path.stat().st_size > 100_000):
            continue
        prompt = read_prompt_from_file(path)
        if not prompt:
            continue
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for guardrail in extract_guardrail_pairs(prompt):
            sentence = " ".join(guardrail["sentence"].split())
            lowered = sentence.casefold()
            normalized = "".join(character for character in lowered if character.isalnum())
            if (not LOW_RISK_TERMS.intersection(lowered.split())
                    or any(term in lowered for term in EXCLUDED_TERMS)
                    or normalized in seen):
                continue
            seen.add(normalized)
            rows.append({
                "source_path": path.relative_to(repo).as_posix(),
                "source_file_sha256": file_hash,
                "source_sentence": sentence,
                "keyword": guardrail["keyword"],
                "selection_score": 10 * len(guardrail["keyword"].split())
                + 3 * sum(term in lowered for term in LOW_RISK_TERMS)
                - len(sentence) / 400,
            })
    return sorted(rows, key=lambda row: (-row["selection_score"], row["source_path"], row["source_sentence"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-source", type=int, default=8)
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error("--per-source must be positive")

    source_root = Path(args.source_root).resolve()
    asset_manifest = source_root.parent / "public_assets_manifest.json"
    if not asset_manifest.is_file():
        raise ValueError("pinned public asset manifest is required")
    raw_manifest = asset_manifest.read_bytes()
    document = json.loads(raw_manifest)
    source_records = document.get("sources") if isinstance(document, dict) else None
    if not isinstance(source_records, list) or len(source_records) != len(SOURCES):
        raise ValueError("public asset manifest must contain exactly four sources")
    by_name = {row.get("name"): row for row in source_records if isinstance(row, dict)}
    if len(by_name) != len(source_records) or set(by_name) != set(SOURCES):
        raise ValueError("public asset manifest source names are missing or duplicated")
    for name, (url, revision) in SOURCES.items():
        record = by_name[name]
        if (record.get("url"), record.get("revision")) != (url, revision) or not record.get("tree"):
            raise ValueError(f"public asset manifest identity mismatch for {name}")
    repositories = sorted(source_root / name for name in by_name)
    if any(not (repo / ".git").is_dir() for repo in repositories):
        raise ValueError("public asset manifest references a missing source repository")

    output = []
    manifest = []
    for repo in repositories:
        record = by_name[repo.name]
        revision = git(repo, "rev-parse", "HEAD")
        url = git(repo, "remote", "get-url", "origin")
        tree = git(repo, "rev-parse", "HEAD^{tree}")
        if git(repo, "status", "--porcelain"):
            raise ValueError(f"source working tree is not clean: {repo.name}")
        if (url, revision, tree) != (record["url"], record["revision"], record["tree"]):
            raise ValueError(f"source checkout differs from manifest: {repo.name}")
        candidates = collect(repo)
        if len(candidates) < args.per_source:
            raise ValueError(f"{repo.name} produced only {len(candidates)} candidates")
        selected = []
        source_paths = set()
        for candidate in candidates:
            if candidate["source_path"] in source_paths:
                continue
            selected.append(candidate)
            source_paths.add(candidate["source_path"])
            if len(selected) == args.per_source:
                break
        if len(selected) < args.per_source:
            raise ValueError(f"{repo.name} has fewer than {args.per_source} eligible source files")
        for index, row in enumerate(selected, 1):
            output.append({"candidate_id": f"{repo.name}-{index:02d}", "source_repo": repo.name,
                           "source_url": url, "source_revision": revision, **row})
        licenses = sorted(path.name for path in repo.iterdir()
                          if path.is_file() and path.name.lower().startswith(("license", "copying")))
        manifest.append({"source_repo": repo.name, "source_url": url, "revision": revision,
                         "tree": tree,
                         "license_files": licenses, "eligible_candidates": len(candidates),
                         "selected_candidates": len(selected)})

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "asset_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "manifest": manifest,
        "candidates": output,
    },
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(manifest), "candidates": len(output),
                      "per_source": args.per_source}, indent=2))


if __name__ == "__main__":
    main()
