#!/usr/bin/env python3
"""Fetch pinned public SPC sources and model snapshots without network drift."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SOURCES = {
    "system_prompts_leaks": ("https://github.com/asgeirtj/system_prompts_leaks.git", "b55f7e37b71f076eb3228faa836954b6610046fe"),
    "leaked-system-prompts": ("https://github.com/jujumilk3/leaked-system-prompts.git", "3afab05da7bbba93d04458e4b44e4239198f69cd"),
    "TheBigPromptLibrary": ("https://github.com/0xeb/TheBigPromptLibrary.git", "cabcdb04b5970211b1b6163d8725ae66bb48c5f0"),
    "system-prompts-and-models-of-ai-tools": ("https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools.git", "1e4203a7d88873c1b37ab2d1c07074fea498c274"),
    "SystemCheck": ("https://github.com/normster/SystemCheck.git", "ed829f6ac7cd4e09f7161fe7dffbebdd220eb024"),
}
MODELS = {
    "llmlingua2": ("microsoft/llmlingua-2-xlm-roberta-large-meetingbank", "ebaba9b0e874dadd3003ffcff828e4397e568089"),
    "llama2_surrogate": ("NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"),
    "qwen3_4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weight_digest(files: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode("utf-8")); digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def fetch_sources(output: Path) -> list[dict]:
    source_dir = output / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (url, revision) in SOURCES.items():
        repo = source_dir / name
        if not (repo / ".git").is_dir():
            subprocess.run(["git", "clone", "--filter=blob:none", url, str(repo)], check=True)
        _git(repo, "fetch", "--depth", "1", "origin", revision)
        _git(repo, "checkout", "--detach", revision)
        actual = _git(repo, "rev-parse", "HEAD")
        if actual != revision:
            raise RuntimeError(f"source revision mismatch for {name}: {actual}")
        records.append({"name": name, "url": url, "revision": actual, "path": str(repo.resolve())})
    return records


def fetch_models(output: Path) -> list[dict]:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise RuntimeError("install huggingface_hub to fetch model snapshots") from error
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (repo_id, revision) in MODELS.items():
        path = Path(snapshot_download(repo_id=repo_id, revision=revision, local_dir=model_dir / name))
        weights = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"})
        if not weights:
            raise RuntimeError(f"no model weights found for {repo_id}")
        records.append({"name": name, "repo_id": repo_id, "revision": revision,
                        "path": str(path.resolve()),
                        "weight_sha256": _weight_digest(weights, path),
                        "weight_files": {str(p.relative_to(path)): _sha256(p) for p in weights}})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", action="store_true", help="also fetch pinned Hugging Face snapshots")
    args = parser.parse_args()
    manifest = {"sources": fetch_sources(args.output), "models": fetch_models(args.output) if args.models else [],
                "model_fetch_status": "PINNED" if args.models else "NOT_REQUESTED"}
    destination = args.output / "public_assets_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(destination), "sources": len(manifest["sources"]), "models": len(manifest["models"])}, indent=2))


if __name__ == "__main__":
    main()
