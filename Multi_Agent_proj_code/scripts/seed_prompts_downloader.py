"""
Download additional seed prompts from public HuggingFace safety datasets
and merge them into data/seed_prompts.json.

Supported sources:
  - HarmBench (Mantas Mazeika et al.)
  - AdvBench (Andy Zou et al.)
  - JailbreakBench

Usage:
  pip install datasets
  python scripts/seed_prompts_downloader.py --source harmbench --max 100
  python scripts/seed_prompts_downloader.py --source advbench --max 50
  python scripts/seed_prompts_downloader.py --source all --max 300
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import hashlib
from pathlib import Path

from rich.console import Console

import config

console = Console()


def load_existing() -> dict:
    path = Path(config.SEED_PROMPTS_PATH)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": "1.0", "prompts": []}


def save(data: dict):
    Path(config.SEED_PROMPTS_PATH).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def make_id(text: str, prefix: str) -> str:
    return f"{prefix}_{hashlib.md5(text.encode()).hexdigest()[:8]}"


def _infer_category(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["bomb", "explosive", "weapon", "firearm", "chemical weapon"]):
        return "weapons"
    if any(w in text_lower for w in ["suicide", "self-harm", "self harm", "end my life"]):
        return "self_harm"
    if any(w in text_lower for w in ["hack", "malware", "ransomware", "sql injection", "phishing"]):
        return "illegal_activities"
    if any(w in text_lower for w in ["fake news", "misinformation", "deepfake", "conspiracy"]):
        return "misinformation"
    if any(w in text_lower for w in ["ignore", "dan ", "jailbreak", "developer mode", "bypass"]):
        return "jailbreaks"
    if any(w in text_lower for w in ["hate", "racist", "supremac", "genocide", "nazi"]):
        return "hate_speech"
    if any(w in text_lower for w in ["track", "stalk", "dox", "privacy", "spy"]):
        return "privacy_violation"
    return "illegal_activities"


def download_harmbench(max_prompts: int = 100) -> list[dict]:
    """Download from HarmBench dataset on HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]datasets package required: pip install datasets[/red]")
        return []

    console.print("Downloading HarmBench...")
    try:
        ds = load_dataset("LLM-LAT/harmful-dataset", split="train")
        prompts = []
        for i, row in enumerate(ds):
            if i >= max_prompts:
                break
            text = row.get("prompt") or row.get("text") or str(row)
            if not text or len(text) < 20:
                continue
            prompts.append({
                "id":       make_id(text, "hb"),
                "category": row.get("category", _infer_category(text)),
                "severity": int(row.get("severity", 3)),
                "text":     text.strip(),
                "source":   "harmbench",
            })
        console.print(f"  → {len(prompts)} prompts from HarmBench")
        return prompts
    except Exception as e:
        console.print(f"[yellow]HarmBench download failed: {e}[/yellow]")
        return []


def download_advbench(max_prompts: int = 100) -> list[dict]:
    """Download from AdvBench (Andy Zou et al.)."""
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]datasets package required: pip install datasets[/red]")
        return []

    console.print("Downloading AdvBench harmful behaviors...")
    try:
        ds = load_dataset("walledai/AdvBench", split="train")
        prompts = []
        for i, row in enumerate(ds):
            if i >= max_prompts:
                break
            text = row.get("prompt") or row.get("goal") or row.get("instruction") or str(row)
            if not text or len(text) < 20:
                continue
            prompts.append({
                "id":       make_id(text, "ab"),
                "category": _infer_category(text),
                "severity": 4,
                "text":     text.strip(),
                "source":   "advbench",
            })
        console.print(f"  → {len(prompts)} prompts from AdvBench")
        return prompts
    except Exception as e:
        console.print(f"[yellow]AdvBench download failed: {e}[/yellow]")
        return []


def download_jailbreakbench(max_prompts: int = 100) -> list[dict]:
    """Download from JailbreakBench."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    console.print("Downloading JailbreakBench...")
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
        prompts = []
        for i, row in enumerate(ds):
            if i >= max_prompts:
                break
            text = row.get("Goal") or row.get("Behavior") or str(row)
            if not text or len(text) < 20:
                continue
            prompts.append({
                "id":       make_id(text, "jbb"),
                "category": row.get("Category", _infer_category(text)),
                "severity": 4,
                "text":     text.strip(),
                "source":   "jailbreakbench",
            })
        console.print(f"  → {len(prompts)} prompts from JailbreakBench")
        return prompts
    except Exception as e:
        console.print(f"[yellow]JailbreakBench download failed: {e}[/yellow]")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download seed prompts from HuggingFace datasets")
    parser.add_argument("--source", choices=["harmbench", "advbench", "jailbreakbench", "all"],
                        default="all")
    parser.add_argument("--max", type=int, default=100, help="Max prompts per source")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    existing = load_existing()
    existing_ids = {p["id"] for p in existing.get("prompts", [])}
    new_prompts: list[dict] = []

    if args.source in ("harmbench", "all"):
        new_prompts.extend(download_harmbench(args.max))
    if args.source in ("advbench", "all"):
        new_prompts.extend(download_advbench(args.max))
    if args.source in ("jailbreakbench", "all"):
        new_prompts.extend(download_jailbreakbench(args.max))

    # Deduplicate
    added = [p for p in new_prompts if p["id"] not in existing_ids]
    console.print(f"\nNew prompts to add: [green]{len(added)}[/green]  "
                  f"(skipped {len(new_prompts) - len(added)} duplicates)")

    if args.dry_run:
        console.print("[yellow]Dry run — nothing saved.[/yellow]")
        for p in added[:5]:
            console.print(f"  [{p['category']}] {p['text'][:80]}...")
        return

    existing["prompts"].extend(added)
    save(existing)
    console.print(f"Saved {len(existing['prompts'])} total prompts to {config.SEED_PROMPTS_PATH}")


if __name__ == "__main__":
    main()
